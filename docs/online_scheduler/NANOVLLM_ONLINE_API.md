# Nano-vLLM Online API 契约

> **状态标签**：`Implemented`（同步/异步/SSE 契约） · `CPU Verified`（46 tests passed） · `GPU Pending`
>
> **契约版本**：v0.1　|　**基线**：`bb823b3`　|　**实现位置**：当前工作树

## 1. 兼容原则

现有 `LLM.generate(prompts, sampling_params, use_tqdm=True)` 行为保留。新增在线接口不改变已有返回结构；调用方只有显式使用 `add_request`、`step_stream`、`AsyncLLMEngine` 或 HTTP 服务时才进入在线路径。

## 2. Python 同步接口

### 2.1 `LLMEngine.add_request`

```python
def add_request(
    self,
    prompt: str | list[int],
    sampling_params: SamplingParams,
    request_id: str | None = None,
) -> str:
    ...
```

- 返回最终采用的 `request_id`。
- 未传 ID 时由引擎生成非空、唯一 ID。
- 同一 engine 生命周期内，任何曾使用过的 ID 再提交都抛出 `DuplicateRequestError`；请求完成后也不可复用。
- prompt token 与 `max_tokens` 的总长度不得超过模型允许长度。
- 请求成功加入后初始状态为 waiting；此方法不直接执行 GPU。

### 2.2 `LLMEngine.abort_request`

```python
def abort_request(self, request_id: str) -> bool:
    ...
```

返回值：

| 返回 | 含义 |
|---|---|
| `True` | 找到活跃请求，已标记终止、移出调度并释放资源 |
| `False` | ID 不存在或请求已经进入终态 |

成功取消必须产生且只产生一次 `finish_reason="abort"` 的终态输出。若同步实现选择由下一次 `step_stream()` 交付该事件，必须在测试中固定这一时序。

### 2.3 `LLMEngine.step_stream`

```python
def step_stream(self) -> StepResult:
    ...
```

实际类型：

```python
@dataclass(frozen=True, slots=True)
class StepResult:
    outputs: tuple[RequestOutput, ...]
    stats: StepStats | None
```

`outputs` 只包含本步有新 token 或刚进入终态的请求；`stats` 必须描述同一个调度步。调用方不能把 `elapsed_ms` 当作单请求 E2E。

### 2.4 输出类型

```python
@dataclass(frozen=True, slots=True)
class RequestOutput:
    request_id: str
    sequence_id: int
    token_id: int | None
    token_ids: tuple[int, ...]
    text: str
    finished: bool
    finish_reason: Literal["stop", "length", "abort"] | None
    timestamp_ns: int

@dataclass(frozen=True, slots=True)
class StepStats:
    step_id: int
    batch_kind: Literal["prefill", "decode", "idle"]
    batch_size: int
    scheduled_tokens: int
    waiting: int
    running: int
    kv_used_blocks: int
    kv_total_blocks: int
    preemptions: int
    forced_prefill: bool
    allocation_blocked: bool
    elapsed_ms: float
    scheduled_request_ids: tuple[str, ...]
    preempted_request_ids: tuple[str, ...]
    waiting_request_ids: tuple[str, ...]
    running_request_ids: tuple[str, ...]
    waiting_before: int
    running_before: int
    decode_streak: int
```

## 3. Python 异步接口

### 3.1 所有权模型

`AsyncLLMEngine` 启动一个 worker 线程。worker 独占同步 `LLMEngine`、CUDA 上下文和调度器；事件循环线程只进行参数校验、命令入队和异步结果分发。

命令类型固定为：

- `ADD`：注册请求和输出通道。
- `ABORT`：在 step 边界取消请求。
- `SHUTDOWN`：停止接收请求，清理活跃请求并退出 worker。

### 3.2 `submit`

```python
stream = await engine.submit(
    prompt="Hello",
    sampling_params=SamplingParams(temperature=0.6, max_tokens=64),
    request_id="demo-001",
)

async for event in stream:
    print(event.text, event.finished)
```

`submit` 返回单请求异步 stream。stream 只出现该请求的事件，最后一个事件必须 `finished=True`。若 worker 失败，stream 以结构化异常结束，不能永久挂起。

### 3.3 `generate` 便利流接口

```python
async for event in engine.generate(
    prompt="Hello",
    sampling_params=SamplingParams(temperature=0.6, max_tokens=64),
):
    print(event.text)
```

它是基于 `submit` 的 async iterator，逐个 yield `RequestOutput`，并在退出时关闭 stream；不是等待全部完成后返回最终对象的 coroutine。

### 3.4 背压

`max_queue_size=256` 当前限制 Scheduler 的 `waiting + running` 活跃请求总数，而不是 Python `_commands` 队列；总容量满时新 `ADD` 返回 `RequestQueueFullError`。每请求输出队列默认容量为 64，慢消费者溢出也会收到该异常并触发取消。调用方应退避或降低并发。

## 4. HTTP/SSE 约定

示例基础地址：`http://127.0.0.1:8000`。服务入口为 `nanovllm-serve` 或 `python -m nanovllm.serve.sse`；真实 CUDA 端到端仍为 GPU Pending。

### 4.1 端点清单

| 方法 | 路径 | 成功响应 | 用途 |
|---|---|---|---|
| `POST` | `/generate` | `200 text/event-stream` | 提交请求并逐事件输出 |
| `DELETE` | `/requests/{id}` | `200 application/json` | 取消活跃请求 |
| `GET` | `/health` | `200 application/json` | worker 与模型健康快照 |
| `GET` | `/metrics` | `200 application/json` | 请求、累计调度事件与最近 step 快照 |

### 4.2 `POST /generate`

请求：

```json
{
  "request_id": "demo-001",
  "prompt": "用一句话解释 KV Cache",
  "temperature": 0.6,
  "max_tokens": 64,
  "ignore_eos": false
}
```

`request_id` 可省略；`prompt` 必填，三个采样参数为顶层字段。服务在发送 SSE headers 之前完成参数、重复 ID 和 admission 背压检查，以便返回正确 HTTP 状态码。

响应 headers 至少包括：

```text
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
```

SSE token 事件：

```text
event: token
data: {"request_id":"demo-001","sequence_id":7,"token_id":123,"token_ids":[123],"text":"KV","finished":false,"finish_reason":null,"timestamp_ns":0}

```

SSE 终态事件：

```text
event: done
data: {"request_id":"demo-001","sequence_id":7,"token_id":456,"token_ids":[123,456],"text":"KV Cache ...","finished":true,"finish_reason":"length","timestamp_ns":0}

```

示例中的 `timestamp_ns=0` 是 schema 占位符，不是合法实测值；实现必须填单调时钟。

客户端断开连接时，服务应提交 ABORT；由于取消在 step 边界生效，断连不等于 GPU kernel 立即停止。

### 4.3 `DELETE /requests/{id}`

成功：

```json
{
  "request_id": "demo-001",
  "aborted": true
}
```

不存在或已终态返回 `404`：

```json
{"detail":"active request not found"}
```

该场景不与重复提交的 `409` 混用。

### 4.4 `GET /health`

健康：

```json
{
  "status": "ok",
  "worker_alive": true,
  "started": true,
  "closed": false
}
```

worker 初始化失败或已异常退出时，当前端点仍返回 `200`，但状态为 unavailable：

```json
{
  "status": "unavailable",
  "worker_alive": false,
  "closed": true
}
```

不得把完整 Python traceback、模型绝对路径或 prompt 返回给未受信任客户端。

### 4.5 `GET /metrics`

当前返回 JSON 快照：

```json
{
  "started": true,
  "closed": false,
  "worker_alive": true,
  "scheduler_policy": "bounded_decode_first",
  "active_requests": 1,
  "submitted_requests": 2,
  "finished_requests": 1,
  "aborted_requests": 0,
  "emitted_tokens": 8,
  "preemptions": 0,
  "forced_prefills": 1,
  "allocation_blocked_steps": 0,
  "last_step": {
    "step_id": 9,
    "batch_kind": "prefill",
    "batch_size": 1,
    "scheduled_tokens": 128,
    "waiting": 0,
    "running": 1,
    "kv_used_blocks": 1,
    "kv_total_blocks": 10,
    "decode_streak": 0,
    "elapsed_ms": 0.0
  }
}
```

它不是 Prometheus 文本协议。当前快照包含调度策略、累计抢占/强制 Prefill/分配阻塞计数，以及最近一步的 batch、队列、KV、Decode streak 和耗时；不包含 request ID 或 prompt。

## 5. 错误码

FastAPI 当前错误体：

```json
{"detail":"request_id already exists: demo-001"}
```

| HTTP | `error.code` | 触发条件 | 客户端动作 |
|---:|---|---|---|
| `409` | `DUPLICATE_REQUEST_ID` | ID 在当前 Engine 生命周期内已使用 | 更换 ID |
| `422` | `INVALID_REQUEST` | 空 prompt、采样参数非法、策略值非法 | 修正参数 |
| `422` | `CONTEXT_LENGTH_EXCEEDED` | prompt 与最大输出超过模型长度 | 缩短输入或输出上限 |
| `429` | `QUEUE_FULL` | `waiting + running` 达到 `max_queue_size` | 退避后重试 |
| `503` | `ENGINE_CLOSED` | 引擎已关闭或正在关闭 | 等待服务恢复 |
| `500` | `WORKER_ERROR` | worker 初始化、CUDA 或模型执行失败 | 查看健康与服务日志 |

SSE 开始前捕获到的 worker `RuntimeError` 映射为 `500`。SSE 已开始后发生的 worker 异常无法再修改 HTTP 状态；当前实现会让异步流抛错并关闭，没有独立的 `event: error` schema。客户端必须把异常断流视为失败，而不是正常 done。

## 6. 可观测性与时间语义

- `timestamp_ns` 用 `time.monotonic_ns()` 一类单调时钟，适合进程内排序，不当作 Unix 时间。
- `elapsed_ms` 是单个 engine step 的耗时。
- TTFT 由服务接收请求到第一个 token 事件计算。
- TPOT 需要逐 token 时间戳计算，输出 token 少于两个时记为不可计算。
- E2E 到终态事件；abort 请求单独统计，不混入正常完成分位数。

## 7. 契约测试清单

- 相同 ID 并发提交：一个成功，一个 `409`。
- 队列满：新请求 `429`，已接收请求不丢失。
- 非法温度、非正 `max_tokens`、过长上下文：`422`。
- 正常 EOS、长度结束、主动取消分别返回 `stop/length/abort`。
- 每个流最后恰好一个 done；done 后没有 token。
- worker 故障时所有未完成流结束并得到错误，健康快照显示 `unavailable`。
- 客户端断连触发取消，最终 KV Block 使用量回落。

当前纯 CPU 门禁为 `46 passed`，且 Ruff、`compileall`、`git diff --check` 通过；这证明状态机、异步隔离、benchmark 与 API 适配的 CPU 路径，不证明真实 CUDA/SSE 性能。开发主机 Python 3.13 仅用于纯 CPU 测试；项目正式运行范围是 `>=3.10,<3.13`，WSL 使用 Python 3.11。
