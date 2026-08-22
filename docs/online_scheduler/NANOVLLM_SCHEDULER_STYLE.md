# Nano-vLLM 在线调度器实现规范

> **状态标签**：`Implemented`（在线实现规范） · `CPU Verified`（46 tests passed） · `GPU Pending`
>
> **适用范围**：在线调度扩展代码、测试与文档　|　**基线**：`bb823b3`

## 1. 设计优先级

1. 正确释放 KV Block，避免泄漏和重复释放。
2. 一个请求只有一个终态，事件不串流。
3. 调度决策可解释、可测试、可统计。
4. 保持上游离线 API 兼容。
5. 在正确性成立后再讨论延迟和吞吐。

## 2. 推荐目录职责

实际文件名以落地代码为准，但职责必须保持单向：

```text
nanovllm/
├── config.py                 # 配置和纯校验
├── engine/
│   ├── sequence.py           # 序列状态与 token 数据
│   ├── block_manager.py      # KV Block 所有权
│   ├── scheduler.py          # 批次选择和后处理
│   ├── llm_engine.py         # 同步请求 API 与 step
│   ├── outputs.py            # RequestOutput / StepStats / StepResult
│   └── async_llm_engine.py   # worker、命令和异步 stream
└── server/
    └── sse.py                # HTTP 参数与 SSE 编码
tests/
├── test_scheduler_policy.py
├── test_online_engine.py
├── test_async_engine.py
└── test_online_api.py
```

HTTP 层不得操作 `Sequence` 或 `BlockManager`；异步层不得复制调度策略；Scheduler 不依赖 HTTP 或 asyncio。

## 3. 命名与类型

- 类与 dataclass：`PascalCase`，如 `RequestOutput`、`StepStats`。
- 函数和字段：`snake_case`，布尔字段用可读谓词，如 `forced_prefill`。
- 策略值使用字符串枚举或 `Literal`，只能是 `prefill_first`、`decode_first`、`bounded_decode_first`。
- 时间单位写入名字：`timestamp_ns`、`elapsed_ms`，禁止裸 `timestamp`/`latency`。
- 累计值和单步值区分：`token_ids` 是累计，`token_id` 是本步新值。
- `request_id` 是外部标识；`sequence_id` 是内部标识，禁止混用。

公开接口必须有类型注解和 docstring，docstring 说明状态变化、线程所有权和异常，不复述函数名。

## 4. 线程与 CUDA 所有权

唯一允许的拓扑：

```text
asyncio / HTTP threads
        │ 只写命令、读事件
        ▼
command queue + Scheduler active-request admission limit
        │
        ▼
single worker thread
        ├─ LLMEngine
        ├─ Scheduler
        ├─ BlockManager
        └─ CUDA context / ModelRunner
```

规则：

- CUDA 初始化、模型执行和销毁在同一 worker 线程完成。
- `waiting`、`running`、请求 ID 映射和 KV Block 状态只在 worker 修改。
- 每个 engine step 前 drain 命令，顺序处理 `ADD/ABORT/SHUTDOWN`；`max_queue_size` 当前限制 Scheduler 的 `waiting + running`，而不是 Python 命令队列长度。
- 不跨线程传递可变 `Sequence`；对外只发不可变快照或防御性复制。
- 不用粗粒度锁让多个线程轮流调用 CUDA；那会让所有权难以证明。
- worker 异常必须转换为引擎失败态，唤醒所有等待消费者。

## 5. 调度器编码规则

调度应分为“选择批次”和“提交状态变化”两个易测阶段。策略判断保持纯净，资源分配和队列移动集中处理。

推荐逻辑：

```python
batch_kind, forced_prefill = self._choose_batch_kind()
if batch_kind == "prefill":
    seqs = self._schedule_prefill()
else:
    seqs = self._schedule_decode()
return ScheduleBatch(...)
```

避免：

- 在 HTTP handler 根据队列长度决定 Prefill/Decode。
- 通过捕获 `AssertionError` 实现正常背压或资源不足流程。
- 在一次循环中无记录地从 waiting/running 反复移动同一序列。
- 将“无法分配”误报为“没有请求”。
- 用全局变量保存连续 Decode 步数。

### 5.1 连续 Decode 计数

- 成功执行 Decode 批次后加一。
- 成功执行 Prefill 后归零。
- 无 waiting 时可连续 Decode，不应虚构 `forced_prefill`。
- 到达上限但 waiting 请求不可分配时，必须避免死循环，并记录 `allocation_blocked`；是否继续 Decode由锁定策略测试确定。
- 策略切换或引擎重置时计数归零。

### 5.2 抢占

抢占是可观察事件，不是异常：

1. 被抢占序列从 running 移除。
2. 已占 KV Block 全部释放。
3. `num_cached_tokens` 清零，`block_table` 清空。
4. `is_prefill=True`，状态回到 waiting。
5. `preemptions` 计数增加。

## 6. 请求状态与事件规则

- ADD 只有在 engine 生命周期 ID 唯一性和 Scheduler 活跃请求总容量校验成功后才对调用方可见。
- ABORT 幂等：第一次对活跃请求返回成功，之后返回失败/未找到。
- 终态只允许 `stop`、`length`、`abort`；worker error 走异常事件，不伪装成正常 finish reason。
- 每个新 token 事件重新生成累计 token 快照；不得把内部可变 list 直接交给消费者。
- 文本由累计 token 解码，避免 byte-pair token 单独 decode 导致拼接错误。
- 慢消费者不得阻塞 CUDA worker；当前单请求输出队列默认容量为 64，溢出会给该 stream 投递 `RequestQueueFullError` 并排队取消。

## 7. 配置校验规范

用户可触发的配置错误使用显式 `ValueError` 或领域异常，不依赖 `assert`。至少校验：

```text
scheduler_policy in allowed policies
max_consecutive_decode_steps > 0
max_queue_size > 0
max_num_batched_tokens > 0
max_num_seqs > 0
0 < gpu_memory_utilization <= 1
1 <= tensor_parallel_size <= 8
kvcache_block_size > 0 and divisible by 256
```

错误消息包含字段名、收到的值和合法范围，但不得包含凭据或完整 prompt。

## 8. 统计与日志规范

- `StepStats` 在后处理完成后采样 waiting/running/KV 数量。
- 计时用单调时钟，范围严格覆盖 `schedule → model run → postprocess`。
- 统计采集不能再次遍历或 decode 大量历史 token。
- 结构化日志至少包含 `step_id`、`batch_kind`、`batch_size`，调试时可含 `request_id`；公开指标禁止 request ID label。
- 不记录 prompt 原文；实验如需关联输入，记录数据集 ID、长度和哈希。

## 9. 异常与清理

使用 `try/finally` 保证以下资源被清理：

- worker 失败：为所有活跃请求发布错误并终止 stream。
- 正常 shutdown：停止接受新 ADD，处理或取消现有请求，再退出模型进程。
- 客户端断开：排队 ABORT，不能由 HTTP 线程直接释放 KV。
- 初始化失败：`/health` 显示不可用，不启动一个假健康服务。

禁止静默 `except Exception: pass`。错误应保留内部 traceback 到服务日志，对客户端只给稳定错误码。

## 10. 测试风格

### 10.1 CPU 单元测试

用 fake tokenizer/fake model runner 把调度与 CUDA 解耦：

- fake runner 根据序列返回确定 token。
- 每个测试固定 prompt token、EOS、最大输出和 KV Block 数。
- 断言完整队列、状态、事件和 Block 集合，不只断言“函数没报错”。
- 并发测试使用超时，失败时能明确指出死锁。
- 测试结束断言没有存活 worker 线程。

### 10.2 GPU 测试

GPU 测试单独标记，默认 CPU 门禁不得因无 GPU 失败。GPU 输出先做正确性门禁，再写性能表。原始数据默认写入 `artifacts/online_scheduler/`，schema 见 `docs/online_scheduler/results/README.md`；不把终端截图当唯一证据。

## 11. 文档状态规范

- 未落地能力写 `Planned` 或 `GPU Pending`，不能写 `Implemented`。
- CPU 测试通过可写 `CPU Verified`，同时附命令和测试数。
- GPU 指标只有存在原始 JSON/JSONL、环境清单和重复运行时才能填写。
- 上游功能与本人新增功能分段书写，禁止用“我们从零实现整个 Nano-vLLM”描述 fork。
- 每次接口字段变化，同时更新 API 文档、schema README 和契约测试。

## 12. 提交前检查

```bash
python -m compileall -q nanovllm tests
python -m pytest -q
git diff --check
```

当前工作树执行 `python -m pytest -q` 得到 `46 passed`，Ruff、`compileall` 与 `git diff --check` 也通过；GPU 仍报告 `NOT RUN / GPU Pending`。开发主机 Python 3.13 只用于纯 CPU 测试，正式运行遵守项目 `>=3.10,<3.13` 并使用 Python 3.11。文档 Agent 不负责 Git 提交，交付前只报告文件和待同步点。
