# Nano-vLLM 在线调度器 7 天开发记录

> **状态标签**：历史开发日志（Day 1～7）。当前状态已更新为 `GPU Verified`；正式结果见[15-run 报告](../../reports/nanovllm-online-rtx4060-20260828.md)。下文的 GPU Pending 条目保留为当时的时间线记录。
>
> **基线**：`bb823b3`　|　记录原则：每天都有输入、输出、验收和证据；未完成项不回填成完成。

## Day 1：读懂基线与锁定边界

**目标**：不改代码，画出真实调用链与状态机。

**已核对**：

- `LLMEngine.generate → add_request → step → Scheduler.schedule → ModelRunner.run → Scheduler.postprocess`。
- `SequenceStatus` 只有 WAITING/RUNNING/FINISHED。
- 基线始终先尝试 Prefill，没有 waiting 批次才 Decode。
- `BlockManager` 用 free/used 集合、引用计数和完整块哈希管理 KV。
- Decode 分配失败会抢占运行序列并重新 Prefill。

**产出**：架构、时序、状态机、不变量和上游/本人边界文档。

**验收**：每条“已实现”都能定位到基线源码；当日尚未落地的在线目标标为 Planned，随后各日按证据更新。

**状态**：`Implemented`（文档审计完成）。

## Day 2：输出类型、请求 ID 与同步在线 API

**目标**：引入 `RequestOutput`、`StepStats`、`StepResult`，扩展同步引擎。

**任务**：

- `add_request(..., request_id=None) -> str`。
- 维护 request ID 与 sequence ID 映射。
- `abort_request(id) -> bool`，覆盖 waiting/running/已终态。
- `step_stream() -> StepResult`，输出累计 token/text 和每步统计。
- 保持已有 `generate()` 返回结构兼容。

**验收**：自生成 ID 唯一；engine 生命周期内曾用 ID 拒绝复用；stop/length/abort 每个请求恰好一个终态；最终 KV Block 释放。

**证据**：`tests/unit/test_llm_engine_online.py`；总门禁 `46 passed`。

**状态**：`Implemented / CPU Verified`。

## Day 3：三种调度策略

**目标**：把 Prefill/Decode 选择从固定分支变成显式策略。

**任务**：

- 配置 `scheduler_policy` 三选一。
- `bounded_decode_first` 使用 `max_consecutive_decode_steps=8`。
- 记录 `forced_prefill`、`allocation_blocked`、`preemptions`。
- 把策略选择与资源分配拆成可单测函数。

**验收**：策略顺序与契约一致；bounded 在 waiting 可执行时不越界；KV 阻塞不死循环；策略切换不破坏上游离线行为。

**证据**：`tests/unit/test_scheduler.py` 覆盖三策略、上限、KV 阻塞、抢占、取消和队列容量。

**状态**：`Implemented / CPU Verified`。

## Day 4：单 worker 异步引擎

**目标**：允许调用方并发提交，同时保持 CUDA 单线程所有权。

**任务**：

- `ADD/ABORT/SHUTDOWN` 命令模型。
- `max_queue_size=256` 限制 Scheduler 的 waiting+running 活跃请求总量。
- worker 在每个 step 边界 drain 命令。
- 每请求独立 async stream；慢消费者不能串扰其他请求。
- worker 异常广播、健康状态和幂等关闭。

**验收**：并发 submit 无重复/丢失；满队列明确背压；abort 最迟在下个 step 边界生效；测试结束无残留线程。

**证据**：`tests/unit/test_async_llm_engine.py` 覆盖 stream 隔离、单 owner 线程、跨线程异常、abort 和幂等 shutdown。

**状态**：`Implemented / CPU Verified`。

## Day 5：HTTP/SSE 与可观测性

**目标**：提供最小在线服务面。

**任务**：

- `POST /generate` 逐 token SSE。
- `DELETE /requests/{id}` 取消。
- `GET /health` 反映 worker 状态。
- `GET /metrics` 暴露请求计数、策略、累计调度事件和最近 step 快照。
- 映射 `409/422/429/500`，断连触发取消。

**验收**：正常 SSE 最后恰好一个 done；SSE 开始后的 worker error 以异常断流结束；参数错误在 headers 前返回；指标不含 request ID 或 prompt。

**证据**：`nanovllm/serve/sse.py` 已实现 lifespan、四端点和错误映射；纯 CPU 门禁已通过。真实 GPU/SSE 端到端仍 Pending。

**状态**：`Implemented / CPU Verified`（适配层）；GPU Pending。

## Day 6：CPU 门禁与故障注入

**目标**：不依赖 GPU 验证调度、并发、状态和错误契约。

**任务**：

- fake tokenizer/model runner 产生确定 token。
- 三策略、Chunked Prefill、抢占、Prefix Cache、取消竞态。
- 队列满、非法配置、重复 ID、worker crash、shutdown。
- `compileall`、pytest、`git diff --check`。

**验收**：全部 CPU 测试通过；无死锁；失败分支能结束所有 stream；Block 不变量成立。

**实施记录**：通过延迟 CUDA/Transformers 导入和 fake runner，使调度、异步引擎与 benchmark 在不加载模型的情况下可测。

**证据**：

```text
python -m pytest -q
46 passed

python -m compileall -q nanovllm benchmarks tests
通过

python -m ruff check nanovllm/__init__.py nanovllm/config.py \
  nanovllm/engine/llm_engine.py nanovllm/engine/scheduler.py \
  nanovllm/engine/sequence.py nanovllm/engine/async_llm_engine.py \
  nanovllm/engine/errors.py nanovllm/engine/outputs.py \
  nanovllm/serve benchmarks tests
通过

git diff --check
通过
```

开发主机使用 Python 3.13，只用于纯 CPU 测试；项目 `requires-python` 为 `>=3.10,<3.13`，正式 WSL 固定 Python 3.11。

**状态**：`Implemented / CPU Verified`。

## Day 7：WSL2 + RTX 4060 复现实验与交付

**目标**：形成可复算而非口头描述的 GPU 证据。

**任务**：

- 保存 Windows/WSL/driver/CUDA/Python/依赖/commit 环境。
- 固定模型、输入输出、并发、种子、warmup 和重复次数。
- 运行三策略 A/B，保存 request/step JSONL。
- 运行 Barrier interference：初始 Decode-heavy 全部首 token 后，以 100 ms 间隔注入长 Prompt。
- 正确性门禁通过后计算 TTFT、TPOT、E2E、吞吐、Goodput、KV 高水位。
- 更新实验报告和项目展示；性能数字只从原始数据生成。

**验收**：每轮有完整 schema、校验状态和 SHA-256；无 GPU 数据时结果表保持空白。

**状态**：`GPU Pending`。

## 节点同步表

| 节点 | 代码 | CPU 测试 | GPU 测试 | 文档 | 最终状态 |
|---|---|---|---|---|---|
| 基线审计 | 上游已有 | 未运行 | 不适用 | 已完成 | Implemented |
| 类型与同步 API | 已实现 | 已通过 | 待运行 | 已同步 | CPU Verified |
| 调度策略 | 已实现 | 已通过 | 待运行 | 已同步 | CPU Verified |
| 异步 worker | 已实现 | 已通过 | 待运行 | 已同步 | CPU Verified |
| HTTP/SSE | 已实现 | 已通过适配层 | 待运行 | 已同步 | CPU Verified / GPU Pending |
| RTX 4060 报告 | 不适用 | 不适用 | 待运行 | 模板已写 | GPU Pending |

## 每日复盘模板

```text
日期：
Commit：
今天改变了什么：
为什么这样设计：
运行了哪些命令：
通过/失败/未运行：
发现的根因：
原始证据路径：
明天只做的一件事：
```

## 待同步点

- GPU 环境就绪后补真实服务启动与 SSE 端到端证据。
- GPU 实验完成后只从 `artifacts/online_scheduler/` 自动汇总数字。
- 任何后续 API 字段、默认策略或错误语义变化都同步更新契约。
