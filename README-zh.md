<p align="center">
  <img width="300" src="assets/logo.png" alt="Nano-vLLM Logo">
</p>

<p align="center">
  <strong>轻量、可读、可验证的大模型推理引擎与在线调度实验平台</strong>
</p>

<p align="center">
  中文 ｜ <a href="README.md">English</a> ｜
  <a href="docs/online_scheduler/DEV_DOCUMENT.md">开发文档</a> ｜
  <a href="docs/online_scheduler/WSL2_RTX4060_RUNBOOK.md">RTX 4060 复现手册</a>
</p>

# Nano-vLLM：面向在线请求的 Decode 优先与防饥饿调度优化

Nano-vLLM 是一个结构紧凑的 vLLM 风格推理引擎。上游项目已经实现离线批量推理、Paged KV Cache、Prefix Cache、Chunked Prefill、Recompute 抢占、Tensor Parallel、FlashAttention 和 Decode CUDA Graph 等核心机制。

当前分支在这些能力之上完成了在线调度二次开发：将原本“批量提交、全部完成后返回”的执行方式扩展为支持动态请求接入、逐 token 流式输出、请求取消、背压、调度指标和可重放实验的单机在线推理原型。核心研究问题是：当 Prefill 与 Decode 竞争同一张 GPU 时，怎样降低在途请求的 token 间抖动，同时避免新请求长期拿不到首 token。

> [!IMPORTANT]
> 当前在线调度代码已经实现，并通过 46 项纯 CPU 契约测试。Milestone 0 已在 RTX 4060 8GB 上完成上游离线 `bench.py` 基线；面向在线调度的 CUDA/SSE 正确性与三策略 A/B 实验仍为 **GPU Pending**。两类实验使用不同接口和指标，不能互相替代。

## 目录

- [项目状态](#项目状态)
- [核心能力](#核心能力)
- [系统架构](#系统架构)
- [三种调度策略](#三种调度策略)
- [安装与环境](#安装与环境)
- [离线推理快速开始](#离线推理快速开始)
- [启动在线 SSE 服务](#启动在线-sse-服务)
- [Python 在线接口](#python-在线接口)
- [可复现实验与指标](#可复现实验与指标)
- [测试与验证](#测试与验证)
- [项目结构](#项目结构)
- [文档导航](#文档导航)
- [能力边界与已知限制](#能力边界与已知限制)
- [上游与二次开发边界](#上游与二次开发边界)

## 项目状态

| 范围                           | 状态           | 当前证据                                                        |
| ------------------------------ | -------------- | --------------------------------------------------------------- |
| 上游离线推理核心               | `Implemented`  | 源码中的 Scheduler、BlockManager、ModelRunner、Attention 等模块 |
| Milestone 0：RTX 4060 离线基线 | `GPU Verified` | 256 sequences、133,966 输出 token、1241.94 output tokens/s      |
| 在线调度二次开发               | `Implemented`  | 三种策略、动态请求、流式输出、取消、异步 worker、SSE、benchmark |
| 纯 CPU 逻辑验证                | `CPU Verified` | `python -m pytest -q`：46 passed                                |
| RTX 4060 在线 SSE 正确性冒烟   | `GPU Pending`  | 尚未归档在线服务的真实 CUDA/SSE 运行产物                        |
| RTX 4060 在线三策略性能结论    | `GPU Pending`  | 不预写 TTFT、TPOT、在线吞吐或显存提升数字                       |

这里的 `CPU Verified` 只证明调度状态机、异步隔离、API 适配、benchmark 和指标计算的无 CUDA 路径，不代表模型已经在 CPU 上完成推理，更不代表 GPU 性能已经验证。

Milestone 0 的 1241.94 output tokens/s 来自单次饱和离线批处理：Qwen3-0.6B、256 sequences、133,966 个生成 token、107.87 秒计时区间。它证明指定 WSL2/RTX 4060 环境能够运行真实模型和上游离线 workload，但不包含 TTFT、TPOT、P50/P95 或在线并发指标。完整环境、命令和限制见 [RTX 4060 离线基线报告](docs/baseline-rtx4060.md)。

## 核心能力

### 上游 Nano-vLLM 已有能力

- Qwen3 模型结构、权重加载与温度采样；
- `waiting/running` 双队列与迭代级动态组批；
- Chunked Prefill；
- Paged KV Cache、Prefix Cache、引用计数与 Block 生命周期管理；
- KV Block 不足时的 Recompute 抢占；
- FlashAttention 变长 Prefill 与分页 KV Decode；
- Triton KV Cache 写入内核；
- Tensor Parallel；
- Decode CUDA Graph；
- 与 vLLM 风格相近的离线 `LLM.generate()` 接口。

### 当前分支新增能力

- `prefill_first`、`decode_first`、`bounded_decode_first` 三种可切换策略；
- 以 `max_consecutive_decode_steps` 控制连续 Decode 上限，并记录强制 Prefill；
- 外部稳定 `request_id`、重复 ID 拒绝和请求长度预检；
- 单步累计 token 事件、`stop/length/abort` 三种终态；
- 在 step 边界安全取消请求并释放 KV Block；
- `AsyncLLMEngine` 单 worker 线程独占 CUDA 和所有可变引擎状态；
- Scheduler 活跃请求总量背压与慢消费者保护；
- FastAPI/SSE 生成、取消、健康检查和 JSON 指标接口；
- fixed、Poisson、长短请求混合和 first-token barrier 干扰负载；
- 逐请求、逐 token、逐 step 的 JSON/JSONL 原始产物与 GPU 遥测；
- TTFT、TPOT、ITL、E2E、吞吐、Goodput 和调度行为汇总。

## 系统架构

```text
HTTP / Python 调用方
        │
        │ ADD / ABORT / SHUTDOWN
        ▼
AsyncLLMEngine 命令通道
        │
        ▼
单 worker 线程
  ├─ 独占 LLMEngine 与 CUDA 上下文
  ├─ 在每个 step 边界处理命令
  └─ 将事件分发到对应请求的异步流
        │
        ▼
Scheduler
  ├─ waiting：等待或继续 Prefill
  ├─ running：正在 Decode
  ├─ BlockManager：Paged KV / Prefix Cache / 回收
  └─ ModelRunner：Prefill / Decode / CUDA Graph / TP
        │
        ▼
RequestOutput + StepStats
        │
        ├─ SSE token / done 事件
        ├─ /metrics 实时快照
        └─ benchmark 原始产物
```

并发发生在“请求提交与结果消费”层，模型执行仍由一个 worker 串行驱动动态批次。这样可以明确 CUDA 上下文、Scheduler、Sequence 和 KV Block 的唯一所有者，避免多个 HTTP handler 同时修改共享状态。

### 请求生命周期

```text
ADD
 └─> WAITING
      ├─ Prefill 完成 ─> RUNNING ─> Decode ─> FINISHED(stop/length)
      ├─ KV 不足 <────── 抢占并释放 Block <────┘
      └─ ABORT ───────────────────────────────> ABORTED
```

取消命令不会中断已经发射的 CUDA kernel，而是在下一个 step 边界生效。成功取消后，请求会从 waiting/running 移除、释放已持有的 KV Block，并产生且只产生一次 `finish_reason="abort"` 的终态事件。

## 三种调度策略

当前 ModelRunner 的一次调用只执行同一种 phase，因此调度器每步选择 Prefill 或 Decode，不在同一模型调用中混合两类 token。

| 策略                   | 选择规则                                     | 主要收益                                 | 主要风险                                          |
| ---------------------- | -------------------------------------------- | ---------------------------------------- | ------------------------------------------------- |
| `prefill_first`        | waiting 中存在可执行请求时优先 Prefill       | 保持上游兼容行为，新请求较快进入模型     | 持续到达的新请求可能拉大在途请求 TPOT/ITL         |
| `decode_first`         | running 非空时优先 Decode                    | 保持已有请求连续吐 token                 | 新请求的 TTFT 可能长期恶化甚至饥饿                |
| `bounded_decode_first` | 优先 Decode；连续达到 K 步后尝试强制 Prefill | 在 TPOT 与 TTFT 之间建立可解释的有界折中 | K 需要按模型、负载和 GPU 实测，不能视作理论最优值 |

在线服务默认使用：

```text
scheduler_policy = bounded_decode_first
max_consecutive_decode_steps = 8
```

`K=8` 是 RTX 4060 实验的起始控制变量，不是已经验证的最优参数。有界策略保证的是：当 waiting 请求可执行时，连续偏向 Decode 的步数受到限制。如果新请求因 KV 容量或驻留序列上限暂时不可分配，调度器会记录 `allocation_blocked=true` 并继续可执行工作；它不提供墙钟时间意义上的绝对 SLO 保证。

## 安装与环境

### 支持范围

- 操作系统：Linux；Windows 用户推荐 WSL2 Ubuntu 22.04；
- Python：`>=3.10,<3.13`；已验证的离线基线使用 Python 3.12.14，在线调度 runbook 推荐 Python 3.11；
- GPU：支持 CUDA 的 NVIDIA GPU；
- 推荐起步模型：Qwen3-0.6B；
- RTX 4060 8GB：建议先使用单卡、短上下文和保守显存配置。

macOS 没有本项目所需的 NVIDIA CUDA/FlashAttention 正式运行环境。开发期间在 macOS + Python 3.13 上完成的测试只覆盖延迟导入后的纯 CPU 契约路径；Python 3.13 也不在项目声明的正式支持范围内。

### 创建环境

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

请先根据当前 NVIDIA 驱动和 CUDA 环境，从 PyTorch 官方安装页面选择匹配的 PyTorch wheel，再在仓库根目录安装项目：

```bash
python -m pip install -e '.[online,test]'
```

其中：

- 默认依赖包含 PyTorch、Triton、Transformers、FlashAttention、tqdm 和 xxhash；
- `online` 安装 FastAPI 与 Uvicorn；
- `test` 安装 pytest 与 pytest-asyncio。

FlashAttention 对 Python、PyTorch、CUDA 和编译器组合较敏感。如果安装失败，应先确认 PyTorch CUDA 基础可用并保留完整构建日志，不要直接使用来源不明的 wheel。

### 下载模型

以下命令与上游示例一致；如果当前 Hugging Face CLI 已调整参数，请以本机 `--help` 为准：

```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir /YOUR/MODEL/PATH \
  --local-dir-use-symlinks False
```

正式实验应记录模型 revision、实际绝对路径和文件清单，但不要把 Hugging Face token 或其他凭据提交到仓库。

### GPU 环境自检

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -c "import transformers, triton, flash_attn; print('imports ok')"
```

完整的 WSL2 安装、驱动边界、常见错误和验收条件见 [WSL2 + RTX 4060 复现实验手册](docs/online_scheduler/WSL2_RTX4060_RUNBOOK.md)。

## 离线推理快速开始

离线接口保持与上游兼容：

```python
from nanovllm import LLM, SamplingParams

model_path = "/YOUR/MODEL/PATH"
llm = LLM(
    model_path,
    enforce_eager=True,
    tensor_parallel_size=1,
)

sampling_params = SamplingParams(
    temperature=0.6,
    max_tokens=128,
)
prompts = [
    "Hello, Nano-vLLM.",
    "请用一句话解释 KV Cache。",
]

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(output["text"])
```

第一次 GPU 冒烟建议使用 `enforce_eager=True`，先排除 CUDA Graph 捕获带来的额外变量；正确性通过后，再单独比较 eager 与 CUDA Graph。

## 启动在线 SSE 服务

RTX 4060 8GB 建议从以下保守配置开始：

```bash
nanovllm-serve \
  --model /YOUR/MODEL/PATH \
  --host 127.0.0.1 \
  --port 8000 \
  --scheduler-policy bounded_decode_first \
  --max-consecutive-decode-steps 8 \
  --max-queue-size 256 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 512 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.75 \
  --enforce-eager
```

也可以使用：

```bash
python -m nanovllm.serve.sse --model /YOUR/MODEL/PATH --enforce-eager
```

### 流式生成

```bash
curl -N -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "request_id": "demo-001",
    "prompt": "用一句话解释 PagedAttention",
    "temperature": 0.6,
    "max_tokens": 64,
    "ignore_eos": false
  }'
```

响应使用标准 SSE 事件：

```text
event: token
data: {"request_id":"demo-001","token_id":123,"token_ids":[123],"text":"分页","finished":false,"finish_reason":null,...}

event: done
data: {"request_id":"demo-001","token_id":456,"token_ids":[123,456],"text":"分页 KV ...","finished":true,"finish_reason":"length",...}
```

`token_ids` 和 `text` 都是截至当前事件的累计快照，不是只包含本步增量；`token_id` 才是本步新生成的 token。

### 取消请求

```bash
curl -X DELETE http://127.0.0.1:8000/requests/demo-001
```

### 健康检查与实时指标

```bash
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/metrics
```

`/metrics` 当前返回 JSON 快照，不是 Prometheus 文本协议。快照包括请求计数、调度策略、累计输出 token、抢占、强制 Prefill、分配阻塞，以及最近一步的 batch、队列、KV Block、Decode streak 和耗时。

### HTTP 状态码

| 状态码 | 场景                                              |
| -----: | ------------------------------------------------- |
|  `200` | 流式生成、成功取消、健康或指标查询                |
|  `404` | 取消不存在或已进入终态的请求                      |
|  `409` | `request_id` 在当前 Engine 生命周期内已使用       |
|  `422` | prompt、采样参数或上下文长度非法                  |
|  `429` | Scheduler 的 `waiting + running` 达到活跃请求上限 |
|  `503` | 引擎已经关闭或正在关闭                            |
|  `500` | worker 初始化、CUDA 或模型执行失败                |

完整字段、错误语义和时序见 [在线 API 契约](docs/online_scheduler/NANOVLLM_ONLINE_API.md)。

## Python 在线接口

不经过 HTTP 时，可以直接使用异步引擎：

```python
import asyncio

from nanovllm import SamplingParams
from nanovllm.engine.async_llm_engine import AsyncLLMEngine


async def main() -> None:
    engine = AsyncLLMEngine(
        "/YOUR/MODEL/PATH",
        scheduler_policy="bounded_decode_first",
        max_consecutive_decode_steps=8,
        max_queue_size=256,
        max_num_seqs=32,
        max_num_batched_tokens=512,
        max_model_len=2048,
        gpu_memory_utilization=0.75,
        enforce_eager=True,
    )
    params = SamplingParams(temperature=0.6, max_tokens=64)

    try:
        async for event in engine.generate(
            prompt="请解释 Decode 优先调度。",
            sampling_params=params,
            request_id="python-demo-001",
        ):
            print(event.text, event.finished, event.finish_reason)
    finally:
        await engine.shutdown()


asyncio.run(main())
```

`AsyncLLMEngine` 只允许由一个 asyncio event loop 使用。每个请求默认有 64 个事件的输出队列；慢消费者使队列溢出时，该请求会收到背压异常并触发取消，避免阻塞 CUDA worker。

## 可复现实验与指标

`bench.py` 仍保留为上游离线吞吐基线。在线调度策略比较应使用 `benchmarks/online_scheduler/`，因为它能重放动态到达、保存逐事件证据并复算尾延迟。

### 混合 Poisson 负载

```bash
python -m benchmarks.online_scheduler.cli \
  --model /YOUR/MODEL/PATH \
  --policy bounded_decode_first \
  --max-consecutive-decode-steps 8 \
  --workload mixed \
  --arrival poisson \
  --repeat-index 0 \
  --output-root artifacts/online_scheduler
```

要比较三种策略，只替换 `--policy`，并保持模型、commit、随机种子、请求 trace、引擎配置和重复次数一致。

### Prefill/Decode 干扰负载

```bash
python -m benchmarks.online_scheduler.cli \
  --model /YOUR/MODEL/PATH \
  --policy bounded_decode_first \
  --workload interference \
  --interference-decode-requests 8 \
  --interference-prefill-requests 8 \
  --injection-interval-ms 100 \
  --repeat-index 0 \
  --output-root artifacts/online_scheduler
```

该 workload 会先让一组 Decode-heavy 请求全部产生首 token，记录 barrier，然后按固定间隔注入长 Prompt，更稳定地观察 Prefill 对在途 Decode 的影响。

### 产物目录

每次运行创建独立目录，不覆盖历史结果：

```text
artifacts/online_scheduler/<run_id>/
├── workload.jsonl       # 可跨策略重放的请求 trace
├── manifest.json        # commit、dirty 状态、模型、软件、GPU 与配置
├── requests.jsonl       # 每请求生命周期与汇总
├── tokens.jsonl         # 每个真实输出 token 的时间戳
├── steps.jsonl          # 每个调度步的队列、KV、抢占和策略状态
├── gpu_telemetry.csv    # nvidia-smi 周期采样，失败时显式记录
├── summary.json         # 从原始事件计算的分位数与吞吐
├── stdout.log
└── failure.json         # 仅失败运行存在，失败证据不删除
```

### 指标口径

| 指标     | 本项目口径                                                                    |
| -------- | ----------------------------------------------------------------------------- |
| TTFT     | 理想到达时间到第一个输出 token                                                |
| ITL      | 相邻输出 token 的时间间隔样本                                                 |
| TPOT     | `(finish - first_token) / (output_tokens - 1)`；少于两个输出 token 时不可计算 |
| E2E      | 理想到达时间到请求终态                                                        |
| 吞吐     | 墙钟区间内的 output tokens/s 与 requests/s                                    |
| Goodput  | 配置 TTFT/TPOT SLO 后，满足 SLO 的有效请求吞吐                                |
| 调度行为 | Prefill/Decode/idle 步数、强制 Prefill、抢占、最大 Decode streak、KV 峰值等   |

正式结论必须先通过请求数、终态唯一性、token 时间、输出长度、KV 不越界和 bounded 策略边界等正确性门禁，再比较性能。详细 schema 见 [在线调度实验原始数据规范](docs/online_scheduler/results/README.md)。

## 测试与验证

在仓库根目录执行：

```bash
python -m compileall -q nanovllm benchmarks tests
python -m pytest -q
git diff --check
```

可选 Ruff 门禁：

```bash
python -m ruff check nanovllm/__init__.py nanovllm/config.py \
  nanovllm/engine/llm_engine.py nanovllm/engine/scheduler.py \
  nanovllm/engine/sequence.py nanovllm/engine/async_llm_engine.py \
  nanovllm/engine/errors.py nanovllm/engine/outputs.py \
  nanovllm/serve benchmarks tests
```

最近一次已保存的纯 CPU 门禁结果：

```text
46 passed
Ruff passed
compileall passed
git diff --check passed
online scheduler GPU NOT RUN / GPU Pending
```

测试使用 fake tokenizer、fake runner、延迟导入和纯 CPU benchmark 组件，覆盖：

- 三种策略的选择顺序与 bounded 强制 Prefill；
- KV 不足抢占、重新入队和释放不变量；
- 请求 ID、累计 token、三种终态与取消；
- 并发 submit、admission 背压、慢消费者和 worker 故障；
- SSE 端点、错误映射和健康/指标快照；
- fixed、Poisson、barrier workload 与指标计算；
- benchmark runner 的到达时序和产物 schema。

## 项目结构

```text
nano-vllm/
├── nanovllm/
│   ├── config.py                     # 模型与调度配置校验
│   ├── engine/
│   │   ├── llm_engine.py             # 离线 API、动态请求、step_stream、取消
│   │   ├── async_llm_engine.py       # 单 CUDA worker、命令通道、异步流
│   │   ├── scheduler.py              # 三策略、Chunked Prefill、抢占
│   │   ├── block_manager.py          # Paged KV、Prefix Cache、引用计数
│   │   ├── model_runner.py           # 输入准备、模型执行、CUDA Graph、TP
│   │   ├── sequence.py               # 请求状态与 token/KV 进度
│   │   ├── outputs.py                # RequestOutput、StepStats、StepResult
│   │   └── errors.py                 # 在线领域异常
│   ├── layers/                       # Attention、Linear、Sampler 等算子层
│   ├── models/qwen3.py               # Qwen3 模型结构
│   └── serve/sse.py                  # FastAPI/SSE 适配层与 CLI
├── benchmarks/online_scheduler/
│   ├── workload.py                   # 固定、Poisson、barrier 负载
│   ├── runner.py                     # 动态注入、运行控制、GPU 遥测
│   ├── metrics.py                    # TTFT/TPOT/ITL/E2E/Goodput
│   └── cli.py                        # 可复现实验入口
├── tests/                            # 46 项纯 CPU 契约测试
├── docs/online_scheduler/            # 开发、API、实验与展示文档
├── reports/                          # 已归档的 RTX 4060 离线基线报告
├── scripts/                          # 环境采集与离线基线 runner
├── example.py                        # 离线推理示例
└── bench.py                          # 上游离线吞吐基线
```

## 文档导航

| 文档                                                                      | 用途                                                       |
| ------------------------------------------------------------------------- | ---------------------------------------------------------- |
| [RTX 4060 离线基线](docs/baseline-rtx4060.md)                             | 已验证的上游 `bench.py` 环境、结果、方法和限制             |
| [基线运行环境](docs/environment.md)                                       | WSL2、Python、PyTorch、CUDA、Triton 与 FlashAttention 版本 |
| [Milestone 0 说明](docs/milestone-00-baseline.md)                         | 离线 GPU 基线目标、验收标准和测量范围                      |
| [小负载生成报告](reports/rtx4060-qwen3-0.6b-baseline.md)                  | 8 请求离线 smoke workload 的机器生成结果                   |
| [在线调度器开发文档](docs/online_scheduler/DEV_DOCUMENT.md)               | 架构、状态机、不变量、配置和测试矩阵                       |
| [在线 API 契约](docs/online_scheduler/NANOVLLM_ONLINE_API.md)             | Python/SSE 字段、时序、状态码和错误语义                    |
| [调度器实现规范](docs/online_scheduler/NANOVLLM_SCHEDULER_STYLE.md)       | 线程所有权、编码约束、异常与清理规范                       |
| [开发日志](docs/online_scheduler/DEVELOPMENT_LOG.md)                      | 七天实施过程、节点证据和当前门禁                           |
| [RTX 4060 实验报告](docs/online_scheduler/EXPERIMENT_REPORT_RTX4060.md)   | 待真实数据填写的实验模板与结论边界                         |
| [WSL2 + RTX 4060 复现手册](docs/online_scheduler/WSL2_RTX4060_RUNBOOK.md) | 环境搭建、GPU 冒烟、正式实验和排障                         |
| [原始数据规范](docs/online_scheduler/results/README.md)                   | manifest、请求、token、step 和 summary schema              |
| [项目展示与面试讲解](docs/online_scheduler/PROJECT_SHOWCASE_ZH.md)        | 贡献边界、核心链路、难点和演示脚本                         |

## 能力边界与已知限制

- 当前是单机、单模型进程的实验型在线服务，不是生产级多租户平台；
- HTTP 层没有鉴权、TLS、配额、持久化、计费和生产级限流，默认只绑定 `127.0.0.1`；
- 调度粒度是 step-level phase scheduling，一次 ModelRunner 调用不混合 Prefill 与 Decode；
- `bounded_decode_first` 限制可执行 waiting 请求存在时的连续 Decode 步数，不承诺绝对墙钟公平或 SLO；
- 请求取消在 step 边界生效，不能立即中断正在执行的 GPU kernel；
- `/metrics` 是进程内 JSON 快照，不是 Prometheus exporter；
- 当前仅支持仓库已有的模型与算子路径，未实现动态模型加载或完整 OpenAI-compatible API；
- Tensor Parallel 来自上游基线，本阶段没有多卡实测结论；
- RTX 4060 离线基线已经验证；在线 SSE 正确性、三策略 TTFT/TPOT、在线吞吐和显存对比仍需按 runbook 生成原始产物后确认。

## 上游与二次开发边界

| 范围                                                                                         | 归属             | 代表代码                                                 |
| -------------------------------------------------------------------------------------------- | ---------------- | -------------------------------------------------------- |
| 模型执行、Paged KV Cache、Prefix Cache、FlashAttention、TP、CUDA Graph、基础 Chunked Prefill | 上游 Nano-vLLM   | `model_runner.py`、`block_manager.py`、`attention.py` 等 |
| 三种 Prefill/Decode 策略与 bounded 防饥饿                                                    | 当前分支二次开发 | `scheduler.py`、`config.py`                              |
| 动态请求 ID、逐 token 输出、取消与统计                                                       | 当前分支二次开发 | `llm_engine.py`、`outputs.py`、`errors.py`               |
| 单 worker 异步引擎与背压                                                                     | 当前分支二次开发 | `async_llm_engine.py`                                    |
| SSE 服务和实时 JSON 指标                                                                     | 当前分支二次开发 | `serve/sse.py`                                           |
| 动态负载、指标与证据归档                                                                     | 当前分支二次开发 | `benchmarks/online_scheduler/`                           |

推荐的准确项目表述是：

> 基于 Nano-vLLM 上游已有的 Paged KV Cache、Chunked Prefill 和模型执行链路，完成面向在线请求的调度二次开发：引入 Decode First 与 Bounded Decode First 策略，通过单 CUDA worker 支持动态接入、逐 token SSE、取消和背压，并以逐请求/逐 token/逐 step 原始产物构建 TTFT、TPOT 与吞吐 A/B 实验链路。

不应表述为“从零实现完整 Nano-vLLM”或在 GPU 数据缺失时声称具体加速百分比。

## License 与致谢

本项目沿用 [MIT License](LICENSE)。感谢 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 提供轻量、可读的推理引擎基线；当前在线调度工作建立在该上游实现之上。
