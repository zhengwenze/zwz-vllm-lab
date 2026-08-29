<p align="center">
  <img width="300" src="assets/logo.png" alt="Nano-vLLM Logo">
</p>

<p align="center">
  <strong>轻量、可读、可验证的大模型推理引擎与在线调度实验平台</strong>
</p>

<p align="center">
  <a href="docs/README.md">文档索引</a> ｜
  <a href="docs/online_scheduler/WSL2_RTX4060_RUNBOOK.md">RTX 4060 复现手册</a>
</p>

# Nano-vLLM：Decode 优先与防饥饿的在线调度二次开发

Nano-vLLM 是一个结构紧凑的 vLLM 风格推理引擎，上游已实现离线批量推理、Paged KV Cache、Prefix Cache、Chunked Prefill、Recompute 抢占、Tensor Parallel、FlashAttention 与 Decode CUDA Graph。

本分支在之上完成**在线调度二次开发**：把"批量提交、全部完成后返回"扩展为支持动态请求接入、逐 token 流式输出、请求取消、背压、调度指标与可重放实验的单机在线推理原型。核心研究问题：Prefill 与 Decode 竞争同一 GPU 时，如何降低在途请求的 token 间抖动，同时避免新请求长期拿不到首 token。

> [!IMPORTANT]
> 在线调度代码已实现并通过 46 项纯 CPU 契约测试；Milestone 0 离线基线与在线 CUDA/SSE 三策略 A/B 均已在 RTX 4060 8GB 完成。在线实验为 **3 策略 × 5 次独立重复（15/15 run 完成）**，脱敏原始证据、环境/配置指纹、聚合结果与 SHA-256 已公开归档。

## 项目状态

| 范围                           | 状态           | 证据                                                   |
| ------------------------------ | -------------- | ------------------------------------------------------ |
| Milestone 0：RTX 4060 离线基线 | `GPU Verified` | 256 序列 / 133,966 token / **1241.94 output tokens/s** |
| 在线调度二次开发               | `Implemented`  | 三策略、动态请求、SSE、取消、背压、benchmark           |
| 纯 CPU 逻辑验证                | `CPU Verified` | `pytest`：46 passed                                    |
| RTX 4060 在线 SSE / 三策略性能 | `GPU Verified` | [15-run 脱敏证据包](docs/results/online-scheduler-rtx4060-20260828/README.md)：请求/Token/Step 原始事件、GPU 遥测、环境指纹与聚合结果 |

固定 mixed Poisson workload（100 请求、8 req/s、seed 20260827）下，5 次重复均值显示：严格 `decode_first` 的 TTFT P50 / 输出吞吐为 108.9 s / 50.75 tok/s；`bounded_decode_first(K=8)` 为 4.580 s / 446.6 tok/s；`prefill_first` 为 2.118 s / 596.5 tok/s。该结论只适用于本仓库记录的模型、硬件与负载，不作跨模型或生产负载推广。

## 核心能力（当前分支新增）

- `prefill_first` / `decode_first` / `bounded_decode_first` 三种策略，用 `max_consecutive_decode_steps` 建立 TTFT↔TPOT 的有界折中；
- 外部稳定 `request_id`、逐 token 累计事件、`stop/length/abort` 三种终态、step 边界安全取消并释放 KV Block；
- `AsyncLLMEngine` 单 CUDA worker 独占，Scheduler 活跃请求背压与慢消费者保护；
- FastAPI/SSE 生成、取消、健康检查与 JSON 指标；fixed/Poisson/混合/干扰负载；
- 逐请求、逐 token、逐 step 的 JSON/JSONL 原始产物与 GPU 遥测，TTFT/TPOT/ITL/E2E/吞吐/Goodput 汇总。

## 安装与环境

支持 Linux；Windows 推荐 WSL2 Ubuntu 22.04，Python `>=3.10,<3.13`，NVIDIA CUDA GPU，推荐起步模型 Qwen3-0.6B。

```bash
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
# 先按本机驱动/CUDA 选择匹配的 PyTorch wheel，再安装项目
python -m pip install -e '.[online,test]'
```

下载模型：

```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir /YOUR/MODEL/PATH --local-dir-use-symlinks False
```

## 快速开始

离线推理（与上游 `LLM.generate` 兼容）：

```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
outputs = llm.generate(["Hello, Nano-vLLM."],
                        SamplingParams(temperature=0.6, max_tokens=128))
print(outputs[0]["text"])
```

启动在线 SSE 服务（RTX 4060 保守配置）：

```bash
nanovllm-serve --model /YOUR/MODEL/PATH \
  --scheduler-policy bounded_decode_first --max-consecutive-decode-steps 8 \
  --max-num-seqs 32 --max-num-batched-tokens 512 --max-model-len 2048 \
  --gpu-memory-utilization 0.75 --enforce-eager
```

流式生成 / 取消 / 健康与指标：

```bash
curl -N -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"request_id":"demo-001","prompt":"用一句话解释 PagedAttention","max_tokens":64}'
curl -X DELETE http://127.0.0.1:8000/requests/demo-001
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/metrics
```

## 可复现实验

在线策略比较使用 `benchmarks/online_scheduler/`（离线 `bench.py` 仅作吞吐基线），比较三策略时保持模型、commit、seed、请求 trace 与重复次数一致：

```bash
python -m benchmarks.online_scheduler.cli \
  --model /YOUR/MODEL/PATH --policy bounded_decode_first \
  --max-consecutive-decode-steps 8 --workload mixed \
  --arrival poisson --repeat-index 0 --output-root artifacts/online_scheduler
```

每次运行生成独立目录：`workload.jsonl`（可重放 trace）、`manifest.json`、`requests/tokens/steps.jsonl`、`gpu_telemetry.csv`、`summary.json`。指标口径 TTFT/ITL/TPOT/E2E/吞吐/Goodput 定义见 [原始数据规范](docs/online_scheduler/results/README.md)。

## 文档导航

- [完整文档索引](docs/README.md)｜[在线调度开发文档](docs/online_scheduler/DEV_DOCUMENT.md)｜[在线 API 契约](docs/online_scheduler/NANOVLLM_ONLINE_API.md)
- [WSL2 + RTX 4060 复现手册](docs/online_scheduler/WSL2_RTX4060_RUNBOOK.md)｜[RTX 4060 离线基线](docs/baseline-rtx4060.md)
- [RTX 4060 在线 15-run 报告](reports/nanovllm-online-rtx4060-20260828.md)｜[脱敏原始证据包](docs/results/online-scheduler-rtx4060-20260828/README.md)
- [项目复盘与下一步](docs/PROJECT_GAPS_AND_NEXT_STEPS_ZH.md)｜[模拟面试 Q&A](docs/interview/NANOVLLM_MOCK_INTERVIEW_QA.md)

## 能力边界与二次开发归属

当前为单机、单模型进程的实验型在线服务；HTTP 层无鉴权/TLS/配额，默认只绑定 `127.0.0.1`；调度为 step-level phase，不混合 Prefill 与 Decode；`bounded_decode_first` 只限定在可执行 waiting 请求存在时的连续 Decode 步数，不承诺绝对 SLO。

推荐的准确表述：基于上游已有的 Paged KV Cache、Chunked Prefill 与模型执行链路，完成在线调度二次开发——引入 Decode First 与 Bounded Decode First，以单 CUDA worker 支持动态接入、逐 token SSE、取消与背压，并以逐请求/逐 token/逐 step 原始产物构建 TTFT、TPOT 与吞吐 A/B 实验链路。**不应表述为“从零实现完整 Nano-vLLM”；性能数字必须同时限定为本次 RTX 4060 / Qwen3-0.6B / 固定 workload 的 5-repeat 结果。**

## License

[MIT License](LICENSE)，致谢 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)。
