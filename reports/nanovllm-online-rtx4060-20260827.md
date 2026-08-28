# Nano-vLLM RTX 4060 在线调度 GPU 实验报告

> 2026-08-28 已升级为 **3 策略 × 5 次独立重复**（15 runs，轮换顺序 + mean/std/CV 聚合），请以 [nanovllm-online-rtx4060-20260828.md](./nanovllm-online-rtx4060-20260828.md) 为终版结论来源；本文件为单轮首测的历史证据。

实验日期：2026-08-27（Asia/Shanghai）  
证据状态：**GPU Verified（单轮首测）**  
Git commit：`73b9118838af8ef17d1587de8bd3cca037d50263`

## 1. 实验范围

本报告记录 Nano-vLLM 在线调度链路在真实 NVIDIA GeForce RTX 4060 8 GB 上的首次三策略 A/B 实验。实验包含 CUDA 模型执行、在线 SSE 冒烟、动态请求重放、逐请求/逐 token/逐 step 记录以及 GPU 遥测。

本次每种策略只运行 1 次，因此数据可以证明链路真实可运行并展示当前 workload 下的显著行为差异，但不能替代多次重复、随机化执行顺序和置信区间分析。

## 2. 环境

| 项目 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4060, 8188 MiB |
| Windows driver | 560.94 |
| WSL | WSL2, Linux 6.18.33.2-microsoft-standard-WSL2 |
| Python | 3.12.14 |
| PyTorch | 2.7.1+cu126 |
| CUDA runtime | 12.6 |
| Triton | 3.3.1 |
| FlashAttention | 2.8.3 |
| Transformers | 4.57.1 |
| Model | Qwen3-0.6B（本地完整权重） |
| 执行模式 | 单卡、Tensor Parallel 1、eager |

CPU/契约门禁结果为 `46 passed`，`compileall` 和 `git diff --check` 均通过。在线 SSE 冒烟成功生成 16 个 token，并得到唯一 `finish_reason=length` 终态。

## 3. 固定实验配置

三种策略使用完全相同的 workload 文件，SHA-256 均为：

`96783189e12290784ff9ae9228bc74113b530637d65dc5576e80872899e5c8e6`

| 参数 | 值 |
| --- | --- |
| Workload | mixed Poisson |
| 请求数 | 100 |
| 到达率 | 8 requests/s |
| Seed | 20260827 |
| 输出 token 总数 | 12,288 |
| Prompt buckets | 128 / 512 / 1024 tokens |
| Output buckets | 128 / 128 / 64 tokens |
| max_model_len | 2048 |
| max_num_seqs | 32 |
| max_num_batched_tokens | 512 |
| max_queue_size | 512 |
| gpu_memory_utilization | 0.75 |
| KV blocks | 134 × 256 tokens |
| GPU telemetry interval | 0.2 s |
| bounded K | 8 decode steps |

## 4. 真实 GPU 结果

| 策略 | 墙钟 s | output tok/s | req/s | TTFT P50 ms | TTFT P95 ms | TPOT P50 ms | TPOT P95 ms | E2E P50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prefill_first | 20.819 | 590.24 | 4.803 | 2,043.71 | 4,971.27 | 43.94 | 46.07 | 7,601.84 |
| decode_first | 241.825 | 50.81 | 0.414 | 112,835.02 | 216,784.43 | 31.32 | 35.22 | 116,973.25 |
| bounded_decode_first (K=8) | 27.151 | 452.58 | 3.683 | 5,033.00 | 11,955.07 | 38.11 | 42.90 | 9,637.34 |

补充分位数：

| 策略 | TTFT mean/P99 ms | TPOT mean/P99 ms | ITL P95 ms | E2E P95 ms |
| --- | ---: | ---: | ---: | ---: |
| prefill_first | 2,214.50 / 5,172.04 | 43.14 / 49.22 | 76.93 | 9,841.12 |
| decode_first | 110,096.93 / 223,294.73 | 31.53 / 35.22 | 37.86 | 220,338.36 |
| bounded_decode_first (K=8) | 5,503.75 / 12,094.47 | 38.45 / 42.97 | 69.95 | 16,075.96 |

## 5. 调度与 GPU 证据

| 策略 | 总 step | Prefill / Decode | 强制 Prefill | 最大 waiting | 最大 running | KV 峰值 | GPU 平均/峰值利用率 | 峰值显存 | 最高温度 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prefill_first | 565 | 96 / 469 | 0 | 34 | 32 | 52.24% | 43.22% / 66% | 6174 MiB | 52 C |
| decode_first | 7816 | 73 / 7743 | 0 | 95 | 4 | 3.73% | 34.25% / 69% | 6257 MiB | 52 C |
| bounded_decode_first (K=8) | 788 | 74 / 714 | 73 | 49 | 28 | 32.09% | 44.66% / 65% | 6265 MiB | 54 C |

严格 `decode_first` 在开放到达负载下让已有请求持续 Decode，新请求长时间留在 waiting：最大 waiting 达 95，而最大 running 只有 4。这使 TPOT 更低，但 TTFT、E2E 和整体吞吐严重恶化。

`bounded_decode_first(K=8)` 实际触发 73 次强制 Prefill，将最大 running 恢复到 28。相对严格 `decode_first`，它将 TTFT P50 从 112.84 秒降到 5.03 秒，并把吞吐从 50.81 提升到 452.58 output token/s；代价是 TPOT P50 从 31.32 ms 上升到 38.11 ms。

在这一个 workload 和单轮数据中，`prefill_first` 的 TTFT 与吞吐最佳，bounded 策略的 TPOT 更低但 TTFT 更高。不能据此声称 bounded 在所有场景优于 prefill；下一阶段应增加 interference workload、多次重复和随机化策略顺序。

## 6. 数据完整性校验

- 三轮均为 offered/admitted/finished `100/100/100`，completion rate 100%。
- 三轮请求 ID 均唯一，所有请求只有一个 finished 终态。
- 三轮实际输出长度全部等于请求长度，token 总数均为 12,288。
- 所有请求满足 `arrival <= first_token <= finish`。
- 所有 step 的 KV 使用量均在 `[0, kv_total_blocks]` 内。
- 三轮均有 GPU 遥测，且没有 `failure.json`。
- Linux 原始产物复制到 Windows 项目后，8 个核心文件逐项通过 SHA-256 校验。

## 7. 原始证据

原始证据位于：

- `artifacts/validation/20260827/bootstrap/`
- `artifacts/online_scheduler/experiments/20260827-initial/20260827T151006.462573Z-prefill_first-poisson-mixed/`
- `artifacts/online_scheduler/experiments/20260827-initial/20260827T151059.920426Z-decode_first-poisson-mixed/`
- `artifacts/online_scheduler/experiments/20260827-initial/20260827T151530.490539Z-bounded_decode_first-poisson-mixed/`

每个正式 run 均包含 `manifest.json`、`workload.jsonl`、`requests.jsonl`、`tokens.jsonl`、`steps.jsonl`、`gpu_telemetry.csv`、`summary.json`、`stdout.log` 和 `checksums.sha256`。
