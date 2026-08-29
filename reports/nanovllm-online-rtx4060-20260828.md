# Nano-vLLM RTX 4060 在线调度 5 次独立重复 GPU 实验报告

实验日期：2026-08-28（Asia/Shanghai）  
证据状态：**GPU Verified（3 策略 × 5 次独立重复 = 15 runs，轮换执行顺序）**  
Git commit：`73b9118838af8ef17d1587de8bd3cca037d50263`（与单轮首测一致）

## 1. 实验范围

本报告是 [单轮首测报告](./nanovllm-online-rtx4060-20260827.md) 的升级：在固定 workload 下对三种在线调度策略各执行 **5 次独立重复**，共 **15 个正式 run**，并把执行顺序**逐轮轮换**，以降低 thermal / CPU cache / 运行顺序造成的偏差。

目标是把“跑了一次、看到显著差异”升级为：**固定 workload 下 5 次独立重复的受控 Scheduler A/B/C benchmark，通过 mean / std / CV 验证结果稳定性，并保留逐请求、逐 token、Scheduler state 与 GPU telemetry 原始证据**。

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

## 3. 固定实验配置（显式固化）

> ⚠️ 该仓库 CLI 的默认 workload 为 `300 requests / 2 req/s / seed=20260822`，与首测报告的 100/8.0/20260827 并不一致。为避免“参数漂移”，正式脚本 `scripts/run_scheduler_rtx4060_repeats.sh` **显式固定**以下参数，每次运行完全相同。

| 参数 | 值 |
| --- | --- |
| Model | Qwen3-0.6B |
| Workload | mixed Poisson |
| num_requests | **100**（CLI 默认 300） |
| request_rate | **8.0 req/s**（CLI 默认 2.0） |
| seed | **20260827**（CLI 默认 20260822） |
| enforce_eager | **true** |
| max_model_len | 2048 |
| max_num_seqs | 32 |
| max_num_batched_tokens | 512 |
| gpu_memory_utilization | 0.75 |
| bounded K | 8 decode steps |
| GPU telemetry interval | 0.2 s |

执行顺序按轮次轮换：`P-D-B → D-B-P → B-P-D → P-D-B → D-B-P`（P=prefill_first, D=decode_first, B=bounded_decode_first）。

## 4. 跨 repeat 聚合结果（每策略 n = 5）

### 4.1 核心指标 mean ± std / CV / min–max

| 策略 | 指标 | mean | std | CV | min | max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| prefill_first | Throughput (tok/s) | 596.5 | 29.1 | 4.9% | 550.2 | 622.5 |
| prefill_first | TTFT P50 | 2.118 s | 328 ms | 15.5% | 1.803 s | 2.675 s |
| prefill_first | TTFT P95 | 5.122 s | 774 ms | 15.1% | 4.269 s | 6.285 s |
| prefill_first | TPOT P50 | 44.15 ms | 2.78 ms | 6.3% | 41.16 ms | 48.68 ms |
| prefill_first | TPOT P95 | 46.92 ms | 3.30 ms | 7.0% | 43.01 ms | 51.66 ms |
| prefill_first | E2E P50 | 7.721 s | 730 ms | 9.5% | 6.989 s | 8.883 s |
| prefill_first | E2E P95 | 9.743 s | 935 ms | 9.6% | 8.949 s | 11.25 s |
| decode_first | Throughput (tok/s) | 50.75 | 1.27 | 2.5% | 49.52 | 52.20 |
| decode_first | TTFT P50 | 108.9 s | 5.78 s | 5.3% | 101.7 s | 117.2 s |
| decode_first | TTFT P95 | 216.5 s | 6.18 s | 2.9% | 209.4 s | 223.0 s |
| decode_first | TPOT P50 | 32.01 ms | 0.87 ms | 2.7% | 30.91 ms | 33.01 ms |
| decode_first | TPOT P95 | 36.34 ms | 2.94 ms | 8.1% | 34.52 ms | 41.48 ms |
| decode_first | E2E P50 | 112.6 s | 5.90 s | 5.2% | 105.1 s | 120.9 s |
| decode_first | E2E P95 | 220.4 s | 6.06 s | 2.7% | 213.7 s | 226.7 s |
| bounded_decode_first | Throughput (tok/s) | 446.6 | 14.6 | 3.3% | 433.5 | 471.0 |
| bounded_decode_first | TTFT P50 | 4.580 s | 323 ms | 7.1% | 4.044 s | 4.833 s |
| bounded_decode_first | TTFT P95 | 12.00 s | 615 ms | 5.1% | 11.00 s | 12.63 s |
| bounded_decode_first | TPOT P50 | 38.87 ms | 1.17 ms | 3.0% | 37.03 ms | 40.08 ms |
| bounded_decode_first | TPOT P95 | 40.52 ms | 1.43 ms | 3.5% | 38.14 ms | 41.65 ms |
| bounded_decode_first | E2E P50 | 9.331 s | 455 ms | 4.9% | 8.635 s | 9.771 s |
| bounded_decode_first | E2E P95 | 16.40 s | 833 ms | 5.1% | 15.02 s | 17.15 s |

### 4.2 每策略一句话汇总

**prefill_first**
- Throughput：596.5 ± 29.1 tok/s（CV = 4.9%）
- TTFT P50：2.118 ± 0.328 s（CV = 15.5%）
- TPOT P50：44.15 ± 2.78 ms（CV = 6.3%）
- E2E P50：7.721 ± 0.730 s（CV = 9.5%）

**decode_first**
- Throughput：50.75 ± 1.27 tok/s（CV = 2.5%）
- TTFT P50：108.9 ± 5.8 s（CV = 5.3%）
- TPOT P50：32.01 ± 0.87 ms（CV = 2.7%）
- E2E P50：112.6 ± 5.9 s（CV = 5.2%）

**bounded_decode_first (K=8)**
- Throughput：446.6 ± 14.6 tok/s（CV = 3.3%）
- TTFT P50：4.580 ± 0.323 s（CV = 7.1%）
- TPOT P50：38.87 ± 1.17 ms（CV = 3.0%）
- E2E P50：9.331 ± 0.455 s（CV = 4.9%）

## 5. 策略对比（由 5 次重复的 mean 计算）

| 对比 | Throughput | TTFT P50 | TTFT P95 | TPOT P50 | TPOT P95 | E2E P50 | E2E P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bounded vs decode | **+780%** | **−95.8%** | −94.5% | +21.4% | +11.5% | −91.7% | −92.6% |
| prefill vs bounded | **+33.6%** | **−53.8%** | −57.3% | +13.6% | +15.8% | −17.3% | −40.6% |

解读：
- 严格 `decode_first` 在开放到达负载下让新请求长期等待（最大 waiting = 95、最大 running = 4），TTFT P50 高达 108.9 s，但 TPOT P50 最低（32.0 ms）。
- `bounded_decode_first(K=8)` 通过最多 8 步连续 Decode 后强制 Prefill，将 TTFT P50 从 108.9 s 压到 4.58 s（−95.8%），吞吐从 50.8 提升到 446.6 tok/s（+780%），代价是 TPOT P50 上升 21.4%（32.0 → 38.9 ms）。5 次重复中 TTFT P50 稳定在 4.04–4.83 s 区间。
- `prefill_first` 在该 workload 下吞吐（596.5 tok/s）与 TTFT（2.12 s）均为最佳，相对 bounded 吞吐 +33.6%、TTFT P50 −53.8%，但 TPOT P50 高 13.6%。

## 6. 调度与 GPU 证据（跨 repeat mean）

| 策略 | 最大 waiting | 最大 running | 强制 Prefill steps | KV 块峰值利用率 | GPU 平均利用率 | GPU 峰值利用率 | 峰值显存 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prefill_first | 34.2 | 32 | 0 | 52.1% | 38.9% | 58% | 6331 MiB |
| decode_first | 95 | 4 | 0 | 3.7% | 27.7% | 48% | 6391 MiB |
| bounded_decode_first | 48.2 | 28 | 73 | 31.9% | 36.7% | 46% | 6339 MiB |

- `decode_first` 最大 running 仅 4（一次只服务少量请求），GPU 利用率最低，与 TTFT 恶化的调度行为一致。
- `bounded_decode_first` 实际触发 73 次强制 Prefill，把最大 running 恢复到 28，这与首测的单轮证据一致（跨 repeat 稳定）。

## 7. 数据完整性校验

- 15/15 个 run 全部有效：`offered/admitted/finished = 100/100/100`，completion rate = 1.0。
- 无任何 `failure.json`。
- 每个 run 均含 `manifest.json`、`workload.jsonl`、`requests.jsonl`、`tokens.jsonl`、`steps.jsonl`、`gpu_telemetry.csv`、`summary.json`、`stdout.log`。
- 每策略恰好 5 个 run（repeat_index = 0..4），workload 指纹（100 / 8.0 / 20260827 / eager）一致。

## 8. 原始证据

- 公开聚合 JSON：[`docs/results/online-scheduler-rtx4060-20260828/aggregate.json`](../docs/results/online-scheduler-rtx4060-20260828/aggregate.json)
- 15 个正式 run 的脱敏原始事件与遥测：[`docs/results/online-scheduler-rtx4060-20260828/runs/`](../docs/results/online-scheduler-rtx4060-20260828/runs/)
- 原始文件哈希与公开文件校验：[`source-sha256.json`](../docs/results/online-scheduler-rtx4060-20260828/source-sha256.json) / [`checksums.sha256`](../docs/results/online-scheduler-rtx4060-20260828/checksums.sha256)
- 生成方式：`python -m benchmarks.online_scheduler.aggregate_repeats --artifacts-root artifacts/online_scheduler/experiments/20260828-ablation --output-json artifacts/online_scheduler/experiments/20260828-ablation/aggregate.json`
- 复现脚本：`scripts/run_scheduler_rtx4060_repeats.sh`
