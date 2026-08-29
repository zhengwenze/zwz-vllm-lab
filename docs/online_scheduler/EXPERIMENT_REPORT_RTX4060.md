# Nano-vLLM 在线调度 RTX 4060 实验报告模板（历史）

> **状态标签**：历史模板。正式实验已为 `GPU Verified`，请阅读[15-run 最终报告](../../reports/nanovllm-online-rtx4060-20260828.md)。
>
> **重要**：本文保留实验前模板，不再代表当前项目状态。正式数字只能引用最终报告及其公开原始证据包。

## 1. 实验问题

在固定模型、请求集合、采样参数和服务版本下：

1. `prefill_first`、`decode_first`、`bounded_decode_first` 对 TTFT、TPOT、E2E 和吞吐分别有什么影响？
2. `bounded_decode_first` 是否把连续 Decode 控制在配置上限内？
3. 请求取消后 KV Block 是否回收？
4. 队列背压和抢占发生时，错误率与尾延迟怎样变化？

## 2. 证据状态

| 项目 | 状态 | 证据位置 |
|---|---|---|
| 实验协议 | `Implemented` | 本文 |
| 原始数据 schema | `Implemented` | `results/README.md` |
| CPU 契约测试 | `CPU Verified` | `46 passed`；Ruff/compileall/diff-check 通过 |
| RTX 4060 环境探测 | `GPU Verified` | 15 个 `manifest.public.json` |
| 策略 A/B 原始数据 | `GPU Verified` | 15-run 脱敏 JSONL/CSV 证据包 |
| 性能结论 | `GPU Verified` | 5-repeat 聚合报告 |

## 3. 固定环境

所有值由命令采集，不凭记忆填写。

| 项 | 值 |
|---|---|
| 日期与时区 | — |
| Git commit | — |
| Windows 版本 | — |
| WSL 版本 / Linux kernel | — |
| Linux 发行版 | — |
| GPU | NVIDIA GeForce RTX 4060 |
| NVIDIA driver | — |
| CUDA runtime/toolkit | — |
| Python | 3.11（正式 WSL 固定；开发主机 3.13 只跑纯 CPU 测试） |
| PyTorch | — |
| Transformers | — |
| Triton | — |
| FlashAttention | — |
| 模型路径与 revision | — |
| 模型 dtype | — |
| `enforce_eager` | — |

环境命令：

```bash
git rev-parse HEAD
uname -a
nvidia-smi
python --version
python -m pip freeze
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

把完整输出保存到实验目录；报告只做摘要。

## 4. 固定工作负载

| 变量 | 固定值 |
|---|---|
| 模型 | Qwen3-0.6B（准确路径/revision 待填） |
| 数据集/生成方式 | — |
| 请求数 | — |
| 输入长度桶 | — |
| `max_tokens` | — |
| `temperature` | — |
| `ignore_eos` | 建议 `true` 用于固定输出长度的性能比较 |
| 到达过程 | closed-loop / fixed-rate / Poisson，三选一并固定 |
| 并发 | — |
| 随机种子 | — |
| warmup 轮数 | — |
| 正式重复轮数 | — |
| 超时 | — |

不要在同一个比较中同时改变调度策略、并发、输入长度和 CUDA Graph 开关。

## 5. 实验矩阵

### 5.1 主实验：策略比较

| Run ID | Policy | Decode 上限 | 并发 | 重复 | 状态 |
|---|---|---|---|---|---|
| P1 | `prefill_first` | 不适用 | — | — | Pending |
| D1 | `decode_first` | 不适用 | — | — | Pending |
| B1 | `bounded_decode_first` | `8` | — | — | Pending |

### 5.2 压力实验

- 小 `max_queue_size`，验证 `429` 与已接收请求不丢失。
- 小 KV Cache 预算，触发抢占并验证最终块回收。
- 生成中取消固定比例请求，分离正常完成与 abort 指标。
- 同一前缀与不同前缀两组负载，观察 Prefix Cache 使用行为。

### 5.3 Barrier interference 实验

`--workload interference` 专门制造 Prefill/Decode 竞争，而不是依赖随机到达“碰巧重叠”：

1. 先提交一组 Decode-heavy 请求。
2. 等这些初始请求全部产生首 token，Runner 记录 `barrier_ns`。
3. 再从该 barrier 起每隔 100 ms 注入一个长 Prompt 请求。
4. 比较三种策略在注入窗口中的 TPOT、TTFT、mode switches、decode gap 与 forced prefill。

`RequestSpec.arrival_anchor` 只能为 `start` 或 `barrier`。barrier 请求的 `arrival_ns = barrier_ns + arrival_offset_ns`，因此排队时间仍从理想到达计算，不因同步 engine step 遮蔽等待。

## 6. 指标定义

| 指标 | 定义 |
|---|---|
| TTFT | 服务接收请求到第一个 token 事件 |
| TPOT | 相邻输出 token 事件间隔；少于两个输出 token 时不可计算 |
| E2E | 服务接收请求到终态事件 |
| Output throughput | 正常完成输出 token 总数 / 正式测量墙钟时间 |
| Goodput | 同时满足预先指定 TTFT/TPOT SLO 的正常完成请求速率 |
| Preemption rate | 抢占次数 / 调度步或请求数，分母必须注明 |
| KV high-watermark | `kv_used_blocks / kv_total_blocks` 的最大值 |
| Error/abort rate | error 与 abort 分开统计，不混入正常性能样本 |
| Mode switches | 相邻非空 batch kind 发生变化的次数 |
| Max decode streak | 观察到的最长连续 Decode 步数 |
| Max decode gap | 相邻 Decode 之间最多插入的非 Decode 步数 |
| Max queue | waiting/running 请求数的运行期峰值 |

benchmark 可由请求 ID 快照真实累计每请求 `waiting_steps` 和 `preemption_count`。当前无法严格归因每请求的重算 token，`recomputed_tokens` 必须保持 `null`，不能用零代替未知。

## 7. 正确性门禁

任一项失败，本轮标记 `INVALID`，不得输出性能优劣结论：

- 接收请求数 = 正常终态 + abort 终态 + error。
- 每个请求恰好一个终态，终态后无 token。
- 累计 token 数不超过配置上限。
- 无重复 request ID 被静默接受。
- KV used 不超过 total；全部请求结束后回到预期基线。
- Bounded 策略在 waiting 可执行时不超过连续 Decode 上限。
- 原始记录包含 commit、环境、配置、请求和逐 step 数据。

环境不完整、采样不足或原始文件缺失时标记 `INCONCLUSIVE`，不要硬写结论。

## 8. 执行步骤

1. 按 `WSL2_RTX4060_RUNBOOK.md` 完成环境检查。
2. 固定 commit 和配置，保存 environment JSON。
3. 运行 CPU 契约测试；失败则停止 GPU 实验。
4. 对每个 case 先 warmup，warmup 数据不进入正式统计。当前 benchmark CLI 已内置独立 warmup。
5. 按随机化或轮换顺序执行各策略，减少温度/后台负载漂移。
6. 每轮生成 request JSONL、step JSONL 和 summary JSON。
7. 运行 schema 与不变量校验。
8. 只有有效重复都通过后，填写汇总表和结论。

当前 benchmark CLI 示例（每个策略分别运行）：

```bash
python -m benchmarks.online_scheduler.cli \
  --model /home/$USER/models/Qwen3-0.6B \
  --policy bounded_decode_first \
  --max-consecutive-decode-steps 8 \
  --workload mixed \
  --arrival poisson \
  --repeat-index 0 \
  --output-root artifacts/online_scheduler
```

替换 `--policy` 运行另两种策略；每次正式重复只改变 `--repeat-index`。CLI 会写 workload、manifest、逐请求、逐 token、逐 step、summary、`stdout.log` 和可选 GPU telemetry；异常时写 `failure.json`，不能删除失败轮。

干扰实验命令：

```bash
python -m benchmarks.online_scheduler.cli \
  --model /home/$USER/models/Qwen3-0.6B \
  --policy bounded_decode_first \
  --workload interference \
  --interference-decode-requests 8 \
  --interference-prefill-requests 8 \
  --injection-interval-ms 100 \
  --repeat-index 0 \
  --output-root artifacts/online_scheduler
```

## 9. 结果表（保持空白直到实测）

### 9.1 请求指标

| Policy | Valid runs | TTFT P50 (ms) | TTFT P95 (ms) | TPOT P50 (ms) | TPOT P95 (ms) | E2E P95 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| `prefill_first` | — | — | — | — | — | — |
| `decode_first` | — | — | — | — | — | — |
| `bounded_decode_first` | — | — | — | — | — | — |

### 9.2 系统指标

| Policy | Output tok/s | Goodput req/s | KV high-watermark | Preemptions | Forced Prefills | Errors |
|---|---:|---:|---:|---:|---:|---:|
| `prefill_first` | — | — | — | — | — | — |
| `decode_first` | — | — | — | — | — | — |
| `bounded_decode_first` | — | — | — | — | — | — |

### 9.3 调度行为

| Policy | Mode switches | Max decode streak | Max decode gap | Max waiting | Max running |
|---|---:|---:|---:|---:|---:|
| `prefill_first` | — | — | — | — | — |
| `decode_first` | — | — | — | — | — |
| `bounded_decode_first` | — | — | — | — | — |

### 9.4 显存与功耗（可选）

| Policy | Peak VRAM | Avg power | GPU util | 采集方法 |
|---|---:|---:|---:|---|
| `prefill_first` | — | — | — | — |
| `decode_first` | — | — | — | — |
| `bounded_decode_first` | — | — | — | — |

## 10. 结论模板

在结果有效前不要删除以下占位提示：

> 历史模板验收状态：固定负载重复实验已于 2026-08-28 完成；具体结论和限制见最终报告。

实测后按以下格式写：

- **观察**：只描述表中可复算的数值和分布。
- **机制解释**：引用 `StepStats` 中的 batch 顺序、forced prefill、抢占和 KV 使用。
- **代价**：说明一个指标改善是否牺牲另一个指标。
- **适用边界**：限制到该模型、该显卡、该负载、该 commit。
- **下一步**：提出只改变一个变量的新实验。

## 11. 失败与异常记录

| 时间 | Run ID | 症状 | 原因 | 处理 | 是否重跑 |
|---|---|---|---|---|---|
| — | — | — | — | — | — |

OOM、worker crash、温度降频、后台 GPU 占用、schema 缺字段都要记录，不能只保留“最好的一轮”。

## 12. 原始证据索引

| 文件 | SHA-256 | 内容 | 状态 |
|---|---|---|---|
| `artifacts/online_scheduler/<run_id>/manifest.json` | — | 环境、配置、workload | Pending |
| `artifacts/online_scheduler/<run_id>/workload.jsonl` | — | 可重放到达与 token 长度 | Pending |
| `artifacts/online_scheduler/<run_id>/requests.jsonl` | — | 逐请求汇总 | Pending |
| `artifacts/online_scheduler/<run_id>/tokens.jsonl` | — | 逐 token 时间戳 | Pending |
| `artifacts/online_scheduler/<run_id>/steps.jsonl` | — | 逐调度步 | Pending |
| `artifacts/online_scheduler/<run_id>/gpu_telemetry.csv` | — | nvidia-smi 遥测 | Pending |
| `artifacts/online_scheduler/<run_id>/summary.json` | — | 可复算汇总 | Pending |
| `artifacts/online_scheduler/<run_id>/stdout.log` | — | CLI 最终摘要 | Pending |
| `artifacts/online_scheduler/<run_id>/failure.json` | — | 失败现场，仅异常时 | Pending |

原始数据字段见 `results/README.md`。
