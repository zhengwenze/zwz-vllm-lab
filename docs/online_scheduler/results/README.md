# 在线调度实验原始数据规范

> **状态标签**：`Implemented`（benchmark artifacts） · `CPU Verified`（46 tests passed） · `GPU Verified`（RTX 4060 三策略 × 5 次重复；[公开证据包](../../results/online-scheduler-rtx4060-20260828/README.md)）

benchmark 默认把真实运行产物写到 `artifacts/online_scheduler/<run_id>/`。2026-08-28 的 15-run 正式实验已将脱敏、压缩后的完整事件流与校验哈希发布到 `docs/results/online-scheduler-rtx4060-20260828/`；本页继续定义通用 schema 与证据规则。

## 1. 实际目录结构

```text
artifacts/online_scheduler/<run_id>/
├── workload.jsonl
├── manifest.json
├── requests.jsonl
├── tokens.jsonl
├── steps.jsonl
├── gpu_telemetry.csv
├── summary.json
├── stdout.log
└── failure.json             # 仅失败时
```

`run_id` 由 UTC 时间、policy 和 workload 组成。CLI 使用 `exist_ok=False`，不会覆盖同名 run；失败轮也应保留日志，不只挑最快结果。

## 2. `workload.jsonl`

每行是一个可重放请求：

```json
{"request_id":"request-000000","arrival_offset_ns":0,"prompt_token_ids":[1,2],"prompt_tokens":2,"output_tokens":128,"bucket":"short","arrival_anchor":"start"}
```

约束：

- ID 唯一，`arrival_offset_ns` 非负且在各 anchor 组内单调。
- prompt token 非空且为非负整数。
- 输出长度为正数。
- `arrival_anchor` 只能为 `start` 或 `barrier`；start 组必须在 barrier 组之前。
- 固定到达、Poisson 到达和 barrier interference 都由种子确定；同一 trace 可跨策略重放。

Barrier interference 先运行 Decode-heavy 初始组；只有全部初始请求产生首 token 后，Runner 才记录 `barrier_ns` 并按 100 ms 间隔注入长 Prompt 组。它用于稳定制造 Prefill/Decode 干扰。

这份 workload 会保存 token IDs。若输入来自真实用户，不得直接写入；应使用获授权的公开数据或合成 token。

## 3. `manifest.json`

实际 CLI 写入：

```json
{
  "run_id": "",
  "created_utc": "",
  "git_commit": "",
  "git_status_porcelain": "",
  "git_dirty": false,
  "nvidia_smi": "",
  "model": "",
  "policy": "bounded_decode_first",
  "policy_params": {"max_consecutive_decode_steps": 8},
  "repeat_index": 0,
  "engine_config": {},
  "workload": {},
  "software": {
    "python": "",
    "platform": "",
    "torch": "",
    "cuda": "",
    "transformers": "",
    "triton": "",
    "flash_attn": null
  },
  "kv_cache": {
    "block_size_tokens": null,
    "num_blocks": null,
    "token_capacity": null
  },
  "warmup": {"completed": true},
  "telemetry": {},
  "benchmark": {"barrier_ns": null}
}
```

模板中的空值不是实测数据。正式 run 必须包含 commit、`git_dirty`、repeat index、模型路径、软件版本、实际 KV Block 数、warmup 和 workload；干扰实验还必须有 `barrier_ns`。

## 4. `requests.jsonl`

每行一个请求汇总：

```json
{"request_id":"request-000000","bucket":"short","arrival_ns":0,"admitted_ns":null,"prompt_tokens":128,"requested_output_tokens":128,"actual_output_tokens":0,"first_token_ns":null,"finish_ns":null,"status":"pending","waiting_steps":0,"preemption_count":0,"recomputed_tokens":null}
```

字段语义：

- `arrival_ns` 是理想到达时间；TTFT/E2E 从它开始，避免 coordinated omission。
- `admitted_ns` 是实际加入引擎时间；两者差是 admission delay。
- `waiting_steps` 从逐 step 的 `waiting_request_ids` 真实累计。
- `preemption_count` 从 `preempted_request_ids` 真实累计。
- 当前协议不能严格归因每请求重算 token，所以 `recomputed_tokens` 必须为 `null`，不能填 `0`。
- `status`、首 token、完成时间和实际输出数共同构成生命周期门禁。

## 5. `tokens.jsonl`

每行一个真实输出 token 事件：

```json
{"request_id":"request-000000","output_index":0,"token_id":123,"emitted_ns":0}
```

同一请求 `output_index` 从零连续递增，`emitted_ns` 单调。TTFT、ITL 和 TPOT 都从这些时间戳复算；不能只保存聚合分位数。

## 6. `steps.jsonl`

每行对应一个 `step_stream()`：

```json
{"step_id":0,"batch_kind":"prefill","batch_size":1,"scheduled_tokens":128,"waiting":0,"running":1,"kv_used_blocks":1,"kv_total_blocks":10,"preemptions":0,"forced_prefill":false,"allocation_blocked":false,"elapsed_ms":0.0,"scheduled_request_ids":["request-000000"],"preempted_request_ids":[],"waiting_request_ids":[],"running_request_ids":["request-000000"],"waiting_before":1,"running_before":0,"decode_streak":0,"runner_start_ns":0,"runner_finish_ns":0}
```

约束：

- `step_id` 在 run 内严格递增；`batch_kind` 为 `prefill/decode/idle`。
- `0 <= kv_used_blocks <= kv_total_blocks`。
- `scheduled_request_ids` 对应本步选中的外部请求。
- `preempted_request_ids` 是本步真实被抢占请求。
- `waiting_request_ids/running_request_ids` 是后处理队列快照；`*_before` 是调度前计数。
- `decode_streak` 支持验证 bounded 策略，不从日志文本猜测。
- `recomputed_tokens` 仍不可由这些字段严格推出，因此保持 unknown。

## 7. `gpu_telemetry.csv`

通过 `nvidia-smi` 周期采样，列包括：

```text
sampled_perf_counter_ns,timestamp,name,uuid,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm
```

遥测是 best effort：不可用时 benchmark 可以继续，但 manifest 必须记录错误，此时功耗/利用率结论标记 `INCONCLUSIVE`。

## 8. `summary.json`

由 `MetricsCollector` 从原始请求、token 和 step 生成：

```json
{
  "requests": {
    "offered": 0,
    "admitted": 0,
    "finished": 0,
    "completion_rate": 0.0,
    "exact_output_length": 0
  },
  "latency": {
    "admission_delay": {},
    "ttft": {},
    "tpot": {},
    "itl": {},
    "e2e": {}
  },
  "throughput": {
    "wall_seconds": 0.0,
    "output_tokens": 0,
    "output_tokens_per_second": 0.0,
    "requests_per_second": 0.0,
    "goodput_requests_per_second": null
  },
  "by_bucket": {},
  "scheduler": {
    "steps": 0,
    "prefill_steps": 0,
    "decode_steps": 0,
    "idle_steps": 0,
    "forced_prefill_steps": 0,
    "allocation_blocked_steps": 0,
    "preemptions": 0,
    "mode_switches": 0,
    "max_decode_streak": 0,
    "max_decode_gap_steps": 0,
    "max_waiting_requests": 0,
    "max_running_requests": 0,
    "peak_kv_block_utilization": null,
    "step_elapsed": {}
  },
  "slo": {"ttft_ms": null, "tpot_ms": null}
}
```

延迟摘要包含 count、mean、min、P50、P95、P99、max，并使用线性插值分位数。Goodput 只有配置至少一个 SLO 时才有定义。

## 9. `stdout.log` 与 `failure.json`

- 成功 run 把 CLI 最终渲染的 run ID、output directory 和 summary 保存到 `stdout.log`，便于无终端历史时审计。
- benchmark 执行异常时写 `failure.json`，包含错误类型、消息、内部 traceback 和失败单调时间；失败 run 不生成虚假 summary，也不应被删除。
- `git_dirty=true` 的 run 可以用于调试，但正式跨策略结论应使用同一干净 commit，或明确列出 dirty patch。

## 10. 正确性门禁

正式汇总前至少检查：

1. offered、admitted、finished 和 exact output length 自洽。
2. 每个 admitted ID 唯一且恰好完成一次；token 不能早于 admission 或晚于 finish。
3. token 时间戳和 step ID 单调。
4. KV 使用不越界，运行结束回到预期基线。
5. Bounded 策略的 `max_decode_streak` 不越过配置边界（waiting 可执行的条件需结合 step 记录）。
6. warmup 不进入正式统计。
7. commit、dirty 状态、workload 和环境字段齐全。

目前 summary 没有自动写 `VALID/INVALID/INCONCLUSIVE` 字段；实验报告必须先执行上述门禁再人工标注结论状态，不能假装已自动判定。

## 11. 隐私、校验和与状态

- 不保存凭据、私有 prompt、用户名或可识别个人信息。
- 不修改历史 run；重跑生成新目录。
- 对正式归档生成 SHA-256：

```bash
sha256sum workload.jsonl manifest.json requests.jsonl tokens.jsonl steps.jsonl gpu_telemetry.csv summary.json stdout.log > checksums.sha256
sha256sum --check checksums.sha256
```

纯 CPU artifact/metrics 测试已通过；最终测试数以主门禁为准。当前没有 RTX 4060 原始 run，所有 GPU 数字继续留空。
