# RTX 4060 在线调度 15-run 公开证据包

状态：**GPU Verified**。2026-08-28 在 NVIDIA GeForce RTX 4060 8 GB / Qwen3-0.6B 上，对 `prefill_first`、`decode_first`、`bounded_decode_first(K=8)` 各执行 5 次独立重复，共 15 个正式 run；15/15 均满足 `offered = admitted = finished = 100`。

核心结果与限制见[聚合实验报告](../../../reports/nanovllm-online-rtx4060-20260828.md)。机器可读的跨 repeat 统计在 [`aggregate.json`](aggregate.json)，每个 run 的环境、软件版本、配置、请求级/Token 级/Step 级事件、GPU 遥测和汇总位于 [`runs/`](runs/)。

## 证据结构

```text
aggregate.json
source-sha256.json
checksums.sha256
runs/<run_id>/
├── manifest.public.json
├── summary.json
├── workload.jsonl.gz
├── requests.jsonl.gz
├── tokens.jsonl.gz
├── steps.jsonl.gz
├── gpu_telemetry.csv.gz
└── stdout.log.gz
```

JSONL/CSV 使用 gzip 仅为控制 Git 体积；解压后仍是原 schema，可由仓库 benchmark 聚合代码复算。`checksums.sha256` 校验所有公开文件；`source-sha256.json` 记录脱敏前本地源文件的 SHA-256 和字节数。

## 脱敏与可审计边界

只做以下机械替换，数值、时间戳、请求 ID、调度事件和指标不变：本机用户名替换为 `<user>`，工作区绝对路径替换为 `<workspace>`，GPU UUID 替换为 `GPU-[REDACTED]`。未发现 API key、Authorization header、prompt 正文或模型输出正文。

实验 manifest 如实保留 `git_dirty=true`：15 个 run 记录的 dirty 内容是当时尚未跟踪的重复聚合脚本、运行脚本和对应测试；引擎实验基准 commit 为 `73b9118838af8ef17d1587de8bd3cca037d50263`。这批结果支持本报告的固定 workload 结论，不支持跨硬件、跨模型或生产流量推广。
