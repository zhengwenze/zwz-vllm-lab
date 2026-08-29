# WSL2 + RTX 4060 复现实验手册

> **状态标签**：`Implemented`（操作手册与 benchmark CLI） · `CPU Verified`（46 tests passed） · `GPU Verified`（RTX 4060 15-run；[证据](../../reports/nanovllm-online-rtx4060-20260828.md)）
>
> **目标**：在 Windows + WSL2 的 RTX 4060 环境复现 Nano-vLLM 在线调度实验，并留下完整原始证据。

## 1. 安全与范围

- 所有命令默认在个人实验机执行，不在生产服务器执行。
- 不在仓库保存 Hugging Face token、代理密码或私有模型凭据。
- 不安装 WSL 内的 Linux NVIDIA display driver。CUDA on WSL 使用 Windows 主机 NVIDIA 驱动；WSL 内只按需要安装 toolkit/runtime。
- 仓库放在 WSL Linux 文件系统（例如 `/home/<user>/src`），避免在 `/mnt/c` 上做高频小文件构建。
- 先验证 CPU/契约测试，再运行 GPU 实验。

官方参考：[Microsoft WSL 安装](https://learn.microsoft.com/en-us/windows/wsl/install)、[NVIDIA CUDA on WSL 指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)、[PyTorch 本地安装选择器](https://pytorch.org/get-started/locally/)。安装命令随版本变化时，以官方页面为准，并把最终命令记入环境文件。

## 2. Windows 主机准备

以管理员 PowerShell 执行：

```powershell
wsl --install
wsl --update
wsl --status
wsl --list --verbose
```

确认目标发行版的 VERSION 为 2。若已有 WSL，不要重复注销发行版；`wsl --unregister` 会删除该发行版数据，本手册不使用该命令。

安装支持 WSL CUDA 的 NVIDIA Windows 驱动并重启。驱动安装只在 Windows 主机完成。

## 3. WSL 内 GPU 探测

进入 WSL：

```powershell
wsl
```

在 Linux shell 执行：

```bash
uname -a
cat /etc/os-release
nvidia-smi
```

验收：

- `nvidia-smi` 能看到 NVIDIA GeForce RTX 4060。
- 没有 “command not found”、driver mismatch 或 GPU 不可见错误。
- 把完整输出保存到本次 run 的 `manifest.json` 或旁证文本。

若失败：

1. 回到 Windows 确认驱动与 `wsl --update`。
2. 执行 `wsl --shutdown` 后重新进入。
3. 不要在 WSL 安装 `cuda-drivers` 或普通 Linux display driver。

## 4. Linux 基础依赖

```bash
sudo apt update
sudo apt install -y git build-essential
```

正式实验固定 Python 3.11。通过发行版仓库、`pyenv` 或 Conda 安装均可，但必须验证 `python3.11 --version`，不要退回开发主机的 Python 3.13；项目声明只支持 `>=3.10,<3.13`。

是否需要 CUDA toolkit 取决于 PyTorch、Triton 和 FlashAttention 的构建方式。先用 PyTorch 官方安装选择器确定与当前驱动兼容的 wheel；不要硬编码一组未经验证的 CUDA 版本。

## 5. 获取代码与创建隔离环境

```bash
mkdir -p /home/$USER/src
cd /home/$USER/src
git clone <YOUR_REPOSITORY_URL> nano-vllm
cd nano-vllm
git rev-parse HEAD
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

`<YOUR_REPOSITORY_URL>` 必须替换为实际仓库地址。为了复现固定版本，记录 commit；需要切换 commit 时使用明确哈希，不使用浮动分支作为实验身份。

## 6. 安装 PyTorch 与项目依赖

先从 PyTorch 官方选择器复制与当前环境匹配的命令，然后验证：

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

通过后安装项目：

```bash
python -m pip install -e '.[online,test]'
```

`flash-attn` 可能需要编译工具、足够内存和与 PyTorch/CUDA 匹配的环境。失败时保存完整日志，先核对版本兼容，不使用来源不明的 wheel。

再次验证：

```bash
python -c "import torch, transformers, triton, flash_attn; print('imports ok'); print(torch.cuda.is_available())"
python -m pip freeze
```

## 7. 下载模型

推荐从 Qwen3-0.6B 开始。示例：

```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir /home/$USER/models/Qwen3-0.6B \
  --local-dir-use-symlinks False
```

若 CLI 参数因版本变化而弃用，以当前 Hugging Face CLI 帮助为准。记录模型 revision、文件清单和实际绝对路径；结果文件不要写访问 token。

## 8. 运行前门禁

```bash
source .venv/bin/activate
python -m compileall -q nanovllm tests
python -m pytest -q
git diff --check
nvidia-smi
```

验收：

- CPU 契约测试通过。
- GPU 可用且没有其他重负载进程。
- 工作树和 commit 状态被记录。
- 任何失败都先修复或标为 blocked，不跳过后写性能结论。

## 9. 单请求 GPU 冒烟

先使用 `enforce_eager=True`，减少 CUDA Graph 对排障的干扰；确认正确后再把 Graph 开关作为独立实验变量。

```bash
python example.py
```

若示例路径仍指向 `~/huggingface/Qwen3-0.6B/`，请用实际配置入口传入模型路径；不要为了跑通而把本机绝对路径提交到公共仓库。

记录：启动是否成功、首个输出、终态、峰值显存和错误日志。冒烟通过只证明能运行，不是性能结论。

## 10. 在线服务冒烟

服务入口已经落地：

```bash
nanovllm-serve \
  --model /home/$USER/models/Qwen3-0.6B \
  --scheduler-policy bounded_decode_first \
  --max-consecutive-decode-steps 8 \
  --max-queue-size 256
```

另一个 shell 验证：

```bash
curl -sS http://127.0.0.1:8000/health
curl -N -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"request_id":"smoke-001","prompt":"Hello","temperature":0.6,"max_tokens":16,"ignore_eos":false}'
curl -sS http://127.0.0.1:8000/metrics
```

也可使用 `python -m nanovllm.serve.sse`。入口存在和 CPU 测试通过不代表本机 GPU 已运行。

## 11. 正式实验流程

为每轮创建独立目录：

```text
artifacts/online_scheduler/<run_id>/
├── manifest.json
├── workload.jsonl
├── requests.jsonl
├── tokens.jsonl
├── steps.jsonl
├── gpu_telemetry.csv
├── summary.json
├── stdout.log
├── failure.json             # 仅失败时
└── checksums.sha256
```

执行顺序：

1. 采集环境、commit、依赖和 GPU 状态。
2. 固定 workload 并保存 config。
3. warmup；warmup 数据单独标记。
4. 轮换运行三个策略，避免总让某策略处在冷机或热机阶段。
5. 每个策略重复相同次数。
6. 运行正确性与 schema 校验。
7. 生成 SHA-256。
8. 只把有效运行汇总进实验报告。

正式命令示例：

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

另外两种策略只替换 `--policy`；每轮重复递增 `--repeat-index`，不要在同一 A/B 中顺便改变负载或引擎上限。

Barrier interference 用例：

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

Runner 会等初始 Decode-heavy 请求全部产出首 token，再记录 barrier 并按 100 ms 间隔注入长 Prompt；这能稳定制造 Prefill 对在途 Decode 的干扰。

计算校验和：

```bash
cd artifacts/online_scheduler/<run_id>
sha256sum manifest.json workload.jsonl requests.jsonl tokens.jsonl steps.jsonl gpu_telemetry.csv summary.json stdout.log > checksums.sha256
```

## 12. 常见问题

### `torch.cuda.is_available()` 为 False

先检查 Windows 驱动、`wsl --update`、WSL 内 `nvidia-smi` 和 PyTorch wheel 是否带 CUDA。不要先改 Nano-vLLM 代码。

### `flash-attn` 安装失败

保存编译日志，核对 Python/PyTorch/CUDA/编译器组合与内存。确认 PyTorch CUDA 基础先通过，再处理 FlashAttention。

### 初始化时 KV Cache blocks 断言失败

可能是模型/Graph warmup 已占显存，或 `gpu_memory_utilization` 留给 KV 的预算不足。关闭其他 GPU 进程、降低模型/上下文/并发，记录变更；不要把失败轮删掉。

### NCCL 初始化失败

单卡 RTX 4060 使用 `tensor_parallel_size=1`。仍失败时记录 NCCL、driver、PyTorch 和端口状态；不要为了绕过错误虚构多卡结果。

### 性能波动很大

检查 Windows 电源模式、GPU 温度/功耗、后台图形或计算负载、WSL 文件位置和实验顺序。增加重复次数，报告分布而非只选最快一次。

## 13. 完成条件

- CPU 契约测试有可复现通过记录。
- RTX 4060 单请求与在线 SSE 冒烟成功。
- 三策略都有相同工作负载、warmup 和重复运行。
- 每轮 schema 完整、正确性门禁通过、SHA-256 可核对。
- `EXPERIMENT_REPORT_RTX4060.md` 的数字可从原始数据重新计算。

未满足任一条时继续保留 `GPU Pending`。

当前 CPU 证据：开发主机 Python 3.13 上 `46 passed`，Ruff、`compileall` 与 `git diff --check` 通过。Python 3.13 超出项目声明的 `>=3.10,<3.13`，所以它只作为纯 CPU 逻辑证据；本手册正式 WSL 环境固定 Python 3.11。
