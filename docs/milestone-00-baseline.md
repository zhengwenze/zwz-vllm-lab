# Milestone 0 — WSL RTX 4060 Baseline

## Goal

Run Qwen3-0.6B with this nano-vLLM fork inside an isolated WSL environment, record a reproducible single-GPU baseline, and keep the benchmark workflow reviewable through Git.

## Validated environment direction

The local setup attempt established that the following combination is compatible with the RTX 4060 / 560-series driver path used for this milestone:

- Python 3.12
- PyTorch 2.7.1
- CUDA runtime 12.6
- FlashAttention 2.8.3 wheel built for Python 3.12 / Torch 2.7 / CUDA 12 / CXX11 ABI TRUE
- Qwen3-0.6B
- WSL2 Ubuntu

The project intentionally avoids changing the previously working Day-15 vLLM environment.

## Why the initial installation took so long

`pyproject.toml` declares broad lower bounds (`torch>=2.4.0`, `triton>=3.0.0`) and an unpinned `flash-attn`. A fresh resolver can therefore select a much newer PyTorch/CUDA stack than the installed NVIDIA driver supports. FlashAttention is additionally sensitive to the exact Python, PyTorch, CUDA and C++ ABI combination, and source installation requires `nvcc`.

For this milestone, prefer a known-compatible binary FlashAttention wheel instead of compiling it from source inside WSL.

## Baseline procedure

From the repository root in the isolated virtual environment:

```bash
python scripts/baseline_rtx4060.py \
  --model ~/huggingface/Qwen3-0.6B \
  --output reports/rtx4060-qwen3-0.6b-baseline.md
```

The runner deliberately refuses to generate a GPU baseline if CUDA is unavailable or the detected GPU is not an RTX 4060-class device.

## What the report records

- GPU and NVIDIA driver
- OS and Python version
- PyTorch and CUDA runtime version
- FlashAttention version
- model path and nano-vLLM execution settings
- request count and generation length
- input/output token counts
- end-to-end batch time
- output-token throughput
- total-token throughput
- peak PyTorch allocated VRAM

## Metric scope

This first milestone is an **offline single-GPU baseline**. The existing nano-vLLM `generate` API does not expose per-token timestamps, so TTFT and TPOT are intentionally not fabricated here. Those service-level latency metrics belong in a later serving benchmark where arrival time and first-token/token-stream timestamps can be measured correctly.

## Acceptance criteria

- [ ] `torch.cuda.is_available()` is true inside the isolated WSL environment.
- [ ] GPU reports RTX 4060.
- [ ] Qwen3-0.6B loads successfully.
- [ ] At least one real generation succeeds.
- [ ] `scripts/baseline_rtx4060.py` completes without error.
- [ ] `reports/rtx4060-qwen3-0.6b-baseline.md` is generated from the actual machine.
- [ ] The generated report is reviewed and committed in a follow-up commit/PR.
