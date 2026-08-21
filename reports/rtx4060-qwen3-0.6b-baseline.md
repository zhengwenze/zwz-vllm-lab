# RTX 4060 / Qwen3-0.6B Baseline

Generated: 2026-08-21 21:35:06 (Asia/Shanghai)

## Environment

- GPU: `NVIDIA GeForce RTX 4060`
- NVIDIA driver: `560.94`
- OS: `Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43`
- Python: `3.12.14`
- PyTorch: `2.7.1+cu126`
- PyTorch CUDA runtime: `12.6`
- FlashAttention: `2.8.3`
- Model: `/home/zwz2025/huggingface/Qwen3-0.6B`
- Engine: `nano-vLLM`
- tensor_parallel_size: `1`
- enforce_eager: `True`
- max_model_len: `4096`

## Workload

- Requests: `8`
- Max output tokens/request: `128`
- Sampling: near-greedy (`temperature=1e-05`, seed `0`)
- Warm-up: 1 request / 16 max output tokens

## Results

- Input tokens: `161`
- Output tokens: `1024`
- End-to-end batch time: `4.665 s`
- Output throughput: `219.52 tokens/s`
- Total token throughput: `254.04 tokens/s`
- Peak PyTorch allocated VRAM: `5.62 GiB`

## Notes

This is an offline single-GPU baseline. It does **not** report TTFT/TPOT because
nano-vLLM's offline `generate` API does not expose per-token timestamps.
Service-level latency metrics should be added in a later serving benchmark.

