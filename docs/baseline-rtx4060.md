# RTX 4060 baseline: Qwen3-0.6B

## Result

The first milestone runs the unmodified upstream Nano-vLLM benchmark on a
single RTX 4060 8 GB under WSL2.

| Metric | Result |
| --- | ---: |
| Requests / sequences | 256 |
| Total generated tokens | 133,966 |
| Measured benchmark time | 107.87 s |
| Offline output throughput | **1241.94 tokens/s** |
| Whole-process wall time | 138.25 s |
| Maximum resident host memory (`time -v`) | 2,301,224 KiB |
| GPU utilization observed during the run | 96% |
| GPU memory observed during the run | 7699 MiB / 8188 MiB |
| GPU temperature observed during the run | 63 C |
| Exit code | 0 |

The upstream README reports 1434.13 output tokens/s on an RTX 4070 Laptop 8 GB
for the same workload. This RTX 4060 result is about 86.6% of that throughput.
The comparison is only a reference because the driver, host CPU, operating
system, power limit, and software versions are not controlled across machines.

## Runner cross-check

The repository's smaller `scripts/baseline_rtx4060.py` workload was also run
after fixing its unsupported `temperature=0.0` setting. With eight prompts and
128 output tokens per prompt, it generated 1024 output tokens in 4.665 seconds
(219.52 output tokens/s) and reported 5.62 GiB peak PyTorch-allocated VRAM. The
generated artifact is in
[`reports/rtx4060-qwen3-0.6b-baseline.md`](../reports/rtx4060-qwen3-0.6b-baseline.md).

This cross-check is a smoke-sized workload and is not directly comparable to
the 256-sequence upstream benchmark above.

## Method

Date: 2026-08-21 (Asia/Shanghai)

Repository base commit before this report:
`bb823b3e06983d71485a8e1f23715ebd87d98ef8`.

Environment details are recorded in [environment.md](environment.md). The model
weights were loaded from `/home/zwz2025/huggingface/Qwen3-0.6B`.

The benchmark command was:

```bash
cd /home/zwz2025/projects/zwz-vllm-lab
source .venv/bin/activate
/usr/bin/time -v python bench.py
```

The workload defined by upstream `bench.py` uses:

- random seed 0;
- 256 sequences;
- input lengths sampled uniformly from 100 to 1024 tokens;
- output lengths sampled uniformly from 100 to 1024 tokens;
- maximum model length 4096;
- tensor parallel size 1;
- CUDA graphs enabled (`enforce_eager=False`);
- one warm-up generation before timing.

The throughput calculation is:

```text
133,966 output tokens / 107.87 seconds = 1241.94 output tokens/second
```

The 138.25-second whole-process wall time also includes model loading, warm-up,
and shutdown, so it is intentionally not used for the throughput calculation.

## Interpretation and limitations

- This measures saturated **offline batch throughput**. It does not measure an
  OpenAI-compatible server, time to first token (TTFT), time per output token
  (TPOT), p50/p95 latency, or concurrent online requests.
- GPU memory, utilization, and temperature are point observations made while
  the benchmark was active, not high-frequency sampled peaks.
- This is one measured run, not a multi-run statistical result.
- The RTX 4060 also drives the Windows desktop, so some VRAM is unavailable to
  WSL2 and background activity can affect the result.
- Torch Dynamo reported that `rms_forward` reached its recompile limit because
  it observed different tensor ranks. The benchmark still completed with exit
  code 0; investigating this warning is a useful optimization follow-up.

## Next measurements

The next benchmark milestone should add an online serving harness and report
TTFT, TPOT, request throughput, p50/p95 latency, and peak VRAM across controlled
concurrency and prompt/output-length combinations.

