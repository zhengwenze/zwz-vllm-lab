#!/usr/bin/env python3
"""Generate a small, reproducible Nano-vLLM baseline report on a single NVIDIA GPU."""

from __future__ import annotations

import argparse
import platform
import subprocess
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams


def command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        return "unavailable"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(Path.home() / "huggingface/Qwen3-0.6B"))
    parser.add_argument("--output", default="reports/rtx4060-qwen3-0.6b-baseline.md")
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing to create an RTX 4060 GPU baseline.")

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    gpu_name = torch.cuda.get_device_name(0)
    if "4060" not in gpu_name:
        raise RuntimeError(f"Expected an RTX 4060-class GPU, got: {gpu_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(str(model_path), enforce_eager=True, tensor_parallel_size=1, max_model_len=4096)

    prompts = [
        "用三句话解释什么是大模型推理中的 KV Cache。",
        "Explain the difference between prefill and decode in LLM inference.",
        "为什么批处理可以提高 GPU 推理吞吐量？",
        "Give three practical causes of CUDA out-of-memory during LLM serving.",
        "简述 PagedAttention 解决了什么问题。",
        "What do TTFT and TPOT measure in an inference service?",
        "解释 continuous batching 的基本思想。",
        "Summarize why reproducible benchmarks need a warm-up phase.",
    ]
    formatted = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    # Warm-up CUDA context/kernels and model execution before timing.
    llm.generate([formatted[0]], SamplingParams(temperature=0.0, max_tokens=16), use_tqdm=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    outputs = llm.generate(formatted, params, use_tqdm=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    input_tokens = sum(len(tokenizer.encode(x)) for x in formatted)
    output_tokens = sum(len(tokenizer.encode(item["text"])) for item in outputs)
    total_tokens = input_tokens + output_tokens
    output_tps = output_tokens / elapsed
    total_tps = total_tokens / elapsed
    peak_vram_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)

    driver = command_output([
        "nvidia-smi",
        "--query-gpu=driver_version",
        "--format=csv,noheader",
    ]).splitlines()[0]

    try:
        import flash_attn
        flash_attn_version = flash_attn.__version__
    except Exception:
        flash_attn_version = "unavailable"

    report = "# RTX 4060 / Qwen3-0.6B Baseline\n\n"
    report += f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    report += "## Environment\n\n"
    report += f"- GPU: `{gpu_name}`\n"
    report += f"- NVIDIA driver: `{driver}`\n"
    report += f"- OS: `{platform.platform()}`\n"
    report += f"- Python: `{platform.python_version()}`\n"
    report += f"- PyTorch: `{torch.__version__}`\n"
    report += f"- PyTorch CUDA runtime: `{torch.version.cuda}`\n"
    report += f"- FlashAttention: `{flash_attn_version}`\n"
    report += f"- Model: `{model_path}`\n"
    report += "- Engine: `nano-vLLM`\n"
    report += "- tensor_parallel_size: `1`\n"
    report += "- enforce_eager: `True`\n"
    report += "- max_model_len: `4096`\n\n"
    report += "## Workload\n\n"
    report += f"- Requests: `{len(formatted)}`\n"
    report += f"- Max output tokens/request: `{args.max_tokens}`\n"
    report += "- Sampling: greedy (`temperature=0.0`)\n"
    report += "- Warm-up: 1 request / 16 max output tokens\n\n"
    report += "## Results\n\n"
    report += f"- Input tokens: `{input_tokens}`\n"
    report += f"- Output tokens: `{output_tokens}`\n"
    report += f"- End-to-end batch time: `{elapsed:.3f} s`\n"
    report += f"- Output throughput: `{output_tps:.2f} tokens/s`\n"
    report += f"- Total token throughput: `{total_tps:.2f} tokens/s`\n"
    report += f"- Peak PyTorch allocated VRAM: `{peak_vram_gib:.2f} GiB`\n\n"
    report += "## Notes\n\n"
    report += (
        "This is an offline single-GPU baseline. It does **not** report TTFT/TPOT because "
        "nano-vLLM's offline `generate` API does not expose per-token timestamps. Service-level "
        "latency metrics should be added in a later serving benchmark.\n"
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved report to: {output_path}")


if __name__ == "__main__":
    main()
