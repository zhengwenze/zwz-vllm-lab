"""Command-line entry point for RTX 4060 online scheduler experiments."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import sys
import time
import traceback

from .runner import GpuTelemetry, OnlineBenchmarkRunner, runtime_manifest
from .workload import (
    RTX4060_DECODE_HEAVY_BUCKETS,
    RTX4060_MIXED_BUCKETS,
    RTX4060_PREFILL_HEAVY_BUCKETS,
    WorkloadBucket,
    make_fixed_trace,
    make_interference_trace,
    make_poisson_trace,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local Hugging Face Qwen3-0.6B model directory")
    parser.add_argument("--policy", choices=("prefill_first", "decode_first", "bounded_decode_first"), required=True)
    parser.add_argument("--max-consecutive-decode-steps", type=int, default=8)
    parser.add_argument(
        "--workload",
        choices=("mixed", "decode-heavy", "prefill-heavy", "fixed", "interference"),
        default="mixed",
    )
    parser.add_argument("--arrival", choices=("poisson", "fixed"), default="poisson")
    parser.add_argument("--num-requests", type=int, default=300)
    parser.add_argument("--request-rate", type=float, default=2.0, help="Poisson arrivals per second")
    parser.add_argument("--interval-ms", type=float, default=500.0, help="Fixed inter-arrival interval")
    parser.add_argument("--prompt-tokens", type=int, default=128, help="Only used by the fixed workload")
    parser.add_argument("--output-tokens", type=int, default=128, help="Only used by the fixed workload")
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--repeat-index", type=int, default=0)
    parser.add_argument("--interference-decode-requests", type=int, default=8)
    parser.add_argument("--interference-prefill-requests", type=int, default=8)
    parser.add_argument("--injection-interval-ms", type=float, default=100.0)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/online_scheduler"))
    parser.add_argument("--ttft-slo-ms", type=float)
    parser.add_argument("--tpot-slo-ms", type=float)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-queue-size", type=int, help="Defaults to at least the complete replay size")
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--warmup-prompt-tokens", type=int, default=128)
    parser.add_argument("--warmup-output-tokens", type=int, default=16)
    parser.add_argument("--max-drain-seconds", type=float, default=300.0)
    parser.add_argument("--telemetry-interval", type=float, default=0.2)
    parser.add_argument("--no-gpu-telemetry", action="store_true")
    return parser


def _buckets(args: argparse.Namespace) -> tuple[WorkloadBucket, ...]:
    if args.workload == "mixed":
        return RTX4060_MIXED_BUCKETS
    if args.workload == "decode-heavy":
        return RTX4060_DECODE_HEAVY_BUCKETS
    if args.workload == "prefill-heavy":
        return RTX4060_PREFILL_HEAVY_BUCKETS
    return (WorkloadBucket("fixed", args.prompt_tokens, args.output_tokens),)


def _trace(args: argparse.Namespace):
    if args.workload == "interference":
        return make_interference_trace(
            decode_requests=args.interference_decode_requests,
            injected_prefill_requests=args.interference_prefill_requests,
            injection_interval_seconds=args.injection_interval_ms / 1000,
            seed=args.seed,
        )
    buckets = _buckets(args)
    name = f"{args.arrival}-{args.workload}"
    if args.arrival == "fixed":
        if len(buckets) != 1:
            raise ValueError("fixed arrival currently requires --workload fixed, decode-heavy, or prefill-heavy")
        bucket = buckets[0]
        return make_fixed_trace(
            num_requests=args.num_requests,
            interval_seconds=args.interval_ms / 1000,
            prompt_tokens=bucket.prompt_tokens,
            output_tokens=bucket.output_tokens,
            seed=args.seed,
            name=name,
        )
    return make_poisson_trace(
        num_requests=args.num_requests,
        request_rate=args.request_rate,
        buckets=buckets,
        seed=args.seed,
        name=name,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    trace = _trace(args)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + f"-{args.policy}-{trace.name}"
    output_dir = args.output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    trace.write_jsonl(output_dir / "workload.jsonl")

    # Delay heavy imports so workload/metric unit tests remain CPU-only.
    import torch
    import transformers
    import triton
    from nanovllm import LLM, SamplingParams

    engine_config = {
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_queue_size": args.max_queue_size or max(512, args.num_requests),
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": 1,
        "enforce_eager": args.enforce_eager,
        "scheduler_policy": args.policy,
        "max_consecutive_decode_steps": args.max_consecutive_decode_steps,
    }
    engine = LLM(args.model, **engine_config)
    warmup_params = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=args.warmup_output_tokens)
    engine.generate([[1] * args.warmup_prompt_tokens], warmup_params, use_tqdm=False)

    try:
        flash_attn_version = version("flash-attn")
    except PackageNotFoundError:
        flash_attn_version = None
    actual_num_kvcache_blocks = engine.model_runner.config.num_kvcache_blocks
    manifest = runtime_manifest()
    manifest.update(
        {
            "run_id": run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model": str(Path(args.model).expanduser().resolve()),
            "policy": args.policy,
            "policy_params": {"max_consecutive_decode_steps": args.max_consecutive_decode_steps},
            "repeat_index": args.repeat_index,
            "engine_config": engine_config,
            "workload": {
                "name": trace.name,
                "seed": trace.seed,
                "num_requests": len(trace.requests),
                "request_rate": args.request_rate if args.arrival == "poisson" else None,
                "interval_ms": args.interval_ms if args.arrival == "fixed" else None,
            },
            "software": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "transformers": transformers.__version__,
                "triton": triton.__version__,
                "flash_attn": flash_attn_version,
            },
            "kv_cache": {
                "block_size_tokens": engine.model_runner.config.kvcache_block_size,
                "num_blocks": actual_num_kvcache_blocks,
                "token_capacity": actual_num_kvcache_blocks * engine.model_runner.config.kvcache_block_size,
            },
            "warmup": {
                "completed": True,
                "prompt_tokens": args.warmup_prompt_tokens,
                "output_tokens": args.warmup_output_tokens,
            },
        }
    )

    try:
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        telemetry = GpuTelemetry(
            output_dir / "gpu_telemetry.csv",
            interval_seconds=args.telemetry_interval,
        )
        if not args.no_gpu_telemetry:
            telemetry.start()
        benchmark_started = time.perf_counter_ns()
        try:
            runner = OnlineBenchmarkRunner(
                engine,
                trace,
                lambda request: SamplingParams(
                    temperature=0.6,
                    ignore_eos=True,
                    max_tokens=request.output_tokens,
                ),
                max_drain_seconds=args.max_drain_seconds,
            )
            result = runner.run()
        except BaseException as exc:
            failure = {
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "failed_perf_counter_ns": time.perf_counter_ns(),
            }
            (output_dir / "failure.json").write_text(
                json.dumps(failure, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            raise
        finally:
            telemetry.stop()
        manifest["telemetry"] = telemetry.status()
        manifest["benchmark"] = {
            "runner_start_ns": result.start_ns,
            "runner_finish_ns": result.finish_ns,
            "barrier_ns": result.barrier_ns,
            "wall_ms": (result.finish_ns - result.start_ns) / 1_000_000,
            "orchestration_start_ns": benchmark_started,
        }
        summary = result.collector.write_artifacts(
            output_dir,
            manifest=manifest,
            ttft_slo_ms=args.ttft_slo_ms,
            tpot_slo_ms=args.tpot_slo_ms,
        )
        console_result = {
            "run_id": run_id,
            "output_dir": str(output_dir),
            "summary": summary,
        }
        rendered_result = json.dumps(console_result, ensure_ascii=False, indent=2)
        (output_dir / "stdout.log").write_text(rendered_result + "\n", encoding="utf-8")
        print(rendered_result)
        return 0
    finally:
        engine.exit()


if __name__ == "__main__":
    raise SystemExit(main())
