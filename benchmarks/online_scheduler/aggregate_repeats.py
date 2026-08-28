"""Cross-repeat aggregation for the RTX 4060 scheduler A/B/C ablation.

Reads one or more online-scheduler run directories (each containing
``manifest.json``, ``summary.json`` and optionally ``gpu_telemetry.csv``),
groups runs by scheduler policy, and reports mean / std / CV / min / max
across the repeats for the core latency, throughput and scheduler metrics.

Only runs whose workload fingerprint matches the fixed ablation workload are
aggregated, so old single-run artifacts are never mixed in by accident.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Fixed ablation workload fingerprint (must match the run script exactly).
REQUIRED_WORKLOAD = {
    "num_requests": 100,
    "request_rate": 8.0,
    "seed": 20260827,
}
REQUIRE_EAGER = True
POLICY_ORDER = ("prefill_first", "decode_first", "bounded_decode_first")


@dataclass(frozen=True, slots=True)
class RepeatStats:
    count: int
    mean: float | None = None
    std: float | None = None
    cv: float | None = None
    minimum: float | None = None
    maximum: float | None = None


def summarize_values(values: Iterable[float]) -> RepeatStats:
    """Return mean / sample std / CV / min / max for a repeat group."""
    finite = [float(value) for value in values]
    if not finite:
        return RepeatStats(count=0)
    count = len(finite)
    mean = statistics.fmean(finite)
    std = statistics.stdev(finite) if count >= 2 else None
    cv = std / mean if std is not None and mean else None
    return RepeatStats(
        count=count,
        mean=mean,
        std=std,
        cv=cv,
        minimum=min(finite),
        maximum=max(finite),
    )


def _matches_workload(manifest: dict[str, Any]) -> bool:
    workload = manifest.get("workload") or {}
    engine = manifest.get("engine_config") or {}
    expected = {
        "num_requests": workload.get("num_requests"),
        "request_rate": workload.get("request_rate"),
        "seed": workload.get("seed"),
    }
    if expected != REQUIRED_WORKLOAD:
        return False
    return bool(engine.get("enforce_eager", False) is REQUIRE_EAGER)


def _gpu_stats(run_dir: Path) -> dict[str, float | None]:
    """Summarize nvidia-smi telemetry when present; otherwise all None."""
    csv_path = run_dir / "gpu_telemetry.csv"
    if not csv_path.is_file():
        return {"utilization_mean": None, "utilization_peak": None, "memory_peak": None}
    utilization: list[float] = []
    memory_used: list[float] = []
    try:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                utilization.append(float(row["utilization.gpu"]))
                memory_used.append(float(row["memory.used"]))
    except (KeyError, ValueError, csv.Error):
        return {"utilization_mean": None, "utilization_peak": None, "memory_peak": None}
    return {
        "utilization_mean": statistics.fmean(utilization) if utilization else None,
        "utilization_peak": max(utilization) if utilization else None,
        "memory_peak": max(memory_used) if memory_used else None,
    }


def _extract_metrics(run_dir: Path) -> dict[str, float | None] | None:
    """Return the per-run metric values, or None when the run is not usable."""
    manifest_path = run_dir / "manifest.json"
    summary_path = run_dir / "summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not _matches_workload(manifest):
        return None
    if summary.get("requests", {}).get("completion_rate") != 1.0:
        return None
    latency = summary.get("latency", {})
    scheduler = summary.get("scheduler", {})
    throughput = summary.get("throughput", {})
    gpu = _gpu_stats(run_dir)

    def get(table: dict[str, Any], *keys: str) -> float | None:
        value: Any = table
        for key in keys:
            value = value.get(key) if isinstance(value, dict) else None
            if value is None:
                return None
        return float(value)

    return {
        "output_tokens_per_second": get(throughput, "output_tokens_per_second"),
        "requests_per_second": get(throughput, "requests_per_second"),
        "ttft_p50_ms": get(latency, "ttft", "p50_ms"),
        "ttft_p95_ms": get(latency, "ttft", "p95_ms"),
        "tpot_p50_ms": get(latency, "tpot", "p50_ms"),
        "tpot_p95_ms": get(latency, "tpot", "p95_ms"),
        "e2e_p50_ms": get(latency, "e2e", "p50_ms"),
        "e2e_p95_ms": get(latency, "e2e", "p95_ms"),
        "itl_p95_ms": get(latency, "itl", "p95_ms"),
        "max_waiting_requests": get(scheduler, "max_waiting_requests"),
        "max_running_requests": get(scheduler, "max_running_requests"),
        "forced_prefill_steps": get(scheduler, "forced_prefill_steps"),
        "peak_kv_block_utilization": get(scheduler, "peak_kv_block_utilization"),
        "gpu_utilization_mean": gpu["utilization_mean"],
        "gpu_utilization_peak": gpu["utilization_peak"],
        "gpu_memory_peak_mib": gpu["memory_peak"],
    }


METRIC_LABELS = {
    "output_tokens_per_second": "throughput (output tok/s)",
    "requests_per_second": "throughput (req/s)",
    "ttft_p50_ms": "TTFT P50 (ms)",
    "ttft_p95_ms": "TTFT P95 (ms)",
    "tpot_p50_ms": "TPOT P50 (ms)",
    "tpot_p95_ms": "TPOT P95 (ms)",
    "e2e_p50_ms": "E2E P50 (ms)",
    "e2e_p95_ms": "E2E P95 (ms)",
    "itl_p95_ms": "ITL P95 (ms)",
    "max_waiting_requests": "max waiting (requests)",
    "max_running_requests": "max running (requests)",
    "forced_prefill_steps": "forced prefill (steps)",
    "peak_kv_block_utilization": "peak KV block utilization",
    "gpu_utilization_mean": "GPU utilization mean (%)",
    "gpu_utilization_peak": "GPU utilization peak (%)",
    "gpu_memory_peak_mib": "GPU memory peak (MiB)",
}


def _format_cell(value: float | None) -> str:
    return "-" if value is None else f"{value:.4g}"


def aggregate_repeats(
    artifacts_root: Path,
    *,
    policy_order: tuple[str, ...] = POLICY_ORDER,
) -> dict[str, Any]:
    """Aggregate every matching run under ``artifacts_root`` by policy."""
    runs: dict[str, dict[str, float | None]] = {}
    run_ids: dict[str, list[str]] = {}
    for run_dir in sorted(artifacts_root.iterdir()):
        if not run_dir.is_dir():
            continue
        metrics = _extract_metrics(run_dir)
        if metrics is None:
            continue
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        policy = manifest["policy"]
        runs.setdefault(policy, {metric: [] for metric in METRIC_LABELS})
        run_ids.setdefault(policy, [])
        run_ids[policy].append(manifest["run_id"])
        for metric, value in metrics.items():
            if value is not None:
                runs[policy][metric].append(value)

    ordered = [policy for policy in policy_order if policy in runs]
    ordered.extend(policy for policy in runs if policy not in policy_order)
    result: dict[str, Any] = {
        "artifacts_root": str(artifacts_root),
        "policy_order": ordered,
        "policies": {},
    }
    for policy in ordered:
        result["policies"][policy] = {
            "run_ids": run_ids[policy],
            "repeat_count": len(run_ids[policy]),
            "metrics": {
                metric: asdict(summarize_values(runs[policy][metric]))
                for metric in METRIC_LABELS
            },
        }
    return result


def _render_report(aggregate: dict[str, Any]) -> str:
    lines: list[str] = [
        f"# Cross-repeat aggregation",
        "",
        f"artifacts_root: `{aggregate['artifacts_root']}`",
        "",
    ]
    for policy in aggregate["policy_order"]:
        entry = aggregate["policies"][policy]
        lines.append(f"## {policy}  (n = {entry['repeat_count']})")
        lines.append("")
        lines.append("| metric | mean | std | CV | min | max |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for metric, stats in entry["metrics"].items():
            if stats["count"] == 0:
                continue
            lines.append(
                f"| {METRIC_LABELS[metric]} | {_format_cell(stats['mean'])} | "
                f"{_format_cell(stats['std'])} | {_format_cell(stats['cv'])} | "
                f"{_format_cell(stats['minimum'])} | {_format_cell(stats['maximum'])} |"
            )
        lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("artifacts/online_scheduler/experiments/20260828-ablation"),
    )
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    aggregate = aggregate_repeats(args.artifacts_root)
    print(_render_report(aggregate))
    if args.output_json is not None:
        args.output_json.write_text(
            json.dumps(aggregate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
