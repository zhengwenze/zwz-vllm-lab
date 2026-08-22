"""Metric collection and raw-artifact serialization for online benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

from .workload import RequestSpec


NANOSECONDS_PER_MILLISECOND = 1_000_000
NANOSECONDS_PER_SECOND = 1_000_000_000


def percentile(values: Iterable[float], quantile: float) -> float | None:
    """Return a linearly interpolated percentile (Hyndman-Fan type 7)."""

    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between zero and one")
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("percentile values must be finite")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize_latencies(values_ms: Iterable[float]) -> dict[str, float | int | None]:
    values = [float(value) for value in values_ms]
    if not values:
        return {"count": 0, "mean_ms": None, "min_ms": None, "p50_ms": None, "p95_ms": None, "p99_ms": None, "max_ms": None}
    if any(value < 0 or not math.isfinite(value) for value in values):
        raise ValueError("latencies must be finite and non-negative")
    return {
        "count": len(values),
        "mean_ms": fmean(values),
        "min_ms": min(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "max_ms": max(values),
    }


@dataclass(slots=True)
class TokenRecord:
    request_id: str
    output_index: int
    token_id: int
    emitted_ns: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "request_id": self.request_id,
            "output_index": self.output_index,
            "token_id": self.token_id,
            "emitted_ns": self.emitted_ns,
        }


@dataclass(slots=True)
class RequestRecord:
    request_id: str
    bucket: str
    arrival_ns: int
    prompt_tokens: int
    requested_output_tokens: int
    admitted_ns: int | None = None
    first_token_ns: int | None = None
    finish_ns: int | None = None
    status: str = "pending"
    waiting_steps: int = 0
    preemption_count: int = 0
    # Recompute tokens depend on prefix-cache hits and are not yet surfaced by
    # the engine. Keep this unknown rather than deriving a misleading value.
    recomputed_tokens: int | None = None
    token_records: list[TokenRecord] = field(default_factory=list)

    @property
    def actual_output_tokens(self) -> int:
        return len(self.token_records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "bucket": self.bucket,
            "arrival_ns": self.arrival_ns,
            "admitted_ns": self.admitted_ns,
            "prompt_tokens": self.prompt_tokens,
            "requested_output_tokens": self.requested_output_tokens,
            "actual_output_tokens": self.actual_output_tokens,
            "first_token_ns": self.first_token_ns,
            "finish_ns": self.finish_ns,
            "status": self.status,
            "waiting_steps": self.waiting_steps,
            "preemption_count": self.preemption_count,
            "recomputed_tokens": self.recomputed_tokens,
        }


class MetricsCollector:
    """Collect request/token/step events with strict lifecycle validation."""

    def __init__(self) -> None:
        self.requests: dict[str, RequestRecord] = {}
        self.tokens: list[TokenRecord] = []
        self.steps: list[dict[str, Any]] = []

    def register_request(self, spec: RequestSpec, *, arrival_ns: int) -> None:
        if spec.request_id in self.requests:
            raise ValueError(f"duplicate request: {spec.request_id}")
        self.requests[spec.request_id] = RequestRecord(
            request_id=spec.request_id,
            bucket=spec.bucket,
            arrival_ns=arrival_ns,
            prompt_tokens=spec.prompt_tokens,
            requested_output_tokens=spec.output_tokens,
        )

    def mark_admitted(self, request_id: str, *, admitted_ns: int) -> None:
        record = self._request(request_id)
        if record.admitted_ns is not None:
            raise ValueError(f"request already admitted: {request_id}")
        if admitted_ns < record.arrival_ns:
            raise ValueError("admitted timestamp precedes ideal arrival")
        record.admitted_ns = admitted_ns
        record.status = "running"

    def record_token(self, request_id: str, *, token_id: int, emitted_ns: int, finished: bool = False) -> None:
        record = self._request(request_id)
        if record.admitted_ns is None:
            raise ValueError(f"token emitted before admission: {request_id}")
        if record.finish_ns is not None:
            raise ValueError(f"token emitted after completion: {request_id}")
        previous_ns = record.token_records[-1].emitted_ns if record.token_records else record.admitted_ns
        if emitted_ns < previous_ns:
            raise ValueError("token timestamps must be monotonic")
        token = TokenRecord(request_id, len(record.token_records), int(token_id), emitted_ns)
        record.token_records.append(token)
        self.tokens.append(token)
        if record.first_token_ns is None:
            record.first_token_ns = emitted_ns
        if finished:
            self.mark_finished(request_id, finish_ns=emitted_ns)

    def mark_finished(self, request_id: str, *, finish_ns: int) -> None:
        record = self._request(request_id)
        if record.finish_ns is not None:
            raise ValueError(f"request already finished: {request_id}")
        if record.first_token_ns is None:
            raise ValueError(f"request finished without output token: {request_id}")
        if finish_ns < record.first_token_ns:
            raise ValueError("finish timestamp precedes first token")
        record.finish_ns = finish_ns
        record.status = "finished"

    def add_step(self, step: dict[str, Any]) -> None:
        step = dict(step)
        for request_id in step.get("waiting_request_ids", ()):
            if request_id in self.requests:
                self.requests[request_id].waiting_steps += 1
        for request_id in step.get("preempted_request_ids", ()):
            if request_id in self.requests:
                self.requests[request_id].preemption_count += 1
        self.steps.append(step)

    def summary(
        self,
        *,
        ttft_slo_ms: float | None = None,
        tpot_slo_ms: float | None = None,
    ) -> dict[str, Any]:
        records = list(self.requests.values())
        finished = [record for record in records if record.finish_ns is not None]
        ttft = [(record.first_token_ns - record.arrival_ns) / NANOSECONDS_PER_MILLISECOND for record in finished]
        e2e = [(record.finish_ns - record.arrival_ns) / NANOSECONDS_PER_MILLISECOND for record in finished]
        admission_delay = [(record.admitted_ns - record.arrival_ns) / NANOSECONDS_PER_MILLISECOND for record in records if record.admitted_ns is not None]
        tpot = [
            (record.finish_ns - record.first_token_ns) / NANOSECONDS_PER_MILLISECOND / (record.actual_output_tokens - 1)
            for record in finished
            if record.actual_output_tokens > 1
        ]
        itl = [
            (right.emitted_ns - left.emitted_ns) / NANOSECONDS_PER_MILLISECOND
            for record in records
            for left, right in zip(record.token_records, record.token_records[1:])
        ]
        if finished:
            wall_seconds = (max(record.finish_ns for record in finished) - min(record.arrival_ns for record in records)) / NANOSECONDS_PER_SECOND
        else:
            wall_seconds = 0.0
        output_tokens = sum(record.actual_output_tokens for record in finished)
        good = []
        if ttft_slo_ms is not None or tpot_slo_ms is not None:
            for record in finished:
                request_ttft = (record.first_token_ns - record.arrival_ns) / NANOSECONDS_PER_MILLISECOND
                request_tpot = (
                    (record.finish_ns - record.first_token_ns) / NANOSECONDS_PER_MILLISECOND / (record.actual_output_tokens - 1)
                    if record.actual_output_tokens > 1
                    else 0.0
                )
                good.append(
                    (ttft_slo_ms is None or request_ttft <= ttft_slo_ms)
                    and (tpot_slo_ms is None or request_tpot <= tpot_slo_ms)
                )
        bucket_summaries = {}
        for bucket in sorted({record.bucket for record in records}):
            bucket_records = [record for record in finished if record.bucket == bucket]
            bucket_summaries[bucket] = {
                "requests": len(bucket_records),
                "ttft": summarize_latencies(
                    (record.first_token_ns - record.arrival_ns) / NANOSECONDS_PER_MILLISECOND
                    for record in bucket_records
                ),
                "tpot": summarize_latencies(
                    (record.finish_ns - record.first_token_ns)
                    / NANOSECONDS_PER_MILLISECOND
                    / (record.actual_output_tokens - 1)
                    for record in bucket_records
                    if record.actual_output_tokens > 1
                ),
                "e2e": summarize_latencies(
                    (record.finish_ns - record.arrival_ns) / NANOSECONDS_PER_MILLISECOND
                    for record in bucket_records
                ),
            }
        batch_kinds = [str(step.get("batch_kind", "unknown")) for step in self.steps]
        mode_switches = sum(left != right for left, right in zip(batch_kinds, batch_kinds[1:]))
        max_decode_streak = 0
        current_decode_streak = 0
        max_decode_gap = 0
        current_decode_gap = 0
        seen_decode = False
        for batch_kind in batch_kinds:
            if batch_kind == "decode":
                current_decode_streak += 1
                max_decode_streak = max(max_decode_streak, current_decode_streak)
                if seen_decode:
                    max_decode_gap = max(max_decode_gap, current_decode_gap)
                seen_decode = True
                current_decode_gap = 0
            else:
                current_decode_streak = 0
                if seen_decode:
                    current_decode_gap += 1
        kv_utilizations = [
            step["kv_used_blocks"] / step["kv_total_blocks"]
            for step in self.steps
            if step.get("kv_total_blocks", 0) > 0
        ]
        scheduler_summary = {
            "steps": len(self.steps),
            "prefill_steps": batch_kinds.count("prefill"),
            "decode_steps": batch_kinds.count("decode"),
            "idle_steps": batch_kinds.count("idle"),
            "forced_prefill_steps": sum(bool(step.get("forced_prefill")) for step in self.steps),
            "allocation_blocked_steps": sum(bool(step.get("allocation_blocked")) for step in self.steps),
            "preemptions": sum(int(step.get("preemptions", 0)) for step in self.steps),
            "mode_switches": mode_switches,
            "max_decode_streak": max_decode_streak,
            "max_decode_gap_steps": max_decode_gap,
            "max_waiting_requests": max(
                (int(step.get("waiting", 0)) for step in self.steps),
                default=0,
            ),
            "max_running_requests": max(
                (int(step.get("running", 0)) for step in self.steps),
                default=0,
            ),
            "peak_kv_block_utilization": max(kv_utilizations) if kv_utilizations else None,
            "step_elapsed": summarize_latencies(float(step["elapsed_ms"]) for step in self.steps if "elapsed_ms" in step),
        }
        return {
            "requests": {
                "offered": len(records),
                "admitted": sum(record.admitted_ns is not None for record in records),
                "finished": len(finished),
                "completion_rate": len(finished) / len(records) if records else 0.0,
                "exact_output_length": sum(
                    record.actual_output_tokens == record.requested_output_tokens for record in finished
                ),
            },
            "latency": {
                "admission_delay": summarize_latencies(admission_delay),
                "ttft": summarize_latencies(ttft),
                "tpot": summarize_latencies(tpot),
                "itl": summarize_latencies(itl),
                "e2e": summarize_latencies(e2e),
            },
            "throughput": {
                "wall_seconds": wall_seconds,
                "output_tokens": output_tokens,
                "output_tokens_per_second": output_tokens / wall_seconds if wall_seconds > 0 else 0.0,
                "requests_per_second": len(finished) / wall_seconds if wall_seconds > 0 else 0.0,
                "goodput_requests_per_second": sum(good) / wall_seconds if good and wall_seconds > 0 else None,
            },
            "by_bucket": bucket_summaries,
            "scheduler": scheduler_summary,
            "slo": {"ttft_ms": ttft_slo_ms, "tpot_ms": tpot_slo_ms},
        }

    def write_artifacts(
        self,
        output_dir: str | Path,
        *,
        manifest: dict[str, Any],
        ttft_slo_ms: float | None = None,
        tpot_slo_ms: float | None = None,
    ) -> dict[str, Any]:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self._write_json(directory / "manifest.json", manifest)
        self._write_jsonl(directory / "requests.jsonl", (record.to_dict() for record in self.requests.values()))
        self._write_jsonl(directory / "tokens.jsonl", (token.to_dict() for token in self.tokens))
        self._write_jsonl(directory / "steps.jsonl", self.steps)
        summary = self.summary(ttft_slo_ms=ttft_slo_ms, tpot_slo_ms=tpot_slo_ms)
        self._write_json(directory / "summary.json", summary)
        return summary

    def _request(self, request_id: str) -> RequestRecord:
        try:
            return self.requests[request_id]
        except KeyError as exc:
            raise KeyError(f"unknown request: {request_id}") from exc

    @staticmethod
    def _write_json(path: Path, value: dict[str, Any]) -> None:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @staticmethod
    def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as stream:
            for value in values:
                stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
