"""Deterministic request traces for online scheduler experiments.

Arrival offsets are part of the trace.  The runner measures latency from those
ideal arrival times rather than from the later instant at which a blocking
engine loop happens to admit a request.  This avoids coordinated omission.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Iterable, Sequence


NANOSECONDS_PER_SECOND = 1_000_000_000
DEFAULT_VOCAB_UPPER_BOUND = 150_000  # Stay below Qwen3 special-token IDs.


@dataclass(frozen=True, slots=True)
class WorkloadBucket:
    """One prompt/output shape in a weighted workload distribution."""

    name: str
    prompt_tokens: int
    output_tokens: int
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("bucket name must not be empty")
        if self.prompt_tokens <= 0:
            raise ValueError("prompt_tokens must be positive")
        if self.output_tokens <= 0:
            raise ValueError("output_tokens must be positive")
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("bucket weight must be finite and positive")


@dataclass(frozen=True, slots=True)
class RequestSpec:
    """A request whose arrival time and token lengths are known in advance."""

    request_id: str
    arrival_offset_ns: int
    prompt_token_ids: tuple[int, ...]
    output_tokens: int
    bucket: str = "fixed"
    arrival_anchor: str = "start"

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.arrival_offset_ns < 0:
            raise ValueError("arrival_offset_ns must be non-negative")
        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty")
        if any(token_id < 0 for token_id in self.prompt_token_ids):
            raise ValueError("prompt token IDs must be non-negative")
        if self.output_tokens <= 0:
            raise ValueError("output_tokens must be positive")
        if self.arrival_anchor not in {"start", "barrier"}:
            raise ValueError("arrival_anchor must be 'start' or 'barrier'")

    @property
    def prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "arrival_offset_ns": self.arrival_offset_ns,
            "prompt_token_ids": list(self.prompt_token_ids),
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "bucket": self.bucket,
            "arrival_anchor": self.arrival_anchor,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "RequestSpec":
        token_ids = value["prompt_token_ids"]
        if not isinstance(token_ids, list):
            raise ValueError("prompt_token_ids must be a JSON array")
        return cls(
            request_id=str(value["request_id"]),
            arrival_offset_ns=int(value["arrival_offset_ns"]),
            prompt_token_ids=tuple(int(token_id) for token_id in token_ids),
            output_tokens=int(value["output_tokens"]),
            bucket=str(value.get("bucket", "fixed")),
            arrival_anchor=str(value.get("arrival_anchor", "start")),
        )


@dataclass(frozen=True, slots=True)
class WorkloadTrace:
    """A complete, replayable arrival trace."""

    name: str
    seed: int
    requests: tuple[RequestSpec, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("trace name must not be empty")
        if not self.requests:
            raise ValueError("trace must contain at least one request")
        ids = [request.request_id for request in self.requests]
        if len(ids) != len(set(ids)):
            raise ValueError("request IDs must be unique")
        anchors = [request.arrival_anchor for request in self.requests]
        if "barrier" in anchors:
            first_barrier = anchors.index("barrier")
            if any(anchor != "start" for anchor in anchors[:first_barrier]):
                raise ValueError("start-anchored requests must precede barrier requests")
            if any(anchor != "barrier" for anchor in anchors[first_barrier:]):
                raise ValueError("barrier-anchored requests must form one trailing group")
        for anchor in {"start", "barrier"}:
            arrivals = [
                request.arrival_offset_ns
                for request in self.requests
                if request.arrival_anchor == anchor
            ]
            if arrivals != sorted(arrivals):
                raise ValueError(
                    f"{anchor}-anchored requests must be ordered by arrival_offset_ns"
                )

    def write_jsonl(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as stream:
            for request in self.requests:
                stream.write(json.dumps(request.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")

    @classmethod
    def read_jsonl(cls, path: str | Path, *, name: str, seed: int) -> "WorkloadTrace":
        requests = []
        with Path(path).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    requests.append(RequestSpec.from_dict(json.loads(line)))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"invalid workload JSONL at line {line_number}: {exc}") from exc
        return cls(name=name, seed=seed, requests=tuple(requests))


def _weighted_bucket(rng: random.Random, buckets: Sequence[WorkloadBucket]) -> WorkloadBucket:
    total = sum(bucket.weight for bucket in buckets)
    selected = rng.random() * total
    cumulative = 0.0
    for bucket in buckets:
        cumulative += bucket.weight
        if selected < cumulative:
            return bucket
    return buckets[-1]


def _make_prompt(rng: random.Random, length: int, vocab_upper_bound: int) -> tuple[int, ...]:
    if vocab_upper_bound <= 1:
        raise ValueError("vocab_upper_bound must be greater than one")
    return tuple(rng.randrange(1, vocab_upper_bound) for _ in range(length))


def _build_trace(
    *,
    name: str,
    seed: int,
    arrival_offsets_ns: Iterable[int],
    buckets: Sequence[WorkloadBucket],
    vocab_upper_bound: int,
    arrival_anchor: str = "start",
    request_id_prefix: str = "request",
) -> WorkloadTrace:
    if not buckets:
        raise ValueError("at least one workload bucket is required")
    shape_rng = random.Random(seed ^ 0x51A7_E5ED)
    prompt_rng = random.Random(seed ^ 0xC0DE_4060)
    requests = []
    for index, arrival_offset_ns in enumerate(arrival_offsets_ns):
        bucket = _weighted_bucket(shape_rng, buckets)
        requests.append(
            RequestSpec(
                request_id=f"{request_id_prefix}-{index:06d}",
                arrival_offset_ns=arrival_offset_ns,
                prompt_token_ids=_make_prompt(prompt_rng, bucket.prompt_tokens, vocab_upper_bound),
                output_tokens=bucket.output_tokens,
                bucket=bucket.name,
                arrival_anchor=arrival_anchor,
            )
        )
    return WorkloadTrace(name=name, seed=seed, requests=tuple(requests))


def make_fixed_trace(
    *,
    num_requests: int,
    interval_seconds: float,
    prompt_tokens: int,
    output_tokens: int,
    seed: int,
    name: str = "fixed",
    vocab_upper_bound: int = DEFAULT_VOCAB_UPPER_BOUND,
) -> WorkloadTrace:
    """Create a trace with a fixed inter-arrival interval and fixed shapes."""

    if num_requests <= 0:
        raise ValueError("num_requests must be positive")
    if not math.isfinite(interval_seconds) or interval_seconds < 0:
        raise ValueError("interval_seconds must be finite and non-negative")
    interval_ns = round(interval_seconds * NANOSECONDS_PER_SECOND)
    arrivals = (index * interval_ns for index in range(num_requests))
    bucket = WorkloadBucket("fixed", prompt_tokens, output_tokens)
    return _build_trace(
        name=name,
        seed=seed,
        arrival_offsets_ns=arrivals,
        buckets=(bucket,),
        vocab_upper_bound=vocab_upper_bound,
    )


def make_poisson_trace(
    *,
    num_requests: int,
    request_rate: float,
    buckets: Sequence[WorkloadBucket],
    seed: int,
    name: str = "poisson",
    vocab_upper_bound: int = DEFAULT_VOCAB_UPPER_BOUND,
) -> WorkloadTrace:
    """Create an open-loop Poisson arrival trace.

    The first request arrives at offset zero. Subsequent inter-arrival times are
    sampled independently, and nanosecond offsets are forced to be monotonic.
    """

    if num_requests <= 0:
        raise ValueError("num_requests must be positive")
    if not math.isfinite(request_rate) or request_rate <= 0:
        raise ValueError("request_rate must be finite and positive")
    arrival_rng = random.Random(seed ^ 0xA771_A1)
    arrivals = [0]
    elapsed_ns = 0
    for _ in range(1, num_requests):
        delta_ns = max(1, round(arrival_rng.expovariate(request_rate) * NANOSECONDS_PER_SECOND))
        elapsed_ns += delta_ns
        arrivals.append(elapsed_ns)
    return _build_trace(
        name=name,
        seed=seed,
        arrival_offsets_ns=arrivals,
        buckets=tuple(buckets),
        vocab_upper_bound=vocab_upper_bound,
    )


def make_interference_trace(
    *,
    decode_requests: int = 8,
    injected_prefill_requests: int = 8,
    injection_interval_seconds: float = 0.1,
    seed: int = 20260822,
    decode_prompt_tokens: int = 128,
    decode_output_tokens: int = 256,
    prefill_prompt_tokens: int = 1024,
    prefill_output_tokens: int = 32,
    name: str = "barrier-interference",
    vocab_upper_bound: int = DEFAULT_VOCAB_UPPER_BOUND,
) -> WorkloadTrace:
    """Create a two-stage trace gated on all stage-one first tokens.

    ``barrier`` offsets are relative to the instant at which every initial
    Decode-heavy request has emitted its first token, not process start.
    """

    if decode_requests <= 0 or injected_prefill_requests <= 0:
        raise ValueError("interference request counts must be positive")
    if not math.isfinite(injection_interval_seconds) or injection_interval_seconds < 0:
        raise ValueError("injection_interval_seconds must be finite and non-negative")
    decode_trace = _build_trace(
        name=name,
        seed=seed,
        arrival_offsets_ns=(0 for _ in range(decode_requests)),
        buckets=(
            WorkloadBucket(
                "decode-stage",
                decode_prompt_tokens,
                decode_output_tokens,
            ),
        ),
        vocab_upper_bound=vocab_upper_bound,
        request_id_prefix="decode",
    )
    interval_ns = round(injection_interval_seconds * NANOSECONDS_PER_SECOND)
    prefill_trace = _build_trace(
        name=name,
        seed=seed ^ 0x1A7E_F00D,
        arrival_offsets_ns=(index * interval_ns for index in range(injected_prefill_requests)),
        buckets=(
            WorkloadBucket(
                "injected-prefill",
                prefill_prompt_tokens,
                prefill_output_tokens,
            ),
        ),
        vocab_upper_bound=vocab_upper_bound,
        arrival_anchor="barrier",
        request_id_prefix="prefill",
    )
    return WorkloadTrace(
        name=name,
        seed=seed,
        requests=decode_trace.requests + prefill_trace.requests,
    )


RTX4060_MIXED_BUCKETS = (
    WorkloadBucket("short", prompt_tokens=128, output_tokens=128, weight=0.60),
    WorkloadBucket("medium", prompt_tokens=512, output_tokens=128, weight=0.30),
    WorkloadBucket("long", prompt_tokens=1024, output_tokens=64, weight=0.10),
)

RTX4060_DECODE_HEAVY_BUCKETS = (
    WorkloadBucket("decode-heavy", prompt_tokens=128, output_tokens=256),
)

RTX4060_PREFILL_HEAVY_BUCKETS = (
    WorkloadBucket("prefill-heavy", prompt_tokens=1024, output_tokens=32),
)
