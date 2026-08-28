from pathlib import Path

import pytest

from benchmarks.online_scheduler.workload import (
    RTX4060_MIXED_BUCKETS,
    WorkloadBucket,
    WorkloadTrace,
    make_fixed_trace,
    make_interference_trace,
    make_poisson_trace,
)


def test_fixed_trace_has_exact_arrivals_and_lengths() -> None:
    trace = make_fixed_trace(
        num_requests=3,
        interval_seconds=0.25,
        prompt_tokens=7,
        output_tokens=5,
        seed=42,
    )

    assert [request.arrival_offset_ns for request in trace.requests] == [0, 250_000_000, 500_000_000]
    assert [request.prompt_tokens for request in trace.requests] == [7, 7, 7]
    assert [request.output_tokens for request in trace.requests] == [5, 5, 5]


def test_poisson_trace_is_deterministic_for_the_same_seed() -> None:
    kwargs = dict(
        num_requests=100,
        request_rate=2.0,
        buckets=RTX4060_MIXED_BUCKETS,
        seed=20260822,
    )

    first = make_poisson_trace(**kwargs)
    second = make_poisson_trace(**kwargs)

    assert first == second
    assert first.requests[0].arrival_offset_ns == 0
    assert all(
        left.arrival_offset_ns < right.arrival_offset_ns
        for left, right in zip(first.requests, first.requests[1:])
    )
    assert {request.bucket for request in first.requests}.issubset({"short", "medium", "long"})


def test_different_seed_changes_poisson_trace() -> None:
    first = make_poisson_trace(
        num_requests=5,
        request_rate=1.0,
        buckets=(WorkloadBucket("shape", 4, 2),),
        seed=1,
    )
    second = make_poisson_trace(
        num_requests=5,
        request_rate=1.0,
        buckets=(WorkloadBucket("shape", 4, 2),),
        seed=2,
    )

    assert first != second


def test_trace_jsonl_round_trip(tmp_path: Path) -> None:
    trace = make_poisson_trace(
        num_requests=10,
        request_rate=3.0,
        buckets=RTX4060_MIXED_BUCKETS,
        seed=9,
        name="round-trip",
    )
    path = tmp_path / "workload.jsonl"

    trace.write_jsonl(path)
    restored = WorkloadTrace.read_jsonl(path, name=trace.name, seed=trace.seed)

    assert restored == trace


@pytest.mark.parametrize("request_rate", [0, -1, float("inf"), float("nan")])
def test_invalid_poisson_rate_is_rejected(request_rate: float) -> None:
    with pytest.raises(ValueError, match="request_rate"):
        make_poisson_trace(
            num_requests=2,
            request_rate=request_rate,
            buckets=(WorkloadBucket("shape", 4, 2),),
            seed=1,
        )


def test_trace_rejects_out_of_order_arrivals() -> None:
    valid = make_fixed_trace(
        num_requests=2,
        interval_seconds=1,
        prompt_tokens=4,
        output_tokens=2,
        seed=1,
    )

    with pytest.raises(ValueError, match="ordered"):
        WorkloadTrace(name="bad", seed=1, requests=tuple(reversed(valid.requests)))


def test_interference_trace_has_a_first_token_barrier_and_fixed_injection_spacing() -> None:
    trace = make_interference_trace(
        decode_requests=2,
        injected_prefill_requests=3,
        injection_interval_seconds=0.1,
        seed=7,
        decode_prompt_tokens=4,
        decode_output_tokens=3,
        prefill_prompt_tokens=8,
        prefill_output_tokens=1,
    )

    start_requests = [request for request in trace.requests if request.arrival_anchor == "start"]
    injected = [request for request in trace.requests if request.arrival_anchor == "barrier"]
    assert len(start_requests) == 2
    assert [request.arrival_offset_ns for request in injected] == [0, 100_000_000, 200_000_000]
    assert {request.bucket for request in injected} == {"injected-prefill"}
