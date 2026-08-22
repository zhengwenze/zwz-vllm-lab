import json

import pytest

from benchmarks.online_scheduler.metrics import MetricsCollector, percentile, summarize_latencies
from benchmarks.online_scheduler.workload import RequestSpec


def request_spec(request_id: str = "request-0") -> RequestSpec:
    return RequestSpec(
        request_id=request_id,
        arrival_offset_ns=0,
        prompt_token_ids=(1, 2, 3),
        output_tokens=3,
        bucket="test",
    )


def test_percentile_uses_linear_interpolation() -> None:
    assert percentile([], 0.50) is None
    assert percentile([0, 10, 20, 30], 0.50) == 15
    assert percentile([0, 10, 20, 30], 0.95) == pytest.approx(28.5)


def test_latency_summary_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        summarize_latencies([1.0, -1.0])


def test_request_metrics_use_ideal_arrival_and_token_timestamps() -> None:
    collector = MetricsCollector()
    collector.register_request(request_spec(), arrival_ns=1_000_000_000)
    collector.mark_admitted("request-0", admitted_ns=1_010_000_000)
    collector.record_token("request-0", token_id=10, emitted_ns=1_110_000_000)
    collector.record_token("request-0", token_id=11, emitted_ns=1_130_000_000)
    collector.record_token("request-0", token_id=12, emitted_ns=1_170_000_000, finished=True)

    summary = collector.summary(ttft_slo_ms=150, tpot_slo_ms=50)

    assert summary["latency"]["admission_delay"]["p50_ms"] == 10
    assert summary["latency"]["ttft"]["p50_ms"] == 110
    assert summary["latency"]["itl"]["mean_ms"] == 30
    assert summary["latency"]["tpot"]["p50_ms"] == 30
    assert summary["latency"]["e2e"]["p50_ms"] == 170
    assert summary["requests"]["completion_rate"] == 1
    assert summary["throughput"]["goodput_requests_per_second"] == pytest.approx(1 / 0.17)


def test_collector_rejects_invalid_lifecycle() -> None:
    collector = MetricsCollector()
    collector.register_request(request_spec(), arrival_ns=100)

    with pytest.raises(ValueError, match="before admission"):
        collector.record_token("request-0", token_id=1, emitted_ns=101)
    with pytest.raises(ValueError, match="precedes ideal arrival"):
        collector.mark_admitted("request-0", admitted_ns=99)


def test_raw_artifacts_are_machine_readable(tmp_path) -> None:
    collector = MetricsCollector()
    collector.register_request(request_spec(), arrival_ns=100)
    collector.mark_admitted("request-0", admitted_ns=100)
    collector.record_token("request-0", token_id=1, emitted_ns=110)
    collector.record_token("request-0", token_id=2, emitted_ns=120)
    collector.record_token("request-0", token_id=3, emitted_ns=130, finished=True)
    collector.add_step({"step_id": 1, "batch_kind": "decode"})

    collector.write_artifacts(tmp_path, manifest={"run_id": "unit-test"})

    assert json.loads((tmp_path / "manifest.json").read_text())["run_id"] == "unit-test"
    assert json.loads((tmp_path / "summary.json").read_text())["requests"]["finished"] == 1
    assert len((tmp_path / "requests.jsonl").read_text().splitlines()) == 1
    assert len((tmp_path / "tokens.jsonl").read_text().splitlines()) == 3
    assert len((tmp_path / "steps.jsonl").read_text().splitlines()) == 1


def test_scheduler_summary_and_request_waiting_are_derived_from_step_trace() -> None:
    collector = MetricsCollector()
    collector.register_request(request_spec(), arrival_ns=100)
    collector.mark_admitted("request-0", admitted_ns=100)
    collector.add_step(
        {
            "batch_kind": "decode",
            "waiting_request_ids": ("request-0",),
            "preempted_request_ids": (),
            "waiting": 1,
            "running": 2,
        }
    )
    collector.add_step(
        {
            "batch_kind": "decode",
            "waiting_request_ids": ("request-0",),
            "preempted_request_ids": ("request-0",),
            "waiting": 1,
            "running": 1,
        }
    )
    collector.add_step({"batch_kind": "prefill", "waiting": 0, "running": 2})
    collector.add_step({"batch_kind": "decode", "waiting": 0, "running": 2})

    summary = collector.summary()

    assert collector.requests["request-0"].waiting_steps == 2
    assert collector.requests["request-0"].preemption_count == 1
    assert summary["scheduler"]["mode_switches"] == 2
    assert summary["scheduler"]["max_decode_streak"] == 2
    assert summary["scheduler"]["max_decode_gap_steps"] == 1
