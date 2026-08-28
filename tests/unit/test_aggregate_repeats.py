import json

import pytest

from benchmarks.online_scheduler.aggregate_repeats import (
    METRIC_LABELS,
    aggregate_repeats,
    summarize_values,
)


def _write_run(
    root, name: str, policy: str, repeat_index: int, throughput: float, ttft_p50: float
) -> None:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    manifest = {
        "run_id": name,
        "policy": policy,
        "repeat_index": repeat_index,
        "workload": {"num_requests": 100, "request_rate": 8.0, "seed": 20260827},
        "engine_config": {"enforce_eager": True},
    }
    summary = {
        "requests": {"completion_rate": 1.0},
        "throughput": {
            "output_tokens_per_second": throughput,
            "requests_per_second": throughput / 100,
        },
        "latency": {
            "ttft": {"p50_ms": ttft_p50, "p95_ms": ttft_p50 * 2},
            "tpot": {"p50_ms": 30.0, "p95_ms": 35.0},
            "e2e": {"p50_ms": 1000.0, "p95_ms": 2000.0},
            "itl": {"p95_ms": 50.0},
        },
        "scheduler": {
            "max_waiting_requests": 10,
            "max_running_requests": 5,
            "forced_prefill_steps": 7,
            "peak_kv_block_utilization": 0.3,
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_summarize_values_reports_cv_and_range() -> None:
    stats = summarize_values([590.24, 588.0, 591.5])
    assert stats.count == 3
    assert stats.mean == pytest.approx(589.9133, rel=1e-4)
    assert stats.std is not None and stats.std > 0
    assert stats.cv is not None and stats.cv == pytest.approx(stats.std / stats.mean)
    assert stats.minimum == 588.0
    assert stats.maximum == 591.5


def test_summarize_values_single_value_has_no_std() -> None:
    stats = summarize_values([50.81])
    assert stats.count == 1
    assert stats.mean == 50.81
    assert stats.std is None
    assert stats.cv is None


def test_aggregate_repeats_groups_by_policy_and_filters_workload(tmp_path) -> None:
    # Three repeats of prefill_first plus one mismatched-workload run.
    _write_run(tmp_path, "r0-prefill", "prefill_first", 0, 590.0, 2043.0)
    _write_run(tmp_path, "r1-prefill", "prefill_first", 1, 588.0, 2060.0)
    _write_run(tmp_path, "r2-prefill", "prefill_first", 2, 591.0, 2050.0)
    wrong = tmp_path / "r0-wrong-workload"
    wrong.mkdir()
    (wrong / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "r0-wrong-workload",
                "policy": "prefill_first",
                "workload": {"num_requests": 300, "request_rate": 2.0, "seed": 999},
                "engine_config": {"enforce_eager": False},
            }
        ),
        encoding="utf-8",
    )
    (wrong / "summary.json").write_text(
        json.dumps(
            {
                "requests": {"completion_rate": 1.0},
                "throughput": {
                    "output_tokens_per_second": 1.0,
                    "requests_per_second": 0.01,
                },
                "latency": {},
                "scheduler": {},
            }
        ),
        encoding="utf-8",
    )

    result = aggregate_repeats(tmp_path)
    assert "prefill_first" in result["policies"]
    entry = result["policies"]["prefill_first"]
    assert entry["repeat_count"] == 3
    assert [metric for metric in METRIC_LABELS if entry["metrics"][metric]["count"] > 0]
    throughput = entry["metrics"]["output_tokens_per_second"]
    assert throughput["count"] == 3
    assert throughput["mean"] == pytest.approx(589.6667, rel=1e-4)
    assert throughput["cv"] is not None


def test_aggregate_repeats_skips_incomplete_completion(tmp_path) -> None:
    run_dir = tmp_path / "r0-incomplete"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "r0-incomplete",
                "policy": "bounded_decode_first",
                "workload": {
                    "num_requests": 100,
                    "request_rate": 8.0,
                    "seed": 20260827,
                },
                "engine_config": {"enforce_eager": True},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "requests": {"completion_rate": 0.99},
                "throughput": {},
                "latency": {},
                "scheduler": {},
            }
        ),
        encoding="utf-8",
    )
    result = aggregate_repeats(tmp_path)
    assert result["policies"] == {}
