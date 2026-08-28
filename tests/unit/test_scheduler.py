import sys
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType

import pytest

# Import the scheduler without executing nanovllm/__init__.py.  The public
# package imports the CUDA engine eagerly, while these state-machine tests are
# deliberately runnable on hosts without torch, transformers, or a GPU.
if "nanovllm" not in sys.modules:
    package = ModuleType("nanovllm")
    package.__path__ = [str(Path(__file__).parents[2] / "nanovllm")]
    sys.modules["nanovllm"] = package

from nanovllm.engine.scheduler import (
    BatchKind,
    Scheduler,
    SchedulerQueueFullError,
)
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


def make_scheduler(
    policy="prefill_first",
    *,
    num_blocks=32,
    max_num_seqs=8,
    max_num_batched_tokens=64,
    max_consecutive_decode_steps=2,
    max_queue_size=32,
):
    Sequence.block_size = 4
    config = SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        eos=0,
        kvcache_block_size=4,
        num_kvcache_blocks=num_blocks,
        scheduler_policy=policy,
        max_consecutive_decode_steps=max_consecutive_decode_steps,
        max_queue_size=max_queue_size,
    )
    return Scheduler(config)


def make_seq(num_prompt_tokens=4, max_tokens=32):
    return Sequence(
        list(range(1, num_prompt_tokens + 1)),
        SamplingParams(temperature=1.0, max_tokens=max_tokens, ignore_eos=True),
    )


def finish_prefill(scheduler, seq, sampled_token=101):
    output = scheduler.schedule()
    assert output.batch_kind == BatchKind.PREFILL
    assert seq in output.seqs
    scheduler.postprocess(output.seqs, [sampled_token] * len(output.seqs), True)
    return output


def assert_block_invariants(scheduler):
    manager = scheduler.block_manager
    free = set(manager.free_block_ids)
    used = manager.used_block_ids
    assert free.isdisjoint(used)
    assert free | used == set(range(len(manager.blocks)))
    for block in manager.blocks:
        assert (block.block_id in used) == (block.ref_count > 0)
    for seq in [*scheduler.waiting, *scheduler.running]:
        assert set(seq.block_table) <= used


def test_prefill_first_preserves_baseline_priority():
    scheduler = make_scheduler("prefill_first")
    running = make_seq()
    scheduler.add(running)
    finish_prefill(scheduler, running)
    waiting = make_seq()
    scheduler.add(waiting)

    output = scheduler.schedule()

    assert output.batch_kind == BatchKind.PREFILL
    assert waiting in output.seqs
    assert output.forced_prefill is False


def test_decode_first_prioritizes_running_sequence():
    scheduler = make_scheduler("decode_first")
    running = make_seq()
    scheduler.add(running)
    finish_prefill(scheduler, running)
    waiting = make_seq()
    scheduler.add(waiting)

    output = scheduler.schedule()

    assert output.batch_kind == BatchKind.DECODE
    assert output.seqs == [running]
    assert list(scheduler.waiting) == [waiting]


def test_bounded_decode_forces_prefill_after_configured_streak():
    scheduler = make_scheduler("bounded_decode_first", max_consecutive_decode_steps=2)
    running = make_seq()
    scheduler.add(running)
    finish_prefill(scheduler, running)
    waiting = make_seq()
    scheduler.add(waiting)

    for token in (201, 202):
        output = scheduler.schedule()
        assert output.batch_kind == BatchKind.DECODE
        scheduler.postprocess(output.seqs, [token], False)

    output = scheduler.schedule()

    assert output.batch_kind == BatchKind.PREFILL
    assert output.forced_prefill is True
    assert waiting in output.seqs
    assert scheduler.consecutive_decode_steps == 0
    assert scheduler.forced_prefill_count == 1


def test_decode_only_work_does_not_age_future_request():
    scheduler = make_scheduler("bounded_decode_first", max_consecutive_decode_steps=1)
    running = make_seq()
    scheduler.add(running)
    finish_prefill(scheduler, running)

    output = scheduler.schedule()
    scheduler.postprocess(output.seqs, [201], False)
    assert scheduler.consecutive_decode_steps == 0

    waiting = make_seq()
    scheduler.add(waiting)
    output = scheduler.schedule()
    assert output.batch_kind == BatchKind.DECODE


def test_forced_prefill_falls_back_when_resident_capacity_is_full():
    scheduler = make_scheduler(
        "bounded_decode_first",
        max_num_seqs=1,
        max_consecutive_decode_steps=1,
    )
    running = make_seq()
    scheduler.add(running)
    finish_prefill(scheduler, running)
    waiting = make_seq()
    scheduler.add(waiting)
    scheduler.consecutive_decode_steps = 1

    output = scheduler.schedule()

    assert output.batch_kind == BatchKind.DECODE
    assert output.forced_prefill is False
    assert output.allocation_blocked is True
    assert scheduler.resident_count == 1


def test_resident_sequences_never_exceed_limit():
    scheduler = make_scheduler("prefill_first", max_num_seqs=1)
    first = make_seq()
    scheduler.add(first)
    finish_prefill(scheduler, first)
    second = make_seq()
    scheduler.add(second)

    output = scheduler.schedule()

    assert output.batch_kind == BatchKind.DECODE
    assert scheduler.resident_count == 1
    assert not second.block_table


def test_partial_prefill_abort_releases_every_block():
    scheduler = make_scheduler(max_num_batched_tokens=2)
    seq = make_seq(num_prompt_tokens=8)
    scheduler.add(seq)
    output = scheduler.schedule()
    assert output.batch_kind == BatchKind.PREFILL
    scheduler.postprocess(output.seqs, [101], True)
    assert seq in scheduler.waiting
    assert seq.block_table

    aborted = scheduler.abort(seq.seq_id)

    assert aborted is seq
    assert seq.status == SequenceStatus.ABORTED
    assert seq.finish_reason == "abort"
    assert not seq.block_table
    assert len(scheduler.block_manager.used_block_ids) == 0
    assert_block_invariants(scheduler)


def test_running_abort_releases_blocks_and_is_idempotent():
    scheduler = make_scheduler()
    seq = make_seq()
    scheduler.add(seq)
    finish_prefill(scheduler, seq)

    assert scheduler.abort(seq.seq_id) is seq
    assert scheduler.abort(seq.seq_id) is None
    assert seq.is_aborted
    assert scheduler.is_finished()
    assert_block_invariants(scheduler)


def test_abort_preserves_shared_prefix_reference():
    scheduler = make_scheduler(num_blocks=16)
    first = make_seq(num_prompt_tokens=8)
    scheduler.add(first)
    finish_prefill(scheduler, first)
    second = make_seq(num_prompt_tokens=8)
    scheduler.add(second)
    finish_prefill(scheduler, second)
    shared_block = set(first.block_table) & set(second.block_table)
    assert shared_block

    scheduler.abort(second.seq_id)

    for block_id in shared_block:
        assert scheduler.block_manager.blocks[block_id].ref_count == 1
        assert block_id in scheduler.block_manager.used_block_ids
    assert_block_invariants(scheduler)


def test_self_preemption_returns_idle_instead_of_asserting():
    scheduler = make_scheduler(num_blocks=1)
    seq = make_seq(num_prompt_tokens=4)
    scheduler.add(seq)
    finish_prefill(scheduler, seq)

    output = scheduler.schedule()

    assert output.batch_kind == BatchKind.IDLE
    assert output.seqs == []
    assert output.preempted_seq_ids == (seq.seq_id,)
    assert output.allocation_blocked is True
    assert seq.status == SequenceStatus.WAITING
    assert seq.num_preemptions == 1
    assert scheduler.total_preemptions == 1
    assert_block_invariants(scheduler)


def test_queue_limit_rejects_excess_waiting_requests():
    scheduler = make_scheduler(max_queue_size=1)
    scheduler.add(make_seq())

    with pytest.raises(SchedulerQueueFullError):
        scheduler.add(make_seq())


def test_queue_limit_counts_running_requests_for_real_backpressure():
    scheduler = make_scheduler(max_queue_size=1)
    running = make_seq()
    scheduler.add(running)
    finish_prefill(scheduler, running)

    with pytest.raises(SchedulerQueueFullError):
        scheduler.add(make_seq())


def test_empty_scheduler_returns_idle_and_legacy_unpacking_works():
    scheduler = make_scheduler()

    output = scheduler.schedule()
    seqs, is_prefill = output

    assert output.batch_kind == BatchKind.IDLE
    assert seqs == []
    assert is_prefill is False


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"scheduler_policy": "unknown"}, "unsupported scheduler_policy"),
        ({"max_consecutive_decode_steps": 0}, "max_consecutive_decode_steps"),
        ({"max_queue_size": 0}, "max_queue_size"),
    ],
)
def test_scheduler_config_validation_is_explicit(monkeypatch, tmp_path, overrides, message):
    fake_transformers = ModuleType("transformers")

    class FakeAutoConfig:
        max_position_embeddings = 4096

        @classmethod
        def from_pretrained(cls, _model):
            return cls()

    fake_transformers.AutoConfig = FakeAutoConfig
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    sys.modules.pop("nanovllm.config", None)
    from nanovllm.config import Config

    with pytest.raises(ValueError, match=message):
        Config(str(tmp_path), **overrides)
