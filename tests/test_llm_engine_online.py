from collections import deque
from types import SimpleNamespace

import pytest

from nanovllm.engine.errors import DuplicateRequestError, RequestTooLongError
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class FakeTokenizer:
    eos_token_id = 0

    def encode(self, prompt):
        return [ord(character) % 31 + 1 for character in prompt]

    def decode(self, token_ids):
        return " ".join(str(token_id) for token_id in token_ids)


class FakeModelRunner:
    def __init__(self, tokens):
        self.tokens = deque(tokens)
        self.calls = 0

    def call(self, method, seqs, is_prefill):
        assert method == "run"
        self.calls += 1
        return [self.tokens.popleft() for _ in seqs]


def make_engine(*, tokens=(101, 102, 103), num_blocks=8, max_queue_size=8):
    Sequence.block_size = 4
    config = SimpleNamespace(
        max_num_seqs=4,
        max_num_batched_tokens=8,
        max_model_len=16,
        gpu_memory_utilization=0.8,
        tensor_parallel_size=1,
        enforce_eager=True,
        eos=0,
        kvcache_block_size=4,
        num_kvcache_blocks=num_blocks,
        scheduler_policy="prefill_first",
        max_consecutive_decode_steps=2,
        max_queue_size=max_queue_size,
    )
    engine = LLMEngine.__new__(LLMEngine)
    engine.config = config
    engine.ps = []
    engine.events = []
    engine._closed = False
    engine._step_id = 0
    engine._request_to_seq = {}
    engine._seq_to_request = {}
    engine._known_request_ids = set()
    engine._pending_outputs = deque()
    engine.tokenizer = FakeTokenizer()
    engine.model_runner = FakeModelRunner(tokens)
    engine.scheduler = Scheduler(config)
    return engine


def test_stream_step_emits_cumulative_tokens_and_cleans_active_mapping():
    engine = make_engine(tokens=(101, 102))
    request_id = engine.add_request(
        [1, 2, 3, 4],
        SamplingParams(max_tokens=2, ignore_eos=True),
        request_id="req-1",
    )

    first = engine.step_stream()
    second = engine.step_stream()

    assert request_id == "req-1"
    assert first.stats.batch_kind == "prefill"
    assert first.outputs[0].token_ids == (101,)
    assert first.outputs[0].finished is False
    assert second.stats.batch_kind == "decode"
    assert second.outputs[0].token_ids == (101, 102)
    assert second.outputs[0].text == "101 102"
    assert second.outputs[0].finish_reason == "length"
    assert engine.is_finished()
    assert not engine._request_to_seq
    assert not engine.scheduler.block_manager.used_block_ids


def test_abort_is_idempotent_and_emits_terminal_event_without_gpu_step():
    engine = make_engine()
    engine.add_request(
        [1, 2, 3, 4],
        SamplingParams(max_tokens=2),
        request_id="cancel-me",
    )

    assert engine.abort_request("cancel-me") is True
    assert engine.abort_request("cancel-me") is False
    result = engine.step_stream()

    assert result.stats.batch_kind == "idle"
    assert result.outputs[0].request_id == "cancel-me"
    assert result.outputs[0].token_id is None
    assert result.outputs[0].finished is True
    assert result.outputs[0].finish_reason == "abort"
    assert engine.model_runner.calls == 0
    assert engine.is_finished()


def test_duplicate_and_impossible_requests_are_rejected_before_queueing():
    engine = make_engine(num_blocks=2)
    params = SamplingParams(max_tokens=1)
    engine.add_request([1], params, request_id="unique")

    with pytest.raises(DuplicateRequestError):
        engine.add_request([2], params, request_id="unique")
    with pytest.raises(RequestTooLongError, match="max_model_len"):
        engine.add_request([1] * 16, params, request_id="too-long")
    assert list(engine._request_to_seq) == ["unique"]


def test_idle_step_never_invokes_model_runner():
    engine = make_engine()

    result = engine.step_stream()

    assert result.stats.batch_kind == "idle"
    assert result.outputs == ()
    assert engine.model_runner.calls == 0


def test_legacy_step_contract_returns_only_finished_sequences():
    engine = make_engine(tokens=(101,))
    engine.add_request(
        [1, 2],
        SamplingParams(max_tokens=1, ignore_eos=True),
        request_id="legacy",
    )

    outputs, scheduled_tokens = engine.step()

    assert len(outputs) == 1
    assert outputs[0][1] == [101]
    assert scheduled_tokens == 2
