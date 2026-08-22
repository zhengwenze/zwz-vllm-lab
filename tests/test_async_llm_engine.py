import asyncio
from collections import deque
from dataclasses import dataclass
from threading import Event, get_ident
from time import perf_counter_ns, sleep

import pytest

from nanovllm.engine.async_llm_engine import AsyncLLMEngine
from nanovllm.engine.errors import DuplicateRequestError, RequestQueueFullError
from nanovllm.engine.outputs import RequestOutput, StepResult, StepStats
from nanovllm.sampling_params import SamplingParams


@dataclass
class FakeRequest:
    request_id: str
    remaining: int
    tokens: list[int]


class FakeSyncEngine:
    def __init__(self, _model, **_kwargs):
        self.requests = deque()
        self.known = set()
        self.pending = deque()
        self.owner_threads = set()
        self.step_id = 0
        self.exited = False

    def _own(self):
        self.owner_threads.add(get_ident())

    def add_request(self, _prompt, sampling_params, request_id=None):
        self._own()
        if request_id in self.known:
            raise DuplicateRequestError(f"duplicate: {request_id}")
        self.known.add(request_id)
        self.requests.append(FakeRequest(request_id, sampling_params.max_tokens, []))
        return request_id

    def abort_request(self, request_id):
        self._own()
        request = next((item for item in self.requests if item.request_id == request_id), None)
        if request is None:
            return False
        self.requests.remove(request)
        self.pending.append(self._output(request, None, True, "abort"))
        return True

    @property
    def has_pending_outputs(self):
        return bool(self.pending)

    def is_finished(self):
        self._own()
        return not self.requests

    def step_stream(self):
        self._own()
        sleep(0.0005)
        outputs = list(self.pending)
        self.pending.clear()
        if self.requests:
            request = self.requests.popleft()
            request.tokens.append(100 + len(request.tokens))
            request.remaining -= 1
            finished = request.remaining == 0
            outputs.append(self._output(request, request.tokens[-1], finished, "length" if finished else None))
            if not finished:
                self.requests.append(request)
            batch_kind = "decode"
        else:
            batch_kind = "idle"
        stats = StepStats(
            step_id=self.step_id,
            batch_kind=batch_kind,
            batch_size=int(batch_kind != "idle"),
            scheduled_tokens=int(batch_kind != "idle"),
            waiting=0,
            running=len(self.requests),
            kv_used_blocks=len(self.requests),
            kv_total_blocks=8,
            preemptions=0,
            forced_prefill=False,
            allocation_blocked=False,
            elapsed_ms=0.1,
        )
        self.step_id += 1
        return StepResult(tuple(outputs), stats)

    def exit(self):
        self._own()
        self.exited = True

    @staticmethod
    def _output(request, token_id, finished, reason):
        return RequestOutput(
            request_id=request.request_id,
            sequence_id=1,
            token_id=token_id,
            token_ids=tuple(request.tokens),
            text=" ".join(str(token) for token in request.tokens),
            finished=finished,
            finish_reason=reason,
            timestamp_ns=perf_counter_ns(),
        )


class ExplodingEngine(FakeSyncEngine):
    def __init__(self, model):
        super().__init__(model)
        self.entered_step = Event()
        self.release_step = Event()

    def step_stream(self):
        self._own()
        self.entered_step.set()
        self.release_step.wait(timeout=2)
        raise RuntimeError("synthetic worker failure")


@pytest.mark.asyncio
async def test_dynamic_streams_are_isolated_and_cuda_owner_is_one_thread():
    fake = FakeSyncEngine("unused")
    engine = AsyncLLMEngine("unused", engine_factory=lambda *_args, **_kwargs: fake)
    first = await engine.submit([1], SamplingParams(max_tokens=2), "first")
    second = await engine.submit([2], SamplingParams(max_tokens=2), "second")

    first_outputs, second_outputs = await asyncio.gather(
        _collect(first),
        _collect(second),
    )
    snapshot = engine.metrics_snapshot()
    await engine.shutdown()

    assert [output.request_id for output in first_outputs] == ["first", "first"]
    assert [output.request_id for output in second_outputs] == ["second", "second"]
    assert first_outputs[-1].token_ids == (100, 101)
    assert second_outputs[-1].token_ids == (100, 101)
    assert len(fake.owner_threads) == 1
    assert get_ident() not in fake.owner_threads
    assert fake.exited is True
    assert snapshot["emitted_tokens"] == 4
    assert snapshot["last_step"]["batch_kind"] == "decode"


@pytest.mark.asyncio
async def test_duplicate_id_error_crosses_worker_boundary():
    fake = FakeSyncEngine("unused")
    engine = AsyncLLMEngine("unused", engine_factory=lambda *_args, **_kwargs: fake)
    await engine.submit([1], SamplingParams(max_tokens=100), "same")

    with pytest.raises(DuplicateRequestError):
        await engine.submit([1], SamplingParams(max_tokens=1), "same")
    await engine.shutdown()


@pytest.mark.asyncio
async def test_abort_delivers_terminal_event_and_shutdown_is_idempotent():
    fake = FakeSyncEngine("unused")
    engine = AsyncLLMEngine("unused", engine_factory=lambda *_args, **_kwargs: fake)
    stream = await engine.submit([1], SamplingParams(max_tokens=100), "cancel")

    assert await engine.abort("cancel") is True
    outputs = await _collect(stream)
    await engine.shutdown()
    await engine.shutdown()

    assert outputs[-1].finished is True
    assert outputs[-1].finish_reason == "abort"


@pytest.mark.asyncio
async def test_slow_consumer_is_aborted_with_explicit_backpressure_error():
    fake = FakeSyncEngine("unused")
    engine = AsyncLLMEngine(
        "unused",
        output_queue_size=1,
        engine_factory=lambda *_args, **_kwargs: fake,
    )
    stream = await engine.submit([1], SamplingParams(max_tokens=100), "slow")
    await asyncio.sleep(0.02)

    with pytest.raises(RequestQueueFullError, match="could not keep up"):
        await stream.__anext__()
    await engine.shutdown()


@pytest.mark.asyncio
async def test_worker_failure_rejects_commands_queued_during_gpu_step():
    fake = ExplodingEngine("unused")
    engine = AsyncLLMEngine("unused", engine_factory=lambda *_args, **_kwargs: fake)
    first = await engine.submit([1], SamplingParams(max_tokens=10), "first")
    assert await asyncio.to_thread(fake.entered_step.wait, 1)
    queued_submit = asyncio.create_task(
        engine.submit([2], SamplingParams(max_tokens=1), "queued")
    )
    await asyncio.sleep(0)
    fake.release_step.set()

    with pytest.raises(RuntimeError, match="synthetic worker failure"):
        await asyncio.wait_for(queued_submit, timeout=1)
    with pytest.raises(RuntimeError, match="synthetic worker failure"):
        await asyncio.wait_for(first.__anext__(), timeout=1)
    await engine.shutdown()


async def _collect(stream):
    return [output async for output in stream]
