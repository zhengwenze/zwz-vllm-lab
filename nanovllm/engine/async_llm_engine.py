from __future__ import annotations

import asyncio
import queue
import threading
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, AsyncIterator, Callable
from uuid import uuid4

from nanovllm.engine.errors import EngineClosedError, RequestQueueFullError
from nanovllm.engine.outputs import RequestOutput, StepStats
from nanovllm.sampling_params import SamplingParams

if TYPE_CHECKING:
    from nanovllm.engine.llm_engine import LLMEngine


class _CommandKind(str, Enum):
    ADD = "add"
    ABORT = "abort"
    SHUTDOWN = "shutdown"


@dataclass(slots=True)
class _StreamState:
    request_id: str
    loop: asyncio.AbstractEventLoop
    queue: asyncio.Queue[object]
    overflowed: bool = False


@dataclass(slots=True)
class _Command:
    kind: _CommandKind
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future | None = None
    request_id: str | None = None
    prompt: str | list[int] | None = None
    sampling_params: SamplingParams | None = None
    stream: _StreamState | None = None


class AsyncRequestStream:
    """An admitted request whose outputs can be consumed asynchronously."""

    def __init__(self, engine: "AsyncLLMEngine", state: _StreamState):
        self.request_id = state.request_id
        self._engine = engine
        self._queue = state.queue
        self._finished = False

    def __aiter__(self) -> "AsyncRequestStream":
        return self

    async def __anext__(self) -> RequestOutput:
        if self._finished:
            raise StopAsyncIteration
        item = await self._queue.get()
        if isinstance(item, BaseException):
            self._finished = True
            raise item
        if not isinstance(item, RequestOutput):
            self._finished = True
            raise RuntimeError(f"unexpected stream item: {type(item)!r}")
        if item.finished:
            self._finished = True
        return item

    async def aclose(self) -> None:
        if not self._finished:
            await self._engine.abort(self.request_id)
            self._finished = True


class AsyncLLMEngine:
    """Thread-isolated online wrapper around the synchronous CUDA engine.

    Exactly one worker thread owns ``LLMEngine`` and all CUDA calls. Commands
    are applied between model steps, so abort never frees KV blocks while a
    forward pass is in flight.
    """

    def __init__(
        self,
        model: str,
        *,
        output_queue_size: int = 64,
        engine_factory: Callable[..., "LLMEngine"] | None = None,
        **engine_kwargs,
    ):
        if output_queue_size < 1:
            raise ValueError("output_queue_size must be at least 1")
        self.model = model
        self.engine_kwargs = engine_kwargs
        self.output_queue_size = output_queue_size
        if engine_factory is None:
            from nanovllm.engine.llm_engine import LLMEngine

            engine_factory = LLMEngine
        self._engine_factory = engine_factory
        self._commands: queue.Queue[_Command] = queue.Queue()
        self._condition = threading.Condition()
        self._state_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._ready_future: asyncio.Future | None = None
        self._owner_loop: asyncio.AbstractEventLoop | None = None
        self._worker_streams: dict[str, _StreamState] = {}
        self._fatal_error: BaseException | None = None
        self._closed = False
        self._started = False
        self._submitted = 0
        self._finished = 0
        self._aborted = 0
        self._emitted_tokens = 0
        self._preemptions = 0
        self._forced_prefills = 0
        self._allocation_blocked_steps = 0
        self._last_step_stats: StepStats | None = None

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        with self._state_lock:
            if self._closed:
                raise EngineClosedError("engine has been closed")
            if self._owner_loop is not None and self._owner_loop is not loop:
                raise RuntimeError("AsyncLLMEngine must be used from one event loop")
            self._owner_loop = loop
            if self._thread is None:
                self._ready_future = loop.create_future()
                self._thread = threading.Thread(
                    target=self._worker_main,
                    name="nanovllm-online-worker",
                    daemon=True,
                )
                self._thread.start()
            ready = self._ready_future
        if ready is not None:
            await asyncio.shield(ready)

    async def submit(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        request_id: str | None = None,
    ) -> AsyncRequestStream:
        await self.start()
        request_id = request_id or uuid4().hex
        loop = asyncio.get_running_loop()
        stream = _StreamState(
            request_id=request_id,
            loop=loop,
            queue=asyncio.Queue(maxsize=self.output_queue_size),
        )
        future = loop.create_future()
        self._enqueue_command(
            _Command(
                kind=_CommandKind.ADD,
                loop=loop,
                future=future,
                request_id=request_id,
                prompt=prompt,
                sampling_params=sampling_params,
                stream=stream,
            )
        )
        await future
        return AsyncRequestStream(self, stream)

    async def generate(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        request_id: str | None = None,
    ) -> AsyncIterator[RequestOutput]:
        stream = await self.submit(prompt, sampling_params, request_id)
        try:
            async for output in stream:
                yield output
        finally:
            await stream.aclose()

    async def abort(self, request_id: str) -> bool:
        await self.start()
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._enqueue_command(
            _Command(
                kind=_CommandKind.ABORT,
                loop=loop,
                future=future,
                request_id=request_id,
            )
        )
        return bool(await future)

    async def shutdown(self) -> None:
        with self._state_lock:
            thread = self._thread
            if self._closed and (thread is None or not thread.is_alive()):
                return
        if thread is None:
            with self._state_lock:
                self._closed = True
            return
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._enqueue_command(_Command(kind=_CommandKind.SHUTDOWN, loop=loop, future=future))
        await future
        await asyncio.to_thread(thread.join, 5.0)

    @property
    def is_alive(self) -> bool:
        thread = self._thread
        return bool(thread and thread.is_alive() and self._fatal_error is None)

    def metrics_snapshot(self) -> dict[str, object]:
        with self._state_lock:
            last_step = self._last_step_stats
            return {
                "started": self._started,
                "closed": self._closed,
                "worker_alive": self.is_alive,
                "scheduler_policy": self.engine_kwargs.get(
                    "scheduler_policy",
                    "prefill_first",
                ),
                "active_requests": len(self._worker_streams),
                "submitted_requests": self._submitted,
                "finished_requests": self._finished,
                "aborted_requests": self._aborted,
                "emitted_tokens": self._emitted_tokens,
                "preemptions": self._preemptions,
                "forced_prefills": self._forced_prefills,
                "allocation_blocked_steps": self._allocation_blocked_steps,
                "last_step": None
                if last_step is None
                else {
                    "step_id": last_step.step_id,
                    "batch_kind": last_step.batch_kind,
                    "batch_size": last_step.batch_size,
                    "scheduled_tokens": last_step.scheduled_tokens,
                    "waiting": last_step.waiting,
                    "running": last_step.running,
                    "kv_used_blocks": last_step.kv_used_blocks,
                    "kv_total_blocks": last_step.kv_total_blocks,
                    "decode_streak": last_step.decode_streak,
                    "elapsed_ms": last_step.elapsed_ms,
                },
            }

    def _enqueue_command(self, command: _Command) -> None:
        with self._state_lock:
            if self._closed or self._fatal_error is not None:
                error = self._fatal_error or EngineClosedError("engine has been closed")
                if command.future is not None:
                    command.loop.call_soon(command.future.set_exception, error)
                return
            self._commands.put(command)
        with self._condition:
            self._condition.notify()

    def _worker_main(self) -> None:
        engine: "LLMEngine" | None = None
        shutdown_acks: list[_Command] = []
        try:
            engine = self._engine_factory(self.model, **self.engine_kwargs)
            with self._state_lock:
                self._started = True
            self._resolve_ready(None)
            should_stop = False
            while not should_stop:
                should_stop, new_shutdown_acks = self._drain_commands(engine)
                shutdown_acks.extend(new_shutdown_acks)
                if should_stop:
                    self._fail_all_streams(EngineClosedError("engine is shutting down"))
                    break
                has_pending = bool(getattr(engine, "has_pending_outputs", False))
                if not engine.is_finished() or has_pending:
                    result = engine.step_stream()
                    for output in result.outputs:
                        self._publish(output)
                    if result.stats is not None:
                        with self._state_lock:
                            self._last_step_stats = result.stats
                            self._preemptions += result.stats.preemptions
                            self._forced_prefills += int(result.stats.forced_prefill)
                            self._allocation_blocked_steps += int(
                                result.stats.allocation_blocked
                            )
                    if result.stats is not None and result.stats.batch_kind == "idle" and not result.outputs:
                        with self._condition:
                            self._condition.wait(timeout=0.001)
                    continue
                with self._condition:
                    if self._commands.empty():
                        self._condition.wait()
        except BaseException as exc:
            with self._state_lock:
                self._fatal_error = exc
            self._resolve_ready(exc)
            self._fail_all_streams(exc)
        finally:
            if engine is not None:
                try:
                    engine.exit()
                except BaseException as exc:
                    with self._state_lock:
                        first_exit_error = self._fatal_error is None
                        if first_exit_error:
                            self._fatal_error = exc
                    if first_exit_error:
                        self._fail_all_streams(exc)
            with self._state_lock:
                self._closed = True
            terminal_error = self._fatal_error or EngineClosedError("engine has been closed")
            self._fail_pending_commands(terminal_error)
            for command in shutdown_acks:
                self._resolve_future(command, self._fatal_error, value=None)

    def _drain_commands(self, engine: "LLMEngine") -> tuple[bool, list[_Command]]:
        should_stop = False
        shutdown_acks = []
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                break
            if command.kind == _CommandKind.SHUTDOWN:
                should_stop = True
                shutdown_acks.append(command)
                break
            if command.kind == _CommandKind.ADD:
                try:
                    assert command.request_id is not None
                    assert command.prompt is not None
                    assert command.sampling_params is not None
                    assert command.stream is not None
                    actual_id = engine.add_request(
                        command.prompt,
                        command.sampling_params,
                        request_id=command.request_id,
                    )
                    if actual_id != command.request_id:
                        raise RuntimeError(
                            f"engine returned request ID {actual_id!r}; expected "
                            f"{command.request_id!r}"
                        )
                    with self._state_lock:
                        self._worker_streams[actual_id] = command.stream
                        self._submitted += 1
                    self._resolve_future(command, None, value=actual_id)
                except BaseException as exc:
                    self._resolve_future(command, exc)
                continue
            if command.kind == _CommandKind.ABORT:
                assert command.request_id is not None
                try:
                    aborted = engine.abort_request(command.request_id)
                    self._resolve_future(command, None, value=aborted)
                except BaseException as exc:
                    self._resolve_future(command, exc)
        return should_stop, shutdown_acks

    def _publish(self, output: RequestOutput) -> None:
        with self._state_lock:
            state = self._worker_streams.get(output.request_id)
            if state is None:
                return
            if output.token_id is not None:
                self._emitted_tokens += 1
            if output.finished:
                self._worker_streams.pop(output.request_id, None)
                if output.finish_reason == "abort":
                    self._aborted += 1
                else:
                    self._finished += 1
        state.loop.call_soon_threadsafe(self._deliver_to_stream, state, output)

    def _deliver_to_stream(self, state: _StreamState, item: object) -> None:
        if state.overflowed:
            return
        try:
            state.queue.put_nowait(item)
        except asyncio.QueueFull:
            state.overflowed = True
            while not state.queue.empty():
                try:
                    state.queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            state.queue.put_nowait(
                RequestQueueFullError(
                    f"consumer for request {state.request_id!r} could not keep up"
                )
            )
            self._enqueue_command(
                _Command(
                    kind=_CommandKind.ABORT,
                    loop=state.loop,
                    request_id=state.request_id,
                )
            )

    def _fail_pending_commands(self, exc: BaseException) -> None:
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return
            self._resolve_future(command, exc)

    def _fail_all_streams(self, exc: BaseException) -> None:
        with self._state_lock:
            streams = list(self._worker_streams.values())
            self._worker_streams.clear()
        for state in streams:
            state.loop.call_soon_threadsafe(self._deliver_to_stream, state, exc)

    def _resolve_ready(self, exc: BaseException | None) -> None:
        future = self._ready_future
        loop = self._owner_loop
        if future is None or loop is None:
            return
        if exc is None:
            loop.call_soon_threadsafe(self._set_future_result, future, None)
        else:
            loop.call_soon_threadsafe(self._set_future_exception, future, exc)

    @staticmethod
    def _resolve_future(
        command: _Command,
        exc: BaseException | None,
        *,
        value: object = None,
    ) -> None:
        if command.future is None:
            return
        if exc is None:
            command.loop.call_soon_threadsafe(
                AsyncLLMEngine._set_future_result,
                command.future,
                value,
            )
        else:
            command.loop.call_soon_threadsafe(
                AsyncLLMEngine._set_future_exception,
                command.future,
                exc,
            )

    @staticmethod
    def _set_future_result(future: asyncio.Future, value: object) -> None:
        if not future.done():
            future.set_result(value)

    @staticmethod
    def _set_future_exception(future: asyncio.Future, exc: BaseException) -> None:
        if not future.done():
            future.set_exception(exc)
