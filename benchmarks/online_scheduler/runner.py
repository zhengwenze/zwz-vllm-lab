"""Open-loop replay runner for the LLMEngine streaming-step protocol."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any, Callable, Protocol

from .metrics import MetricsCollector
from .workload import RequestSpec, WorkloadTrace


class EngineProtocol(Protocol):
    """The minimal online interface required from nano-vLLM."""

    def add_request(self, prompt: list[int], sampling_params: Any, request_id: str | None = None) -> str: ...

    def step_stream(self) -> Any: ...

    def is_finished(self) -> bool: ...


SamplingParamsFactory = Callable[[RequestSpec], Any]


@dataclass(frozen=True, slots=True)
class RunnerResult:
    collector: MetricsCollector
    start_ns: int
    finish_ns: int
    barrier_ns: int | None = None


class OnlineBenchmarkRunner:
    """Replay ideal arrivals without hiding queueing during blocking GPU steps."""

    def __init__(
        self,
        engine: EngineProtocol,
        trace: WorkloadTrace,
        sampling_params_factory: SamplingParamsFactory,
        *,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
        sleep: Callable[[float], None] = time.sleep,
        idle_poll_seconds: float = 0.001,
        max_drain_seconds: float = 300.0,
    ) -> None:
        if idle_poll_seconds <= 0:
            raise ValueError("idle_poll_seconds must be positive")
        if max_drain_seconds <= 0:
            raise ValueError("max_drain_seconds must be positive")
        self.engine = engine
        self.trace = trace
        self.sampling_params_factory = sampling_params_factory
        self.clock_ns = clock_ns
        self.sleep = sleep
        self.idle_poll_seconds = idle_poll_seconds
        self.max_drain_ns = round(max_drain_seconds * 1_000_000_000)

    def run(self) -> RunnerResult:
        collector = MetricsCollector()
        start_ns = self.clock_ns()
        start_requests = tuple(
            request for request in self.trace.requests if request.arrival_anchor == "start"
        )
        barrier_requests = tuple(
            request for request in self.trace.requests if request.arrival_anchor == "barrier"
        )
        for request in start_requests:
            collector.register_request(request, arrival_ns=start_ns + request.arrival_offset_ns)

        next_start_request = 0
        next_barrier_request = 0
        barrier_ns: int | None = None
        drain_deadline_ns: int | None = None
        while (
            next_start_request < len(start_requests)
            or barrier_ns is None and bool(barrier_requests)
            or next_barrier_request < len(barrier_requests)
            or not self.engine.is_finished()
        ):
            now_ns = self.clock_ns()
            while (
                next_start_request < len(start_requests)
                and start_ns + start_requests[next_start_request].arrival_offset_ns <= now_ns
            ):
                self._admit(collector, start_requests[next_start_request], now_ns)
                next_start_request += 1

            if (
                barrier_requests
                and barrier_ns is None
                and next_start_request == len(start_requests)
                and all(
                    collector.requests[request.request_id].first_token_ns is not None
                    for request in start_requests
                )
            ):
                barrier_ns = self.clock_ns()
                for request in barrier_requests:
                    collector.register_request(
                        request,
                        arrival_ns=barrier_ns + request.arrival_offset_ns,
                    )

            while (
                barrier_ns is not None
                and next_barrier_request < len(barrier_requests)
                and barrier_ns + barrier_requests[next_barrier_request].arrival_offset_ns <= now_ns
            ):
                self._admit(collector, barrier_requests[next_barrier_request], now_ns)
                next_barrier_request += 1

            all_admitted = (
                next_start_request == len(start_requests)
                and next_barrier_request == len(barrier_requests)
            )
            if all_admitted and drain_deadline_ns is None:
                drain_deadline_ns = now_ns + self.max_drain_ns

            if not self.engine.is_finished():
                step_started_ns = self.clock_ns()
                result = self.engine.step_stream()
                step_finished_ns = self.clock_ns()
                self._collect_step(collector, result, step_started_ns, step_finished_ns)
            else:
                next_arrival_ns = self._next_arrival_ns(
                    start_ns=start_ns,
                    start_requests=start_requests,
                    next_start_request=next_start_request,
                    barrier_ns=barrier_ns,
                    barrier_requests=barrier_requests,
                    next_barrier_request=next_barrier_request,
                )
                if next_arrival_ns is None:
                    continue
                delay_seconds = max(0, next_arrival_ns - self.clock_ns()) / 1_000_000_000
                self.sleep(min(delay_seconds, self.idle_poll_seconds))

            if drain_deadline_ns is not None and self.clock_ns() > drain_deadline_ns and not self.engine.is_finished():
                raise TimeoutError(
                    f"engine did not drain within {self.max_drain_ns / 1_000_000_000:.1f}s after final admission"
                )

        return RunnerResult(
            collector=collector,
            start_ns=start_ns,
            finish_ns=self.clock_ns(),
            barrier_ns=barrier_ns,
        )

    def _admit(
        self,
        collector: MetricsCollector,
        request: RequestSpec,
        admitted_ns: int,
    ) -> None:
        returned_id = self.engine.add_request(
            list(request.prompt_token_ids),
            self.sampling_params_factory(request),
            request_id=request.request_id,
        )
        if returned_id != request.request_id:
            raise RuntimeError(
                f"engine returned request ID {returned_id!r}; expected {request.request_id!r}"
            )
        collector.mark_admitted(request.request_id, admitted_ns=admitted_ns)

    @staticmethod
    def _next_arrival_ns(
        *,
        start_ns: int,
        start_requests: tuple[RequestSpec, ...],
        next_start_request: int,
        barrier_ns: int | None,
        barrier_requests: tuple[RequestSpec, ...],
        next_barrier_request: int,
    ) -> int | None:
        if next_start_request < len(start_requests):
            return start_ns + start_requests[next_start_request].arrival_offset_ns
        if barrier_ns is not None and next_barrier_request < len(barrier_requests):
            return barrier_ns + barrier_requests[next_barrier_request].arrival_offset_ns
        return None

    @staticmethod
    def _collect_step(
        collector: MetricsCollector,
        result: Any,
        step_started_ns: int,
        step_finished_ns: int,
    ) -> None:
        try:
            outputs = result.outputs
            stats = result.stats
        except AttributeError as exc:
            raise TypeError("step_stream() must return an object with outputs and stats attributes") from exc

        for output in outputs:
            request_id = str(output.request_id)
            emitted_ns = int(output.timestamp_ns)
            token_id = output.token_id
            finished = bool(output.finished)
            if token_id is not None:
                collector.record_token(
                    request_id,
                    token_id=int(token_id),
                    emitted_ns=emitted_ns,
                    finished=finished,
                )
            elif finished:
                collector.mark_finished(request_id, finish_ns=emitted_ns)

        if stats is None:
            raise TypeError("step_stream() returned no StepStats for an active engine step")

        known_fields = (
            "step_id",
            "batch_kind",
            "batch_size",
            "scheduled_tokens",
            "waiting",
            "running",
            "kv_used_blocks",
            "kv_total_blocks",
            "preemptions",
            "forced_prefill",
            "allocation_blocked",
            "elapsed_ms",
            "scheduled_request_ids",
            "preempted_request_ids",
            "waiting_request_ids",
            "running_request_ids",
            "waiting_before",
            "running_before",
            "decode_streak",
        )
        step = {name: getattr(stats, name) for name in known_fields}
        step.update({"runner_start_ns": step_started_ns, "runner_finish_ns": step_finished_ns})
        collector.add_step(step)


class GpuTelemetry:
    """Best-effort nvidia-smi telemetry that never blocks benchmark execution."""

    QUERY_FIELDS = (
        "timestamp",
        "name",
        "uuid",
        "utilization.gpu",
        "memory.used",
        "memory.total",
        "temperature.gpu",
        "power.draw",
        "clocks.sm",
    )

    def __init__(self, output_path: str | Path, *, interval_seconds: float = 0.2) -> None:
        if interval_seconds <= 0:
            raise ValueError("telemetry interval must be positive")
        self.output_path = Path(output_path)
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples = 0
        self.error: str | None = None

    def start(self) -> bool:
        if self._thread is not None:
            raise RuntimeError("telemetry has already been started")
        try:
            self._query()
        except (FileNotFoundError, subprocess.SubprocessError, OSError) as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return False
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(("sampled_perf_counter_ns", *self.QUERY_FIELDS))
        self._thread = threading.Thread(target=self._loop, name="gpu-telemetry", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 3))

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self._thread is not None,
            "interval_seconds": self.interval_seconds,
            "samples": self.samples,
            "error": self.error,
        }

    def _loop(self) -> None:
        while not self._stop.is_set():
            sampled_ns = time.perf_counter_ns()
            try:
                rows = self._query()
                with self.output_path.open("a", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    for row in rows:
                        writer.writerow((sampled_ns, *row))
                        self.samples += 1
            except (FileNotFoundError, subprocess.SubprocessError, OSError) as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                return
            self._stop.wait(self.interval_seconds)

    def _query(self) -> list[list[str]]:
        command = [
            "nvidia-smi",
            f"--query-gpu={','.join(self.QUERY_FIELDS)}",
            "--format=csv,noheader,nounits",
        ]
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=3)
        return [next(csv.reader([line])) for line in completed.stdout.splitlines() if line.strip()]


def runtime_manifest() -> dict[str, Any]:
    """Collect reproducibility metadata without importing CUDA libraries."""

    def command_output(command: list[str]) -> str | None:
        try:
            return subprocess.run(command, check=True, capture_output=True, text=True, timeout=5).stdout.strip()
        except (FileNotFoundError, subprocess.SubprocessError, OSError):
            return None

    git_status = command_output(["git", "status", "--porcelain"])
    return {
        "pid": os.getpid(),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(git_status),
        "git_status_porcelain": git_status,
        "nvidia_smi": command_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,power.limit", "--format=csv,noheader,nounits"]),
    }
