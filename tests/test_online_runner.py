from benchmarks.online_scheduler.runner import OnlineBenchmarkRunner
from benchmarks.online_scheduler.workload import RequestSpec, WorkloadTrace, make_fixed_trace
from nanovllm.engine.outputs import RequestOutput, StepResult, StepStats


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000_000_000

    def __call__(self) -> int:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += round(seconds * 1_000_000_000)


class FakeEngine:
    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.active: list[str] = []
        self.step_id = 0

    def add_request(self, prompt, sampling_params, request_id=None):
        self.active.append(request_id)
        return request_id

    def is_finished(self):
        return not self.active

    def step_stream(self):
        self.clock.now += 10_000_000
        request_id = self.active.pop(0)
        self.step_id += 1
        output = RequestOutput(request_id, self.step_id, 7, (7,), "", True, "length", self.clock.now)
        stats = StepStats(self.step_id, "decode", 1, 1, 0, len(self.active), 1, 10, 0, False, False, 10.0)
        return StepResult((output,), stats)


def test_runner_counts_queueing_from_ideal_arrival() -> None:
    clock = FakeClock()
    trace = make_fixed_trace(
        num_requests=2,
        interval_seconds=0.005,
        prompt_tokens=2,
        output_tokens=1,
        seed=1,
    )
    runner = OnlineBenchmarkRunner(
        FakeEngine(clock),
        trace,
        lambda request: {"max_tokens": request.output_tokens},
        clock_ns=clock,
        sleep=clock.sleep,
    )

    result = runner.run()

    first = result.collector.requests["request-000000"]
    second = result.collector.requests["request-000001"]
    assert first.admitted_ns == result.start_ns
    assert second.arrival_ns == result.start_ns + 5_000_000
    # The second arrival occurs during a blocking 10ms engine step and is
    # admitted afterwards; its latency still starts at the 5ms ideal arrival.
    assert second.admitted_ns == result.start_ns + 10_000_000
    assert second.first_token_ns - second.arrival_ns == 15_000_000
    assert len(result.collector.steps) == 2


def test_barrier_arrivals_start_only_after_initial_requests_emit_first_token() -> None:
    clock = FakeClock()
    trace = WorkloadTrace(
        name="barrier",
        seed=1,
        requests=(
            RequestSpec("decode-0", 0, (1, 2), 1, "decode-stage"),
            RequestSpec(
                "prefill-0",
                0,
                (3, 4),
                1,
                "injected-prefill",
                arrival_anchor="barrier",
            ),
        ),
    )
    runner = OnlineBenchmarkRunner(
        FakeEngine(clock),
        trace,
        lambda request: {"max_tokens": request.output_tokens},
        clock_ns=clock,
        sleep=clock.sleep,
    )

    result = runner.run()

    initial = result.collector.requests["decode-0"]
    injected = result.collector.requests["prefill-0"]
    assert result.barrier_ns == initial.first_token_ns
    assert injected.arrival_ns == result.barrier_ns
    assert injected.admitted_ns == result.barrier_ns
