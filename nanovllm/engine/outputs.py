from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class RequestOutput:
    """One cumulative, user-visible generation event."""

    request_id: str
    sequence_id: int
    token_id: int | None
    token_ids: tuple[int, ...]
    text: str
    finished: bool
    finish_reason: str | None
    timestamp_ns: int


@dataclass(frozen=True, slots=True)
class StepStats:
    """Scheduler and KV-cache state captured after one engine step."""

    step_id: int
    batch_kind: str
    batch_size: int
    scheduled_tokens: int
    waiting: int
    running: int
    kv_used_blocks: int
    kv_total_blocks: int
    preemptions: int
    forced_prefill: bool
    allocation_blocked: bool
    elapsed_ms: float
    scheduled_request_ids: tuple[str, ...] = field(default_factory=tuple)
    preempted_request_ids: tuple[str, ...] = field(default_factory=tuple)
    waiting_request_ids: tuple[str, ...] = field(default_factory=tuple)
    running_request_ids: tuple[str, ...] = field(default_factory=tuple)
    waiting_before: int = 0
    running_before: int = 0
    decode_streak: int = 0


@dataclass(frozen=True, slots=True)
class StepResult:
    """Streaming outputs and observability data produced by one engine step."""

    outputs: tuple[RequestOutput, ...] = field(default_factory=tuple)
    stats: StepStats | None = None
