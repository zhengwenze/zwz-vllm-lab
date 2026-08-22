from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Iterator

from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager

if TYPE_CHECKING:
    from nanovllm.config import Config


class BatchKind(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    IDLE = "idle"


@dataclass(slots=True)
class SchedulerOutput:
    seqs: list[Sequence]
    batch_kind: BatchKind
    num_scheduled_tokens: int = 0
    forced_prefill: bool = False
    preempted_seq_ids: tuple[int, ...] = ()
    allocation_blocked: bool = False

    @property
    def is_prefill(self) -> bool:
        return self.batch_kind == BatchKind.PREFILL

    def __iter__(self) -> Iterator:
        """Keep the historical ``seqs, is_prefill = schedule()`` contract."""
        yield self.seqs
        yield self.is_prefill


class SchedulerQueueFullError(RuntimeError):
    pass


class Scheduler:

    def __init__(self, config: "Config"):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.policy = config.scheduler_policy
        self.max_consecutive_decode_steps = config.max_consecutive_decode_steps
        self.max_queue_size = config.max_queue_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.consecutive_decode_steps = 0
        self.total_preemptions = 0
        self.forced_prefill_count = 0

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        active_requests = len(self.waiting) + len(self.running)
        if active_requests >= self.max_queue_size:
            raise SchedulerQueueFullError(
                f"scheduler request capacity is full ({self.max_queue_size})"
            )
        self.waiting.append(seq)

    @property
    def resident_count(self) -> int:
        partial_prefills = sum(bool(seq.block_table) for seq in self.waiting)
        return len(self.running) + partial_prefills

    def schedule(self) -> SchedulerOutput:
        if not self.waiting:
            # Decode-only work before a request arrives must not consume the new
            # request's bounded waiting allowance.
            self.consecutive_decode_steps = 0

        if not self.waiting and not self.running:
            return SchedulerOutput([], BatchKind.IDLE)

        if self.waiting and self.running:
            if self.policy == "prefill_first":
                output = self._schedule_prefill()
                if output.seqs:
                    return self._record(output)
                fallback = self._schedule_decode()
                fallback.allocation_blocked |= output.allocation_blocked
                return self._record(fallback)
            if self.policy == "decode_first":
                return self._record(self._schedule_decode())
            if self.consecutive_decode_steps >= self.max_consecutive_decode_steps:
                output = self._schedule_prefill(forced=True)
                if output.seqs:
                    return self._record(output)
                fallback = self._schedule_decode()
                fallback.allocation_blocked |= output.allocation_blocked
                return self._record(fallback)
            return self._record(self._schedule_decode())

        if self.waiting:
            return self._record(self._schedule_prefill())
        return self._record(self._schedule_decode())

    def _record(self, output: SchedulerOutput) -> SchedulerOutput:
        if output.batch_kind == BatchKind.PREFILL:
            self.consecutive_decode_steps = 0
            if output.forced_prefill:
                self.forced_prefill_count += 1
        elif output.batch_kind == BatchKind.DECODE:
            if self.waiting:
                self.consecutive_decode_steps += 1
            else:
                self.consecutive_decode_steps = 0
        return output

    def _schedule_prefill(self, forced: bool = False) -> SchedulerOutput:
        scheduled_seqs = []
        num_batched_tokens = 0
        allocation_blocked = False

        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                if self.resident_count >= self.max_num_seqs:
                    allocation_blocked = True
                    break
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    allocation_blocked = True
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        return SchedulerOutput(
            scheduled_seqs,
            BatchKind.PREFILL if scheduled_seqs else BatchKind.IDLE,
            num_scheduled_tokens=num_batched_tokens,
            forced_prefill=forced and bool(scheduled_seqs),
            allocation_blocked=allocation_blocked,
        )

    def _schedule_decode(self) -> SchedulerOutput:
        scheduled_seqs = []
        preempted_seq_ids = []
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    victim = self.running.pop()
                    preempted_seq_ids.append(victim.seq_id)
                    self.preempt(victim)
                else:
                    preempted_seq_ids.append(seq.seq_id)
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        self.running.extendleft(reversed(scheduled_seqs))
        return SchedulerOutput(
            scheduled_seqs,
            BatchKind.DECODE if scheduled_seqs else BatchKind.IDLE,
            num_scheduled_tokens=len(scheduled_seqs),
            preempted_seq_ids=tuple(preempted_seq_ids),
            allocation_blocked=bool(preempted_seq_ids),
        )

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        seq.num_scheduled_tokens = 0
        seq.num_preemptions += 1
        self.total_preemptions += 1
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def abort(self, seq_id: int) -> Sequence | None:
        seq = next((seq for seq in self.waiting if seq.seq_id == seq_id), None)
        queue = self.waiting
        if seq is None:
            seq = next((seq for seq in self.running if seq.seq_id == seq_id), None)
            queue = self.running
        if seq is None:
            return None
        queue.remove(seq)
        if seq.block_table:
            self.block_manager.deallocate(seq)
        seq.num_scheduled_tokens = 0
        seq.status = SequenceStatus.ABORTED
        seq.finish_reason = "abort"
        return seq

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                seq.finish_reason = "stop" if token_id == self.eos and not seq.ignore_eos else "length"
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
