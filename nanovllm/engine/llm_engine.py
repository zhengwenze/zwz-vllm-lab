import atexit
from collections import deque
from dataclasses import fields
from time import perf_counter, perf_counter_ns
from uuid import uuid4

from tqdm.auto import tqdm

from nanovllm.engine.errors import (
    DuplicateRequestError,
    RequestQueueFullError,
    RequestTooLongError,
)
from nanovllm.engine.outputs import RequestOutput, StepResult, StepStats
from nanovllm.engine.scheduler import (
    BatchKind,
    Scheduler,
    SchedulerOutput,
    SchedulerQueueFullError,
)
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class LLMEngine:
    """Synchronous inference engine with offline and online stepping APIs."""

    def __init__(self, model: str, **kwargs):
        # Heavy CUDA dependencies stay local so scheduler/metrics tests can
        # import the package on a CPU-only development host.
        import torch.multiprocessing as mp
        from transformers import AutoTokenizer

        from nanovllm.config import Config
        from nanovllm.engine.model_runner import ModelRunner

        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {key: value for key, value in kwargs.items() if key in config_fields}
        self.config: Config = Config(model, **config_kwargs)
        Sequence.block_size = self.config.kvcache_block_size
        self.ps = []
        self.events = []
        self._closed = False
        self._step_id = 0
        self._request_to_seq: dict[str, Sequence] = {}
        self._seq_to_request: dict[int, str] = {}
        self._known_request_ids: set[str] = set()
        self._pending_outputs: deque[RequestOutput] = deque()

        ctx = mp.get_context("spawn")
        for rank in range(1, self.config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(self.config, rank, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(self.config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model, use_fast=True)
        self.config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(self.config)
        atexit.register(self.exit)

    def exit(self) -> None:
        if self._closed:
            return
        self._closed = True
        model_runner = getattr(self, "model_runner", None)
        if model_runner is not None:
            model_runner.call("exit")
            del self.model_runner
        for process in getattr(self, "ps", []):
            process.join()

    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        request_id: str | None = None,
    ) -> str:
        """Validate and admit one request, returning its stable external ID."""

        if self._closed:
            raise RuntimeError("engine has been closed")
        request_id = request_id or uuid4().hex
        if not request_id:
            raise ValueError("request_id must not be empty")
        if request_id in self._known_request_ids:
            raise DuplicateRequestError(f"request_id already exists: {request_id}")
        if not isinstance(sampling_params, SamplingParams):
            raise TypeError("sampling_params must be a SamplingParams instance")
        if sampling_params.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        token_ids = self.tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
        if not token_ids:
            raise ValueError("prompt must contain at least one token")
        if any(not isinstance(token_id, int) or token_id < 0 for token_id in token_ids):
            raise ValueError("prompt token IDs must be non-negative integers")

        total_tokens = len(token_ids) + sampling_params.max_tokens
        if total_tokens > self.config.max_model_len:
            raise RequestTooLongError(
                f"request needs {total_tokens} tokens but max_model_len is "
                f"{self.config.max_model_len}"
            )
        required_blocks = (
            total_tokens + self.config.kvcache_block_size - 1
        ) // self.config.kvcache_block_size
        if required_blocks > self.config.num_kvcache_blocks:
            raise RequestTooLongError(
                f"request needs {required_blocks} KV blocks but only "
                f"{self.config.num_kvcache_blocks} are available"
            )

        seq = Sequence(token_ids, sampling_params)
        try:
            self.scheduler.add(seq)
        except SchedulerQueueFullError as exc:
            raise RequestQueueFullError(str(exc)) from exc
        self._known_request_ids.add(request_id)
        self._request_to_seq[request_id] = seq
        self._seq_to_request[seq.seq_id] = request_id
        return request_id

    def abort_request(self, request_id: str) -> bool:
        """Abort an active request and defer its terminal event to step_stream."""

        seq = self._request_to_seq.get(request_id)
        if seq is None:
            return False
        aborted = self.scheduler.abort(seq.seq_id)
        if aborted is None:
            return False
        self._pending_outputs.append(self._make_output(aborted, token_id=None))
        self._forget_active_request(request_id, aborted.seq_id)
        return True

    @property
    def has_pending_outputs(self) -> bool:
        return bool(self._pending_outputs)

    def step_stream(self) -> StepResult:
        """Run one phase and return cumulative token events plus step metrics."""

        started = perf_counter()
        waiting_before = len(self.scheduler.waiting)
        running_before = len(self.scheduler.running)
        pending = list(self._drain_pending_outputs())
        if self.scheduler.is_finished():
            scheduler_output = SchedulerOutput([], BatchKind.IDLE)
            return StepResult(
                tuple(pending),
                self._make_stats(
                    scheduler_output,
                    started,
                    waiting_before=waiting_before,
                    running_before=running_before,
                ),
            )

        scheduler_output = self.scheduler.schedule()
        scheduled_request_ids = tuple(
            self._seq_to_request[seq.seq_id] for seq in scheduler_output.seqs
        )
        preempted_request_ids = tuple(
            self._seq_to_request[seq_id]
            for seq_id in scheduler_output.preempted_seq_ids
            if seq_id in self._seq_to_request
        )
        streamed = pending
        if scheduler_output.seqs:
            before_counts = {
                seq.seq_id: seq.num_completion_tokens for seq in scheduler_output.seqs
            }
            token_ids = self.model_runner.call(
                "run",
                scheduler_output.seqs,
                scheduler_output.is_prefill,
            )
            self.scheduler.postprocess(
                scheduler_output.seqs,
                token_ids,
                scheduler_output.is_prefill,
            )
            for seq in scheduler_output.seqs:
                if seq.num_completion_tokens == before_counts[seq.seq_id]:
                    continue
                token_id = seq.completion_token_ids[-1]
                output = self._make_output(seq, token_id=token_id)
                streamed.append(output)
                if seq.is_finished:
                    request_id = self._seq_to_request.get(seq.seq_id)
                    if request_id is not None:
                        self._forget_active_request(request_id, seq.seq_id)

        return StepResult(
            tuple(streamed),
            self._make_stats(
                scheduler_output,
                started,
                scheduled_request_ids=scheduled_request_ids,
                preempted_request_ids=preempted_request_ids,
                waiting_before=waiting_before,
                running_before=running_before,
            ),
        )

    def step(self) -> tuple[list[tuple[int, list[int]]], int]:
        """Historical offline step API retained for existing callers."""

        result = self.step_stream()
        outputs = [
            (output.sequence_id, list(output.token_ids))
            for output in result.outputs
            if output.finished and output.finish_reason != "abort"
        ]
        stats = result.stats
        if stats is None or stats.batch_kind == BatchKind.IDLE.value:
            num_tokens = 0
        elif stats.batch_kind == BatchKind.PREFILL.value:
            num_tokens = stats.scheduled_tokens
        else:
            num_tokens = -stats.scheduled_tokens
        return outputs, num_tokens

    def is_finished(self) -> bool:
        return self.scheduler.is_finished() and not self._pending_outputs

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict[str, object]]:
        pbar = tqdm(
            total=len(prompts),
            desc="Generating",
            dynamic_ncols=True,
            disable=not use_tqdm,
        )
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        if len(prompts) != len(sampling_params):
            raise ValueError("prompts and sampling_params must have the same length")
        for prompt, params in zip(prompts, sampling_params):
            self.add_request(prompt, params)
        outputs: dict[int, list[int]] = {}
        prefill_throughput = decode_throughput = 0.0
        while not self.is_finished():
            started = perf_counter()
            output, num_tokens = self.step()
            elapsed = perf_counter() - started
            if num_tokens > 0:
                prefill_throughput = num_tokens / elapsed
            elif num_tokens < 0:
                decode_throughput = -num_tokens / elapsed
            pbar.set_postfix(
                {
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                }
            )
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        ordered = [outputs[seq_id] for seq_id in sorted(outputs)]
        return [
            {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
            for token_ids in ordered
        ]

    def _make_output(self, seq: Sequence, token_id: int | None) -> RequestOutput:
        request_id = self._seq_to_request[seq.seq_id]
        token_ids = tuple(seq.completion_token_ids)
        return RequestOutput(
            request_id=request_id,
            sequence_id=seq.seq_id,
            token_id=token_id,
            token_ids=token_ids,
            text=self.tokenizer.decode(token_ids),
            finished=seq.is_finished,
            finish_reason=seq.finish_reason,
            timestamp_ns=perf_counter_ns(),
        )

    def _make_stats(
        self,
        output: SchedulerOutput,
        started: float,
        *,
        scheduled_request_ids: tuple[str, ...] = (),
        preempted_request_ids: tuple[str, ...] = (),
        waiting_before: int = 0,
        running_before: int = 0,
    ) -> StepStats:
        manager = self.scheduler.block_manager
        stats = StepStats(
            step_id=self._step_id,
            batch_kind=output.batch_kind.value,
            batch_size=len(output.seqs),
            scheduled_tokens=output.num_scheduled_tokens,
            waiting=len(self.scheduler.waiting),
            running=len(self.scheduler.running),
            kv_used_blocks=len(manager.used_block_ids),
            kv_total_blocks=len(manager.blocks),
            preemptions=len(output.preempted_seq_ids),
            forced_prefill=output.forced_prefill,
            allocation_blocked=output.allocation_blocked,
            elapsed_ms=(perf_counter() - started) * 1000,
            scheduled_request_ids=scheduled_request_ids,
            preempted_request_ids=preempted_request_ids,
            waiting_request_ids=tuple(
                self._seq_to_request[seq.seq_id]
                for seq in self.scheduler.waiting
                if seq.seq_id in self._seq_to_request
            ),
            running_request_ids=tuple(
                self._seq_to_request[seq.seq_id]
                for seq in self.scheduler.running
                if seq.seq_id in self._seq_to_request
            ),
            waiting_before=waiting_before,
            running_before=running_before,
            decode_streak=self.scheduler.consecutive_decode_steps,
        )
        self._step_id += 1
        return stats

    def _drain_pending_outputs(self):
        while self._pending_outputs:
            yield self._pending_outputs.popleft()

    def _forget_active_request(self, request_id: str, seq_id: int) -> None:
        self._request_to_seq.pop(request_id, None)
        self._seq_to_request.pop(seq_id, None)
