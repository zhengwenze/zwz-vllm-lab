import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    scheduler_policy: str = "prefill_first"
    max_consecutive_decode_steps: int = 8
    max_queue_size: int = 256

    def __post_init__(self):
        if not os.path.isdir(self.model):
            raise ValueError(f"model path does not exist: {self.model}")
        if self.max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens must be positive")
        if self.max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be positive")
        if self.max_model_len <= 0:
            raise ValueError("max_model_len must be positive")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if self.kvcache_block_size <= 0 or self.kvcache_block_size % 256 != 0:
            raise ValueError("kvcache_block_size must be a positive multiple of 256")
        if not 1 <= self.tensor_parallel_size <= 8:
            raise ValueError("tensor_parallel_size must be between 1 and 8")
        if self.scheduler_policy not in {
            "prefill_first",
            "decode_first",
            "bounded_decode_first",
        }:
            raise ValueError(f"unsupported scheduler_policy: {self.scheduler_policy}")
        if self.max_consecutive_decode_steps <= 0:
            raise ValueError("max_consecutive_decode_steps must be positive")
        if self.max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
