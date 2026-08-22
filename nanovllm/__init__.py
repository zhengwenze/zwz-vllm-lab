"""Public API with lazy CUDA imports for CPU-only tooling and tests."""

from typing import TYPE_CHECKING

from nanovllm.sampling_params import SamplingParams

if TYPE_CHECKING:
    from nanovllm.engine.async_llm_engine import AsyncLLMEngine
    from nanovllm.llm import LLM

__all__ = ["AsyncLLMEngine", "LLM", "SamplingParams"]


def __getattr__(name: str):
    if name == "LLM":
        from nanovllm.llm import LLM

        return LLM
    if name == "AsyncLLMEngine":
        from nanovllm.engine.async_llm_engine import AsyncLLMEngine

        return AsyncLLMEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
