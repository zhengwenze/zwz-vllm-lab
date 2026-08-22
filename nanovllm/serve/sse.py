import argparse
import json
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import TYPE_CHECKING

from nanovllm.engine.async_llm_engine import AsyncLLMEngine
from nanovllm.engine.errors import (
    DuplicateRequestError,
    EngineClosedError,
    RequestQueueFullError,
    RequestTooLongError,
)
from nanovllm.sampling_params import SamplingParams

if TYPE_CHECKING:
    from fastapi import FastAPI


def create_app(engine: AsyncLLMEngine) -> "FastAPI":
    """Create a thin FastAPI/SSE adapter around ``AsyncLLMEngine``."""

    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise RuntimeError(
            "online serving dependencies are missing; install nano-vllm[online]"
        ) from exc

    class GenerateRequest(BaseModel):
        prompt: str | list[int]
        request_id: str | None = None
        temperature: float = Field(default=0.6, gt=1e-10)
        max_tokens: int = Field(default=128, ge=1)
        ignore_eos: bool = False

    @asynccontextmanager
    async def lifespan(_app):
        await engine.start()
        try:
            yield
        finally:
            await engine.shutdown()

    app = FastAPI(
        title="Nano-vLLM Online Scheduler",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health() -> dict[str, object]:
        snapshot = engine.metrics_snapshot()
        return {
            "status": "ok" if snapshot["worker_alive"] else "unavailable",
            **snapshot,
        }

    @app.get("/metrics")
    async def metrics() -> dict[str, object]:
        return engine.metrics_snapshot()

    @app.delete("/requests/{request_id}")
    async def abort_request(request_id: str) -> dict[str, object]:
        aborted = await engine.abort(request_id)
        if not aborted:
            raise HTTPException(status_code=404, detail="active request not found")
        return {"request_id": request_id, "aborted": True}

    @app.post("/generate")
    async def generate(body: GenerateRequest):
        params = SamplingParams(
            temperature=body.temperature,
            max_tokens=body.max_tokens,
            ignore_eos=body.ignore_eos,
        )
        try:
            stream = await engine.submit(body.prompt, params, body.request_id)
        except DuplicateRequestError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except RequestQueueFullError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except (RequestTooLongError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except EngineClosedError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        async def event_source():
            try:
                async for output in stream:
                    event = "done" if output.finished else "token"
                    payload = json.dumps(asdict(output), ensure_ascii=False)
                    yield f"event: {event}\ndata: {payload}\n\n"
            finally:
                await stream.aclose()

        return StreamingResponse(
            event_source(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve nano-vLLM over SSE")
    parser.add_argument("--model", required=True, help="Local Hugging Face model directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--scheduler-policy",
        choices=("prefill_first", "decode_first", "bounded_decode_first"),
        default="bounded_decode_first",
    )
    parser.add_argument("--max-consecutive-decode-steps", type=int, default=8)
    parser.add_argument("--max-queue-size", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "online serving dependencies are missing; install nano-vllm[online]"
        ) from exc

    engine = AsyncLLMEngine(
        args.model,
        scheduler_policy=args.scheduler_policy,
        max_consecutive_decode_steps=args.max_consecutive_decode_steps,
        max_queue_size=args.max_queue_size,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )
    uvicorn.run(create_app(engine), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
