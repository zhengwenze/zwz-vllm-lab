from time import perf_counter_ns

from fastapi.testclient import TestClient

from nanovllm.engine.errors import DuplicateRequestError
from nanovllm.engine.outputs import RequestOutput
from nanovllm.serve.sse import create_app


class FakeStream:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.outputs)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def aclose(self):
        self.closed = True


class FakeAsyncEngine:
    def __init__(self):
        self.started = False
        self.closed = False
        self.streams = []

    async def start(self):
        self.started = True

    async def shutdown(self):
        self.closed = True

    async def submit(self, _prompt, _params, request_id=None):
        if request_id == "duplicate":
            raise DuplicateRequestError("request_id already exists: duplicate")
        request_id = request_id or "generated"
        stream = FakeStream(
            [
                RequestOutput(
                    request_id=request_id,
                    sequence_id=1,
                    token_id=101,
                    token_ids=(101,),
                    text="hello",
                    finished=False,
                    finish_reason=None,
                    timestamp_ns=perf_counter_ns(),
                ),
                RequestOutput(
                    request_id=request_id,
                    sequence_id=1,
                    token_id=102,
                    token_ids=(101, 102),
                    text="hello world",
                    finished=True,
                    finish_reason="length",
                    timestamp_ns=perf_counter_ns(),
                ),
            ]
        )
        self.streams.append(stream)
        return stream

    async def abort(self, request_id):
        return request_id == "active"

    def metrics_snapshot(self):
        return {
            "started": self.started,
            "closed": self.closed,
            "worker_alive": self.started and not self.closed,
            "active_requests": 0,
            "submitted_requests": len(self.streams),
            "finished_requests": len(self.streams),
            "aborted_requests": 0,
            "emitted_tokens": len(self.streams) * 2,
        }


def test_sse_generate_health_metrics_and_lifecycle():
    engine = FakeAsyncEngine()
    app = create_app(engine)

    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        response = client.post(
            "/generate",
            json={"prompt": "hello", "request_id": "req-sse", "max_tokens": 2},
        )
        metrics = client.get("/metrics").json()

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: token" in response.text
    assert "event: done" in response.text
    assert '"request_id": "req-sse"' in response.text
    assert metrics["submitted_requests"] == 1
    assert engine.streams[0].closed is True
    assert engine.closed is True


def test_sse_maps_duplicate_and_missing_abort_errors():
    app = create_app(FakeAsyncEngine())

    with TestClient(app) as client:
        duplicate = client.post(
            "/generate",
            json={"prompt": [1], "request_id": "duplicate"},
        )
        missing = client.delete("/requests/missing")
        active = client.delete("/requests/active")

    assert duplicate.status_code == 409
    assert missing.status_code == 404
    assert active.status_code == 200
