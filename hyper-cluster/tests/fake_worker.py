"""A real (in-process) worker that speaks the Hyper-Cluster worker protocol.

The tests spin this up with uvicorn on a random free port so the head node
talks to it over genuine HTTP + Server-Sent Events -- no mocking of the
transport layer, which means the failover, hold-and-requeue and streaming
paths are exercised for real.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


def free_port() -> int:
    """Find an unused TCP port on the loopback interface."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass
class WorkerState:
    """Mutable knobs of a fake worker (used to simulate slow/dead nodes)."""

    worker_id: str
    model_loaded: str = "Qwen2.5-7B-Instruct-GGUF"
    model_family: str = "qwen2"
    vram_total_gb: float = 24.0
    vram_free_gb: float = 20.0
    tokens_per_sec: float = 60.0
    delay_ms: float = 5.0
    fail: bool = False
    drain: bool = False
    inflight: int = 0
    requests_served: int = 0
    answer_prefix: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def _answer_for(state: WorkerState, prompt: str) -> list[str]:
    """Produce a deterministic, prompt-aware answer split into chunks."""
    marker = state.worker_id
    body = prompt.strip().replace("\n", " ")[:180]
    text = (
        f"[{marker}] {state.answer_prefix or 'answer'}: {body} "
        f"tokens={state.tokens_per_sec:.0f} model={state.model_loaded}"
    )
    words = text.split(" ")
    chunks: list[str] = []
    for index, word in enumerate(words):
        chunks.append(word + (" " if index < len(words) - 1 else ""))
    return chunks


def build_worker_app(state: WorkerState) -> FastAPI:
    """Create the FastAPI application implementing the worker protocol."""
    app = FastAPI(title=f"fake-worker-{state.worker_id}")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        if state.fail:
            return JSONResponse(status_code=503, content={"status": "down"})
        return {
            "status": "ok",
            "protocol_version": "1.0",
            "worker_id": state.worker_id,
            "model_loaded": state.model_loaded,
            "draining": state.drain,
            "vram_free_gb": state.vram_free_gb,
            "vram_total_gb": state.vram_total_gb,
        }

    @app.post("/v1/chat")
    async def chat(payload: dict[str, Any], request: Request) -> Any:
        if state.fail:
            return JSONResponse(status_code=503, content={"error": "worker is down"})
        if state.drain:
            return JSONResponse(status_code=503, content={"error": "worker is draining"})

        async def _stream() -> AsyncIterator[str]:
            started = time.perf_counter()
            prompt = "\n".join(str(m.get("content", "")) for m in payload.get("messages", []))
            state.inflight += 1
            try:
                for index, chunk in enumerate(_answer_for(state, prompt)):
                    await asyncio.sleep(state.delay_ms / 1000.0)
                    if request.client is not None and await request.is_disconnected():
                        break
                    frame = {
                        "request_id": payload.get("request_id", ""),
                        "text": chunk,
                        "index": index,
                        "source": "cluster",
                        "draft": False,
                    }
                    yield f"data: {json.dumps(frame)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                state.inflight = max(0, state.inflight - 1)
                state.requests_served += 1
                state.extra["last_total_ms"] = (time.perf_counter() - started) * 1000.0

        return StreamingResponse(_stream(), media_type="text/event-stream")

    @app.post("/v1/chat/sync")
    async def chat_sync(payload: dict[str, Any]) -> Any:
        if state.fail:
            return JSONResponse(status_code=503, content={"error": "worker is down"})
        started = time.perf_counter()
        prompt = "\n".join(str(m.get("content", "")) for m in payload.get("messages", []))
        chunks = _answer_for(state, prompt)
        total_ms = (time.perf_counter() - started) * 1000.0 + state.delay_ms * len(chunks)
        text = "".join(chunks)
        return {
            "request_id": payload.get("request_id", ""),
            "text": text,
            "model": state.model_loaded,
            "worker_id": state.worker_id,
            "tier": "cluster",
            "source": "cluster",
            "tokens": len(chunks),
            "ttft_ms": state.delay_ms,
            "total_ms": total_ms,
            "tokens_per_sec": state.tokens_per_sec,
            "finish_reason": "stop",
        }

    @app.post("/v1/model/load")
    async def load_model(payload: dict[str, Any]) -> Any:
        state.model_loaded = str(payload.get("model") or state.model_loaded)
        state.model_family = str(payload.get("family") or state.model_family)
        return {
            "worker_id": state.worker_id,
            "model": state.model_loaded,
            "loaded": True,
            "load_ms": 12.0,
            "vram_used_gb": 6.0,
            "error": None,
        }

    @app.post("/v1/admin/drain")
    async def drain(payload: dict[str, Any]) -> Any:
        state.drain = True
        return {"draining": True, "worker_id": state.worker_id}

    return app


class WorkerServer:
    """Runs a fake worker in a background thread on its own port."""

    def __init__(self, **kwargs: Any) -> None:
        # ``model`` is accepted as a friendlier alias of ``model_loaded``.
        if "model" in kwargs:
            kwargs["model_loaded"] = kwargs.pop("model")
        self.state = WorkerState(worker_id=kwargs.pop("worker_id", "wk-test"), **kwargs)
        self.port = free_port()
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def worker_id(self) -> str:
        return self.state.worker_id

    def start(self, timeout: float = 10.0) -> WorkerServer:
        """Start the server and block until it accepts connections."""
        config = uvicorn.Config(
            build_worker_app(self.state),
            host="127.0.0.1",
            port=self.port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)

        def _run() -> None:
            asyncio.run(self._server.serve())  # type: ignore[func-returns-value]

        self._thread = threading.Thread(target=_run, daemon=True, name=f"fake-worker-{self.port}")
        self._thread.start()
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.2):
                    self._ready.set()
                    return self
            except OSError:
                time.sleep(0.05)
        raise RuntimeError(f"fake worker {self.state.worker_id} did not start in {timeout}s")

    def stop(self) -> None:
        """Stop the server."""
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def __enter__(self) -> WorkerServer:
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()


__all__ = ["WorkerServer", "WorkerState", "build_worker_app", "free_port"]
