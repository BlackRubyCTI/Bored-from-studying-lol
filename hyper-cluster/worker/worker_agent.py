"""The mesh worker runtime.

Lifecycle
---------
1. load the resident model (llama.cpp / upstream / simulator);
2. **benchmark itself** (GPU, CPU, FLOPS, head-node latency, decode speed);
3. register with the head node, advertising the measured capabilities;
4. heartbeat every ``heartbeat_seconds`` until the session TTL is refreshed;
5. serve ``/v1/chat`` (SSE), ``/v1/chat/sync``, ``/v1/model/load`` and
   ``/v1/admin/drain``;
6. drain gracefully on ``SIGTERM`` so the rotator can retire it without
   dropping in-flight requests.

The agent is written to survive a dead head node: registration retries with
exponential backoff, and the worker keeps serving local traffic meanwhile.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import signal
import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from core.logging_setup import get_logger, setup_logging
from core.protocol import (
    ChatCompletion,
    ChatMessage,
    ChatRequest,
    ModelLoadResult,
    TokenChunk,
    WorkerAck,
    WorkerHeartbeat,
    WorkerHello,
    WorkerStatus,
)
from worker.config import WorkerConfig
from worker.inference_engine import GenerationParams, InferenceEngine, ModelRuntime
from worker.telemetry import Benchmarker, BenchmarkReport, capabilities_from_report

LOGGER = get_logger("worker.agent")

CHATML = "chatml"
LLAMA3 = "llama3"
PLAIN = "plain"


def build_prompt(messages: list[ChatMessage], template: str = CHATML) -> str:
    """Render a conversation into a single prompt string."""
    if template == PLAIN:
        return "\n\n".join(f"{m.role.capitalize()}: {m.content}" for m in messages)
    if template == LLAMA3:
        parts = []
        for message in messages:
            parts.append(f"<|start_header_id|>{message.role}<|end_header_id|>\n\n{message.content}<|eot_id|>")
        parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return "".join(parts)
    parts = []
    for message in messages:
        parts.append(f"<|im_start|>{message.role}\n{message.content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


class WorkerAgent:
    """The worker process."""

    def __init__(
        self,
        config: WorkerConfig | None = None,
        *,
        engine: InferenceEngine | None = None,
        head_url: str | None = None,
    ) -> None:
        self.config = config or WorkerConfig.from_env()
        if head_url:
            self.config.head_url = head_url.rstrip("/")
        self.worker_id = self.config.resolved_worker_id()
        self.engine = engine or InferenceEngine(
            backend=self.config.backend,
            upstream_url=self.config.upstream_url,
            upstream_model=self.config.upstream_model,
            upstream_api_key=self.config.upstream_api_key,
            default_runtime=ModelRuntime(
                name=Path(self.config.model_filename).stem if self.config.model_filename else "",
                path=self.config.model_path,
                repo=self.config.model_repo,
                filename=self.config.model_filename,
                family=self.config.model_family,
                params_b=self.config.model_params_b,
                context=self.config.context_size,
                gpu_layers=self.config.gpu_layers,
                threads=self.config.threads,
                sha256=self.config.model_sha256,
            ),
        )
        self.report: BenchmarkReport | None = None
        self.status = WorkerStatus.BOOTING
        self.draining = False
        self.inflight = 0
        self.queue_depth = 0
        self.total_requests = 0
        self.total_tokens = 0
        self._client = httpx.AsyncClient(timeout=30.0)
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._register_task: asyncio.Task[None] | None = None
        self._backoff = self.config.reconnect_backoff_seconds
        self._stopped = asyncio.Event()

    # ------------------------------------------------------------------ #
    # Head-node plumbing
    # ------------------------------------------------------------------ #
    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if self.config.mesh_token:
            headers["x-mesh-token"] = self.config.mesh_token
        return headers

    async def register(self) -> bool:
        """Register with the head node; returns ``True`` on success."""
        capabilities = self._capabilities()
        hello = WorkerHello(
            worker_id=self.worker_id,
            host=self.config.bind_host if self.config.bind_host != "0.0.0.0" else "127.0.0.1",
            port=self.config.bind_port,
            capabilities=capabilities,
            model_loaded=self.engine.info().model,
            model_family=self.config.model_family,
            model_params_b=self.config.model_params_b,
            started_at=time.time(),
            metadata={
                "backend": self.engine.info().backend,
                "provider": self.config.provider,
                "region": self.config.region,
                "session_ends_at": self.config.session_ends_at,
            },
        )
        try:
            response = await self._client.post(
                f"{self.config.head_url}/v1/workers/register",
                json=hello.model_dump(mode="json"),
                headers=self._headers(),
            )
        except httpx.HTTPError as exc:
            LOGGER.warning("registration failed", extra={"hc_error": str(exc), "hc_head": self.config.head_url})
            return False
        if response.status_code >= 400:
            LOGGER.error(
                "registration rejected",
                extra={"hc_status": response.status_code, "hc_body": response.text[:200]},
            )
            return False
        try:
            ack = WorkerAck(**response.json())
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("malformed registration ack", extra={"hc_error": str(exc)})
            return False
        self.status = WorkerStatus.IDLE
        self._backoff = self.config.reconnect_backoff_seconds
        LOGGER.info(
            "registered with head node",
            extra={
                "hc_worker": self.worker_id,
                "hc_ttl": ack.ttl_seconds,
                "hc_drain_before": ack.drain_before_seconds,
                "hc_backend": self.engine.info().backend,
            },
        )
        await self._broadcast_event("worker.ready", {"worker_id": self.worker_id, "model": self.engine.info().model})
        return True

    def _capabilities(self) -> Any:
        if self.report is not None:
            return capabilities_from_report(
                self.report,
                provider=self.config.provider,
                region=self.config.region,
                families=[self.config.model_family, "qwen2", "llama", "gemma2", "phi3", "mistral"],
                context=self.config.context_size,
                max_concurrency=self.config.max_concurrency,
                model_params_b=self.config.model_params_b,
            )
        from core.protocol import Capabilities

        return Capabilities(
            provider=self.config.provider,
            region=self.config.region,
            backend=self.engine.info().backend,
            device="cpu",
            supported_families=[self.config.model_family],
            supported_context=self.config.context_size,
            max_concurrency=self.config.max_concurrency,
        )

    async def heartbeat(self) -> bool:
        """Send one heartbeat."""
        info = self.engine.info()
        beat = WorkerHeartbeat(
            worker_id=self.worker_id,
            status=self.status,
            inflight=self.inflight,
            queue_depth=self.queue_depth,
            tokens_per_sec=info.last_tokens_per_sec,
            ttft_ms=info.last_ttft_ms,
            vram_free_gb=self.report.gpu.vram_free_gb if self.report and self.report.gpu else 0.0,
            vram_total_gb=self.report.gpu.vram_total_gb if self.report and self.report.gpu else 0.0,
            model_loaded=info.model,
            mesh_latency_ms=self.report.head_latency.p50_ms if self.report else 0.0,
            ts=time.time(),
        )
        try:
            response = await self._client.post(
                f"{self.config.head_url}/v1/workers/{self.worker_id}/heartbeat",
                json=beat.model_dump(mode="json"),
                headers=self._headers(),
            )
        except httpx.HTTPError as exc:
            LOGGER.warning("heartbeat failed", extra={"hc_error": str(exc)})
            return False
        if response.status_code == 404:
            LOGGER.warning("head node forgot this worker; re-registering")
            await self.register()
            return False
        return response.status_code < 400

    async def _broadcast_event(self, event: str, data: dict[str, Any]) -> None:
        """Best-effort notification to the head node's event bus."""
        try:
            await self._client.post(
                f"{self.config.head_url}/v1/workers/{self.worker_id}/events",
                json={"event": event, "data": data},
                headers=self._headers(),
            )
        except httpx.HTTPError:
            pass

    # ------------------------------------------------------------------ #
    # Background loops
    # ------------------------------------------------------------------ #
    async def _registration_loop(self) -> None:
        """Keep trying to (re)register until the agent stops."""
        while not self._stopped.is_set():
            if await self.register():
                return
            await asyncio.sleep(self._backoff)
            self._backoff = min(self._backoff * 2, self.config.reconnect_backoff_max_seconds)

    async def _heartbeat_loop(self) -> None:
        """Heartbeat until the agent stops."""
        while not self._stopped.is_set():
            try:
                await asyncio.wait_for(self._stopped.wait(), timeout=self.config.heartbeat_seconds)
                return
            except asyncio.TimeoutError:
                pass
            if self.draining:
                continue
            ok = await self.heartbeat()
            if not ok:
                await asyncio.sleep(self._backoff)
                self._backoff = min(self._backoff * 2, self.config.reconnect_backoff_max_seconds)
                if not await self.register():
                    continue
                self._backoff = self.config.reconnect_backoff_seconds
            else:
                self._backoff = self.config.reconnect_backoff_seconds

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        """Load the model, benchmark, register and start the heartbeat."""
        self.status = WorkerStatus.BOOTING
        try:
            info = self.engine.load()
            LOGGER.info(
                "model ready",
                extra={"hc_backend": info.backend, "hc_model": info.model, "hc_detail": info.detail},
            )
        except Exception as exc:  # noqa: BLE001 - the agent must still come up
            LOGGER.error("model load failed", extra={"hc_error": str(exc)})
            self.status = WorkerStatus.DEAD
        self.report = await Benchmarker(
            worker_id=self.worker_id,
            head_url=self.config.head_url,
        ).run(self.engine)
        LOGGER.info("self-benchmark complete", extra={"hc_report": json.dumps(self.report.to_dict())[:600]})
        self._register_task = asyncio.create_task(self._registration_loop(), name="hc-worker-register")
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop(), name="hc-worker-heartbeat")

    async def stop(self) -> None:
        """Drain, stop the loops and release resources."""
        self.draining = True
        self.status = WorkerStatus.DRAINING
        self._stopped.set()
        for task in (self._heartbeat_task, self._register_task):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        with contextlib.suppress(Exception):
            await self._client.post(
                f"{self.config.head_url}/v1/workers/{self.worker_id}/drain",
                json={"worker_id": self.worker_id},
                headers=self._headers(),
            )
        with contextlib.suppress(Exception):
            await self._client.aclose()
        self.status = WorkerStatus.DEAD
        LOGGER.info("worker stopped", extra={"hc_worker": self.worker_id})

    # ------------------------------------------------------------------ #
    # Request handling
    # ------------------------------------------------------------------ #
    def _prompt_for(self, request: ChatRequest) -> str:
        return build_prompt(request.messages)

    def _params_for(self, request: ChatRequest) -> GenerationParams:
        return GenerationParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=list(request.stop),
            seed=request.seed,
        )

    async def _generate(self, request: ChatRequest) -> AsyncIterator[str]:
        """Yield text fragments for a request."""
        prompt = self._prompt_for(request)
        params = self._params_for(request)
        for piece in self.engine.stream(prompt, params):
            yield piece

    def _busy(self) -> JSONResponse:
        return JSONResponse(status_code=503, content={"error": "worker is busy", "worker_id": self.worker_id})

    # ------------------------------------------------------------------ #
    # HTTP surface
    # ------------------------------------------------------------------ #
    def create_app(self) -> FastAPI:
        """Build the FastAPI application implementing the worker protocol."""
        app = FastAPI(title=f"hyper-cluster-worker-{self.worker_id}", version="1.0.0")

        @app.get("/health")
        async def health() -> dict[str, Any]:
            info = self.engine.info()
            return {
                "status": "ok" if not self.draining else "draining",
                "protocol_version": "1.0",
                "worker_id": self.worker_id,
                "model_loaded": info.model,
                "backend": info.backend,
                "draining": self.draining,
                "inflight": self.inflight,
                "vram_free_gb": self.report.gpu.vram_free_gb if self.report and self.report.gpu else 0.0,
                "vram_total_gb": self.report.gpu.vram_total_gb if self.report and self.report.gpu else 0.0,
                "tokens_per_sec": info.last_tokens_per_sec,
            }

        @app.get("/v1/telemetry")
        async def telemetry() -> dict[str, Any]:
            return {
                "worker_id": self.worker_id,
                "report": self.report.to_dict() if self.report else None,
                "engine": self.engine.info().to_dict(),
                "inflight": self.inflight,
                "total_requests": self.total_requests,
                "total_tokens": self.total_tokens,
            }

        @app.post("/v1/chat")
        async def chat(payload: ChatRequest, request: Request) -> Any:
            if self.draining:
                return JSONResponse(status_code=503, content={"error": "worker is draining"})
            if self.inflight >= self.config.max_concurrency:
                return self._busy()

            async def _stream() -> AsyncIterator[str]:
                self.inflight += 1
                self.total_requests += 1
                started = time.perf_counter()
                try:
                    index = 0
                    async for piece in self._generate(payload):
                        if await request.is_disconnected():
                            break
                        chunk = TokenChunk(
                            request_id=payload.request_id,
                            text=piece,
                            index=index,
                            source="cluster",
                            draft=False,
                        )
                        index += 1
                        self.total_tokens += 1
                        yield f"data: {chunk.model_dump_json()}\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    self.inflight = max(0, self.inflight - 1)
                    elapsed = time.perf_counter() - started
                    LOGGER.debug(
                        "served request",
                        extra={"hc_request": payload.request_id, "hc_ms": round(elapsed * 1000, 1)},
                    )

            return StreamingResponse(_stream(), media_type="text/event-stream")

        @app.post("/v1/chat/sync")
        async def chat_sync(payload: ChatRequest) -> Any:
            if self.draining:
                return JSONResponse(status_code=503, content={"error": "worker is draining"})
            if self.inflight >= self.config.max_concurrency:
                return self._busy()
            self.inflight += 1
            self.total_requests += 1
            started = time.perf_counter()
            first_at: float | None = None
            parts: list[str] = []
            try:
                for piece in self.engine.stream(self._prompt_for(payload), self._params_for(payload)):
                    if first_at is None:
                        first_at = time.perf_counter()
                    parts.append(piece)
            finally:
                self.inflight = max(0, self.inflight - 1)
            elapsed = time.perf_counter() - started
            text = "".join(parts)
            tokens = max(1, len(text) // 4)
            completion = ChatCompletion(
                request_id=payload.request_id,
                text=text,
                model=self.engine.info().model,
                worker_id=self.worker_id,
                tier="cluster",
                source="cluster",
                mode=payload.mode,
                tokens=tokens,
                ttft_ms=((first_at - started) * 1000.0) if first_at else elapsed * 1000.0,
                total_ms=elapsed * 1000.0,
                tokens_per_sec=tokens / elapsed if elapsed > 0 else 0.0,
                finish_reason="stop",
            )
            return completion.model_dump(mode="json")

        @app.post("/v1/model/load")
        async def model_load(payload: dict[str, Any]) -> Any:
            try:
                runtime = ModelRuntime(
                    name=str(payload.get("model") or "unknown"),
                    path=self.config.model_path,
                    repo=self.config.model_repo,
                    filename=str(payload.get("model") or ""),
                    family=str(payload.get("family") or self.config.model_family),
                    params_b=float(payload.get("params_b") or self.config.model_params_b),
                    context=self.config.context_size,
                    gpu_layers=self.config.gpu_layers,
                    threads=self.config.threads,
                )
                started = time.perf_counter()
                info = self.engine.load(runtime)
                elapsed = (time.perf_counter() - started) * 1000.0
                self.config.model_family = runtime.family
                return ModelLoadResult(
                    worker_id=self.worker_id,
                    model=info.model,
                    loaded=True,
                    load_ms=elapsed,
                    vram_used_gb=info.vram_used_gb,
                ).model_dump(mode="json")
            except Exception as exc:  # noqa: BLE001
                return ModelLoadResult(
                    worker_id=self.worker_id,
                    model=str(payload.get("model") or ""),
                    loaded=False,
                    error=str(exc),
                ).model_dump(mode="json")

        @app.post("/v1/admin/drain")
        async def drain(payload: dict[str, Any] | None = None) -> Any:
            self.draining = True
            self.status = WorkerStatus.DRAINING
            LOGGER.info("drain requested", extra={"hc_worker": self.worker_id})
            return {"draining": True, "worker_id": self.worker_id}

        @app.post("/v1/admin/undrain")
        async def undrain(payload: dict[str, Any] | None = None) -> Any:
            self.draining = False
            self.status = WorkerStatus.IDLE
            return {"draining": False, "worker_id": self.worker_id}

        return app


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """Command line interface of the worker."""
    parser = argparse.ArgumentParser(prog="worker.worker_agent", description="Hyper-Cluster mesh worker.")
    parser.add_argument("--head-url", default=None, help="head node URL")
    parser.add_argument("--mesh-token", default=None, help="mesh token")
    parser.add_argument("--host", default=None, help="bind address")
    parser.add_argument("--port", type=int, default=None, help="bind port")
    parser.add_argument("--model-path", default=None, help="path to the local GGUF")
    parser.add_argument("--model-repo", default=None, help="Hugging Face repo of the model")
    parser.add_argument("--model-filename", default=None, help="GGUF filename inside the repo")
    parser.add_argument("--model-family", default=None, help="model family (qwen2, llama, ...)")
    parser.add_argument("--model-params-b", type=float, default=None, help="parameter count in billions")
    parser.add_argument("--context", type=int, default=None, help="context window")
    parser.add_argument("--gpu-layers", type=int, default=None, help="layers to offload to the GPU (-1 = auto)")
    parser.add_argument("--backend", default=None, choices=["auto", "llama.cpp", "upstream", "simulator"], help="inference backend")
    parser.add_argument("--upstream-url", default=None, help="OpenAI-compatible upstream URL")
    parser.add_argument("--upstream-model", default=None, help="model name on the upstream server")
    parser.add_argument("--provider", default=None, help="provider label reported to the head node")
    parser.add_argument("--region", default=None, help="region label reported to the head node")
    parser.add_argument("--max-concurrency", type=int, default=None, help="concurrent requests")
    parser.add_argument("--log-level", default=None, help="DEBUG/INFO/WARNING")
    parser.add_argument("--log-text", action="store_true", help="human readable logs")
    parser.add_argument("--telemetry-only", action="store_true", help="print the self-benchmark and exit")
    return parser


def config_from_args(args: argparse.Namespace) -> WorkerConfig:
    """Merge CLI flags into the environment-derived configuration."""
    config = WorkerConfig.from_env()
    for attribute, value in (
        ("head_url", args.head_url),
        ("mesh_token", args.mesh_token),
        ("bind_host", args.host),
        ("bind_port", args.port),
        ("model_path", args.model_path),
        ("model_repo", args.model_repo),
        ("model_filename", args.model_filename),
        ("model_family", args.model_family),
        ("model_params_b", args.model_params_b),
        ("context_size", args.context),
        ("gpu_layers", args.gpu_layers),
        ("backend", args.backend),
        ("upstream_url", args.upstream_url),
        ("upstream_model", args.upstream_model),
        ("provider", args.provider),
        ("region", args.region),
        ("max_concurrency", args.max_concurrency),
        ("log_level", args.log_level),
    ):
        if value is not None:
            setattr(config, attribute, value)
    if args.log_text:
        config.log_json = False
    return config


async def _serve(config: WorkerConfig, agent: WorkerAgent) -> None:
    import uvicorn

    server = uvicorn.Server(
        uvicorn.Config(
            agent.create_app(),
            host=config.bind_host,
            port=config.bind_port,
            log_level=config.log_level.lower(),
            access_log=False,
        )
    )
    await agent.start()
    LOGGER.info(
        "worker serving",
        extra={
            "hc_worker": agent.worker_id,
            "hc_bind": f"{config.bind_host}:{config.bind_port}",
            "hc_model": agent.engine.info().model,
            "hc_backend": agent.engine.info().backend,
        },
    )
    stop_event = asyncio.Event()

    def _request_stop(*_: Any) -> None:
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, _request_stop)

    serve_task = asyncio.create_task(server.serve())
    stop_task = asyncio.create_task(stop_event.wait())
    done, pending = await asyncio.wait({serve_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
    for task in done:
        with contextlib.suppress(Exception):
            await task
    await agent.stop()


def main(argv: list[str] | None = None) -> int:
    """Start the worker."""
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    setup_logging(config.log_level, json_mode=config.log_json, log_file=config.log_file or None)

    if args.telemetry_only:
        agent = WorkerAgent(config)
        report = asyncio.run(
            Benchmarker(worker_id=agent.worker_id, head_url=config.head_url).run(agent.engine)
        )
        print(json.dumps(report.to_dict(), indent=2))
        return 0

    asyncio.run(_serve(config, WorkerAgent(config)))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
