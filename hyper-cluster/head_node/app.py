"""FastAPI control plane for the head node.

Routes
------
``GET  /``                      service banner
``GET  /health``                liveness + coarse cluster health
``GET  /v1/status``             full telemetry snapshot (dashboard polls this)
``GET  /v1/workers``            worker inventory
``POST /v1/workers/register``   worker self-registration (mesh token)
``POST /v1/workers/{id}/heartbeat``  liveness beacon
``POST /v1/workers/{id}/drain``  graceful retirement
``DELETE /v1/workers/{id}``     hard removal
``POST /v1/chat``               non-streaming completion
``POST /v1/chat/stream``        Server-Sent Events completion
``GET  /v1/events``             live telemetry stream (SSE)
``GET  /v1/models``             model catalogue + active model
``GET  /v1/requests/{id}``      status (and result) of any request
``GET  /v1/queue``              hold-and-requeue buffer
``GET  /v1/credentials``        credential vault snapshot (admin)
``POST /v1/credentials``        register a provider account (admin)

Milestone 4 -- the memory and the sleep-time engine:
``GET  /v1/memory``             the state of the persistent memory
``GET  /v1/memory/documents``   the indexed documents
``POST /v1/memory/documents``   upload a file (it is indexed immediately)
``POST /v1/memory/search``      hybrid retrieval over the memory
``POST /v1/memory/sync``        push / pull the memory snapshot (admin)
``GET  /v1/evolution/status``   the state of the sleep-time engine
``POST /v1/evolution/run``      run one self-evolution cycle now (admin)
``POST /v1/evolution/prompt``   set the system prompt of the cluster (admin)
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from core.config import Settings, get_settings
from core.credentials import Credential, get_vault
from core.errors import HyperClusterError
from core.ids import now_s
from core.logging_setup import get_logger, setup_logging
from core.protocol import (
    ChatRequest,
    WorkerAck,
    WorkerHeartbeat,
    WorkerHello,
)
from head_node.auto_spawner import SpawnRequest
from head_node.events import EventBus
from head_node.orchestrator import Orchestrator
from head_node.request_store import RequestLedger
from head_node.worker_registry import WorkerRegistry
from tools.sandbox import PythonSandbox, ToolExecutor

LOGGER = get_logger("head_node.app")

SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def sse_payload(event: str, data: dict[str, Any]) -> str:
    """Render one Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def create_app(
    settings: Settings | None = None,
    *,
    orchestrator: Orchestrator | None = None,
) -> FastAPI:
    """Build the FastAPI application (factory kept separate for tests)."""
    resolved = settings or get_settings()
    setup_logging(resolved.log_level, json_mode=resolved.log_json, log_file=resolved.log_file or None)
    resolved.ensure_dirs()

    registry = WorkerRegistry(
        ttl_seconds=resolved.worker_ttl_seconds,
        max_concurrency_per_worker=resolved.max_concurrency_per_worker,
    )
    events = EventBus()
    ledger = RequestLedger()
    tools = ToolExecutor(
        sandbox=PythonSandbox(
            timeout=resolved.sandbox_timeout_seconds,
            memory_limit_mb=resolved.sandbox_memory_limit_mb,
        ),
    ) if resolved.sandbox_enabled else None
    engine = orchestrator or Orchestrator(
        resolved,
        registry=registry,
        events=events,
        ledger=ledger,
        tools=tools,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = resolved
        app.state.orchestrator = engine
        app.state.events = events
        app.state.ledger = ledger
        await engine.startup()
        LOGGER.info(
            "head node ready",
            extra={
                "hc_cluster": resolved.cluster_name,
                "hc_port": resolved.head_port,
                "hc_instance": resolved.instance_id,
            },
        )
        try:
            yield
        finally:
            await engine.shutdown()

    app = FastAPI(
        title="Hyper-Cluster Omega",
        version="1.0.0",
        description="Zero-limit autonomous AI operating system -- head node control plane.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved.cors_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------ #
    # Authentication
    # ------------------------------------------------------------------ #
    async def require_client_auth(
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> str:
        """Authenticate a phone/browser client."""
        expected = resolved.api_key
        if not expected:
            return "anonymous"
        presented = x_api_key
        if not presented and authorization and authorization.lower().startswith("bearer "):
            presented = authorization[7:].strip()
        if not presented or presented != expected:
            raise HTTPException(status_code=401, detail="invalid or missing API key")
        return "client"

    async def require_mesh_auth(
        authorization: str | None = Header(default=None),
        x_mesh_token: str | None = Header(default=None),
    ) -> str:
        """Authenticate a mesh worker."""
        expected = resolved.mesh_token
        if not expected:
            return "anonymous"
        presented = x_mesh_token
        if not presented and authorization and authorization.lower().startswith("bearer "):
            presented = authorization[7:].strip()
        if not presented or presented != expected:
            raise HTTPException(status_code=401, detail="invalid or missing mesh token")
        return "worker"

    async def require_admin_auth(
        authorization: str | None = Header(default=None),
        x_admin_token: str | None = Header(default=None),
    ) -> str:
        """Authenticate an administrative call."""
        expected = resolved.admin_token or resolved.api_key
        if not expected:
            return "anonymous"
        presented = x_admin_token
        if not presented and authorization and authorization.lower().startswith("bearer "):
            presented = authorization[7:].strip()
        if not presented or presented != expected:
            raise HTTPException(status_code=401, detail="invalid or missing admin token")
        return "admin"

    # ------------------------------------------------------------------ #
    # Error handling
    # ------------------------------------------------------------------ #
    @app.exception_handler(HyperClusterError)
    async def hypercluster_error_handler(request: Request, exc: HyperClusterError) -> JSONResponse:
        LOGGER.warning(
            "request rejected",
            extra={"hc_error": exc.code, "hc_message": exc.message, "hc_path": request.url.path},
        )
        return JSONResponse(status_code=exc.http_status, content=exc.to_dict())

    # ------------------------------------------------------------------ #
    # Service level
    # ------------------------------------------------------------------ #
    @app.get("/")
    async def root() -> dict[str, Any]:
        return {
            "service": "hyper-cluster-omega",
            "version": "1.0.0",
            "cluster": resolved.cluster_name,
            "instance": resolved.instance_id,
            "protocol": "1.0",
            "docs": "/docs",
            "status": "/v1/status",
        }

    @app.get("/health")
    async def health() -> dict[str, Any]:
        status = engine.status()
        return {
            "status": "ok" if status.healthy else "degraded",
            "ts": now_s(),
            "workers": status.worker_count,
            "healthy_workers": status.healthy_workers,
            "edge": status.edge_healthy,
            "queued": status.queued_requests,
        }

    @app.get("/v1/status")
    async def cluster_status(_: str = Depends(require_client_auth)) -> dict[str, Any]:
        status = engine.status()
        payload = status.model_dump(mode="json")
        payload["hold_queue"] = engine.hold_queue.stats()
        payload["hold_queue_entries"] = engine.hold_queue.snapshot()
        payload["edge"] = engine.edge.stats()
        payload["ledger"] = engine.ledger.stats()
        payload["settings"] = resolved.redacted()
        return payload

    # ------------------------------------------------------------------ #
    # Worker mesh
    # ------------------------------------------------------------------ #
    @app.get("/v1/workers")
    async def list_workers(_: str = Depends(require_client_auth)) -> dict[str, Any]:
        return {
            "workers": [record.model_dump(mode="json") for record in engine.registry.snapshot()],
            "totals": engine.registry.totals(),
        }

    @app.post("/v1/workers/register", response_model=WorkerAck)
    async def register_worker(
        hello: WorkerHello,
        _: str = Depends(require_mesh_auth),
    ) -> WorkerAck:
        record = engine.registry.register(hello)
        await engine.events.emit(
            "worker.registered",
            {
                "worker_id": record.worker_id,
                "provider": record.provider,
                "vram_gb": record.vram_total_gb,
                "model": record.model_loaded,
            },
        )
        await engine.notify_worker_changed(record.worker_id)
        return WorkerAck(
            accepted=True,
            worker_id=record.worker_id,
            assigned_worker_id=record.worker_id,
            ttl_seconds=engine.registry.ttl_seconds,
            refresh_before_seconds=max(30, engine.registry.ttl_seconds // 4),
            drain_before_seconds=resolved.handoff_lead_seconds,
            head_url=f"http://{resolved.head_host}:{resolved.head_port}",
            message="registered",
        )

    @app.post("/v1/workers/{worker_id}/heartbeat")
    async def worker_heartbeat(
        worker_id: str,
        beat: WorkerHeartbeat,
        _: str = Depends(require_mesh_auth),
    ) -> dict[str, Any]:
        if beat.worker_id and beat.worker_id != worker_id:
            raise HTTPException(status_code=400, detail="worker_id mismatch")
        record = engine.registry.heartbeat(beat)
        if record is None:
            raise HTTPException(status_code=404, detail="unknown worker")
        engine._capacity_event.set()  # noqa: SLF001 - internal wake-up for held requests
        return {"ok": True, "expires_at": record.expires_at, "seconds_to_expiry": record.seconds_to_expiry}

    @app.post("/v1/workers/{worker_id}/events")
    async def worker_event(
        worker_id: str,
        payload: dict[str, Any],
        _: str = Depends(require_mesh_auth),
    ) -> dict[str, Any]:
        """Let a worker push a notification into the cluster event stream."""
        event = str(payload.get("event") or "worker.event")
        data = dict(payload.get("data") or {})
        data.setdefault("worker_id", worker_id)
        await engine.events.emit(event, data)
        return {"ok": True}

    @app.post("/v1/workers/{worker_id}/drain")
    async def drain_worker(worker_id: str, _: str = Depends(require_admin_auth)) -> dict[str, Any]:
        record = engine.registry.drain(worker_id, reason="admin request")
        if record is None:
            raise HTTPException(status_code=404, detail="unknown worker")
        await engine.client.drain(record)
        await engine.events.emit("worker.draining", {"worker_id": worker_id})
        return {"ok": True, "worker_id": worker_id, "draining": True}

    @app.delete("/v1/workers/{worker_id}")
    async def remove_worker(worker_id: str, _: str = Depends(require_admin_auth)) -> dict[str, Any]:
        record = engine.registry.remove(worker_id)
        if record is None:
            raise HTTPException(status_code=404, detail="unknown worker")
        await engine.events.emit("worker.expired", {"worker_id": worker_id, "reason": "removed"})
        return {"ok": True, "worker_id": worker_id}

    # ------------------------------------------------------------------ #
    # Chat
    # ------------------------------------------------------------------ #
    @app.post("/v1/chat")
    async def chat(payload: ChatRequest, _: str = Depends(require_client_auth)):
        if payload.stream:
            return StreamingResponse(
                engine_event_stream(engine, payload),
                media_type="text/event-stream",
                headers=SSE_HEADERS,
            )
        completion = await engine.submit(payload)
        status_code = 202 if completion.error == "deferred" else 200
        return JSONResponse(status_code=status_code, content=completion.model_dump(mode="json"))

    @app.post("/v1/chat/stream")
    async def chat_stream(payload: ChatRequest, _: str = Depends(require_client_auth)) -> StreamingResponse:
        return StreamingResponse(
            engine_event_stream(engine, payload),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    @app.get("/v1/requests/{request_id}")
    async def request_status(request_id: str, _: str = Depends(require_client_auth)) -> dict[str, Any]:
        status = engine.ledger.get(request_id)
        if status is None:
            raise HTTPException(status_code=404, detail="unknown request")
        completion = engine.ledger.result(request_id)
        payload: dict[str, Any] = {"status": status.model_dump(mode="json")}
        if completion is not None:
            payload["result"] = completion.model_dump(mode="json")
        return payload

    # ------------------------------------------------------------------ #
    # Telemetry stream
    # ------------------------------------------------------------------ #
    @app.get("/v1/events")
    async def event_stream(_: str = Depends(require_client_auth)) -> StreamingResponse:
        return StreamingResponse(
            telemetry_stream(engine),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    # ------------------------------------------------------------------ #
    # Models, queue, credentials
    # ------------------------------------------------------------------ #
    @app.get("/v1/models")
    async def models(_: str = Depends(require_client_auth)) -> dict[str, Any]:
        catalog = engine.scheduler.catalog
        return {
            "active_model": engine.scheduler.active_model(),
            "models": [spec.model_dump(mode="json") for spec in catalog.models],
            "edge_models": catalog.edge_models,
        }

    @app.get("/v1/tools")
    async def tools_catalogue(_: str = Depends(require_client_auth)) -> dict[str, Any]:
        """The tool schemas the model can call."""
        executor = engine.tools
        return {
            "enabled": executor is not None,
            "tools": executor.list_tools() if executor is not None else [],
            "max_rounds": resolved.max_tool_rounds,
        }

    @app.get("/v1/queue")
    async def queue(_: str = Depends(require_client_auth)) -> dict[str, Any]:
        return {
            "stats": engine.hold_queue.stats(),
            "entries": engine.hold_queue.snapshot(),
        }

    @app.get("/v1/credentials")
    async def credentials(_: str = Depends(require_admin_auth)) -> dict[str, Any]:
        return {"credentials": get_vault(resolved.credential_file).snapshot()}

    # ------------------------------------------------------------------ #
    # Cluster capacity: VRAM pool, auto-scaler, spawner
    # ------------------------------------------------------------------ #
    @app.get("/v1/cluster/vram")
    async def cluster_vram(_: str = Depends(require_client_auth)) -> dict[str, Any]:
        """The live, measured VRAM pool of the mesh."""
        pool = engine.selector.pool()
        return {
            "pool": pool.to_dict(),
            "demand": engine.selector.demand.to_dict(),
            "target_model": engine.selector.target_model(pool).name
            if engine.selector.target_model(pool)
            else None,
        }

    @app.get("/v1/models/plan")
    async def models_plan(_: str = Depends(require_client_auth)) -> dict[str, Any]:
        """The scaling plan the auto-scaler would apply right now (dry run)."""
        decisions = engine.selector.plan()
        return {
            "enabled": resolved.autoscaler_enabled,
            "decisions": [decision.to_dict() for decision in decisions],
            "demand": engine.selector.demand.to_dict(),
        }

    @app.post("/v1/cluster/autoscale")
    async def run_autoscale(_: str = Depends(require_admin_auth)) -> dict[str, Any]:
        """Apply the scaling plan immediately (hot-swap the models)."""
        applied = await engine.autoscale()
        return {
            "ok": True,
            "applied": [decision.to_dict() for decision in applied],
        }

    @app.get("/v1/speculative/pair")
    async def speculative_pair(_: str = Depends(require_client_auth)) -> dict[str, Any]:
        """The draft/verifier pairing computed from the live pool."""
        pair = engine.selector.draft_pair()
        return {"pair": pair.to_dict() if pair else None}

    @app.get("/v1/spawner")
    async def spawner_status(_: str = Depends(require_admin_auth)) -> dict[str, Any]:
        """The state of the headless provisioner and the TTL rotator."""
        return engine.spawner.status()

    @app.post("/v1/spawner/provision")
    async def spawner_provision(
        payload: dict[str, Any],
        _: str = Depends(require_admin_auth),
    ) -> dict[str, Any]:
        """Provision compute on demand (multi-account failover)."""
        request = SpawnRequest(
            provider=str(payload.get("provider") or ""),
            region=str(payload.get("region") or "unknown"),
            model=payload.get("model"),
            family=payload.get("family"),
            params_b=payload.get("params_b"),
            vram_gb=float(payload.get("vram_gb") or 0.0),
            ttl_seconds=int(payload.get("ttl_seconds") or resolved.spawner_default_ttl_seconds),
            labels={str(k): str(v) for k, v in (payload.get("labels") or {}).items()},
            head_url=resolved.public_base_url
            or f"http://{resolved.head_host}:{resolved.head_port}",
            mesh_token=resolved.mesh_token,
        )
        result = await engine.spawner.provision(request)
        return result.to_dict()

    @app.post("/v1/spawner/rotate")
    async def spawner_rotate(
        payload: dict[str, Any],
        _: str = Depends(require_admin_auth),
    ) -> dict[str, Any]:
        """Force the pre-emptive handover of one worker."""
        worker_id = str(payload.get("worker_id") or "")
        record = engine.registry.get(worker_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"unknown worker: {worker_id}")
        action = await engine.spawner.rotate(record)
        return action.to_dict()

    # ------------------------------------------------------------------ #
    # Memory and self-evolution (milestone 4)
    # ------------------------------------------------------------------ #
    def _require_memory() -> Any:
        if engine.memory is None:
            raise HTTPException(status_code=409, detail="the memory is disabled")
        return engine.memory

    @app.get("/v1/memory")
    async def memory_status(_: str = Depends(require_client_auth)) -> dict[str, Any]:
        """The state of the persistent memory."""
        if engine.memory is None:
            return {"enabled": False}
        return {"enabled": True, **engine.memory.stats()}

    @app.get("/v1/memory/documents")
    async def memory_documents(_: str = Depends(require_client_auth)) -> dict[str, Any]:
        """The documents that are indexed."""
        memory = _require_memory()
        return {"documents": [record.to_dict() for record in memory.documents()]}

    @app.post("/v1/memory/search")
    async def memory_search(
        payload: dict[str, Any],
        _: str = Depends(require_client_auth),
    ) -> dict[str, Any]:
        """Search the memory (hybrid retrieval)."""
        memory = _require_memory()
        query = str(payload.get("query") or "")
        if not query:
            raise HTTPException(status_code=422, detail="query is required")
        hits = memory.search(
            query,
            k=int(payload.get("k") or 5),
            sources=payload.get("sources") or None,
        )
        return {"query": query, "hits": [hit.to_dict() for hit in hits]}

    @app.post("/v1/memory/documents")
    async def memory_upload(
        file: UploadFile = File(...),
        _: str = Depends(require_client_auth),
    ) -> dict[str, Any]:
        """Upload a document -- it is indexed immediately."""
        memory = _require_memory()
        payload = await file.read()
        if not payload:
            raise HTTPException(status_code=422, detail="the uploaded file is empty")
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=415, detail="only text files can be indexed") from None
        record = memory.add_document(
            file.filename or "upload",
            text,
            metadata={"source_path": file.filename or "", "size": len(payload)},
        )
        return {"ok": True, "document": record.to_dict()}

    @app.post("/v1/memory/sync")
    async def memory_sync(
        payload: dict[str, Any],
        _: str = Depends(require_admin_auth),
    ) -> dict[str, Any]:
        """Push the memory to Drive / the Hub, or pull it back."""
        memory = _require_memory()
        action = str(payload.get("action") or "push")
        targets = tuple(payload.get("targets") or ("drive", "hub"))
        drive = None
        hub = None
        if "drive" in targets and resolved.drive_token:
            from memory.sync import DriveClient

            drive = DriveClient(
                token=resolved.drive_token,
                api_base=resolved.drive_api_base,
                upload_base=resolved.drive_upload_base,
            )
        if "hub" in targets and resolved.hub_token and resolved.hub_repo:
            from memory.sync import HubClient

            hub = HubClient(token=resolved.hub_token, base_url=resolved.hub_base_url)
        if drive is None and hub is None:
            raise HTTPException(status_code=409, detail="no sync target is configured")
        from memory.sync import MemorySync

        sync = MemorySync(
            drive=drive,
            hub=hub,
            drive_folder=resolved.drive_folder,
            hub_repo=resolved.hub_repo,
        )
        try:
            if action == "pull":
                snapshots = await sync.pull(targets=targets)
                restored = 0
                for snapshot in snapshots:
                    restored = memory.restore(snapshot)
                return {"ok": True, "action": "pull", "restored": restored}
            receipts = await sync.push(memory.snapshot(), targets=targets)
            return {
                "ok": True,
                "action": "push",
                "receipts": [receipt.to_dict() for receipt in receipts],
            }
        finally:
            if drive is not None:
                await drive.aclose()
            if hub is not None:
                await hub.aclose()

    @app.get("/v1/evolution/status")
    async def evolution_status(_: str = Depends(require_client_auth)) -> dict[str, Any]:
        """The state of the sleep-time self-evolution engine."""
        if engine.autopilot is None:
            return {"enabled": False}
        return {"enabled": True, **engine.autopilot.status()}

    @app.post("/v1/evolution/run")
    async def evolution_run(
        payload: dict[str, Any],
        _: str = Depends(require_admin_auth),
    ) -> dict[str, Any]:
        """Run one self-evolution cycle now."""
        if engine.autopilot is None:
            raise HTTPException(status_code=409, detail="the evolution engine is disabled")
        report = await engine.autopilot.run_cycle(dataset_size=payload.get("dataset_size"))
        return report.to_dict()

    @app.post("/v1/evolution/prompt")
    async def evolution_prompt(
        payload: dict[str, Any],
        _: str = Depends(require_admin_auth),
    ) -> dict[str, Any]:
        """Set the system prompt of the cluster."""
        if engine.autopilot is None:
            raise HTTPException(status_code=409, detail="the evolution engine is disabled")
        prompt = str(payload.get("prompt") or "")
        if not prompt:
            raise HTTPException(status_code=422, detail="prompt is required")
        engine.autopilot.system_prompt = prompt
        return {"ok": True, "system_prompt": engine.autopilot.system_prompt}

    @app.post("/v1/credentials")
    async def add_credential(
        payload: dict[str, Any],
        _: str = Depends(require_admin_auth),
    ) -> dict[str, Any]:
        try:
            credential = Credential.from_dict(payload)
        except HyperClusterError as exc:
            raise HTTPException(status_code=422, detail=exc.message) from exc
        if not credential.token:
            raise HTTPException(status_code=422, detail="token is required")
        get_vault(resolved.credential_file).add(credential)
        return {"ok": True, "credential": credential.to_dict(redact=True)}

    return app


async def engine_event_stream(engine: Orchestrator, payload: ChatRequest) -> AsyncIterator[str]:
    """Translate orchestrator events into SSE frames."""
    try:
        async for event in engine.stream(payload):
            yield sse_payload(event.event, event.data)
    except Exception as exc:  # noqa: BLE001 - never leave the SSE hanging
        LOGGER.exception("stream failed", extra={"hc_request": payload.request_id})
        yield sse_payload("error", {"message": str(exc), "type": type(exc).__name__})


async def telemetry_stream(engine: Orchestrator) -> AsyncIterator[str]:
    """Push live cluster telemetry to the dashboard."""
    queue = await engine.events.subscribe()
    try:
        yield sse_payload("hello", {"ts": now_s(), "cluster": engine.settings.cluster_name})
        while True:
            try:
                event = await asyncio_wait(queue, timeout=15.0)
            except TimeoutError:
                yield sse_payload("ping", {"ts": now_s()})
                continue
            yield sse_payload(event.event, event.data)
    finally:
        await engine.events.unsubscribe(queue)


async def asyncio_wait(queue: Any, timeout: float) -> Any:
    """``asyncio.wait_for`` on a queue with a timeout error translation."""
    import asyncio

    try:
        return await asyncio.wait_for(queue.get(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise TimeoutError from exc


__all__ = ["create_app", "sse_payload", "SSE_HEADERS"]
