"""The orchestrator: the single brain of the cluster.

Responsibilities
----------------
* **Mode routing** -- ``fast`` / ``deep`` / ``council`` / ``sandbox`` are picked
  automatically when the client asks for ``auto``.
* **Speculative cascading** -- the first tokens come from a sub-second edge
  provider while the GPU cluster verifies and deepens the answer in parallel.
  The phone therefore starts rendering in ~200 ms even when the GPU is behind a
  VPN.
* **Hold & requeue** -- when no worker can serve a request it is parked in
  memory, replayed on the next healthy worker, and *never* surfaces as a 5xx.
* **Council mode** -- a panel of distinct models answers, critiques each other
  and synthesises one definitive answer.
* **Failover** -- a worker that dies mid-stream is transparently retried on the
  next-best worker.

The orchestrator emits a stream of :class:`StreamEvent` objects; the FastAPI
layer turns them into Server-Sent Events for the phone.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from core.catalog import find_model
from core.config import Settings
from core.credentials import CredentialVault, get_vault
from core.errors import (
    HoldTimeoutError,
    NoCapacityError,
    WorkerUnreachableError,
)
from core.ids import mono_ms, now_s
from core.logging_setup import get_logger, log_context
from core.protocol import (
    ChatCompletion,
    ChatMessage,
    ChatRequest,
    ChunkSource,
    ClusterStatus,
    EventType,
    RequestMode,
    RequestState,
    StreamEvent,
    WorkerTier,
)
from head_node.council import CouncilEngine, CouncilOutcome, similarity
from head_node.edge_bridge import EdgeBridge
from head_node.events import EventBus
from head_node.hold_queue import HoldQueue
from head_node.request_store import RequestLedger
from head_node.scheduler import Scheduler, SchedulerDecision
from head_node.worker_client import WorkerClient
from head_node.worker_registry import WorkerRecord, WorkerRegistry
from tools.sandbox import ToolExecutor, build_tool_prompt, parse_tool_call

LOGGER = get_logger("head_node.orchestrator")

#: How long we are willing to wait for the *cluster* to verify an edge draft.
VERIFICATION_BUDGET_SECONDS = 25.0

#: Similarity above which an edge draft is considered "confirmed" by the cluster.
VERIFICATION_THRESHOLD = 0.55


@dataclass
class RequestStream:
    """Per-request event pipe + completion future."""

    request_id: str
    queue: asyncio.Queue[StreamEvent] = field(default_factory=asyncio.Queue)
    future: asyncio.Future[ChatCompletion] | None = None
    created_at: float = field(default_factory=now_s)
    tokens_emitted: int = 0

    def __post_init__(self) -> None:
        if self.future is None:
            self.future = asyncio.get_running_loop().create_future()

    async def emit(self, event: str, data: dict[str, Any] | None = None) -> None:
        """Push an event towards the client."""
        payload = data or {}
        if event == "token":
            self.tokens_emitted += 1
            payload.setdefault("index", self.tokens_emitted - 1)
        await self.queue.put(StreamEvent(event=event, request_id=self.request_id, data=payload))

    def resolve(self, completion: ChatCompletion) -> None:
        if self.future is not None and not self.future.done():
            self.future.set_result(completion)

    def reject(self, error: BaseException) -> None:
        if self.future is not None and not self.future.done():
            self.future.set_exception(error)
            self.future.add_done_callback(_drain_future)

    async def events(self) -> AsyncIterator[StreamEvent]:
        """Yield events until the terminal one arrives."""
        while True:
            event = await self.queue.get()
            yield event
            if event.event in ("done", "error", "deferred"):
                return


class Orchestrator:
    """Coordinates the whole cluster on behalf of the phone."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        registry: WorkerRegistry | None = None,
        scheduler: Scheduler | None = None,
        hold_queue: HoldQueue | None = None,
        edge: EdgeBridge | None = None,
        council: CouncilEngine | None = None,
        client: WorkerClient | None = None,
        events: EventBus | None = None,
        ledger: RequestLedger | None = None,
        vault: CredentialVault | None = None,
        tools: ToolExecutor | None = None,
    ) -> None:
        self.settings = settings if settings is not None else Settings.from_env()
        # NOTE: every collaborator is resolved with an explicit ``is not None``
        # check.  Several of these objects define ``__len__`` and are therefore
        # *falsy when empty* -- using ``a or b`` would silently swap in a
        # different instance (and desynchronise the registry from the
        # scheduler).
        self.registry = (
            registry
            if registry is not None
            else WorkerRegistry(
                ttl_seconds=self.settings.worker_ttl_seconds,
                max_concurrency_per_worker=self.settings.max_concurrency_per_worker,
            )
        )
        self.scheduler = scheduler if scheduler is not None else Scheduler(self.settings, self.registry)
        self.hold_queue = (
            hold_queue
            if hold_queue is not None
            else HoldQueue(
                max_size=self.settings.hold_queue_max_size,
                ttl_seconds=self.settings.hold_request_ttl_seconds,
            )
        )
        self.events = events if events is not None else EventBus()
        self.ledger = ledger if ledger is not None else RequestLedger()
        self.edge = (
            edge
            if edge is not None
            else EdgeBridge(self.settings, vault=vault if vault is not None else get_vault(self.settings.credential_file))
        )
        self.client = client if client is not None else WorkerClient(self.settings)
        self.council = (
            council if council is not None else CouncilEngine(self.settings, self.registry, self.scheduler, self.client)
        )
        self.tools = tools
        self._capacity_event = asyncio.Event()
        self._background: list[asyncio.Task[Any]] = []
        self._closing = False
        self._started_at = now_s()
        self._inflight = 0
        self._tokens_generated = 0

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    async def startup(self) -> None:
        """Start the background maintenance loops."""
        if self._background:
            return
        self._closing = False
        self._background.append(asyncio.create_task(self._capacity_loop(), name="hc-capacity-loop"))
        self._background.append(asyncio.create_task(self._maintenance_loop(), name="hc-maintenance-loop"))
        LOGGER.info("orchestrator started", extra={"hc_workers": len(self.registry)})

    async def shutdown(self) -> None:
        """Stop background loops and release resources."""
        self._closing = True
        for task in self._background:
            task.cancel()
        for task in self._background:
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._background.clear()
        await self.edge.aclose()
        await self.client.aclose()
        LOGGER.info("orchestrator stopped")

    async def _capacity_loop(self) -> None:
        """Wake held requests as soon as capacity appears."""
        while not self._closing:
            try:
                if self.hold_queue.waiting_count() > 0 and self.registry.healthy_workers():
                    self._capacity_event.set()
                await asyncio.sleep(0.4)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - the loop must never die
                LOGGER.exception("capacity loop error")
                await asyncio.sleep(1.0)

    async def _maintenance_loop(self) -> None:
        """Expire dead workers, refresh deferred work, publish telemetry."""
        while not self._closing:
            try:
                expired = self.registry.sweep()
                for record in expired:
                    await self.events.emit(
                        EventType.WORKER_EXPIRED,
                        {"worker_id": record.worker_id, "provider": record.provider},
                    )
                self.hold_queue.expire()
                if self.hold_queue.deferred_count() and self.registry.healthy_workers():
                    await self._replay_deferred()
                await asyncio.sleep(2.0)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                LOGGER.exception("maintenance loop error")
                await asyncio.sleep(3.0)

    async def _replay_deferred(self) -> None:
        """Re-execute deferred requests now that capacity is back."""
        for entry in self.hold_queue.deferred():
            decision = self.scheduler.select(entry.request)
            if not decision.has_capacity:
                return
            self.hold_queue.claim(entry.request_id)
            asyncio.create_task(self._replay_one(entry))

    async def _replay_one(self, entry: Any) -> None:
        """Replay a deferred request in the background; result goes to ledger."""
        with log_context(request=entry.request_id, mode="replay"):
            try:
                completion = await self._execute_cluster(entry.request, None)
                self.ledger.complete(completion)
                await self.events.emit(
                    EventType.REQUEST_STATE,
                    {"request_id": entry.request_id, "state": "completed", "replayed": True},
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("deferred replay failed", extra={"hc_request": entry.request_id, "hc_error": str(exc)})
                self.ledger.set_state(entry.request_id, RequestState.FAILED, error=str(exc))
            finally:
                self.hold_queue.complete(entry.request_id, ChatCompletion(request_id=entry.request_id))

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    async def submit(self, request: ChatRequest) -> ChatCompletion:
        """Run a request to completion (non-streaming)."""
        stream = RequestStream(request_id=request.request_id)
        self.ledger.record_request(request)
        task = asyncio.create_task(self._execute(request, stream), name=f"hc-req-{request.request_id}")
        task.add_done_callback(_consume_task_result)
        timeout = request.deadline_seconds or self.settings.request_timeout_seconds
        try:
            return await asyncio.wait_for(asyncio.shield(stream.future), timeout=timeout + 5.0)
        except asyncio.TimeoutError as exc:
            stream.reject(exc)
            raise NoCapacityError(
                "request timed out before the cluster could answer",
                details={"request_id": request.request_id},
            ) from exc

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        """Run a request and yield its events as they happen."""
        stream = RequestStream(request_id=request.request_id)
        self.ledger.record_request(request)
        task = asyncio.create_task(self._execute(request, stream), name=f"hc-stream-{request.request_id}")
        task.add_done_callback(_consume_task_result)
        try:
            async for event in stream.events():
                yield event
        finally:
            if task.done():
                _consume_task_result(task)

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #
    async def _execute(self, request: ChatRequest, stream: RequestStream | None) -> ChatCompletion:
        """Dispatch a request according to its (resolved) mode."""
        with log_context(request=request.request_id, mode=request.mode.value):
            try:
                completion = await self._dispatch(request, stream)
            except HoldTimeoutError as exc:
                self.ledger.set_state(request.request_id, RequestState.DEFERRED, error=str(exc))
                completion = ChatCompletion(
                    request_id=request.request_id,
                    error="deferred",
                    finish_reason="deferred",
                    raw={"reason": str(exc), "replayable": True},
                )
                if stream is not None:
                    await stream.emit(
                        "deferred",
                        {
                            "request_id": request.request_id,
                            "message": (
                                "All workers are busy; your request is queued and will be answered "
                                f"automatically. Pull /v1/requests/{request.request_id} for the result."
                            ),
                            "reason": str(exc),
                        },
                    )
                    stream.resolve(completion)
                self.ledger.complete(completion)
                return completion
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("request failed", extra={"hc_request": request.request_id})
                self.ledger.set_state(request.request_id, RequestState.FAILED, error=str(exc))
                completion = ChatCompletion(
                    request_id=request.request_id,
                    error=str(exc),
                    finish_reason="error",
                )
                if stream is not None:
                    await stream.emit("error", {"message": str(exc), "type": type(exc).__name__})
                    stream.resolve(completion)
                self.ledger.complete(completion)
                return completion
            else:
                self.ledger.complete(completion)
                if stream is not None:
                    await stream.emit("done", completion.model_dump(mode="json"))
                    stream.resolve(completion)
                return completion

    async def _dispatch(self, request: ChatRequest, stream: RequestStream | None) -> ChatCompletion:
        """Resolve the mode and route to the matching pipeline."""
        mode = self.scheduler.classify(request)
        request.mode = mode
        self._inflight += 1
        try:
            if stream is not None:
                await stream.emit(
                    "start",
                    {
                        "mode": mode.value,
                        "model": request.model,
                        "max_tokens": request.max_tokens,
                        "tier": "auto",
                        "ts": now_s(),
                    },
                )
            if mode is RequestMode.COUNCIL:
                return await self._run_council(request, stream)
            if mode is RequestMode.SANDBOX and self.tools is not None and self.settings.sandbox_enabled:
                return await self._run_sandbox(request, stream)
            if mode is RequestMode.FAST and self.settings.speculative_enabled:
                return await self._run_speculative(request, stream)
            return await self._execute_cluster(request, stream)
        finally:
            self._inflight = max(0, self._inflight - 1)

    # ------------------------------------------------------------------ #
    async def _execute_cluster(
        self,
        request: ChatRequest,
        stream: RequestStream | None,
        *,
        emit_tokens: bool = True,
        sync: bool = False,
    ) -> ChatCompletion:
        """Serve a request from the pooled GPU workers, with failover."""
        mode = self.scheduler.classify(request)
        excluded: set[str] = set()
        attempts = 0
        last_error: str | None = None

        while attempts < 4:
            attempts += 1
            decision = self.scheduler.select(request, mode=mode, exclude=excluded)
            if decision.primary is None:
                if not self.settings.hold_enabled:
                    raise NoCapacityError(
                        "no worker has free capacity right now",
                        details={"reason": decision.reason},
                    )
                decision = await self._await_capacity(request, stream, decision, excluded)
                if decision.primary is None:
                    raise NoCapacityError(
                        "no worker has free capacity right now",
                        details={"reason": decision.reason, "held": True},
                    )

            record = self.registry.get(decision.primary.worker_id)
            if record is None:
                excluded.add(decision.primary.worker_id)
                continue
            if not self.registry.claim(record.worker_id):
                excluded.add(record.worker_id)
                continue

            self.ledger.set_state(request.request_id, RequestState.DISPATCHED, worker_id=record.worker_id)
            if stream is not None:
                await stream.emit(
                    "dispatch",
                    {
                        "worker_id": record.worker_id,
                        "provider": record.provider,
                        "model": record.model_loaded,
                        "ttft_estimate_ms": round(decision.primary.ttft_ms, 2),
                        "score": round(decision.primary.score, 2),
                    },
                )
            if decision.requires_model_load:
                await self._ensure_model(record, decision.target_model)
            try:
                if sync:
                    completion = await self._run_on_worker_sync(record, request)
                else:
                    completion = await self._run_on_worker(record, request, stream, emit_tokens=emit_tokens)
                self._tokens_generated += completion.tokens
                return completion
            except WorkerUnreachableError as exc:
                last_error = str(exc)
                excluded.add(record.worker_id)
                self.registry.mark_lost(record.worker_id, reason=str(exc))
                await self.events.emit(
                    EventType.WORKER_LOST,
                    {"worker_id": record.worker_id, "error": str(exc)},
                )
                LOGGER.warning(
                    "worker failed, failing over",
                    extra={"hc_worker": record.worker_id, "hc_error": str(exc)},
                )
            finally:
                self.registry.release(record.worker_id)
                self._capacity_event.set()

        raise NoCapacityError(
            "every candidate worker failed",
            details={"last_error": last_error, "attempts": attempts},
        )

    async def _ensure_model(self, record: WorkerRecord, model_name: str | None) -> bool:
        """Hot-load the model the scheduler decided on (best effort).

        The VRAM auto-scaler (milestone 3) will drive model selection across the
        whole pool; this keeps a *single* request honest: if the scheduler
        picked a worker because it can serve a bigger model, the worker is
        actually told to load it before the request runs.
        """
        if not model_name:
            return False
        if record.model_loaded and record.model_loaded.lower() == model_name.lower():
            return False
        spec = find_model(self.scheduler.catalog, model_name)
        if spec is None:
            return False
        try:
            result = await self.client.load_model(record, spec)
        except WorkerUnreachableError as exc:
            LOGGER.warning(
                "model hot-load failed; continuing with the resident model",
                extra={"hc_worker": record.worker_id, "hc_model": model_name, "hc_error": str(exc)},
            )
            return False
        if not result.loaded:
            LOGGER.warning(
                "worker refused the model",
                extra={"hc_worker": record.worker_id, "hc_model": model_name, "hc_error": result.error},
            )
            return False
        self.registry.set_model(record.worker_id, spec.name, family=spec.family, params_b=spec.params_b)
        await self.events.emit(
            EventType.MODEL_CHANGED,
            {"worker_id": record.worker_id, "model": spec.name, "load_ms": result.load_ms},
        )
        return True

    async def _run_on_worker_sync(
        self, record: WorkerRecord, request: ChatRequest
    ) -> ChatCompletion:
        """Run a *blocking* completion on a worker (used by the tool loop)."""
        started = mono_ms()
        completion = await self.client.chat(record, request)
        completion.worker_id = completion.worker_id or record.worker_id
        completion.model = completion.model or record.model_loaded
        completion.total_ms = completion.total_ms or (mono_ms() - started)
        self.registry.record_latency(
            record.worker_id,
            ttft_ms=completion.ttft_ms,
            total_ms=completion.total_ms,
            tokens=completion.tokens,
            tokens_per_sec=completion.tokens_per_sec,
        )
        return completion

    async def _run_sandbox(
        self, request: ChatRequest, stream: RequestStream | None
    ) -> ChatCompletion:
        """Agentic tool loop: the model calls tools, we execute them safely."""
        assert self.tools is not None
        schemas = self.tools.list_tools()
        question = request.to_prompt()
        system_prompt = build_tool_prompt(schemas, question)

        messages: list[ChatMessage] = [ChatMessage(role="system", content=system_prompt)]
        for message in request.messages:
            if message.role == "system":
                continue
            messages.append(message)

        transcript: list[dict[str, Any]] = []
        final_text = ""
        raw = ""
        rounds = max(1, self.settings.max_tool_rounds)
        started = mono_ms()

        for round_index in range(rounds):
            probe = ChatRequest(
                request_id=f"{request.request_id}-tool-{round_index}",
                messages=list(messages),
                mode=RequestMode.DEEP,
                max_tokens=min(request.max_tokens, 900),
                temperature=request.temperature,
                stream=False,
            )
            completion = await self._execute_cluster(probe, None, sync=True)
            raw = completion.text or ""
            call = parse_tool_call(raw)
            if call is None:
                final_text = raw.strip()
                break
            if "final" in call:
                final_text = str(call.get("final") or "").strip()
                break
            tool_name = str(call.get("tool") or "")
            # Tool execution is blocking (subprocess / network) -> keep the
            # event loop free so other requests keep streaming.
            result = await asyncio.to_thread(self.tools.execute_call, call)
            transcript.append(
                {
                    "round": round_index,
                    "tool": tool_name,
                    "ok": result.ok,
                    "output": result.output[:2000],
                    "error": result.error,
                    "duration_ms": round(result.duration_ms, 1),
                }
            )
            if stream is not None:
                await stream.emit(
                    "tool",
                    {
                        "round": round_index,
                        "tool": tool_name,
                        "ok": result.ok,
                        "output": result.output[:2000],
                        "error": result.error,
                    },
                )
            messages.append(ChatMessage(role="assistant", content=raw[:2000]))
            messages.append(
                ChatMessage(
                    role="tool",
                    content=f"[tool result: {tool_name}]\n{result.output[:4000]}",
                )
            )

        if not final_text:
            final_text = raw.strip()
        if stream is not None and final_text:
            for piece in _chunk_text(final_text):
                await stream.emit("token", {"text": piece, "source": ChunkSource.SYNTHESIS.value, "draft": False})
        return ChatCompletion(
            request_id=request.request_id,
            text=final_text,
            model=None,
            worker_id=None,
            tier=WorkerTier.CLUSTER,
            source=ChunkSource.SYNTHESIS,
            mode=RequestMode.SANDBOX,
            tokens=max(1, len(final_text) // 4),
            total_ms=mono_ms() - started,
            verified=True,
            finish_reason="stop",
            raw={"tools": transcript, "rounds": len(transcript)},
            council=None,
        )

    async def _run_on_worker(
        self,
        record: WorkerRecord,
        request: ChatRequest,
        stream: RequestStream | None,
        *,
        emit_tokens: bool = True,
    ) -> ChatCompletion:
        """Stream one completion from a worker and measure it."""
        started = mono_ms()
        first_token_at: float | None = None
        parts: list[str] = []
        tokens = 0
        async for chunk in self.client.stream_chat(record, request):
            if not chunk.text:
                continue
            if first_token_at is None:
                first_token_at = mono_ms()
            parts.append(chunk.text)
            tokens += 1
            if stream is not None and emit_tokens:
                await stream.emit(
                    "token",
                    {
                        "text": chunk.text,
                        "source": chunk.source.value,
                        "draft": chunk.draft,
                    },
                )
        ttft_ms = (first_token_at - started) if first_token_at is not None else (mono_ms() - started)
        total_ms = mono_ms() - started
        tokens_per_sec = (tokens / (total_ms / 1000.0)) if total_ms > 0 else 0.0
        self.registry.record_latency(
            record.worker_id,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            tokens=tokens,
            tokens_per_sec=tokens_per_sec,
        )
        if stream is not None:
            self.ledger.set_state(request.request_id, RequestState.STREAMING, worker_id=record.worker_id)
        return ChatCompletion(
            request_id=request.request_id,
            text="".join(parts),
            model=record.model_loaded,
            worker_id=record.worker_id,
            tier=WorkerTier.CLUSTER,
            source=ChunkSource.CLUSTER,
            mode=self.scheduler.classify(request),
            tokens=tokens,
            usage={"prompt_tokens": 0, "completion_tokens": tokens, "total_tokens": tokens},
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            tokens_per_sec=tokens_per_sec,
            finish_reason="stop",
        )

    # ------------------------------------------------------------------ #
    async def _await_capacity(
        self,
        request: ChatRequest,
        stream: RequestStream | None,
        decision: SchedulerDecision,
        excluded: set[str],
    ) -> SchedulerDecision:
        """Park the request until a worker frees up (or the deadline hits)."""
        try:
            entry = self.hold_queue.enqueue(
                request,
                deadline_seconds=request.deadline_seconds or self.settings.hold_request_ttl_seconds,
            )
        except Exception as exc:  # queue full
            raise NoCapacityError(
                "cluster is saturated and the hold queue is full",
                details={"reason": str(exc)},
            ) from exc

        held_started = mono_ms()
        if stream is not None:
            await stream.emit(
                "hold",
                {
                    "reason": decision.reason or "no capacity",
                    "position": self.hold_queue.waiting_count(),
                    "deadline_in_s": round(entry.seconds_to_deadline, 1),
                    "workers": len(self.registry),
                },
            )
            self.ledger.set_state(request.request_id, RequestState.HELD)
        await self.events.emit(
            EventType.QUEUE_UPDATED,
            {"queued": self.hold_queue.waiting_count(), "reason": "no_capacity"},
        )

        try:
            while True:
                fresh = self.scheduler.select(request, mode=request.mode, exclude=excluded)
                if fresh.has_capacity:
                    if stream is not None:
                        await stream.emit(
                            "resume",
                            {"held_ms": round(mono_ms() - held_started, 1), "worker_id": fresh.primary.worker_id},
                        )
                    return fresh
                remaining = entry.seconds_to_deadline
                if remaining <= 0:
                    self.hold_queue.defer(entry.request_id, "deadline exceeded while waiting for capacity")
                    raise HoldTimeoutError(
                        "no capacity became available before the deadline",
                        details={"request_id": request.request_id, "held_ms": round(mono_ms() - held_started, 1)},
                    )
                self._capacity_event.clear()
                try:
                    await asyncio.wait_for(self._capacity_event.wait(), timeout=min(remaining, 0.4))
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            self.hold_queue.fail(request.request_id, "client disconnected")
            raise
        except HoldTimeoutError:
            raise
        except Exception as exc:  # noqa: BLE001
            self.hold_queue.fail(request.request_id, str(exc))
            raise

    # ------------------------------------------------------------------ #
    async def _run_speculative(
        self, request: ChatRequest, stream: RequestStream | None
    ) -> ChatCompletion:
        """Edge-draft + cluster-verify cascading."""
        candidates = self.edge.candidates()
        choice = self.scheduler.select_edge(candidates, request, mode=request.mode)
        if choice is None:
            LOGGER.info("no edge endpoint available; falling back to cluster-only")
            return await self._execute_cluster(request, stream)

        credential = self.edge.vault.get(choice.provider, choice.label)
        if credential is None:
            return await self._execute_cluster(request, stream)

        draft_parts: list[str] = []
        draft_error: str | None = None

        async def _draft() -> None:
            nonlocal draft_error
            try:
                async for text in self.edge.stream(credential, request):
                    draft_parts.append(text)
                    if stream is not None:
                        await stream.emit(
                            "token",
                            {"text": text, "source": ChunkSource.EDGE.value, "draft": True},
                        )
            except Exception as exc:  # noqa: BLE001 - the draft is best effort
                draft_error = str(exc)
                LOGGER.warning("edge draft failed", extra={"hc_error": str(exc), "hc_provider": choice.provider})

        draft_job = asyncio.create_task(_draft(), name="hc-draft")

        # The cluster verifies in parallel -- but only if it has capacity right
        # now, otherwise the edge answer stands on its own.
        cluster_completion: ChatCompletion | None = None
        decision = self.scheduler.select(request, mode=request.mode)
        if decision.has_capacity:
            try:
                cluster_completion = await asyncio.wait_for(
                    self._execute_cluster(request, None, emit_tokens=False),
                    timeout=VERIFICATION_BUDGET_SECONDS,
                )
            except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001
                LOGGER.info("cluster verification unavailable", extra={"hc_error": str(exc)})
                cluster_completion = None

        try:
            await asyncio.wait_for(asyncio.shield(draft_job), timeout=VERIFICATION_BUDGET_SECONDS)
        except asyncio.TimeoutError:
            draft_job.cancel()
        except Exception:  # noqa: BLE001
            pass

        draft_text = "".join(draft_parts)
        started = mono_ms()
        if stream is not None:
            await stream.emit(
                "edge",
                {
                    "provider": choice.provider,
                    "label": choice.label,
                    "model": choice.model,
                    "chars": len(draft_text),
                    "error": draft_error,
                },
            )

        if cluster_completion is not None and cluster_completion.text.strip():
            score = similarity(draft_text, cluster_completion.text)
            if score >= VERIFICATION_THRESHOLD:
                if stream is not None:
                    await stream.emit(
                        "verify",
                        {
                            "verdict": "confirmed",
                            "similarity": round(score, 3),
                            "verified_by": cluster_completion.worker_id,
                            "model": cluster_completion.model,
                        },
                    )
                cluster_completion.verified = True
                cluster_completion.verified_by = cluster_completion.worker_id
                cluster_completion.raw = {
                    "edge_provider": choice.provider,
                    "edge_chars": len(draft_text),
                    "similarity": round(score, 3),
                }
                cluster_completion.ttft_ms = max(1.0, (mono_ms() - started))
                return cluster_completion
            # Divergence: the cluster is authoritative, tell the client to swap.
            if stream is not None:
                await stream.emit(
                    "revise",
                    {
                        "reason": "cluster diverged from edge draft",
                        "similarity": round(score, 3),
                        "text": cluster_completion.text,
                    },
                )
            cluster_completion.verified = True
            cluster_completion.verified_by = cluster_completion.worker_id
            cluster_completion.raw = {
                "edge_provider": choice.provider,
                "edge_chars": len(draft_text),
                "similarity": round(score, 3),
                "revised": True,
            }
            return cluster_completion

        # Edge-only answer (no cluster capacity / cluster failed).
        if not draft_text.strip():
            raise NoCapacityError(
                "no execution target produced an answer",
                details={"edge_error": draft_error},
            )
        if stream is not None:
            await stream.emit(
                "verify",
                {"verdict": "unverified", "reason": "cluster unavailable", "verified_by": None},
            )
        elapsed_ms = mono_ms() - started
        return ChatCompletion(
            request_id=request.request_id,
            text=draft_text,
            model=choice.model,
            worker_id=None,
            tier=WorkerTier.EDGE,
            source=ChunkSource.EDGE,
            mode=request.mode,
            tokens=max(1, len(draft_text) // 4),
            total_ms=elapsed_ms,
            tokens_per_sec=(len(draft_text) / 4) / max(elapsed_ms / 1000.0, 0.001),
            verified=False,
            finish_reason="stop",
            raw={"edge_provider": choice.provider, "label": choice.label, "error": draft_error},
        )

    # ------------------------------------------------------------------ #
    async def _run_council(self, request: ChatRequest, stream: RequestStream | None) -> ChatCompletion:
        """Multi-agent deliberation with cross-critique and synthesis."""
        members = self.council.select_members(request)
        if not members:
            if self.settings.hold_enabled:
                await self._await_capacity(request, stream, self.scheduler.select(request), set())
                members = self.council.select_members(request)
            if not members:
                raise NoCapacityError("no workers available for council mode")

        async def _sink(text: str, worker_id: str) -> None:
            if stream is not None:
                await stream.emit(
                    "token",
                    {"text": text, "source": ChunkSource.SYNTHESIS.value, "draft": False},
                )

        outcome: CouncilOutcome = await self.council.deliberate(request, members=members, sink=_sink)
        if stream is not None:
            await stream.emit("council", outcome.to_dict())
        if not outcome.final_text.strip():
            raise NoCapacityError(
                "the council could not reach an answer",
                details={"error": outcome.error},
            )
        return ChatCompletion(
            request_id=request.request_id,
            text=outcome.final_text,
            model=None,
            worker_id=outcome.synthesis_by,
            tier=WorkerTier.CLUSTER,
            source=ChunkSource.SYNTHESIS,
            mode=RequestMode.COUNCIL,
            tokens=max(1, len(outcome.final_text) // 4),
            total_ms=outcome.total_ms,
            verified=outcome.consensus >= 0.35,
            finish_reason="stop",
            council=outcome.to_dict(),
        )

    # ------------------------------------------------------------------ #
    # Telemetry
    # ------------------------------------------------------------------ #
    def status(self) -> ClusterStatus:
        """Aggregate cluster status for the mobile dashboard."""
        totals = self.registry.totals()
        workers = self.registry.snapshot()
        edge_healthy = self.edge.healthy_providers()
        return ClusterStatus(
            cluster_name=self.settings.cluster_name,
            instance_id=self.settings.instance_id,
            ts=now_s(),
            healthy=bool(totals["healthy"]) or bool(edge_healthy),
            worker_count=totals["workers"],
            healthy_workers=totals["healthy"],
            draining_workers=totals["draining"],
            total_vram_gb=totals["total_vram_gb"],
            free_vram_gb=totals["free_vram_gb"],
            total_tflops=totals["total_tflops"],
            active_model=self.scheduler.active_model(),
            model_catalog_size=len(self.scheduler.catalog.models),
            queued_requests=self.hold_queue.waiting_count(),
            held_requests=len(self.hold_queue),
            in_flight=self._inflight,
            tokens_per_sec_aggregate=totals["tokens_per_sec"],
            edge_providers=self.edge.provider_names(),
            edge_healthy=edge_healthy,
            workers=workers,
            mode=RequestMode.AUTO,
            autopilot=False,
            uptime_seconds=round(now_s() - self._started_at, 1),
        )

    def request_status(self, request_id: str) -> Any:
        """Look up a request in the ledger."""
        return self.ledger.get(request_id)

    def request_result(self, request_id: str) -> ChatCompletion | None:
        """Fetch a finished request's result."""
        return self.ledger.result(request_id)

    # ------------------------------------------------------------------ #
    async def notify_worker_changed(self, worker_id: str) -> None:
        """Called by the API layer when a worker registers/changes."""
        record = self.registry.get(worker_id)
        self._capacity_event.set()
        if record is not None:
            await self.events.emit(
                EventType.WORKER_REGISTERED,
                {
                    "worker_id": worker_id,
                    "provider": record.provider,
                    "vram_gb": record.vram_total_gb,
                    "model": record.model_loaded,
                },
            )


def _chunk_text(text: str, size: int = 48) -> list[str]:
    """Split a final answer into word-aligned chunks for smooth streaming."""
    words = text.split(" ")
    chunks: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}" if current else word
        if len(candidate) >= size and current:
            chunks.append(current + " ")
            current = word
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _drain_future(future: asyncio.Future[Any]) -> None:
    """Retrieve a future's exception so asyncio does not warn about it."""
    if not future.cancelled():
        future.exception()


def _consume_task_result(task: asyncio.Task[Any]) -> None:
    """Swallow background task exceptions so asyncio stays quiet."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        LOGGER.warning("background task failed", extra={"hc_error": str(exc)})


__all__ = ["Orchestrator", "RequestStream", "VERIFICATION_THRESHOLD"]
