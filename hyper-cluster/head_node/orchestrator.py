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
from head_node.auto_spawner import AutoSpawner
from head_node.council import CouncilEngine, CouncilOutcome, similarity
from head_node.edge_bridge import EdgeBridge
from head_node.events import EventBus
from head_node.hold_queue import HoldQueue
from head_node.model_selector import ModelSelector, ScalingDecision
from head_node.request_store import RequestLedger
from head_node.scheduler import Scheduler, SchedulerDecision
from head_node.worker_client import WorkerClient
from head_node.worker_registry import WorkerRecord, WorkerRegistry
from memory.vector_memory import VectorMemory
from tools.sandbox import ToolExecutor, build_tool_prompt, parse_tool_call
from training.autopilot_evolver import AutopilotEvolver

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
        selector: ModelSelector | None = None,
        spawner: AutoSpawner | None = None,
        memory: VectorMemory | None = None,
        autopilot: AutopilotEvolver | None = None,
        notifier: Any | None = None,
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
        self.selector = (
            selector
            if selector is not None
            else ModelSelector(self.settings, self.registry, self.scheduler.catalog)
        )
        self.spawner = (
            spawner
            if spawner is not None
            else AutoSpawner(self.settings, self.registry, vault=self.edge.vault)
        )
        self.memory = (
            memory
            if memory is not None
            else (
                VectorMemory(
                    self.settings.memory_dir,
                    chunk_size=self.settings.memory_chunk_size,
                    chunk_overlap=self.settings.memory_chunk_overlap,
                    max_chunks=self.settings.memory_max_chunks,
                )
                if self.settings.memory_enabled
                else None
            )
        )
        self.autopilot = (
            autopilot
            if autopilot is not None
            else self._build_autopilot()
        )
        self.notifier = (
            notifier
            if notifier is not None
            else self._build_notifier()
        )
        self._capacity_event = asyncio.Event()
        self._background: list[asyncio.Task[Any]] = []
        self._closing = False
        self._started_at = now_s()
        self._inflight = 0
        self._tokens_generated = 0

    # ------------------------------------------------------------------ #
    # The sleep-time engine
    # ------------------------------------------------------------------ #
    def _build_autopilot(self) -> AutopilotEvolver | None:
        """Wire the sleep-time self-evolution engine (Milestone 4)."""
        if not self.settings.evolution_enabled:
            return None
        from training.checkpoint_streamer import CheckpointStreamer
        from training.distributed_trainer import DistributedTrainer
        from training.prompt_evolver import PromptEvolver, load_prompt_from_memory
        from training.self_play import SelfPlayEngine

        trainer = DistributedTrainer(
            checkpoint_dir=self.settings.evolution_checkpoint_dir,
            dim=self.settings.evolution_train_dim,
            rank=self.settings.evolution_train_rank,
            vocab_size=self.settings.evolution_train_vocab,
            learning_rate=self.settings.evolution_train_learning_rate,
            steps=self.settings.evolution_train_steps,
            shard_fn=self._shard_dispatch,
        )
        self_play = SelfPlayEngine(self.registry, self.client)
        streamer: CheckpointStreamer | None = None
        if self.settings.drive_token or self.settings.hub_token:
            from memory.sync import DriveClient, HubClient

            drive = (
                DriveClient(
                    token=self.settings.drive_token,
                    api_base=self.settings.drive_api_base,
                    upload_base=self.settings.drive_upload_base,
                )
                if self.settings.drive_token
                else None
            )
            hub = (
                HubClient(token=self.settings.hub_token, base_url=self.settings.hub_base_url)
                if self.settings.hub_token and self.settings.hub_repo
                else None
            )
            streamer = CheckpointStreamer(
                drive=drive,
                hub=hub,
                drive_folder=self.settings.drive_folder,
                hub_repo=self.settings.hub_repo,
            )
        prompt_evolver = PromptEvolver(
            self.registry,
            self.client,
            baseline_prompt=load_prompt_from_memory(self.memory),
        )
        return AutopilotEvolver(
            self.settings,
            self.registry,
            self.client,
            memory=self.memory,
            trainer=trainer,
            self_play=self_play,
            streamer=streamer,
            prompt_evolver=prompt_evolver,
            events=self.events,
        )

    def _build_notifier(self) -> Any | None:
        """Wire the phone alerts (Milestone 5)."""
        if not self.settings.notifier_enabled:
            return None
        from mobile.notifier import Notifier

        notifier = Notifier(
            telegram_token=self.settings.notifier_telegram_token,
            telegram_chat_id=self.settings.notifier_telegram_chat_id,
            webhook_url=self.settings.notifier_webhook_url,
            webhook_secret=self.settings.notifier_webhook_secret,
            events=self.settings.notifier_events or None,
            min_interval_seconds=self.settings.notifier_min_interval_seconds,
        )
        return notifier if notifier.enabled else None

    def _shard_dispatch(self, payload: dict[str, Any]) -> Any:
        """Send one training shard to a worker (round-robin over the mesh)."""
        workers = self.registry.healthy_workers()
        if not workers:
            raise NoCapacityError("no worker is available for the training shard")
        index = getattr(self, "_shard_cursor", 0)
        self._shard_cursor = (index + 1) % len(workers)
        record = workers[index % len(workers)]
        return self.client.train_shard(record, payload)

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
        self._background.append(asyncio.create_task(self._autoscale_loop(), name="hc-autoscale-loop"))
        self._background.append(asyncio.create_task(self._spawner_loop(), name="hc-spawner-loop"))
        if self.autopilot is not None:
            self.autopilot.start()
        if self.notifier is not None:
            self._background.append(
                asyncio.create_task(
                    self.notifier.run(self.events.subscribe, self.events.unsubscribe),
                    name="hc-notifier-loop",
                )
            )
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
        if self.autopilot is not None:
            await self.autopilot.aclose()
        if self.notifier is not None:
            await self.notifier.aclose()
        await self.edge.aclose()
        await self.client.aclose()
        await self.spawner.aclose()
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

    async def _autoscale_loop(self) -> None:
        """Keep the models on the pool in line with the live demand."""
        while not self._closing:
            try:
                await asyncio.sleep(max(2.0, float(self.settings.autoscaler_min_dwell_seconds) / 4.0))
                if not self.settings.autoscaler_enabled:
                    continue
                await self.autoscale()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - the loop must never die
                LOGGER.exception("autoscaler loop error")
                await asyncio.sleep(3.0)

    async def autoscale(self) -> list[ScalingDecision]:
        """Compute and apply the current scaling plan.

        The decision logic lives in :class:`ModelSelector`; this method is the
        part that *executes* it: the worker is told to hot-swap its model and
        the registry is updated so the scheduler sees the new resident model.
        """
        decisions = self.selector.plan()
        applied: list[ScalingDecision] = []
        for decision in decisions:
            if not decision.changes_model or not decision.model:
                continue
            record = self.registry.get(decision.worker_id)
            if record is None:
                continue
            spec = find_model(self.selector.catalog, decision.model)
            if spec is None:
                continue
            loaded = await self._ensure_model(record, spec.name)
            if loaded:
                applied.append(decision)
                self.selector.mark_applied(decision.worker_id)
                await self.events.emit(
                    EventType.MODEL_CHANGED,
                    {
                        "worker_id": decision.worker_id,
                        "model": spec.name,
                        "previous_model": decision.previous_model,
                        "direction": decision.direction,
                        "reason": decision.reason,
                    },
                )
        return applied

    async def _spawner_loop(self) -> None:
        """Provision and rotate compute without human intervention."""
        while not self._closing:
            try:
                await asyncio.sleep(max(2.0, float(self.settings.spawner_tick_seconds)))
                if not self.settings.spawner_enabled:
                    continue
                actions = await self.spawner.tick()
                for action in actions:
                    await self.events.emit(
                        EventType.CAPACITY,
                        {
                            "kind": action.kind,
                            "worker_id": action.worker_id,
                            "replacement": action.replacement,
                            "provider": action.provider,
                            "detail": action.detail,
                            "ok": action.ok,
                        },
                    )
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - the loop must never die
                LOGGER.exception("spawner loop error")
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
        self._user_activity(request)
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
        self._user_activity(request)
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
    # The user always wins
    # ------------------------------------------------------------------ #
    def _user_activity(self, request: ChatRequest) -> None:
        """Called for every incoming request.

        Two things happen here, both on the hot path and both cheap:

        * the sleep-time engine is **preempted** -- the training stops at its
          next checkpoint and the workers come back to the user;
        * the conversation is **remembered** -- the durable facts are
          distilled into the memory and the retrieval context for this
          request is attached to it.
        """
        if self.autopilot is not None:
            self.autopilot.notify_activity("user request")
        if self.memory is None:
            return
        try:
            self.memory.distill_facts(request.messages, metadata={"request_id": request.request_id})
        except Exception:  # noqa: BLE001 - the memory must never break a request
            LOGGER.warning("memory distillation failed", extra={"hc_request": request.request_id})

    def _memory_context(self, request: ChatRequest) -> str:
        """The retrieval context for a request (empty when there is none)."""
        if self.memory is None or not self.settings.memory_enabled:
            return ""
        question = ""
        for message in reversed(request.messages):
            if message.role == "user":
                question = message.content
                break
        if not question:
            return ""
        try:
            return self.memory.context(
                question,
                k=self.settings.memory_context_top_k,
                max_chars=self.settings.memory_context_chars,
            )
        except Exception:  # noqa: BLE001
            LOGGER.warning("memory retrieval failed", extra={"hc_request": request.request_id})
            return ""

    def _with_memory(self, request: ChatRequest) -> ChatRequest:
        """Attach the retrieved memory to the request (as a system message)."""
        context = self._memory_context(request)
        if not context:
            return request
        messages = list(request.messages)
        system = next((message for message in messages if message.role == "system"), None)
        if system is not None:
            system = system.model_copy(update={"content": f"{system.content}\n\n{context}"})
            messages = [system if message is system else message for message in messages]
        else:
            messages.insert(0, ChatMessage(role="system", content=context))
        return request.model_copy(update={"messages": messages})

    def _remember(self, request: ChatRequest, completion: ChatCompletion) -> None:
        """Store the exchange in the memory (the cluster learns from it)."""
        if self.memory is None or not completion.text:
            return
        question = ""
        for message in reversed(request.messages):
            if message.role == "user":
                question = message.content
                break
        if not question:
            return
        try:
            self.memory.record_exchange(
                question,
                completion.text,
                metadata={"request_id": request.request_id, "mode": request.mode.value},
            )
        except Exception:  # noqa: BLE001
            LOGGER.warning("memory write failed", extra={"hc_request": request.request_id})

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
                self._remember(request, completion)
                if stream is not None:
                    await stream.emit("done", completion.model_dump(mode="json"))
                    stream.resolve(completion)
                return completion

    async def _dispatch(self, request: ChatRequest, stream: RequestStream | None) -> ChatCompletion:
        """Resolve the mode and route to the matching pipeline."""
        mode = self.scheduler.classify(request)
        request.mode = mode
        self.selector.record_demand(mode)
        request = self._with_memory(request)
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
            # No free API is reachable: draft *inside* the cluster with a small
            # resident model and verify with the big one.
            local = await self._run_local_speculative(request, stream)
            if local is not None:
                return local
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
    async def _run_local_speculative(
        self, request: ChatRequest, stream: RequestStream | None
    ) -> ChatCompletion | None:
        """Draft inside the cluster: a small model first, the big model verifies.

        The pairing is computed by the model selector from the *live* pool: a
        draft model that is at least 4x smaller and measurably faster than the
        verifier.  Returns ``None`` when no usable pair exists.
        """
        pair = self.selector.draft_pair()
        if pair is None or not pair.draft_model or not pair.verifier_model:
            return None
        draft_record = self.registry.get(pair.draft_worker)
        verifier_record = self.registry.get(pair.verifier_worker)
        if draft_record is None or verifier_record is None:
            return None
        if not draft_record.healthy or not verifier_record.healthy:
            return None

        draft_parts: list[str] = []
        draft_error: str | None = None

        async def _draft() -> None:
            nonlocal draft_error
            try:
                async for chunk in self.client.stream_chat(draft_record, request):
                    draft_parts.append(chunk.text or "")
                    if stream is not None:
                        await stream.emit(
                            "token",
                            {"text": chunk.text or "", "source": ChunkSource.DRAFT.value, "draft": True},
                        )
            except Exception as exc:  # noqa: BLE001 - the draft is best effort
                draft_error = str(exc)
                LOGGER.warning(
                    "cluster draft failed",
                    extra={"hc_error": str(exc), "hc_worker": draft_record.worker_id},
                )

        draft_job = asyncio.create_task(_draft(), name="hc-local-draft")

        verifier_completion: ChatCompletion | None = None
        try:
            verifier_completion = await asyncio.wait_for(
                self._execute_cluster(request, None, emit_tokens=False),
                timeout=VERIFICATION_BUDGET_SECONDS,
            )
        except Exception as exc:  # noqa: BLE001 - the verifier is best effort
            LOGGER.info("cluster verifier unavailable", extra={"hc_error": str(exc)})
            verifier_completion = None

        try:
            await asyncio.wait_for(asyncio.shield(draft_job), timeout=VERIFICATION_BUDGET_SECONDS)
        except asyncio.TimeoutError:
            draft_job.cancel()
        except Exception:  # noqa: BLE001
            pass

        draft_text = "".join(draft_parts)
        if stream is not None:
            await stream.emit(
                "edge",
                {
                    "provider": "cluster",
                    "label": pair.draft_worker,
                    "model": pair.draft_model,
                    "chars": len(draft_text),
                    "error": draft_error,
                    "paired_with": pair.verifier_model,
                },
            )

        if verifier_completion is not None and verifier_completion.text.strip():
            score = similarity(draft_text, verifier_completion.text)
            verified = score >= VERIFICATION_THRESHOLD
            if stream is not None:
                await stream.emit(
                    "verify",
                    {
                        "verdict": "confirmed" if verified else "revised",
                        "similarity": round(score, 3),
                        "verified_by": verifier_completion.worker_id,
                        "model": verifier_completion.model,
                        "draft_model": pair.draft_model,
                    },
                )
            verifier_completion.verified = True
            verifier_completion.verified_by = verifier_completion.worker_id
            verifier_completion.raw = {
                "draft_worker": pair.draft_worker,
                "draft_model": pair.draft_model,
                "verifier_model": pair.verifier_model,
                "similarity": round(score, 3),
                "draft_chars": len(draft_text),
                "estimated_speedup": pair.estimated_speedup,
            }
            return verifier_completion

        if not draft_text.strip():
            return None
        return ChatCompletion(
            request_id=request.request_id,
            text=draft_text,
            model=pair.draft_model,
            worker_id=pair.draft_worker,
            tier=WorkerTier.CLUSTER,
            source=ChunkSource.DRAFT,
            mode=request.mode,
            tokens=max(1, len(draft_text) // 4),
            verified=False,
            finish_reason="stop",
            raw={"draft_worker": pair.draft_worker, "error": draft_error},
        )

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
            autopilot=bool(self.autopilot is not None and self.autopilot.enabled),
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
