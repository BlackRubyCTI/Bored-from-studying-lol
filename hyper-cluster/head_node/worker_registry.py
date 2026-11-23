"""Authoritative registry of every worker in the mesh.

The registry is the single source of truth about *who is alive*, *how fast
they are*, *how much VRAM they still have* and *when their session expires*.
It is intentionally synchronous and lock-protected: every operation is O(1)
and never blocks the event loop, which lets the orchestrator call it from
inside hot request paths.

Session lifecycle
-----------------
``register`` -> ``READY`` (idle, fresh TTL)
``heartbeat`` -> refreshes TTL + telemetry
``drain`` -> stops receiving new work, finishes in-flight requests
``expire`` -> TTL elapsed without a heartbeat -> removed and reported

The 10-minute pre-emptive handoff is driven by :meth:`expiring_within`, which
the auto-spawner (milestone 3) polls to provision a replacement *before* the
session dies.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from core.ids import now_s
from core.logging_setup import get_logger
from core.protocol import (
    Capabilities,
    WorkerHeartbeat,
    WorkerHello,
    WorkerSnapshot,
    WorkerStatus,
    WorkerTier,
)

LOGGER = get_logger("head_node.registry")

#: Statuses in which a worker may receive new work.
SERVING_STATUSES = (WorkerStatus.IDLE, WorkerStatus.BUSY)


def _ewma(old: float, new: float, alpha: float) -> float:
    """Exponentially weighted moving average helper."""
    if old <= 0.0:
        return new
    return (1.0 - alpha) * old + alpha * new


def _node_throughput(caps: Capabilities) -> float:
    """The throughput the worker measured in its own benchmark."""
    try:
        return float(caps.extra.get("throughput_tps") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _node_factor(caps: Capabilities, hello: WorkerHello) -> float:
    """``tokens/s * B`` -- the hardware factor of the node.

    Together with the size of the model a node hosts, this lets the head
    node estimate how fast *another* model would run on the same hardware.
    """
    throughput = _node_throughput(caps)
    params = float(caps.extra.get("benchmark_params_b") or hello.model_params_b or 0.0)
    if throughput <= 0.0 or params <= 0.0:
        return 0.0
    return throughput * params


@dataclass
class WorkerRecord:
    """Live state of a single worker."""

    worker_id: str
    host: str = "127.0.0.1"
    port: int = 8081
    provider: str = "local"
    region: str = "unknown"
    tier: WorkerTier = WorkerTier.CLUSTER
    status: WorkerStatus = WorkerStatus.BOOTING
    draining: bool = False
    capabilities: Capabilities = field(default_factory=Capabilities)
    model_loaded: str | None = None
    model_family: str | None = None
    model_params_b: float | None = None
    vram_total_gb: float = 0.0
    vram_free_gb: float = 0.0
    compute_tflops: float = 0.0
    tokens_per_sec: float = 0.0
    #: ``tokens/s * B`` of the node's own benchmark.  It is a property of the
    #: hardware, so it survives a model hot-swap and lets the head node
    #: estimate the throughput of a *different* model on the same node.
    node_tps_per_b: float = 0.0
    ttft_ms: float = 0.0
    mesh_latency_ms: float = 0.0
    inflight: int = 0
    queue_depth: int = 0
    registered_at: float = field(default_factory=now_s)
    last_heartbeat: float = field(default_factory=now_s)
    expires_at: float = 0.0
    consecutive_misses: int = 0
    total_requests: int = 0
    total_tokens: int = 0
    ewma_ttft_ms: float = 0.0
    ewma_total_ms: float = 0.0
    ewma_tokens_per_sec: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    @property
    def healthy(self) -> bool:
        """A worker is healthy when it is alive, not draining and not expired."""
        return (
            not self.draining
            and self.status in SERVING_STATUSES
            and self.expires_at > now_s()
        )

    @property
    def busy(self) -> bool:
        return self.inflight > 0 or self.status is WorkerStatus.BUSY

    @property
    def seconds_to_expiry(self) -> float:
        return max(0.0, self.expires_at - now_s())

    @property
    def max_concurrency(self) -> int:
        return max(1, self.capabilities.max_concurrency)

    @property
    def load_factor(self) -> float:
        return self.inflight / max(1, self.max_concurrency)

    @property
    def has_capacity(self) -> bool:
        return self.healthy and self.inflight < self.max_concurrency

    @property
    def address(self) -> str:
        return f"http://{self.host}:{self.port}"

    def base_url(self) -> str:
        return self.address

    def supports_family(self, family: str) -> bool:
        if not family:
            return True
        families = [f.lower() for f in self.capabilities.supported_families]
        if not families:
            return True
        return family.lower() in families

    def to_snapshot(self) -> WorkerSnapshot:
        """Immutable projection for the API."""
        return WorkerSnapshot(
            worker_id=self.worker_id,
            host=self.host,
            port=self.port,
            provider=self.provider,
            region=self.region,
            status=self.status,
            tier=self.tier,
            model_loaded=self.model_loaded,
            model_family=self.model_family,
            vram_total_gb=round(self.vram_total_gb, 3),
            vram_free_gb=round(self.vram_free_gb, 3),
            tokens_per_sec=round(self.tokens_per_sec, 3),
            mesh_latency_ms=round(self.mesh_latency_ms, 3),
            inflight=self.inflight,
            queue_depth=self.queue_depth,
            registered_at=self.registered_at,
            last_heartbeat=self.last_heartbeat,
            expires_at=self.expires_at,
            seconds_to_expiry=round(self.seconds_to_expiry, 2),
            draining=self.draining,
            healthy=self.healthy,
            compute_tflops=round(self.compute_tflops, 3),
            supported_families=list(self.capabilities.supported_families),
            backend=(self.capabilities.backend or "unknown").strip() or "unknown",
            max_concurrency=self.max_concurrency,
        )

    def to_dict(self) -> dict[str, Any]:
        data = self.to_snapshot().model_dump()
        data["status"] = self.status.value
        data["tier"] = self.tier.value
        data["model_params_b"] = self.model_params_b
        data["ttft_ms"] = round(self.ttft_ms, 3)
        data["total_requests"] = self.total_requests
        data["total_tokens"] = self.total_tokens
        data["ewma_ttft_ms"] = round(self.ewma_ttft_ms, 3)
        data["ewma_total_ms"] = round(self.ewma_total_ms, 3)
        data["ewma_tokens_per_sec"] = round(self.ewma_tokens_per_sec, 3)
        data["metadata"] = dict(self.metadata)
        return data


class WorkerRegistry:
    """Thread-safe registry of workers."""

    def __init__(self, *, ttl_seconds: int = 3600, max_concurrency_per_worker: int = 2) -> None:
        self._workers: dict[str, WorkerRecord] = {}
        self._lock = threading.RLock()
        self._ttl_seconds = max(60, int(ttl_seconds))
        self._max_concurrency = max(1, int(max_concurrency_per_worker))

    # ------------------------------------------------------------------ #
    @property
    def ttl_seconds(self) -> int:
        return self._ttl_seconds

    def set_ttl(self, ttl_seconds: int) -> None:
        """Change the session TTL (applied to new registrations)."""
        self._ttl_seconds = max(60, int(ttl_seconds))

    # ------------------------------------------------------------------ #
    def register(self, hello: WorkerHello, *, ttl_seconds: int | None = None) -> WorkerRecord:
        """Register (or refresh) a worker and return its record."""
        now = now_s()
        effective_ttl = int(ttl_seconds or self._ttl_seconds)
        with self._lock:
            existing = self._workers.get(hello.worker_id)
            caps = hello.capabilities
            if existing is None:
                record = WorkerRecord(
                    worker_id=hello.worker_id,
                    host=hello.host,
                    port=hello.port,
                    provider=caps.provider or "local",
                    region=caps.region,
                    capabilities=caps,
                    model_loaded=hello.model_loaded,
                    model_family=hello.model_family,
                    model_params_b=hello.model_params_b,
                    vram_total_gb=caps.vram_total_gb,
                    vram_free_gb=caps.vram_free_gb,
                    compute_tflops=caps.compute_tflops,
                    tokens_per_sec=_node_throughput(caps),
                    node_tps_per_b=_node_factor(caps, hello),
                    mesh_latency_ms=caps.mesh_latency_ms,
                    registered_at=now,
                    last_heartbeat=now,
                    expires_at=now + effective_ttl,
                    status=WorkerStatus.IDLE,
                    metadata=dict(hello.metadata),
                )
                self._workers[hello.worker_id] = record
                LOGGER.info(
                    "worker registered",
                    extra={
                        "hc_worker": record.worker_id,
                        "hc_provider": record.provider,
                        "hc_vram": record.vram_total_gb,
                        "hc_model": record.model_loaded,
                    },
                )
                return record

            # Refresh an existing registration (worker re-announced itself).
            existing.host = hello.host
            existing.port = hello.port
            existing.capabilities = caps
            existing.provider = caps.provider or existing.provider
            existing.region = caps.region or existing.region
            existing.model_loaded = hello.model_loaded or existing.model_loaded
            existing.model_family = hello.model_family or existing.model_family
            existing.model_params_b = hello.model_params_b or existing.model_params_b
            existing.vram_total_gb = caps.vram_total_gb
            existing.vram_free_gb = caps.vram_free_gb
            existing.compute_tflops = caps.compute_tflops
            existing.tokens_per_sec = _node_throughput(caps) or existing.tokens_per_sec
            existing.node_tps_per_b = _node_factor(caps, hello) or existing.node_tps_per_b
            existing.mesh_latency_ms = caps.mesh_latency_ms or existing.mesh_latency_ms
            existing.last_heartbeat = now
            existing.expires_at = now + effective_ttl
            existing.consecutive_misses = 0
            if existing.status in (WorkerStatus.DEAD, WorkerStatus.BOOTING):
                existing.status = WorkerStatus.IDLE
            existing.metadata.update(hello.metadata)
            LOGGER.info("worker re-registered", extra={"hc_worker": existing.worker_id})
            return existing

    # ------------------------------------------------------------------ #
    def heartbeat(self, beat: WorkerHeartbeat) -> WorkerRecord | None:
        """Apply a heartbeat; returns ``None`` for unknown *or* dead workers.

        A worker that was seen failing is **not** resurrected by a heartbeat:
        it has to register again to start a new session.  Otherwise a
        half-alive worker (its HTTP server gone but its process alive) could
        silently re-enter the routing pool.
        """
        with self._lock:
            record = self._workers.get(beat.worker_id)
            if record is None:
                return None
            if record.status is WorkerStatus.DEAD:
                LOGGER.warning(
                    "heartbeat from a dead worker; it must register again",
                    extra={"hc_worker": beat.worker_id},
                )
                return None
            now = now_s()
            record.last_heartbeat = now
            record.expires_at = now + self._ttl_seconds
            record.consecutive_misses = 0
            record.status = beat.status
            record.inflight = beat.inflight
            record.queue_depth = beat.queue_depth
            record.tokens_per_sec = beat.tokens_per_sec or record.tokens_per_sec
            record.ttft_ms = beat.ttft_ms or record.ttft_ms
            record.vram_free_gb = beat.vram_free_gb or record.vram_free_gb
            record.vram_total_gb = beat.vram_total_gb or record.vram_total_gb
            record.mesh_latency_ms = beat.mesh_latency_ms or record.mesh_latency_ms
            if beat.model_loaded:
                record.model_loaded = beat.model_loaded
            if beat.tokens_per_sec > 0:
                record.ewma_tokens_per_sec = _ewma(record.ewma_tokens_per_sec, beat.tokens_per_sec, 0.3)
            return record

    def touch(self, worker_id: str, *, ttl_seconds: int | None = None) -> WorkerRecord | None:
        """Refresh a worker's TTL without changing its telemetry."""
        with self._lock:
            record = self._workers.get(worker_id)
            if record is None:
                return None
            record.last_heartbeat = now_s()
            record.expires_at = record.last_heartbeat + int(ttl_seconds or self._ttl_seconds)
            record.consecutive_misses = 0
            return record

    def set_status(self, worker_id: str, status: WorkerStatus, *, model_loaded: str | None = None) -> WorkerRecord | None:
        """Update a worker's status (and optionally the loaded model)."""
        with self._lock:
            record = self._workers.get(worker_id)
            if record is None:
                return None
            record.status = status
            if model_loaded is not None:
                record.model_loaded = model_loaded
            return record

    def set_model(self, worker_id: str, model: str, *, family: str | None = None, params_b: float | None = None) -> WorkerRecord | None:
        """Record that a worker now serves a different model."""
        with self._lock:
            record = self._workers.get(worker_id)
            if record is None:
                return None
            record.model_loaded = model
            if family:
                record.model_family = family
            if params_b:
                record.model_params_b = params_b
            return record

    def drain(self, worker_id: str, *, reason: str = "") -> WorkerRecord | None:
        """Stop sending new work to a worker; in-flight work continues."""
        with self._lock:
            record = self._workers.get(worker_id)
            if record is None:
                return None
            record.draining = True
            record.status = WorkerStatus.DRAINING
            if reason:
                record.metadata["drain_reason"] = reason
            LOGGER.info("worker draining", extra={"hc_worker": worker_id, "hc_reason": reason or "manual"})
            return record

    def undrain(self, worker_id: str) -> WorkerRecord | None:
        """Return a draining worker to service."""
        with self._lock:
            record = self._workers.get(worker_id)
            if record is None:
                return None
            record.draining = False
            if record.status is WorkerStatus.DRAINING:
                record.status = WorkerStatus.IDLE
            return record

    def remove(self, worker_id: str) -> WorkerRecord | None:
        """Remove a worker entirely."""
        with self._lock:
            record = self._workers.pop(worker_id, None)
            if record is not None:
                record.status = WorkerStatus.DEAD
                LOGGER.info("worker removed", extra={"hc_worker": worker_id})
            return record

    def mark_lost(self, worker_id: str, *, reason: str = "") -> WorkerRecord | None:
        """Mark a worker as unreachable but keep it visible for diagnostics.

        A worker that failed to serve a request is considered dead straight
        away: it is removed from the routing pool immediately (so no further
        traffic is sent to it) while the record itself survives until its TTL
        expires, which keeps the failure visible in the dashboard.
        """
        with self._lock:
            record = self._workers.get(worker_id)
            if record is None:
                return None
            record.consecutive_misses += 1
            record.metadata["lost_reason"] = reason
            record.status = WorkerStatus.DEAD
            record.inflight = 0
            LOGGER.warning("worker marked lost", extra={"hc_worker": worker_id, "hc_reason": reason})
            return record

    # ------------------------------------------------------------------ #
    def get(self, worker_id: str) -> WorkerRecord | None:
        with self._lock:
            return self._workers.get(worker_id)

    def list_workers(
        self,
        *,
        healthy_only: bool = False,
        serving_only: bool = False,
        providers: Iterable[str] | None = None,
    ) -> list[WorkerRecord]:
        """Snapshot list of workers filtered by the given criteria."""
        provider_filter = {p.lower() for p in providers} if providers else None
        with self._lock:
            records = list(self._workers.values())
        out = []
        for record in records:
            if healthy_only and not record.healthy:
                continue
            if serving_only and not (record.has_capacity and not record.draining):
                continue
            if provider_filter and record.provider.lower() not in provider_filter:
                continue
            out.append(record)
        return sorted(out, key=lambda r: (r.draining, r.status.value, r.worker_id))

    def healthy_workers(self) -> list[WorkerRecord]:
        return self.list_workers(healthy_only=True, serving_only=True)

    def expiring_within(self, seconds: float) -> list[WorkerRecord]:
        """Workers whose session expires within ``seconds`` (rotator input)."""
        with self._lock:
            records = list(self._workers.values())
        horizon = now_s() + seconds
        return sorted(
            (r for r in records if not r.draining and r.expires_at <= horizon),
            key=lambda r: r.expires_at,
        )

    def session_ending_within(self, seconds: float) -> list[WorkerRecord]:
        """Workers whose *session* (kernel, job, instance) ends within ``seconds``.

        A worker reports the absolute end of its session in its registration
        metadata (``session_ends_at``).  Unlike the liveness TTL -- which every
        heartbeat refreshes -- this deadline really does approach, so it is
        the signal the pre-emptive rotator acts on.
        """
        with self._lock:
            records = list(self._workers.values())
        horizon = now_s() + seconds
        ending: list[WorkerRecord] = []
        for record in records:
            if record.draining:
                continue
            raw = record.metadata.get("session_ends_at")
            try:
                session_end = float(raw)
            except (TypeError, ValueError):
                continue
            if session_end <= 0:
                continue
            if session_end <= horizon:
                ending.append(record)
        return sorted(ending, key=lambda r: float(r.metadata.get("session_ends_at", 0.0)))

    # ------------------------------------------------------------------ #
    def claim(self, worker_id: str, count: int = 1) -> bool:
        """Reserve capacity on a worker; ``False`` when it is saturated."""
        with self._lock:
            record = self._workers.get(worker_id)
            if record is None or not record.healthy:
                return False
            if record.inflight + count > record.max_concurrency:
                return False
            record.inflight += count
            record.status = WorkerStatus.BUSY
            return True

    def release(self, worker_id: str, count: int = 1) -> None:
        """Return capacity to a worker.

        A worker that was marked dead keeps its DEAD status: releasing the
        claim after a failure must never put it back into the routing pool.
        """
        with self._lock:
            record = self._workers.get(worker_id)
            if record is None:
                return
            record.inflight = max(0, record.inflight - count)
            if record.status is WorkerStatus.DEAD:
                return
            if record.inflight == 0 and not record.draining:
                record.status = WorkerStatus.IDLE

    def record_latency(
        self,
        worker_id: str,
        *,
        ttft_ms: float,
        total_ms: float,
        tokens: int = 0,
        tokens_per_sec: float = 0.0,
    ) -> WorkerRecord | None:
        """Fold an observed request into the worker's EWMA latency profile."""
        with self._lock:
            record = self._workers.get(worker_id)
            if record is None:
                return None
            record.total_requests += 1
            record.total_tokens += max(0, tokens)
            if ttft_ms > 0:
                record.ewma_ttft_ms = _ewma(record.ewma_ttft_ms, ttft_ms, 0.3)
            if total_ms > 0:
                record.ewma_total_ms = _ewma(record.ewma_total_ms, total_ms, 0.3)
            if tokens_per_sec > 0:
                record.ewma_tokens_per_sec = _ewma(record.ewma_tokens_per_sec, tokens_per_sec, 0.3)
                record.tokens_per_sec = tokens_per_sec
            record.ttft_ms = record.ewma_ttft_ms or record.ttft_ms
            return record

    # ------------------------------------------------------------------ #
    def sweep(self, *, now: float | None = None) -> list[WorkerRecord]:
        """Expire workers whose TTL elapsed; returns the expired records."""
        moment = now_s() if now is None else now
        expired: list[WorkerRecord] = []
        with self._lock:
            for worker_id, record in list(self._workers.items()):
                if record.expires_at <= moment:
                    record.status = WorkerStatus.DEAD
                    record.inflight = 0
                    expired.append(record)
                    LOGGER.warning("worker expired", extra={"hc_worker": worker_id})
            for record in expired:
                self._workers.pop(record.worker_id, None)
        return expired

    def totals(self) -> dict[str, Any]:
        """Aggregate capacity of the whole mesh."""
        with self._lock:
            records = list(self._workers.values())
        healthy = [r for r in records if r.healthy]
        return {
            "workers": len(records),
            "healthy": len(healthy),
            "draining": len([r for r in records if r.draining]),
            "total_vram_gb": round(sum(r.vram_total_gb for r in records), 3),
            "free_vram_gb": round(sum(r.vram_free_gb for r in records), 3),
            "total_tflops": round(sum(r.compute_tflops for r in records), 3),
            "inflight": sum(r.inflight for r in records),
            "tokens_per_sec": round(sum(r.tokens_per_sec for r in healthy), 3),
            "providers": sorted({r.provider for r in records}),
        }

    def snapshot(self) -> list[WorkerSnapshot]:
        """All workers as API-safe snapshots."""
        with self._lock:
            records = list(self._workers.values())
        return [r.to_snapshot() for r in sorted(records, key=lambda r: r.worker_id)]

    def __len__(self) -> int:  # pragma: no cover - trivial
        with self._lock:
            return len(self._workers)


__all__ = ["WorkerRecord", "WorkerRegistry", "SERVING_STATUSES"]
