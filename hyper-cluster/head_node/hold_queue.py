"""Hold-and-requeue buffer.

When the cluster cannot serve a request *right now* (every GPU is busy, a
worker just died, a model is being hot-swapped, the mesh is draining) the
request is **held** in memory instead of being failed.  A held request:

* keeps its place in a FIFO fairness queue,
* survives worker churn -- it is replayed on the next healthy worker,
* has a deadline; when the deadline passes the request is **deferred**, not
  dropped: the result store keeps it and the background replayer executes it
  as soon as capacity returns,
* never surfaces as a raw 500/503 to the phone.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.errors import HoldTimeoutError
from core.ids import now_s
from core.logging_setup import get_logger
from core.protocol import ChatCompletion, ChatRequest, RequestState, RequestStatus

LOGGER = get_logger("head_node.hold_queue")


class HoldState(str, Enum):
    """Lifecycle of a held request."""

    WAITING = "waiting"      # queued, waiting for capacity
    ASSIGNED = "assigned"    # a worker has been reserved for it
    COMPLETED = "completed"  # finished successfully
    FAILED = "failed"        # exhausted its retries
    DEFERRED = "deferred"    # deadline passed; queued for automatic replay


@dataclass
class HeldRequest:
    """One request parked in the hold buffer."""

    request_id: str
    request: ChatRequest
    state: HoldState = HoldState.WAITING
    enqueued_at: float = field(default_factory=now_s)
    deadline_at: float = 0.0
    attempts: int = 0
    worker_id: str | None = None
    error: str | None = None
    future: asyncio.Future[ChatCompletion] | None = None

    @property
    def waiting_seconds(self) -> float:
        return now_s() - self.enqueued_at

    @property
    def seconds_to_deadline(self) -> float:
        return max(0.0, self.deadline_at - now_s())

    @property
    def expired(self) -> bool:
        return self.deadline_at > 0 and now_s() >= self.deadline_at

    def to_status(self) -> RequestStatus:
        return RequestStatus(
            request_id=self.request_id,
            state={
                HoldState.WAITING: RequestState.HELD,
                HoldState.ASSIGNED: RequestState.DISPATCHED,
                HoldState.COMPLETED: RequestState.COMPLETED,
                HoldState.FAILED: RequestState.FAILED,
                HoldState.DEFERRED: RequestState.DEFERRED,
            }[self.state],
            mode=self.request.mode,
            created_at=self.enqueued_at,
            updated_at=now_s(),
            held_ms=round(self.waiting_seconds * 1000.0, 2),
            attempts=self.attempts,
            worker_id=self.worker_id,
            error=self.error,
        )


class HoldQueue:
    """Thread-safe FIFO buffer of parked requests."""

    def __init__(self, *, max_size: int = 512, ttl_seconds: int = 900) -> None:
        self._max_size = max(1, int(max_size))
        self._ttl_seconds = max(5, int(ttl_seconds))
        self._entries: dict[str, HeldRequest] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def ttl_seconds(self) -> int:
        return self._ttl_seconds

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def size(self) -> int:
        """Number of parked requests."""
        return len(self)

    def waiting_count(self) -> int:
        with self._lock:
            return sum(1 for e in self._entries.values() if e.state is HoldState.WAITING)

    def deferred_count(self) -> int:
        with self._lock:
            return sum(1 for e in self._entries.values() if e.state is HoldState.DEFERRED)

    # ------------------------------------------------------------------ #
    def enqueue(self, request: ChatRequest, *, deadline_seconds: float | None = None) -> HeldRequest:
        """Park a request and return its handle.

        Raises:
            HoldQueueFull: when the buffer is saturated (the caller then
                answers with a graceful ``202`` instead of an error).
        """
        now = now_s()
        ttl = float(deadline_seconds if deadline_seconds is not None else self._ttl_seconds)
        entry = HeldRequest(
            request_id=request.request_id,
            request=request,
            enqueued_at=now,
            deadline_at=now + max(1.0, ttl),
            future=asyncio.get_running_loop().create_future() if _in_loop() else None,
        )
        with self._lock:
            if len(self._entries) >= self._max_size:
                raise HoldQueueFull(
                    f"hold queue is full ({self._max_size} parked requests)",
                    request_id=request.request_id,
                )
            self._entries[request.request_id] = entry
        LOGGER.info(
            "request held",
            extra={"hc_request": request.request_id, "hc_waiting": self.waiting_count()},
        )
        return entry

    def get(self, request_id: str) -> HeldRequest | None:
        with self._lock:
            return self._entries.get(request_id)

    def remove(self, request_id: str) -> HeldRequest | None:
        with self._lock:
            return self._entries.pop(request_id, None)

    def ready(self) -> list[HeldRequest]:
        """Waiting entries in FIFO order (oldest first)."""
        with self._lock:
            waiting = [e for e in self._entries.values() if e.state is HoldState.WAITING]
        return sorted(waiting, key=lambda e: e.enqueued_at)

    def deferred(self) -> list[HeldRequest]:
        """Deferred entries awaiting automatic replay."""
        with self._lock:
            deferred = [e for e in self._entries.values() if e.state is HoldState.DEFERRED]
        return sorted(deferred, key=lambda e: e.enqueued_at)

    def claim(self, request_id: str, worker_id: str | None = None) -> HeldRequest | None:
        """Mark a waiting entry as assigned to a worker."""
        with self._lock:
            entry = self._entries.get(request_id)
            if entry is None or entry.state is not HoldState.WAITING:
                return None
            entry.state = HoldState.ASSIGNED
            entry.worker_id = worker_id
            entry.attempts += 1
            return entry

    def assign(self, request_id: str, worker_id: str) -> HeldRequest | None:
        """Take a waiting entry out of the buffer and hand it to a worker.

        Unlike :meth:`claim` -- which keeps the entry in the buffer so the
        deferred pool can still see it -- an assigned request is no longer
        waiting for capacity.  It is removed from the buffer, so the queue
        depth and the ``/v1/status`` counters stay honest and the request can
        not be replayed a second time.
        """
        with self._lock:
            entry = self._entries.get(request_id)
            if entry is None or entry.state is not HoldState.WAITING:
                return None
            entry.state = HoldState.ASSIGNED
            entry.worker_id = worker_id
            entry.attempts += 1
            self._entries.pop(request_id, None)
        LOGGER.info(
            "held request assigned to a worker",
            extra={"hc_request": request_id, "hc_worker": worker_id},
        )
        return entry

    def complete(self, request_id: str, completion: ChatCompletion) -> HeldRequest | None:
        """Resolve a parked request with a successful result."""
        with self._lock:
            entry = self._entries.pop(request_id, None)
        if entry is None:
            return None
        entry.state = HoldState.COMPLETED
        if entry.future is not None and not entry.future.done():
            entry.future.set_result(completion)
        return entry

    def fail(self, request_id: str, error: str) -> HeldRequest | None:
        """Resolve a parked request with a failure."""
        with self._lock:
            entry = self._entries.pop(request_id, None)
        if entry is None:
            return None
        entry.state = HoldState.FAILED
        entry.error = error
        if entry.future is not None and not entry.future.done():
            entry.future.set_exception(HoldTimeoutError(error))
            _consume_future_exception(entry.future)
        return entry

    def defer(self, request_id: str, reason: str = "deadline exceeded") -> HeldRequest | None:
        """Move a parked request to the deferred pool (never dropped)."""
        with self._lock:
            entry = self._entries.get(request_id)
            if entry is None:
                return None
            entry.state = HoldState.DEFERRED
            entry.error = reason
        LOGGER.warning(
            "request deferred for replay",
            extra={"hc_request": request_id, "hc_reason": reason},
        )
        return entry

    def expire(self) -> list[HeldRequest]:
        """Move every expired waiting entry to the deferred pool."""
        expired: list[HeldRequest] = []
        with self._lock:
            for entry in self._entries.values():
                if entry.state is HoldState.WAITING and entry.expired:
                    expired.append(entry)
        for entry in expired:
            self.defer(entry.request_id, "deadline exceeded while waiting for capacity")
        return expired

    def snapshot(self) -> list[dict[str, Any]]:
        """Serialisable view of the buffer (for the mobile dashboard)."""
        with self._lock:
            entries = list(self._entries.values())
        return sorted(
            (
                {
                    "request_id": e.request_id,
                    "state": e.state.value,
                    "enqueued_at": e.enqueued_at,
                    "waiting_ms": round(e.waiting_seconds * 1000.0, 2),
                    "deadline_at": e.deadline_at,
                    "attempts": e.attempts,
                    "mode": e.request.mode.value,
                    "error": e.error,
                }
                for e in entries
            ),
            key=lambda item: item["enqueued_at"],
        )

    def stats(self) -> dict[str, Any]:
        with self._lock:
            entries = list(self._entries.values())
        return {
            "size": len(entries),
            "waiting": sum(1 for e in entries if e.state is HoldState.WAITING),
            "assigned": sum(1 for e in entries if e.state is HoldState.ASSIGNED),
            "deferred": sum(1 for e in entries if e.state is HoldState.DEFERRED),
            "max_size": self._max_size,
        }


class HoldQueueFull(Exception):
    """Raised when the hold buffer cannot accept another request."""

    def __init__(self, message: str, *, request_id: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.request_id = request_id


def _consume_future_exception(future: asyncio.Future[Any]) -> None:
    """Mark a future's exception as retrieved (nobody may await it)."""

    def _drain(fut: asyncio.Future[Any]) -> None:
        if not fut.cancelled():
            fut.exception()

    future.add_done_callback(_drain)


def _in_loop() -> bool:
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


__all__ = ["HoldState", "HeldRequest", "HoldQueue", "HoldQueueFull"]
