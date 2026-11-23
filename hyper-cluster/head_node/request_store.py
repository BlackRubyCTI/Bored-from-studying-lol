"""Ledger of every request the control plane has seen.

The ledger is what makes *deferred* work recoverable: when a request had to be
parked (all workers busy, worker died, model hot-swap in flight) the phone
gets a ``202`` with a request id; minutes later the cluster can still deliver
the answer and the phone can fetch it with ``GET /v1/requests/{request_id}``.

Entries are kept in memory with a bounded history and are optionally mirrored
to disk so a head-node restart does not lose the outcome of in-flight work.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from core.ids import now_s
from core.logging_setup import get_logger
from core.protocol import ChatCompletion, ChatRequest, RequestState, RequestStatus

LOGGER = get_logger("head_node.request_store")


class RequestLedger:
    """Bounded, optionally disk-backed ledger of request outcomes."""

    def __init__(self, *, max_entries: int = 2000, persist_path: str | Path | None = None) -> None:
        self._entries: dict[str, RequestStatus] = {}
        self._completions: dict[str, ChatCompletion] = {}
        self._lock = threading.RLock()
        self._max_entries = max(16, int(max_entries))
        self._persist_path = Path(persist_path).expanduser() if persist_path else None
        self._order: list[str] = []

    # ------------------------------------------------------------------ #
    def record_request(self, request: ChatRequest) -> RequestStatus:
        """Create (or refresh) the ledger entry for a request."""
        with self._lock:
            status = self._entries.get(request.request_id)
            if status is None:
                status = RequestStatus(
                    request_id=request.request_id,
                    state=RequestState.QUEUED,
                    mode=request.mode,
                    created_at=now_s(),
                    updated_at=now_s(),
                )
                self._entries[request.request_id] = status
                self._order.append(request.request_id)
                self._evict_locked()
            status.updated_at = now_s()
            return status

    def set_state(
        self,
        request_id: str,
        state: RequestState,
        *,
        worker_id: str | None = None,
        error: str | None = None,
        held_ms: float | None = None,
    ) -> RequestStatus | None:
        """Advance the lifecycle state of a request."""
        with self._lock:
            status = self._entries.get(request_id)
            if status is None:
                return None
            status.state = state
            status.updated_at = now_s()
            if worker_id is not None:
                status.worker_id = worker_id
            if error is not None:
                status.error = error
            if held_ms is not None:
                status.held_ms = held_ms
            return status

    def complete(self, completion: ChatCompletion) -> RequestStatus | None:
        """Store the final result of a request."""
        with self._lock:
            status = self._entries.get(completion.request_id)
            if status is None:
                status = RequestStatus(request_id=completion.request_id, created_at=now_s())
                self._entries[completion.request_id] = status
                self._order.append(completion.request_id)
            if completion.error == "deferred":
                status.state = RequestState.DEFERRED
            elif completion.error:
                status.state = RequestState.FAILED
            else:
                status.state = RequestState.COMPLETED
            status.updated_at = now_s()
            status.text = completion.text
            status.worker_id = completion.worker_id or status.worker_id
            status.error = completion.error
            self._completions[completion.request_id] = completion
            self._evict_locked()
            return status

    def get(self, request_id: str) -> RequestStatus | None:
        with self._lock:
            return self._entries.get(request_id)

    def result(self, request_id: str) -> ChatCompletion | None:
        with self._lock:
            return self._completions.get(request_id)

    def recent(self, limit: int = 50) -> list[RequestStatus]:
        with self._lock:
            ids = self._order[-limit:]
            return [self._entries[i] for i in ids if i in self._entries]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            entries = list(self._entries.values())
        by_state: dict[str, int] = {}
        for entry in entries:
            by_state[entry.state.value] = by_state.get(entry.state.value, 0) + 1
        return {"total": len(entries), "by_state": by_state}

    # ------------------------------------------------------------------ #
    def _evict_locked(self) -> None:
        while len(self._order) > self._max_entries:
            oldest = self._order.pop(0)
            self._entries.pop(oldest, None)
            self._completions.pop(oldest, None)

    def persist(self) -> None:
        """Write the ledger to disk (best effort)."""
        if not self._persist_path:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "entries": {rid: status.model_dump() for rid, status in self._entries.items()},
                "completions": {
                    rid: completion.model_dump() for rid, completion in self._completions.items()
                },
            }
            self._persist_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        except OSError as exc:  # pragma: no cover - disk issues must not kill traffic
            LOGGER.warning("failed to persist request ledger", extra={"hc_error": str(exc)})

    def restore(self) -> int:
        """Load a previously persisted ledger; returns the number of entries."""
        if not self._persist_path or not self._persist_path.is_file():
            return 0
        try:
            payload = json.loads(self._persist_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover
            LOGGER.warning("failed to restore request ledger", extra={"hc_error": str(exc)})
            return 0
        restored = 0
        with self._lock:
            for rid, data in (payload.get("entries") or {}).items():
                try:
                    status = RequestStatus(**data)
                except Exception:  # noqa: BLE001 - tolerate schema drift
                    continue
                self._entries[rid] = status
                self._order.append(rid)
                restored += 1
            for rid, data in (payload.get("completions") or {}).items():
                try:
                    self._completions[rid] = ChatCompletion(**data)
                except Exception:  # noqa: BLE001
                    continue
            self._evict_locked()
        return restored


__all__ = ["RequestLedger"]
