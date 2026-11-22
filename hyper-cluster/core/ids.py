"""Identifier and clock helpers.

Every component stamps messages with a monotonic-ish wall clock in *milliseconds*
so latency can be computed across processes without clock-skew surprises
(we always compare deltas, never absolute timestamps from different hosts).
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Final

_REQUEST_PREFIX: Final[str] = "req"
_WORKER_PREFIX: Final[str] = "wk"
_TRACE_PREFIX: Final[str] = "trc"
_JOB_PREFIX: Final[str] = "job"


def now_ms() -> int:
    """Wall-clock milliseconds since the Unix epoch."""
    return int(time.time() * 1000)


def now_s() -> float:
    """Wall-clock seconds since the Unix epoch (float)."""
    return time.time()


def mono_ms() -> float:
    """Monotonic milliseconds, safe for intra-process duration measurement."""
    return time.monotonic() * 1000.0


def _token(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def new_request_id() -> str:
    """Unique id for a chat/completion request."""
    return _token(_REQUEST_PREFIX)


def new_worker_id(provider: str = "local") -> str:
    """Unique id for a mesh worker, namespaced by the provider that spawned it."""
    safe_provider = "".join(ch if ch.isalnum() else "-" for ch in provider.lower())
    return f"{_WORKER_PREFIX}_{safe_provider}_{os.getpid()}_{uuid.uuid4().hex[:8]}"


def new_trace_id() -> str:
    """Unique id for a distributed trace (one request -> many worker hops)."""
    return _token(_TRACE_PREFIX)


def new_job_id() -> str:
    """Unique id for a background job (evolve, train, sync, ...)."""
    return _token(_JOB_PREFIX)


def short_id(identifier: str, length: int = 8) -> str:
    """Human friendly short form of an identifier, safe for log lines."""
    cleaned = identifier.replace("_", "").replace("-", "")
    return cleaned[:length] if cleaned else identifier[:length]


__all__ = [
    "now_ms",
    "now_s",
    "mono_ms",
    "new_request_id",
    "new_worker_id",
    "new_trace_id",
    "new_job_id",
    "short_id",
]
