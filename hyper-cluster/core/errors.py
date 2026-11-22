"""Typed exception hierarchy shared by every Hyper-Cluster component.

The controller code (FastAPI, orchestrator, workers) never raises bare
``Exception``; every failure mode has a named type so the orchestrator can
translate it into a *graceful* response instead of a raw 500 leaking to the
phone.
"""

from __future__ import annotations

from typing import Any


class HyperClusterError(Exception):
    """Root of the Hyper-Cluster exception tree.

    Args:
        message: Human readable description.
        code: Stable machine readable slug used in API payloads and SSE events.
        details: Optional JSON-serialisable context.
    """

    code = "hypercluster_error"
    http_status = 500

    def __init__(self, message: str, *, code: str | None = None, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {"error": self.code, "message": self.message, "details": self.details}


class ConfigError(HyperClusterError):
    """Raised when configuration is missing or contradictory."""

    code = "config_error"
    http_status = 500


class AuthError(HyperClusterError):
    """Raised when a caller presents a missing/incorrect credential."""

    code = "auth_error"
    http_status = 401


class ProtocolError(HyperClusterError):
    """Raised when a peer speaks an incompatible version of the wire protocol."""

    code = "protocol_error"
    http_status = 400


class ValidationError(HyperClusterError):
    """Raised when a request payload fails semantic validation."""

    code = "validation_error"
    http_status = 422


class NoCapacityError(HyperClusterError):
    """Raised when no worker can currently serve a request.

    The orchestrator converts this into a *hold* (retry later) rather than an
    error surfaced to the user.
    """

    code = "no_capacity"
    http_status = 503


class WorkerUnreachableError(HyperClusterError):
    """Raised when a worker accepted a request but then became unreachable."""

    code = "worker_unreachable"
    http_status = 502


class WorkerBusyError(HyperClusterError):
    """Raised when a worker is saturated and cannot accept more concurrency."""

    code = "worker_busy"
    http_status = 503


class ModelNotReadyError(HyperClusterError):
    """Raised when a worker does not have (or cannot load) the requested model."""

    code = "model_not_ready"
    http_status = 503


class HoldTimeoutError(HyperClusterError):
    """Raised when a held request outlived its deadline without capacity.

    The request is *not* dropped: it stays in the deferred store and is
    replayed automatically once capacity returns.
    """

    code = "hold_timeout"
    http_status = 202


class ProviderQuotaError(HyperClusterError):
    """Raised when an external provider reports rate limit / quota exhaustion."""

    code = "provider_quota"
    http_status = 429


class ProviderAuthError(HyperClusterError):
    """Raised when an external provider credential is rejected."""

    code = "provider_auth"
    http_status = 401


class ProviderUnavailableError(HyperClusterError):
    """Raised when an external provider endpoint is unreachable."""

    code = "provider_unavailable"
    http_status = 502


class CredentialStoreError(HyperClusterError):
    """Raised when the credential store cannot be read or written."""

    code = "credential_store_error"
    http_status = 500


__all__ = [
    "HyperClusterError",
    "ConfigError",
    "AuthError",
    "ProtocolError",
    "ValidationError",
    "NoCapacityError",
    "WorkerUnreachableError",
    "WorkerBusyError",
    "ModelNotReadyError",
    "HoldTimeoutError",
    "ProviderQuotaError",
    "ProviderAuthError",
    "ProviderUnavailableError",
    "CredentialStoreError",
]
