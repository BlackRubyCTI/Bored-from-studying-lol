"""Worker-side configuration.

Every value can be supplied through the environment (the bootstrap script
writes them into the worker's env file), so a worker can be re-provisioned on
a fresh machine with **zero** manual steps.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ENV_PREFIX = "HYPER_WORKER_"


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(f"{ENV_PREFIX}{name}", default)


def _int(name: str, default: int) -> int:
    raw = _env(name)
    try:
        return int(raw) if raw not in (None, "") else default
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    raw = _env(name)
    try:
        return float(raw) if raw not in (None, "") else default
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(slots=True)
class WorkerConfig:
    """Everything a worker needs to know about itself."""

    # --- identity ---------------------------------------------------------
    worker_id: str = ""
    provider: str = "local"
    region: str = "unknown"

    # --- control plane ----------------------------------------------------
    head_url: str = "http://127.0.0.1:8080"
    mesh_token: str = ""
    bind_host: str = "0.0.0.0"
    bind_port: int = 8081
    heartbeat_seconds: int = 20
    ttl_seconds: int = 3600
    reconnect_backoff_seconds: float = 2.0
    reconnect_backoff_max_seconds: float = 60.0
    #: Absolute unix timestamp when the *session* (kernel, job, instance)
    #: ends.  The head node uses it to replace the worker before it dies.
    session_ends_at: float = 0.0

    # --- model ------------------------------------------------------------
    model_path: str = ""
    model_repo: str = ""
    model_filename: str = ""
    model_family: str = "qwen2"
    model_params_b: float = 7.0
    model_sha256: str = ""
    context_size: int = 8192
    gpu_layers: int = -1
    threads: int = 0

    # --- inference backends ------------------------------------------------
    #: ``auto`` picks llama.cpp when available, then an OpenAI-compatible
    #: upstream, then the deterministic simulator.
    backend: str = "auto"
    upstream_url: str = ""
    upstream_model: str = ""
    upstream_api_key: str = ""

    # --- capacity ----------------------------------------------------------
    max_concurrency: int = 2
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95

    # --- tooling -----------------------------------------------------------
    sandbox_enabled: bool = True
    sandbox_timeout_seconds: int = 20
    sandbox_workdir: str = ""

    # --- observability -----------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = True
    log_file: str = ""

    # --- mesh --------------------------------------------------------------
    tailscale_auth_key: str = ""
    tailscale_hostname: str = ""
    tailscale_tags: list[str] = field(default_factory=list)
    mesh_enabled: bool = True

    # ------------------------------------------------------------------ #
    @classmethod
    def from_env(cls) -> WorkerConfig:
        """Build the configuration from the process environment."""
        tags = _env("TAGS", "") or ""
        return cls(
            worker_id=_env("ID", "") or "",
            provider=_env("PROVIDER", "local") or "local",
            region=_env("REGION", "unknown") or "unknown",
            head_url=(_env("HEAD_URL", "http://127.0.0.1:8080") or "http://127.0.0.1:8080").rstrip("/"),
            mesh_token=_env("MESH_TOKEN", "") or "",
            bind_host=_env("BIND_HOST", "0.0.0.0") or "0.0.0.0",
            bind_port=_int("BIND_PORT", 8081),
            heartbeat_seconds=_int("HEARTBEAT_SECONDS", 20),
            ttl_seconds=_int("TTL_SECONDS", 3600),
            session_ends_at=_float("SESSION_ENDS_AT", 0.0),
            model_path=_env("MODEL_PATH", "") or "",
            model_repo=_env("MODEL_REPO", "") or "",
            model_filename=_env("MODEL_FILENAME", "") or "",
            model_family=_env("MODEL_FAMILY", "qwen2") or "qwen2",
            model_params_b=_float("MODEL_PARAMS_B", 7.0),
            model_sha256=_env("MODEL_SHA256", "") or "",
            context_size=_int("CONTEXT_SIZE", 8192),
            gpu_layers=_int("GPU_LAYERS", -1),
            threads=_int("THREADS", 0),
            backend=_env("BACKEND", "auto") or "auto",
            upstream_url=_env("UPSTREAM_URL", "") or "",
            upstream_model=_env("UPSTREAM_MODEL", "") or "",
            upstream_api_key=_env("UPSTREAM_API_KEY", "") or "",
            max_concurrency=_int("MAX_CONCURRENCY", 2),
            max_tokens=_int("MAX_TOKENS", 2048),
            temperature=_float("TEMPERATURE", 0.7),
            top_p=_float("TOP_P", 0.95),
            sandbox_enabled=_bool("SANDBOX_ENABLED", True),
            sandbox_timeout_seconds=_int("SANDBOX_TIMEOUT_SECONDS", 20),
            sandbox_workdir=_env("SANDBOX_WORKDIR", "") or "",
            log_level=(_env("LOG_LEVEL", "INFO") or "INFO").upper(),
            log_json=_bool("LOG_JSON", True),
            log_file=_env("LOG_FILE", "") or "",
            tailscale_auth_key=_env("TAILSCALE_AUTH_KEY", "") or "",
            tailscale_hostname=_env("TAILSCALE_HOSTNAME", "") or "",
            tailscale_tags=[t.strip() for t in tags.split(",") if t.strip()],
            mesh_enabled=_bool("MESH_ENABLED", True),
        )

    # ------------------------------------------------------------------ #
    def resolved_worker_id(self) -> str:
        """Worker id, generated on first use and stable afterwards."""
        if self.worker_id:
            return self.worker_id
        import socket
        import uuid

        host = socket.gethostname().split(".")[0][:24]
        self.worker_id = f"wk_{self.provider}_{host}_{uuid.uuid4().hex[:6]}"
        return self.worker_id

    def workdir(self) -> Path:
        """Scratch directory used by the sandbox and the model cache."""
        base = Path(self.sandbox_workdir).expanduser() if self.sandbox_workdir else Path.home() / ".hypercluster"
        base.mkdir(parents=True, exist_ok=True)
        return base

    def redacted(self) -> dict[str, Any]:
        """Safe-to-log view (tokens masked)."""
        data = {f: getattr(self, f) for f in self.__slots__}
        for secret in ("mesh_token", "tailscale_auth_key", "upstream_api_key"):
            data[secret] = "***" if data[secret] else ""
        return data


__all__ = ["WorkerConfig", "ENV_PREFIX"]
