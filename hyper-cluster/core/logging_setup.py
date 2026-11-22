"""Structured logging for the whole cluster.

Two renderings are supported and can be switched at runtime:

* ``json``  -- one JSON object per line, ideal for log shippers and for the
  mobile dashboard's live telemetry panel.
* ``console`` -- human friendly, colourised, aligned output for interactive
  terminals.

Every record carries the ambient *context* (request id, trace id, worker id,
component) so a single log line can be correlated across the mesh.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s :: %(message)s"

#: Ambient context merged into every record emitted inside the block.
context_var: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar("hypercluster_log_context", default={})

#: Fields that must never be emitted verbatim.
_SECRET_HINTS = ("token", "secret", "password", "api_key", "authorization", "cookie")

_configured = threading.Event()
_configured_lock = threading.Lock()


class ContextFilter(logging.Filter):
    """Injects the ambient context dict onto each record."""

    def filter(self, record: logging.LogRecord) -> bool:
        extra_context = dict(context_var.get() or {})
        existing = getattr(record, "context", None)
        if isinstance(existing, dict):
            merged = dict(existing)
            merged.update(extra_context)
            extra_context = merged
        record.context = extra_context
        for key, value in extra_context.items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: ("***" if any(hint in k.lower() for hint in _SECRET_HINTS) else _redact(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_redact(item) for item in value)
    return value


class JsonFormatter(logging.Formatter):
    """Renders each record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S") + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        context = getattr(record, "context", None) or {}
        for key, value in context.items():
            if key not in payload and value is not None:
                payload[key] = _redact(value)
        # Any explicitly attached extras (``logger.info(..., extra={...})``).
        for key, value in record.__dict__.items():
            if key.startswith("hc_") and value is not None:
                name = key[3:]
                if any(hint in name.lower() for hint in _SECRET_HINTS):
                    payload[name] = "***"
                else:
                    payload[name] = _redact(value)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class ConsoleFormatter(logging.Formatter):
    """Compact, colourised, human readable formatter."""

    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[1;41m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        level = record.levelname
        color = self.COLORS.get(level, "") if _supports_colour() else ""
        reset = self.RESET if color else ""
        stamp = self.formatTime(record, "%H:%M:%S")
        context = getattr(record, "context", None) or {}
        ctx = "".join(f" {k}={v}" for k, v in context.items() if v is not None)
        head = f"{color}{stamp} {level:<8}{reset} {record.name:<28}"
        message = record.getMessage()
        out = f"{head} {message}{ctx}"
        if record.exc_info:
            out += "\n" + self.formatException(record.exc_info)
        return out


def _supports_colour() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stderr.isatty()


def setup_logging(
    level: str | int | None = None,
    *,
    json_mode: bool | None = None,
    log_file: str | Path | None = None,
    force: bool = False,
) -> logging.Logger:
    """Configure the root logger.

    Safe to call repeatedly; only the first call (or a call with ``force``)
    mutates handlers.
    """
    with _configured_lock:
        root = logging.getLogger()
        if _configured.is_set() and not force:
            return root

        resolved_level = level
        if resolved_level is None:
            resolved_level = os.environ.get("HYPER_LOG_LEVEL", "INFO")
        if isinstance(resolved_level, str):
            resolved_level = getattr(logging, resolved_level.upper(), logging.INFO)

        if json_mode is None:
            raw_json = os.environ.get("HYPER_LOG_JSON", "1")
            json_mode = str(raw_json).strip().lower() not in ("0", "false", "no", "off")

        for handler in list(root.handlers):
            root.removeHandler(handler)

        formatter = JsonFormatter() if json_mode else ConsoleFormatter()
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setFormatter(formatter)
        stream_handler.addFilter(ContextFilter())
        root.addHandler(stream_handler)

        if log_file:
            path = Path(log_file).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(path, encoding="utf-8")
            file_handler.setFormatter(JsonFormatter())
            file_handler.addFilter(ContextFilter())
            root.addHandler(file_handler)

        root.setLevel(resolved_level)
        # Third party libraries are noisy at INFO; keep them at WARNING.
        for noisy in ("httpx", "httpcore", "urllib3", "asyncio", "uvicorn.access"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        _configured.set()
        return root


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger that carries the ambient context."""
    return logging.getLogger(name)


@contextmanager
def log_context(**kwargs: Any) -> Iterator[None]:
    """Bind extra key/value pairs to every log record emitted in the block."""
    current = dict(context_var.get() or {})
    current.update({k: v for k, v in kwargs.items() if v is not None})
    token = context_var.set(current)
    try:
        yield
    finally:
        context_var.reset(token)


def bind_context(**kwargs: Any) -> None:
    """Bind context for the remainder of the current context (no reset)."""
    current = dict(context_var.get() or {})
    current.update({k: v for k, v in kwargs.items() if v is not None})
    context_var.set(current)


def clear_context() -> None:
    """Drop all ambient context."""
    context_var.set({})


__all__ = [
    "setup_logging",
    "get_logger",
    "log_context",
    "bind_context",
    "clear_context",
    "JsonFormatter",
    "ConsoleFormatter",
    "context_var",
]
