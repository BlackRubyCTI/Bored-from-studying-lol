"""Multi-account credential vault.

The vault keeps *several* credentials per provider (one per free account), tracks
their health/quota state and hands out the healthiest one on demand.  It is the
foundation used by:

* the edge bridge (sub-second fallback brains),
* the auto-spawner (headless compute pools),
* the rotator (silent failover between accounts).

Storage
-------
``~/.hypercluster/credentials.json`` (mode ``0600``)::

    {
      "groq": [
        {"label": "acct-a", "token": "gsk_...", "base_url": "", "models": ["..."], "rpm_limit": 30},
        {"label": "acct-b", "token": "gsk_...", "base_url": "", "models": []}
      ]
    }

Environment variables override the file so the operator can inject secrets
without ever writing them to disk:

``HYPER_CRED_GROQ_ACCT_A=gsk_...`` -> provider ``groq``, label ``acct-a``.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.errors import CredentialStoreError
from core.ids import now_s
from core.providers import PROVIDERS, ProviderSpec, get_provider

HEALTH_UNKNOWN = "unknown"
HEALTH_HEALTHY = "healthy"
HEALTH_DEGRADED = "degraded"
HEALTH_EXHAUSTED = "exhausted"
HEALTH_INVALID = "invalid"
HEALTH_DISABLED = "disabled"


@dataclass
class Credential:
    """One account for one provider."""

    provider: str
    label: str = "default"
    token: str = ""
    base_url: str = ""
    models: list[str] = field(default_factory=list)
    rpm_limit: int = 30
    health: str = HEALTH_UNKNOWN
    last_probe: float = 0.0
    last_error: str = ""
    requests_made: int = 0
    rate_window_start: float = 0.0
    rate_window_count: int = 0
    disabled: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    @property
    def key(self) -> str:
        return f"{self.provider}:{self.label}"

    def spec(self) -> ProviderSpec | None:
        return get_provider(self.provider)

    def effective_base_url(self) -> str:
        if self.base_url:
            return self.base_url.rstrip("/")
        spec = self.spec()
        return spec.base_url if spec else ""

    def available_models(self) -> list[str]:
        if self.models:
            return list(self.models)
        spec = self.spec()
        return list(spec.models) if spec else []

    def default_model(self) -> str:
        spec = self.spec()
        return spec.default_model if spec else ""

    def rate_limited(self) -> bool:
        """True when the current one-minute window is exhausted."""
        if self.rpm_limit <= 0:
            return False
        now = now_s()
        if now - self.rate_window_start >= 60.0:
            return False
        return self.rate_window_count >= self.rpm_limit

    def record_request(self) -> None:
        """Record one outgoing request against the rate window."""
        now = now_s()
        if now - self.rate_window_start >= 60.0:
            self.rate_window_start = now
            self.rate_window_count = 0
        self.rate_window_count += 1
        self.requests_made += 1

    def to_dict(self, *, redact: bool = True) -> dict[str, Any]:
        data = {
            "provider": self.provider,
            "label": self.label,
            "base_url": self.base_url,
            "models": list(self.models),
            "rpm_limit": self.rpm_limit,
            "health": self.health,
            "last_probe": self.last_probe,
            "last_error": self.last_error,
            "requests_made": self.requests_made,
            "disabled": self.disabled,
            "metadata": dict(self.metadata),
        }
        data["token"] = "***" if redact else self.token
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Credential:
        provider = str(data.get("provider", "")).strip().lower()
        if not provider:
            raise CredentialStoreError("credential entry is missing 'provider'")
        return cls(
            provider=provider,
            label=str(data.get("label", "default")),
            token=str(data.get("token", "")),
            base_url=str(data.get("base_url", "")),
            models=[str(m) for m in data.get("models", []) or []],
            rpm_limit=int(data.get("rpm_limit", 30) or 30),
            health=str(data.get("health", HEALTH_UNKNOWN)),
            last_probe=float(data.get("last_probe", 0.0) or 0.0),
            last_error=str(data.get("last_error", "")),
            requests_made=int(data.get("requests_made", 0) or 0),
            disabled=bool(data.get("disabled", False)),
            metadata=dict(data.get("metadata", {}) or {}),
        )


class CredentialVault:
    """Thread-safe, multi-account credential store.

    The vault is intentionally *process local* but file backed, so the head node
    and the auto-spawner share the same source of truth across restarts.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path).expanduser() if path else None
        self._credentials: dict[str, list[Credential]] = {}
        self._lock = threading.RLock()
        self._loaded = False

    # ------------------------------------------------------------------ #
    @property
    def path(self) -> Path | None:
        return self._path

    def load(self, *, force: bool = False) -> CredentialVault:
        """Load credentials from disk and overlay environment variables."""
        with self._lock:
            if self._loaded and not force:
                return self
            self._credentials = {}
            if self._path and self._path.is_file():
                try:
                    raw = json.loads(self._path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise CredentialStoreError(f"cannot read credential store {self._path}: {exc}") from exc
                if isinstance(raw, dict):
                    for provider, entries in raw.items():
                        bucket: list[Credential] = []
                        if isinstance(entries, list):
                            for entry in entries:
                                if not isinstance(entry, dict):
                                    continue
                                entry.setdefault("provider", provider)
                                bucket.append(Credential.from_dict(entry))
                        elif isinstance(entries, dict):
                            entries.setdefault("provider", provider)
                            bucket.append(Credential.from_dict(entries))
                        self._credentials[provider.lower()] = bucket
            self._overlay_environment()
            self._loaded = True
            return self

    def _overlay_environment(self) -> None:
        """Merge ``HYPER_CRED_<PROVIDER>_<LABEL>`` variables into the vault."""
        prefix = "HYPER_CRED_"
        for key, value in os.environ.items():
            if not key.startswith(prefix) or not value:
                continue
            remainder = key[len(prefix) :].lower()
            for provider in PROVIDERS:
                if remainder.startswith(provider + "_"):
                    label = remainder[len(provider) + 1 :]
                    break
            else:
                provider, _, label = remainder.partition("_")
            if not provider:
                continue
            self._upsert(Credential(provider=provider, label=label or "default", token=value))

    def save(self) -> None:
        """Persist the vault to disk with ``0600`` permissions."""
        if not self._path:
            return
        with self._lock:
            payload = {
                provider: [cred.to_dict(redact=False) for cred in creds] for provider, creds in self._credentials.items()
            }
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self._path.with_suffix(self._path.suffix + ".tmp")
                tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
                os.chmod(tmp, 0o600)
                tmp.replace(self._path)
                os.chmod(self._path, 0o600)
            except OSError as exc:
                raise CredentialStoreError(f"cannot write credential store {self._path}: {exc}") from exc

    # ------------------------------------------------------------------ #
    def _upsert(self, credential: Credential) -> Credential:
        bucket = self._credentials.setdefault(credential.provider, [])
        for index, existing in enumerate(bucket):
            if existing.label == credential.label:
                bucket[index] = credential
                return credential
        bucket.append(credential)
        return credential

    def add(self, credential: Credential) -> Credential:
        """Add or replace a credential."""
        with self._lock:
            if not self._loaded:
                self.load()
            stored = self._upsert(credential)
            self.save()
            return stored

    def remove(self, provider: str, label: str) -> bool:
        """Delete one credential; returns ``True`` when something was removed."""
        with self._lock:
            bucket = self._credentials.get(provider.lower(), [])
            remaining = [cred for cred in bucket if cred.label != label]
            if len(remaining) == len(bucket):
                return False
            self._credentials[provider.lower()] = remaining
            self.save()
            return True

    def get(self, provider: str, label: str | None = None) -> Credential | None:
        """Return a specific credential, or the first for the provider."""
        with self._lock:
            if not self._loaded:
                self.load()
            bucket = self._credentials.get(provider.lower(), [])
            if not bucket:
                return None
            if label is None:
                return bucket[0]
            for cred in bucket:
                if cred.label == label:
                    return cred
            return None

    def list(self, provider: str | None = None) -> list[Credential]:
        """All credentials (optionally filtered by provider)."""
        with self._lock:
            if not self._loaded:
                self.load()
            if provider is None:
                out: list[Credential] = []
                for bucket in self._credentials.values():
                    out.extend(bucket)
                return sorted(out, key=lambda c: (c.provider, c.label))
            return sorted(self._credentials.get(provider.lower(), []), key=lambda c: c.label)

    def providers(self) -> list[str]:
        """Providers that have at least one credential registered."""
        with self._lock:
            if not self._loaded:
                self.load()
            return sorted(self._credentials.keys())

    def pick(self, provider: str, *, exclude: Iterable[str] = ()) -> Credential | None:
        """Pick the healthiest usable credential for ``provider``.

        Accounts that are disabled, invalid, exhausted or currently
        rate-limited are skipped, so the caller silently fails over to another
        account (or another provider).  ``None`` means *no account of this
        provider can serve traffic right now*.
        """
        excluded = set(exclude)
        candidates = [
            cred
            for cred in self.list(provider)
            if cred.token
            and not cred.disabled
            and cred.health not in (HEALTH_INVALID, HEALTH_EXHAUSTED)
            and cred.label not in excluded
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda c: (_health_rank(c.health), c.rate_limited(), c.last_error != "", c.label))
        for cred in candidates:
            if not cred.rate_limited():
                return cred
        return None

    def pick_any(self, provider: str) -> Credential | None:
        """Return any credential for ``provider``, ignoring rate limits."""
        for cred in self.list(provider):
            if cred.token and not cred.disabled and cred.health not in (HEALTH_INVALID, HEALTH_EXHAUSTED):
                return cred
        return None

    def mark(self, provider: str, label: str, *, health: str | None = None, error: str | None = None) -> None:
        """Update the health state of one credential."""
        with self._lock:
            cred = self.get(provider, label)
            if cred is None:
                return
            if health is not None:
                cred.health = health
            if error is not None:
                cred.last_error = error
            cred.last_probe = now_s()

    def record_usage(self, provider: str, label: str) -> None:
        """Increment the rate window counter for one credential."""
        with self._lock:
            cred = self.get(provider, label)
            if cred is not None:
                cred.record_request()

    def snapshot(self) -> list[dict[str, Any]]:
        """Redacted view of the whole vault (safe for the mobile dashboard)."""
        return [cred.to_dict(redact=True) for cred in self.list()]

    def has_usable(self, provider: str) -> bool:
        """True when at least one account for ``provider`` can serve traffic."""
        return self.pick(provider) is not None


def _health_rank(health: str) -> int:
    return {
        HEALTH_HEALTHY: 0,
        HEALTH_UNKNOWN: 1,
        HEALTH_DEGRADED: 2,
        HEALTH_DISABLED: 3,
        HEALTH_EXHAUSTED: 4,
        HEALTH_INVALID: 5,
    }.get(health, 1)


_default_vault: CredentialVault | None = None


def get_vault(path: str | Path | None = None) -> CredentialVault:
    """Process-wide credential vault singleton."""
    global _default_vault
    if _default_vault is None:
        _default_vault = CredentialVault(path).load()
    return _default_vault


def reset_vault() -> None:
    """Drop the cached vault (test helper)."""
    global _default_vault
    _default_vault = None


__all__ = [
    "HEALTH_UNKNOWN",
    "HEALTH_HEALTHY",
    "HEALTH_DEGRADED",
    "HEALTH_EXHAUSTED",
    "HEALTH_INVALID",
    "HEALTH_DISABLED",
    "Credential",
    "CredentialVault",
    "get_vault",
    "reset_vault",
]
