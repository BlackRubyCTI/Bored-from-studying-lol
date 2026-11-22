"""Edge bridge -- the "Free Frontier API" tier.

Free inference APIs (Groq, Cerebras, SambaNova, OpenRouter, Google AI Studio,
Fireworks) are wired into the cluster as *instant* accelerators:

* they produce the first tokens of an answer in well under a second, so the
  phone starts rendering immediately even if the GPU mesh is 200 ms away;
* they act as a zero-downtime fallback brain when every GPU worker is busy,
  draining or dead;
* they are the "speculative draft" model that the heavy cluster verifies.

Every provider is reached through its OpenAI-compatible endpoint, which makes
the whole tier one code path with automatic failover between accounts.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx

from core.config import Settings
from core.credentials import Credential, CredentialVault, get_vault
from core.errors import (
    ProviderAuthError,
    ProviderQuotaError,
    ProviderUnavailableError,
)
from core.logging_setup import get_logger
from core.protocol import ChatRequest
from core.providers import get_provider
from head_node.scheduler import EdgeCandidate

LOGGER = get_logger("head_node.edge_bridge")

#: Conservative per-provider throughput estimates (tokens/second) used before
#: the bridge has observed real traffic.
DEFAULT_TPS: dict[str, float] = {
    "groq": 260.0,
    "cerebras": 220.0,
    "sambanova": 130.0,
    "fireworks": 150.0,
    "google_ai_studio": 90.0,
    "openrouter": 55.0,
}


@dataclass
class EdgeObservation:
    """Rolling latency/throughput statistics for one edge endpoint."""

    provider: str
    label: str
    base_latency_ms: float = 120.0
    tokens_per_sec: float = 0.0
    samples: int = 0
    last_error: str = ""
    last_used: float = 0.0

    def record(self, *, latency_ms: float, tokens: int = 0, elapsed_ms: float = 0.0) -> None:
        alpha = 0.3
        self.base_latency_ms = (1 - alpha) * self.base_latency_ms + alpha * max(1.0, latency_ms)
        if elapsed_ms > 0 and tokens > 0:
            tps = tokens / (elapsed_ms / 1000.0)
            self.tokens_per_sec = (1 - alpha) * self.tokens_per_sec + alpha * tps if self.tokens_per_sec else tps
        self.samples += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "label": self.label,
            "base_latency_ms": round(self.base_latency_ms, 2),
            "tokens_per_sec": round(self.tokens_per_sec, 2),
            "samples": self.samples,
            "last_error": self.last_error,
        }


@dataclass
class EdgeBridge:
    """Multi-account, failover-capable bridge to the free inference tier."""

    settings: Settings
    vault: CredentialVault = field(default_factory=get_vault)
    client: httpx.AsyncClient | None = None
    _observations: dict[str, EdgeObservation] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.settings.edge_timeout_seconds, connect=5.0),
                limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
                headers={"user-agent": "hyper-cluster-edge/1.0"},
            )

    # ------------------------------------------------------------------ #
    async def aclose(self) -> None:
        if self.client is not None:
            await self.client.aclose()
            self.client = None

    def _observation(self, provider: str, label: str) -> EdgeObservation:
        key = f"{provider}:{label}"
        obs = self._observations.get(key)
        if obs is None:
            obs = EdgeObservation(provider=provider, label=label, tokens_per_sec=DEFAULT_TPS.get(provider, 60.0))
            self._observations[key] = obs
        return obs

    # ------------------------------------------------------------------ #
    def provider_names(self) -> list[str]:
        """Providers the bridge is allowed to use (configuration + registry)."""
        configured = [p.lower() for p in self.settings.speculative_edge_providers]
        names = [p for p in configured if get_provider(p) is not None]
        if not names:
            names = [p for p in DEFAULT_TPS if get_provider(p) is not None]
        return names

    def candidates(self) -> list[EdgeCandidate]:
        """Every usable edge endpoint with a live credential."""
        out: list[EdgeCandidate] = []
        for provider in self.provider_names():
            credential = self.vault.pick(provider)
            if credential is None:
                out.append(
                    EdgeCandidate(
                        provider=provider,
                        available=False,
                        reason="no usable credential",
                    )
                )
                continue
            obs = self._observation(provider, credential.label)
            out.append(
                EdgeCandidate(
                    provider=provider,
                    label=credential.label,
                    model=credential.default_model(),
                    predicted_tps=obs.tokens_per_sec or DEFAULT_TPS.get(provider, 60.0),
                    base_latency_ms=obs.base_latency_ms,
                    available=True,
                    reason="healthy",
                )
            )
        return out

    def healthy_providers(self) -> list[str]:
        """Names of providers that currently have a usable account."""
        return [c.provider for c in self.candidates() if c.available]

    # ------------------------------------------------------------------ #
    def _payload(self, credential: Credential, request: ChatRequest, *, stream: bool) -> dict[str, Any]:
        spec = get_provider(credential.provider)
        model = credential.default_model()
        if request.model:
            model = request.model
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
            "stream": stream,
        }
        if request.stop:
            body["stop"] = request.stop
        if request.seed is not None:
            body["seed"] = request.seed
        if spec is not None and not spec.openai_compatible:
            body["model"] = model
        return body

    def _endpoint(self, credential: Credential) -> tuple[str, dict[str, str]]:
        spec = get_provider(credential.provider)
        if spec is None:
            raise ProviderUnavailableError(f"unknown edge provider: {credential.provider}")
        base = credential.effective_base_url() or spec.base_url
        headers = {"content-type": "application/json"}
        headers.update(spec.header_for(credential.token))
        return f"{base.rstrip('/')}{spec.chat_path}", headers

    # ------------------------------------------------------------------ #
    async def stream(
        self, credential: Credential, request: ChatRequest
    ) -> AsyncIterator[str]:
        """Stream raw text fragments from an edge provider."""
        if self.client is None:  # pragma: no cover - defensive
            raise ProviderUnavailableError("edge bridge is closed")
        url, headers = self._endpoint(credential)
        body = self._payload(credential, request, stream=True)
        started = time.perf_counter()
        try:
            async with self.client.stream("POST", url, json=body, headers=headers) as response:
                if response.status_code in (401, 403):
                    self.vault.mark(
                        credential.provider, credential.label, health="invalid", error=f"HTTP {response.status_code}"
                    )
                    raise ProviderAuthError(
                        f"{credential.provider} rejected the credential",
                        details={"provider": credential.provider, "label": credential.label},
                    )
                if response.status_code == 429:
                    self.vault.mark(
                        credential.provider, credential.label, health="exhausted", error="rate limited"
                    )
                    raise ProviderQuotaError(
                        f"{credential.provider} rate limit reached",
                        details={"provider": credential.provider, "label": credential.label},
                    )
                if response.status_code >= 400:
                    body_text = (await response.aread()).decode("utf-8", "replace")[:300]
                    self.vault.mark(
                        credential.provider, credential.label, health="degraded", error=f"HTTP {response.status_code}"
                    )
                    raise ProviderUnavailableError(
                        f"{credential.provider} returned HTTP {response.status_code}",
                        details={"provider": credential.provider, "body": body_text},
                    )
                self.vault.mark(credential.provider, credential.label, health="healthy", error="")
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload in ("", "[DONE]"):
                        if payload == "[DONE]":
                            break
                        continue
                    try:
                        parsed = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    for choice in parsed.get("choices", []):
                        delta = choice.get("delta") or {}
                        content = delta.get("content")
                        if content:
                            yield content
                        elif choice.get("text"):
                            yield choice["text"]
        except httpx.HTTPError as exc:
            self.vault.mark(
                credential.provider, credential.label, health="degraded", error=str(exc)
            )
            raise ProviderUnavailableError(
                f"{credential.provider} is unreachable: {exc}",
                details={"provider": credential.provider},
            ) from exc
        finally:
            latency_ms = (time.perf_counter() - started) * 1000.0
            obs = self._observation(credential.provider, credential.label)
            obs.base_latency_ms = 0.7 * obs.base_latency_ms + 0.3 * min(latency_ms, 400.0)
            obs.last_used = time.time()
            self.vault.record_usage(credential.provider, credential.label)

    async def complete(self, credential: Credential, request: ChatRequest) -> str:
        """Run a non-streaming completion through the edge provider."""
        if self.client is None:  # pragma: no cover - defensive
            raise ProviderUnavailableError("edge bridge is closed")
        url, headers = self._endpoint(credential)
        body = self._payload(credential, request, stream=False)
        started = time.perf_counter()
        try:
            response = await self.client.post(url, json=body, headers=headers)
        except httpx.HTTPError as exc:
            self.vault.mark(credential.provider, credential.label, health="degraded", error=str(exc))
            raise ProviderUnavailableError(
                f"{credential.provider} is unreachable: {exc}",
                details={"provider": credential.provider},
            ) from exc
        if response.status_code in (401, 403):
            self.vault.mark(credential.provider, credential.label, health="invalid", error="auth rejected")
            raise ProviderAuthError(f"{credential.provider} rejected the credential")
        if response.status_code == 429:
            self.vault.mark(credential.provider, credential.label, health="exhausted", error="rate limited")
            raise ProviderQuotaError(f"{credential.provider} rate limit reached")
        if response.status_code >= 400:
            self.vault.mark(
                credential.provider, credential.label, health="degraded", error=f"HTTP {response.status_code}"
            )
            raise ProviderUnavailableError(
                f"{credential.provider} returned HTTP {response.status_code}",
                details={"body": response.text[:300]},
            )
        try:
            parsed = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderUnavailableError(f"{credential.provider} returned invalid JSON") from exc
        self.vault.mark(credential.provider, credential.label, health="healthy", error="")
        text = ""
        choices = parsed.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            text = message.get("content") or choices[0].get("text") or ""
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        usage = parsed.get("usage") or {}
        tokens = int(usage.get("completion_tokens") or 0)
        self._observation(credential.provider, credential.label).record(
            latency_ms=elapsed_ms * 0.35, tokens=tokens, elapsed_ms=elapsed_ms
        )
        self.vault.record_usage(credential.provider, credential.label)
        return text

    # ------------------------------------------------------------------ #
    async def probe(self, provider: str) -> bool:
        """Health-probe one provider (used by the rotator and the dashboard)."""
        spec = get_provider(provider)
        if spec is None:
            return False
        credential = self.vault.pick(provider)
        if credential is None:
            return False
        if self.client is None:  # pragma: no cover
            return False
        url = f"{credential.effective_base_url() or spec.base_url}{spec.health_path}"
        headers = dict(spec.header_for(credential.token))
        try:
            response = await self.client.get(url, headers=headers, timeout=10.0)
        except httpx.HTTPError as exc:
            self.vault.mark(provider, credential.label, health="degraded", error=str(exc))
            return False
        if response.status_code in (401, 403):
            self.vault.mark(provider, credential.label, health="invalid", error="auth rejected")
            return False
        if response.status_code == 429:
            self.vault.mark(provider, credential.label, health="exhausted", error="rate limited")
            return False
        healthy = response.status_code < 400
        self.vault.mark(
            provider,
            credential.label,
            health="healthy" if healthy else "degraded",
            error="" if healthy else f"HTTP {response.status_code}",
        )
        return healthy

    def stats(self) -> dict[str, Any]:
        """Telemetry for the mobile dashboard."""
        return {
            "providers": self.provider_names(),
            "healthy": self.healthy_providers(),
            "observations": [obs.to_dict() for obs in self._observations.values()],
            "credentials": self.vault.snapshot(),
        }


__all__ = ["EdgeBridge", "EdgeObservation", "DEFAULT_TPS"]
