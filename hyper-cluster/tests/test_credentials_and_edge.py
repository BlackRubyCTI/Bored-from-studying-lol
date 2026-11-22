"""Credential vault and edge-bridge tests."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import httpx
import pytest

from core.config import Settings
from core.credentials import (
    HEALTH_HEALTHY,
    HEALTH_INVALID,
    Credential,
    CredentialVault,
)
from core.errors import ProviderAuthError, ProviderQuotaError, ProviderUnavailableError
from core.protocol import ChatMessage, ChatRequest
from head_node.edge_bridge import EdgeBridge


# --------------------------------------------------------------------------- #
# Vault
# --------------------------------------------------------------------------- #
def test_vault_persists_credentials_with_tight_permissions(tmp_path: Path) -> None:
    path = tmp_path / "credentials.json"
    vault = CredentialVault(path)
    vault.add(Credential(provider="groq", label="acct-a", token="gsk_secret"))
    assert path.is_file()
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600
    payload = json.loads(path.read_text())
    assert payload["groq"][0]["token"] == "gsk_secret"


def test_vault_never_leaks_tokens_in_snapshots(tmp_path: Path) -> None:
    vault = CredentialVault(tmp_path / "credentials.json")
    vault.add(Credential(provider="groq", label="acct-a", token="gsk_secret"))
    snapshot = vault.snapshot()
    assert snapshot[0]["token"] == "***"
    assert "gsk_secret" not in json.dumps(snapshot)


def test_vault_fails_over_to_the_next_account(tmp_path: Path) -> None:
    vault = CredentialVault(tmp_path / "credentials.json")
    vault.add(Credential(provider="groq", label="acct-a", token="a"))
    vault.add(Credential(provider="groq", label="acct-b", token="b"))
    first = vault.pick("groq")
    assert first is not None
    vault.mark("groq", first.label, health=HEALTH_INVALID, error="401")
    second = vault.pick("groq")
    assert second is not None
    assert second.label != first.label


def test_vault_skips_disabled_and_rate_limited_accounts(tmp_path: Path) -> None:
    vault = CredentialVault(tmp_path / "credentials.json")
    vault.add(Credential(provider="groq", label="acct-a", token="a"))
    vault.add(Credential(provider="groq", label="acct-b", token="b"))
    vault.get("groq", "acct-a").disabled = True
    vault.get("groq", "acct-b").rpm_limit = 1
    vault.get("groq", "acct-b").record_request()
    assert vault.pick("groq") is None
    assert vault.has_usable("groq") is False


def test_vault_reads_environment_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HYPER_CRED_CEREBRAS_ACCT_X", "csk_env")
    vault = CredentialVault(tmp_path / "credentials.json").load(force=True)
    credential = vault.get("cerebras", "acct_x")
    assert credential is not None
    assert credential.token == "csk_env"


def test_vault_removal(tmp_path: Path) -> None:
    vault = CredentialVault(tmp_path / "credentials.json")
    vault.add(Credential(provider="groq", label="acct-a", token="a"))
    assert vault.remove("groq", "acct-a") is True
    assert vault.remove("groq", "acct-a") is False
    assert vault.get("groq") is None


# --------------------------------------------------------------------------- #
# Edge bridge
# --------------------------------------------------------------------------- #
def _request() -> ChatRequest:
    return ChatRequest(messages=[ChatMessage(role="user", content="hello edge")], max_tokens=32)


def _bridge(tmp_path: Path, handler) -> tuple[EdgeBridge, CredentialVault]:
    vault = CredentialVault(tmp_path / "credentials.json")
    vault.add(Credential(provider="groq", label="acct-a", token="gsk_test"))
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    settings = Settings(data_dir=str(tmp_path), speculative_edge_providers=["groq"])
    return EdgeBridge(settings, vault=vault, client=client), vault


async def test_edge_bridge_streams_tokens(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        assert request.headers["authorization"] == "Bearer gsk_test"
        body = (
            'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    bridge, _ = _bridge(tmp_path, handler)
    credential = bridge.vault.get("groq")
    assert credential is not None
    chunks = [chunk async for chunk in bridge.stream(credential, _request())]
    assert "".join(chunks) == "Hello"
    await bridge.aclose()


async def test_edge_bridge_marks_quota_exhaustion(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="rate limited")

    bridge, vault = _bridge(tmp_path, handler)
    credential = bridge.vault.get("groq")
    assert credential is not None
    with pytest.raises(ProviderQuotaError):
        await bridge.complete(credential, _request())
    assert vault.get("groq", "acct-a").health == "exhausted"
    await bridge.aclose()


async def test_edge_bridge_marks_invalid_credentials(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="bad key")

    bridge, vault = _bridge(tmp_path, handler)
    credential = bridge.vault.get("groq")
    assert credential is not None
    with pytest.raises(ProviderAuthError):
        await bridge.complete(credential, _request())
    assert vault.get("groq", "acct-a").health == "invalid"
    await bridge.aclose()


async def test_edge_bridge_reports_unavailable_provider(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    bridge, _ = _bridge(tmp_path, handler)
    credential = bridge.vault.get("groq")
    assert credential is not None
    with pytest.raises(ProviderUnavailableError):
        await bridge.complete(credential, _request())
    await bridge.aclose()


async def test_edge_candidates_reflect_the_vault(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": []})

    bridge, _ = _bridge(tmp_path, handler)
    candidates = bridge.candidates()
    assert len(candidates) == 1
    assert candidates[0].provider == "groq"
    assert candidates[0].available is True
    assert bridge.healthy_providers() == ["groq"]
    await bridge.aclose()


async def test_edge_probe_updates_health(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "llama-3.3-70b-versatile"}]})

    bridge, vault = _bridge(tmp_path, handler)
    assert await bridge.probe("groq") is True
    assert vault.get("groq", "acct-a").health == HEALTH_HEALTHY
    await bridge.aclose()


def test_edge_candidate_requires_a_credential(tmp_path: Path) -> None:
    vault = CredentialVault(tmp_path / "credentials.json")
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    settings = Settings(data_dir=str(tmp_path), speculative_edge_providers=["groq"])
    bridge = EdgeBridge(settings, vault=vault, client=client)
    candidates = bridge.candidates()
    assert candidates[0].available is False
    assert "credential" in candidates[0].reason
