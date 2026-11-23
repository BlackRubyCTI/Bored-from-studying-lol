"""The autopilot switch of the phone and the notifier of the orchestrator."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from core.config import Settings
from head_node.app import create_app
from head_node.orchestrator import Orchestrator
from training.autopilot_evolver import AutopilotEvolver


@pytest.fixture
def settings(tmp_path) -> Settings:
    """The settings of a cluster that learns (the engine is on)."""
    return Settings(
        data_dir=str(tmp_path / "data"),
        log_level="WARNING",
        log_json=False,
        api_key="",
        mesh_token="",
        admin_token="",
        evolution_enabled=True,
    )


@pytest.fixture
def client(settings: Settings) -> TestClient:
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


# --------------------------------------------------------------------------- #
# The switch
# --------------------------------------------------------------------------- #
def test_the_status_exposes_the_autopilot(client: TestClient) -> None:
    payload = client.get("/v1/status").json()
    assert "autopilot" in payload
    assert isinstance(payload["autopilot"], bool)


def test_the_autopilot_can_be_turned_off(client: TestClient) -> None:
    response = client.post("/v1/evolution/toggle", json={"enabled": False})
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["enabled"] is False
    assert client.get("/v1/status").json()["autopilot"] is False


def test_the_autopilot_can_be_turned_on_again(client: TestClient) -> None:
    client.post("/v1/evolution/toggle", json={"enabled": False})
    response = client.post("/v1/evolution/toggle", json={"enabled": True})
    assert response.status_code == 200
    assert response.json()["enabled"] is True
    assert client.get("/v1/status").json()["autopilot"] is True


def test_the_switch_toggles_without_a_payload(client: TestClient) -> None:
    first = client.post("/v1/evolution/toggle", json={}).json()
    second = client.post("/v1/evolution/toggle", json={}).json()
    assert first["enabled"] is not second["enabled"]


def test_the_switch_needs_the_admin_token(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        log_level="WARNING",
        api_key="",
        admin_token="admin-secret",
        evolution_enabled=True,
    )
    app = create_app(settings)
    with TestClient(app) as test_client:
        assert test_client.post("/v1/evolution/toggle", json={"enabled": False}).status_code == 401
        ok = test_client.post(
            "/v1/evolution/toggle",
            json={"enabled": False},
            headers={"x-admin-token": "admin-secret"},
        )
        assert ok.status_code == 200


def test_the_switch_is_refused_without_the_engine(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        log_level="WARNING",
        api_key="",
        evolution_enabled=False,
    )
    app = create_app(settings)
    with TestClient(app) as test_client:
        response = test_client.post("/v1/evolution/toggle", json={"enabled": False})
        assert response.status_code == 409


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_the_engine_can_be_disabled_at_runtime(tmp_path) -> None:
    """Disabling the engine stops it immediately (the chat wins)."""
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        log_level="WARNING",
        evolution_enabled=True,
    )
    orchestrator = Orchestrator(settings)
    try:
        assert orchestrator.autopilot is not None
        assert orchestrator.autopilot.enabled
        orchestrator.autopilot.set_enabled(False)
        assert not orchestrator.autopilot.enabled
        assert orchestrator.autopilot.status()["enabled"] is False
        orchestrator.autopilot.set_enabled(True)
        assert orchestrator.autopilot.enabled
    finally:
        await orchestrator.shutdown()


@pytest.mark.asyncio
async def test_the_status_of_the_engine(tmp_path) -> None:
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        log_level="WARNING",
        evolution_enabled=True,
    )
    orchestrator = Orchestrator(settings)
    try:
        status = orchestrator.autopilot.status()
        assert "enabled" in status
        assert "state" in status
        assert "preempt_budget_ms" in status
    finally:
        await orchestrator.shutdown()


# --------------------------------------------------------------------------- #
# The notifier
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_the_orchestrator_builds_the_notifier(tmp_path) -> None:
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        log_level="WARNING",
        notifier_enabled=True,
        notifier_telegram_token="token",
        notifier_telegram_chat_id="42",
    )
    orchestrator = Orchestrator(settings)
    try:
        assert orchestrator.notifier is not None
        assert orchestrator.notifier.enabled
        assert orchestrator.notifier.targets == ["telegram"]
    finally:
        await orchestrator.shutdown()


@pytest.mark.asyncio
async def test_there_is_no_notifier_when_it_is_disabled(tmp_path) -> None:
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        log_level="WARNING",
        notifier_enabled=False,
    )
    orchestrator = Orchestrator(settings)
    try:
        assert orchestrator.notifier is None
    finally:
        await orchestrator.shutdown()


@pytest.mark.asyncio
async def test_the_notifier_follows_the_orchestrator_events(tmp_path) -> None:
    """An event of the bus reaches the notifier (which would push it)."""
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        log_level="WARNING",
        notifier_enabled=True,
        notifier_webhook_url="https://push.example/hook",
    )
    orchestrator = Orchestrator(settings)
    try:
        notifier = orchestrator.notifier
        assert notifier is not None
        assert notifier.webhook_url == "https://push.example/hook"
        await orchestrator.startup()
        await asyncio.sleep(0.05)
        # the notifier is subscribed to the bus
        assert orchestrator.events.subscriber_count >= 1
    finally:
        await orchestrator.shutdown()


def test_the_autopilot_class_exposes_the_switch() -> None:
    assert hasattr(AutopilotEvolver, "set_enabled")
