"""The milestone-4 control plane: memory upload, search, sync and evolution."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from core.config import Settings
from head_node.app import create_app
from head_node.events import EventBus
from head_node.orchestrator import Orchestrator
from head_node.request_store import RequestLedger
from head_node.scheduler import Scheduler
from head_node.worker_registry import WorkerRegistry


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        data_dir=str(tmp_path / "data"),
        log_level="WARNING",
        log_json=False,
        api_key="",
        mesh_token="",
        admin_token="",
        worker_ttl_seconds=60,
        heartbeat_interval_seconds=5,
        hold_request_ttl_seconds=30,
        request_timeout_seconds=20,
        memory_enabled=True,
        evolution_enabled=True,
        evolution_idle_seconds=3600,
        evolution_dataset_size=4,
    )


@pytest.fixture
def client(settings: Settings) -> TestClient:
    registry = WorkerRegistry(ttl_seconds=settings.worker_ttl_seconds)
    engine = Orchestrator(
        settings,
        registry=registry,
        scheduler=Scheduler(settings, registry),
        events=EventBus(),
        ledger=RequestLedger(),
    )
    return TestClient(create_app(settings, orchestrator=engine))


def test_the_memory_reports_its_state(client: TestClient) -> None:
    response = client.get("/v1/memory")
    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is True
    assert body["chunks"] == 0


def test_an_uploaded_file_is_indexed_and_searchable(client: TestClient) -> None:
    response = client.post(
        "/v1/memory/documents",
        files={"file": ("pricing.txt", b"The pro plan costs 20 euros per month.", "text/plain")},
    )
    assert response.status_code == 200
    document = response.json()["document"]
    assert document["name"] == "pricing.txt"
    assert document["chunks"] >= 1

    listed = client.get("/v1/memory/documents")
    assert listed.status_code == 200
    assert any(item["name"] == "pricing.txt" for item in listed.json()["documents"])

    found = client.post("/v1/memory/search", json={"query": "how much is the pro plan", "k": 3})
    assert found.status_code == 200
    hits = found.json()["hits"]
    assert hits
    assert "20 euros" in hits[0]["text"]
    assert hits[0]["source"] == "document"


def test_an_empty_upload_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/v1/memory/documents",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert response.status_code == 422


def test_a_binary_upload_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/v1/memory/documents",
        files={"file": ("blob.bin", b"\x00\x01\x02\xff", "application/octet-stream")},
    )
    assert response.status_code == 415


def test_a_search_without_a_query_is_rejected(client: TestClient) -> None:
    response = client.post("/v1/memory/search", json={})
    assert response.status_code == 422


def test_the_evolution_status_is_exposed(client: TestClient) -> None:
    response = client.get("/v1/evolution/status")
    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is True
    assert body["state"] in ("idle", "waiting-for-idle", "training", "preempted")
    assert body["preempt_budget_ms"] == 2000.0


def test_the_system_prompt_can_be_set(client: TestClient) -> None:
    response = client.post("/v1/evolution/prompt", json={"prompt": "answer in one line"})
    assert response.status_code == 200
    assert response.json()["system_prompt"] == "answer in one line"
    status = client.get("/v1/evolution/status")
    assert status.json()["system_prompt"] == "answer in one line"


def test_a_sync_without_a_target_is_refused(client: TestClient) -> None:
    response = client.post("/v1/memory/sync", json={"action": "push"})
    assert response.status_code == 409
