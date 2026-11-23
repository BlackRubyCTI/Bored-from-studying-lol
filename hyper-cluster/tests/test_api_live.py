"""End-to-end API tests against the real FastAPI control plane."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from core.config import Settings
from head_node.app import create_app
from tests.conftest import make_hello
from tests.fake_worker import WorkerServer


@pytest.fixture
def worker() -> WorkerServer:
    server = WorkerServer(worker_id="wk-api", tokens_per_sec=100.0, delay_ms=1.0)
    server.start()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture
def client(settings: Settings) -> TestClient:
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


def test_root_and_health(client: TestClient) -> None:
    root = client.get("/")
    assert root.status_code == 200
    assert root.json()["service"] == "hyper-cluster-omega"

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] in ("ok", "degraded")


def test_status_endpoint_exposes_full_telemetry(client: TestClient) -> None:
    payload = client.get("/v1/status").json()
    assert payload["cluster_name"]
    assert "workers" in payload
    assert "hold_queue" in payload
    assert "edge" in payload
    assert "ledger" in payload


def test_worker_registration_and_inventory(client: TestClient, worker: WorkerServer) -> None:
    payload = make_hello("wk-api", worker.port).model_dump(mode="json")
    response = client.post("/v1/workers/register", json=payload)
    assert response.status_code == 200
    ack = response.json()
    assert ack["accepted"] is True
    assert ack["worker_id"] == "wk-api"
    assert ack["ttl_seconds"] > 0

    inventory = client.get("/v1/workers").json()
    assert inventory["totals"]["workers"] == 1
    assert inventory["workers"][0]["worker_id"] == "wk-api"


def test_heartbeat_updates_worker(client: TestClient, worker: WorkerServer) -> None:
    client.post("/v1/workers/register", json=make_hello("wk-api", worker.port).model_dump(mode="json"))
    beat = {
        "worker_id": "wk-api",
        "status": "busy",
        "inflight": 1,
        "tokens_per_sec": 88.5,
        "vram_free_gb": 18.0,
        "vram_total_gb": 24.0,
    }
    response = client.post("/v1/workers/wk-api/heartbeat", json=beat)
    assert response.status_code == 200
    assert response.json()["ok"] is True
    inventory = client.get("/v1/workers").json()
    assert inventory["workers"][0]["status"] == "busy"
    assert inventory["workers"][0]["tokens_per_sec"] == 88.5


def test_chat_returns_completion(client: TestClient, worker: WorkerServer) -> None:
    client.post("/v1/workers/register", json=make_hello("wk-api", worker.port).model_dump(mode="json"))
    body = {
        "messages": [{"role": "user", "content": "hello from the phone"}],
        "mode": "deep",
        "max_tokens": 64,
        "stream": False,
    }
    response = client.post("/v1/chat", json=body)
    assert response.status_code == 200
    payload = response.json()
    assert payload["error"] is None
    assert "wk-api" in payload["text"]
    assert payload["worker_id"] == "wk-api"


def test_chat_stream_emits_server_sent_events(client: TestClient, worker: WorkerServer) -> None:
    client.post("/v1/workers/register", json=make_hello("wk-api", worker.port).model_dump(mode="json"))
    body = {
        "messages": [{"role": "user", "content": "stream me"}],
        "mode": "deep",
        "max_tokens": 32,
    }
    with client.stream("POST", "/v1/chat/stream", json=body) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        names: list[str] = []
        frames: list[dict] = []
        current = ""
        for line in response.iter_lines():
            if line.startswith("event:"):
                current = line[6:].strip()
            elif line.startswith("data:"):
                payload = json.loads(line[5:].strip())
                frames.append(payload)
                names.append(current)
    assert frames
    assert "start" in names
    assert "token" in names
    assert names[-1] == "done"


def test_chat_without_capacity_is_deferred_not_an_error(client: TestClient) -> None:
    body = {
        "messages": [{"role": "user", "content": "no workers around"}],
        "mode": "deep",
        "max_tokens": 16,
        "stream": False,
        "deadline_seconds": 0.5,
    }
    response = client.post("/v1/chat", json=body)
    assert response.status_code == 202
    payload = response.json()
    assert payload["error"] == "deferred"


def test_request_status_is_retrievable(client: TestClient, worker: WorkerServer) -> None:
    client.post("/v1/workers/register", json=make_hello("wk-api", worker.port).model_dump(mode="json"))
    body = {
        "messages": [{"role": "user", "content": "track me"}],
        "mode": "deep",
        "max_tokens": 32,
        "stream": False,
    }
    response = client.post("/v1/chat", json=body)
    request_id = response.json()["request_id"]
    status = client.get(f"/v1/requests/{request_id}")
    assert status.status_code == 200
    payload = status.json()
    assert payload["status"]["request_id"] == request_id
    assert payload["status"]["state"] == "completed"
    assert payload["result"]["text"]


def test_models_and_queue_endpoints(client: TestClient) -> None:
    models = client.get("/v1/models").json()
    assert models["models"]
    assert all("vram_required_gb" in model for model in models["models"])

    queue = client.get("/v1/queue").json()
    assert queue["stats"]["size"] == 0
    assert queue["entries"] == []


def test_api_key_is_enforced_when_configured(settings: Settings, worker: WorkerServer) -> None:
    settings.api_key = "super-secret-key"
    app = create_app(settings)
    with TestClient(app) as secured:
        assert secured.get("/v1/status").status_code == 401
        assert secured.get("/v1/status", headers={"X-API-Key": "super-secret-key"}).status_code == 200
        body = {"messages": [{"role": "user", "content": "hi"}], "stream": False}
        assert secured.post("/v1/chat", json=body).status_code == 401


def test_mesh_token_is_enforced_for_registration(settings: Settings) -> None:
    settings.mesh_token = "mesh-secret"
    app = create_app(settings)
    with TestClient(app) as secured:
        payload = make_hello("wk-x", 9999).model_dump(mode="json")
        assert secured.post("/v1/workers/register", json=payload).status_code == 401
        ok = secured.post(
            "/v1/workers/register",
            json=payload,
            headers={"X-Mesh-Token": "mesh-secret"},
        )
        assert ok.status_code == 200


def test_unknown_request_returns_404(client: TestClient) -> None:
    assert client.get("/v1/requests/does-not-exist").status_code == 404


def test_drain_and_remove_worker(client: TestClient, worker: WorkerServer) -> None:
    client.post("/v1/workers/register", json=make_hello("wk-api", worker.port).model_dump(mode="json"))
    drained = client.post("/v1/workers/wk-api/drain")
    assert drained.status_code == 200
    inventory = client.get("/v1/workers").json()
    assert inventory["workers"][0]["draining"] is True
    removed = client.delete("/v1/workers/wk-api")
    assert removed.status_code == 200
    assert client.get("/v1/workers").json()["totals"]["workers"] == 0
