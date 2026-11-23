"""The Mobile Mission Control: the PWA shell and the API proxy."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from core.config import Settings
from head_node.app import create_app
from mobile.dashboard import SHELL_FILES, STATIC_DIR, create_dashboard


@pytest.fixture
def head_app(settings: Settings):
    """The real control plane."""
    return create_app(settings)


@pytest.fixture
def dashboard(settings: Settings, head_app):
    """The dashboard, proxying the in-process control plane."""
    head_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=head_app),
        base_url="http://head",
    )
    app = create_dashboard(settings, head_url="http://head", client=head_client)
    with TestClient(app) as test_client:
        yield test_client
    # the client is owned by the fixture, not by the app
    test_client.close()


# --------------------------------------------------------------------------- #
# The shell
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path", sorted(SHELL_FILES))
def test_every_shell_file_is_served(dashboard: TestClient, path: str) -> None:
    response = dashboard.get(path)
    assert response.status_code == 200, path
    assert response.content


def test_the_manifest_is_served_as_a_manifest(dashboard: TestClient) -> None:
    response = dashboard.get("/manifest.webmanifest")
    assert response.status_code == 200
    assert "manifest" in response.headers["content-type"]
    manifest = json.loads(response.text)
    assert manifest["display"] == "standalone"
    assert manifest["theme_color"]


def test_the_shell_is_the_pwa_and_not_the_api(dashboard: TestClient) -> None:
    """The catch-all must not swallow /api/..."""
    response = dashboard.get("/api/v1/status")
    assert response.status_code == 200
    assert "cluster_name" in response.json()


def test_an_unknown_path_is_a_404(dashboard: TestClient) -> None:
    assert dashboard.get("/nope").status_code == 404


def test_the_static_files_exist_on_disk() -> None:
    for name in SHELL_FILES.values():
        assert (STATIC_DIR / name).is_file(), name


# --------------------------------------------------------------------------- #
# The proxy
# --------------------------------------------------------------------------- #
def test_the_proxy_forwards_the_status(dashboard: TestClient) -> None:
    payload = dashboard.get("/api/v1/status").json()
    assert payload["cluster_name"]
    assert "healthy_workers" in payload


def test_the_proxy_forwards_a_post(dashboard: TestClient) -> None:
    response = dashboard.post("/api/v1/memory/search", json={"query": "cluster", "k": 2})
    assert response.status_code == 200
    assert "hits" in response.json()


def test_the_proxy_forwards_the_upload(dashboard: TestClient, tmp_path: Path) -> None:
    document = tmp_path / "paper.txt"
    document.write_text("the cluster remembers this line", encoding="utf-8")
    with document.open("rb") as handle:
        response = dashboard.post(
            "/api/v1/memory/documents",
            files={"file": (document.name, handle, "text/plain")},
        )
    assert response.status_code == 200
    assert response.json()["document"]["chunks"] >= 1


def test_the_proxy_keeps_the_status_code(dashboard: TestClient) -> None:
    assert dashboard.get("/api/v1/nope").status_code == 404


def test_the_config_endpoint_describes_the_cluster(dashboard: TestClient) -> None:
    payload = dashboard.get("/api/config").json()
    assert payload["head"] == "http://head"
    assert payload["features"]["rag"] is True
    assert payload["requires_key"] is False


def test_the_health_endpoint_reports_the_head(dashboard: TestClient) -> None:
    payload = dashboard.get("/health").json()
    assert payload["dashboard"] == "ok"
    assert payload["head_status"] == 200


# --------------------------------------------------------------------------- #
# The authentication
# --------------------------------------------------------------------------- #
def test_the_api_key_is_required_when_the_cluster_has_one(tmp_path: Path) -> None:
    settings = Settings(data_dir=str(tmp_path / "data"), log_level="WARNING", api_key="secret-key")
    head = create_app(settings)
    head_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=head), base_url="http://head")
    app = create_dashboard(settings, head_url="http://head", client=head_client)
    with TestClient(app) as client:
        assert client.get("/api/v1/status").status_code == 401
        ok = client.get("/api/v1/status", headers={"x-api-key": "secret-key"})
        assert ok.status_code == 200


# --------------------------------------------------------------------------- #
# The streaming
# --------------------------------------------------------------------------- #
def test_a_slow_stream_is_relayed_without_buffering(settings: Settings) -> None:
    """A stream that stays open must trickle -- not time out.

    The head node holds a request while the cluster has no capacity; the
    phone must still see the first events immediately.  The upstream is a
    *real* HTTP server here: only then does the read timeout of the proxy
    matter.
    """
    import asyncio

    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse

    from tests.worker_harness import ThreadedServer, free_port

    fake = FastAPI()

    @fake.get("/v1/chat/stream")
    async def chat() -> StreamingResponse:
        async def slow_stream():
            yield "event: start\ndata: {}\n\n"
            await asyncio.sleep(2.0)
            yield "event: token\ndata: {\"text\": \"late\"}\n\n"

        return StreamingResponse(slow_stream(), media_type="text/event-stream")

    server = ThreadedServer(fake, free_port()).start()
    try:
        head_client = httpx.AsyncClient(timeout=httpx.Timeout(0.5))
        app = create_dashboard(settings, head_url=server.base_url, client=head_client)
        with TestClient(app) as client:
            with client.stream("GET", "/api/v1/chat/stream") as response:
                assert response.status_code == 200
                assert "text/event-stream" in response.headers["content-type"]
                first = next(response.iter_lines())
                assert first.startswith("event:")
    finally:
        server.stop()


def test_the_sse_stream_is_proxied(dashboard: TestClient) -> None:
    """The chat stream reaches the phone as Server-Sent Events."""
    with dashboard.stream(
        "POST",
        "/api/v1/chat/stream",
        json={"messages": [{"role": "user", "content": "hello"}], "stream": True},
    ) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        chunks = []
        for line in response.iter_lines():
            if line.startswith("event:"):
                chunks.append(line.split(":", 1)[1].strip())
            if len(chunks) > 2:
                break
    assert chunks, "the stream produced no events"
