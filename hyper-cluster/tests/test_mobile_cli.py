"""The Termux command line (``hc``)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from mobile import cli


class FakeCluster:
    """A fake head node (the CLI talks to it over HTTP)."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/v1/status":
            return httpx.Response(
                200,
                json={
                    "cluster_name": "omega",
                    "healthy": True,
                    "worker_count": 3,
                    "healthy_workers": 2,
                    "draining_workers": 1,
                    "free_vram_gb": 12.5,
                    "total_vram_gb": 24.0,
                    "tokens_per_sec_aggregate": 42.0,
                    "active_model": "qwen/qwen2.5-7b",
                    "queued_requests": 2,
                    "in_flight": 1,
                    "edge_healthy": ["groq"],
                },
            )
        if path == "/v1/workers":
            return httpx.Response(
                200,
                json={
                    "workers": [
                        {
                            "worker_id": "wk-1",
                            "model_loaded": "qwen2.5-7b",
                            "vram_free_gb": 12.0,
                            "tokens_per_sec": 30.0,
                            "healthy": True,
                            "draining": False,
                        }
                    ],
                    "totals": {},
                },
            )
        if path == "/v1/memory":
            return httpx.Response(200, json={"chunks": 12, "documents": 3, "by_source": {"pdf": 2}})
        if path == "/v1/memory/search":
            return httpx.Response(
                200,
                json={"hits": [{"source": "pdf", "score": 0.91, "text": "the answer"}]},
            )
        if path == "/v1/memory/documents":
            return httpx.Response(200, json={"document": {"name": "paper.pdf", "chunks": 4}, "pdf_pages": 3})
        if path == "/v1/evolution/status":
            return httpx.Response(
                200,
                json={
                    "state": "idle",
                    "phase": "idle",
                    "enabled": True,
                    "idle_for": 10.0,
                    "idle_seconds": 300,
                    "preempt_latency_ms": 56,
                    "preempt_budget_ms": 2000,
                    "cycles": 3,
                },
            )
        if path == "/v1/evolution/run":
            return httpx.Response(200, json={"seeds": 8, "verified": 5, "training": {"loss_start": 1.0, "loss_end": 0.5}})
        if path == "/v1/chat/stream":
            body = (
                'event: token\ndata: {"text": "hello"}\n\n'
                'event: token\ndata: {"text": " world"}\n\n'
                'event: done\ndata: {"text": "hello world"}\n\n'
            )
            return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
        return httpx.Response(404, json={"detail": "not found"})


@pytest.fixture
def cluster(monkeypatch: pytest.MonkeyPatch) -> FakeCluster:
    """Point the CLI at a fake head node."""
    fake = FakeCluster()
    original = httpx.Client

    def factory(*args, **kwargs) -> httpx.Client:
        kwargs["transport"] = httpx.MockTransport(fake.handler)
        return original(*args, **kwargs)

    monkeypatch.setattr(cli.httpx, "Client", factory)
    return fake


def make_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> cli.Client:
    monkeypatch.setattr(cli, "CONFIG_PATH", tmp_path / "client.json")
    return cli.Client(head_url="http://head", api_key="key")


# --------------------------------------------------------------------------- #
# The parser
# --------------------------------------------------------------------------- #
def test_every_command_is_registered() -> None:
    parser = cli.build_parser()
    samples = {
        "status": [],
        "workers": [],
        "chat": ["hello"],
        "ask": [],
        "watch": [],
        "upload": ["paper.pdf"],
        "memory": [],
        "evolution": [],
        "qr": [],
        "config": [],
    }
    for command, extra in samples.items():
        assert parser.parse_args([command, *extra]).command == command


def test_an_unknown_command_is_refused() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["teleport"])


def test_the_api_key_is_sent() -> None:
    client = cli.Client(head_url="http://head", api_key="secret")
    assert client._headers()["x-api-key"] == "secret"
    client.close()


# --------------------------------------------------------------------------- #
# The commands
# --------------------------------------------------------------------------- #
def test_the_status_command_prints_the_telemetry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cluster: FakeCluster
) -> None:
    client = make_client(tmp_path, monkeypatch)
    assert cli.cmd_status(client, cli.build_parser().parse_args(["status"])) == 0
    out = capsys.readouterr().out
    assert "omega" in out
    assert "workers" in out
    assert "tok/s" in out
    client.close()


def test_the_workers_command_lists_the_mesh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cluster: FakeCluster
) -> None:
    client = make_client(tmp_path, monkeypatch)
    assert cli.cmd_workers(client, cli.build_parser().parse_args(["workers"])) == 0
    out = capsys.readouterr().out
    assert "wk-1" in out
    client.close()


def test_the_chat_command_streams_the_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cluster: FakeCluster
) -> None:
    client = make_client(tmp_path, monkeypatch)
    args = cli.build_parser().parse_args(["chat", "hello", "there"])
    assert cli.cmd_chat(client, args) == 0
    out = capsys.readouterr().out
    assert "hello world" in out
    client.close()


def test_the_memory_command_prints_the_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cluster: FakeCluster
) -> None:
    client = make_client(tmp_path, monkeypatch)
    args = cli.build_parser().parse_args(["memory", "what"])
    assert cli.cmd_memory(client, args) == 0
    out = capsys.readouterr().out
    assert "the answer" in out
    client.close()


def test_the_memory_command_without_a_query_prints_the_statistics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cluster: FakeCluster
) -> None:
    client = make_client(tmp_path, monkeypatch)
    args = cli.build_parser().parse_args(["memory"])
    assert cli.cmd_memory(client, args) == 0
    out = capsys.readouterr().out
    assert "chunks" in out
    assert "12" in out
    client.close()


def test_the_upload_command_indexes_the_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cluster: FakeCluster
) -> None:
    document = tmp_path / "paper.pdf"
    document.write_bytes(b"%PDF-1.4 fake")
    client = make_client(tmp_path, monkeypatch)
    args = cli.build_parser().parse_args(["upload", str(document)])
    assert cli.cmd_upload(client, args) == 0
    out = capsys.readouterr().out
    assert "paper.pdf" in out
    assert "chunks" in out
    client.close()


def test_the_upload_command_refuses_a_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cluster: FakeCluster
) -> None:
    client = make_client(tmp_path, monkeypatch)
    args = cli.build_parser().parse_args(["upload", str(tmp_path / "nope.pdf")])
    with pytest.raises(SystemExit):
        cli.cmd_upload(client, args)
    client.close()


def test_the_evolution_command_prints_the_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cluster: FakeCluster
) -> None:
    client = make_client(tmp_path, monkeypatch)
    args = cli.build_parser().parse_args(["evolution"])
    assert cli.cmd_evolution(client, args) == 0
    out = capsys.readouterr().out
    assert "preempt" in out
    assert "56" in out
    client.close()


def test_the_evolution_run_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cluster: FakeCluster
) -> None:
    client = make_client(tmp_path, monkeypatch)
    args = cli.build_parser().parse_args(["evolution", "run"])
    assert cli.cmd_evolution(client, args) == 0
    out = capsys.readouterr().out
    assert "cycle finished" in out
    client.close()


def test_the_qr_command_draws_the_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cluster: FakeCluster
) -> None:
    client = make_client(tmp_path, monkeypatch)
    args = cli.build_parser().parse_args(["qr", "http://192.168.1.10:8082"])
    assert cli.cmd_qr(client, args) == 0
    out = capsys.readouterr().out
    assert "█" in out
    client.close()


def test_the_config_command_remembers_the_head_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "CONFIG_PATH", tmp_path / "client.json")
    client = cli.Client(head_url="http://head")
    args = cli.build_parser().parse_args(["config", "http://10.0.0.1:9000", "--api-key", "abc"])
    assert cli.cmd_config(client, args) == 0
    stored = json.loads((tmp_path / "client.json").read_text())
    assert stored["head_url"] == "http://10.0.0.1:9000"
    assert stored["api_key"] == "abc"
    client.close()


# --------------------------------------------------------------------------- #
# The configuration file
# --------------------------------------------------------------------------- #
def test_the_configuration_is_read_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "client.json"
    path.write_text(json.dumps({"head_url": "http://10.0.0.2:1234", "api_key": "k"}))
    monkeypatch.setattr(cli, "CONFIG_PATH", path)
    client = cli.Client()
    assert client.head_url == "http://10.0.0.2:1234"
    assert client.api_key == "k"
    client.close()


def test_a_broken_configuration_is_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "client.json"
    path.write_text("{not json")
    monkeypatch.setattr(cli, "CONFIG_PATH", path)
    client = cli.Client()
    assert client.head_url.startswith("http")
    client.close()
