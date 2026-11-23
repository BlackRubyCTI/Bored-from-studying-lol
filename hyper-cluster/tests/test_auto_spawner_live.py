"""Tests for the headless provisioners and the pre-emptive TTL rotator.

The compute providers are external services, so the tests run a *real* HTTP
server that implements their documented endpoints (Kaggle kernels, Hugging
Face Spaces, GitHub Actions dispatch, OCI Compute).  The provisioner code
under test is the production code: it builds the documented requests, signs
them and reacts to the responses.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from core.config import Settings
from core.credentials import Credential, CredentialVault
from core.protocol import Capabilities, WorkerHello
from head_node.auto_spawner import (
    AutoSpawner,
    EdgeApiProvisioner,
    GitHubActionsProvisioner,
    HuggingFaceSpaceProvisioner,
    KaggleProvisioner,
    OracleProvisioner,
    SpawnRequest,
)
from head_node.worker_registry import WorkerRegistry


class ProviderAPI:
    """A local stand-in for the compute providers' REST APIs."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.broken_tokens: set[str] = set()
        self.created: dict[str, Any] = {}

    def record(self, entry: dict[str, Any]) -> None:
        self.requests.append(entry)

    def find(self, needle: str) -> list[dict[str, Any]]:
        return [r for r in self.requests if needle in r["path"]]


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    api: ProviderAPI

    def log_message(self, *_args: Any) -> None:
        return

    # -- helpers ------------------------------------------------------- #
    def _body(self) -> bytes:
        length = int(self.headers.get("content-length") or 0)
        return self.rfile.read(length) if length else b""

    def _send(self, payload: Any, status: int = 200) -> None:
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _auth_token(self) -> str:
        header = self.headers.get("authorization", "")
        if header.startswith("Basic "):
            import base64

            return base64.b64decode(header[6:]).decode("utf-8")
        return header.replace("Bearer ", "")

    # -- verbs --------------------------------------------------------- #
    def do_POST(self) -> None:  # noqa: N802 - http.server API
        body = self._body()
        token = self._auth_token()
        self.api.record(
            {
                "method": "POST",
                "path": self.path,
                "body": body.decode("utf-8", "replace"),
                "authorization": self.headers.get("authorization", ""),
                "token": token,
                "content_type": self.headers.get("content-type", ""),
            }
        )
        if token in self.api.broken_tokens:
            self._send({"error": "account suspended"}, status=500)
            return

        if self.path.endswith("/kernels/push"):
            self.api.created["kernel"] = self.path
            self._send({"id": "kernel-id", "status": "queued"})
            return
        if self.path.endswith("/api/repos/create"):
            self.api.created["repo"] = json.loads(body or b"{}").get("name", "")
            self._send({"url": "https://huggingface.co/spaces/created"}, status=201)
            return
        if "/upload/main/" in self.path:
            self.api.created.setdefault("uploads", []).append(self.path)
            self._send({"ok": True})
            return
        if "/actions/workflows/" in self.path and self.path.endswith("/dispatches"):
            self.api.created["dispatch"] = json.loads(body or b"{}")
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path.endswith("/20160918/instances"):
            self.api.created["instance"] = json.loads(body or b"{}")
            self._send({"data": {"id": "ocid1.instance.oc1..omega"}}, status=200)
            return
        self._send({"error": "not found"}, status=404)

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        self.api.record(
            {
                "method": "GET",
                "path": self.path,
                "body": "",
                "authorization": self.headers.get("authorization", ""),
                "token": self._auth_token(),
                "content_type": "",
            }
        )
        if "/actions/runs" in self.path:
            self._send({"workflow_runs": [{"id": 4242, "status": "queued"}]})
            return
        if "/models" in self.path:
            self._send({"data": [{"id": "llama-3.3-70b-versatile"}]})
            return
        if "/kernels/status/" in self.path:
            self._send({"status": "complete"})
            return
        self._send({"error": "not found"}, status=404)

    def do_DELETE(self) -> None:  # noqa: N802 - http.server API
        self.api.record(
            {
                "method": "DELETE",
                "path": self.path,
                "body": "",
                "authorization": self.headers.get("authorization", ""),
                "token": self._auth_token(),
                "content_type": "",
            }
        )
        self._send({"ok": True})


@pytest.fixture
def provider_api() -> ProviderAPI:
    api = ProviderAPI()
    _Handler.api = api
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[0], server.server_address[1]
    api.base_url = f"http://{host}:{port}"  # type: ignore[attr-defined]
    try:
        yield api
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        data_dir=str(tmp_path / "data"),
        log_level="WARNING",
        log_json=False,
        spawner_enabled=True,
        spawner_providers=["kaggle"],
        spawner_min_workers=0,
        spawner_tick_seconds=1,
        handoff_lead_seconds=600,
        # A routable head URL is a precondition for provisioning anything: the
        # default head_host is 0.0.0.0 (a bind address) and M6 refuses to hand
        # that to a remote node.  The *refusal* has its own dedicated test in
        # tests/test_m6_truth_and_jail.py.
        public_base_url="https://head.example.test",
    )


def _vault(tmp_path, accounts: list[Credential]) -> CredentialVault:
    vault = CredentialVault(tmp_path / "credentials.json")
    for credential in accounts:
        vault.add(credential)
    return vault


def _spawner(settings: Settings, registry: WorkerRegistry, vault: CredentialVault, provider_api) -> AutoSpawner:
    base = provider_api.base_url  # type: ignore[attr-defined]
    return AutoSpawner(
        settings,
        registry,
        vault,
        provisioners={
            "kaggle": KaggleProvisioner(settings, base_url=base),
            "hf_spaces": HuggingFaceSpaceProvisioner(settings, base_url=f"{base}/api"),
            "github_actions": GitHubActionsProvisioner(settings, base_url=base),
            "oracle": OracleProvisioner(settings, base_url=base),
            "edge_api": EdgeApiProvisioner(settings, base_url=base),
        },
    )


def _request(provider: str = "kaggle", **overrides) -> SpawnRequest:
    payload = {
        "provider": provider,
        "region": "eu-west-1",
        "model": "Qwen2.5-7B-Instruct-GGUF",
        "family": "qwen2",
        "ttl_seconds": 3600,
        "head_url": "http://head:8080",
        "mesh_token": "mesh-token",
        "labels": {"kernel": "hyper-cluster-eu-west-1"},
    }
    payload.update(overrides)
    return SpawnRequest(**payload)


# --------------------------------------------------------------------------- #
# Kaggle
# --------------------------------------------------------------------------- #
async def test_kaggle_provision_pushes_a_kernel(settings, tmp_path, provider_api) -> None:
    vault = _vault(
        tmp_path,
        [
            Credential(
                provider="kaggle",
                label="acct-a",
                token="alice:secret",
                metadata={"owner": "alice"},
            )
        ],
    )
    spawner = _spawner(settings, WorkerRegistry(), vault, provider_api)
    result = await spawner.provision(_request())
    assert result.ok is True, result.error
    assert result.instance_id == "alice/hyper-cluster-eu-west-1"

    push = provider_api.find("/kernels/push")[0]
    assert push["content_type"].startswith("multipart/form-data")
    assert "Basic " in push["authorization"]
    assert "alice:secret" == push["token"]
    assert '"enable_gpu": true' in push["body"].replace("'", '"')
    assert "HYPER_WORKER_HEAD_URL" in push["body"]
    assert "HYPER_WORKER_MESH_TOKEN" in push["body"]
    assert "git clone" in push["body"]
    assert "./worker/bootstrap.sh --install" in push["body"]
    await spawner.aclose()


async def test_multi_account_failover(settings, tmp_path, provider_api) -> None:
    """The first account fails -> the spawner silently uses the second one."""
    vault = _vault(
        tmp_path,
        [
            Credential(provider="kaggle", label="acct-a", token="alice:secret", metadata={"owner": "alice"}),
            Credential(provider="kaggle", label="acct-b", token="bob:secret", metadata={"owner": "bob"}),
        ],
    )
    provider_api.broken_tokens.add("alice:secret")
    spawner = _spawner(settings, WorkerRegistry(), vault, provider_api)
    result = await spawner.provision(_request())
    assert result.ok is True, result.error
    assert result.account_label == "acct-b"
    # The failing account is marked as degraded in the vault.
    assert vault.get("kaggle", "acct-a").health == "degraded"
    assert vault.get("kaggle", "acct-b").health == "healthy"
    await spawner.aclose()


async def test_provisioning_without_an_account_fails_cleanly(settings, tmp_path, provider_api) -> None:
    vault = _vault(tmp_path, [])
    spawner = _spawner(settings, WorkerRegistry(), vault, provider_api)
    result = await spawner.provision(_request())
    assert result.ok is False
    assert "no usable account" in result.error
    await spawner.aclose()


async def test_unknown_provider_is_rejected(settings, tmp_path, provider_api) -> None:
    vault = _vault(tmp_path, [])
    spawner = _spawner(settings, WorkerRegistry(), vault, provider_api)
    result = await spawner.provision(_request(provider="nope"))
    assert result.ok is False
    assert "no provisioner" in result.error
    await spawner.aclose()


# --------------------------------------------------------------------------- #
# Hugging Face Spaces
# --------------------------------------------------------------------------- #
async def test_huggingface_provision_creates_and_uploads(settings, tmp_path, provider_api) -> None:
    vault = _vault(
        tmp_path,
        [Credential(provider="hf_spaces", label="acct-a", token="hf_tok", metadata={"owner": "bob"})],
    )
    spawner = _spawner(settings, WorkerRegistry(), vault, provider_api)
    result = await spawner.provision(_request(provider="hf_spaces"))
    assert result.ok is True, result.error
    assert result.instance_id == "bob/hyper-cluster-eu-west-1"
    uploads = provider_api.created.get("uploads", [])
    assert len(uploads) == 3
    assert any("Dockerfile" in path for path in uploads)
    assert any("README.md" in path for path in uploads)
    assert any("worker_bootstrap.sh" in path for path in uploads)
    await spawner.aclose()


# --------------------------------------------------------------------------- #
# GitHub Actions
# --------------------------------------------------------------------------- #
async def test_github_actions_provision_dispatches_a_workflow(settings, tmp_path, provider_api) -> None:
    vault = _vault(
        tmp_path,
        [
            Credential(
                provider="github_actions",
                label="acct-a",
                token="ghp_tok",
                metadata={"repo": "me/cluster", "workflow": "worker.yml", "ref": "main"},
            )
        ],
    )
    spawner = _spawner(settings, WorkerRegistry(), vault, provider_api)
    result = await spawner.provision(_request(provider="github_actions"))
    assert result.ok is True, result.error
    assert result.instance_id == "me/cluster#4242"
    dispatch = provider_api.created["dispatch"]
    assert dispatch["ref"] == "main"
    assert dispatch["inputs"]["head_url"] == "http://head:8080"
    assert dispatch["inputs"]["mesh_token"] == "mesh-token"
    await spawner.aclose()


# --------------------------------------------------------------------------- #
# Oracle
# --------------------------------------------------------------------------- #
def _rsa_key() -> str:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")


async def test_oracle_provision_signs_the_request(settings, tmp_path, provider_api) -> None:
    vault = _vault(
        tmp_path,
        [
            Credential(
                provider="oracle",
                label="acct-a",
                token=_rsa_key(),
                metadata={
                    "tenancy": "ocid1.tenancy.oc1..omega",
                    "user": "ocid1.user.oc1..omega",
                    "fingerprint": "aa:bb:cc",
                    "compartment": "ocid1.compartment.oc1..omega",
                    "image": "ocid1.image.oc1..omega",
                    "availability_domain": "EU-FRANKFURT-1-AD-1",
                    "subnet_id": "ocid1.subnet.oc1..omega",
                    "ssh_public_key": "ssh-ed25519 AAAA test",
                    "region": "eu-frankfurt-1",
                },
            )
        ],
    )
    spawner = _spawner(settings, WorkerRegistry(), vault, provider_api)
    result = await spawner.provision(_request(provider="oracle", region="eu-frankfurt-1"))
    assert result.ok is True, result.error
    assert result.instance_id == "ocid1.instance.oc1..omega"
    launch = provider_api.created["instance"]
    assert launch["shape"] == "VM.Standard.A1.Flex"
    assert launch["shapeConfig"] == {"ocpus": 4, "memoryInGBs": 24}
    assert launch["compartmentId"] == "ocid1.compartment.oc1..omega"
    assert launch["createVnicDetails"]["subnetId"] == "ocid1.subnet.oc1..omega"
    posted = provider_api.find("/20160918/instances")[0]
    assert 'Signature version="1"' in posted["authorization"]
    assert "keyId=" in posted["authorization"]
    await spawner.aclose()


async def test_oracle_provision_reports_missing_metadata(settings, tmp_path, provider_api) -> None:
    vault = _vault(
        tmp_path,
        [Credential(provider="oracle", label="acct-a", token=_rsa_key(), metadata={})],
    )
    spawner = _spawner(settings, WorkerRegistry(), vault, provider_api)
    result = await spawner.provision(_request(provider="oracle"))
    assert result.ok is False
    assert "compartment" in result.error
    await spawner.aclose()


# --------------------------------------------------------------------------- #
# Edge capacity
# --------------------------------------------------------------------------- #
async def test_edge_provision_verifies_the_account_live(settings, tmp_path, provider_api) -> None:
    vault = _vault(
        tmp_path,
        [
            Credential(
                provider="groq",
                label="acct-a",
                token="gsk_tok",
                base_url=f"{provider_api.base_url}/edge",  # type: ignore[attr-defined]
            )
        ],
    )
    spawner = _spawner(settings, WorkerRegistry(), vault, provider_api)
    result = await spawner.provision(_request(provider="edge_api", labels={"edge_provider": "groq"}))
    assert result.ok is True, result.error
    assert result.instance_id == "groq:acct-a"
    assert result.meta["model"] == "llama-3.3-70b-versatile"
    probe = provider_api.find("/models")[0]
    assert probe["token"] == "gsk_tok"
    await spawner.aclose()


# --------------------------------------------------------------------------- #
# The pre-emptive TTL rotator
# --------------------------------------------------------------------------- #
def _register_worker(
    registry: WorkerRegistry,
    worker_id: str,
    *,
    provider: str = "kaggle",
    session_ends_in: float | None = 300.0,
) -> None:
    """Register a worker; by default its session ends in 5 minutes."""
    metadata = {"session_ends_at": time.time() + session_ends_in} if session_ends_in else {}
    registry.register(
        WorkerHello(
            worker_id=worker_id,
            host="127.0.0.1",
            port=8081,
            capabilities=Capabilities(
                vram_total_gb=16.0,
                vram_free_gb=12.0,
                provider=provider,
                region="eu-west-1",
            ),
            model_loaded="Qwen2.5-7B-Instruct-GGUF",
            model_family="qwen2",
            model_params_b=7.0,
            metadata=metadata,
        ),
        ttl_seconds=120,
    )


async def test_rotator_replaces_an_expiring_worker(settings, tmp_path, provider_api) -> None:
    vault = _vault(
        tmp_path,
        [Credential(provider="kaggle", label="acct-a", token="alice:secret", metadata={"owner": "alice"})],
    )
    registry = WorkerRegistry()
    _register_worker(registry, "wk-old")
    spawner = _spawner(settings, registry, vault, provider_api)

    # The session dies in 120 s -> inside the 10 minute hand-over window.
    actions = await spawner.tick()
    assert any(a.kind == "rotate" and a.ok for a in actions), actions
    assert "wk-old" in spawner._pending  # noqa: SLF001 - internal state

    # The old worker is still serving while the replacement boots.
    assert registry.get("wk-old").healthy is True
    assert registry.get("wk-old").draining is False

    # The replacement registers -> the hand-over completes.
    _register_worker(registry, "wk-new")
    actions = await spawner.tick()
    handoff = [a for a in actions if a.kind == "handoff"]
    assert handoff, actions
    assert handoff[0].worker_id == "wk-old"
    assert handoff[0].replacement == "wk-new"
    assert registry.get("wk-old").draining is True
    await spawner.aclose()


async def test_rotator_ignores_workers_with_a_long_session(settings, tmp_path, provider_api) -> None:
    """A worker whose session runs for hours must not be replaced."""
    vault = _vault(
        tmp_path,
        [Credential(provider="kaggle", label="acct-a", token="alice:secret", metadata={"owner": "alice"})],
    )
    registry = WorkerRegistry()
    _register_worker(registry, "wk-long", session_ends_in=3600.0)
    spawner = _spawner(settings, registry, vault, provider_api)
    actions = await spawner.tick()
    assert not [a for a in actions if a.kind == "rotate"], actions
    await spawner.aclose()


async def test_rotator_replaces_a_silent_worker(settings, tmp_path, provider_api) -> None:
    """A worker that stopped heartbeating is replaced as well."""
    vault = _vault(
        tmp_path,
        [Credential(provider="kaggle", label="acct-a", token="alice:secret", metadata={"owner": "alice"})],
    )
    registry = WorkerRegistry()
    _register_worker(registry, "wk-silent", session_ends_in=None)
    registry.mark_lost("wk-silent", reason="no heartbeat")
    spawner = _spawner(settings, registry, vault, provider_api)
    actions = await spawner.tick()
    assert [a for a in actions if a.kind == "rotate" and a.ok], actions
    await spawner.aclose()


async def test_rotator_waits_for_the_replacement(settings, tmp_path, provider_api) -> None:
    """The old worker keeps serving until the replacement is really there."""
    vault = _vault(
        tmp_path,
        [Credential(provider="kaggle", label="acct-a", token="alice:secret", metadata={"owner": "alice"})],
    )
    registry = WorkerRegistry()
    _register_worker(registry, "wk-old")
    spawner = _spawner(settings, registry, vault, provider_api)
    await spawner.tick()
    # A worker that registered *before* the replacement was provisioned is not
    # a replacement.
    registry.get("wk-old").registered_at = time.time() - 3600
    _register_worker(registry, "wk-early")
    registry.get("wk-early").registered_at = time.time() - 3600
    actions = await spawner.tick()
    assert not [a for a in actions if a.kind == "handoff"], actions
    assert registry.get("wk-old").draining is False
    await spawner.aclose()


async def test_disabled_spawner_does_nothing(settings, tmp_path, provider_api) -> None:
    settings.spawner_enabled = False
    vault = _vault(
        tmp_path,
        [Credential(provider="kaggle", label="acct-a", token="alice:secret", metadata={"owner": "alice"})],
    )
    registry = WorkerRegistry()
    _register_worker(registry, "wk-old")
    spawner = _spawner(settings, registry, vault, provider_api)
    assert await spawner.tick() == []
    await spawner.aclose()


async def test_spawner_keeps_the_minimum_worker_floor(settings, tmp_path, provider_api) -> None:
    settings.spawner_min_workers = 2
    vault = _vault(
        tmp_path,
        [Credential(provider="kaggle", label="acct-a", token="alice:secret", metadata={"owner": "alice"})],
    )
    registry = WorkerRegistry()
    spawner = _spawner(settings, registry, vault, provider_api)
    actions = await spawner.tick()
    scale_outs = [a for a in actions if a.kind == "scale-out"]
    assert len(scale_outs) == 2
    assert all(a.ok for a in scale_outs)
    await spawner.aclose()


async def test_terminate_calls_the_provider(settings, tmp_path, provider_api) -> None:
    vault = _vault(
        tmp_path,
        [Credential(provider="kaggle", label="acct-a", token="alice:secret", metadata={"owner": "alice"})],
    )
    spawner = _spawner(settings, WorkerRegistry(), vault, provider_api)
    result = await spawner.provision(_request())
    assert result.ok is True
    assert await spawner.terminate(result.instance_id) is True
    assert provider_api.find("/kernels/delete/")[0]["method"] == "DELETE"
    await spawner.aclose()


async def test_status_is_redacted(settings, tmp_path, provider_api) -> None:
    vault = _vault(
        tmp_path,
        [Credential(provider="kaggle", label="acct-a", token="alice:secret", metadata={"owner": "alice"})],
    )
    spawner = _spawner(settings, WorkerRegistry(), vault, provider_api)
    status = spawner.status()
    payload = json.dumps(status)
    assert "alice:secret" not in payload
    assert status["enabled"] is True
    await spawner.aclose()


async def test_provider_errors_never_echo_a_secret(settings, tmp_path, provider_api) -> None:
    """A provider that echoes the token back must not leak it."""
    vault = _vault(
        tmp_path,
        [Credential(provider="kaggle", label="acct-a", token="alice:supersecret", metadata={"owner": "alice"})],
    )
    spawner = _spawner(settings, WorkerRegistry(), vault, provider_api)
    result = await spawner.provision(_request())
    assert result.ok is True

    # Simulate a provider that answers with the token in the error body.
    scrubbed = spawner._scrub(  # noqa: SLF001 - the seam under test
        "request failed: token alice:supersecret was rejected",
        [Credential(provider="kaggle", label="acct-a", token="alice:supersecret")],
    )
    assert "alice:supersecret" not in scrubbed
    assert "***" in scrubbed
    await spawner.aclose()
