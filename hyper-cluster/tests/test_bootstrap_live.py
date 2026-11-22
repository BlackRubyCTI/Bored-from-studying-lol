"""Tests for the zero-touch worker bootstrap script.

The script is executed for real (``bash``), against a local HTTP origin and a
local cache directory, so the download / checksum / cache logic is verified
end to end.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = PROJECT_ROOT / "worker" / "bootstrap.sh"


def run_bootstrap(script: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run a snippet with bootstrap.sh sourced."""
    environment = dict(os.environ)
    environment.update(env or {})
    return subprocess.run(
        ["bash", "-c", f"set -euo pipefail\nsource '{BOOTSTRAP}'\n{script}\n"],
        capture_output=True,
        text=True,
        timeout=120,
        env=environment,
        cwd=str(PROJECT_ROOT),
    )


class _ArtifactHandler(BaseHTTPRequestHandler):
    payload = b"hyper-cluster-bootstrap-artifact"

    protocol_version = "HTTP/1.1"

    def log_message(self, *_args) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        if self.path.startswith("/artifact"):
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(self.payload)))
            self.end_headers()
            self.wfile.write(self.payload)
        elif self.path.startswith("/broken"):
            body = b"corrupted"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()


@pytest.fixture(scope="module")
def origin() -> str:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ArtifactHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[0], server.server_address[1]
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()


def test_bootstrap_script_is_valid_bash() -> None:
    completed = subprocess.run(
        ["bash", "-n", str(BOOTSTRAP)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_bootstrap_reports_the_environment() -> None:
    completed = subprocess.run(
        [str(BOOTSTRAP), "--print-env"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "HYPER_WORKER_HEAD_URL=" in completed.stdout
    assert "HYPER_WORKER_MODEL_PATH=" in completed.stdout


def test_bootstrap_check_mode_changes_nothing(tmp_path: Path) -> None:
    env = {
        "HYPER_WORKER_CACHE": str(tmp_path / "cache"),
        "HYPER_WORKER_DATA": str(tmp_path / "data"),
    }
    completed = subprocess.run(
        [str(BOOTSTRAP), "--check"],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, **env},
    )
    assert completed.returncode == 0, completed.stderr
    assert '"level":"INFO"' in completed.stdout
    assert not (tmp_path / "data" / "venv").exists()


def test_verify_sha256_accepts_and_rejects(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"payload")
    digest = hashlib.sha256(b"payload").hexdigest()
    completed = run_bootstrap(
        f"verify_sha256 '{artifact}' '{digest}' && echo MATCH_OK\n"
        f"if verify_sha256 '{artifact}' '{'0' * 64}'; then echo WRONG; else echo MISMATCH_OK; fi"
    )
    assert completed.returncode == 0, completed.stderr
    assert "MATCH_OK" in completed.stdout
    assert "MISMATCH_OK" in completed.stdout


def test_fetch_artifact_downloads_verifies_and_caches(origin: str, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    target = cache / "artifact.bin"
    digest = hashlib.sha256(_ArtifactHandler.payload).hexdigest()
    env = {"HYPER_WORKER_CACHE": str(cache)}

    first = run_bootstrap(
        f"fetch_artifact '{origin}/artifact' '{target}' '{digest}' 'artifact'",
        env=env,
    )
    assert first.returncode == 0, first.stderr
    assert target.exists()
    assert target.read_bytes() == _ArtifactHandler.payload
    assert "checksum" in first.stdout

    # The second call must reuse the cached artefact (no download).
    second = run_bootstrap(
        f"fetch_artifact '{origin}/artifact' '{target}' '{digest}' 'artifact'",
        env=env,
    )
    assert second.returncode == 0, second.stderr
    assert "already cached" in second.stdout


def test_fetch_artifact_discards_a_corrupted_download(origin: str, tmp_path: Path) -> None:
    target = tmp_path / "cache" / "artifact.bin"
    completed = run_bootstrap(
        f"fetch_artifact '{origin}/broken' '{target}' '{'0' * 64}' 'artifact' || true"
    )
    assert "checksum mismatch" in completed.stderr
    assert not target.exists()


def test_fetch_artifact_fails_on_a_404(origin: str, tmp_path: Path) -> None:
    target = tmp_path / "cache" / "missing.bin"
    completed = run_bootstrap(
        f"fetch_artifact '{origin}/missing' '{target}' '' 'artifact'"
    )
    assert completed.returncode == 1
    assert "failed to download" in completed.stderr
    assert not target.exists()


def test_env_file_is_generated_with_the_model_path(tmp_path: Path) -> None:
    env_file = tmp_path / "worker.env"
    completed = run_bootstrap(
        "write_env\n",
        env={
            "HYPER_WORKER_DATA": str(tmp_path / "data"),
            "HYPER_WORKER_ENV_FILE": str(env_file),
            "HYPER_WORKER_HEAD_URL": "http://head.example:8080",
            "HYPER_WORKER_MESH_TOKEN": "mesh-token",
        },
    )
    assert completed.returncode == 0, completed.stderr
    content = env_file.read_text(encoding="utf-8")
    assert "HYPER_WORKER_HEAD_URL=http://head.example:8080" in content
    assert "HYPER_WORKER_MESH_TOKEN=mesh-token" in content
    assert "HYPER_WORKER_MODEL_PATH=" in content
    assert oct(env_file.stat().st_mode)[-3:] == "600"


def test_platform_detection_identifies_this_host() -> None:
    completed = run_bootstrap("detect_platform; echo \"$PLATFORM/$CPU_ARCH\"")
    assert completed.returncode == 0, completed.stderr
    platform = completed.stdout.strip().splitlines()[-1]
    assert platform.startswith("linux/") or platform.startswith("darwin/")


def test_service_unit_is_generated(tmp_path: Path) -> None:
    """The systemd unit must be rendered from the resolved environment."""
    unit = tmp_path / "hyper-cluster-worker.service"
    completed = run_bootstrap(
        f"write_unit '{unit}'",
        env={
            "HYPER_WORKER_DATA": str(tmp_path / "data"),
            "HYPER_WORKER_VENV": str(tmp_path / "venv"),
            "HYPER_WORKER_ENV_FILE": str(tmp_path / "worker.env"),
        },
    )
    assert completed.returncode == 0, completed.stderr
    content = unit.read_text(encoding="utf-8")
    assert "[Unit]" in content
    assert "[Service]" in content
    assert "ExecStart=" in content
    assert "Restart=always" in content
    assert "-m worker.worker_agent" in content


def test_bootstrap_is_idempotent(tmp_path: Path) -> None:
    """Running the provisioning steps twice must not change anything."""
    env = {
        "HYPER_WORKER_CACHE": str(tmp_path / "cache"),
        "HYPER_WORKER_DATA": str(tmp_path / "data"),
    }
    script = (
        "write_env\n"
        "write_unit \"${HYPER_WORKER_DATA}/unit.service\"\n"
    )
    first = run_bootstrap(script, env=env)
    assert first.returncode == 0, first.stderr
    env_file = Path(env["HYPER_WORKER_DATA"]) / "worker.env"
    unit = Path(env["HYPER_WORKER_DATA"]) / "unit.service"
    first_env = env_file.read_text(encoding="utf-8")
    first_unit = unit.read_text(encoding="utf-8")

    second = run_bootstrap(script, env=env)
    assert second.returncode == 0, second.stderr
    assert env_file.read_text(encoding="utf-8") == first_env
    assert unit.read_text(encoding="utf-8") == first_unit

def test_unit_path_falls_back_to_the_user_scope(tmp_path: Path) -> None:
    """A non-root host must still get a unit (rootless systemd)."""
    completed = run_bootstrap(
        "unit=$(unit_path | head -1); scope=$(unit_path | tail -1); echo \"${scope} ${unit}\""
    )
    assert completed.returncode == 0, completed.stderr
    scope, unit = completed.stdout.strip().splitlines()[-1].split(" ", 1)
    assert scope in ("system", "user")
    assert unit.endswith("hyper-cluster-worker.service")


def test_systemd_usable_never_raises() -> None:
    completed = run_bootstrap("if systemd_usable; then echo YES; else echo NO; fi")
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().splitlines()[-1] in ("YES", "NO")


def test_start_service_falls_back_to_the_background_worker(tmp_path: Path) -> None:
    """Without systemd the worker must still be started."""
    completed = run_bootstrap(
        "systemd_usable() { return 1; }\n"
        "start_background_worker() { echo \"BACKGROUND_STARTED ${DATA_DIR}\"; }\n"
        "start_service\n"
    )
    assert completed.returncode == 0, completed.stderr
    assert "BACKGROUND_STARTED" in completed.stdout


def test_background_worker_writes_its_log_and_pid(tmp_path: Path) -> None:
    """The background fallback records where it logs and which pid it runs."""
    data_dir = tmp_path / "data"
    env_file = data_dir / "worker.env"
    env_file.parent.mkdir(parents=True, exist_ok=True)
    env_file.write_text("HYPER_WORKER_BIND_PORT=0\n", encoding="utf-8")
    completed = run_bootstrap(
        "start_background_worker\nsleep 1\n"
        "test -f \"${HYPER_WORKER_DATA}/worker.pid\" && echo PID_OK\n"
        "test -f \"${HYPER_WORKER_DATA}/worker.log\" && echo LOG_OK\n",
        env={"HYPER_WORKER_DATA": str(data_dir)},
    )
    assert completed.returncode == 0, completed.stderr
    assert "PID_OK" in completed.stdout
    assert "LOG_OK" in completed.stdout
    pid = int((data_dir / "worker.pid").read_text(encoding="utf-8").strip())
    try:
        os.kill(pid, 15)
    except ProcessLookupError:
        pass
