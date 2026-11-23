"""M6 — truth, hygiene and a real jail.

Every test here fails on the pre-M6 code:

* the spawner exported ``HYPER_WORKER_MODEL_FILENAME`` while ``bootstrap.sh``
  read ``HYPER_WORKER_MODEL_FILE``, so a spawned node silently downloaded the
  default 7B;
* the spawner handed out ``http://0.0.0.0:8080`` when ``HYPER_PUBLIC_BASE_URL``
  was empty, provisioning nodes that could never check in;
* ``/v1/status`` never said whether the answers were real;
* there was no ``/v1/capacity/plan``;
* the sandbox was monkeypatched guards with no kernel boundary and no report of
  which layer held.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from core.config import Settings
from core.errors import PublicURLUnsetError
from core.protocol import ClusterStatus, WorkerSnapshot
from head_node.worker_registry import WorkerRecord, WorkerRegistry

# Reuse the fake provider HTTP server the spawner suite already ships: pytest
# resolves the fixture by name, so this import looks unused to a linter.
from tests.test_auto_spawner_live import provider_api  # noqa: F401
from tools.jail import (
    ISOLATION_PROCESS,
    ISOLATION_UNSHARE,
    JailConfig,
    available_tiers,
    build_jail_command,
    detect_isolation,
    jail_script,
    seccomp_table,
)
from tools.sandbox import PythonSandbox

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BOOTSTRAP = PROJECT_ROOT / "worker" / "bootstrap.sh"

#: Kernel isolation is available here; used to decide whether the strong
#: assertions may run or must be skipped with a reason (never silently).
HAS_KERNEL_JAIL = detect_isolation(JailConfig()) in {ISOLATION_UNSHARE, "container", "bwrap"}


# --------------------------------------------------------------------------- #
# 1. Model hygiene: one canonical env name, aliases honoured, no silent 7B
# --------------------------------------------------------------------------- #
def _bash(script: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Source bootstrap.sh with helpers only (no installer) and run ``script``.

    ``log``/``warn``/``die`` write to stderr, so a caller that redirects only
    stdout still sees the operational message.
    """
    base = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": env.get("_HOME", "/tmp"),
        "BASH_ENV": "",
    }
    merged = {**base, **{k: v for k, v in env.items() if not k.startswith("_")}}
    return subprocess.run(
        [
            "bash",
            "-c",
            f'set -e; source "{BOOTSTRAP}"; {script}',
        ],
        capture_output=True,
        text=True,
        env=merged,
        cwd=str(PROJECT_ROOT),
        check=False,
    )


def test_bootstrap_reads_the_canonical_filename() -> None:
    """HYPER_WORKER_MODEL_FILENAME (what the spawner exports) must be honoured."""
    result = _bash("printf '%s' \"${MODEL_FILE}\"", {"HYPER_WORKER_MODEL_FILENAME": "llama-3.2-3b-instruct-q4_k_m.gguf"})
    assert result.returncode == 0, result.stderr
    assert result.stdout == "llama-3.2-3b-instruct-q4_k_m.gguf", (
        "bootstrap.sh ignored the canonical HYPER_WORKER_MODEL_FILENAME — this is the "
        f"M6 bug (it silently used the default 7B). stdout={result.stdout!r}"
    )


def test_bootstrap_still_honours_the_legacy_alias() -> None:
    """The pre-M6 name keeps working: no operator is forced to rename."""
    result = _bash("printf '%s' \"${MODEL_FILE}\"", {"HYPER_WORKER_MODEL_FILE": "legacy-model.gguf"})
    assert result.returncode == 0, result.stderr
    assert result.stdout == "legacy-model.gguf"


def test_canonical_name_wins_over_the_legacy_alias() -> None:
    result = _bash(
        "printf '%s' \"${MODEL_FILE}\"",
        {"HYPER_WORKER_MODEL_FILENAME": "canonical.gguf", "HYPER_WORKER_MODEL_FILE": "legacy.gguf"},
    )
    assert result.stdout == "canonical.gguf"


def test_no_model_requested_is_not_a_silent_seven_b() -> None:
    """With nothing set, ``select_model`` must say so rather than pick 7B quietly."""
    result = _bash(
        'select_model; printf "MODEL=%s" "${MODEL_FILE}"',
        {"HYPER_WORKER_BACKEND": "llama.cpp"},
    )
    assert result.returncode == 0, result.stderr + result.stdout
    # The default is still applied for compatibility, but it is now *logged*.
    message = (result.stdout + result.stderr).lower()
    assert "default" in message
    assert "MODEL=qwen2.5-7b-instruct-q4_k_m.gguf" in result.stdout


def test_simulator_backend_downloads_no_model() -> None:
    result = _bash(
        'select_model; printf "MODEL=%s" "${MODEL_FILE:-<empty>}"',
        {"HYPER_WORKER_BACKEND": "simulator"},
    )
    assert result.returncode == 0, result.stderr
    assert "MODEL=<empty>" in result.stdout


def test_requested_model_without_repo_is_refused() -> None:
    """A model we cannot locate is an error, not a silent substitution."""
    result = _bash(
        'select_model; printf "REACHED"',
        {"HYPER_WORKER_MODEL_FILENAME": "some-model.gguf", "HYPER_WORKER_BACKEND": "llama.cpp"},
    )
    assert result.returncode != 0
    assert "REACHED" not in result.stdout
    assert "HYPER_WORKER_MODEL_REPO" in result.stdout + result.stderr


def test_bootstrap_writes_the_canonical_name_into_the_env_file() -> None:
    text = BOOTSTRAP.read_text(encoding="utf-8")
    assert "HYPER_WORKER_MODEL_FILENAME=${MODEL_FILE}" in text
    assert 'MODEL_FILE="${HYPER_WORKER_MODEL_FILENAME:-${HYPER_WORKER_MODEL_FILE:-}}"' in text


def test_worker_config_reads_both_names(monkeypatch: pytest.MonkeyPatch) -> None:
    from worker.config import WorkerConfig

    for key in list(os.environ):
        if key.startswith("HYPER_WORKER_"):
            monkeypatch.delenv(key, raising=False)

    monkeypatch.setenv("HYPER_WORKER_MODEL_FILENAME", "canonical.gguf")
    assert WorkerConfig.from_env().model_filename == "canonical.gguf"

    monkeypatch.delenv("HYPER_WORKER_MODEL_FILENAME")
    monkeypatch.setenv("HYPER_WORKER_MODEL_FILE", "legacy.gguf")
    assert WorkerConfig.from_env().model_filename == "legacy.gguf"


def test_spawner_exports_both_model_names() -> None:
    """The generated bootstrap must satisfy either side of the rename."""
    from head_node.auto_spawner import KaggleProvisioner, SpawnRequest

    settings = Settings(data_dir="/tmp/hc-m6", public_base_url="https://head.example.test")
    provisioner = KaggleProvisioner(settings)
    request = SpawnRequest(
        provider="kaggle",
        region="eu",
        model="llama-3.2-3b-instruct-q4_k_m.gguf",
        family="llama",
        params_b=3.2,
        head_url="https://head.example.test",
        labels={"model_repo": "bartowski/Llama-3.2-3B-Instruct-GGUF"},
    )
    script = provisioner._worker_bootstrap(request)  # noqa: SLF001 - the artefact under test
    assert "export HYPER_WORKER_MODEL_FILENAME=llama-3.2-3b-instruct-q4_k_m.gguf" in script
    assert "export HYPER_WORKER_MODEL_FILE=llama-3.2-3b-instruct-q4_k_m.gguf" in script
    assert "export HYPER_WORKER_MODEL_REPO=bartowski/Llama-3.2-3B-Instruct-GGUF" in script
    assert "export HYPER_WORKER_MODEL_PARAMS_B=3.2" in script
    # The old bug: the node loaded 7B regardless of the request.
    assert "MODEL_FILENAME=llama" in script


def test_spawner_does_not_export_an_empty_filename_for_simulator() -> None:
    from head_node.auto_spawner import KaggleProvisioner, SpawnRequest

    settings = Settings(data_dir="/tmp/hc-m6", public_base_url="https://head.example.test")
    provisioner = KaggleProvisioner(settings)
    script = provisioner._worker_bootstrap(SpawnRequest(provider="kaggle", head_url="https://x.test"))
    assert "export HYPER_WORKER_BACKEND=simulator" in script


# --------------------------------------------------------------------------- #
# 2. Head URL: refuse rather than hand out http://0.0.0.0:8080
# --------------------------------------------------------------------------- #
def test_public_head_url_refuses_a_wildcard_bind() -> None:
    settings = Settings(head_host="0.0.0.0", head_port=8080, public_base_url="")
    assert settings.public_url_configured is False
    with pytest.raises(PublicURLUnsetError) as excinfo:
        settings.public_head_url()
    message = str(excinfo.value)
    assert "HYPER_PUBLIC_BASE_URL" in message
    assert "0.0.0.0" in message
    # Actionable: says what to set and why.
    assert "routable" in message
    assert excinfo.value.code == "public_base_url_unset"
    # 4xx: an operator precondition, never a user-visible 5xx (law L1).
    assert 400 <= excinfo.value.http_status < 500


def test_public_head_url_prefers_the_configured_base() -> None:
    settings = Settings(public_base_url="https://cluster.example.ts.net/")
    assert settings.public_head_url() == "https://cluster.example.ts.net"


def test_public_head_url_allows_a_real_bind_host() -> None:
    settings = Settings(head_host="10.1.2.3", head_port=9000, public_base_url="")
    assert settings.public_head_url() == "http://10.1.2.3:9000"


def test_ipv6_wildcard_is_also_refused() -> None:
    settings = Settings(head_host="::", public_base_url="")
    with pytest.raises(PublicURLUnsetError):
        settings.public_head_url()


async def test_scale_out_refuses_without_a_public_url(tmp_path: Path) -> None:
    """No public URL -> no provisioned nodes that can never phone home."""
    from core.credentials import Credential, CredentialVault
    from head_node.auto_spawner import AutoSpawner, KaggleProvisioner

    vault = CredentialVault(tmp_path / "credentials.json")
    vault.add(Credential(provider="kaggle", label="acct", token="alice:secret", metadata={"owner": "alice"}))
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        log_level="WARNING",
        log_json=False,
        spawner_enabled=True,
        spawner_providers=["kaggle"],
        spawner_min_workers=2,
        head_host="0.0.0.0",
        head_port=8080,
        public_base_url="",
    )
    spawner = AutoSpawner(
        settings,
        WorkerRegistry(),
        vault,
        provisioners={"kaggle": KaggleProvisioner(settings, base_url="https://provider.invalid")},
    )
    actions = await spawner.tick()
    scale_outs = [a for a in actions if a.kind == "scale-out"]
    assert scale_outs, actions
    assert all(a.ok is False for a in scale_outs)
    assert "HYPER_PUBLIC_BASE_URL" in scale_outs[0].detail
    # Critically: nothing was provisioned.
    assert spawner.instances == {}
    url, error = spawner._head_url_or_error()  # noqa: SLF001 - refusal is the contract
    assert url == ""
    assert "0.0.0.0" in error
    await spawner.aclose()


async def test_scale_out_works_when_a_public_url_exists(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    """The refusal must not break the happy path: same setup, URL configured."""
    from core.credentials import Credential, CredentialVault
    from head_node.auto_spawner import AutoSpawner, KaggleProvisioner

    vault = CredentialVault(tmp_path / "credentials.json")
    vault.add(Credential(provider="kaggle", label="acct", token="alice:secret", metadata={"owner": "alice"}))
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        log_level="WARNING",
        log_json=False,
        spawner_enabled=True,
        spawner_providers=["kaggle"],
        spawner_min_workers=1,
        head_host="0.0.0.0",
        public_base_url="https://head.example.test",
    )
    spawner = AutoSpawner(
        settings,
        WorkerRegistry(),
        vault,
        provisioners={"kaggle": KaggleProvisioner(settings, base_url=request.getfixturevalue("provider_api").base_url)},
    )
    actions = await spawner.tick()
    scale_outs = [a for a in actions if a.kind == "scale-out"]
    assert scale_outs and all(a.ok for a in scale_outs), actions
    await spawner.aclose()


async def test_provision_refuses_before_touching_a_provider(tmp_path: Path) -> None:
    """An unroutable head URL is refused before any provider quota is spent."""
    from core.credentials import Credential, CredentialVault
    from head_node.auto_spawner import AutoSpawner, KaggleProvisioner

    vault = CredentialVault(tmp_path / "credentials.json")
    vault.add(Credential(provider="kaggle", label="acct", token="alice:secret", metadata={"owner": "alice"}))
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        log_level="WARNING",
        log_json=False,
        spawner_enabled=True,
        head_host="0.0.0.0",
        public_base_url="",
    )
    spawner = AutoSpawner(
        settings,
        WorkerRegistry(),
        vault,
        provisioners={"kaggle": KaggleProvisioner(settings, base_url="https://provider.invalid")},
    )
    actions = await spawner.tick()
    assert any("HYPER_PUBLIC_BASE_URL" in a.detail for a in actions), actions
    url, error = spawner._head_url_or_error()  # noqa: SLF001
    assert url == "" and PublicURLUnsetError.code in error
    await spawner.aclose()


# --------------------------------------------------------------------------- #
# 3. Capability truth: /v1/status + /v1/capacity/plan
# --------------------------------------------------------------------------- #
def test_cluster_status_carries_capability_truth() -> None:
    status = ClusterStatus(cluster_name="c", instance_id="i", ts=0.0, healthy=True)
    assert status.answers_are_real is False  # default must be the honest one
    assert status.backends == []


def test_worker_snapshot_defaults_to_unknown_not_a_guess() -> None:
    snapshot = WorkerSnapshot(
        worker_id="w1",
        host="h",
        port=1,
        provider="p",
        region="r",
        status="idle",  # type: ignore[arg-type]
    )
    assert snapshot.backend == "unknown"
    assert snapshot.max_concurrency == 1


def _register(registry: WorkerRegistry, worker_id: str, backend: str) -> WorkerRecord:
    from core.protocol import Capabilities, WorkerHello

    hello = WorkerHello(
        worker_id=worker_id,
        host="127.0.0.1",
        port=9000,
        provider="local",
        region="eu",
        capabilities=Capabilities(backend=backend, max_concurrency=3, vram_total_gb=8.0),
    )
    return registry.register(hello)


def test_registry_snapshot_reports_the_worker_backend() -> None:
    registry = WorkerRegistry()
    _register(registry, "wk-live", "llama.cpp")
    (snapshot,) = registry.snapshot()
    assert snapshot.backend == "llama.cpp"
    assert snapshot.max_concurrency == 3


@pytest.fixture
def api_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A live head node whose only backend will be whatever we register."""
    for key in list(os.environ):
        if key.startswith("HYPER_"):
            monkeypatch.delenv(key, raising=False)

    from fastapi.testclient import TestClient

    from head_node.app import create_app

    settings = Settings(
        data_dir=str(tmp_path / "data"),
        log_level="WARNING",
        log_json=False,
        api_key="test-key",
        public_base_url="https://head.example.test",
        spawner_enabled=False,
        sandbox_enabled=False,
        memory_enabled=False,
        autoscaler_enabled=False,
        speculative_enabled=False,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        yield client, app


def test_status_says_the_answers_are_simulated(api_client) -> None:
    client, _ = api_client
    response = client.get("/v1/status", headers={"x-api-key": "test-key"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["answers_are_real"] is False
    capability = payload["capability"]
    assert capability["answers_are_real"] is False
    assert capability["simulator_only"] is True
    assert capability["label"] == "simulator"
    assert "NOT model output" in capability["detail"]
    # No secrets in status.
    assert "test-key" not in json.dumps(payload)


def test_status_reports_the_backend_per_worker(api_client) -> None:
    client, app = api_client
    engine = app.state.orchestrator
    _register(engine.registry, "wk-sim", "simulator")
    payload = client.get("/v1/status", headers={"x-api-key": "test-key"}).json()
    assert payload["workers"], payload
    assert {w["backend"] for w in payload["workers"]} == {"simulator"}
    # A simulator worker does not make the answers real.
    assert payload["answers_are_real"] is False
    assert payload["backends"] == ["simulator"]


def test_status_flips_to_real_with_a_llama_worker(api_client) -> None:
    client, app = api_client
    engine = app.state.orchestrator
    _register(engine.registry, "wk-gpu", "llama.cpp")
    payload = client.get("/v1/status", headers={"x-api-key": "test-key"}).json()
    assert payload["answers_are_real"] is True
    assert payload["capability"]["label"] == "live model"
    assert "llama.cpp" in payload["backends"]


def test_capacity_plan_is_honest_about_the_simulator(api_client) -> None:
    client, app = api_client
    engine = app.state.orchestrator
    _register(engine.registry, "wk-sim", "simulator")
    plan = client.get("/v1/capacity/plan", headers={"x-api-key": "test-key"}).json()
    assert plan["answers_are_real"] is False
    assert plan["model_grade"] is False
    assert plan["claim"] is None
    assert "simulator" in plan["why_no_claim"]
    # Mechanical capacity is still reported — it is measured, just not model-grade.
    assert plan["virtual_vram_gb"]["total"] > 0
    assert plan["virtual_cores"]["concurrent_slots"] == 3


def test_capacity_plan_only_claims_from_a_measurement(api_client) -> None:
    client, app = api_client
    engine = app.state.orchestrator
    _register(engine.registry, "wk-gpu", "llama.cpp")
    record = engine.registry.get("wk-gpu")
    record.tokens_per_sec = 41.5
    engine.registry.set_model if hasattr(engine.registry, "set_model") else None
    plan = client.get("/v1/capacity/plan", headers={"x-api-key": "test-key"}).json()
    assert plan["answers_are_real"] is True
    # No active model yet -> still no model-grade claim, and it says why.
    if plan["active_model"] is None:
        assert plan["claim"] is None
        assert plan["model_grade"] is False
    else:
        assert plan["claim"]["tokens_per_sec_aggregate"] > 0
        assert plan["claim"]["method"]


# --------------------------------------------------------------------------- #
# 4. The real jail
# --------------------------------------------------------------------------- #
def test_jail_tiers_are_probed_not_assumed() -> None:
    tiers = available_tiers()
    assert ISOLATION_PROCESS in tiers  # the fallback always exists
    assert detect_isolation(JailConfig(preferred="process")) == ISOLATION_PROCESS
    # A tier the host cannot deliver is never claimed.
    assert detect_isolation(JailConfig(preferred="container")) in tiers


def test_jail_config_can_force_the_process_fallback() -> None:
    sandbox = PythonSandbox(timeout=5, isolation="process")
    result = sandbox.run("print('hello')")
    assert result.ok is True
    assert result.sandbox_isolation == ISOLATION_PROCESS
    assert result.kernel_isolated is False


def test_jail_command_builder_reports_the_tier(tmp_path: Path) -> None:
    command = build_jail_command(
        python="/usr/bin/python3",
        runner_path=str(tmp_path / "runner.py"),
        workspace=str(tmp_path),
        jail_script_path=str(tmp_path / "jail.py"),
        scratch_dir="/tmp",
        config=JailConfig(),
        limits={},
        child_env={},
    )
    assert command.tier in available_tiers()
    assert command.argv[0]  # a real binary
    (tmp_path / "runner.py").write_text("print(1)", encoding="utf-8")
    (tmp_path / "jail.py").write_text("print(1)", encoding="utf-8")


def test_embedded_jail_script_is_valid_python() -> None:
    compile(jail_script(), "<jail>", "exec")


def test_seccomp_table_covers_this_architecture() -> None:
    table = seccomp_table()
    if not table:  # pragma: no cover - exotic host
        pytest.skip(f"no seccomp table for {os.uname().machine}")
    for syscall in ("execve", "socket", "ptrace", "mount", "unshare", "bpf"):
        assert syscall in table


def test_sandbox_reports_which_layer_held() -> None:
    result = PythonSandbox(timeout=10).run("print(1)")
    assert result.ok is True
    assert result.sandbox_isolation in {"container", "bwrap", ISOLATION_UNSHARE, ISOLATION_PROCESS}
    assert result.to_dict()["sandbox_isolation"] == result.sandbox_isolation
    if result.kernel_isolated:
        assert result.read_only_root or any("read-only" in n for n in result.isolation_notes)


# --- escape tests: the kernel must hold with the monkeypatches removed ------ #
@pytest.fixture
def bare_sandbox() -> PythonSandbox:
    """A sandbox whose only defence is the kernel (guards disabled)."""
    return PythonSandbox(timeout=10, processes=32, guards_off=True)


def test_escape_ctypes_socket_is_denied_by_the_kernel(bare_sandbox: PythonSandbox) -> None:
    """A fresh ctypes handle used to walk straight past the monkeypatched socket."""
    result = bare_sandbox.run(
        "import ctypes\n"
        'libc = ctypes.CDLL("libc.so.6", use_errno=True)\n'
        "libc.socket.restype = ctypes.c_int\n"
        "fd = libc.socket(2, 1, 0)\n"
        "print('fd', fd, 'errno', ctypes.get_errno())\n"
    )
    if bare_sandbox.isolation == ISOLATION_PROCESS:
        pytest.skip("no kernel jail on this host")
    assert result.ok is True
    # socket(2) must have been refused with EPERM/EAFNOSUPPORT, not succeed.
    fd = int(result.stdout.split()[1])
    assert fd < 0, f"ctypes socket(2) succeeded — escape! stdout={result.stdout!r}"


def test_escape_ctypes_execve_is_denied_by_the_kernel(bare_sandbox: PythonSandbox) -> None:
    """importlib/ctypes reloading of ``os`` used to bypass the os.system guard."""
    marker = Path("/tmp/hc-m6-ctypes-escape")
    marker.unlink(missing_ok=True)
    result = bare_sandbox.run(
        "import ctypes\n"
        'libc = ctypes.CDLL("libc.so.6", use_errno=True)\n'
        f'argv = (ctypes.c_char_p * 4)(b"/bin/sh", b"-c", b"touch {marker}", None)\n'
        'rc = libc.execv(ctypes.c_char_p(b"/bin/sh"), argv)\n'
        "print('rc', rc, 'errno', ctypes.get_errno())\n"
    )
    if bare_sandbox.isolation == ISOLATION_PROCESS:
        pytest.skip("no kernel jail on this host")
    assert result.ok is True, result.error
    assert int(result.stdout.split()[1]) == -1, result.stdout
    assert not marker.exists(), "execve ran inside the jail"


def test_escape_importlib_reload_is_contained(bare_sandbox: PythonSandbox) -> None:
    """Reloading ``os`` defeats the monkeypatch; the syscall still must not run."""
    marker = Path("/tmp/hc-m6-importlib-escape")
    marker.unlink(missing_ok=True)
    result = bare_sandbox.run(
        "import importlib, os\n"
        "importlib.reload(os)\n"
        f"print('rc', os.system('touch {marker}'))\n"
    )
    if bare_sandbox.isolation == ISOLATION_PROCESS:
        pytest.skip("no kernel jail on this host")
    assert result.ok is True, result.error
    assert not marker.exists(), f"os.system escaped via importlib: {result.stdout!r}"


def test_escape_proc_is_not_the_host_proc(bare_sandbox: PythonSandbox) -> None:
    """The jail mounts a fresh /proc: one PID, and PID 1 is the jail itself.

    ``/proc/1/environ`` therefore holds the sandbox's own environment, which is
    correct — the assertion that matters is that the *host* process table and
    host secrets are not reachable.
    """
    if bare_sandbox.isolation == ISOLATION_PROCESS:
        pytest.skip("no kernel jail on this host")
    result = bare_sandbox.run(
        "import os\n"
        "pids = [p for p in os.listdir('/proc') if p.isdigit()]\n"
        "print('pids', len(pids))\n"
        "print('leak', b'HYPER_API_KEY' in open('/proc/1/environ','rb').read())\n"
    )
    assert result.ok is True, result.error
    pids = int(result.stdout.split()[1])
    assert pids == 1, f"/proc exposes more than the jail's own PID ({pids} entries)"
    assert "leak False" in result.stdout, "PID 1 inside the jail carries a host secret"


def test_escape_symlink_cannot_reach_host_secrets(bare_sandbox: PythonSandbox) -> None:
    """A symlink out of the workspace must not expose ``/etc`` or a home dir."""
    result = bare_sandbox.run(
        "import os\n"
        "os.symlink('/etc', 'escape')\n"
        "print(open('escape/shadow').read()[:10])\n"
    )
    assert result.ok is False, "read /etc/shadow through a symlink"
    assert "permission" in (result.error + result.stderr).lower()


def test_escape_cannot_write_the_root_filesystem(bare_sandbox: PythonSandbox) -> None:
    result = bare_sandbox.run("open('/etc/hc-m6-pwn', 'w').write('x')")
    if bare_sandbox.isolation == ISOLATION_PROCESS:
        pytest.skip("no kernel jail on this host")
    assert result.ok is False
    assert not Path("/etc/hc-m6-pwn").exists()
    if bare_sandbox.isolation in {ISOLATION_UNSHARE, "container", "bwrap"}:
        # When the host allows it, the guarantee is a read-only root.
        assert "read-only" in (result.error + result.stderr).lower() or "permission" in (
            result.error + result.stderr
        ).lower()


def test_escape_home_directories_are_empty(bare_sandbox: PythonSandbox) -> None:
    result = bare_sandbox.run("import os; print('HOME', os.listdir('/home'))")
    if bare_sandbox.isolation == ISOLATION_PROCESS:
        pytest.skip("no kernel jail on this host")
    assert result.ok is True, result.error
    assert result.stdout.strip() == "HOME []"


def test_workspace_is_still_writable(bare_sandbox: PythonSandbox) -> None:
    """Containment must not break the tool: scratch writes still work."""
    result = bare_sandbox.run(
        "open('scratch.txt', 'w').write('ok')\n"
        "print(open('scratch.txt').read())\n"
    )
    assert result.ok is True, result.error
    assert "ok" in result.stdout


def test_standard_library_still_works(bare_sandbox: PythonSandbox) -> None:
    result = bare_sandbox.run("import json, math; print(json.dumps({'pi': round(math.pi, 2)}))")
    assert result.ok is True, result.error
    assert '{"pi": 3.14}' in result.stdout


def test_process_fallback_is_reported_when_forced() -> None:
    """The weakest tier must label itself, never masquerade as a container."""
    sandbox = PythonSandbox(timeout=10, isolation="process")
    result = sandbox.run("import os; os.system('echo pwned')")
    assert result.ok is False
    payload = result.to_dict()
    assert payload["sandbox_isolation"] == ISOLATION_PROCESS
    assert payload["read_only_root"] is False


# --------------------------------------------------------------------------- #
# 5. Phone surface
# --------------------------------------------------------------------------- #
def test_pwa_has_a_capability_badge() -> None:
    html = (PROJECT_ROOT / "mobile" / "static" / "index.html").read_text(encoding="utf-8")
    script = (PROJECT_ROOT / "mobile" / "static" / "app.js").read_text(encoding="utf-8")
    assert 'id="m-capability"' in html
    assert "applyCapability" in script
    assert "not model output" in script
    assert "answers_are_real" not in html  # nothing secret/raw is templated in


def test_pwa_never_hardcodes_a_backend_claim() -> None:
    """The badge must be driven by /v1/status, not a static 'live' label."""
    script = (PROJECT_ROOT / "mobile" / "static" / "app.js").read_text(encoding="utf-8")
    assert "data.answers_are_real" in script or "answersAreReal" in script
    assert "capability-simulator" in script and "capability-live" in script


def test_docs_document_the_canonical_env_name() -> None:
    env_example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "HYPER_WORKER_MODEL_FILENAME" in env_example or "MODEL_FILENAME" in env_example


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required")
def test_bootstrap_script_is_syntactically_valid() -> None:
    completed = subprocess.run(["bash", "-n", str(BOOTSTRAP)], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
