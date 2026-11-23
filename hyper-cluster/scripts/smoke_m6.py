#!/usr/bin/env python3
"""Live simulation for Milestone 6 -- truth, hygiene and a real jail.

Everything here is real: a real head node over HTTP, real worker agents, the
real sandbox, and the real kernel isolation primitives.  Nothing is mocked.

The simulation proves the four M6 claims:

* **no silent model** -- the spawner's generated bootstrap and
  ``worker/bootstrap.sh`` agree on the model name; a requested model is never
  replaced by the default 7B, and the simulator downloads nothing at all;
* **no unroutable head URL** -- with ``HYPER_PUBLIC_BASE_URL`` empty the
  spawner refuses and provisions nothing, instead of handing a remote node
  ``http://0.0.0.0:8080``;
* **the truth is on the wire** -- ``/v1/status`` reports the per-worker backend
  and ``answers_are_real``, and ``/v1/capacity/plan`` refuses to attach a
  model-grade claim to simulator answers;
* **the jail is real** -- ``ctypes``, ``importlib``, ``/proc`` and symlink
  escapes are stopped by the kernel with the monkeypatched guards disabled, and
  the result says which isolation tier held.

    python scripts/smoke_m6.py
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import httpx  # noqa: E402

from core.config import Settings  # noqa: E402
from core.errors import PublicURLUnsetError  # noqa: E402
from tools.jail import ISOLATION_PROCESS, JailConfig, available_tiers, detect_isolation  # noqa: E402
from tools.sandbox import PythonSandbox  # noqa: E402

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"

_results: list[tuple[bool, str]] = []


def check(condition: bool, label: str, detail: str = "") -> bool:
    _results.append((bool(condition), label))
    suffix = f" — {detail}" if detail else ""
    print(f"  {PASS if condition else FAIL} {label}{suffix}")
    return bool(condition)


def section(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")


def _bash(script: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    bootstrap = PROJECT_ROOT / "worker" / "bootstrap.sh"
    merged = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp", **env}
    return subprocess.run(
        ["bash", "-c", f'set -e; source "{bootstrap}"; {script}'],
        capture_output=True,
        text=True,
        env=merged,
        cwd=str(PROJECT_ROOT),
        check=False,
    )


# --------------------------------------------------------------------------- #
# 1. No silent model
# --------------------------------------------------------------------------- #
def prove_model_hygiene() -> None:
    section("1. Model hygiene — zero silent 7B spawns")

    canonical = _bash('printf "%s" "${MODEL_FILE}"', {"HYPER_WORKER_MODEL_FILENAME": "llama-3.2-3b-instruct-q4_k_m.gguf"})
    check(
        canonical.stdout == "llama-3.2-3b-instruct-q4_k_m.gguf",
        "bootstrap.sh honours the canonical HYPER_WORKER_MODEL_FILENAME",
        canonical.stdout or canonical.stderr[:120],
    )

    legacy = _bash('printf "%s" "${MODEL_FILE}"', {"HYPER_WORKER_MODEL_FILE": "legacy-only.gguf"})
    check(legacy.stdout == "legacy-only.gguf", "the legacy HYPER_WORKER_MODEL_FILE alias still works")

    both = _bash(
        'printf "%s" "${MODEL_FILE}"',
        {"HYPER_WORKER_MODEL_FILENAME": "canonical.gguf", "HYPER_WORKER_MODEL_FILE": "legacy.gguf"},
    )
    check(both.stdout == "canonical.gguf", "the canonical name wins when both are set")

    simulator = _bash(
        'select_model >/dev/null 2>&1; printf "%s" "${MODEL_FILE:-<none>}"',
        {"HYPER_WORKER_BACKEND": "simulator"},
    )
    check(simulator.stdout == "<none>", "a simulator node downloads no weights at all")

    missing_repo = _bash(
        'select_model >/dev/null; printf "REACHED"',
        {"HYPER_WORKER_MODEL_FILENAME": "unlocatable.gguf", "HYPER_WORKER_BACKEND": "llama.cpp"},
    )
    check(
        missing_repo.returncode != 0 and "REACHED" not in missing_repo.stdout,
        "a requested model with no repo is refused, not swapped for the 7B",
    )

    from head_node.auto_spawner import KaggleProvisioner, SpawnRequest

    settings = Settings(data_dir=tempfile.mkdtemp(), public_base_url="https://head.example.test")
    provisioner = KaggleProvisioner(settings)
    script = provisioner._worker_bootstrap(  # noqa: SLF001 - the artefact under test
        SpawnRequest(
            provider="kaggle",
            model="llama-3.2-3b-instruct-q4_k_m.gguf",
            family="llama",
            params_b=3.2,
            head_url="https://head.example.test",
            labels={"model_repo": "bartowski/Llama-3.2-3B-Instruct-GGUF"},
        )
    )
    check(
        "export HYPER_WORKER_MODEL_FILENAME=llama-3.2-3b-instruct-q4_k_m.gguf" in script
        and "export HYPER_WORKER_MODEL_FILE=llama-3.2-3b-instruct-q4_k_m.gguf" in script,
        "the generated bootstrap exports both the canonical name and the legacy alias",
    )
    check(
        "export HYPER_WORKER_MODEL_PARAMS_B=3.2" in script,
        "the generated bootstrap carries the requested parameter count (VRAM accounting)",
    )

    # The decisive check: feed the generated script's exports into the real
    # bootstrap and confirm the model it would load is the requested one.
    exports = {
        line.split("=", 1)[0].replace("export ", ""): line.split("=", 1)[1]
        for line in script.splitlines()
        if line.startswith("export ") and "=" in line
    }
    resolved = _bash('printf "%s" "${MODEL_FILE}"', exports)
    check(
        resolved.stdout == "llama-3.2-3b-instruct-q4_k_m.gguf",
        "end-to-end: the spawner's export resolves to the requested model in bootstrap.sh",
        resolved.stdout or resolved.stderr[:120],
    )


# --------------------------------------------------------------------------- #
# 2. No unroutable head URL
# --------------------------------------------------------------------------- #
async def prove_head_url_refusal() -> None:
    section("2. Head URL honesty — refuse instead of http://0.0.0.0:8080")

    settings = Settings(head_host="0.0.0.0", head_port=8080, public_base_url="")
    refused = False
    message = ""
    try:
        settings.public_head_url()
    except PublicURLUnsetError as exc:
        refused = True
        message = exc.message
        check(400 <= exc.http_status < 500, "the refusal is a 4xx operator error, never a user-visible 5xx")
    check(refused, "Settings refuses to produce a URL from a wildcard bind host")
    check("HYPER_PUBLIC_BASE_URL" in message, "the error names the variable to set")
    check("routable" in message, "the error explains why (a bind address is not routable)")

    good = Settings(public_base_url="https://cluster.example.ts.net/")
    check(good.public_head_url() == "https://cluster.example.ts.net", "a configured public URL is used verbatim")

    from core.credentials import Credential, CredentialVault
    from head_node.auto_spawner import AutoSpawner, KaggleProvisioner

    tmp = Path(tempfile.mkdtemp())
    vault = CredentialVault(tmp / "credentials.json")
    vault.add(Credential(provider="kaggle", label="acct", token="a:b", metadata={"owner": "alice"}))
    spawner_settings = Settings(
        data_dir=str(tmp / "data"),
        log_level="ERROR",
        log_json=False,
        spawner_enabled=True,
        spawner_providers=["kaggle"],
        spawner_min_workers=2,
        head_host="0.0.0.0",
        public_base_url="",
    )
    registry = None
    from head_node.worker_registry import WorkerRegistry

    registry = WorkerRegistry()
    spawner = AutoSpawner(
        spawner_settings,
        registry,
        vault,
        provisioners={"kaggle": KaggleProvisioner(spawner_settings, base_url="https://provider.invalid")},
    )
    actions = await spawner.tick()
    scale_outs = [a for a in actions if a.kind == "scale-out"]
    check(bool(scale_outs), "a scale-out round is attempted when workers are below the floor")
    check(all(not a.ok for a in scale_outs), "every scale-out action is refused")
    check(
        any("HYPER_PUBLIC_BASE_URL" in a.detail for a in scale_outs),
        "the refusal carries the actionable message",
    )
    check(spawner.instances == {}, "nothing was provisioned — no quota was burned")
    await spawner.aclose()


# --------------------------------------------------------------------------- #
# 3. The API tells the truth, over real HTTP
# --------------------------------------------------------------------------- #
async def prove_status_truth() -> None:
    section("3. Capability truth — the API never claims model-grade simulator output")

    from tests.worker_harness import HeadServer, free_port

    tmp = Path(tempfile.mkdtemp())
    settings = Settings(
        data_dir=str(tmp / "data"),
        log_level="ERROR",
        log_json=False,
        api_key="smoke-key",
        mesh_token="smoke-mesh",
        public_base_url="https://head.example.test",
        spawner_enabled=False,
        sandbox_enabled=False,
        memory_enabled=False,
        autoscaler_enabled=False,
        speculative_enabled=False,
    )
    head = HeadServer(settings)
    head.start()
    try:
        base = f"http://127.0.0.1:{head.port}"
        headers = {"x-api-key": "smoke-key"}
        async with httpx.AsyncClient(base_url=base, headers=headers, timeout=10.0) as client:
            status = (await client.get("/v1/status")).json()
            check("answers_are_real" in status, "/v1/status exposes answers_are_real")
            check(status["answers_are_real"] is False, "with no workers, answers are not claimed to be real")
            capability = status.get("capability", {})
            check(capability.get("label") == "simulator", "the capability label says simulator")
            check("NOT model output" in (capability.get("detail") or ""), "the detail says answers are not model output")
            check("smoke-key" not in json.dumps(status), "no API key leaks into /v1/status")
            check("smoke-mesh" not in json.dumps(status), "no mesh token leaks into /v1/status")

            plan = (await client.get("/v1/capacity/plan")).json()
            check(plan["model_grade"] is False, "capacity/plan refuses a model-grade claim with no model")
            check(plan["claim"] is None, "capacity/plan publishes no claim")
            check(bool(plan.get("why_no_claim")), "capacity/plan explains why there is no claim")
            check("virtual_cores" in plan and "virtual_vram_gb" in plan, "mechanical capacity is still reported")

            # A real (llama.cpp) worker flips the answer, and the plan follows.
            from core.protocol import Capabilities, WorkerHello

            hello = WorkerHello(
                worker_id="wk-gpu-smoke",
                host="127.0.0.1",
                port=free_port(),
                provider="local",
                region="smoke",
                capabilities=Capabilities(backend="llama.cpp", max_concurrency=4, vram_total_gb=24.0, vram_free_gb=20.0),
            )
            ack = await client.post("/v1/workers/register", json=hello.model_dump(mode="json"), headers={"x-mesh-token": "smoke-mesh"})
            check(ack.status_code == 200, "a llama.cpp worker registers over real HTTP", str(ack.status_code))
            check(
                "head_url" in ack.json() and "0.0.0.0" not in ack.json()["head_url"],
                "the registration ack advertises a routable head URL",
            )

            status = (await client.get("/v1/status")).json()
            check(status["answers_are_real"] is True, "a healthy llama.cpp worker flips answers_are_real to true")
            check(status["backends"] == ["llama.cpp"], "the backend list names llama.cpp")
            worker = status["workers"][0]
            check(worker["backend"] == "llama.cpp", "the per-worker backend is on the wire")
            check(worker["max_concurrency"] == 4, "the worker's concurrency slots are on the wire")
            check(status["capability"]["label"] == "live model", "the capability label becomes live model")

            plan = (await client.get("/v1/capacity/plan")).json()
            check(plan["answers_are_real"] is True, "capacity/plan sees the real backend")
            check(plan["virtual_cores"]["concurrent_slots"] == 4, "virtual cores are summed from measured capacity")
            check(plan["virtual_vram_gb"]["total"] == 24.0, "the virtual VRAM pool is reported from measurement")

            # The PWA is served and carries the badge — phone-verifiable.
            page = await client.get("/")
            check(page.status_code == 200, "the PWA is served over the same origin")
            static_dir = PROJECT_ROOT / "mobile" / "static"
            app_js = (static_dir / "app.js").read_text(encoding="utf-8")
            index_html = (static_dir / "index.html").read_text(encoding="utf-8")
            check('id="m-capability"' in index_html, "the PWA has a capability badge element")
            check("applyCapability" in app_js, "the PWA badge is driven by /v1/status")
            check("not model output" in app_js, "the PWA spells out that simulator answers are not model output")
            check("capability-live" in app_js and "capability-simulator" in app_js, "the PWA badge has both states")
    finally:
        head.stop()


# --------------------------------------------------------------------------- #
# 4. The jail is real
# --------------------------------------------------------------------------- #
def prove_the_jail() -> None:
    section("4. The real jail — the kernel holds with the guards removed")

    tiers = available_tiers()
    tier = detect_isolation(JailConfig())
    check(tier in tiers, "the reported tier is one the host actually offers", f"tiers={tiers}")
    check(ISOLATION_PROCESS in tiers, "the process fallback always exists")

    # Guards OFF: only the kernel can stop these.  This is the whole claim.
    sandbox = PythonSandbox(timeout=15, processes=32, guards_off=True)
    check(
        sandbox.isolation == tier,
        "the sandbox reports the same tier as the probe",
        f"{sandbox.isolation} vs {tier}",
    )

    kind = sandbox.run("print('hello from the jail')")
    check(kind.ok and "hello" in kind.stdout, "ordinary code still runs inside the jail", kind.error[:120])
    check(kind.to_dict()["sandbox_isolation"] == sandbox.isolation, "every result carries sandbox_isolation")
    check(kind.kernel_isolated or sandbox.isolation == ISOLATION_PROCESS, "the result says whether the kernel held")

    socket_escape = sandbox.run(
        "import ctypes\n"
        'libc = ctypes.CDLL("libc.so.6", use_errno=True)\n'
        "libc.socket.restype = ctypes.c_int\n"
        "print('fd', libc.socket(2, 1, 0), 'errno', ctypes.get_errno())\n"
    )
    fd = int(socket_escape.stdout.split()[1]) if socket_escape.stdout.split()[1].lstrip("-").isdigit() else 1
    check(fd < 0, "ctypes socket(2) is denied by the kernel (no monkeypatch involved)", socket_escape.stdout.strip()[:80])

    marker = Path("/tmp/hc-smoke-m6-execve")
    marker.unlink(missing_ok=True)
    exec_escape = sandbox.run(
        "import ctypes\n"
        'libc = ctypes.CDLL("libc.so.6", use_errno=True)\n'
        f'argv = (ctypes.c_char_p * 4)(b"/bin/sh", b"-c", b"touch {marker}", None)\n'
        'print("rc", libc.execv(ctypes.c_char_p(b"/bin/sh"), argv))\n'
    )
    check("-1" in exec_escape.stdout, "ctypes execve(2) is denied by the kernel", exec_escape.stdout.strip()[:80])
    check(not marker.exists(), "the execve escape created no file anywhere on the host")

    importlib_escape = sandbox.run(
        "import importlib, os\n"
        "importlib.reload(os)\n"
        "print('rc', os.system('touch /tmp/hc-smoke-m6-importlib'))\n"
    )
    check(
        not Path("/tmp/hc-smoke-m6-importlib").exists(),
        "reloading os via importlib does not defeat the jail",
        importlib_escape.stdout.strip()[:80],
    )

    proc = sandbox.run("import os; print('pids', len(os.listdir('/proc')))")
    pids = int(proc.stdout.split()[1]) if len(proc.stdout.split()) > 1 else 999
    check(pids < 64, "the jail has a fresh /proc, not the host process table", f"{pids} entries")

    secrets = ["/etc/shadow", "/root/.ssh/id_rsa", "/home/user/.bash_history"]
    leaked = []
    for path in secrets:
        result = sandbox.run(f"print(open({path!r}).read()[:32])")
        if result.ok:
            leaked.append(path)
    check(not leaked, "host secrets are unreachable", f"leaked={leaked}")

    # /proc/1 inside the jail is the jail itself, carrying only the sandbox
    # environment.  The property that matters is that the *host* process table
    # is not mounted anywhere the child can reach.
    host_procs = sandbox.run(
        "import os\n"
        "pids = [p for p in os.listdir('/proc') if p.isdigit()]\n"
        "print('host_pids', len(pids))\n"
        "print('pid1_env_host_leak', b'HYPER_API_KEY' in open('/proc/1/environ','rb').read())\n"
    )
    check("host_pids 1" in host_procs.stdout, "only the jail's own PID is visible in /proc", host_procs.stdout.strip()[:80])
    check(
        "pid1_env_host_leak False" in host_procs.stdout,
        "PID 1 inside the jail carries no host secrets (it is the jail, not the host init)",
    )

    symlink = sandbox.run("import os; os.symlink('/etc', 'escape'); print(open('escape/shadow').read()[:8])")
    check(not symlink.ok, "a symlink out of the workspace cannot read /etc/shadow")

    frozen = sandbox.run("open('/etc/hc-smoke-pwn', 'w').write('x')")
    check(not frozen.ok, "the root filesystem cannot be written")
    if frozen.read_only_root:
        check(True, "the root filesystem is reported read-only")
    else:
        check(
            any("read-only" in n for n in frozen.isolation_notes),
            "when the host forbids a read-only root, the result says so instead of pretending",
            "; ".join(frozen.isolation_notes)[:100],
        )

    writable = sandbox.run("open('scratch.txt','w').write('ok'); print(open('scratch.txt').read())")
    check(writable.ok and "ok" in writable.stdout, "the workspace itself is still writable (the tool works)")

    # The fallback must label itself honestly.
    fallback = PythonSandbox(timeout=10, isolation="process")
    forced = fallback.run("print(1)")
    check(
        forced.sandbox_isolation == ISOLATION_PROCESS and not forced.kernel_isolated,
        "a forced process-only sandbox reports 'process' and never claims a container",
    )
    check(fallback.run("import os; os.system('echo pwned')").ok is False, "the process fallback still blocks os.system")


# --------------------------------------------------------------------------- #
async def main() -> int:
    started = time.time()
    print("\033[1m" + "=" * 64)
    print("  Hyper-Cluster Omega — Milestone 6 smoke: truth, hygiene, real jail")
    print("=" * 64 + "\033[0m")

    prove_model_hygiene()
    await prove_head_url_refusal()
    await prove_status_truth()
    prove_the_jail()

    passed = sum(1 for ok, _ in _results if ok)
    total = len(_results)
    failed = [label for ok, label in _results if not ok]
    print(f"\n{'=' * 64}")
    print(f"  {passed}/{total} smoke checks passed in {time.time() - started:.1f}s")
    if failed:
        print("  failures:")
        for label in failed:
            print(f"    - {label}")
    print(f"{'=' * 64}\n")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
