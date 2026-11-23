"""Tests for the Tailscale mesh automation.

The ``tailscale`` CLI is an external dependency, so the tests drive a
faithful stand-in binary that implements the same subcommands and JSON
payloads (``status --json``, ``ip -4``, ``version``, ``up``, ``serve``,
``funnel``, ``set``).  The mesh code under test is the real thing: it parses
the CLI output, builds the arguments and reacts to failures.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from worker.mesh import MeshCoordinator, TailscaleMesh

STATUS_PAYLOAD = {
    "BackendState": "Running",
    "CurrentTailnet": {"Name": "omega.tailnet.ts.net"},
    "Self": {
        "ID": "self-1",
        "HostName": "worker-a",
        "DNSName": "worker-a.omega.tailnet.ts.net.",
        "TailscaleIPs": ["100.101.102.103"],
        "OS": "linux",
        "Online": True,
        "Tags": ["tag:hypercluster-worker"],
    },
    "Peer": {
        "peer-1": {
            "ID": "peer-1",
            "HostName": "worker-b",
            "DNSName": "worker-b.omega.tailnet.ts.net.",
            "TailscaleIPs": ["100.101.102.104"],
            "OS": "linux",
            "Online": True,
            "Tags": ["tag:hypercluster-worker"],
        },
        "peer-2": {
            "ID": "peer-2",
            "HostName": "worker-c",
            "DNSName": "worker-c.omega.tailnet.ts.net.",
            "TailscaleIPs": ["100.101.102.105"],
            "OS": "linux",
            "Online": False,
        },
    },
}


def _write_fake_tailscale(directory: Path, *, state: dict) -> Path:
    """Create an executable stand-in for the tailscale CLI."""
    script = directory / "tailscale"
    log = directory / "calls.log"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        f"LOG = {str(log)!r}\n"
        f"STATE = {str(directory / 'state.json')!r}\n"
        f"PAYLOAD = {json.dumps(STATUS_PAYLOAD)!r}\n"
        "args = sys.argv[1:]\n"
        "with open(LOG, 'a') as handle:\n"
        "    handle.write(' '.join(args) + '\\n')\n"
        "def load():\n"
        "    try:\n"
        "        with open(STATE) as handle:\n"
        "            return json.load(handle)\n"
        "    except Exception:\n"
        "        return {}\n"
        "def save(data):\n"
        "    with open(STATE, 'w') as handle:\n"
        "        json.dump(data, handle)\n"
        "if not args:\n"
        "    sys.exit(1)\n"
        "command = args[0]\n"
        "if command == 'status':\n"
        "    data = load()\n"
        "    if data.get('fail'):\n"
        "        sys.stderr.write('status: backend stopped\\n')\n"
        "        sys.exit(1)\n"
        "    print(json.dumps(data.get('status', {})))\n"
        "elif command == 'ip':\n"
        "    data = load()\n"
        "    print(data.get('ip', ''))\n"
        "elif command == 'version':\n"
        "    print('1.76.0')\n"
        "elif command == 'up':\n"
        "    data = load()\n"
        "    if not data.get('accept_up'):\n"
        "        sys.stderr.write('failed to join tailnet\\n')\n"
        "        sys.exit(1)\n"
        "    data['up_args'] = args[1:]\n"
        "    data['status'] = json.loads(PAYLOAD)\n"
        "    data['status']['BackendState'] = 'Running'\n"
        "    save(data)\n"
        "    print('Success.')\n"
        "elif command in ('serve', 'funnel'):\n"
        "    data = load()\n"
        "    if not data.get('accept_serve'):\n"
        "        sys.stderr.write('failed to configure serve\\n')\n"
        "        sys.exit(1)\n"
        "    data[command + '_args'] = args[1:]\n"
        "    save(data)\n"
        "    print('Available within your tailnet:')\n"
        "elif command == 'set':\n"
        "    data = load()\n"
        "    data['set_args'] = args[1:]\n"
        "    save(data)\n"
        "    print('ok')\n"
        "else:\n"
        "    sys.stderr.write('unknown command\\n')\n"
        "    sys.exit(2)\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    (directory / "state.json").write_text(
        json.dumps({"status": STATUS_PAYLOAD, "ip": "100.101.102.103"}),
        encoding="utf-8",
    )
    return script


@pytest.fixture
def mesh_env(tmp_path: Path) -> tuple[TailscaleMesh, Path, Path]:
    binary = _write_fake_tailscale(tmp_path, state={})
    mesh = TailscaleMesh(binary=str(binary), timeout=10)
    return mesh, tmp_path / "calls.log", tmp_path / "state.json"


def _read_calls(log: Path) -> list[str]:
    if not log.exists():
        return []
    return [line.strip() for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_state(state: Path) -> dict:
    return json.loads(state.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
def test_mesh_reports_availability(mesh_env) -> None:
    mesh, _log, _state = mesh_env
    assert mesh.available() is True


def test_mesh_degrades_when_the_binary_is_missing() -> None:
    mesh = TailscaleMesh(binary="/nonexistent/tailscale")
    assert mesh.available() is False
    status = mesh.status()
    assert status.available is False
    assert "not installed" in status.error
    assert mesh.ip() == ""
    assert mesh.version() == ""
    assert mesh.up(auth_key="k") == (False, "tailscale binary not installed")
    assert mesh.serve(8081)[0] is False
    assert mesh.funnel(8081)[0] is False


def test_status_parses_the_self_node_and_the_peers(mesh_env) -> None:
    mesh, _log, _state = mesh_env
    status = mesh.status()
    assert status.available is True
    assert status.authenticated is True
    assert status.backend_state == "Running"
    assert status.tailnet == "omega.tailnet.ts.net"
    assert status.dns_name == "worker-a.omega.tailnet.ts.net"
    assert status.self_node is not None
    assert status.self_node.address == "100.101.102.103"
    assert status.self_node.is_self is True
    assert len(status.peers) == 2
    offline = [peer for peer in status.peers if not peer.online]
    assert len(offline) == 1
    assert offline[0].host == "worker-c"


def test_status_reports_a_cli_failure(mesh_env) -> None:
    mesh, _log, state = mesh_env
    payload = _read_state(state)
    payload["fail"] = True
    state.write_text(json.dumps(payload), encoding="utf-8")
    status = mesh.status()
    assert status.available is True
    assert status.error


def test_up_joins_with_an_auth_key_and_tags(mesh_env) -> None:
    mesh, _log, state = mesh_env
    payload = _read_state(state)
    payload["accept_up"] = True
    state.write_text(json.dumps(payload), encoding="utf-8")

    ok, message = mesh.up(auth_key="tskey-secret", hostname="worker-a", tags=["hypercluster-worker"])
    assert ok is True, message
    args = _read_state(state)["up_args"]
    assert "--authkey=tskey-secret" in args
    assert "--hostname=worker-a" in args
    assert "--advertise-tags=tag:hypercluster-worker" in args
    assert "--accept-dns=false" in args


def test_up_reports_a_rejected_auth_key(mesh_env) -> None:
    mesh, _log, state = mesh_env
    payload = _read_state(state)
    payload["accept_up"] = False
    state.write_text(json.dumps(payload), encoding="utf-8")
    ok, message = mesh.up(auth_key="bad-key")
    assert ok is False
    assert "failed to join" in message


def test_serve_advertises_the_worker_port(mesh_env) -> None:
    mesh, _log, state = mesh_env
    payload = _read_state(state)
    payload["accept_serve"] = True
    state.write_text(json.dumps(payload), encoding="utf-8")
    ok, message = mesh.serve(8081)
    assert ok is True, message
    args = _read_state(state)["serve_args"]
    assert "--bg" in args
    assert "http://127.0.0.1:8081/" in args


def test_funnel_publishes_the_worker_port(mesh_env) -> None:
    mesh, _log, state = mesh_env
    payload = _read_state(state)
    payload["accept_serve"] = True
    state.write_text(json.dumps(payload), encoding="utf-8")
    ok, _message = mesh.funnel(8081)
    assert ok is True
    args = _read_state(state)["funnel_args"]
    assert "http://127.0.0.1:8081" in args


def test_set_prefs_uses_dashed_flags(mesh_env) -> None:
    mesh, _log, state = mesh_env
    ok, _message = mesh.set_prefs(shields_up=True, operator="omega")
    assert ok is True
    args = _read_state(state)["set_args"]
    assert "--shields-up=true" in args
    assert "--operator=omega" in args


def test_version_and_ip_are_read_from_the_cli(mesh_env) -> None:
    mesh, _log, _state = mesh_env
    assert mesh.version().startswith("1.76.0")
    assert mesh.ip() == "100.101.102.103"


def test_acl_policy_is_tag_scoped(mesh_env) -> None:
    mesh, _log, _state = mesh_env
    policy = json.loads(mesh.generate_acl())
    assert policy["tagOwners"]["tag:hypercluster-worker"] == ["autogroup:admin"]
    accepts = {(entry["src"][0], tuple(entry["dst"])) for entry in policy["acls"]}
    assert ("tag:hypercluster-head", ("*:*",)) in accepts
    assert (
        "tag:hypercluster-worker",
        ("tag:hypercluster-head:*", "tag:hypercluster-worker:*"),
    ) in accepts


def test_acl_policy_is_written_to_disk(mesh_env, tmp_path: Path) -> None:
    mesh, _log, _state = mesh_env
    target = tmp_path / "acl" / "policy.json"
    written = mesh.write_acl(str(target), tag_owners=["user@example.com"])
    assert Path(written).exists()
    policy = json.loads(Path(written).read_text(encoding="utf-8"))
    assert policy["tagOwners"]["tag:hypercluster-head"] == ["user@example.com"]


def test_coordinator_join_waits_for_the_backend(mesh_env) -> None:
    mesh, _log, state = mesh_env
    payload = _read_state(state)
    payload["accept_up"] = True
    state.write_text(json.dumps(payload), encoding="utf-8")
    coordinator = MeshCoordinator(mesh, port=8081)
    assert coordinator.join(auth_key="tskey-secret", hostname="worker-a", timeout=5) is True
    assert _read_state(state)["status"]["BackendState"] == "Running"


def test_coordinator_join_fails_on_a_rejected_key(mesh_env) -> None:
    mesh, _log, state = mesh_env
    payload = _read_state(state)
    payload["accept_up"] = False
    state.write_text(json.dumps(payload), encoding="utf-8")
    coordinator = MeshCoordinator(mesh, port=8081)
    assert coordinator.join(auth_key="bad", timeout=2) is False


def test_coordinator_discovery_returns_nothing_without_a_mesh() -> None:
    coordinator = MeshCoordinator(TailscaleMesh(binary="/nonexistent/tailscale"), port=8081)
    assert coordinator.discover() == []
