"""Tailscale mesh automation.

The mesh lets ephemeral workers (Kaggle kernels, HF Spaces, GitHub runners)
join the cluster over a private, encrypted tailnet instead of exposing ports
to the public internet.  This module drives the ``tailscale`` CLI:

* join the tailnet with an auth key and tags,
* advertise the worker's own HTTP port (``serve``) or publish it publicly
  (``funnel``),
* discover peers (other workers) and their tailnet addresses,
* generate a tag-based ACL policy so every worker can only reach the head
  node and its peers,
* probe peers for reachability and latency.

Every call degrades gracefully: when the ``tailscale`` binary is missing the
class reports ``available=False`` instead of raising, so a worker can still
join over the LAN.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

from core.ids import now_s
from core.logging_setup import get_logger

LOGGER = get_logger("worker.mesh")


@dataclass
class MeshNode:
    """One node of the tailnet."""

    id: str = ""
    host: str = ""
    dns_name: str = ""
    tailscale_ips: list[str] = field(default_factory=list)
    os: str = ""
    online: bool = False
    tags: list[str] = field(default_factory=list)
    is_self: bool = False

    @property
    def address(self) -> str:
        return self.tailscale_ips[0] if self.tailscale_ips else self.dns_name

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "host": self.host,
            "dns_name": self.dns_name,
            "tailscale_ips": list(self.tailscale_ips),
            "os": self.os,
            "online": self.online,
            "tags": list(self.tags),
            "is_self": self.is_self,
            "address": self.address,
        }


@dataclass
class MeshStatus:
    """Snapshot of the tailnet as seen by this node."""

    available: bool = False
    authenticated: bool = False
    backend_state: str = ""
    tailnet: str = ""
    dns_name: str = ""
    self_node: MeshNode | None = None
    peers: list[MeshNode] = field(default_factory=list)
    error: str = ""
    ts: float = field(default_factory=now_s)

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "authenticated": self.authenticated,
            "backend_state": self.backend_state,
            "tailnet": self.tailnet,
            "dns_name": self.dns_name,
            "self_node": self.self_node.to_dict() if self.self_node else None,
            "peers": [peer.to_dict() for peer in self.peers],
            "error": self.error,
            "ts": self.ts,
        }


class TailscaleMesh:
    """Thin, dependency-free wrapper around the ``tailscale`` CLI."""

    def __init__(self, binary: str = "tailscale", *, timeout: float = 30.0) -> None:
        self.binary = binary
        self.timeout = timeout

    # ------------------------------------------------------------------ #
    def available(self) -> bool:
        """True when the ``tailscale`` binary is installed."""
        return shutil.which(self.binary) is not None

    def _run(self, args: list[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
        """Run a tailscale subcommand."""
        if not self.available():
            raise FileNotFoundError("the tailscale binary is not installed")
        try:
            completed = subprocess.run(
                [self.binary, *args],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            LOGGER.warning("tailscale command failed", extra={"hc_args": " ".join(args), "hc_error": str(exc)})
            raise
        if check and completed.returncode != 0:
            raise RuntimeError(f"tailscale {' '.join(args)} failed: {completed.stderr.strip()[:300]}")
        return completed

    # ------------------------------------------------------------------ #
    def status(self) -> MeshStatus:
        """Parse ``tailscale status --json``."""
        result = MeshStatus(available=self.available())
        if not result.available:
            result.error = "tailscale binary not installed"
            return result
        try:
            completed = self._run(["status", "--json"])
        except Exception as exc:  # noqa: BLE001
            result.error = str(exc)
            return result
        if completed.returncode != 0:
            result.error = completed.stderr.strip()[:300] or "tailscale status failed"
            return result
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            result.error = f"unparsable tailscale status: {exc}"
            return result

        result.backend_state = str(payload.get("BackendState", ""))
        result.authenticated = result.backend_state in ("Running", "NeedsLogin")
        self_node = payload.get("Self") or {}
        result.dns_name = str(self_node.get("DNSName", "")).rstrip(".")
        result.tailnet = str(payload.get("CurrentTailnet", {}).get("Name", "")) if isinstance(
            payload.get("CurrentTailnet"), dict
        ) else ""
        if self_node:
            result.self_node = _node_from_payload(self_node, is_self=True)
        peers: list[MeshNode] = []
        peers_payload = payload.get("Peer")
        if isinstance(peers_payload, dict):
            peers_payload = list(peers_payload.values())
        for peer in peers_payload or []:
            if isinstance(peer, dict):
                peers.append(_node_from_payload(peer))
        result.peers = peers
        return result

    # ------------------------------------------------------------------ #
    def ip(self) -> str:
        """The tailnet IPv4 address of this node."""
        if not self.available():
            return ""
        try:
            completed = self._run(["ip", "-4"])
        except Exception:  # noqa: BLE001
            return ""
        return completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""

    def version(self) -> str:
        """The installed tailscale version."""
        if not self.available():
            return ""
        try:
            completed = self._run(["version"])
        except Exception:  # noqa: BLE001
            return ""
        return completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""

    # ------------------------------------------------------------------ #
    def up(
        self,
        *,
        auth_key: str = "",
        hostname: str = "",
        tags: list[str] | None = None,
        accept_dns: bool = False,
        ssh: bool = False,
        extra_args: list[str] | None = None,
    ) -> tuple[bool, str]:
        """Join (or update) the tailnet."""
        if not self.available():
            return False, "tailscale binary not installed"
        args = ["up", "--accept-dns=" + ("true" if accept_dns else "false")]
        if auth_key:
            args.append(f"--authkey={auth_key}")
        if hostname:
            args.append(f"--hostname={hostname}")
        if tags:
            formatted = ",".join(tag if tag.startswith("tag:") else f"tag:{tag}" for tag in tags)
            args.append(f"--advertise-tags={formatted}")
        if ssh:
            args.append("--ssh")
        args.extend(extra_args or [])
        try:
            completed = self._run(args)
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)
        ok = completed.returncode == 0
        message = (completed.stderr or completed.stdout).strip()[:400]
        LOGGER.info("tailscale up", extra={"hc_ok": ok, "hc_message": message})
        return ok, message

    def set_prefs(self, **prefs: Any) -> tuple[bool, str]:
        """Update tailscale preferences (``tailscale set``)."""
        if not self.available():
            return False, "tailscale binary not installed"
        args = ["set"]
        for key, value in prefs.items():
            flag = key.replace("_", "-")
            if isinstance(value, bool):
                args.append(f"--{flag}={'true' if value else 'false'}")
            else:
                args.append(f"--{flag}={value}")
        try:
            completed = self._run(args)
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)
        return completed.returncode == 0, (completed.stderr or completed.stdout).strip()[:300]

    def serve(self, port: int, *, path: str = "/", background: bool = True) -> tuple[bool, str]:
        """Advertise a local HTTP port inside the tailnet."""
        if not self.available():
            return False, "tailscale binary not installed"
        args = ["serve"]
        if background:
            args.append("--bg")
        args.extend(["--https=443", f"http://127.0.0.1:{port}{path}"])
        try:
            completed = self._run(args)
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)
        ok = completed.returncode == 0
        return ok, (completed.stderr or completed.stdout).strip()[:300]

    def funnel(self, port: int, *, background: bool = True) -> tuple[bool, str]:
        """Publish a local HTTP port to the public internet through the tailnet."""
        if not self.available():
            return False, "tailscale binary not installed"
        args = ["funnel"]
        if background:
            args.append("--bg")
        args.extend(["--https=443", f"http://127.0.0.1:{port}"])
        try:
            completed = self._run(args)
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)
        ok = completed.returncode == 0
        return ok, (completed.stderr or completed.stdout).strip()[:300]

    def logout(self) -> tuple[bool, str]:
        """Leave the tailnet."""
        if not self.available():
            return False, "tailscale binary not installed"
        try:
            completed = self._run(["logout"])
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)
        return completed.returncode == 0, completed.stderr.strip()[:200]

    # ------------------------------------------------------------------ #
    def generate_acl(
        self,
        *,
        head_node_tag: str = "tag:hypercluster-head",
        worker_tag: str = "tag:hypercluster-worker",
        tag_owners: list[str] | None = None,
        extra_hosts: dict[str, list[str]] | None = None,
    ) -> str:
        """Generate a tag-based tailnet ACL policy.

        The worker tag may reach the head node and its peers; the head node
        tag may reach everything.  Tag ownership is required for tagged nodes.
        """
        owners = tag_owners or ["autogroup:admin"]
        extra = extra_hosts or {}
        policy = {
            "tagOwners": {
                head_node_tag: owners,
                worker_tag: owners,
            },
            "acls": [
                {"action": "accept", "src": [head_node_tag], "dst": ["*:*"]},
                {
                    "action": "accept",
                    "src": [worker_tag],
                    "dst": [f"{head_node_tag}:*", f"{worker_tag}:*"],
                },
            ],
            "nodeAttrs": [
                {"target": [f"{worker_tag}:*"], "attr": ["node:agent"]},
            ],
            "autoApprovers": {
                "routes": {},
                "exitNode": [worker_tag],
            },
        }
        if extra:
            policy["hosts"] = extra
        return json.dumps(policy, indent=2)

    def write_acl(self, path: str, **kwargs: Any) -> str:
        """Write the ACL policy to a file (for the admin to paste into Tailscale)."""
        from pathlib import Path

        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        content = self.generate_acl(**kwargs)
        target.write_text(content, encoding="utf-8")
        return str(target)


def _node_from_payload(payload: dict[str, Any], *, is_self: bool = False) -> MeshNode:
    """Convert one ``tailscale status`` peer entry into a :class:`MeshNode`."""
    addresses = payload.get("TailscaleIPs") or []
    if isinstance(addresses, str):
        addresses = [addresses]
    tags: list[str] = []
    tags_payload = payload.get("Tags")
    if isinstance(tags_payload, list):
        tags = [str(tag) for tag in tags_payload]
    return MeshNode(
        id=str(payload.get("ID", "")),
        host=str(payload.get("HostName", "")),
        dns_name=str(payload.get("DNSName", "")).rstrip("."),
        tailscale_ips=[str(ip) for ip in addresses],
        os=str(payload.get("OS", "")),
        online=bool(payload.get("Online", False)),
        tags=tags,
        is_self=is_self,
    )


class MeshCoordinator:
    """Discovers peer workers inside the tailnet."""

    def __init__(self, mesh: TailscaleMesh | None = None, *, port: int = 8081, timeout: float = 3.0) -> None:
        self.mesh = mesh or TailscaleMesh()
        self.port = port
        self.timeout = timeout

    def discover(self) -> list[dict[str, Any]]:
        """Return the reachable worker peers (HTTP ``/health`` answered)."""
        import httpx

        status = self.mesh.status()
        if not status.available or status.self_node is None:
            return []
        found: list[dict[str, Any]] = []
        peers = [status.self_node, *status.peers]
        with httpx.Client(timeout=self.timeout) as client:
            for peer in peers:
                for address in peer.tailscale_ips:
                    url = f"http://{address}:{self.port}/health"
                    try:
                        response = client.get(url)
                    except httpx.HTTPError:
                        continue
                    if response.status_code >= 400:
                        continue
                    try:
                        payload = response.json()
                    except ValueError:
                        continue
                    found.append(
                        {
                            "worker_id": payload.get("worker_id", ""),
                            "url": f"http://{address}:{self.port}",
                            "address": address,
                            "model": payload.get("model_loaded"),
                            "backend": payload.get("backend"),
                            "tailnet_host": peer.host,
                        }
                    )
                    break
        return found

    def join(
        self,
        *,
        auth_key: str,
        hostname: str = "",
        tags: list[str] | None = None,
        ssh: bool = False,
        timeout: float = 60.0,
    ) -> bool:
        """Join the tailnet and wait until the node is running."""
        ok, message = self.mesh.up(auth_key=auth_key, hostname=hostname, tags=tags, ssh=ssh)
        if not ok:
            LOGGER.error("failed to join the tailnet", extra={"hc_message": message})
            return False
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = self.mesh.status()
            if status.backend_state == "Running":
                return True
            time.sleep(1.0)
        return False


__all__ = ["MeshNode", "MeshStatus", "TailscaleMesh", "MeshCoordinator"]
