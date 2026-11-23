#!/usr/bin/env python3
"""Live simulation for Milestone 3 -- the headless provisioner and the
VRAM-aware auto-scaler.

Everything in this script is real: a real head node, real worker agents over
HTTP, real model hot-swaps, a real credential vault and a real provisioner
talking to a local stand-in for the compute providers' REST APIs (Kaggle
kernels, Hugging Face Spaces, GitHub Actions, OCI Compute).

The simulation proves the two guarantees that matter:

* **zero-downtime hot-swaps** -- a worker is replaced *before* its session
  dies, and traffic keeps flowing during the whole hand-over;
* **dynamic model scaling** -- the pool upshifts to a bigger model when the
  demand (and the VRAM) allow it, and downshifts again when they do not.

    python scripts/smoke_m3.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import httpx  # noqa: E402

from core.config import Settings  # noqa: E402
from core.credentials import Credential, CredentialVault  # noqa: E402
from head_node.auto_spawner import (  # noqa: E402
    AutoSpawner,
    GitHubActionsProvisioner,
    HuggingFaceSpaceProvisioner,
    KaggleProvisioner,
    OracleProvisioner,
)
from head_node.events import EventBus  # noqa: E402
from head_node.model_selector import ModelSelector  # noqa: E402
from head_node.orchestrator import Orchestrator  # noqa: E402
from head_node.request_store import RequestLedger  # noqa: E402
from head_node.worker_registry import WorkerRegistry  # noqa: E402
from tests.worker_harness import HeadServer, WorkerHarness, patch_capabilities  # noqa: E402

PASS, FAIL = "✅", "❌"
_results: list[tuple[bool, str]] = []


def check(condition: bool, label: str, detail: str = "") -> bool:
    """Record and print a check."""
    _results.append((condition, label))
    suffix = f" — {detail}" if detail else ""
    print(f"  {PASS if condition else FAIL} {label}{suffix}")
    return condition


# --------------------------------------------------------------------------- #
# The compute providers' API (local stand-in)
# --------------------------------------------------------------------------- #
class _ProviderHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    calls: list[dict[str, Any]] = []

    def log_message(self, *_args: Any) -> None:
        return

    def _send(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        length = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(length) if length else b""
        self.calls.append({"method": "POST", "path": self.path, "body": body.decode("utf-8", "replace")})
        if self.path.endswith("/kernels/push"):
            self._send({"status": "queued"})
            return
        self._send({"ok": True})

    def do_DELETE(self) -> None:  # noqa: N802 - http.server API
        self.calls.append({"method": "DELETE", "path": self.path, "body": ""})
        self._send({"ok": True})


class ProviderAPI:
    """A tiny local implementation of the compute providers' REST endpoints."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        _ProviderHandler.calls = self.calls
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _ProviderHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address[0], self.server.server_address[1]
        self.base_url = f"http://{host}:{port}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


# --------------------------------------------------------------------------- #
async def wait_for_workers(client: httpx.AsyncClient, base: str, count: int, timeout: float = 25.0):
    """Wait until ``count`` workers are registered."""
    deadline = time.time() + timeout
    workers: list[dict] = []
    while time.time() < deadline:
        response = await client.get(f"{base}/v1/workers")
        if response.status_code == 200:
            workers = response.json()["workers"]
            if len(workers) >= count:
                return workers
        await asyncio.sleep(0.2)
    return workers


async def wait_for_model(
    client: httpx.AsyncClient,
    base: str,
    *,
    timeout: float = 25.0,
    different_from: str | None = None,
) -> str:
    """Wait until the worker serves the expected (or a different) model."""
    deadline = time.time() + timeout
    model = ""
    while time.time() < deadline:
        try:
            response = await client.get(f"{base}/health")
            if response.status_code == 200:
                model = str(response.json().get("model_loaded") or "")
                if model and (different_from is None or model != different_from):
                    return model
        except httpx.HTTPError:
            pass
        await asyncio.sleep(0.25)
    return model


async def wait_until(predicate: Any, *, timeout: float = 25.0, interval: float = 0.25) -> bool:
    """Wait for a (synchronous) predicate to become true."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return bool(predicate())



def fresh_dir(path: str) -> Path:
    """A clean data directory -- every run of a smoke starts from scratch.

    Without it the state of the previous run leaks into the next one (the
    memory grows, the ranking drifts) and the smoke stops being repeatable.
    """
    import shutil

    target = Path(path)
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)
    return target



async def main() -> int:
    api = ProviderAPI()
    settings = Settings(
        data_dir=str(fresh_dir("/tmp/hypercluster-smoke-m3")),
        log_level="WARNING",
        log_json=False,
        worker_ttl_seconds=1800,
        request_timeout_seconds=60,
        handoff_lead_seconds=600,
        autoscaler_enabled=True,
        autoscaler_safety_factor=1.15,
        autoscaler_min_gain=0.05,
        autoscaler_min_dwell_seconds=0,
        autoscaler_max_changes_per_tick=1,
        spawner_enabled=True,
        spawner_providers=["kaggle"],
        spawner_min_workers=0,
        spawner_tick_seconds=2,
        credential_file="",
    )
    registry = WorkerRegistry(
        ttl_seconds=settings.worker_ttl_seconds,
        max_concurrency_per_worker=settings.max_concurrency_per_worker,
    )
    vault = CredentialVault()
    vault.add(
        Credential(
            provider="kaggle",
            label="acct-a",
            token="alice:secret",
            metadata={"owner": "alice"},
        )
    )
    selector = ModelSelector(settings, registry)
    spawner = AutoSpawner(
        settings,
        registry,
        vault,
        provisioners={
            "kaggle": KaggleProvisioner(settings, base_url=api.base_url),
            "hf_spaces": HuggingFaceSpaceProvisioner(settings, base_url=f"{api.base_url}/api"),
            "github_actions": GitHubActionsProvisioner(settings, base_url=api.base_url),
            "oracle": OracleProvisioner(settings, base_url=api.base_url),
        },
    )
    engine = Orchestrator(
        settings,
        registry=registry,
        events=EventBus(),
        ledger=RequestLedger(),
        selector=selector,
        spawner=spawner,
    )
    head = HeadServer(settings, orchestrator=engine).start()
    print(f"\n▶ control plane on {head.base_url}")

    workers: list[WorkerHarness] = []
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # ------------------------------------------------------------ #
            print("\n▶ live VRAM pooling")
            big = WorkerHarness(
                head_url=head.base_url,
                model="Qwen2.5-7B-Instruct-GGUF",
                family="qwen2",
                provider="kaggle",
                region="eu-west-1",
                vram_total_gb=24.0,
                vram_free_gb=18.0,
                mesh_latency_ms=3.0,
            )
            patch_capabilities(big)
            big.start()
            await big.boot()
            workers.append(big)

            small = WorkerHarness(
                head_url=head.base_url,
                model="Qwen2.5-0.5B-Instruct-GGUF",
                family="qwen2",
                params_b=0.5,
                provider="kaggle",
                region="eu-west-1",
                vram_total_gb=8.0,
                vram_free_gb=6.0,
                mesh_latency_ms=8.0,
            )
            patch_capabilities(small)
            small.start()
            await small.boot()
            workers.append(small)

            registered = await wait_for_workers(client, head.base_url, 2)
            check(len(registered) == 2, "both workers registered", str(len(registered)))

            pool = (await client.get(f"{head.base_url}/v1/cluster/vram")).json()
            check(pool["pool"]["healthy_workers"] == 2, "the VRAM pool sees both workers")
            check(pool["pool"]["total_gb"] == 32.0, "the pooled VRAM is measured", str(pool["pool"]["total_gb"]))
            check(pool["pool"]["reserved_gb"] > 0, "the resident models are accounted for")
            check(pool["demand"]["total"] == 0, "the demand profile starts empty")

            # ------------------------------------------------------------ #
            print("\n▶ dynamic model scaling (upshift)")
            deep = await client.post(
                f"{head.base_url}/v1/chat",
                json={
                    "messages": [{"role": "user", "content": "compare the architectures and analyse the trade-offs"}],
                    "mode": "deep",
                    "max_tokens": 64,
                    "stream": False,
                },
            )
            check(deep.status_code == 200, "a deep request was served")
            check(deep.json()["error"] is None, "the deep request produced an answer")

            plan = (await client.get(f"{head.base_url}/v1/models/plan")).json()
            check(bool(plan["decisions"]), "the auto-scaler produced a plan")
            check(
                plan["demand"]["level"] == "deep",
                "the demand profile noticed the deep traffic",
                json.dumps(plan["demand"]),
            )

            # The background auto-scaler hot-swaps the biggest node.
            model = await wait_for_model(client, big.base_url, timeout=25.0)
            check(model == "Qwen2.5-32B-Instruct-GGUF", "the pool upshifted to the 32B", str(model))

            after = (await client.get(f"{head.base_url}/v1/cluster/vram")).json()
            check(
                after["pool"]["reserved_gb"] > pool["pool"]["reserved_gb"],
                "the pool now accounts for the bigger model",
                f"{pool['pool']['reserved_gb']} -> {after['pool']['reserved_gb']}",
            )

            hot = await client.post(
                f"{head.base_url}/v1/chat",
                json={
                    "messages": [{"role": "user", "content": "is the cluster still serving during the swap?"}],
                    "mode": "fast",
                    "max_tokens": 32,
                    "stream": False,
                },
            )
            check(hot.status_code == 200 and hot.json()["error"] is None, "the cluster served during the hot-swap")

            # ------------------------------------------------------------ #
            print("\n▶ speculative draft / verifier pairing")
            pair = (await client.get(f"{head.base_url}/v1/speculative/pair")).json()["pair"]
            check(pair is not None, "the pool exposes a draft/verifier pair")
            if pair:
                check(pair["draft_worker"] == small.agent.worker_id, "the small worker is the drafter")
                check(pair["verifier_worker"] == big.agent.worker_id, "the big worker is the verifier")
                check(pair["estimated_speedup"] > 1.0, "the pairing is predicted to be faster", str(pair["estimated_speedup"]))

            # ------------------------------------------------------------ #
            print("\n▶ dynamic model scaling (downshift)")
            for _ in range(4):
                await client.post(
                    f"{head.base_url}/v1/chat",
                    json={
                        "messages": [{"role": "user", "content": "hi"}],
                        "mode": "fast",
                        "max_tokens": 16,
                        "stream": False,
                    },
                )
            plan = (await client.get(f"{head.base_url}/v1/models/plan")).json()
            check(plan["demand"]["level"] == "fast", "the demand profile followed the fast traffic")
            model = await wait_for_model(client, big.base_url, timeout=25.0, different_from="Qwen2.5-32B-Instruct-GGUF")
            check(model != "Qwen2.5-32B-Instruct-GGUF", "the pool downshifted again", str(model))
            released = (await client.get(f"{head.base_url}/v1/cluster/vram")).json()
            check(
                released["pool"]["reserved_gb"] < after["pool"]["reserved_gb"],
                "the downshift released VRAM",
                f"{after['pool']['reserved_gb']} -> {released['pool']['reserved_gb']}",
            )

            # ------------------------------------------------------------ #
            print("\n▶ pre-emptive TTL hand-over (zero downtime)")
            expiring = WorkerHarness(
                head_url=head.base_url,
                model="Qwen2.5-7B-Instruct-GGUF",
                family="qwen2",
                provider="kaggle",
                region="eu-west-1",
                vram_total_gb=16.0,
                vram_free_gb=12.0,
                mesh_latency_ms=5.0,
                session_ends_in=300.0,  # the session dies in 5 minutes
            )
            patch_capabilities(expiring)
            expiring.start()
            await expiring.boot()
            workers.append(expiring)

            served: list[int] = []
            stop = asyncio.Event()

            async def _traffic() -> None:
                while not stop.is_set():
                    response = await client.post(
                        f"{head.base_url}/v1/chat",
                        json={
                            "messages": [{"role": "user", "content": "ping during the hand-over"}],
                            "mode": "fast",
                            "max_tokens": 24,
                            "stream": False,
                        },
                    )
                    served.append(response.status_code)
                    await asyncio.sleep(0.3)

            traffic = asyncio.create_task(_traffic())
            await asyncio.sleep(1.5)
            check(bool(served), "traffic is flowing before the hand-over", f"{len(served)} requests")

            # The rotator replaces the worker whose session ends in 5 minutes
            # -- long before the session actually dies.
            rotated = await wait_until(
                lambda: any(
                    instance.get("replaces") == expiring.agent.worker_id
                    for instance in spawner.status()["instances"]
                ),
                timeout=30.0,
            )
            check(rotated, "the rotator provisioned a replacement for the expiring worker")
            check(
                any("/kernels/push" in call["path"] for call in api.calls),
                "the provisioner hit the provider API",
                f"{len(api.calls)} calls",
            )

            old = registry.get(expiring.agent.worker_id)
            check(old is not None and old.healthy is True, "the old worker still serves during the boot")

            # The replacement node joins (the provisioned worker phones home).
            replacement = WorkerHarness(
                head_url=head.base_url,
                model="Qwen2.5-7B-Instruct-GGUF",
                family="qwen2",
                provider="kaggle",
                region="eu-west-1",
                vram_total_gb=16.0,
                vram_free_gb=12.0,
                mesh_latency_ms=6.0,
            )
            patch_capabilities(replacement)
            replacement.start()
            await replacement.boot()
            workers.append(replacement)

            handed = await wait_until(
                lambda: bool(
                    registry.get(expiring.agent.worker_id)
                    and registry.get(expiring.agent.worker_id).draining
                ),
                timeout=30.0,
            )
            check(handed, "the hand-over completed")
            check(
                bool(registry.get(expiring.agent.worker_id).draining),
                "the old worker is draining",
            )
            check(
                bool(
                    registry.get(replacement.agent.worker_id)
                    and registry.get(replacement.agent.worker_id).healthy
                ),
                "the replacement is serving",
            )

            await asyncio.sleep(1.0)
            stop.set()
            await traffic
            failures = [code for code in served if code != 200]
            check(
                not failures,
                "no request failed during the hand-over",
                f"{len(served)} requests, {len(failures)} failures",
            )

    finally:
        for harness in workers:
            await harness.shutdown()
            harness.stop()
        head.stop()
        api.close()

    passed = sum(1 for ok, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 60}")
    print(f"  {passed}/{total} checks passed")
    print(f"{'=' * 60}\n")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
