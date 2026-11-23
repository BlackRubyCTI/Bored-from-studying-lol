#!/usr/bin/env python3
"""Live smoke test for Milestone 2.

Boots the real control plane and *real* worker agents (uvicorn), then
exercises the whole Milestone-2 surface:

    self-benchmark -> registration -> capabilities -> streaming chat ->
    tool catalogue -> sandboxed python -> tool-call loop -> failover ->
    mesh discovery -> shutdown.

Exits non-zero on the first failed expectation.

    python scripts/smoke_m2.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import httpx  # noqa: E402

from core.config import Settings  # noqa: E402
from tests.worker_harness import HeadServer, WorkerHarness, patch_capabilities  # noqa: E402
from tools.sandbox import PythonSandbox, ToolExecutor  # noqa: E402

PASS, FAIL = "✅", "❌"
_results: list[tuple[bool, str]] = []


def check(condition: bool, label: str, detail: str = "") -> bool:
    """Record and print a check."""
    _results.append((condition, label))
    suffix = f" — {detail}" if detail else ""
    print(f"  {PASS if condition else FAIL} {label}{suffix}")
    return condition


async def wait_for_workers(client: httpx.AsyncClient, base: str, count: int, timeout: float = 25.0):
    """Wait until ``count`` workers are registered."""
    deadline = asyncio.get_event_loop().time() + timeout
    workers: list[dict] = []
    while asyncio.get_event_loop().time() < deadline:
        response = await client.get(f"{base}/v1/workers")
        if response.status_code == 200:
            workers = response.json()["workers"]
            if len(workers) >= count:
                return workers
        await asyncio.sleep(0.2)
    return workers



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
    settings = Settings(
        data_dir=str(fresh_dir("/tmp/hypercluster-smoke-m2")),
        log_level="WARNING",
        log_json=False,
        worker_ttl_seconds=300,
        request_timeout_seconds=60,
        sandbox_enabled=True,
        max_tool_rounds=4,
    )
    head = HeadServer(settings).start()
    print(f"\n▶ control plane on {head.base_url}")

    workers: list[WorkerHarness] = []
    try:
        # ---------------------------------------------------------------- #
        print("\n▶ self-benchmarking worker")
        first = WorkerHarness(
            head_url=head.base_url,
            model="Qwen2.5-7B-Instruct-GGUF",
            family="qwen2",
            mesh_latency_ms=2.0,
        )
        patch_capabilities(first)
        first.start()
        await first.boot()
        workers.append(first)

        async with httpx.AsyncClient(timeout=60.0) as client:
            health = (await client.get(f"{first.base_url}/health")).json()
            check(health.get("status") == "ok", "worker /health answers", health.get("status", "?"))
            telemetry = (await client.get(f"{first.base_url}/v1/telemetry")).json()
            report = telemetry.get("report") or {}
            check(bool(report.get("cpu")), "worker reports its CPU profile")
            check("flops" in report, "worker ran a FLOPS benchmark")
            check(bool(report.get("gpu") or report.get("cpu")), "hardware telemetry is present")

            registered = await wait_for_workers(client, head.base_url, 1)
            check(len(registered) == 1, "worker registered with the control plane")
            record = registered[0] if registered else {}
            check(record.get("healthy") is True, "the registered worker is healthy")
            check(bool(record.get("model_loaded")), "the worker advertises its model", str(record.get("model_loaded")))
            check(record.get("vram_total_gb", 0) > 0, "the worker advertises its VRAM")

            # ------------------------------------------------------------ #
            print("\n▶ streaming inference")
            events: list[str] = []
            text = ""
            async with client.stream(
                "POST",
                f"{head.base_url}/v1/chat/stream",
                json={
                    "messages": [{"role": "user", "content": "hello from the phone"}],
                    "mode": "deep",
                    "max_tokens": 64,
                },
            ) as response:
                check(response.status_code == 200, "SSE stream accepted", str(response.status_code))
                current = ""
                async for line in response.aiter_lines():
                    if line.startswith("event:"):
                        current = line[6:].strip()
                    elif line.startswith("data:"):
                        events.append(current)
                        if current == "token":
                            text += json.loads(line[5:].strip()).get("text", "")
            check("token" in events, "the stream emitted tokens")
            check(events[-1] == "done", "the stream ended with done", events[-1] if events else "none")
            check(len(text) > 0, "the streamed answer is non-empty")

            # ------------------------------------------------------------ #
            print("\n▶ tool layer")
            catalogue = (await client.get(f"{head.base_url}/v1/tools")).json()
            names = [schema["function"]["name"] for schema in catalogue.get("tools", [])]
            check("python" in names, "the python tool is advertised")
            check("web_search" in names, "the web_search tool is advertised")

            sandbox = PythonSandbox(timeout=10)
            result = sandbox.run("print(6 * 7)")
            check(result.ok and "42" in result.stdout, "the sandbox executes python")
            blocked = sandbox.run("import socket; socket.socket().connect(('1.1.1.1', 80))")
            check(not blocked.ok, "the sandbox blocks raw network access")
            spawn = sandbox.run("import os; os.system('echo pwned')")
            check(not spawn.ok, "the sandbox blocks process spawning")

            executor = ToolExecutor()
            tool_result = executor.execute("python", {"code": "print('tool executor ok')"})
            check(tool_result.ok, "the tool executor dispatches python")

            response = await client.post(
                f"{head.base_url}/v1/chat",
                json={
                    "messages": [
                        {"role": "user", "content": 'compute [[call:python]] {"code": "print(6*7)"}'}
                    ],
                    "tools": [{"type": "function", "function": {"name": "python"}}],
                    "max_tokens": 128,
                    "stream": False,
                },
            )
            check(response.status_code == 200, "the tool loop accepted the request")
            payload = response.json()
            transcript = (payload.get("raw") or {}).get("tools") or []
            check(bool(transcript), "the model called a tool")
            check(transcript and transcript[0].get("tool") == "python", "the python tool was executed")
            check(transcript and transcript[0].get("ok") is True, "the tool call succeeded")
            check(transcript and "42" in transcript[0].get("output", ""), "the tool output reached the model")
            check("42" in payload.get("text", ""), "the final answer uses the tool output")

            # ------------------------------------------------------------ #
            print("\n▶ multi-worker failover")
            second = WorkerHarness(
                head_url=head.base_url,
                model="Llama-3.1-8B-Instruct-GGUF",
                family="llama",
                mesh_latency_ms=40.0,
            )
            patch_capabilities(second)
            second.start()
            await second.boot()
            workers.append(second)

            registered = await wait_for_workers(client, head.base_url, 2)
            check(len(registered) == 2, "both workers are registered")

            requests = [
                client.post(
                    f"{head.base_url}/v1/chat",
                    json={
                        "messages": [{"role": "user", "content": f"question {index}"}],
                        "mode": "deep",
                        "max_tokens": 32,
                        "stream": False,
                    },
                )
                for index in range(3)
            ]
            await asyncio.sleep(0.05)
            await second.crash()
            responses = await asyncio.gather(*requests, return_exceptions=True)
            answered = [r for r in responses if isinstance(r, httpx.Response) and r.status_code == 200]
            check(len(answered) == 3, "no request was dropped during the failover", f"{len(answered)}/3")
            check(all(r.json().get("error") is None for r in answered), "every answered request succeeded")

            deadline = asyncio.get_event_loop().time() + 15
            noticed = False
            while asyncio.get_event_loop().time() < deadline:
                inventory = (await client.get(f"{head.base_url}/v1/workers")).json()["workers"]
                if any(
                    w["worker_id"] == second.agent.worker_id and not w["healthy"] for w in inventory
                ):
                    noticed = True
                    break
                await asyncio.sleep(0.25)
            check(noticed, "the control plane noticed the dead worker")

            # ------------------------------------------------------------ #
            print("\n▶ mesh")
            from worker.mesh import MeshCoordinator, TailscaleMesh  # noqa: PLC0415

            mesh = TailscaleMesh(binary="/nonexistent/tailscale")
            status = mesh.status()
            check(status.available is False, "the mesh degrades gracefully without tailscale")
            check(mesh.generate_acl(), "the mesh can generate a tag-scoped ACL policy")
            check(MeshCoordinator(mesh).discover() == [], "peer discovery is safe without a tailnet")
    finally:
        for harness in workers:
            await harness.shutdown()
            harness.stop()
        head.stop()

    passed = sum(1 for ok, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 60}")
    print(f"  {passed}/{total} checks passed")
    print(f"{'=' * 60}\n")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
