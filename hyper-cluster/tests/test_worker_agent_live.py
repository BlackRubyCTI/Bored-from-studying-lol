"""Live tests for the mesh worker agent and the multi-worker failover path.

These run a *real* head node and *real* worker agents over HTTP + SSE.
"""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from core.config import Settings
from tests.worker_harness import HeadServer, WorkerHarness, patch_capabilities


async def _wait_for_workers(client: httpx.AsyncClient, base: str, count: int, timeout: float = 20.0) -> list[dict]:
    """Wait until ``count`` workers are registered with the head node."""
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


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        data_dir=str(tmp_path / "data"),
        log_level="WARNING",
        log_json=False,
        worker_ttl_seconds=120,
        request_timeout_seconds=30,
        council_size=3,
        council_critique_rounds=0,
    )


async def test_worker_registers_with_measured_capabilities(settings: Settings) -> None:
    head = HeadServer(settings).start()
    worker = WorkerHarness(head_url=head.base_url, model="Qwen2.5-7B-Instruct-GGUF")
    patch_capabilities(worker)
    worker.start()
    try:
        await worker.boot()
        async with httpx.AsyncClient(timeout=15.0) as client:
            workers = await _wait_for_workers(client, head.base_url, 1)
            assert len(workers) == 1, "the worker did not register"
            record = workers[0]
            assert record["provider"] == "local"
            assert record["model_loaded"] == "Qwen2.5-7B-Instruct-GGUF"
            assert record["vram_total_gb"] == 24.0
            assert record["healthy"] is True
            assert record["seconds_to_expiry"] > 0
    finally:
        await worker.shutdown()
        worker.stop()
        head.stop()


async def test_worker_serves_streaming_chat(settings: Settings) -> None:
    head = HeadServer(settings).start()
    worker = WorkerHarness(head_url=head.base_url, model="Qwen2.5-7B-Instruct-GGUF")
    patch_capabilities(worker)
    worker.start()
    try:
        await worker.boot()
        async with httpx.AsyncClient(timeout=30.0) as client:
            await _wait_for_workers(client, head.base_url, 1)
            events: list[str] = []
            text = ""
            async with client.stream(
                "POST",
                f"{head.base_url}/v1/chat/stream",
                json={
                    "messages": [{"role": "user", "content": "hello from the phone"}],
                    "mode": "deep",
                    "max_tokens": 48,
                },
            ) as response:
                assert response.status_code == 200
                current = ""
                async for line in response.aiter_lines():
                    if line.startswith("event:"):
                        current = line[6:].strip()
                    elif line.startswith("data:"):
                        events.append(current)
                        if current == "token":
                            import json

                            text += json.loads(line[5:].strip()).get("text", "")
            assert "token" in events
            assert events[-1] == "done"
            assert "hello from the phone" in text
    finally:
        await worker.shutdown()
        worker.stop()
        head.stop()


async def test_worker_reports_health_and_telemetry(settings: Settings) -> None:
    head = HeadServer(settings).start()
    worker = WorkerHarness(head_url=head.base_url)
    patch_capabilities(worker)
    worker.start()
    try:
        await worker.boot()
        async with httpx.AsyncClient(timeout=15.0) as client:
            health = (await client.get(f"{worker.base_url}/health")).json()
            assert health["status"] == "ok"
            assert health["worker_id"]
            assert health["backend"] == "simulator"

            telemetry = (await client.get(f"{worker.base_url}/v1/telemetry")).json()
            assert telemetry["report"] is not None
            assert telemetry["report"]["cpu"]["cores"] >= 1
            assert "flops" in telemetry["report"]
    finally:
        await worker.shutdown()
        worker.stop()
        head.stop()


async def test_drain_marks_the_worker_unavailable(settings: Settings) -> None:
    head = HeadServer(settings).start()
    worker = WorkerHarness(head_url=head.base_url)
    patch_capabilities(worker)
    worker.start()
    try:
        await worker.boot()
        async with httpx.AsyncClient(timeout=15.0) as client:
            await _wait_for_workers(client, head.base_url, 1)
            drained = await client.post(f"{head.base_url}/v1/workers/{worker.agent.worker_id}/drain")
            assert drained.status_code == 200
            workers = (await client.get(f"{head.base_url}/v1/workers")).json()["workers"]
            assert workers[0]["draining"] is True
            assert workers[0]["healthy"] is False
    finally:
        await worker.shutdown()
        worker.stop()
        head.stop()


async def test_multi_worker_failover_never_drops_a_request(settings: Settings) -> None:
    """Kill the fastest worker: every request must still be answered."""
    head = HeadServer(settings).start()
    workers = []
    for index in range(3):
        harness = WorkerHarness(
            head_url=head.base_url,
            model=["Qwen2.5-7B-Instruct-GGUF", "Llama-3.1-8B-Instruct-GGUF", "gemma-2-9b-it-GGUF"][index],
            family=["qwen2", "llama", "gemma2"][index],
            mesh_latency_ms=[2.0, 60.0, 120.0][index],
        )
        patch_capabilities(harness)
        harness.start()
        await harness.boot()
        workers.append(harness)

    try:
        async with httpx.AsyncClient(timeout=40.0) as client:
            await _wait_for_workers(client, head.base_url, 3)
            assert len((await client.get(f"{head.base_url}/v1/workers")).json()["workers"]) == 3

            # Fire a batch of requests, then kill the fastest worker mid-flight.
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
                for index in range(4)
            ]
            fastest = workers[0]
            await asyncio.sleep(0.05)
            await fastest.crash()  # hard kill: the socket dies mid-life
            responses = await asyncio.gather(*requests, return_exceptions=True)

            answered = [r for r in responses if isinstance(r, httpx.Response) and r.status_code == 200]
            assert len(answered) == 4, f"expected 4 answered requests, got {len(answered)}"
            for response in answered:
                payload = response.json()
                assert payload["error"] is None
                assert payload["text"]

            # The head node must have noticed the dead worker.
            deadline = time.time() + 15
            lost = False
            while time.time() < deadline:
                workers_now = (await client.get(f"{head.base_url}/v1/workers")).json()["workers"]
                if any(w["worker_id"] == fastest.agent.worker_id and not w["healthy"] for w in workers_now):
                    lost = True
                    break
                await asyncio.sleep(0.25)
            assert lost, "the head node never noticed the dead worker"
    finally:
        for harness in workers:
            await harness.shutdown()
            harness.stop()
        head.stop()


async def test_worker_reconnects_when_the_head_node_returns(settings: Settings) -> None:
    """A worker started before the head node must register once it appears."""
    head = HeadServer(settings)
    worker = WorkerHarness(head_url=head.base_url)
    patch_capabilities(worker)
    worker.start()
    try:
        await worker.boot()  # the head node is down -> registration retries
        await asyncio.sleep(0.5)
        head.start()
        async with httpx.AsyncClient(timeout=20.0) as client:
            workers = await _wait_for_workers(client, head.base_url, 1, timeout=25.0)
            assert len(workers) == 1, "the worker never re-registered"
    finally:
        await worker.shutdown()
        worker.stop()
        head.stop()
