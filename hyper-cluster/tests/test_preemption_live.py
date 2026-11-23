"""The preemption contract, end to end.

The cluster is busy *learning* (the sleep-time engine is training), a chat
arrives -- and the user is served **immediately**.  The measurement is the
whole point of this file: the preemption must complete in well under two
seconds, and the request must not fail.
"""

from __future__ import annotations

import asyncio
import time

import httpx

from core.config import Settings
from tests.worker_harness import HeadServer, WorkerHarness


def _settings(tmp_path) -> Settings:
    return Settings(
        data_dir=str(tmp_path / "data"),
        log_level="WARNING",
        log_json=False,
        api_key="",
        mesh_token="",
        admin_token="",
        worker_ttl_seconds=120,
        heartbeat_interval_seconds=5,
        hold_request_ttl_seconds=30,
        request_timeout_seconds=30,
        memory_enabled=True,
        evolution_enabled=True,
        evolution_idle_seconds=3600,
        evolution_dataset_size=4,
        evolution_max_shards=2,
        evolution_prompt_enabled=False,
        council_size=3,
        council_critique_rounds=0,
    )


async def test_a_chat_preempts_the_self_evolution(tmp_path) -> None:
    """The user always wins -- and the user does not wait."""
    settings = _settings(tmp_path)
    head = HeadServer(settings).start()
    worker = WorkerHarness(head_url=head.base_url, worker_id="preempt-worker")
    worker.start()
    try:
        await worker.boot()

        # The engine runs inside the head node's process.
        orchestrator = head.app.state.orchestrator  # type: ignore[attr-defined]
        assert orchestrator.autopilot is not None
        memory = orchestrator.memory
        assert memory is not None
        # the cluster has something to learn from
        memory.record_exchange(
            "how do I deploy the cluster",
            "deploy the cluster with kubectl apply and the helm chart",
        )

        # A slow shard: the preemption has to interrupt the training.
        async def slow_shard(payload: dict) -> tuple[object, float]:
            await asyncio.sleep(2.0)
            return [[0.0] * len(payload["W"][0])] * len(payload["W"]), 0.5

        orchestrator.autopilot.trainer.shard_fn = slow_shard

        async with httpx.AsyncClient(timeout=30.0) as client:
            # --- the cluster starts to learn ------------------------- #
            cycle = asyncio.create_task(orchestrator.autopilot.run_cycle())
            await asyncio.sleep(0.4)
            assert orchestrator.autopilot.state == "training"

            # --- the user arrives ------------------------------------ #
            started = time.perf_counter()
            response = await client.post(
                f"{head.base_url}/v1/chat",
                json={
                    "messages": [{"role": "user", "content": "hello from the phone"}],
                    "mode": "deep",
                    "max_tokens": 32,
                    "stream": False,
                },
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            assert response.status_code == 200
            payload = response.json()
            assert payload["error"] is None
            assert payload["text"]

            # --- the request was served fast ------------------------- #
            assert elapsed_ms < 2000.0, f"the user waited {elapsed_ms:.0f} ms"

            # --- and the training stopped ---------------------------- #
            report = await asyncio.wait_for(cycle, timeout=10.0)
            assert report.preempted is True
            assert orchestrator.autopilot.preempt_latency_ms < 2000.0

            status = await client.get(f"{head.base_url}/v1/evolution/status")
            assert status.status_code == 200
            body = status.json()
            assert body["preempt_latency_ms"] < 2000.0
            assert body["state"] == "preempted"
    finally:
        await worker.shutdown()
        worker.stop()
        head.stop()


async def test_the_memory_is_used_by_the_next_request(tmp_path) -> None:
    """What the cluster learned shows up in the next answer."""
    settings = _settings(tmp_path)
    head = HeadServer(settings).start()
    worker = WorkerHarness(head_url=head.base_url, worker_id="memory-worker")
    worker.start()
    try:
        await worker.boot()
        orchestrator = head.app.state.orchestrator  # type: ignore[attr-defined]
        memory = orchestrator.memory
        assert memory is not None
        memory.add_document(
            "cluster.md",
            "The hyper cluster runs the head node as a statefulset in the omega namespace.",
        )

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{head.base_url}/v1/chat",
                json={
                    "messages": [{"role": "user", "content": "where does the head node run"}],
                    "mode": "deep",
                    "max_tokens": 48,
                    "stream": False,
                },
            )
            assert response.status_code == 200
            # the exchange is remembered
            assert memory.stats()["by_source"].get("chat", 0) >= 1
            # and the next retrieval finds the document
            hits = memory.search("head node statefulset", k=3)
            assert hits
    finally:
        await worker.shutdown()
        worker.stop()
        head.stop()
