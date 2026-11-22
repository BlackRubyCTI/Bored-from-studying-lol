"""The worker's brain: hot-reload and the training shard (live HTTP)."""

from __future__ import annotations

import numpy as np
import pytest

from core.config import Settings
from tests.worker_harness import HeadServer, WorkerHarness
from training.qlora import BrainArtifact, _pack


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        data_dir=str(tmp_path / "data"),
        log_level="WARNING",
        log_json=False,
        api_key="",
        mesh_token="",
        admin_token="",
        worker_ttl_seconds=60,
        heartbeat_interval_seconds=5,
        hold_request_ttl_seconds=30,
        request_timeout_seconds=20,
        memory_enabled=True,
        evolution_enabled=True,
        evolution_idle_seconds=3600,
    )


def _artifact() -> BrainArtifact:
    vocab = ["kubectl", "apply", "helm", "chart"]
    artifact = BrainArtifact(dim=16, rank=2, vocab=vocab, steps=4, loss=0.25)
    payload = np.arange(16 * len(vocab), dtype=np.float32).reshape(16, len(vocab))
    artifact.base_quantized = _pack(payload.astype(np.int8))
    artifact.base_scale = _pack(np.ones((16,), dtype=np.float32))
    artifact.A = _pack(np.eye(16, 2, dtype=np.float32))
    artifact.B = _pack(np.zeros((2, len(vocab)), dtype=np.float32))
    artifact.merged = _pack(payload)
    return artifact


async def test_the_worker_accepts_a_hot_reloaded_brain(settings: Settings) -> None:
    head = HeadServer(settings).start()
    worker = WorkerHarness(head_url=head.base_url, worker_id="brain-worker")
    worker.start()
    try:
        await worker.boot()
        import httpx

        artifact = _artifact()
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                f"{worker.base_url}/v1/brain/load",
                json={"worker_id": worker.agent.worker_id, "artifact": artifact.encode()},
            )
            assert response.status_code == 200
            body = response.json()
            assert body["ok"] is True
            assert body["brain"]["revision"] >= 1
            assert body["brain"]["vocab_size"] == 4
            assert body["brain"]["sha256"]

            # the worker reports the brain in its health
            health = await client.get(f"{worker.base_url}/health")
            assert health.status_code == 200
            assert health.json()["brain"]["revision"] >= 1
    finally:
        await worker.shutdown()
        worker.stop()
        head.stop()


async def test_the_worker_rejects_a_broken_artifact(settings: Settings) -> None:
    head = HeadServer(settings).start()
    worker = WorkerHarness(head_url=head.base_url, worker_id="brain-worker")
    worker.start()
    try:
        await worker.boot()
        import httpx

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                f"{worker.base_url}/v1/brain/load",
                json={"worker_id": worker.agent.worker_id, "artifact": "not-base64-at-all"},
            )
            assert response.status_code >= 400
    finally:
        await worker.shutdown()
        worker.stop()
        head.stop()


async def test_the_worker_computes_a_training_shard(settings: Settings) -> None:
    head = HeadServer(settings).start()
    worker = WorkerHarness(head_url=head.base_url, worker_id="brain-worker")
    worker.start()
    try:
        await worker.boot()
        import httpx

        X = [[1.0, 0.0, 0.5], [0.0, 1.0, 0.25]]
        Y = [[0.0, 1.0, 0.5], [1.0, 0.0, 0.25]]
        W = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{worker.base_url}/v1/train/shard",
                json={"X": X, "Y": Y, "W": W, "l2": 0.0},
            )
            assert response.status_code == 200
            body = response.json()
            assert body["loss"] >= 0.0
            gradient = np.asarray(body["gradient"], dtype=np.float64)
            assert gradient.shape == (3, 3)
            assert np.any(gradient != 0.0)
    finally:
        await worker.shutdown()
        worker.stop()
        head.stop()


async def test_the_brain_primes_the_prompt(settings: Settings) -> None:
    head = HeadServer(settings).start()
    worker = WorkerHarness(head_url=head.base_url, worker_id="brain-worker")
    worker.start()
    try:
        await worker.boot()
        import httpx

        artifact = _artifact()
        async with httpx.AsyncClient(timeout=15.0) as client:
            await client.post(
                f"{worker.base_url}/v1/brain/load",
                json={"worker_id": worker.agent.worker_id, "artifact": artifact.encode()},
            )
            response = await client.post(
                f"{worker.base_url}/v1/chat/sync",
                json={
                    "messages": [{"role": "user", "content": "how do I deploy with kubectl"}],
                    "max_tokens": 32,
                    "stream": False,
                },
            )
            assert response.status_code == 200
            body = response.json()
            assert body["text"]
    finally:
        await worker.shutdown()
        worker.stop()
        head.stop()
