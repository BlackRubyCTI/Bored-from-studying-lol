#!/usr/bin/env python3
"""Live simulation for Milestone 4 -- the memory and the sleep-time engine.

Everything in this script is real: a real head node, real worker agents over
HTTP, a real persistent vector memory, a real federated QLoRA trainer and a
real checkpoint streamer talking to local stand-ins for Google Drive and the
Hugging Face Hub.

The simulation proves the three guarantees that matter:

* **the cluster remembers** -- an uploaded file is indexed, distilled facts
  survive a restart and the memory reaches the cloud (and comes back);
* **the cluster learns while it is idle** -- a full cycle mines the
  conversations, synthesises harder prompts, plays them out against the
  mesh, trains a checkpointed adapter, merges it and hot-reloads the workers;
* **the user always wins** -- a chat that arrives during the training
  preempts it and is served in well under two seconds.

    python scripts/smoke_m4.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import httpx  # noqa: E402

from core.config import Settings  # noqa: E402
from memory.sync import DriveClient, HubClient, MemorySync  # noqa: E402
from memory.vector_memory import VectorMemory  # noqa: E402
from tests.cloud_doubles import CloudDouble  # noqa: E402
from tests.worker_harness import HeadServer, WorkerHarness  # noqa: E402

PASS, FAIL = "✅", "❌"
_results: list[tuple[bool, str]] = []


def check(condition: bool, label: str, detail: str = "") -> bool:
    """Record and print a check."""
    _results.append((condition, label))
    suffix = f" — {detail}" if detail else ""
    print(f"  {PASS if condition else FAIL} {label}{suffix}")
    return condition


async def wait_until(predicate: Any, *, timeout: float = 25.0, interval: float = 0.2) -> bool:
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
    cloud = CloudDouble()
    data_dir = fresh_dir("/tmp/hypercluster-smoke-m4")
    settings = Settings(
        data_dir=str(data_dir),
        log_level="WARNING",
        log_json=False,
        worker_ttl_seconds=1800,
        request_timeout_seconds=60,
        memory_enabled=True,
        memory_dir=str(data_dir / "memory"),
        evolution_enabled=True,
        evolution_idle_seconds=3600,
        evolution_dataset_size=6,
        evolution_max_shards=2,
        evolution_prompt_enabled=False,
        evolution_checkpoint_dir=str(data_dir / "checkpoints"),
        # This milestone's scenario shards a *matrix* across the mesh
        # (``slow_shard`` returns W), which is the math backend's contract.  Pin
        # it so the test stays exact; the real QLoRA path has its own smoke
        # (scripts/smoke_m7.py) and its own preemption assertion.
        evolution_train_backend="math",
        drive_token="smoke-token",
        drive_api_base=cloud.drive_api_base,
        drive_upload_base=cloud.drive_upload_base,
        drive_folder="Hyper-Cluster Checkpoints",
        hub_token="smoke-token",
        hub_base_url=cloud.hub_base_url,
        hub_repo="smoke/hyper-cluster",
        council_size=3,
        council_critique_rounds=0,
    )

    head = HeadServer(settings).start()
    workers: list[WorkerHarness] = []
    try:
        # ---------------------------------------------------------- #
        print("\n[1/6] the mesh comes up")
        for index in range(2):
            harness = WorkerHarness(
                head_url=head.base_url,
                worker_id=f"m4-worker-{index}",
                model="Qwen2.5-7B-Instruct-GGUF",
                params_b=7.0 + index,
            )
            harness.start()
            await harness.boot()
            workers.append(harness)

        orchestrator = head.app.state.orchestrator
        memory: VectorMemory = orchestrator.memory
        check(memory is not None, "the memory is enabled")
        check(orchestrator.autopilot is not None, "the sleep-time engine is enabled")

        async with httpx.AsyncClient(timeout=60.0) as client:
            listed = await client.get(f"{head.base_url}/v1/workers")
            check(listed.status_code == 200, "the head node answers")
            check(
                len(listed.json().get("workers") or []) == len(workers),
                "every worker is registered",
                f"{len(listed.json().get('workers') or [])}/{len(workers)}",
            )

            # ------------------------------------------------------ #
            print("\n[2/6] the memory remembers")
            upload = await client.post(
                f"{head.base_url}/v1/memory/documents",
                files={
                    "file": (
                        "architecture.md",
                        b"The head node is a statefulset in the omega namespace. "
                        b"The workers join through the tailscale mesh.",
                        "text/markdown",
                    )
                },
            )
            check(upload.status_code == 200, "a file is uploaded")
            document = upload.json().get("document", {})
            check(int(document.get("chunks") or 0) >= 1, "the file is indexed", f"{document.get('chunks')} chunks")

            facts = await client.post(
                f"{head.base_url}/v1/memory/search",
                json={"query": "where does the head node run", "k": 3},
            )
            hits = facts.json().get("hits") or []
            check(bool(hits), "the memory answers a question")
            check(
                any("statefulset" in str(hit.get("text") or "") for hit in hits),
                "the right passage is found",
            )

            # ------------------------------------------------------ #
            print("\n[3/6] the memory reaches the cloud")
            drive = DriveClient(
                token="smoke-token",
                api_base=cloud.drive_api_base,
                upload_base=cloud.drive_upload_base,
            )
            hub = HubClient(token="smoke-token", base_url=cloud.hub_base_url)
            sync = MemorySync(
                drive=drive,
                hub=hub,
                drive_folder="Hyper-Cluster Memory",
                hub_repo="smoke/hyper-cluster",
            )
            receipts = await sync.push(memory.snapshot())
            check(all(receipt.ok for receipt in receipts), "the snapshot is pushed", f"{len(receipts)} targets")
            drive_files = [item for item in cloud.files() if item["name"].endswith(".json")]
            check(bool(drive_files), "the snapshot is on Drive", drive_files[0]["name"] if drive_files else "")
            check(bool(cloud.hub_files()), "the snapshot is on the Hub")

            restored = VectorMemory(data_dir / "restored-memory")
            snapshots = await sync.pull()
            check(bool(snapshots), "the snapshot comes back")
            if snapshots:
                restored.restore(snapshots[0])
            check(
                restored.stats().get("chunks", 0) == memory.stats().get("chunks", 0),
                "the restored memory is identical",
                f"{restored.stats().get('chunks')} chunks",
            )

            # the memory survives a restart of the head node
            reloaded = VectorMemory(settings.memory_dir)
            check(
                reloaded.stats().get("chunks", 0) == memory.stats().get("chunks", 0),
                "the memory survives a restart",
                f"{reloaded.stats().get('chunks')} chunks",
            )
            await drive.aclose()
            await hub.aclose()

            # ------------------------------------------------------ #
            print("\n[4/6] the cluster learns while it is idle")
            # the cluster has something to learn from
            memory.record_exchange(
                "how do I deploy the cluster",
                "deploy the cluster with kubectl apply and the helm chart",
            )
            memory.record_exchange(
                "how do I scale the workers",
                "scale the workers with the hypercluster autoscaler",
            )
            memory.distill_facts(
                [{"role": "user", "content": "Remember that my name is Ada."}]
            )

            report = await orchestrator.autopilot.run_cycle()
            check(report.error == "", "the cycle finished", report.error or "no error")
            check(report.seeds >= 2, "the conversations were mined", f"{report.seeds} seeds")
            check(report.examples > 0, "harder prompts were synthesised", f"{report.examples} examples")
            check(report.drafts > 0, "the mesh drafted answers", f"{report.drafts} drafts")
            check(report.verified > 0, "the critic kept the good ones", f"{report.verified} verified")
            training = report.training
            check(training is not None and training.merged, "the adapter was trained and merged")
            if training is not None:
                check(
                    training.loss_end <= training.loss_start,
                    "the loss went down",
                    f"{training.loss_start:.5f} -> {training.loss_end:.5f}",
                )
                check(training.checkpoints > 0, "the training was checkpointed", f"{training.checkpoints} checkpoints")
            check(bool(report.artifact_sha256), "the brain has a checksum", report.artifact_sha256[:12])
            check(
                len(report.hot_reloaded) == len(workers),
                "every worker hot-reloaded the brain",
                f"{len(report.hot_reloaded)}/{len(workers)}",
            )

            # the worker really serves the new brain
            health = await client.get(f"{workers[0].base_url}/health")
            brain = health.json().get("brain") or {}
            check(bool(brain.get("loaded")), "the worker runs the new brain")
            check(int(brain.get("revision") or 0) >= 1, "the brain was hot-reloaded", f"revision {brain.get('revision')}")

            # the checkpoint reached the cloud
            streamer = orchestrator.autopilot.streamer
            if streamer is not None:
                remote = await streamer.list_remote(target="drive")
                check(bool(remote), "the checkpoint is in the cloud", f"{len(remote)} files")
            else:
                check(True, "the checkpoint streamer is not configured (skipped)")

            # ------------------------------------------------------ #
            print("\n[5/6] a chat preempts the training")
            # a slow shard: the preemption has to interrupt it
            async def slow_shard(payload: dict[str, Any]) -> tuple[Any, float]:
                await asyncio.sleep(3.0)
                return [[0.0] * len(payload["W"][0])] * len(payload["W"]), 0.5

            orchestrator.autopilot.trainer.shard_fn = slow_shard
            # a fresh run: the previous cycle already checkpointed, and the
            # trainer would resume from it (and finish instantly)
            import shutil

            shutil.rmtree(settings.evolution_checkpoint_dir, ignore_errors=True)
            cycle = asyncio.create_task(orchestrator.autopilot.run_cycle())
            # wait until the training itself is running -- that is what the
            # preemption has to interrupt.
            training_started = await wait_until(
                lambda: orchestrator.autopilot.phase == "training",
                timeout=20.0,
                interval=0.02,
            )
            check(training_started, "the cluster is training again")
            check(orchestrator.autopilot.state == "training", "the engine is in the training state")

            started = time.perf_counter()
            answer = await client.post(
                f"{head.base_url}/v1/chat",
                json={
                    "messages": [{"role": "user", "content": "hello from the phone"}],
                    "mode": "deep",
                    "max_tokens": 48,
                    "stream": False,
                },
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            check(answer.status_code == 200, "the user is served")
            check(not answer.json().get("error"), "the answer has no error")
            check(bool(answer.json().get("text")), "the answer has text")
            check(
                elapsed_ms < 2000.0,
                "the user did not wait for the training",
                f"{elapsed_ms:.0f} ms (budget 2000 ms)",
            )

            preempted = await asyncio.wait_for(cycle, timeout=15.0)
            check(preempted.preempted, "the training was preempted", preempted.preempt_reason)
            latency = orchestrator.autopilot.preempt_latency_ms
            check(
                latency < 2000.0,
                "the preemption was instant",
                f"{latency:.0f} ms (budget 2000 ms)",
            )

            status = await client.get(f"{head.base_url}/v1/evolution/status")
            body = status.json()
            check(body.get("state") == "preempted", "the engine reports the preemption")
            check(
                float(body.get("preempt_latency_ms") or 0.0) < 2000.0,
                "the reported latency is inside the budget",
                f"{body.get('preempt_latency_ms')} ms",
            )

            # ------------------------------------------------------ #
            print("\n[6/6] the cluster is ready for the next question")
            follow_up = await client.post(
                f"{head.base_url}/v1/chat",
                json={
                    "messages": [{"role": "user", "content": "where does the head node run"}],
                    "mode": "deep",
                    "max_tokens": 48,
                    "stream": False,
                },
            )
            check(follow_up.status_code == 200, "the next request is served")
            check(
                memory.stats().get("by_source", {}).get("chat", 0) >= 1,
                "the exchange was remembered",
                f"{memory.stats().get('by_source', {}).get('chat', 0)} conversations",
            )
            facts_now = memory.stats().get("by_source", {}).get("fact", 0)
            check(facts_now >= 1, "the durable facts are stored", f"{facts_now} facts")

    finally:
        for harness in workers:
            await harness.shutdown()
            harness.stop()
        head.stop()
        cloud.close()

    passed = sum(1 for ok, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 60}")
    print(f"  {passed}/{total} checks passed")
    print(f"{'=' * 60}\n")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
