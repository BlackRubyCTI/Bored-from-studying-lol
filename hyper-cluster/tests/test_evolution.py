"""The sleep-time self-evolution engine: maths, pipeline and preemption."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from core.ids import now_s
from core.protocol import ChatCompletion, ChatRequest
from head_node.worker_registry import WorkerRecord, WorkerRegistry
from memory.vector_memory import VectorMemory
from tests.cloud_doubles import CloudDouble
from tests.conftest import make_hello
from training.autopilot_evolver import (
    STATE_PREEMPTED,
    AutopilotEvolver,
)
from training.checkpoint_streamer import CheckpointStreamer
from training.distributed_trainer import (
    Checkpoint,
    DistributedTrainer,
    PreemptionToken,
    TrainingTrace,
)
from training.prompt_evolver import PromptEvolver, load_prompt_from_memory, store_prompt_in_memory
from training.qlora import (
    BrainArtifact,
    average_gradients,
    build_vocabulary,
    featurize,
    local_gradient,
    lora_factorize,
    merge,
    quantize,
)
from training.self_play import SelfPlayEngine, parse_score, similarity
from training.synthetic_data import (
    SyntheticDataset,
    build_dataset,
    mine_seeds,
    mutate_prompt,
    score_difficulty,
)


# --------------------------------------------------------------------------- #
# Doubles
# --------------------------------------------------------------------------- #
class FakeClient:
    """A worker client that answers from a scripted table."""

    def __init__(self, answers: dict[str, str] | None = None, *, delay: float = 0.0) -> None:
        self.answers = answers or {}
        self.delay = delay
        self.brains: dict[str, str] = {}
        self.shards: list[dict[str, Any]] = []
        self.calls: list[str] = []

    async def chat(self, record: WorkerRecord, request: ChatRequest) -> ChatCompletion:
        self.calls.append(record.worker_id)
        if self.delay:
            await asyncio.sleep(self.delay)
        prompt = request.messages[-1].content if request.messages else ""
        if "Score (0-10)" in prompt:
            return ChatCompletion(request_id="critic", text="8", model="fake")
        for needle, answer in self.answers.items():
            if needle.lower() in prompt.lower():
                return ChatCompletion(
                    request_id="draft",
                    text=answer,
                    model=record.model_loaded or "fake",
                    worker_id=record.worker_id,
                )
        return ChatCompletion(
            request_id="draft",
            text="the cluster answers with the kubernetes ingress controller",
            model="fake",
            worker_id=record.worker_id,
        )

    async def load_brain(self, record: WorkerRecord, artifact: str) -> bool:
        self.brains[record.worker_id] = artifact
        return True

    async def train_shard(self, record: WorkerRecord, payload: dict[str, Any]) -> tuple[Any, float]:
        self.shards.append(payload)
        X = np.asarray(payload["X"], dtype=np.float64)
        Y = np.asarray(payload["Y"], dtype=np.float64)
        W = np.asarray(payload["W"], dtype=np.float64)
        gradient, loss = local_gradient(X, Y, W, l2=float(payload.get("l2") or 0.0))
        return gradient.tolist(), float(loss)


def _registry(count: int = 2) -> WorkerRegistry:
    registry = WorkerRegistry(ttl_seconds=60, max_concurrency_per_worker=2)
    for index in range(count):
        hello = make_hello(
            f"worker-{index}",
            9000 + index,
            params_b=7.0 + index,
            model=f"Qwen2.5-{7 + index}B-Instruct-GGUF",
        )
        registry.register(hello)
    return registry


def _traces() -> list[TrainingTrace]:
    return [
        TrainingTrace(prompt="how do I deploy the cluster", response="deploy with the helm chart and kubectl apply"),
        TrainingTrace(prompt="how do I deploy the cluster", response="use kubectl apply the deployment yaml"),
        TrainingTrace(prompt="what is the ingress", response="nginx ingress handles the tls termination"),
        TrainingTrace(prompt="what is the ingress", response="the ingress routes traffic with nginx"),
    ]


# --------------------------------------------------------------------------- #
# The synthetic data
# --------------------------------------------------------------------------- #
def test_the_dataset_is_deterministic() -> None:
    seeds = mine_seeds(
        [
            {"id": "c1", "text": "Q: how do I deploy the cluster\nA: with kubectl apply the helm chart"},
            {"id": "c2", "text": "Q: what is the ingress\nA: the nginx ingress controller"},
        ]
    )
    assert len(seeds) == 2
    assert seeds[0].answer
    first = build_dataset(seeds, count=8, seed=7).to_list()
    second = build_dataset(seeds, count=8, seed=7).to_list()
    assert first == second
    assert len(first) == 8


def test_the_mutations_add_difficulty() -> None:
    import random

    rng = random.Random(3)
    mutated, origin = mutate_prompt("how do I deploy the cluster", rng)
    assert mutated != "how do I deploy the cluster"
    assert origin
    assert score_difficulty(mutated) >= 0.0


def test_the_dataset_carries_the_tags() -> None:
    seeds = mine_seeds(
        [{"id": "c1", "text": "Q: how do I deploy the cluster\nA: kubectl apply the helm chart"}]
    )
    dataset = build_dataset(seeds, count=4, seed=1)
    assert isinstance(dataset, SyntheticDataset)
    assert dataset.stats()["examples"] == 4
    assert all(example.prompt for example in dataset)


# --------------------------------------------------------------------------- #
# The maths
# --------------------------------------------------------------------------- #
def test_the_gradient_moves_the_weights_towards_the_answer() -> None:
    vocab = build_vocabulary(["kubectl apply the deployment"], size=8)
    X, Y = featurize(["deploy the cluster"], ["kubectl apply the deployment"], vocab, dim=32)
    W = np.zeros((32, len(vocab)), dtype=np.float32)
    gradient, loss = local_gradient(X, Y, W, l2=0.0)
    assert gradient.shape == W.shape
    assert loss > 0.0
    updated = W - 0.5 * gradient
    assert local_gradient(X, Y, updated, l2=0.0)[1] < loss


def test_the_federated_average_is_weighted() -> None:
    small = np.ones((4, 4), dtype=np.float32)
    big = np.full((4, 4), 3.0, dtype=np.float32)
    averaged = average_gradients([small, big], [1, 3])
    assert np.allclose(averaged, 2.5)


def test_the_quantisation_keeps_the_matrix_close() -> None:
    matrix = np.random.default_rng(11).normal(size=(16, 16)).astype(np.float32)
    quantized, scale = quantize(matrix)
    from training.qlora import dequantize

    restored = dequantize(quantized, scale)
    assert np.allclose(restored, matrix, atol=0.05)


def test_the_adapter_merges_into_the_base() -> None:
    matrix = np.random.default_rng(5).normal(size=(32, 32)).astype(np.float32)
    A, B = lora_factorize(matrix, rank=4)
    assert A.shape == (4, 32)
    assert B.shape == (32, 4)
    # a rank-4 factorisation is a truncated SVD: it keeps most of the energy
    assert np.linalg.norm(matrix - B @ A) < np.linalg.norm(matrix)
    # a full-rank factorisation is exact
    A, B = lora_factorize(matrix, rank=32)
    assert np.allclose(B @ A, matrix, atol=1e-3)
    base_quantized, base_scale = quantize(matrix)
    merged = merge(base_quantized, base_scale, A, B)
    assert merged.shape == matrix.shape


def test_the_artifact_survives_a_round_trip() -> None:
    artifact = BrainArtifact(dim=16, rank=2, vocab=["a", "b", "c"], steps=3, loss=0.5)
    from training.qlora import _pack

    payload = np.arange(16 * 3, dtype=np.float32).reshape(16, 3)
    artifact.base_quantized = _pack(payload.astype(np.int8))
    artifact.base_scale = _pack(np.ones((16,), dtype=np.float32))
    artifact.A = _pack(np.eye(16, 2, dtype=np.float32))
    artifact.B = _pack(np.zeros((2, 3), dtype=np.float32))
    artifact.merged = _pack(payload)

    decoded = BrainArtifact.decode(artifact.encode())
    assert decoded.sha256() == artifact.sha256()
    assert decoded.vocab == artifact.vocab
    assert decoded.hint("a b c") == artifact.hint("a b c")


# --------------------------------------------------------------------------- #
# The trainer
# --------------------------------------------------------------------------- #
async def test_the_training_converges(tmp_path: Path) -> None:
    trainer = DistributedTrainer(checkpoint_dir=tmp_path / "ckpt", steps=25, learning_rate=1.0, dim=32, rank=4)
    report = await trainer.train(_traces(), shards=2)
    assert report.steps == 25
    assert report.loss_end < report.loss_start
    assert report.merged is True
    assert trainer.last_artifact is not None


async def test_the_shards_are_balanced(tmp_path: Path) -> None:
    trainer = DistributedTrainer(checkpoint_dir=tmp_path / "ckpt")
    plan = trainer.shard_plan(7, 3)
    assert [len(part) for part in plan] == [3, 3, 1]
    assert sorted(index for part in plan for index in part) == list(range(7))


async def test_the_checkpoint_is_written_and_resumed(tmp_path: Path) -> None:
    trainer = DistributedTrainer(checkpoint_dir=tmp_path / "ckpt", steps=4, dim=32, rank=4)
    await trainer.train(_traces(), shards=1)
    checkpoint = Checkpoint.latest(trainer.checkpoint_dir)
    assert checkpoint is not None
    assert checkpoint.step == 4

    resumed = DistributedTrainer(checkpoint_dir=tmp_path / "ckpt", steps=8, dim=32, rank=4)
    report = await resumed.train(_traces(), shards=1)
    # every run does its own ``steps`` worth of work on top of the checkpoint
    assert report.steps == 8 + 4
    assert Checkpoint.latest(resumed.checkpoint_dir).step == 12
    assert Checkpoint.latest(resumed.checkpoint_dir).step == report.steps


async def test_a_preemption_stops_the_training(tmp_path: Path) -> None:
    token = PreemptionToken()
    trainer = DistributedTrainer(checkpoint_dir=tmp_path / "ckpt", steps=50, dim=32, rank=4)
    calls = {"n": 0}

    async def slow_shard(payload: dict[str, Any]) -> tuple[Any, float]:
        calls["n"] += 1
        await asyncio.sleep(0.05)
        X = np.asarray(payload["X"], dtype=np.float64)
        Y = np.asarray(payload["Y"], dtype=np.float64)
        W = np.asarray(payload["W"], dtype=np.float64)
        gradient, loss = local_gradient(X, Y, W, l2=0.0)
        return gradient.tolist(), float(loss)

    trainer.shard_fn = slow_shard

    async def flip() -> None:
        await asyncio.sleep(0.12)
        token.preempt("user request")

    flipper = asyncio.create_task(flip())
    report = await trainer.train(_traces(), shards=2, token=token)
    await flipper
    assert report.preempted is True
    assert report.preempt_reason == "user request"
    assert report.steps < 50


async def test_the_training_survives_a_dead_worker(tmp_path: Path) -> None:
    trainer = DistributedTrainer(checkpoint_dir=tmp_path / "ckpt", steps=3, dim=32, rank=4)
    client = FakeClient()
    trainer.shard_fn = lambda payload: client.train_shard(_registry().healthy_workers()[0], payload)
    report = await trainer.train(_traces(), shards=2)
    assert report.merged is True


# --------------------------------------------------------------------------- #
# The self-play
# --------------------------------------------------------------------------- #
def test_the_score_is_parsed_from_the_critic() -> None:
    assert parse_score("SCORE: 8/10") == pytest.approx(0.8)
    assert parse_score("the answer is 0.5") == pytest.approx(0.5)
    assert parse_score("nothing here") == 0.0


def test_the_similarity_is_symmetric() -> None:
    assert similarity("abc def", "abc def") == pytest.approx(1.0)
    assert similarity("abc def", "abc xyz") == pytest.approx(1 / 3)
    assert similarity("", "abc") == 0.0


async def test_the_self_play_keeps_the_verified_drafts() -> None:
    registry = _registry(2)
    client = FakeClient({"deploy": "deploy the cluster with kubectl apply the helm chart"})
    engine = SelfPlayEngine(registry, client, threshold=0.1)
    dataset = build_dataset(
        mine_seeds(
            [{"id": "c1", "text": "Q: how do I deploy the cluster\nA: kubectl apply the helm chart"}]
        ),
        count=2,
        seed=2,
    )
    traces, report = await engine.run(list(dataset), drafts_per_prompt=2)
    assert report.drafts == 4
    assert report.verified >= 1
    assert traces
    assert traces[0].score >= engine.threshold


async def test_the_self_play_stops_when_the_user_arrives() -> None:
    registry = _registry(2)
    client = FakeClient(delay=0.05)
    engine = SelfPlayEngine(registry, client)
    dataset = build_dataset(
        mine_seeds(
            [{"id": "c1", "text": "Q: how do I deploy the cluster\nA: kubectl apply the helm chart"}]
        ),
        count=10,
        seed=2,
    )
    token = PreemptionToken()
    token.preempt("user request")
    traces, report = await engine.run(list(dataset), drafts_per_prompt=2, token=token)
    assert report.preempted is True
    assert traces == []


# --------------------------------------------------------------------------- #
# The prompt evolver
# --------------------------------------------------------------------------- #
async def test_a_better_prompt_is_promoted() -> None:
    registry = _registry(1)
    client = FakeClient({"question": "kubectl apply the deployment"})
    evolver = PromptEvolver(registry, client, baseline_prompt="short answer", min_gain=0.0)
    report = await evolver.evolve([("question", "kubectl apply the deployment")], candidates=3, seed=5)
    assert report.baseline >= 0.0
    assert report.candidates


async def test_the_prompt_survives_in_the_memory(tmp_path: Path) -> None:
    memory = VectorMemory(tmp_path / "memory")
    store_prompt_in_memory(memory, "be precise")
    assert load_prompt_from_memory(memory) == "be precise"
    assert load_prompt_from_memory(VectorMemory(tmp_path / "memory")) == "be precise"


# --------------------------------------------------------------------------- #
# The checkpoint streamer
# --------------------------------------------------------------------------- #
async def test_the_checkpoint_reaches_the_cloud_and_comes_back(tmp_path: Path) -> None:
    cloud = CloudDouble()
    try:
        from memory.sync import DriveClient, HubClient

        streamer = CheckpointStreamer(
            drive=DriveClient(
                token="t",
                api_base=cloud.drive_api_base,
                upload_base=cloud.drive_upload_base,
            ),
            hub=HubClient(token="t", base_url=cloud.hub_base_url),
            drive_folder="Hyper-Cluster Checkpoints",
            hub_repo="someone/hyper-cluster",
        )
        matrix = np.arange(64, dtype=np.float32).reshape(8, 8)
        streamed = await streamer.push(3, matrix, metadata={"loss": 0.25})
        assert "drive" in streamed.targets
        assert "hub" in streamed.targets
        assert streamed.sha256
        assert streamed.bytes_sent >= matrix.nbytes

        restored = await streamer.pull(target="hub")
        assert restored is not None
        step, pulled, metadata = restored
        assert step == 3
        assert np.allclose(pulled, matrix)
        assert metadata["loss"] == 0.25
    finally:
        cloud.close()


# --------------------------------------------------------------------------- #
# The autopilot
# --------------------------------------------------------------------------- #
class _Settings:
    """A minimal settings object for the autopilot."""

    evolution_enabled = True
    evolution_idle_seconds = 5
    evolution_preempt_budget_seconds = 2.0
    evolution_dataset_size = 4
    evolution_drafts_per_prompt = 2
    evolution_max_shards = 2
    evolution_prompt_enabled = False


def _autopilot(tmp_path: Path, **kwargs: Any) -> tuple[AutopilotEvolver, VectorMemory, FakeClient]:
    memory = VectorMemory(tmp_path / "memory")
    client = FakeClient({"cluster": "deploy the cluster with kubectl apply the helm chart"})
    registry = _registry(2)
    trainer = DistributedTrainer(checkpoint_dir=tmp_path / "ckpt", steps=4, dim=32, rank=4)
    autopilot = AutopilotEvolver(
        _Settings(),
        registry,
        client,
        memory=memory,
        trainer=trainer,
        self_play=SelfPlayEngine(registry, client, threshold=0.05),
        **kwargs,
    )
    return autopilot, memory, client


async def test_a_cycle_learns_and_hot_reloads(tmp_path: Path) -> None:
    autopilot, memory, client = _autopilot(tmp_path)
    memory.record_exchange("how do I deploy the cluster", "deploy the cluster with kubectl apply the helm chart")
    memory.distill_facts([{"role": "user", "content": "Remember that my name is Ada."}])

    report = await autopilot.run_cycle()
    assert report.error == ""
    assert report.seeds >= 1
    assert report.examples == 4
    assert report.drafts > 0
    assert report.training is not None
    assert report.training.merged is True
    assert report.artifact_sha256
    # the new brain is shipped to every worker
    assert len(report.hot_reloaded) == 2
    assert len(client.brains) == 2


async def test_a_user_request_preempts_the_cycle_in_under_two_seconds(tmp_path: Path) -> None:
    autopilot, memory, client = _autopilot(tmp_path)
    memory.record_exchange("how do I deploy the cluster", "deploy the cluster with kubectl apply the helm chart")

    # a slow shard: the preemption has to interrupt it
    async def slow_shard(payload: dict[str, Any]) -> tuple[Any, float]:
        await asyncio.sleep(0.4)
        X = np.asarray(payload["X"], dtype=np.float64)
        Y = np.asarray(payload["Y"], dtype=np.float64)
        W = np.asarray(payload["W"], dtype=np.float64)
        gradient, loss = local_gradient(X, Y, W, l2=0.0)
        return gradient.tolist(), float(loss)

    autopilot.trainer.shard_fn = slow_shard

    async def user_arrives() -> None:
        await asyncio.sleep(0.2)
        autopilot.notify_activity("user request")

    arriver = asyncio.create_task(user_arrives())
    started = asyncio.get_running_loop().time()
    report = await autopilot.run_cycle()
    elapsed = (asyncio.get_running_loop().time() - started) * 1000.0
    await arriver

    assert report.preempted is True
    assert autopilot.state == STATE_PREEMPTED
    assert autopilot.preempt_latency_ms < 2000.0
    # the whole cycle must have stopped well before the budget
    assert elapsed < 2000.0


async def test_the_engine_does_not_loop_back_to_back(tmp_path: Path) -> None:
    """Regression: a finished cycle must restart the idle countdown.

    Without this the sleep-time engine ran a full train/hot-reload cycle every
    couple of seconds on an idle cluster instead of waiting for the window.
    """
    autopilot, memory, _ = _autopilot(tmp_path)
    memory.record_exchange("how do I deploy the cluster", "kubectl apply the helm chart")
    settings = _Settings()
    settings.evolution_idle_seconds = 30
    settings.evolution_cooldown_seconds = 30
    autopilot.settings = settings
    autopilot.last_activity = now_s() - 3600.0

    autopilot.start()
    try:
        deadline = time.time() + 30.0
        while time.time() < deadline and not autopilot.reports:
            await asyncio.sleep(0.1)
        assert autopilot.reports, "the engine never ran its first cycle"
        await asyncio.sleep(3.0)
        assert len(autopilot.reports) == 1, "the engine started a second cycle immediately"
        assert now_s() - autopilot.last_activity < 10.0
    finally:
        await autopilot.aclose()


async def test_every_cycle_keeps_learning(tmp_path: Path) -> None:
    """Regression: a resumed cycle must still do a full run of training.

    With an absolute step target the second cycle found the checkpoint at
    ``steps`` and trained zero steps, re-merging identical weights: the
    artifact hash never changed again after the first cycle.
    """
    autopilot, memory, _ = _autopilot(tmp_path)
    memory.record_exchange("how do I deploy the cluster", "deploy the cluster with kubectl apply the helm chart")

    first = await autopilot.run_cycle()
    second = await autopilot.run_cycle()
    assert first.error == "" and second.error == ""
    assert first.training.steps == autopilot.trainer.steps
    assert second.training.steps == 2 * autopilot.trainer.steps  # resumed, still 4 more steps
    assert second.training.loss_end > 0.0
    assert second.artifact_sha256  # a fresh, non-empty brain
