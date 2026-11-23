"""Distributed, checkpointed QLoRA training with an automatic merge.

The trainer runs on the head node and *delegates the math* to the workers:
every step is one communication round in which each worker computes the
gradient of its own data shard and the head node averages the gradients
(federated averaging).  A checkpoint is written after every step, so a
crashed -- or *preempted* -- run resumes exactly where it stopped.

When the training finishes, the dense update is factorised into a rank-``r``
LoRA adapter, merged into the (int8-quantised) base matrix and shipped back
to the workers as a **brain artifact** -- the hot-reload of the upgraded
brain.

The whole loop is preemptible: :class:`PreemptionToken` is checked between
every shard round, so a user message stops the training within milliseconds
and the workers go back to serving the user.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.ids import mono_ms, now_s
from core.logging_setup import get_logger
from training.qlora import (
    DEFAULT_DIM,
    DEFAULT_L2,
    DEFAULT_RANK,
    DEFAULT_VOCAB_SIZE,
    BrainArtifact,
    apply_update,
    average_gradients,
    build_vocabulary,
    featurize,
    local_gradient,
    lora_factorize,
    merge,
    quantize,
    unpack,
)

LOGGER = get_logger("training.distributed_trainer")


class PreemptedError(RuntimeError):
    """Raised inside the training loop when a user request preempts it."""


class PreemptionToken:
    """The switch that a user request flips.

    The token is *synchronous* to flip: the orchestrator calls
    :meth:`preempt` on the request path, and every coroutine that is
    training notices at its next checkpoint.  The timestamp of the request
    is kept so the preemption latency can be reported (and asserted on).
    """

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self.reason: str = ""
        self.requested_at: float = 0.0
        self.acknowledged_at: float = 0.0

    def preempt(self, reason: str = "user request") -> None:
        """Ask the training to stop."""
        if self._event.is_set():
            return
        self.reason = reason
        self.requested_at = now_s()
        self._event.set()

    def reset(self) -> None:
        """Clear the token for the next run."""
        self._event = asyncio.Event()
        self.reason = ""
        self.requested_at = 0.0
        self.acknowledged_at = 0.0

    @property
    def preempted(self) -> bool:
        return self._event.is_set()

    def acknowledge(self) -> float:
        """Record that the preemption was noticed (the latency clock stops)."""
        if not self._event.is_set():
            return 0.0
        if not self.acknowledged_at:
            self.acknowledged_at = now_s()
        return self.acknowledged_at

    def guard(self) -> None:
        """Raise when the training must stop."""
        if self._event.is_set():
            self.acknowledged_at = now_s()
            raise PreemptedError(self.reason or "preempted")

    async def wait(self, timeout: float | None = None) -> bool:
        """Wait for a preemption (used to race the network calls).

        ``timeout=None`` waits until the token is flipped; a timeout of
        ``0`` (or less) only polls the current state.
        """
        if timeout is not None and timeout <= 0:
            if self._event.is_set():
                self.acknowledged_at = now_s()
                return True
            return False
        try:
            if timeout is None:
                await self._event.wait()
            else:
                await asyncio.wait_for(self._event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return False
        self.acknowledged_at = now_s()
        return True


@dataclass
class TrainingTrace:
    """One verified training example."""

    prompt: str
    response: str
    score: float = 0.0
    worker_id: str = ""
    critic: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "response": self.response,
            "score": round(self.score, 4),
            "worker_id": self.worker_id,
            "critic": self.critic,
            "tags": list(self.tags),
        }


@dataclass
class Checkpoint:
    """A training checkpoint (adapter + optimiser state + metrics)."""

    step: int
    loss: float
    matrix: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=now_s)

    def save(self, directory: str | Path) -> Path:
        """Write the checkpoint atomically."""
        target = Path(directory).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"checkpoint-{self.step:06d}.npz"
        tmp = path.with_suffix(".npz.tmp")
        buffer = _npz_bytes(self.matrix)
        tmp.write_bytes(buffer)
        tmp.replace(path)
        meta_path = target / f"checkpoint-{self.step:06d}.json"
        meta_tmp = meta_path.with_suffix(".json.tmp")
        meta_tmp.write_text(
            json.dumps(
                {
                    "step": self.step,
                    "loss": self.loss,
                    "created_at": self.created_at,
                    **self.metadata,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        meta_tmp.replace(meta_path)
        (target / "latest.json").write_text(
            json.dumps({"step": self.step, "loss": self.loss, "created_at": self.created_at}),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, directory: str | Path, step: int) -> Checkpoint | None:
        """Load a checkpoint by step."""
        target = Path(directory).expanduser() / f"checkpoint-{step:06d}.npz"
        if not target.is_file():
            return None
        matrix = _npz_load(target)
        meta_path = target.with_suffix(".json")
        metadata: dict[str, Any] = {}
        if meta_path.is_file():
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                metadata = {}
        return cls(
            step=int(metadata.get("step") or step),
            loss=float(metadata.get("loss") or 0.0),
            matrix=matrix,
            metadata=metadata,
            created_at=float(metadata.get("created_at") or now_s()),
        )

    @classmethod
    def latest(cls, directory: str | Path) -> Checkpoint | None:
        """The newest checkpoint of a directory."""
        target = Path(directory).expanduser()
        latest = target / "latest.json"
        if not latest.is_file():
            return None
        try:
            step = int(json.loads(latest.read_text(encoding="utf-8")).get("step") or 0)
        except (OSError, json.JSONDecodeError, ValueError):
            return None
        return cls.load(target, step)


@dataclass
class TrainingReport:
    """What one training run produced."""

    steps: int = 0
    loss_start: float = 0.0
    loss_end: float = 0.0
    examples: int = 0
    shards: int = 0
    checkpoints: int = 0
    rank: int = DEFAULT_RANK
    dim: int = DEFAULT_DIM
    vocab_size: int = 0
    artifact_sha256: str = ""
    merged: bool = False
    preempted: bool = False
    preempt_reason: str = ""
    duration_ms: float = 0.0
    started_at: float = field(default_factory=now_s)

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "loss_start": round(self.loss_start, 6),
            "loss_end": round(self.loss_end, 6),
            "examples": self.examples,
            "shards": self.shards,
            "checkpoints": self.checkpoints,
            "rank": self.rank,
            "dim": self.dim,
            "vocab_size": self.vocab_size,
            "artifact_sha256": self.artifact_sha256,
            "merged": self.merged,
            "preempted": self.preempted,
            "preempt_reason": self.preempt_reason,
            "duration_ms": round(self.duration_ms, 2),
            "started_at": self.started_at,
        }


def _npz_bytes(matrix: np.ndarray) -> bytes:
    import io

    buffer = io.BytesIO()
    np.savez_compressed(buffer, matrix=matrix)
    return buffer.getvalue()


def _npz_load(path: Path) -> np.ndarray:

    with np.load(path, allow_pickle=False) as data:
        return np.array(data["matrix"])


# --------------------------------------------------------------------------- #
# The trainer
# --------------------------------------------------------------------------- #
class DistributedTrainer:
    """Federated, checkpointed QLoRA training across the worker mesh."""

    def __init__(
        self,
        *,
        checkpoint_dir: str | Path,
        dim: int = DEFAULT_DIM,
        rank: int = DEFAULT_RANK,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        l2: float = DEFAULT_L2,
        learning_rate: float = 0.5,
        steps: int = 8,
        checkpoint_every: int = 1,
        shard_fn: Callable[..., Any] | None = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir).expanduser()
        self.dim = max(16, int(dim))
        self.rank = max(1, int(rank))
        self.vocab_size = max(8, int(vocab_size))
        self.l2 = float(l2)
        self.learning_rate = float(learning_rate)
        self.steps = max(1, int(steps))
        self.checkpoint_every = max(1, int(checkpoint_every))
        #: The callable that computes a shard's gradient on a worker.
        self.shard_fn = shard_fn
        #: The artifact of the last finished run (the merged brain).
        self.last_artifact: BrainArtifact | None = None

    # ------------------------------------------------------------------ #
    def shard_plan(self, count: int, shards: int) -> list[list[int]]:
        """Split the indices of a dataset into ``shards`` contiguous parts."""
        shards = max(1, int(shards))
        if count <= 0:
            return []
        plan: list[list[int]] = []
        size = max(1, (count + shards - 1) // shards)
        for start in range(0, count, size):
            plan.append(list(range(start, min(count, start + size))))
        return plan

    # ------------------------------------------------------------------ #
    async def train(
        self,
        traces: Sequence[TrainingTrace],
        *,
        shards: int = 2,
        token: PreemptionToken | None = None,
        resume: bool = True,
        on_step: Callable[[int, float], Any] | None = None,
    ) -> TrainingReport:
        """Run the federated training loop.

        ``shards`` is the number of worker shards the dataset is split
        into.  When a ``shard_fn`` is configured, the gradient of every
        shard is computed by a worker; otherwise the head node computes it
        locally (the maths is identical).
        """
        started = mono_ms()
        report = TrainingReport(examples=len(traces), shards=shards)
        if not traces:
            report.duration_ms = mono_ms() - started
            return report

        vocab = build_vocabulary([trace.response for trace in traces], self.vocab_size)
        report.vocab_size = len(vocab)
        prompts = [trace.prompt for trace in traces]
        answers = [trace.response for trace in traces]
        X, Y = featurize(prompts, answers, vocab, self.dim)

        plan = self.shard_plan(len(traces), shards)
        report.shards = len(plan)

        W = np.zeros((self.dim, len(vocab)), dtype=np.float32)
        start_step = 0
        if resume:
            checkpoint = Checkpoint.latest(self.checkpoint_dir)
            if checkpoint is not None and checkpoint.matrix.shape == W.shape:
                W = checkpoint.matrix.astype(np.float32)
                start_step = checkpoint.step
                LOGGER.info("training resumed", extra={"hc_step": start_step})
        # ``steps`` is how much work *this run* does.  Resuming continues from
        # the checkpoint, it does not shrink the run: with an absolute target
        # (``range(start_step, self.steps)``) every cycle after the first one
        # trained zero steps and re-merged the same weights, so the "sleep-time
        # learning" never learned anything new.
        target_step = start_step + self.steps

        weights = [len(indices) for indices in plan] or [1]
        report.loss_start = 0.0

        try:
            for step in range(start_step, target_step):
                if token is not None:
                    token.guard()
                gradients: list[np.ndarray] = []
                losses: list[float] = []
                for indices in plan:
                    if token is not None:
                        token.guard()
                    if self.shard_fn is not None:
                        gradient, loss = await self._remote_shard(X, Y, W, indices, token)
                    else:
                        gradient, loss = local_gradient(X[indices], Y[indices], W, l2=self.l2)
                    gradients.append(gradient)
                    losses.append(loss)
                merged_gradient = average_gradients(gradients, weights)
                W = apply_update(W, merged_gradient, self.learning_rate)
                report.steps = step + 1
                report.loss_end = float(np.mean(losses)) if losses else 0.0
                if report.loss_start == 0.0:
                    report.loss_start = report.loss_end
                if on_step is not None:
                    result = on_step(report.steps, report.loss_end)
                    if asyncio.iscoroutine(result):
                        await result
                if report.steps % self.checkpoint_every == 0:
                    Checkpoint(
                        step=report.steps,
                        loss=report.loss_end,
                        matrix=W.copy(),
                        metadata={"examples": len(traces), "vocab": len(vocab)},
                    ).save(self.checkpoint_dir)
                    report.checkpoints += 1
        except PreemptedError as exc:
            if token is not None:
                token.acknowledge()
            report.preempted = True
            report.preempt_reason = str(exc)
            report.duration_ms = mono_ms() - started
            LOGGER.info(
                "training preempted",
                extra={"hc_reason": str(exc), "hc_steps": report.steps},
            )
            return report

        artifact = self.merge(W, vocab, steps=report.steps, loss=report.loss_end)
        self.last_artifact = artifact
        report.artifact_sha256 = artifact.sha256()
        report.merged = True
        report.duration_ms = mono_ms() - started
        LOGGER.info(
            "training finished",
            extra={
                "hc_steps": report.steps,
                "hc_loss": round(report.loss_end, 6),
                "hc_sha": report.artifact_sha256[:12],
            },
        )
        return report

    async def _remote_shard(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        W: np.ndarray,
        indices: Sequence[int],
        token: PreemptionToken | None,
    ) -> tuple[np.ndarray, float]:
        """Ask a worker for the gradient of one shard."""
        assert self.shard_fn is not None
        payload = {
            "X": X[list(indices)].tolist(),
            "Y": Y[list(indices)].tolist(),
            "W": W.tolist(),
            "l2": self.l2,
        }
        result = self.shard_fn(payload)
        if not asyncio.iscoroutine(result):
            gradient, loss = result
            return np.asarray(gradient, dtype=np.float32), float(loss)
        if token is None:
            gradient, loss = await result
            return np.asarray(gradient, dtype=np.float32), float(loss)

        # Race the network call against the preemption: the user must not
        # wait for a slow shard.
        pending = asyncio.ensure_future(result)
        waiter = asyncio.ensure_future(token.wait(None))
        try:
            await asyncio.wait({pending, waiter}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            if not pending.done():
                pending.cancel()
            waiter.cancel()
        if pending.done():
            gradient, loss = await pending
            return np.asarray(gradient, dtype=np.float32), float(loss)
        # The user is back: drop the shard and stop the training.
        token.guard()
        raise PreemptedError(token.reason or "preempted")

    # ------------------------------------------------------------------ #
    def merge(
        self,
        W: np.ndarray,
        vocab: Sequence[str],
        *,
        steps: int = 0,
        loss: float = 0.0,
    ) -> BrainArtifact:
        """Factorise the dense update and merge it into the base matrix."""
        delta = np.asarray(W, dtype=np.float32)
        A, B = lora_factorize(delta, self.rank)
        base_quantized, base_scale = quantize(delta)
        merged = merge(base_quantized, base_scale, A, B)
        artifact = BrainArtifact(
            dim=self.dim,
            rank=self.rank,
            vocab=list(vocab),
            steps=steps,
            loss=loss,
            metadata={"training": "qlora", "merge": "svd"},
        )
        artifact.base_quantized = _pack_array(base_quantized)
        artifact.base_scale = _pack_array(base_scale)
        artifact.A = _pack_array(A)
        artifact.B = _pack_array(B)
        artifact.merged = _pack_array(merged)
        return artifact

    # ------------------------------------------------------------------ #
    def cleanup(self) -> None:
        """Remove the checkpoints (they live on Drive / the Hub)."""
        shutil.rmtree(self.checkpoint_dir, ignore_errors=True)

    def checkpoints(self) -> list[dict[str, Any]]:
        """The checkpoints on disk, newest first."""
        if not self.checkpoint_dir.is_dir():
            return []
        found: list[dict[str, Any]] = []
        for path in sorted(self.checkpoint_dir.glob("checkpoint-*.json")):
            try:
                found.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return list(reversed(found))


def _pack_array(array: np.ndarray) -> bytes:
    import io

    buffer = io.BytesIO()
    np.savez_compressed(buffer, payload=array)
    return buffer.getvalue()


def shard_gradient(payload: dict[str, Any]) -> tuple[list[list[float]], float]:
    """The worker-side shard computation (used by the ``/v1/train/shard`` route)."""
    X = np.asarray(payload.get("X") or [], dtype=np.float32)
    Y = np.asarray(payload.get("Y") or [], dtype=np.float32)
    W = np.asarray(payload.get("W") or [], dtype=np.float32)
    l2 = float(payload.get("l2") or DEFAULT_L2)
    gradient, loss = local_gradient(X, Y, W, l2=l2)
    return gradient.tolist(), loss


__all__ = [
    "Checkpoint",
    "DistributedTrainer",
    "PreemptedError",
    "PreemptionToken",
    "TrainingReport",
    "TrainingTrace",
    "shard_gradient",
    "unpack",
]
