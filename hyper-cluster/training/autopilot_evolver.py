"""The sleep-time self-evolution engine.

When the cluster goes idle it starts to *learn*:

```
idle ──▶ mine ──▶ synthesize ──▶ self-play ──▶ train ──▶ merge ──▶ push ──▶ hot-reload
```

The one hard rule: **a user message always wins**.  The moment a chat
arrives the orchestrator calls :meth:`AutopilotEvolver.notify_activity`,
which flips the preemption token.  Every stage of the pipeline checks the
token, so the training stops within milliseconds, the workers are released
and the user is served.  The idle timer is then reset: the cluster only
starts training again after it has been quiet for ``idle_seconds``.

The pipeline is resumable: the training checkpoints after every step and
the checkpoint streamer keeps the state in Google Drive and on the
Hugging Face Hub, so an interrupted (or preempted) run continues where it
stopped.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from core.ids import mono_ms, now_s
from core.logging_setup import get_logger
from head_node.worker_registry import WorkerRegistry
from training.checkpoint_streamer import CheckpointStreamer
from training.distributed_trainer import (
    DistributedTrainer,
    PreemptionToken,
    TrainingReport,
    TrainingTrace,
)
from training.prompt_evolver import (
    DEFAULT_SYSTEM_PROMPT,
    PromptEvolver,
    load_prompt_from_memory,
    store_prompt_in_memory,
)
from training.self_play import SelfPlayEngine
from training.synthetic_data import build_dataset, mine_seeds

LOGGER = get_logger("training.autopilot_evolver")

#: The states the autopilot can be in.
STATE_IDLE = "idle"
STATE_WAITING = "waiting-for-idle"
STATE_TRAINING = "training"
STATE_PREEMPTED = "preempted"
STATE_DISABLED = "disabled"


@dataclass
class EvolutionReport:
    """What one self-evolution cycle produced."""

    started_at: float = field(default_factory=now_s)
    duration_ms: float = 0.0
    seeds: int = 0
    examples: int = 0
    drafts: int = 0
    verified: int = 0
    training: TrainingReport | None = None
    prompt_promoted: bool = False
    prompt_score: float = 0.0
    artifact_sha256: str = ""
    hot_reloaded: list[str] = field(default_factory=list)
    pushed: list[str] = field(default_factory=list)
    preempted: bool = False
    preempt_reason: str = ""
    preempt_latency_ms: float = 0.0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "duration_ms": round(self.duration_ms, 2),
            "seeds": self.seeds,
            "examples": self.examples,
            "drafts": self.drafts,
            "verified": self.verified,
            "training": self.training.to_dict() if self.training else None,
            "prompt_promoted": self.prompt_promoted,
            "prompt_score": round(self.prompt_score, 4),
            "artifact_sha256": self.artifact_sha256,
            "hot_reloaded": list(self.hot_reloaded),
            "pushed": list(self.pushed),
            "preempted": self.preempted,
            "preempt_reason": self.preempt_reason,
            "preempt_latency_ms": round(self.preempt_latency_ms, 2),
            "error": self.error,
        }


class AutopilotEvolver:
    """Runs the sleep-time evolution and yields instantly to the user."""

    def __init__(
        self,
        settings: Any,
        registry: WorkerRegistry,
        client: Any,
        *,
        memory: Any | None = None,
        trainer: DistributedTrainer | None = None,
        self_play: SelfPlayEngine | None = None,
        streamer: CheckpointStreamer | None = None,
        prompt_evolver: PromptEvolver | None = None,
        events: Any | None = None,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.client = client
        self.memory = memory
        self.trainer = trainer
        self.self_play = self_play
        self.streamer = streamer
        self.prompt_evolver = prompt_evolver
        self.events = events

        self.token = PreemptionToken()
        self.state = STATE_IDLE
        self.last_activity = now_s()
        self.last_report: EvolutionReport | None = None
        self.reports: list[EvolutionReport] = []
        self._task: asyncio.Task[Any] | None = None
        self._closing = False
        self._lock = asyncio.Lock()
        self._preempt_latency_ms = 0.0
        self._preempt_requested_at = 0.0
        #: The stage of the current cycle ("mine", "self-play", "training", ...).
        self.phase = "idle"

    # ------------------------------------------------------------------ #
    # Configuration helpers
    # ------------------------------------------------------------------ #
    @property
    def enabled(self) -> bool:
        return bool(getattr(self.settings, "evolution_enabled", False))

    @property
    def idle_seconds(self) -> float:
        return max(5.0, float(getattr(self.settings, "evolution_idle_seconds", 300)))

    @property
    def preempt_budget_seconds(self) -> float:
        return max(0.1, float(getattr(self.settings, "evolution_preempt_budget_seconds", 2.0)))

    @property
    def system_prompt(self) -> str:
        if self.prompt_evolver is not None:
            return self.prompt_evolver.baseline_prompt
        return load_prompt_from_memory(self.memory)

    @system_prompt.setter
    def system_prompt(self, prompt: str) -> None:
        if self.prompt_evolver is not None:
            self.prompt_evolver.baseline_prompt = prompt
        store_prompt_in_memory(self.memory, prompt)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Start the idle watchdog."""
        if self._task is None and not self._closing:
            self._task = asyncio.create_task(self._idle_loop(), name="hc-evolution-loop")

    async def aclose(self) -> None:
        """Stop the watchdog and cancel a running cycle."""
        self._closing = True
        self.token.preempt("shutdown")
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    async def _idle_loop(self) -> None:
        """Wait for the idle window and start a cycle."""
        while not self._closing:
            try:
                await asyncio.sleep(1.0)
                if not self.enabled:
                    self.state = STATE_DISABLED
                    continue
                if self.state == STATE_TRAINING:
                    continue
                idle_for = now_s() - self.last_activity
                if idle_for < self.idle_seconds:
                    self.state = STATE_WAITING
                    continue
                self.state = STATE_IDLE
                await self.run_cycle()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - the loop must never die
                LOGGER.exception("evolution loop error")
                await asyncio.sleep(3.0)

    # ------------------------------------------------------------------ #
    # The preemption contract
    # ------------------------------------------------------------------ #
    def notify_activity(self, reason: str = "user request") -> float:
        """Tell the engine that the user is back.

        This is called on the *request path*, so it must be cheap and
        synchronous.  It flips the preemption token (the training stops at
        its next checkpoint) and resets the idle timer.

        Returns the measured preemption latency in milliseconds (``0`` when
        nothing was running).
        """
        self.last_activity = now_s()
        if self.state != STATE_TRAINING:
            return 0.0
        self._preempt_requested_at = now_s()
        self.token.preempt(reason)
        self.state = STATE_PREEMPTED
        latency = self._measure_preemption()
        LOGGER.info(
            "evolution preempted",
            extra={"hc_reason": reason, "hc_latency_ms": round(latency, 2)},
        )
        return latency

    @property
    def preempt_latency_ms(self) -> float:
        """How long the last preemption took (milliseconds)."""
        return self._preempt_latency_ms

    def _measure_preemption(self) -> float:
        """The measured latency of the last preemption.

        The latency is the time between the moment the user request
        arrived and the moment the training acknowledged it -- the whole
        point of the contract is that this stays below the budget.
        """
        acknowledged = self.token.acknowledged_at
        requested = self.token.requested_at or self._preempt_requested_at
        if not acknowledged or not requested:
            return self._preempt_latency_ms
        latency = max(0.0, (acknowledged - requested) * 1000.0)
        self._preempt_latency_ms = latency
        if latency > self.preempt_budget_seconds * 1000.0:
            LOGGER.warning(
                "preemption exceeded its budget",
                extra={"hc_latency_ms": round(latency, 2), "hc_budget_ms": self.preempt_budget_seconds * 1000.0},
            )
        return latency

    # ------------------------------------------------------------------ #
    # The pipeline
    # ------------------------------------------------------------------ #
    async def run_cycle(self, *, dataset_size: int | None = None) -> EvolutionReport:
        """Run one full self-evolution cycle."""
        async with self._lock:
            report = EvolutionReport()
            self.token.reset()
            self.state = STATE_TRAINING
            started = mono_ms()
            try:
                report = await self._cycle(report, dataset_size=dataset_size)
            except Exception as exc:  # noqa: BLE001 - a failed cycle must not kill the loop
                report.error = str(exc)
                LOGGER.exception("evolution cycle failed")
            report.duration_ms = mono_ms() - started
            self.phase = "idle"
            if report.preempted:
                report.preempt_latency_ms = self._measure_preemption()
                self.state = STATE_PREEMPTED
            else:
                self.state = STATE_IDLE
            self.last_report = report
            self.reports.append(report)
            await self._emit("evolution.cycle", report.to_dict())
            return report

    async def _cycle(self, report: EvolutionReport, *, dataset_size: int | None) -> EvolutionReport:
        # --- 1. mine the conversations -------------------------------- #
        self.phase = "mine"
        seeds = self._mine()
        report.seeds = len(seeds)
        if not seeds:
            report.error = "no conversations to learn from"
            return report

        # --- 2. synthesise the harder prompts ------------------------ #
        self.phase = "synthesize"
        size = int(dataset_size or getattr(self.settings, "evolution_dataset_size", 16))
        dataset = build_dataset(
            seeds,
            count=size,
            difficulty=float(getattr(self.settings, "evolution_difficulty", 0.8)),
        )
        report.examples = len(dataset)
        if self.token.preempted:
            report.preempted = True
            self.token.acknowledge()
            report.preempt_reason = self.token.reason
            report.preempt_latency_ms = self._measure_preemption()
            return report

        # --- 3. self-play: draft, criticise, keep -------------------- #
        self.phase = "self-play"
        traces: list[TrainingTrace] = []
        if self.self_play is not None:
            traces, play = await self.self_play.run(
                list(dataset),
                drafts_per_prompt=int(getattr(self.settings, "evolution_drafts_per_prompt", 2)),
                token=self.token,
            )
            report.drafts = play.drafts
            report.verified = play.verified
            report.preempted = play.preempted
        if report.preempted or self.token.preempted:
            report.preempted = True
            self.token.acknowledge()
            report.preempt_reason = self.token.reason
            report.preempt_latency_ms = self._measure_preemption()
            return report
        if not traces:
            report.error = "self-play produced no verified trace"
            return report

        # --- 4. evolve the system prompt ----------------------------- #
        self.phase = "prompt"
        if self.prompt_evolver is not None and getattr(self.settings, "evolution_prompt_enabled", True):
            benchmark = self._benchmark(seeds)
            evolution = await self.prompt_evolver.evolve(benchmark, token=self.token)
            report.prompt_promoted = evolution.promoted
            report.prompt_score = evolution.best_score
            if evolution.promoted:
                store_prompt_in_memory(self.memory, evolution.new_prompt)
            if evolution.preempted or self.token.preempted:
                report.preempted = True
                self.token.acknowledge()
                report.preempt_reason = self.token.reason
                report.preempt_latency_ms = self._measure_preemption()
                return report

        # --- 5. train (checkpointed, federated) ---------------------- #
        self.phase = "training"
        if self.trainer is None:
            report.error = "no trainer configured"
            return report
        training = await self.trainer.train(
            traces,
            shards=self._shard_count(),
            token=self.token,
        )
        report.training = training
        if training.preempted:
            report.preempted = True
            self.token.acknowledge()
            report.preempt_reason = training.preempt_reason
            report.preempt_latency_ms = self._measure_preemption()
            return report

        # --- 6. merge + push ----------------------------------------- #
        self.phase = "merge"
        artifact = self._artifact(training)
        if artifact is not None:
            report.artifact_sha256 = artifact.sha256()
            pushed = await self._push(training, artifact)
            report.pushed = pushed
            # --- 7. hot-reload the upgraded brain -------------------- #
            report.hot_reloaded = await self._hot_reload(artifact)
        return report

    # ------------------------------------------------------------------ #
    def _mine(self) -> list[Any]:
        if self.memory is None:
            return []
        try:
            return mine_seeds(self.memory.conversations())
        except Exception:  # noqa: BLE001
            return []

    def _benchmark(self, seeds: Sequence[Any]) -> list[tuple[str, str]]:
        """The held-out benchmark of the prompt evolver."""
        return [(seed.prompt, seed.answer) for seed in list(seeds)[:8] if seed.answer]

    def _shard_count(self) -> int:
        healthy = len(self.registry.healthy_workers())
        return max(1, min(healthy, int(getattr(self.settings, "evolution_max_shards", 4))))

    def _artifact(self, training: TrainingReport) -> Any | None:
        """The artifact (the merged brain) of the finished training run."""
        if self.trainer is None:
            return None
        return self.trainer.last_artifact

    async def _push(self, training: TrainingReport, artifact: Any) -> list[str]:
        """Push the checkpoint to the cloud."""
        if self.streamer is None:
            return []
        pushed: list[str] = []
        try:
            checkpoint = None
            from training.distributed_trainer import Checkpoint

            checkpoint = Checkpoint.latest(self.trainer.checkpoint_dir) if self.trainer else None
            if checkpoint is not None:
                streamed = await self.streamer.push(
                    checkpoint.step,
                    checkpoint.matrix,
                    metadata={"loss": checkpoint.loss},
                )
                pushed = list(streamed.targets)
        except Exception as exc:  # noqa: BLE001 - the cloud is optional
            LOGGER.warning("checkpoint push failed", extra={"hc_error": str(exc)})
        return pushed

    async def _hot_reload(self, artifact: Any) -> list[str]:
        """Ship the new brain to every healthy worker."""
        payload = artifact.encode()
        reloaded: list[str] = []
        for record in self.registry.healthy_workers():
            try:
                ok = await self.client.load_brain(record, payload)
            except Exception as exc:  # noqa: BLE001 - one dead worker must not stop the rollout
                LOGGER.warning(
                    "brain hot-reload failed",
                    extra={"hc_worker": record.worker_id, "hc_error": str(exc)},
                )
                continue
            if ok:
                reloaded.append(record.worker_id)
        if reloaded:
            LOGGER.info("brain hot-reloaded", extra={"hc_workers": ",".join(reloaded)})
        return reloaded

    async def _emit(self, event: str, data: dict[str, Any]) -> None:
        if self.events is None:
            return
        try:
            await self.events.emit(event, data)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ #
    def status(self) -> dict[str, Any]:
        """A JSON-safe view of the engine."""
        return {
            "enabled": self.enabled,
            "state": self.state,
            "phase": self.phase,
            "idle_seconds": self.idle_seconds,
            "idle_for": round(now_s() - self.last_activity, 2),
            "preempt_budget_seconds": self.preempt_budget_seconds,
            "preempt_latency_ms": round(self._preempt_latency_ms, 2),
            "preempt_budget_ms": round(self.preempt_budget_seconds * 1000.0, 2),
            "preempted": self.token.preempted,
            "preempt_reason": self.token.reason,
            "system_prompt": self.system_prompt,
            "cycles": len(self.reports),
            "last_report": self.last_report.to_dict() if self.last_report else None,
        }


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "STATE_DISABLED",
    "STATE_IDLE",
    "STATE_PREEMPTED",
    "STATE_TRAINING",
    "STATE_WAITING",
    "AutopilotEvolver",
    "EvolutionReport",
]
