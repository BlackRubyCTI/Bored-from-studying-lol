"""Prompt evolution: the system prompt is a hyper-parameter too.

A cluster that trains its weights can also *tune its instructions*.  The
evolver

1. mutates the current system prompt (heuristically and with the model
   itself),
2. benchmarks every candidate on a held-out set of real questions,
3. promotes the candidate that beats the incumbent -- and only then.

The promotion is written to the memory, so the new prompt survives a
restart and is picked up by the next inference request.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from core.ids import now_s
from core.logging_setup import get_logger
from core.protocol import ChatMessage, ChatRequest
from head_node.worker_registry import WorkerRecord, WorkerRegistry
from training.self_play import similarity

LOGGER = get_logger("training.prompt_evolver")

#: The default system prompt of the cluster.
DEFAULT_SYSTEM_PROMPT = (
    "You are Hyper-Cluster Omega, a precise and honest assistant. "
    "Answer the question directly, say what you do not know, and keep the answer short."
)

#: The mutations that are applied to the incumbent prompt.
MUTATIONS: tuple[tuple[str, str], ...] = (
    ("evidence", "Cite the evidence you used for every non-obvious claim."),
    ("risk", "State the biggest risk in your answer and how to reduce it."),
    ("example", "Give one concrete example for every abstract statement."),
    ("uncertainty", "Say explicitly how confident you are and what would change your mind."),
    ("structure", "Answer with a short conclusion first, then the reasoning."),
    ("audience", "Write for a technical reader who knows the domain."),
)


@dataclass
class PromptCandidate:
    """One candidate system prompt and its benchmark score."""

    prompt: str
    score: float = 0.0
    baseline: float = 0.0
    origin: str = "mutation"
    samples: int = 0
    created_at: float = field(default_factory=now_s)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "score": round(self.score, 4),
            "baseline": round(self.baseline, 4),
            "origin": self.origin,
            "samples": self.samples,
            "created_at": self.created_at,
        }


@dataclass
class PromptEvolutionReport:
    """What one evolution round produced."""

    baseline: float = 0.0
    best_score: float = 0.0
    promoted: bool = False
    previous_prompt: str = ""
    new_prompt: str = ""
    candidates: list[PromptCandidate] = field(default_factory=list)
    duration_ms: float = 0.0
    preempted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": round(self.baseline, 4),
            "best_score": round(self.best_score, 4),
            "promoted": self.promoted,
            "previous_prompt": self.previous_prompt,
            "new_prompt": self.new_prompt,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "duration_ms": round(self.duration_ms, 2),
            "preempted": self.preempted,
        }


def mutate_prompt_text(prompt: str, rng: random.Random) -> tuple[str, str]:
    """Apply one mutation to a system prompt."""
    name, instruction = MUTATIONS[rng.randrange(len(MUTATIONS))]
    return f"{prompt.rstrip()} {instruction}".strip(), name


class PromptEvolver:
    """Mutates, benchmarks and promotes the system prompt."""

    def __init__(
        self,
        registry: WorkerRegistry,
        client: Any,
        *,
        baseline_prompt: str = DEFAULT_SYSTEM_PROMPT,
        min_gain: float = 0.05,
        max_tokens: int = 400,
        temperature: float = 0.2,
    ) -> None:
        self.registry = registry
        self.client = client
        self.baseline_prompt = baseline_prompt
        self.min_gain = float(min_gain)
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.history: list[PromptEvolutionReport] = []

    # ------------------------------------------------------------------ #
    def _worker(self) -> WorkerRecord | None:
        healthy = self.registry.healthy_workers()
        if not healthy:
            return None
        return max(healthy, key=lambda record: (record.model_params_b or 0.0))

    async def _answer(self, record: WorkerRecord, system: str, question: str) -> str:
        request = ChatRequest(
            messages=[
                ChatMessage(role="system", content=system),
                ChatMessage(role="user", content=question),
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=False,
        )
        completion = await self.client.chat(record, request)
        return completion.text or ""

    # ------------------------------------------------------------------ #
    async def evaluate(
        self,
        prompt: str,
        benchmark: Sequence[tuple[str, str]],
        *,
        token: Any | None = None,
    ) -> float:
        """Score a system prompt on a benchmark of ``(question, reference)``."""
        record = self._worker()
        if record is None or not benchmark:
            return 0.0
        total = 0.0
        for question, reference in benchmark:
            if token is not None and getattr(token, "preempted", False):
                break
            try:
                answer = await self._answer(record, prompt, question)
            except Exception:  # noqa: BLE001 - a dead worker just costs a point
                continue
            total += similarity(answer, reference)
        return total / max(1, len(benchmark))

    async def evolve(
        self,
        benchmark: Sequence[tuple[str, str]],
        *,
        candidates: int = 4,
        seed: int = 4242,
        token: Any | None = None,
    ) -> PromptEvolutionReport:
        """Run one evolution round and promote the winner (if it wins)."""
        from core.ids import mono_ms

        started = mono_ms()
        rng = random.Random(seed)
        report = PromptEvolutionReport(previous_prompt=self.baseline_prompt)
        if not benchmark:
            report.duration_ms = mono_ms() - started
            return report

        report.baseline = await self.evaluate(self.baseline_prompt, benchmark, token=token)
        pool: list[PromptCandidate] = []
        seen = {self.baseline_prompt.strip().lower()}
        for _ in range(max(1, int(candidates))):
            if token is not None and getattr(token, "preempted", False):
                report.preempted = True
                break
            mutated, origin = mutate_prompt_text(self.baseline_prompt, rng)
            key = mutated.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            score = await self.evaluate(mutated, benchmark, token=token)
            pool.append(
                PromptCandidate(prompt=mutated, score=score, baseline=report.baseline, origin=origin)
            )
        report.candidates = pool
        best = max(pool, key=lambda candidate: candidate.score, default=None)
        if best is not None:
            report.best_score = best.score
            if best.score > report.baseline + self.min_gain:
                self.baseline_prompt = best.prompt
                report.promoted = True
                report.new_prompt = best.prompt
        report.duration_ms = mono_ms() - started
        self.history.append(report)
        LOGGER.info(
            "prompt evolution finished",
            extra={
                "hc_baseline": round(report.baseline, 4),
                "hc_best": round(report.best_score, 4),
                "hc_promoted": report.promoted,
            },
        )
        return report


def load_prompt_from_memory(memory: Any, key: str = "system_prompt") -> str:
    """Read the promoted system prompt from the memory."""
    if memory is None:
        return DEFAULT_SYSTEM_PROMPT
    try:
        for chunk in memory.facts():
            if chunk.metadata.get("kind") == key:
                return chunk.text
    except Exception:  # noqa: BLE001 - a broken memory must not break inference
        pass
    return DEFAULT_SYSTEM_PROMPT


def store_prompt_in_memory(memory: Any, prompt: str, key: str = "system_prompt") -> None:
    """Persist the promoted system prompt in the memory."""
    if memory is None or not prompt:
        return
    try:
        memory.add_fact(prompt, metadata={"kind": key, "origin": "prompt-evolver"})
    except Exception:  # noqa: BLE001
        pass


def extract_questions(text: str) -> list[str]:
    """The questions of a text (used to build the benchmark)."""
    questions = [
        match.strip()
        for match in re.findall(r"Q:\s*(.+)", text or "")
        if len(match.strip()) > 8
    ]
    return questions


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "MUTATIONS",
    "PromptCandidate",
    "PromptEvolutionReport",
    "PromptEvolver",
    "extract_questions",
    "load_prompt_from_memory",
    "mutate_prompt_text",
    "store_prompt_in_memory",
]
