"""Self-play: the cluster argues with itself and keeps the good answers.

For every synthetic prompt the engine

1. fans the prompt out to several workers (each one drafts an answer),
2. asks the *critic* -- the worker with the strongest resident model -- to
   score every draft against the reference,
3. keeps only the drafts the critic verified.

The kept traces are the training set of the sleep-time engine.  Everything
is preemptible: the fan-out is cancelled as soon as a user request
arrives, and the workers are released immediately.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from core.logging_setup import get_logger
from core.protocol import ChatMessage, ChatRequest
from head_node.worker_registry import WorkerRecord, WorkerRegistry
from training.distributed_trainer import PreemptionToken, TrainingTrace
from training.synthetic_data import SyntheticExample

LOGGER = get_logger("training.self_play")

#: The critic is asked for a score between 0 and 10.
CRITICAL_SYSTEM = (
    "You are a strict reviewer. Compare the candidate answer with the reference answer. "
    "Reply with a single number between 0 and 10 (10 = equivalent and complete) and "
    "nothing else."
)


@dataclass
class Draft:
    """One candidate answer of the self-play fan-out."""

    worker_id: str
    text: str
    score: float = 0.0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "text": self.text,
            "score": round(self.score, 4),
            "error": self.error,
        }


@dataclass
class SelfPlayReport:
    """What one self-play round produced."""

    prompts: int = 0
    drafts: int = 0
    verified: int = 0
    rejected: int = 0
    workers: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    preempted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompts": self.prompts,
            "drafts": self.drafts,
            "verified": self.verified,
            "rejected": self.rejected,
            "workers": list(self.workers),
            "duration_ms": round(self.duration_ms, 2),
            "preempted": self.preempted,
        }


def parse_score(text: str) -> float:
    """Extract the critic's score from its answer."""
    match = re.search(r"(\d+(?:\.\d+)?)", (text or "").strip())
    if not match:
        return 0.0
    try:
        value = float(match.group(1))
    except ValueError:
        return 0.0
    if value > 10:
        # The critic was asked for 0-10, but some models answer 0-100.
        value = value / 100.0 if value <= 100 else 10.0
    elif value <= 1.0:
        # Some models answer on a 0-1 scale.
        value = value * 10.0
    return max(0.0, min(10.0, value)) / 10.0


def similarity(left: str, right: str) -> float:
    """A cheap token-overlap similarity (used when no critic is available)."""
    left_tokens = set(re.findall(r"[a-z0-9']+", (left or "").lower()))
    right_tokens = set(re.findall(r"[a-z0-9']+", (right or "").lower()))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


class SelfPlayEngine:
    """Fans prompts out over the mesh and keeps the verified drafts."""

    def __init__(
        self,
        registry: WorkerRegistry,
        client: Any,
        *,
        threshold: float = 0.55,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> None:
        self.registry = registry
        self.client = client
        self.threshold = float(threshold)
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)

    # ------------------------------------------------------------------ #
    def _workers(self, count: int) -> list[WorkerRecord]:
        healthy = [record for record in self.registry.healthy_workers()]
        healthy.sort(key=lambda record: (record.model_params_b or 0.0), reverse=True)
        if count <= 0:
            return healthy
        return healthy[:count]

    def _critic(self, workers: Sequence[WorkerRecord]) -> WorkerRecord | None:
        """The critic is the worker with the strongest resident model."""
        if not workers:
            return None
        return max(workers, key=lambda record: (record.model_params_b or 0.0))

    async def _ask(
        self,
        record: WorkerRecord,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        request = ChatRequest(
            messages=messages,
            max_tokens=max_tokens or self.max_tokens,
            temperature=self.temperature if temperature is None else temperature,
            stream=False,
        )
        completion = await self.client.chat(record, request)
        return completion.text or ""

    # ------------------------------------------------------------------ #
    async def run(
        self,
        examples: Sequence[SyntheticExample],
        *,
        drafts_per_prompt: int = 2,
        token: PreemptionToken | None = None,
        use_critic: bool = True,
    ) -> tuple[list[TrainingTrace], SelfPlayReport]:
        """Run the self-play round over the examples."""
        from core.ids import mono_ms

        started = mono_ms()
        report = SelfPlayReport(prompts=len(examples))
        traces: list[TrainingTrace] = []
        workers = self._workers(max(1, int(drafts_per_prompt)))
        report.workers = [record.worker_id for record in workers]
        if not workers:
            report.duration_ms = mono_ms() - started
            return traces, report
        critic = self._critic(workers)

        for example in examples:
            if token is not None and token.preempted:
                report.preempted = True
                break
            drafts: list[Draft] = []
            fan_out = workers[: max(1, int(drafts_per_prompt))]
            results = await asyncio.gather(
                *[
                    self._draft(record, example)
                    for record in fan_out
                ],
                return_exceptions=True,
            )
            for record, result in zip(fan_out, results, strict=False):
                if isinstance(result, BaseException):
                    drafts.append(Draft(worker_id=record.worker_id, text="", error=str(result)))
                    continue
                drafts.append(result)
            report.drafts += len(drafts)

            if token is not None and token.preempted:
                report.preempted = True
                break

            scored = await self._score(drafts, example, critic, token=token, use_critic=use_critic)
            best = max(scored, key=lambda draft: draft.score, default=None)
            if best is None or not best.text:
                report.rejected += 1
                continue
            if best.score < self.threshold:
                report.rejected += 1
                continue
            report.verified += 1
            traces.append(
                TrainingTrace(
                    prompt=example.prompt,
                    response=best.text,
                    score=best.score,
                    worker_id=best.worker_id,
                    critic=critic.worker_id if critic else "",
                    tags=list(example.tags),
                )
            )
        report.duration_ms = mono_ms() - started
        LOGGER.info(
            "self-play finished",
            extra={
                "hc_prompts": report.prompts,
                "hc_drafts": report.drafts,
                "hc_verified": report.verified,
                "hc_preempted": report.preempted,
            },
        )
        return traces, report

    # ------------------------------------------------------------------ #
    async def _draft(self, record: WorkerRecord, example: SyntheticExample) -> Draft:
        messages = [
            ChatMessage(role="system", content="You are a helpful assistant. Answer precisely."),
            ChatMessage(role="user", content=example.prompt),
        ]
        try:
            text = await self._ask(record, messages)
        except Exception as exc:  # noqa: BLE001 - one bad worker must not stop the round
            return Draft(worker_id=record.worker_id, text="", error=str(exc))
        return Draft(worker_id=record.worker_id, text=text)

    async def _score(
        self,
        drafts: Sequence[Draft],
        example: SyntheticExample,
        critic: WorkerRecord | None,
        *,
        token: PreemptionToken | None,
        use_critic: bool,
    ) -> list[Draft]:
        """Score every draft (with the critic when there is one)."""
        scored: list[Draft] = []
        for draft in drafts:
            if not draft.text:
                scored.append(draft)
                continue
            if critic is None or not use_critic:
                draft.score = similarity(draft.text, example.reference)
                scored.append(draft)
                continue
            if token is not None and token.preempted:
                return scored
            messages = [
                ChatMessage(role="system", content=CRITICAL_SYSTEM),
                ChatMessage(
                    role="user",
                    content=(
                        f"Reference answer:\n{example.reference}\n\n"
                        f"Candidate answer:\n{draft.text}\n\nScore (0-10):"
                    ),
                ),
            ]
            try:
                verdict = await self._ask(critic, messages, max_tokens=16, temperature=0.0)
            except Exception:  # noqa: BLE001 - fall back to the local similarity
                draft.score = similarity(draft.text, example.reference)
                scored.append(draft)
                continue
            score = parse_score(verdict)
            if score <= 0.0:
                # The critic did not give a usable verdict: fall back to the
                # local similarity instead of throwing the draft away.
                score = similarity(draft.text, example.reference)
            draft.score = score
            scored.append(draft)
        return scored


__all__ = [
    "CRITICAL_SYSTEM",
    "Draft",
    "SelfPlayEngine",
    "SelfPlayReport",
    "parse_score",
    "similarity",
]
