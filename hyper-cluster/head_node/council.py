"""Mixture-of-Agents "Council Mode".

A hard prompt is not answered by a single model.  The council:

1. **Fans out** -- N distinct workers (ideally with *different* models) answer
   the same question independently and in parallel.
2. **Cross-critiques** -- each member reviews the other answers and reports
   factual errors, contradictions and gaps.
3. **Synthesises** -- the strongest member merges the answers and the critiques
   into one definitive reply.

Consensus (semantic agreement between the candidates) is measured and reported
so the dashboard can show how much the council actually disagreed.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from core.config import Settings
from core.errors import WorkerUnreachableError
from core.logging_setup import get_logger
from core.protocol import ChatMessage, ChatRequest, RequestMode
from head_node.scheduler import Scheduler
from head_node.worker_client import WorkerClient
from head_node.worker_registry import WorkerRecord, WorkerRegistry

LOGGER = get_logger("head_node.council")

TokenSink = Callable[[str, str], Awaitable[None]]


@dataclass
class CouncilMember:
    """One participant of the council."""

    worker_id: str
    provider: str = "local"
    model: str | None = None
    text: str = ""
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    tokens: int = 0
    error: str | None = None
    intelligence: float = 0.5

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "provider": self.provider,
            "model": self.model,
            "chars": len(self.text),
            "ttft_ms": round(self.ttft_ms, 2),
            "total_ms": round(self.total_ms, 2),
            "tokens": self.tokens,
            "error": self.error,
            "excerpt": self.text[:280],
        }


@dataclass
class Critique:
    """One member's review of another member's answer."""

    by_worker: str
    about_worker: str
    text: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "by_worker": self.by_worker,
            "about_worker": self.about_worker,
            "text": self.text[:600],
            "error": self.error,
        }


@dataclass
class CouncilOutcome:
    """The result of a deliberation."""

    final_text: str = ""
    members: list[CouncilMember] = field(default_factory=list)
    critiques: list[Critique] = field(default_factory=list)
    synthesis_by: str | None = None
    rounds: int = 0
    total_ms: float = 0.0
    consensus: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "members": [m.to_dict() for m in self.members],
            "critiques": [c.to_dict() for c in self.critiques],
            "synthesis_by": self.synthesis_by,
            "rounds": self.rounds,
            "total_ms": round(self.total_ms, 2),
            "consensus": round(self.consensus, 3),
            "error": self.error,
        }


CRITIQUE_SYSTEM = (
    "You are a ruthless technical reviewer. You are given a question and several candidate answers "
    "produced by different expert models. For EACH candidate, report: (1) factual errors, "
    "(2) contradictions with the other candidates, (3) missing steps, (4) the single strongest idea "
    "to keep. Be terse, concrete and never praise. Do not produce a final answer."
)

SYNTHESIS_SYSTEM = (
    "You are the chair of an expert panel. You are given a question, several candidate answers and the "
    "panel's peer critiques. Produce ONE definitive, correct, complete answer that resolves every "
    "contradiction, keeps the strongest ideas and drops the errors. Never mention the panel, the "
    "candidates or the critiques; just answer the question."
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _tokens(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9']+", _normalize(text)) if len(word) > 2}


def similarity(left: str, right: str) -> float:
    """Word-set Jaccard similarity in ``0..1``."""
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 1.0 if (not a and not b) else 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


class CouncilEngine:
    """Runs multi-agent deliberations over the worker pool."""

    def __init__(
        self,
        settings: Settings,
        registry: WorkerRegistry,
        scheduler: Scheduler,
        client: WorkerClient,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.scheduler = scheduler
        self.client = client

    # ------------------------------------------------------------------ #
    def select_members(
        self, request: ChatRequest, *, size: int | None = None
    ) -> list[WorkerRecord]:
        """Pick up to ``size`` workers, preferring *distinct* models."""
        wanted = max(1, int(size or self.settings.council_size))
        decision = self.scheduler.select(request, mode=RequestMode.COUNCIL)
        ordered: list[WorkerRecord] = []
        seen_models: set[str] = set()
        candidates = [s for s in ([decision.primary] + list(decision.alternatives)) if s is not None]
        records: list[WorkerRecord] = []
        for score in candidates:
            if not score.eligible:
                continue
            record = self.registry.get(score.worker_id)
            if record is not None:
                records.append(record)
        # First pass: one worker per distinct model (diversity first).
        for record in records:
            key = (record.model_loaded or "unknown").lower()
            if key in seen_models:
                continue
            seen_models.add(key)
            ordered.append(record)
            if len(ordered) >= wanted:
                return ordered
        # Second pass: fill up with the fastest remaining workers.
        for record in records:
            if record in ordered:
                continue
            ordered.append(record)
            if len(ordered) >= wanted:
                break
        return ordered

    # ------------------------------------------------------------------ #
    async def _ask(
        self, record: WorkerRecord, request: ChatRequest
    ) -> CouncilMember:
        """Run one member's independent answer."""
        started = time.perf_counter()
        member = CouncilMember(worker_id=record.worker_id, provider=record.provider, model=record.model_loaded)
        try:
            completion = await self.client.chat(record, request)
        except WorkerUnreachableError as exc:
            member.error = str(exc)
            member.total_ms = (time.perf_counter() - started) * 1000.0
            return member
        member.text = completion.text
        member.ttft_ms = completion.ttft_ms
        member.total_ms = completion.total_ms or (time.perf_counter() - started) * 1000.0
        member.tokens = completion.tokens
        spec = self.scheduler.catalog
        for candidate in spec.models:
            if record.model_loaded and candidate.name.lower() == record.model_loaded.lower():
                member.intelligence = candidate.intelligence
                break
        return member

    async def _critique(
        self, record: WorkerRecord, request: ChatRequest, members: Sequence[CouncilMember]
    ) -> list[Critique]:
        """Ask one member to review the other candidates."""
        critiques: list[Critique] = []
        others = [m for m in members if m.worker_id != record.worker_id and m.ok]
        if not others:
            return critiques
        dossier = "\n\n".join(
            f"### CANDIDATE {index + 1} (worker {m.worker_id}, model {m.model})\n{m.text[:2600]}"
            for index, m in enumerate(others)
        )
        prompt = ChatRequest(
            request_id=f"{request.request_id}-critique-{record.worker_id}",
            messages=[
                ChatMessage(role="system", content=CRITIQUE_SYSTEM),
                ChatMessage(
                    role="user",
                    content=(
                        f"QUESTION:\n{request.to_prompt()[:4000]}\n\n"
                        f"CANDIDATE ANSWERS:\n{dossier}\n\n"
                        "Review each candidate separately."
                    ),
                ),
            ],
            mode=RequestMode.DEEP,
            max_tokens=min(700, self.settings.council_max_tokens),
            temperature=0.2,
            stream=False,
        )
        try:
            completion = await self.client.chat(record, prompt)
        except WorkerUnreachableError as exc:
            for member in others:
                critiques.append(
                    Critique(by_worker=record.worker_id, about_worker=member.worker_id, error=str(exc))
                )
            return critiques
        blob = completion.text
        # The critic answers about every candidate; split the blob per member.
        for index, member in enumerate(others):
            critiques.append(
                Critique(
                    by_worker=record.worker_id,
                    about_worker=member.worker_id,
                    text=blob if len(others) == 1 else _section_for(blob, index, len(others)),
                )
            )
        return critiques

    async def _synthesise(
        self,
        record: WorkerRecord,
        request: ChatRequest,
        members: Sequence[CouncilMember],
        critiques: Sequence[Critique],
        sink: TokenSink | None,
    ) -> tuple[str, str | None]:
        """Merge everything into one definitive answer."""
        if len(members) == 1 and not critiques:
            return members[0].text, record.worker_id
        dossier = "\n\n".join(
            f"### CANDIDATE {index + 1} (worker {m.worker_id}, model {m.model}, "
            f"intelligence {m.intelligence:.2f})\n{m.text[:4000]}"
            for index, m in enumerate(members)
            if m.ok
        )
        critique_blob = "\n\n".join(
            f"- reviewer {c.by_worker} on {c.about_worker}: {c.text[:900]}"
            for c in critiques
            if c.text and not c.error
        )
        prompt = ChatRequest(
            request_id=f"{request.request_id}-synthesis",
            messages=[
                ChatMessage(role="system", content=SYNTHESIS_SYSTEM),
                ChatMessage(
                    role="user",
                    content=(
                        f"QUESTION:\n{request.to_prompt()[:6000]}\n\n"
                        f"CANDIDATE ANSWERS:\n{dossier}\n\n"
                        f"PEER CRITIQUES:\n{critique_blob or '(none)'}\n\n"
                        "Deliver the definitive answer."
                    ),
                ),
            ],
            mode=RequestMode.DEEP,
            max_tokens=self.settings.council_max_tokens,
            temperature=0.3,
            stream=sink is not None,
        )
        if sink is None:
            try:
                completion = await self.client.chat(record, prompt)
            except WorkerUnreachableError:
                fallback = next((m.text for m in members if m.ok), "")
                return fallback, None
            return completion.text, record.worker_id

        text_parts: list[str] = []
        try:
            async for chunk in self.client.stream_chat(record, prompt):
                if chunk.text:
                    text_parts.append(chunk.text)
                    await sink(chunk.text, record.worker_id)
        except WorkerUnreachableError:
            fallback = next((m.text for m in members if m.ok), "")
            if fallback:
                await sink(fallback, record.worker_id)
            return fallback, None
        return "".join(text_parts), record.worker_id

    # ------------------------------------------------------------------ #
    async def deliberate(
        self,
        request: ChatRequest,
        *,
        members: Sequence[WorkerRecord] | None = None,
        sink: TokenSink | None = None,
        rounds: int | None = None,
    ) -> CouncilOutcome:
        """Run the full council deliberation."""
        started = time.perf_counter()
        outcome = CouncilOutcome()
        chosen = list(members) if members else self.select_members(request)
        if not chosen:
            outcome.error = "no workers available for council"
            outcome.total_ms = (time.perf_counter() - started) * 1000.0
            return outcome

        # 1. Fan out.
        results = await asyncio.gather(*(self._ask(record, request) for record in chosen), return_exceptions=True)
        members_out: list[CouncilMember] = []
        for record, result in zip(chosen, results, strict=False):
            if isinstance(result, CouncilMember):
                members_out.append(result)
            else:
                members_out.append(
                    CouncilMember(worker_id=record.worker_id, provider=record.provider, error=str(result))
                )
        outcome.members = members_out

        # 2. Cross-critique (bounded number of rounds).
        critique_rounds = max(0, int(rounds if rounds is not None else self.settings.council_critique_rounds))
        for round_index in range(critique_rounds):
            reviewers = [r for r in chosen if r.worker_id in {m.worker_id for m in members_out if m.ok}]
            if not reviewers:
                break
            gathered = await asyncio.gather(
                *(self._critique(record, request, members_out) for record in reviewers),
                return_exceptions=True,
            )
            for result in gathered:
                if isinstance(result, list):
                    outcome.critiques.extend(result)
            outcome.rounds = round_index + 1

        # 3. Synthesise with the strongest member.
        viable = [m for m in members_out if m.ok]
        if not viable:
            outcome.error = "all council members failed"
            outcome.total_ms = (time.perf_counter() - started) * 1000.0
            return outcome
        chair_record = None
        best_intelligence = -1.0
        for member in viable:
            if member.intelligence > best_intelligence:
                best_intelligence = member.intelligence
                chair_record = self.registry.get(member.worker_id)
        if chair_record is None:
            chair_record = self.registry.get(chosen[0].worker_id) or chosen[0]

        final_text, synthesis_by = await self._synthesise(chair_record, request, members_out, outcome.critiques, sink)
        outcome.final_text = final_text
        outcome.synthesis_by = synthesis_by
        outcome.consensus = _consensus(viable)
        outcome.total_ms = (time.perf_counter() - started) * 1000.0
        return outcome


def _consensus(members: Sequence[CouncilMember]) -> float:
    """Mean pairwise similarity between the viable candidates."""
    viable = [m for m in members if m.ok]
    if len(viable) < 2:
        return 1.0 if viable else 0.0
    scores: list[float] = []
    for i in range(len(viable)):
        for j in range(i + 1, len(viable)):
            scores.append(similarity(viable[i].text, viable[j].text))
    return sum(scores) / len(scores) if scores else 0.0


def _section_for(blob: str, index: int, total: int) -> str:
    """Best-effort split of a multi-candidate critique into per-candidate parts."""
    markers = [f"CANDIDATE {n + 1}" for n in range(total)]
    positions: list[int] = []
    upper = blob.upper()
    for marker in markers:
        pos = upper.find(marker)
        positions.append(pos if pos != -1 else -1)
    if all(p == -1 for p in positions) or positions[index] == -1:
        return blob
    start = positions[index]
    end = len(blob)
    for pos in positions[index + 1 :]:
        if pos > start:
            end = pos
            break
    return blob[start:end].strip()


__all__ = [
    "CouncilMember",
    "Critique",
    "CouncilOutcome",
    "CouncilEngine",
    "similarity",
]
