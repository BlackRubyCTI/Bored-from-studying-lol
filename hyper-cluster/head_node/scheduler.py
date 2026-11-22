"""Latency-aware scheduler.

The scheduler answers one question: *which worker (or which edge endpoint)
should serve this request right now, and why?*

Scoring model
-------------
For every candidate worker we estimate the two numbers the user actually
feels::

    ttft_ms   = 2 x mesh_round_trip + queue_delay + load_penalty + cold_start
    decode_ms = expected_tokens / expected_tokens_per_second x 1000

and combine them with the configured weights into a single score (lower is
better).  The estimate is continuously corrected with *observed* latencies
(EWMA) reported by the workers themselves, so the ranking adapts to a worker
that suddenly got slow because a TPU throttled or the VPN tunnel degraded.

Tiers
-----
``edge``     -- sub-second free inference APIs (Groq / Cerebras / ...).  Always
                 wins on raw latency, used for speculative drafting and as a
                 zero-downtime fallback brain.
``cluster``  -- pooled, self-hosted GPU workers.  Slower to reach, but they
                 host the large models and can be hot-swapped freely.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from core.catalog import default_catalog, find_model
from core.config import Settings
from core.logging_setup import get_logger
from core.protocol import ChatRequest, ModelCatalog, ModelSpec, RequestMode, WorkerTier
from head_node.worker_registry import WorkerRecord, WorkerRegistry

LOGGER = get_logger("head_node.scheduler")

#: Words that mark a genuinely hard prompt worth a council deliberation.
_COMPLEXITY_MARKERS = (
    "prove", "derive", "step by step", "step-by-step", "analyz", "analyse", "compare",
    "trade-off", "tradeoff", "architecture", "debug", "root cause", "optimi", "design a",
    "why does", "why is", "explain why", "critique", "evaluate", "strategy", "algorithm",
    "mathemat", "calculate", "prove that", "differences between", "implications",
)

#: Markers that mark a trivial prompt best served by the fastest edge node.
_TRIVIAL_MARKERS = ("hi", "hello", "hey", "thanks", "thank you", "ok", "okay", "yes", "no", "lol")


@dataclass(slots=True)
class SchedulerWeights:
    """Relative importance of each latency component."""

    latency: float = 1.0
    queue: float = 0.85
    load: float = 0.5
    cold_start: float = 1.0


@dataclass
class WorkerScore:
    """The scheduler's opinion about one worker."""

    worker_id: str
    provider: str = "local"
    eligible: bool = False
    reason: str = ""
    tier: WorkerTier = WorkerTier.CLUSTER
    score: float = math.inf
    ttft_ms: float = 0.0
    queue_delay_ms: float = 0.0
    load_penalty_ms: float = 0.0
    cold_start_ms: float = 0.0
    decode_ms: float = 0.0
    total_ms: float = 0.0
    expected_tps: float = 0.0
    model_loaded: str | None = None
    target_model: str | None = None
    requires_model_load: bool = False
    model_intelligence: float = 0.5
    breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "provider": self.provider,
            "eligible": self.eligible,
            "reason": self.reason,
            "score": round(self.score, 3),
            "ttft_ms": round(self.ttft_ms, 2),
            "queue_delay_ms": round(self.queue_delay_ms, 2),
            "load_penalty_ms": round(self.load_penalty_ms, 2),
            "cold_start_ms": round(self.cold_start_ms, 2),
            "decode_ms": round(self.decode_ms, 2),
            "total_ms": round(self.total_ms, 2),
            "expected_tps": round(self.expected_tps, 2),
            "model_loaded": self.model_loaded,
            "target_model": self.target_model,
            "requires_model_load": self.requires_model_load,
        }


@dataclass
class EdgeCandidate:
    """A sub-second external endpoint usable as a draft/fallback brain."""

    provider: str
    label: str = "default"
    model: str = ""
    predicted_tps: float = 0.0
    base_latency_ms: float = 0.0
    score: float = math.inf
    available: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "label": self.label,
            "model": self.model,
            "predicted_tps": round(self.predicted_tps, 2),
            "base_latency_ms": round(self.base_latency_ms, 2),
            "score": round(self.score, 3),
            "available": self.available,
            "reason": self.reason,
        }


@dataclass
class SchedulerDecision:
    """The full scheduling outcome for one request."""

    mode: RequestMode = RequestMode.AUTO
    tier: WorkerTier = WorkerTier.CLUSTER
    primary: WorkerScore | None = None
    alternatives: list[WorkerScore] = field(default_factory=list)
    edge: EdgeCandidate | None = None
    target_model: str | None = None
    requires_model_load: bool = False
    reason: str = ""
    expected_tokens: int = 0

    @property
    def has_capacity(self) -> bool:
        return self.primary is not None and self.primary.eligible

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "tier": self.tier.value,
            "primary": self.primary.to_dict() if self.primary else None,
            "alternatives": [a.to_dict() for a in self.alternatives],
            "edge": self.edge.to_dict() if self.edge else None,
            "target_model": self.target_model,
            "requires_model_load": self.requires_model_load,
            "reason": self.reason,
            "expected_tokens": self.expected_tokens,
            "has_capacity": self.has_capacity,
        }


class Scheduler:
    """Chooses the fastest capable execution target for a request."""

    def __init__(
        self,
        settings: Settings,
        registry: WorkerRegistry,
        *,
        catalog: ModelCatalog | None = None,
        weights: SchedulerWeights | None = None,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.catalog = catalog or default_catalog()
        self.weights = weights or SchedulerWeights(
            latency=settings.scheduler_latency_weight,
            queue=settings.scheduler_queue_weight,
            load=settings.scheduler_load_weight,
        )
        self._last_decisions: dict[str, SchedulerDecision] = {}

    # ------------------------------------------------------------------ #
    def set_catalog(self, catalog: ModelCatalog) -> None:
        """Swap the model catalogue (used by the VRAM auto-scaler)."""
        self.catalog = catalog

    def active_model(self) -> str | None:
        """Model the cluster currently considers active."""
        if self.catalog.active_model:
            return self.catalog.active_model
        best = self._best_cluster_model()
        return best.name if best else None

    def _best_cluster_model(self) -> ModelSpec | None:
        """Highest-intelligence model that fits in the live pool."""
        totals = self.registry.totals()
        free_vram = float(totals.get("free_vram_gb", 0.0))
        if free_vram <= 0:
            return None
        candidates = [m for m in self.catalog.models if m.vram_required_gb <= free_vram]
        if not candidates:
            return None
        candidates.sort(key=lambda m: (m.intelligence, m.params_b), reverse=True)
        return candidates[0]

    # ------------------------------------------------------------------ #
    def classify(self, request: ChatRequest) -> RequestMode:
        """Decide which reasoning mode an ``auto`` request really needs."""
        if request.mode is not RequestMode.AUTO:
            return request.mode
        if request.tools:
            return RequestMode.DEEP
        text = " ".join(m.content for m in request.messages[-3:]).lower().strip()
        if not text:
            return RequestMode.FAST
        words = re.findall(r"[\w']+", text)
        if len(words) <= 4 and any(w in _TRIVIAL_MARKERS for w in words):
            return RequestMode.FAST
        hits = sum(1 for marker in _COMPLEXITY_MARKERS if marker in text)
        complexity = hits * 2 + min(len(words) / 60.0, 6.0) + max(0, len(request.messages) - 2) * 0.75
        if complexity >= 6.0:
            return RequestMode.COUNCIL
        if complexity >= 3.0:
            return RequestMode.DEEP
        return RequestMode.FAST

    def expected_tokens(self, request: ChatRequest, mode: RequestMode) -> int:
        """How many tokens we realistically expect to decode."""
        ceiling = max(1, int(request.max_tokens))
        if mode is RequestMode.FAST:
            return min(ceiling, 384)
        if mode is RequestMode.DEEP:
            return min(ceiling, 1200)
        if mode is RequestMode.COUNCIL:
            return ceiling
        return min(ceiling, 768)

    # ------------------------------------------------------------------ #
    def _estimate_tps(self, record: WorkerRecord) -> float:
        """Best available throughput estimate for a worker."""
        if record.ewma_tokens_per_sec > 0:
            return record.ewma_tokens_per_sec
        if record.tokens_per_sec > 0:
            return record.tokens_per_sec
        params_b = record.model_params_b or 7.0
        device = (record.capabilities.device or "").lower()
        if device == "cpu" or record.capabilities.compute_tflops < 5.0:
            base = 9.0 / max(params_b, 0.5)
        else:
            base = 240.0 / max(params_b, 0.5)
        vram_factor = min(1.0, max(record.vram_total_gb, 4.0) / 24.0) ** 0.5
        return max(0.5, min(base * (0.5 + 0.5 * vram_factor), 220.0))

    def _model_for(self, request: ChatRequest, mode: RequestMode) -> ModelSpec | None:
        """Resolve the model spec this request wants."""
        if request.model:
            spec = find_model(self.catalog, request.model)
            if spec is not None:
                return spec
            return None
        if mode is RequestMode.COUNCIL:
            # A council deliberately spans several models -- no single target.
            return None
        totals = self.registry.totals()
        free_vram = float(totals.get("free_vram_gb", 0.0))
        if mode is RequestMode.DEEP or mode is RequestMode.COUNCIL:
            best = self._best_cluster_model()
            if best is not None:
                return best
        if free_vram <= 0:
            return None
        fits = [m for m in self.catalog.models if m.vram_required_gb <= free_vram]
        if not fits:
            return None
        if mode is RequestMode.FAST:
            fits.sort(key=lambda m: (m.params_b, -m.intelligence))
        else:
            fits.sort(key=lambda m: (m.intelligence, m.params_b), reverse=True)
        return fits[0]

    # ------------------------------------------------------------------ #
    def score_worker(
        self,
        record: WorkerRecord,
        request: ChatRequest,
        mode: RequestMode,
        *,
        target: ModelSpec | None,
        expected_tokens: int,
    ) -> WorkerScore:
        """Score one worker for one request."""
        score = WorkerScore(
            worker_id=record.worker_id,
            provider=record.provider,
            tier=record.tier,
            model_loaded=record.model_loaded,
            target_model=request.model or (target.name if target else None),
        )

        if not record.healthy:
            score.reason = f"unhealthy ({record.status.value})"
            return score
        if record.draining:
            score.reason = "draining"
            return score
        if not record.has_capacity:
            score.reason = "saturated"
            return score

        # --- model fit ------------------------------------------------- #
        requires_load = False
        loaded_matches = False
        if request.model:
            if record.model_loaded and record.model_loaded.lower() == request.model.lower():
                loaded_matches = True
            elif target is not None:
                if not record.supports_family(target.family):
                    score.reason = f"family '{target.family}' unsupported"
                    return score
                if record.vram_free_gb < target.vram_required_gb * 0.92:
                    score.reason = "insufficient vram for requested model"
                    return score
                requires_load = True
            else:
                score.reason = "requested model not in catalogue"
                return score
        else:
            if mode is RequestMode.COUNCIL:
                # Council members deliberate with whatever they already serve:
                # *model diversity* is the whole point of a council, so a
                # family mismatch is not disqualifying here.
                if record.model_loaded:
                    loaded_matches = True
                else:
                    requires_load = True
            elif target is not None:
                if record.model_loaded and record.model_loaded.lower() == target.name.lower():
                    loaded_matches = True
                elif not record.supports_family(target.family):
                    score.reason = f"family '{target.family}' unsupported"
                    return score
                else:
                    requires_load = True
        score.requires_model_load = requires_load

        # --- latency model --------------------------------------------- #
        round_trip = max(1.0, record.mesh_latency_ms) * 2.0
        network_overhead = 25.0
        expected_tps = self._estimate_tps(record)

        service_ms = (expected_tokens / max(expected_tps, 0.5)) * 1000.0
        concurrency = max(1, record.max_concurrency)
        queue_delay_ms = service_ms * (record.inflight / concurrency)
        load_penalty_ms = record.load_factor * 180.0

        cold_start_ms = 0.0
        if requires_load:
            if target is not None:
                cold_start_ms = self.settings.scheduler_cold_start_penalty_ms * min(
                    2.0, target.vram_required_gb / 8.0
                )
            else:
                cold_start_ms = self.settings.scheduler_cold_start_penalty_ms

        ttft_ms = round_trip + network_overhead + queue_delay_ms + load_penalty_ms + cold_start_ms
        decode_ms = (expected_tokens / max(expected_tps, 0.5)) * 1000.0
        total_ms = ttft_ms + decode_ms

        raw = (
            self.weights.latency * ttft_ms
            + self.weights.queue * queue_delay_ms
            + self.weights.load * load_penalty_ms
            + self.weights.cold_start * cold_start_ms
        )

        # Intelligence shaping: deep modes pay a little latency for a better
        # brain, fast modes prefer the smallest adequate model.
        intelligence = 0.5
        if record.model_loaded:
            spec = find_model(self.catalog, record.model_loaded)
            if spec is not None:
                intelligence = spec.intelligence
        elif target is not None:
            intelligence = target.intelligence
        score.model_intelligence = intelligence
        if mode in (RequestMode.DEEP, RequestMode.COUNCIL):
            raw *= 1.0 - 0.07 * intelligence
        elif mode is RequestMode.FAST:
            raw *= 1.0 + 0.04 * intelligence
        if loaded_matches:
            raw *= 0.92  # warm model: small but constant advantage

        score.eligible = True
        score.score = raw
        score.ttft_ms = ttft_ms
        score.queue_delay_ms = queue_delay_ms
        score.load_penalty_ms = load_penalty_ms
        score.cold_start_ms = cold_start_ms
        score.decode_ms = decode_ms
        score.total_ms = total_ms
        score.expected_tps = expected_tps
        score.reason = "eligible"
        score.breakdown = {
            "round_trip": round_trip,
            "network_overhead": network_overhead,
            "queue_delay": queue_delay_ms,
            "load_penalty": load_penalty_ms,
            "cold_start": cold_start_ms,
            "decode": decode_ms,
        }
        return score

    # ------------------------------------------------------------------ #
    def select(
        self,
        request: ChatRequest,
        *,
        mode: RequestMode | None = None,
        workers: Sequence[WorkerRecord] | None = None,
        exclude: Iterable[str] = (),
    ) -> SchedulerDecision:
        """Rank the workers for a request and pick the best one."""
        resolved_mode = mode or self.classify(request)
        candidates = list(workers) if workers is not None else self.registry.list_workers(serving_only=True)
        excluded = set(exclude)
        candidates = [c for c in candidates if c.worker_id not in excluded]

        target = self._model_for(request, resolved_mode)
        expected_tokens = self.expected_tokens(request, resolved_mode)

        scored = [
            self.score_worker(record, request, resolved_mode, target=target, expected_tokens=expected_tokens)
            for record in candidates
        ]
        eligible = [s for s in scored if s.eligible]
        eligible.sort(key=lambda s: (s.score, s.worker_id))
        ineligible = [s for s in scored if not s.eligible]

        decision = SchedulerDecision(
            mode=resolved_mode,
            tier=WorkerTier.CLUSTER,
            primary=eligible[0] if eligible else None,
            alternatives=eligible[1:],
            target_model=request.model or (target.name if target else None),
            requires_model_load=bool(eligible and eligible[0].requires_model_load),
            expected_tokens=expected_tokens,
        )
        if decision.primary is None:
            if ineligible:
                decision.reason = f"no eligible worker ({ineligible[0].reason})"
            else:
                decision.reason = "no worker with free capacity"
        else:
            decision.reason = (
                f"selected {decision.primary.worker_id} "
                f"(ttft={decision.primary.ttft_ms:.0f}ms total={decision.primary.total_ms:.0f}ms)"
            )
        return decision

    # ------------------------------------------------------------------ #
    def select_edge(
        self,
        candidates: Sequence[EdgeCandidate],
        request: ChatRequest,
        *,
        mode: RequestMode | None = None,
    ) -> EdgeCandidate | None:
        """Pick the best sub-second edge endpoint for speculative drafting."""
        resolved_mode = mode or self.classify(request)
        expected_tokens = self.expected_tokens(request, resolved_mode)
        usable = [c for c in candidates if c.available and c.predicted_tps > 0]
        if not usable:
            return None
        best: EdgeCandidate | None = None
        for candidate in usable:
            ttft = candidate.base_latency_ms + 40.0
            decode_ms = (expected_tokens / max(candidate.predicted_tps, 1.0)) * 1000.0
            candidate.score = ttft + decode_ms * 0.35  # drafts are cheap, bias to latency
            if best is None or candidate.score < best.score:
                best = candidate
        return best

    # ------------------------------------------------------------------ #
    def remember(self, request: ChatRequest, decision: SchedulerDecision) -> None:
        """Cache the last decision for a request (debugging / telemetry)."""
        self._last_decisions[request.request_id] = decision

    def last_decision(self, request_id: str) -> SchedulerDecision | None:
        return self._last_decisions.get(request_id)


__all__ = [
    "SchedulerWeights",
    "WorkerScore",
    "EdgeCandidate",
    "SchedulerDecision",
    "Scheduler",
]
