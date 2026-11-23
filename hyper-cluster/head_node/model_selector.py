"""Live VRAM pooling and automatic model up/down-scaling.

The cluster is a *pool* of GPUs whose size changes every minute: a Kaggle
kernel expires, a new 24 GB node registers, a user asks for a 70B answer, the
edge tier starts serving alone.  The selector turns that moving target into
three concrete answers:

1. **How much VRAM do we actually have?**  :meth:`ModelSelector.pool` sums the
   *measured* VRAM of every live worker and distinguishes the VRAM that is
   free right now (headroom) from the VRAM that a resident model occupies
   (reserved).  A worker can hot-swap its model, so the capacity that matters
   for "can we serve a 70B?" is the capacity of the *biggest single* worker,
   not the sum of the pool.

2. **Which model should the cluster serve next?**  :meth:`ModelSelector.plan`
   compares the demand profile (fast / deep / council traffic) with the pool
   and produces a list of :class:`ScalingDecision` objects -- an *upshift*
   (hot-swap a bigger, smarter model onto the worker that can hold it) or a
   *downshift* (hot-swap a smaller model to release VRAM).  Hysteresis
   (minimum dwell time, minimum intelligence gain, at most N changes per
   tick) makes the decisions stable: the cluster never oscillates between two
   models.

3. **Who drafts and who verifies?**  :meth:`ModelSelector.draft_pair` pairs a
   small, fast resident model with a large one for *cluster-internal*
   speculative decoding: the draft produces the answer in milliseconds, the
   verifier confirms it with the big model.  The pair is only offered when the
   draft is genuinely faster and at least 4x smaller, because speculation that
   is slower than the verifier is pure overhead.

The module is deliberately *pure*: it reads the registry and the catalogue and
returns decisions.  Executing them (hot-swapping a model over HTTP) is the
orchestrator's job, which keeps the decision logic trivially testable and
free of I/O.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from core.catalog import DRAFT_CANDIDATES, ModelCatalog, default_catalog, find_model
from core.config import Settings
from core.logging_setup import get_logger
from core.protocol import ModelSpec, RequestMode
from head_node.worker_registry import WorkerRecord, WorkerRegistry

LOGGER = get_logger("head_node.model_selector")

#: Actions a scaling decision can ask for.
ACTION_HOLD = "hold"
ACTION_LOAD = "load"
ACTION_UNLOAD = "unload"

#: Demand levels the selector reacts to.
DEMAND_FAST = "fast"
DEMAND_DEEP = "deep"
DEMAND_COUNCIL = "council"


@dataclass
class WorkerVRAM:
    """The VRAM situation of one live worker."""

    worker_id: str
    provider: str = "local"
    region: str = "unknown"
    total_gb: float = 0.0
    free_gb: float = 0.0
    reserved_gb: float = 0.0
    capacity_gb: float = 0.0
    loaded_model: str | None = None
    loaded_family: str | None = None
    loaded_params_b: float = 0.0
    tokens_per_sec: float = 0.0
    tflops: float = 0.0
    healthy: bool = False
    #: ``tokens/s * B`` -- the hardware factor of the node.
    node_tps_per_b: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "provider": self.provider,
            "region": self.region,
            "total_gb": round(self.total_gb, 3),
            "free_gb": round(self.free_gb, 3),
            "reserved_gb": round(self.reserved_gb, 3),
            "capacity_gb": round(self.capacity_gb, 3),
            "loaded_model": self.loaded_model,
            "loaded_family": self.loaded_family,
            "loaded_params_b": round(self.loaded_params_b, 3),
            "tokens_per_sec": round(self.tokens_per_sec, 3),
            "tflops": round(self.tflops, 3),
            "healthy": self.healthy,
            "node_tps_per_b": round(self.node_tps_per_b, 3),
        }


@dataclass
class VRAMPool:
    """The pooled VRAM of the whole cluster."""

    workers: list[WorkerVRAM] = field(default_factory=list)
    total_gb: float = 0.0
    free_gb: float = 0.0
    reserved_gb: float = 0.0
    capacity_gb: float = 0.0
    max_worker_capacity_gb: float = 0.0
    by_provider: dict[str, float] = field(default_factory=dict)
    by_family: dict[str, float] = field(default_factory=dict)
    healthy_workers: int = 0
    observed_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_gb": round(self.total_gb, 3),
            "free_gb": round(self.free_gb, 3),
            "reserved_gb": round(self.reserved_gb, 3),
            "capacity_gb": round(self.capacity_gb, 3),
            "max_worker_capacity_gb": round(self.max_worker_capacity_gb, 3),
            "by_provider": {k: round(v, 3) for k, v in self.by_provider.items()},
            "by_family": {k: round(v, 3) for k, v in self.by_family.items()},
            "healthy_workers": self.healthy_workers,
            "workers": [w.to_dict() for w in self.workers],
            "observed_at": self.observed_at,
        }


@dataclass
class DemandProfile:
    """The recent traffic mix, used to decide how big the model should be."""

    fast: int = 0
    deep: int = 0
    council: int = 0

    @property
    def total(self) -> int:
        return self.fast + self.deep + self.council

    @property
    def level(self) -> str:
        """The dominant demand level."""
        if self.total <= 0:
            return DEMAND_FAST
        if self.council >= max(self.fast, self.deep) and self.council > 0:
            return DEMAND_COUNCIL
        if self.deep >= self.fast:
            return DEMAND_DEEP
        return DEMAND_FAST

    @property
    def intensity(self) -> float:
        """0..1 -- how much brain the current traffic demands."""
        if self.total <= 0:
            return 0.0
        weighted = self.fast * 1 + self.deep * 3 + self.council * 5
        return min(1.0, weighted / (self.total * 5.0))

    def record(self, mode: RequestMode | str) -> None:
        """Record one served request."""
        name = mode.value if isinstance(mode, RequestMode) else str(mode)
        if name == DEMAND_COUNCIL:
            self.council += 1
        elif name == DEMAND_DEEP:
            self.deep += 1
        else:
            self.fast += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "fast": self.fast,
            "deep": self.deep,
            "council": self.council,
            "total": self.total,
            "level": self.level,
            "intensity": round(self.intensity, 3),
        }


@dataclass
class ScalingDecision:
    """One instruction for the cluster: hot-swap a model, or do nothing."""

    action: str = ACTION_HOLD
    worker_id: str = ""
    model: str | None = None
    previous_model: str | None = None
    reason: str = ""
    required_vram_gb: float = 0.0
    freed_vram_gb: float = 0.0
    intelligence_before: float = 0.0
    intelligence_after: float = 0.0
    direction: str = "none"

    @property
    def changes_model(self) -> bool:
        return self.action in (ACTION_LOAD, ACTION_UNLOAD)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "worker_id": self.worker_id,
            "model": self.model,
            "previous_model": self.previous_model,
            "reason": self.reason,
            "required_vram_gb": round(self.required_vram_gb, 3),
            "freed_vram_gb": round(self.freed_vram_gb, 3),
            "intelligence_before": round(self.intelligence_before, 3),
            "intelligence_after": round(self.intelligence_after, 3),
            "direction": self.direction,
        }


@dataclass
class SpeculativePair:
    """A draft model paired with the verifier that checks its output."""

    draft_model: str = ""
    verifier_model: str = ""
    draft_worker: str = ""
    verifier_worker: str = ""
    draft_params_b: float = 0.0
    verifier_params_b: float = 0.0
    draft_tokens_per_sec: float = 0.0
    verifier_tokens_per_sec: float = 0.0
    estimated_speedup: float = 1.0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "draft_model": self.draft_model,
            "verifier_model": self.verifier_model,
            "draft_worker": self.draft_worker,
            "verifier_worker": self.verifier_worker,
            "draft_params_b": round(self.draft_params_b, 3),
            "verifier_params_b": round(self.verifier_params_b, 3),
            "draft_tokens_per_sec": round(self.draft_tokens_per_sec, 3),
            "verifier_tokens_per_sec": round(self.verifier_tokens_per_sec, 3),
            "estimated_speedup": round(self.estimated_speedup, 3),
            "reason": self.reason,
        }


class ModelSelector:
    """Decides which model the live pool should serve."""

    def __init__(
        self,
        settings: Settings | None = None,
        registry: WorkerRegistry | None = None,
        catalog: ModelCatalog | None = None,
    ) -> None:
        self.settings = settings if settings is not None else Settings.from_env()
        self.registry = registry if registry is not None else WorkerRegistry(
            ttl_seconds=self.settings.worker_ttl_seconds,
            max_concurrency_per_worker=self.settings.max_concurrency_per_worker,
        )
        self.catalog = catalog if catalog is not None else default_catalog()
        self._last_change_at: dict[str, float] = {}
        self._demand = DemandProfile()

    # ------------------------------------------------------------------ #
    # Demand tracking
    # ------------------------------------------------------------------ #
    def record_demand(self, mode: RequestMode | str) -> None:
        """Record a served request so the selector knows what the cluster needs."""
        self._demand.record(mode)

    @property
    def demand(self) -> DemandProfile:
        """The rolling demand profile."""
        return self._demand

    def reset_demand(self) -> None:
        """Forget the demand history (used by tests and after a rebalance)."""
        self._demand = DemandProfile()

    # ------------------------------------------------------------------ #
    # VRAM pooling
    # ------------------------------------------------------------------ #
    def _resident_vram(self, record: WorkerRecord) -> float:
        """How much VRAM the resident model occupies (catalogue estimate)."""
        spec = find_model(self.catalog, record.model_loaded)
        if spec is not None:
            return spec.vram_required_gb
        params = record.model_params_b or 0.0
        if params <= 0:
            return 0.0
        # Fallback heuristic for models that are not in the catalogue.
        return round(params * 0.78, 3)

    def _describe(self, record: WorkerRecord) -> WorkerVRAM:
        reserved = self._resident_vram(record)
        free = max(0.0, record.vram_free_gb)
        capacity = free + reserved
        # A worker that never reported its VRAM is treated as a CPU node.
        total = max(record.vram_total_gb, capacity)
        spec = find_model(self.catalog, record.model_loaded)
        params = record.model_params_b or (spec.params_b if spec else 0.0)
        family = record.model_family or (spec.family if spec else None)
        return WorkerVRAM(
            worker_id=record.worker_id,
            provider=record.provider,
            region=record.region,
            total_gb=round(total, 3),
            free_gb=round(free, 3),
            reserved_gb=round(reserved, 3),
            capacity_gb=round(capacity, 3),
            loaded_model=record.model_loaded,
            loaded_family=family,
            loaded_params_b=params,
            tokens_per_sec=round(record.ewma_tokens_per_sec or record.tokens_per_sec, 3),
            tflops=round(record.compute_tflops, 3),
            healthy=record.healthy,
            node_tps_per_b=round(record.node_tps_per_b, 3),
        )

    def pool(self) -> VRAMPool:
        """The live, measured VRAM pool of the cluster."""
        records = self.registry.list_workers()
        workers = [self._describe(record) for record in records]
        healthy = [w for w in workers if w.healthy]
        pool = VRAMPool(
            workers=workers,
            total_gb=round(sum(w.total_gb for w in workers), 3),
            free_gb=round(sum(w.free_gb for w in healthy), 3),
            reserved_gb=round(sum(w.reserved_gb for w in healthy), 3),
            capacity_gb=round(sum(w.capacity_gb for w in healthy), 3),
            max_worker_capacity_gb=round(max((w.capacity_gb for w in healthy), default=0.0), 3),
            healthy_workers=len(healthy),
        )
        for worker in healthy:
            pool.by_provider[worker.provider] = round(
                pool.by_provider.get(worker.provider, 0.0) + worker.capacity_gb, 3
            )
            family = worker.loaded_family or "unknown"
            pool.by_family[family] = round(pool.by_family.get(family, 0.0) + worker.capacity_gb, 3)
        return pool

    # ------------------------------------------------------------------ #
    # Target selection
    # ------------------------------------------------------------------ #
    def fits(self, spec: ModelSpec, worker: WorkerVRAM) -> bool:
        """True when ``spec`` fits on ``worker`` with the configured safety margin."""
        if worker.capacity_gb <= 0:
            return False
        return spec.vram_required_gb * self.settings.autoscaler_safety_factor <= worker.capacity_gb

    def candidates(self, pool: VRAMPool, *, demand: str = DEMAND_DEEP) -> list[ModelSpec]:
        """Catalogue models that the live pool could host, best first."""
        healthy = [w for w in pool.workers if w.healthy]
        if not healthy:
            return []
        fits: list[ModelSpec] = []
        for spec in self.catalog.models:
            for worker in healthy:
                if not self.fits(spec, worker):
                    continue
                if worker.loaded_family and spec.family != worker.loaded_family:
                    # A worker can only serve models of a family it supports.
                    if worker.loaded_family not in ("", "unknown"):
                        continue
                fits.append(spec)
                break
        if demand == DEMAND_FAST:
            fits.sort(key=lambda m: (m.params_b, -m.intelligence))
        elif demand == DEMAND_COUNCIL:
            fits.sort(key=lambda m: (m.intelligence, m.params_b), reverse=True)
        else:
            fits.sort(key=lambda m: (m.intelligence, m.params_b), reverse=True)
        return fits

    def target_model(self, pool: VRAMPool | None = None, *, demand: str | None = None) -> ModelSpec | None:
        """The model the cluster should serve for the current demand."""
        resolved_pool = pool if pool is not None else self.pool()
        level = demand or self._demand.level
        candidates = self.candidates(resolved_pool, demand=level)
        if not candidates:
            return None
        if level == DEMAND_FAST:
            # Fast traffic still deserves a competent (but small) model.
            for spec in candidates:
                if spec.intelligence >= 0.25:
                    return spec
            return candidates[0]
        return candidates[0]

    # ------------------------------------------------------------------ #
    # Scaling plan
    # ------------------------------------------------------------------ #
    def _dwell_ok(self, worker_id: str) -> bool:
        last = self._last_change_at.get(worker_id)
        if last is None:
            return True
        return (time.time() - last) >= self.settings.autoscaler_min_dwell_seconds

    def mark_applied(self, worker_id: str) -> None:
        """Record that a worker was just re-balanced (starts the dwell timer)."""
        self._last_change_at[worker_id] = time.time()

    def plan(self, demand: DemandProfile | None = None) -> list[ScalingDecision]:
        """Decide what the pool should do about its models.

        The rule is simple and demand-driven: the pool converges on the
        *target* model of the current demand level.  A worker that hosts a
        weaker model is upshifted, a worker that hosts a stronger one is
        downshifted, and a worker that already hosts the target is left
        alone.  At most ``autoscaler_max_changes_per_tick`` changes are
        planned per tick so the cluster never thrashes.
        """
        profile = demand if demand is not None else self._demand
        pool = self.pool()
        decisions: list[ScalingDecision] = []
        if pool.healthy_workers == 0:
            return [
                ScalingDecision(
                    action=ACTION_HOLD,
                    reason="no healthy worker in the pool",
                )
            ]

        level = profile.level
        target = self.target_model(pool, demand=level)
        budget = max(1, int(self.settings.autoscaler_max_changes_per_tick))
        applied = 0
        min_gain = self.settings.autoscaler_min_gain

        # --- downshift: the resident model is heavier than the demand ---- #
        # Fast traffic never needs a frontier model, so releasing the big
        # weights comes first -- the freed VRAM is what lets the next deep
        # request upshift somewhere else.
        if applied < budget and target is not None:
            for worker in sorted(
                (w for w in pool.workers if w.healthy and w.loaded_model),
                key=lambda w: w.reserved_gb,
                reverse=True,
            ):
                if applied >= budget:
                    break
                if not self._dwell_ok(worker.worker_id):
                    continue
                resident = find_model(self.catalog, worker.loaded_model)
                if resident is None:
                    continue
                if resident.intelligence <= target.intelligence + min_gain:
                    # The resident model is exactly what the demand asks for.
                    continue
                smaller = self._smaller_fit(pool, worker, resident, preferred=target)
                if smaller is None:
                    continue
                decisions.append(
                    ScalingDecision(
                        action=ACTION_LOAD,
                        worker_id=worker.worker_id,
                        model=smaller.name,
                        previous_model=resident.name,
                        reason=(
                            f"downshift to {smaller.name} "
                            f"(frees {max(0.0, resident.vram_required_gb - smaller.vram_required_gb):.1f}GB, "
                            f"demand level '{level}' wants {target.name})"
                        ),
                        required_vram_gb=smaller.vram_required_gb,
                        freed_vram_gb=max(0.0, resident.vram_required_gb - smaller.vram_required_gb),
                        intelligence_before=resident.intelligence,
                        intelligence_after=smaller.intelligence,
                        direction="down",
                    )
                )
                applied += 1
                break

        # --- upshift: put the target model on the worker that can host it --
        if applied < budget and target is not None:
            ranked = sorted(
                (w for w in pool.workers if w.healthy),
                key=lambda w: (w.capacity_gb, w.tokens_per_sec),
                reverse=True,
            )
            for worker in ranked:
                if applied >= budget:
                    break
                if not self._dwell_ok(worker.worker_id):
                    continue
                if not self.fits(target, worker):
                    # This node cannot host the target model.
                    continue
                resident = find_model(self.catalog, worker.loaded_model)
                resident_intelligence = resident.intelligence if resident else 0.0
                if resident_intelligence >= target.intelligence - min_gain:
                    # The worker already serves a model of the right size.
                    continue
                decisions.append(
                    ScalingDecision(
                        action=ACTION_LOAD,
                        worker_id=worker.worker_id,
                        model=target.name,
                        previous_model=worker.loaded_model,
                        reason=(
                            f"upshift to {target.name} "
                            f"(intelligence {resident_intelligence:.2f} -> {target.intelligence:.2f}, "
                            f"{target.vram_required_gb:.1f}GB fits in {worker.capacity_gb:.1f}GB)"
                        ),
                        required_vram_gb=target.vram_required_gb,
                        freed_vram_gb=0.0,
                        intelligence_before=resident_intelligence,
                        intelligence_after=target.intelligence,
                        direction="up",
                    )
                )
                applied += 1
                break

        if not decisions:
            decisions.append(
                ScalingDecision(
                    action=ACTION_HOLD,
                    reason=(
                        f"the pool already matches the demand level '{level}'"
                        + (f" (target {target.name})" if target else "")
                    ),
                )
            )
        return decisions

    def _smaller_fit(
        self,
        pool: VRAMPool,
        worker: WorkerVRAM,
        resident: ModelSpec,
        *,
        preferred: ModelSpec | None = None,
    ) -> ModelSpec | None:
        """The best model smaller than ``resident`` that fits on ``worker``.

        ``preferred`` is the model the current demand asks for; when it fits,
        it wins over any other candidate -- the pool converges on the target
        instead of on the largest model that happens to fit.
        """
        options = [
            spec
            for spec in self.catalog.models
            if spec.params_b < resident.params_b and self.fits(spec, worker)
        ]
        if not options:
            return None
        if preferred is not None and any(o.name == preferred.name for o in options):
            return preferred
        options.sort(key=lambda m: (m.intelligence, m.params_b), reverse=True)
        return options[0]

    # ------------------------------------------------------------------ #
    # Speculative draft / verifier pairing
    # ------------------------------------------------------------------ #
    def _expected_tps(self, worker: WorkerVRAM) -> float:
        """Estimate how fast a worker runs its resident model.

        The benchmark of the node gives us ``tokens/s * B`` -- a property of
        the hardware.  Re-applying it to the model the node hosts *now* keeps
        the estimate correct after a hot-swap, when the last measurement is
        stale.
        """
        params = max(worker.loaded_params_b, 0.5)
        if worker.node_tps_per_b > 0:
            return worker.node_tps_per_b / params
        if worker.tokens_per_sec > 0:
            return worker.tokens_per_sec
        if worker.tflops >= 20:
            return 240.0 / params
        return 12.0 / params

    def draft_pair(self, *, pool: VRAMPool | None = None) -> SpeculativePair | None:
        """Pair a fast small model with a large verifier for speculation."""
        resolved_pool = pool if pool is not None else self.pool()
        healthy = [w for w in resolved_pool.workers if w.healthy]
        if len(healthy) < 1:
            return None

        def _params(worker: WorkerVRAM) -> float:
            if worker.loaded_params_b > 0:
                return worker.loaded_params_b
            spec = find_model(self.catalog, worker.loaded_model)
            return spec.params_b if spec else 0.0

        candidates = [w for w in healthy if _params(w) > 0]
        if not candidates:
            return None

        # The verifier is the brainiest resident model.
        verifier = max(candidates, key=lambda w: (_params(w), self._expected_tps(w)))
        verifier_params = _params(verifier)

        # The draft must be a resident model that is much smaller *and* faster.
        drafts = [
            w
            for w in candidates
            if w.worker_id != verifier.worker_id
            and _params(w) * 4.0 <= verifier_params
            and self._expected_tps(w) > self._expected_tps(verifier)
        ]
        if not drafts:
            # Same worker hosting both (the draft is tiny) is still valid.
            drafts = [
                w
                for w in candidates
                if w.worker_id == verifier.worker_id
                and _params(w) * 4.0 <= verifier_params
            ]
        if not drafts:
            return None

        draft = max(drafts, key=lambda w: self._expected_tps(w))
        draft_params = _params(draft)
        if draft_params <= 0 or verifier_params <= 0:
            return None

        draft_tps = self._expected_tps(draft)
        verifier_tps = self._expected_tps(verifier)
        if draft_tps <= 0 or verifier_tps <= 0:
            return None
        # Classic speculative decoding cost model, in seconds per token:
        #   E[t] = (1 - a) * t_target + a * (t_draft + e * t_target)
        # with a = acceptance rate and e = the cost of verifying one drafted
        # token.  The speed-up is t_target / E[t].
        acceptance = 0.7
        verify_overhead = 0.1
        target_cost = 1.0 / verifier_tps
        draft_cost = 1.0 / draft_tps
        expected = (1.0 - acceptance) * target_cost + acceptance * (
            draft_cost + verify_overhead * target_cost
        )
        speedup = target_cost / max(expected, 1e-9)
        return SpeculativePair(
            draft_model=draft.loaded_model or "",
            verifier_model=verifier.loaded_model or "",
            draft_worker=draft.worker_id,
            verifier_worker=verifier.worker_id,
            draft_params_b=draft_params,
            verifier_params_b=verifier_params,
            draft_tokens_per_sec=round(draft_tps, 2),
            verifier_tokens_per_sec=round(verifier_tps, 2),
            estimated_speedup=round(min(speedup, 2.5), 3),
            reason=(
                f"draft {draft.loaded_model} ({draft_params:.1f}B) verifies "
                f"{verifier.loaded_model} ({verifier_params:.1f}B)"
            ),
        )

    def draft_model_for(self, worker: WorkerVRAM) -> ModelSpec | None:
        """The catalogue's preferred draft model for a worker."""
        for name in DRAFT_CANDIDATES:
            spec = find_model(self.catalog, name)
            if spec is not None and self.fits(spec, worker):
                return spec
        return None

    # ------------------------------------------------------------------ #
    def status(self) -> dict[str, Any]:
        """A JSON-safe view of the selector state (for the dashboard)."""
        pool = self.pool()
        return {
            "pool": pool.to_dict(),
            "demand": self._demand.to_dict(),
            "target_model": (self.target_model(pool) or ModelSpec(name="")).name or None,
            "draft_pair": (self.draft_pair(pool=pool) or SpeculativePair()).to_dict()
            if pool.healthy_workers
            else None,
        }


__all__ = [
    "ACTION_HOLD",
    "ACTION_LOAD",
    "ACTION_UNLOAD",
    "DEMAND_FAST",
    "DEMAND_DEEP",
    "DEMAND_COUNCIL",
    "DemandProfile",
    "ModelSelector",
    "ScalingDecision",
    "SpeculativePair",
    "VRAMPool",
    "WorkerVRAM",
]
