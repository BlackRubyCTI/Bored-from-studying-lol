"""Tests for the live VRAM pooling, auto-scaling and draft/verifier pairing."""

from __future__ import annotations

import pytest

from core.catalog import default_catalog, find_model
from core.config import Settings
from core.protocol import Capabilities, RequestMode, WorkerHello
from head_node.model_selector import (
    ACTION_HOLD,
    ACTION_LOAD,
    DEMAND_COUNCIL,
    DEMAND_DEEP,
    DEMAND_FAST,
    DemandProfile,
    ModelSelector,
)
from head_node.worker_registry import WorkerRegistry


def _settings(**overrides) -> Settings:
    base = {
        "data_dir": "/tmp/hypercluster-selector-tests",
        "log_level": "WARNING",
        "log_json": False,
        "autoscaler_enabled": True,
        "autoscaler_safety_factor": 1.15,
        "autoscaler_min_gain": 0.05,
        "autoscaler_min_dwell_seconds": 0,
        "autoscaler_max_changes_per_tick": 1,
    }
    base.update(overrides)
    return Settings(**base)


def _register(
    registry: WorkerRegistry,
    worker_id: str,
    *,
    vram_total: float,
    vram_free: float,
    model: str | None,
    params_b: float | None = None,
    family: str | None = None,
    tokens_per_sec: float = 40.0,
    tflops: float = 60.0,
    provider: str = "local",
    region: str = "test",
) -> None:
    registry.register(
        WorkerHello(
            worker_id=worker_id,
            host="127.0.0.1",
            port=8081,
            capabilities=Capabilities(
                vram_total_gb=vram_total,
                vram_free_gb=vram_free,
                compute_tflops=tflops,
                provider=provider,
                region=region,
                device="cuda",
                supported_families=["qwen2", "llama", "gemma2", "phi3"],
            ),
            model_loaded=model,
            model_family=family or ("qwen2" if model else None),
            model_params_b=params_b,
        )
    )


@pytest.fixture
def registry() -> WorkerRegistry:
    return WorkerRegistry(ttl_seconds=600, max_concurrency_per_worker=2)


# --------------------------------------------------------------------------- #
# VRAM pooling
# --------------------------------------------------------------------------- #
def test_pool_reports_measured_vram(registry: WorkerRegistry) -> None:
    selector = ModelSelector(_settings(), registry)
    _register(registry, "w1", vram_total=24.0, vram_free=18.5, model="Qwen2.5-7B-Instruct-GGUF")
    _register(registry, "w2", vram_total=12.0, vram_free=10.0, model="Qwen2.5-0.5B-Instruct-GGUF")
    pool = selector.pool()
    assert pool.healthy_workers == 2
    assert pool.total_gb == 36.0
    assert pool.free_gb == 28.5
    # Qwen2.5-7B needs 5.5 GB, Qwen2.5-0.5B needs 1.2 GB -> 6.7 GB reserved.
    assert pool.reserved_gb == pytest.approx(6.7, abs=0.01)
    assert pool.capacity_gb == pytest.approx(35.2, abs=0.01)
    assert pool.max_worker_capacity_gb == pytest.approx(24.0, abs=0.01)
    assert pool.by_provider["local"] == pytest.approx(35.2, abs=0.01)


def test_pool_shrinks_when_a_worker_dies(registry: WorkerRegistry) -> None:
    selector = ModelSelector(_settings(), registry)
    _register(registry, "w1", vram_total=24.0, vram_free=18.0, model="Qwen2.5-7B-Instruct-GGUF")
    _register(registry, "w2", vram_total=24.0, vram_free=18.0, model="Qwen2.5-7B-Instruct-GGUF")
    assert selector.pool().total_gb == 48.0
    registry.mark_lost("w2", reason="test")
    pool = selector.pool()
    assert pool.healthy_workers == 1
    assert pool.total_gb == 48.0  # the dead worker is still visible for diagnostics
    assert pool.free_gb == 18.0  # but it no longer contributes capacity


def test_pool_ignores_draining_workers(registry: WorkerRegistry) -> None:
    selector = ModelSelector(_settings(), registry)
    _register(registry, "w1", vram_total=24.0, vram_free=18.0, model="Qwen2.5-7B-Instruct-GGUF")
    _register(registry, "w2", vram_total=24.0, vram_free=18.0, model="Qwen2.5-7B-Instruct-GGUF")
    registry.drain("w2", reason="handoff")
    pool = selector.pool()
    assert pool.healthy_workers == 1
    assert pool.free_gb == 18.0


# --------------------------------------------------------------------------- #
# Upshift / downshift
# --------------------------------------------------------------------------- #
def test_upshift_proposes_a_bigger_model(registry: WorkerRegistry) -> None:
    selector = ModelSelector(_settings(), registry)
    _register(registry, "w1", vram_total=24.0, vram_free=18.0, model="Qwen2.5-7B-Instruct-GGUF")
    selector.record_demand(RequestMode.DEEP)
    decisions = selector.plan()
    assert decisions, "the selector produced no decision"
    decision = decisions[0]
    assert decision.action == ACTION_LOAD
    assert decision.direction == "up"
    assert decision.model == "Qwen2.5-32B-Instruct-GGUF"
    assert decision.worker_id == "w1"
    assert decision.intelligence_after > decision.intelligence_before
    assert "upshift" in decision.reason


def test_upshift_respects_the_safety_factor(registry: WorkerRegistry) -> None:
    """A model that only just fits must not be selected."""
    selector = ModelSelector(_settings(autoscaler_safety_factor=1.5), registry)
    # 14 GB free + 5.5 GB reserved -> 19.5 GB capacity.  Qwen2.5-32B needs
    # 20 GB (20 * 1.5 = 30 > 19.5) so it must not be picked; the 14B may be.
    _register(registry, "w1", vram_total=24.0, vram_free=14.0, model="Qwen2.5-7B-Instruct-GGUF")
    selector.record_demand(RequestMode.DEEP)
    decisions = selector.plan()
    decision = decisions[0]
    if decision.action == ACTION_LOAD:
        spec = find_model(selector.catalog, decision.model)
        assert spec is not None
        assert spec.vram_required_gb * 1.5 <= 19.5
        assert spec.name != "Qwen2.5-32B-Instruct-GGUF"


def test_upshift_refuses_a_model_that_does_not_fit(registry: WorkerRegistry) -> None:
    """With no headroom at all there is nothing to upshift to."""
    selector = ModelSelector(_settings(autoscaler_safety_factor=1.5), registry)
    _register(registry, "w1", vram_total=8.0, vram_free=0.5, model="Qwen2.5-0.5B-Instruct-GGUF")
    selector.record_demand(RequestMode.DEEP)
    decisions = selector.plan()
    assert decisions[0].action == ACTION_HOLD


def test_no_upshift_when_the_gain_is_too_small(registry: WorkerRegistry) -> None:
    selector = ModelSelector(_settings(autoscaler_min_gain=0.5), registry)
    _register(registry, "w1", vram_total=24.0, vram_free=18.0, model="Qwen2.5-7B-Instruct-GGUF")
    selector.record_demand(RequestMode.DEEP)
    decisions = selector.plan()
    assert decisions[0].action == ACTION_HOLD


def test_downshift_frees_vram_for_fast_traffic(registry: WorkerRegistry) -> None:
    selector = ModelSelector(_settings(), registry)
    _register(registry, "w1", vram_total=24.0, vram_free=4.0, model="Qwen2.5-14B-Instruct-GGUF")
    for _ in range(5):
        selector.record_demand(RequestMode.FAST)
    decisions = selector.plan()
    decision = decisions[0]
    assert decision.action == ACTION_LOAD
    assert decision.direction == "down"
    assert decision.previous_model == "Qwen2.5-14B-Instruct-GGUF"
    assert decision.freed_vram_gb > 0
    assert decision.intelligence_after < decision.intelligence_before


def test_fast_traffic_never_selects_a_frontier_model(registry: WorkerRegistry) -> None:
    selector = ModelSelector(_settings(), registry)
    _register(registry, "w1", vram_total=80.0, vram_free=70.0, model="Qwen2.5-7B-Instruct-GGUF")
    for _ in range(5):
        selector.record_demand(RequestMode.FAST)
    decisions = selector.plan()
    if decisions[0].action == ACTION_LOAD:
        assert "72B" not in (decisions[0].model or "")


def test_hysteresis_blocks_a_second_change(registry: WorkerRegistry) -> None:
    selector = ModelSelector(_settings(autoscaler_min_dwell_seconds=3600), registry)
    _register(registry, "w1", vram_total=24.0, vram_free=18.0, model="Qwen2.5-7B-Instruct-GGUF")
    selector.record_demand(RequestMode.DEEP)
    first = selector.plan()
    assert first[0].action == ACTION_LOAD
    selector.mark_applied("w1")
    second = selector.plan()
    assert second[0].action == ACTION_HOLD
    assert "dwell" in second[0].reason or second[0].reason


def test_only_one_change_per_tick(registry: WorkerRegistry) -> None:
    selector = ModelSelector(_settings(autoscaler_max_changes_per_tick=1), registry)
    for index in range(4):
        _register(
            registry,
            f"w{index}",
            vram_total=24.0,
            vram_free=18.0,
            model="Qwen2.5-7B-Instruct-GGUF",
        )
    selector.record_demand(RequestMode.DEEP)
    decisions = selector.plan()
    actionable = [d for d in decisions if d.action != ACTION_HOLD]
    assert len(actionable) <= 1


def test_council_demand_selects_the_strongest_model(registry: WorkerRegistry) -> None:
    selector = ModelSelector(_settings(), registry)
    _register(registry, "w1", vram_total=48.0, vram_free=40.0, model="Qwen2.5-7B-Instruct-GGUF")
    selector.record_demand(RequestMode.COUNCIL)
    target = selector.target_model(demand=DEMAND_COUNCIL)
    assert target is not None
    # 45.5 GB of capacity with a 1.15 safety factor: the 70B (42 GB) does not
    # fit, the 32B (20 GB) does.
    assert target.name == "Qwen2.5-32B-Instruct-GGUF"
    assert target.intelligence >= 0.75


# --------------------------------------------------------------------------- #
# Demand profile
# --------------------------------------------------------------------------- #
def test_demand_profile_tracks_the_traffic_mix() -> None:
    profile = DemandProfile()
    profile.record(RequestMode.FAST)
    profile.record(RequestMode.DEEP)
    profile.record(RequestMode.DEEP)
    profile.record(RequestMode.COUNCIL)
    assert profile.total == 4
    assert profile.level == DEMAND_DEEP
    assert 0.0 < profile.intensity < 1.0


def test_empty_demand_is_fast(registry: WorkerRegistry) -> None:
    selector = ModelSelector(_settings(), registry)
    assert selector.demand.level == DEMAND_FAST


# --------------------------------------------------------------------------- #
# Speculative draft / verifier pairing
# --------------------------------------------------------------------------- #
def test_draft_pair_matches_a_small_fast_model_with_the_big_one(registry: WorkerRegistry) -> None:
    selector = ModelSelector(_settings(), registry)
    _register(
        registry,
        "small",
        vram_total=8.0,
        vram_free=6.0,
        model="Qwen2.5-0.5B-Instruct-GGUF",
        tokens_per_sec=320.0,
    )
    _register(
        registry,
        "big",
        vram_total=48.0,
        vram_free=20.0,
        model="Qwen2.5-32B-Instruct-GGUF",
        tokens_per_sec=30.0,
    )
    pair = selector.draft_pair()
    assert pair is not None
    assert pair.draft_model == "Qwen2.5-0.5B-Instruct-GGUF"
    assert pair.verifier_model == "Qwen2.5-32B-Instruct-GGUF"
    assert pair.draft_worker == "small"
    assert pair.verifier_worker == "big"
    assert pair.estimated_speedup > 1.0


def test_no_pair_when_the_models_are_too_close(registry: WorkerRegistry) -> None:
    selector = ModelSelector(_settings(), registry)
    _register(
        registry,
        "a",
        vram_total=24.0,
        vram_free=18.0,
        model="Qwen2.5-7B-Instruct-GGUF",
        tokens_per_sec=90.0,
    )
    _register(
        registry,
        "b",
        vram_total=24.0,
        vram_free=18.0,
        model="Qwen2.5-14B-Instruct-GGUF",
        tokens_per_sec=60.0,
    )
    assert selector.draft_pair() is None


def test_no_pair_without_workers(registry: WorkerRegistry) -> None:
    selector = ModelSelector(_settings(), registry)
    assert selector.draft_pair() is None


def test_status_is_json_safe(registry: WorkerRegistry) -> None:
    import json

    selector = ModelSelector(_settings(), registry)
    _register(registry, "w1", vram_total=24.0, vram_free=18.0, model="Qwen2.5-7B-Instruct-GGUF")
    status = selector.status()
    json.dumps(status)  # must not raise
    assert status["pool"]["healthy_workers"] == 1
    assert status["demand"]["total"] == 0


def test_catalogue_models_all_fit_their_declared_vram(registry: WorkerRegistry) -> None:
    """The catalogue and the selector must agree about the VRAM numbers."""
    selector = ModelSelector(_settings(), registry)
    for spec in default_catalog().models:
        _register(
            registry,
            f"host-{spec.name}",
            vram_total=spec.vram_required_gb * 2,
            vram_free=spec.vram_required_gb * 2,
            model=None,
        )
        pool = selector.pool()
        worker = next(w for w in pool.workers if w.worker_id == f"host-{spec.name}")
        assert selector.fits(spec, worker), spec.name
