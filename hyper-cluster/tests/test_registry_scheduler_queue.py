"""Worker registry, scheduler and hold-queue unit tests."""

from __future__ import annotations

import time

import pytest

from core.config import Settings
from core.protocol import ChatMessage, ChatRequest, RequestMode, WorkerHeartbeat, WorkerStatus
from head_node.hold_queue import HoldQueue, HoldQueueFull, HoldState
from head_node.scheduler import EdgeCandidate, Scheduler
from head_node.worker_registry import WorkerRegistry
from tests.conftest import make_hello


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def test_register_creates_healthy_worker(registry: WorkerRegistry) -> None:
    record = registry.register(make_hello("wk-a", 9001))
    assert record.worker_id == "wk-a"
    assert record.healthy is True
    assert record.vram_total_gb == 24.0
    assert len(registry) == 1


def test_heartbeat_refreshes_ttl_and_telemetry(registry: WorkerRegistry) -> None:
    registry.register(make_hello("wk-a", 9001))
    before = registry.get("wk-a").expires_at
    time.sleep(0.01)
    registry.heartbeat(
        WorkerHeartbeat(
            worker_id="wk-a",
            status=WorkerStatus.BUSY,
            inflight=1,
            tokens_per_sec=42.5,
            vram_free_gb=18.0,
            vram_total_gb=24.0,
        )
    )
    record = registry.get("wk-a")
    assert record.status is WorkerStatus.BUSY
    assert record.tokens_per_sec == 42.5
    assert record.expires_at > before


def test_expired_workers_are_swept(registry: WorkerRegistry) -> None:
    registry.register(make_hello("wk-a", 9001))
    registry.set_ttl(60)
    record = registry.get("wk-a")
    record.expires_at = time.time() - 1.0
    expired = registry.sweep()
    assert [r.worker_id for r in expired] == ["wk-a"]
    assert registry.get("wk-a") is None


def test_draining_worker_is_not_healthy(registry: WorkerRegistry) -> None:
    registry.register(make_hello("wk-a", 9001))
    registry.drain("wk-a", reason="test")
    record = registry.get("wk-a")
    assert record.draining is True
    assert record.healthy is False
    assert registry.healthy_workers() == []


def test_capacity_claim_and_release(registry: WorkerRegistry) -> None:
    registry.register(make_hello("wk-a", 9001, max_concurrency=2))
    assert registry.claim("wk-a") is True
    assert registry.claim("wk-a") is True
    assert registry.claim("wk-a") is False
    registry.release("wk-a")
    assert registry.claim("wk-a") is True


def test_latency_ewma_converges(registry: WorkerRegistry) -> None:
    registry.register(make_hello("wk-a", 9001))
    for _ in range(10):
        registry.record_latency("wk-a", ttft_ms=100.0, total_ms=900.0, tokens=90, tokens_per_sec=100.0)
    record = registry.get("wk-a")
    assert 95.0 <= record.ewma_ttft_ms <= 105.0
    assert record.tokens_per_sec == 100.0


def test_expiring_within_reports_soon_to_die_workers(registry: WorkerRegistry) -> None:
    registry.register(make_hello("wk-a", 9001))
    registry.register(make_hello("wk-b", 9002))
    soon = registry.expiring_within(3600)
    assert len(soon) == 2
    assert soon[0].seconds_to_expiry <= soon[1].seconds_to_expiry


def test_totals_aggregate_capacity(registry: WorkerRegistry) -> None:
    registry.register(make_hello("wk-a", 9001, vram=24.0, free_vram=20.0, tflops=60.0))
    registry.register(make_hello("wk-b", 9002, vram=16.0, free_vram=12.0, tflops=40.0))
    totals = registry.totals()
    assert totals["workers"] == 2
    assert totals["healthy"] == 2
    assert totals["total_vram_gb"] == 40.0
    assert totals["total_tflops"] == 100.0


# --------------------------------------------------------------------------- #
# Scheduler
# --------------------------------------------------------------------------- #
def _request(text: str = "hello", mode: RequestMode = RequestMode.AUTO) -> ChatRequest:
    return ChatRequest(messages=[ChatMessage(role="user", content=text)], mode=mode)


def test_classify_routes_simple_prompts_to_fast(settings: Settings, registry: WorkerRegistry) -> None:
    scheduler = Scheduler(settings, registry)
    assert scheduler.classify(_request("hi")) is RequestMode.FAST
    assert scheduler.classify(_request("thanks!")) is RequestMode.FAST


def test_classify_routes_hard_prompts_to_council(settings: Settings, registry: WorkerRegistry) -> None:
    scheduler = Scheduler(settings, registry)
    hard = "Prove step by step that the following optimization algorithm converges, and analyze the trade-offs"
    assert scheduler.classify(_request(hard)) is RequestMode.COUNCIL


def test_classify_respects_explicit_mode(settings: Settings, registry: WorkerRegistry) -> None:
    scheduler = Scheduler(settings, registry)
    assert scheduler.classify(_request("hi", RequestMode.DEEP)) is RequestMode.DEEP


def test_select_prefers_the_faster_worker(settings: Settings, registry: WorkerRegistry) -> None:
    registry.register(make_hello("wk-fast", 9001, latency_ms=2.0))
    registry.register(make_hello("wk-slow", 9002, latency_ms=200.0))
    scheduler = Scheduler(settings, registry)
    decision = scheduler.select(_request())
    assert decision.primary is not None
    assert decision.primary.worker_id == "wk-fast"
    assert decision.has_capacity is True


def test_select_skips_saturated_workers(settings: Settings, registry: WorkerRegistry) -> None:
    registry.register(make_hello("wk-a", 9001, max_concurrency=1))
    registry.claim("wk-a")
    registry.register(make_hello("wk-b", 9002))
    scheduler = Scheduler(settings, registry)
    decision = scheduler.select(_request())
    assert decision.primary is not None
    assert decision.primary.worker_id == "wk-b"


def test_select_reports_no_capacity_when_everything_is_busy(settings: Settings, registry: WorkerRegistry) -> None:
    registry.register(make_hello("wk-a", 9001, max_concurrency=1))
    registry.claim("wk-a")
    scheduler = Scheduler(settings, registry)
    decision = scheduler.select(_request())
    assert decision.primary is None
    assert decision.has_capacity is False
    assert "capacity" in decision.reason.lower()


def test_unknown_model_is_rejected(settings: Settings, registry: WorkerRegistry) -> None:
    registry.register(make_hello("wk-a", 9001))
    scheduler = Scheduler(settings, registry)
    request = _request()
    request.model = "NotARealModel-XL"
    decision = scheduler.select(request)
    assert decision.primary is None
    assert "catalogue" in decision.reason.lower()


def test_model_load_penalty_is_applied(settings: Settings, registry: WorkerRegistry) -> None:
    registry.register(make_hello("wk-a", 9001, model="Qwen2.5-7B-Instruct-GGUF", free_vram=20.0))
    scheduler = Scheduler(settings, registry)
    request = _request()
    request.model = "Qwen2.5-14B-Instruct-GGUF"
    decision = scheduler.select(request)
    assert decision.primary is not None
    assert decision.requires_model_load is True
    assert decision.primary.cold_start_ms > 0


def test_loaded_model_beats_cold_start(settings: Settings, registry: WorkerRegistry) -> None:
    registry.register(make_hello("wk-a", 9001, model="Qwen2.5-72B-Instruct-GGUF", vram=48.0, free_vram=44.0))
    scheduler = Scheduler(settings, registry)
    request = _request()
    request.model = "Qwen2.5-72B-Instruct-GGUF"
    decision = scheduler.select(request)
    assert decision.primary is not None
    assert decision.requires_model_load is False
    assert decision.primary.cold_start_ms == 0.0


def test_select_edge_prefers_lowest_latency(settings: Settings, registry: WorkerRegistry) -> None:
    scheduler = Scheduler(settings, registry)
    candidates = [
        EdgeCandidate(provider="slow", predicted_tps=100.0, base_latency_ms=400.0, available=True),
        EdgeCandidate(provider="fast", predicted_tps=120.0, base_latency_ms=60.0, available=True),
        EdgeCandidate(provider="down", predicted_tps=0.0, base_latency_ms=0.0, available=False),
    ]
    best = scheduler.select_edge(candidates, _request())
    assert best is not None
    assert best.provider == "fast"


# --------------------------------------------------------------------------- #
# Hold queue
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_hold_queue_enqueue_and_complete() -> None:
    queue = HoldQueue(max_size=4, ttl_seconds=30)
    request = _request()
    entry = queue.enqueue(request)
    assert queue.waiting_count() == 1
    assert entry.state is HoldState.WAITING
    queue.claim(request.request_id, worker_id="wk-a")
    assert queue.get(request.request_id).state is HoldState.ASSIGNED
    queue.complete(request.request_id, None)  # type: ignore[arg-type]
    assert queue.waiting_count() == 0


@pytest.mark.asyncio
async def test_hold_queue_defers_expired_requests() -> None:
    queue = HoldQueue(max_size=4, ttl_seconds=1)
    request = _request()
    entry = queue.enqueue(request)
    entry.deadline_at = time.time() - 1.0
    expired = queue.expire()
    assert len(expired) == 1
    assert queue.deferred_count() == 1
    assert queue.get(request.request_id).state is HoldState.DEFERRED


@pytest.mark.asyncio
async def test_hold_queue_enforces_capacity() -> None:
    queue = HoldQueue(max_size=1, ttl_seconds=30)
    queue.enqueue(_request())
    with pytest.raises(HoldQueueFull):
        queue.enqueue(_request())


@pytest.mark.asyncio
async def test_hold_queue_preserves_fifo_order() -> None:
    queue = HoldQueue(max_size=8, ttl_seconds=30)
    first = _request("first")
    second = _request("second")
    queue.enqueue(first)
    time.sleep(0.01)
    queue.enqueue(second)
    ready = queue.ready()
    assert [entry.request_id for entry in ready] == [first.request_id, second.request_id]


def test_hold_queue_snapshot_reports_position() -> None:
    queue = HoldQueue(max_size=8, ttl_seconds=30)
    queue.enqueue(_request())
    snapshot = queue.snapshot()
    assert len(snapshot) == 1
    assert snapshot[0]["state"] == "waiting"
