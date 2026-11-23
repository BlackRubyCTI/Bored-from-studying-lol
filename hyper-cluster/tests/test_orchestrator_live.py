"""Live orchestrator tests -- these run a real worker over real HTTP/SSE."""

from __future__ import annotations

import asyncio
import time

import pytest

from core.protocol import ChatCompletion, ChatMessage, ChatRequest, RequestMode
from head_node.orchestrator import Orchestrator
from tests.conftest import make_hello
from tests.fake_worker import WorkerServer


@pytest.fixture
def worker_a() -> WorkerServer:
    server = WorkerServer(worker_id="wk-a", model="Qwen2.5-7B-Instruct-GGUF", tokens_per_sec=120.0, delay_ms=1.0)
    server.start()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture
def worker_b() -> WorkerServer:
    server = WorkerServer(worker_id="wk-b", model="Llama-3.1-8B-Instruct-GGUF", tokens_per_sec=200.0, delay_ms=1.0)
    server.start()
    try:
        yield server
    finally:
        server.stop()


def _request(text: str = "hello cluster", mode: RequestMode = RequestMode.DEEP, **kwargs) -> ChatRequest:
    return ChatRequest(
        messages=[ChatMessage(role="user", content=text)],
        mode=mode,
        max_tokens=64,
        stream=True,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Happy paths
# --------------------------------------------------------------------------- #
async def test_submit_routes_to_a_registered_worker(
    orchestrator: Orchestrator, worker_a: WorkerServer
) -> None:
    orchestrator.registry.register(make_hello("wk-a", worker_a.port, provider="local"))
    completion = await orchestrator.submit(_request())
    assert completion.error is None
    assert "wk-a" in completion.text
    assert completion.worker_id == "wk-a"
    assert completion.tokens > 0
    assert completion.total_ms > 0


async def test_stream_emits_tokens_then_done(
    orchestrator: Orchestrator, worker_a: WorkerServer
) -> None:
    orchestrator.registry.register(make_hello("wk-a", worker_a.port))
    events = []
    async for event in orchestrator.stream(_request()):
        events.append((event.event, event.data))
    names = [name for name, _ in events]
    assert "start" in names
    assert "token" in names
    assert names[-1] == "done"
    token_text = "".join(data["text"] for name, data in events if name == "token")
    assert "wk-a" in token_text


async def test_fast_mode_uses_edge_when_available(
    orchestrator: Orchestrator, worker_a: WorkerServer
) -> None:
    """With no edge credentials the fast path falls back to the cluster."""
    orchestrator.registry.register(make_hello("wk-a", worker_a.port))
    completion = await orchestrator.submit(_request("hi", RequestMode.FAST))
    assert completion.error is None
    assert completion.text


# --------------------------------------------------------------------------- #
# Failover
# --------------------------------------------------------------------------- #
async def test_worker_failure_fails_over_to_another_worker(
    orchestrator: Orchestrator, worker_a: WorkerServer, worker_b: WorkerServer
) -> None:
    orchestrator.registry.register(make_hello("wk-a", worker_a.port, latency_ms=1.0))
    orchestrator.registry.register(make_hello("wk-b", worker_b.port, latency_ms=90.0))
    worker_a.state.fail = True

    completion = await orchestrator.submit(_request())
    assert completion.error is None
    assert completion.worker_id == "wk-b"
    assert "wk-b" in completion.text


# --------------------------------------------------------------------------- #
# Hold & requeue
# --------------------------------------------------------------------------- #
async def test_request_is_held_then_deferred_when_no_capacity(
    orchestrator: Orchestrator,
) -> None:
    request = _request("please answer me", RequestMode.DEEP, deadline_seconds=0.6)
    completion = await orchestrator.submit(request)
    assert completion.error == "deferred"
    assert orchestrator.hold_queue.deferred_count() == 1
    status = orchestrator.ledger.get(request.request_id)
    assert status is not None
    assert status.state.value == "deferred"


async def test_deferred_request_is_replayed_when_capacity_returns(
    orchestrator: Orchestrator, worker_a: WorkerServer
) -> None:
    request = _request("replay me please", RequestMode.DEEP, deadline_seconds=0.5)
    completion = await orchestrator.submit(request)
    assert completion.error == "deferred"

    # A worker joins: the background maintenance loop replays the deferred work.
    orchestrator.registry.register(make_hello("wk-a", worker_a.port))
    deadline = time.time() + 20.0
    result: ChatCompletion | None = None
    while time.time() < deadline:
        result = orchestrator.ledger.result(request.request_id)
        if result is not None and result.text and not result.error:
            break
        await asyncio.sleep(0.2)
    assert result is not None, "deferred request was never replayed"
    assert "wk-a" in result.text


async def test_held_request_is_resumed_when_capacity_arrives(
    orchestrator: Orchestrator, worker_a: WorkerServer
) -> None:
    """A request that is still waiting is served as soon as a worker joins."""
    request = _request("hold me", RequestMode.DEEP, deadline_seconds=20.0)
    task = asyncio.create_task(orchestrator.submit(request))
    await asyncio.sleep(0.3)
    assert orchestrator.hold_queue.waiting_count() == 1
    orchestrator.registry.register(make_hello("wk-a", worker_a.port))
    completion = await asyncio.wait_for(task, timeout=15.0)
    assert completion.error is None
    assert "wk-a" in completion.text


# --------------------------------------------------------------------------- #
# Council mode
# --------------------------------------------------------------------------- #
async def test_council_mode_deliberates_across_two_workers(
    orchestrator: Orchestrator, worker_a: WorkerServer, worker_b: WorkerServer
) -> None:
    orchestrator.registry.register(make_hello("wk-a", worker_a.port, model="Qwen2.5-7B-Instruct-GGUF"))
    orchestrator.registry.register(
        make_hello("wk-b", worker_b.port, model="Llama-3.1-8B-Instruct-GGUF", family="llama")
    )
    request = _request(
        "Prove step by step that the sum of two even integers is even, and analyze the trade-offs",
        RequestMode.COUNCIL,
    )
    completion = await orchestrator.submit(request)
    assert completion.error is None
    assert completion.council is not None
    assert len(completion.council["members"]) == 2
    assert completion.council["consensus"] >= 0.0
    assert completion.text


async def test_council_emits_member_events(
    orchestrator: Orchestrator, worker_a: WorkerServer, worker_b: WorkerServer
) -> None:
    orchestrator.registry.register(make_hello("wk-a", worker_a.port))
    orchestrator.registry.register(make_hello("wk-b", worker_b.port, model="Llama-3.1-8B-Instruct-GGUF", family="llama"))
    request = _request("compare these two architectures and analyze", RequestMode.COUNCIL)
    events = []
    async for event in orchestrator.stream(request):
        events.append(event.event)
    assert "council" in events
    assert "done" in events


# --------------------------------------------------------------------------- #
# Telemetry
# --------------------------------------------------------------------------- #
async def test_status_reflects_workers_and_queue(
    orchestrator: Orchestrator, worker_a: WorkerServer
) -> None:
    orchestrator.registry.register(make_hello("wk-a", worker_a.port))
    status = orchestrator.status()
    assert status.worker_count == 1
    assert status.healthy_workers == 1
    assert status.total_vram_gb == 24.0
    assert status.cluster_name == orchestrator.settings.cluster_name


async def test_held_requests_leave_the_queue_once_they_are_served(
    orchestrator: Orchestrator, worker_a: WorkerServer
) -> None:
    """Regression: a resumed request must not stay in the hold buffer.

    Before the fix the buffer was never drained, so ``waiting``/``held``
    counters grew forever and the same request could be replayed twice.
    """
    orchestrator.registry.register(make_hello("wk-a", worker_a.port))
    worker_a.state.delay_ms = 60.0
    try:
        requests = [_request(f"burst {index}", RequestMode.DEEP) for index in range(6)]
        completions = await asyncio.gather(*(orchestrator.submit(request) for request in requests))
        assert all(completion.error is None for completion in completions)
        assert orchestrator.hold_queue.waiting_count() == 0
        assert len(orchestrator.hold_queue) == 0
        status = orchestrator.status()
        assert status.queued_requests == 0
        assert status.held_requests == 0
    finally:
        worker_a.state.delay_ms = 1.0
