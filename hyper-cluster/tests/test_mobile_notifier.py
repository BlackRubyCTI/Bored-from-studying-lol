"""The phone alerts: Telegram, the webhook and the event bus wiring."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from head_node.events import EventBus
from mobile.notifier import DEFAULT_EVENTS, Alert, Notifier


class Recorder:
    """A fake HTTP transport that records the pushes."""

    def __init__(self, *, fail_with: int = 0) -> None:
        self.calls: list[httpx.Request] = []
        self.fail_with = fail_with

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        if self.fail_with:
            return httpx.Response(self.fail_with, json={"ok": False})
        return httpx.Response(200, json={"ok": True})

    def bodies(self) -> list[dict]:
        import json

        return [json.loads(call.content.decode()) for call in self.calls]


def build_notifier(recorder: Recorder, **kwargs) -> Notifier:
    client = httpx.AsyncClient(transport=httpx.MockTransport(recorder.handle_async_request))
    return Notifier(client=client, **kwargs)


# --------------------------------------------------------------------------- #
# The configuration
# --------------------------------------------------------------------------- #
def test_a_notifier_without_transports_is_disabled() -> None:
    notifier = Notifier()
    assert not notifier.enabled
    assert notifier.targets == []


def test_the_targets_are_the_configured_transports() -> None:
    notifier = Notifier(telegram_token="t", telegram_chat_id="42")
    assert notifier.enabled
    assert notifier.targets == ["telegram"]

    notifier = Notifier(webhook_url="https://push.example/hook")
    assert notifier.targets == ["webhook"]

    notifier = Notifier(telegram_token="t", telegram_chat_id="42", webhook_url="https://push.example/hook")
    assert notifier.targets == ["telegram", "webhook"]


def test_a_telegram_token_without_a_chat_is_not_a_target() -> None:
    notifier = Notifier(telegram_token="t")
    assert notifier.targets == []


# --------------------------------------------------------------------------- #
# The events
# --------------------------------------------------------------------------- #
def test_the_default_events_are_the_interesting_ones() -> None:
    notifier = Notifier()
    assert notifier.wants("model.changed")
    assert notifier.wants("worker.lost")
    assert notifier.wants("evolution.cycle")
    assert not notifier.wants("request.state")


def test_a_prefix_covers_a_whole_family() -> None:
    notifier = Notifier(events=["worker"])
    assert notifier.wants("worker.lost")
    assert notifier.wants("worker.registered")
    assert not notifier.wants("model.changed")


def test_an_empty_list_means_everything() -> None:
    notifier = Notifier(events=[])
    assert notifier.wants("anything.at.all")


def test_the_alert_carries_a_title_and_a_body() -> None:
    notifier = Notifier()
    alert = notifier.render("model.changed", {"model": "qwen2.5-7b"})
    assert alert.title == "A new model is live"
    assert "qwen2.5-7b" in alert.body
    assert alert.level == "info"


def test_a_lost_worker_is_a_warning() -> None:
    notifier = Notifier()
    alert = notifier.render("worker.lost", {"worker_id": "wk-1"})
    assert alert.level == "warning"
    assert "wk-1" in alert.body


def test_an_unknown_event_still_renders() -> None:
    notifier = Notifier()
    alert = notifier.render("weird.event", {"a": 1})
    assert alert.title
    assert "a" in alert.body


# --------------------------------------------------------------------------- #
# The transports
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_the_telegram_transport_posts_the_message() -> None:
    recorder = Recorder()
    notifier = build_notifier(
        recorder,
        telegram_token="bot-token",
        telegram_chat_id="42",
    )
    results = await notifier.send(Alert(title="hello", body="world"))
    assert [result.ok for result in results] == [True]
    request = recorder.calls[0]
    assert "bot-token" in str(request.url)
    assert request.url.host == "api.telegram.org"
    body = recorder.bodies()[0]
    assert body["chat_id"] == "42"
    assert "hello" in body["text"]
    await notifier.aclose()


@pytest.mark.asyncio
async def test_the_webhook_transport_posts_the_alert() -> None:
    recorder = Recorder()
    notifier = build_notifier(
        recorder,
        webhook_url="https://push.example/hook",
        webhook_secret="s3cret",
    )
    results = await notifier.send(Alert(title="hello", body="world"))
    assert [result.ok for result in results] == [True]
    request = recorder.calls[0]
    assert str(request.url) == "https://push.example/hook"
    assert request.headers["x-hyper-secret"] == "s3cret"
    body = recorder.bodies()[0]
    assert body["title"] == "hello"
    assert body["level"] == "info"
    await notifier.aclose()


@pytest.mark.asyncio
async def test_a_failing_transport_never_breaks_the_others() -> None:
    recorder = Recorder(fail_with=500)
    notifier = build_notifier(
        recorder,
        telegram_token="bot-token",
        telegram_chat_id="42",
        webhook_url="https://push.example/hook",
    )
    results = await notifier.send(Alert(title="hello"))
    assert len(results) == 2
    assert all(not result.ok for result in results)
    assert notifier.failures == 2
    assert notifier.sent == []
    await notifier.aclose()


@pytest.mark.asyncio
async def test_an_event_that_is_not_wanted_is_not_pushed() -> None:
    recorder = Recorder()
    notifier = build_notifier(recorder, webhook_url="https://push.example/hook")
    results = await notifier.handle_event("request.state", {"request_id": "r1"})
    assert results == []
    assert recorder.calls == []
    await notifier.aclose()


@pytest.mark.asyncio
async def test_a_wanted_event_is_pushed() -> None:
    recorder = Recorder()
    notifier = build_notifier(recorder, webhook_url="https://push.example/hook")
    results = await notifier.handle_event("model.changed", {"model": "m"})
    assert len(results) == 1
    assert recorder.calls
    await notifier.aclose()


@pytest.mark.asyncio
async def test_the_throttle_drops_the_burst() -> None:
    recorder = Recorder()
    notifier = build_notifier(
        recorder,
        webhook_url="https://push.example/hook",
        min_interval_seconds=60.0,
    )
    await notifier.send(Alert(title="one"))
    await notifier.send(Alert(title="two"))
    assert len(recorder.calls) == 1
    await notifier.aclose()


# --------------------------------------------------------------------------- #
# The event bus
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_the_notifier_follows_the_event_bus() -> None:
    recorder = Recorder()
    notifier = build_notifier(recorder, webhook_url="https://push.example/hook")
    bus = EventBus()
    task = asyncio.create_task(notifier.run(bus.subscribe, bus.unsubscribe))
    await asyncio.sleep(0.05)
    await bus.emit("model.changed", {"model": "qwen"})
    for _ in range(50):
        if recorder.calls:
            break
        await asyncio.sleep(0.02)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert recorder.calls, "the event bus did not reach the notifier"
    body = recorder.bodies()[0]
    assert body["event"] == "model.changed"
    assert "qwen" in body["body"]
    await notifier.aclose()


@pytest.mark.asyncio
async def test_the_orchestrator_wires_the_notifier(tmp_path) -> None:
    """The head node builds the notifier and subscribes it to the bus."""
    from core.config import Settings
    from head_node.orchestrator import Orchestrator

    settings = Settings(
        data_dir=str(tmp_path / "data"),
        log_level="WARNING",
        log_json=False,
        notifier_enabled=True,
        notifier_webhook_url="https://push.example/hook",
        notifier_events=["model.changed"],
    )
    orchestrator = Orchestrator(settings)
    try:
        assert orchestrator.notifier is not None
        assert orchestrator.notifier.enabled
        assert orchestrator.notifier.webhook_url == "https://push.example/hook"
        stats = orchestrator.notifier.stats()
        assert stats["enabled"] is True
        assert stats["targets"] == ["webhook"]
    finally:
        await orchestrator.shutdown()


def test_the_default_events_are_stable() -> None:
    """The defaults are part of the contract with the phone."""
    assert "model.changed" in DEFAULT_EVENTS
    assert "worker.lost" in DEFAULT_EVENTS
    assert "evolution.cycle" in DEFAULT_EVENTS
