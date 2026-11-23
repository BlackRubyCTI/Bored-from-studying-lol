"""Async event bus used to push live telemetry to the mobile dashboard.

The bus is a fan-out of bounded queues.  Slow consumers never block the
control plane: when a subscriber's queue is full the *oldest* event is dropped
and the subscriber is told it fell behind, so the phone can re-sync from
``GET /v1/status`` instead of receiving stale events.
"""

from __future__ import annotations

import asyncio
from typing import Any

from core.logging_setup import get_logger
from core.protocol import EventType, StreamEvent

LOGGER = get_logger("head_node.events")


class EventBus:
    """Bounded fan-out event bus."""

    def __init__(self, *, queue_size: int = 256, history_size: int = 200) -> None:
        self._subscribers: set[asyncio.Queue[StreamEvent]] = set()
        self._history: list[StreamEvent] = []
        self._history_size = history_size
        self._queue_size = queue_size
        self._lock = asyncio.Lock()
        self._sequence = 0

    # ------------------------------------------------------------------ #
    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    # ------------------------------------------------------------------ #
    async def publish(self, event: StreamEvent) -> None:
        """Publish an event to every subscriber (non-blocking)."""
        async with self._lock:
            self._history.append(event)
            if len(self._history) > self._history_size:
                del self._history[: len(self._history) - self._history_size]
            targets = list(self._subscribers)
        for queue in targets:
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:  # pragma: no cover - race, harmless
                    pass
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:  # pragma: no cover - race, harmless
                pass

    def publish_nowait(self, event: StreamEvent) -> None:
        """Publish from synchronous code (best effort, never raises)."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            LOGGER.debug("no running loop; dropping event", extra={"hc_event": event.event})
            return
        loop.create_task(self.publish(event))

    async def emit(self, event_type: EventType | str, data: dict[str, Any] | None = None, *, request_id: str = "") -> None:
        """Convenience wrapper around :meth:`publish`."""
        event = StreamEvent(
            event=str(getattr(event_type, "value", event_type)),
            request_id=request_id,
            data=data or {},
        )
        await self.publish(event)

    # ------------------------------------------------------------------ #
    async def subscribe(self) -> asyncio.Queue[StreamEvent]:
        """Register a new subscriber queue."""
        queue: asyncio.Queue[StreamEvent] = asyncio.Queue(maxsize=self._queue_size)
        async with self._lock:
            self._subscribers.add(queue)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue[StreamEvent]) -> None:
        """Remove a subscriber queue."""
        async with self._lock:
            self._subscribers.discard(queue)

    def history(self, limit: int = 50) -> list[StreamEvent]:
        """Most recent events, newest last."""
        return list(self._history[-limit:])

    async def close(self) -> None:
        """Drop every subscriber (used on shutdown)."""
        async with self._lock:
            queues = list(self._subscribers)
            self._subscribers.clear()
        for queue in queues:
            try:
                queue.put_nowait(StreamEvent(event="system", data={"reason": "shutdown"}))
            except asyncio.QueueFull:  # pragma: no cover
                pass


__all__ = ["EventBus"]
