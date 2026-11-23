"""Phone alerts: the cluster talks back.

The phone should not have to poll to know that something happened: a new
model shipped, a worker died, the cluster learned something, a credential
needs attention.  The notifier subscribes to the head node's event bus and
pushes the interesting events to the phone through

* **Telegram** -- the bot API (the phone gets a real push notification);
* **a webhook** -- any URL that accepts a JSON body (ntfy, a push relay, a
  small server of your own, ...).

Both transports are best effort: a failed push is logged and dropped, it
never blocks the event bus or the request path.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import httpx

from core.ids import mono_ms, now_s
from core.logging_setup import get_logger

LOGGER = get_logger("mobile.notifier")

#: The events that are worth a phone alert by default.
DEFAULT_EVENTS: tuple[str, ...] = (
    "model.changed",
    "worker.lost",
    "worker.expired",
    "capacity.updated",
    "evolution.cycle",
    "system",
)

#: A human readable title for every event.
EVENT_TITLES: dict[str, str] = {
    "model.changed": "A new model is live",
    "worker.registered": "A worker joined",
    "worker.updated": "A worker changed",
    "worker.draining": "A worker is draining",
    "worker.expired": "A worker expired",
    "worker.lost": "A worker was lost",
    "capacity.updated": "The cluster scaled",
    "queue.updated": "The queue changed",
    "request.state": "A request finished",
    "evolution.cycle": "The cluster learned something",
    "system": "The cluster says hello",
}


@dataclass
class Alert:
    """One message for the phone."""

    title: str
    body: str = ""
    level: str = "info"
    event: str = ""
    created_at: float = field(default_factory=now_s)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "body": self.body,
            "level": self.level,
            "event": self.event,
            "created_at": self.created_at,
        }

    def to_text(self) -> str:
        """The plain-text body (Telegram)."""
        if self.body:
            return f"{self.title}\n\n{self.body}"
        return self.title


@dataclass
class PushResult:
    """The outcome of one push."""

    ok: bool
    target: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "target": self.target, "error": self.error}


class Notifier:
    """Pushes the events of the cluster to the phone."""

    def __init__(
        self,
        *,
        telegram_token: str = "",
        telegram_chat_id: str = "",
        webhook_url: str = "",
        webhook_secret: str = "",
        events: Iterable[str] | None = None,
        client: httpx.AsyncClient | None = None,
        timeout: float = 15.0,
        min_interval_seconds: float = 0.0,
    ) -> None:
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        self.webhook_url = webhook_url
        self.webhook_secret = webhook_secret
        self.events = tuple(events) if events is not None else DEFAULT_EVENTS
        self.timeout = timeout
        self.min_interval_seconds = max(0.0, float(min_interval_seconds))
        self._client = client
        self._owns_client = client is None
        self._last_sent_at = 0.0
        self.sent: list[Alert] = []
        self.failures = 0

    # ------------------------------------------------------------------ #
    @property
    def enabled(self) -> bool:
        """Whether at least one transport is configured."""
        return bool((self.telegram_token and self.telegram_chat_id) or self.webhook_url)

    @property
    def targets(self) -> list[str]:
        found: list[str] = []
        if self.telegram_token and self.telegram_chat_id:
            found.append("telegram")
        if self.webhook_url:
            found.append("webhook")
        return found

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------ #
    def wants(self, event: str) -> bool:
        """Whether an event is worth a push."""
        if not self.events:
            return True
        if event in self.events:
            return True
        # a prefix match lets "worker" cover every worker event
        return any(event.startswith(f"{prefix}.") for prefix in self.events)

    def render(self, event: str, data: dict[str, Any]) -> Alert:
        """Turn an event into an alert."""
        title = EVENT_TITLES.get(event, event.replace(".", " ").capitalize())
        body = self._summarise(event, data)
        level = "warning" if event in ("worker.lost", "worker.expired", "system") else "info"
        return Alert(title=title, body=body, level=level, event=event)

    @staticmethod
    def _summarise(event: str, data: dict[str, Any]) -> str:
        if not isinstance(data, dict):
            return ""
        if event == "model.changed":
            model = data.get("model") or "?"
            return f"The cluster now serves {model}."
        if event in ("worker.lost", "worker.expired", "worker.draining", "worker.registered"):
            worker = data.get("worker_id") or "?"
            return f"Worker {worker} ({event.split('.')[-1]})."
        if event == "capacity.updated":
            return " ".join(f"{key}={value}" for key, value in list(data.items())[:6])
        if event == "evolution.cycle":
            verified = data.get("verified")
            sha = str(data.get("artifact_sha256") or "")[:12]
            if verified:
                return f"{verified} verified traces, new brain {sha}."
            return f"A cycle finished ({sha})."
        if event == "system":
            return str(data.get("message") or data.get("reason") or json.dumps(data)[:200])
        return json.dumps(data)[:400]

    # ------------------------------------------------------------------ #
    async def send(self, alert: Alert) -> list[PushResult]:
        """Push one alert through every configured transport."""
        if not self.enabled:
            return []
        if self.min_interval_seconds:
            now = mono_ms() / 1000.0
            if now - self._last_sent_at < self.min_interval_seconds:
                return []
        self._last_sent_at = mono_ms() / 1000.0
        results: list[PushResult] = []
        for target in self.targets:
            try:
                if target == "telegram":
                    await self._send_telegram(alert)
                else:
                    await self._send_webhook(alert)
                results.append(PushResult(ok=True, target=target))
            except Exception as exc:  # noqa: BLE001 - a push must never break the bus
                self.failures += 1
                LOGGER.warning("push failed", extra={"hc_target": target, "hc_error": str(exc)})
                results.append(PushResult(ok=False, target=target, error=str(exc)))
        if any(result.ok for result in results):
            self.sent.append(alert)
        return results

    async def _send_telegram(self, alert: Alert) -> None:
        response = await self._http().post(
            f"https://api.telegram.org/bot{self.telegram_token}/sendMessage",
            json={
                "chat_id": self.telegram_chat_id,
                "text": alert.to_text(),
                "disable_notification": alert.level == "info",
            },
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"telegram refused the message: HTTP {response.status_code}")

    async def _send_webhook(self, alert: Alert) -> None:
        headers = {"content-type": "application/json"}
        if self.webhook_secret:
            headers["x-hyper-secret"] = self.webhook_secret
        response = await self._http().post(
            self.webhook_url,
            json=alert.to_dict(),
            headers=headers,
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"the webhook refused the message: HTTP {response.status_code}")

    # ------------------------------------------------------------------ #
    async def handle_event(self, event: str, data: dict[str, Any]) -> list[PushResult]:
        """Push an event (if it is worth one)."""
        if not self.wants(event):
            return []
        alert = self.render(event, data)
        return await self.send(alert)

    async def run(self, subscribe: Any, unsubscribe: Any) -> None:
        """Forward the events of the bus until the task is cancelled."""
        queue = await subscribe()
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    continue
                try:
                    await self.handle_event(event.event, event.data)
                except Exception:  # noqa: BLE001
                    LOGGER.exception("notifier failed")
        finally:
            try:
                await unsubscribe(queue)
            except Exception:  # noqa: BLE001
                pass

    def stats(self) -> dict[str, Any]:
        """A JSON-safe view of the notifier."""
        return {
            "enabled": self.enabled,
            "targets": self.targets,
            "events": list(self.events),
            "sent": len(self.sent),
            "failures": self.failures,
            "last": self.sent[-1].to_dict() if self.sent else None,
        }


__all__ = ["DEFAULT_EVENTS", "EVENT_TITLES", "Alert", "Notifier", "PushResult"]
