"""Loop-aware HTTP clients.

``httpx`` binds its connection pool to the event loop that first uses it.
The head node and the worker agent can, however, be driven from more than
one loop: the harness runs the HTTP server inside a thread while the
lifecycle (``start``/``stop``) runs in the test's loop, and a worker that
is restarted inside a supervisor may end up on a new loop too.

:class:`LoopBoundClient` keeps one client *per running loop* and hands the
caller the right one, so a request never crosses a loop boundary.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx


def _current_loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def _close_soon(client: httpx.AsyncClient, loop: asyncio.AbstractEventLoop | None) -> None:
    """Close a client that belongs to another (still running) loop."""
    if loop is None or not loop.is_running():
        return
    try:
        asyncio.run_coroutine_threadsafe(client.aclose(), loop)
    except RuntimeError:  # pragma: no cover - the loop died in the meantime
        pass


class LoopBoundClient:
    """An ``httpx.AsyncClient`` that always belongs to the running loop."""

    def __init__(self, client: httpx.AsyncClient | None = None, **options: Any) -> None:
        self._options = options
        #: An injected client is used as-is (its owner knows its loop).
        self._injected = client
        self._client: httpx.AsyncClient | None = client
        self._loop: asyncio.AbstractEventLoop | None = _current_loop()

    # ------------------------------------------------------------------ #
    def bind(self) -> httpx.AsyncClient:
        """The client for the loop that is running right now."""
        if self._injected is not None:
            return self._injected
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # pragma: no cover - only outside of a loop
            loop = None
        if self._client is None or self._loop is not loop:
            old, old_loop = self._client, self._loop
            self._client = httpx.AsyncClient(**self._options)
            self._loop = loop
            if old is not None:
                _close_soon(old, old_loop)
        return self._client

    async def aclose(self) -> None:
        """Close the client (from its own loop when it is not ours)."""
        client = self._client
        self._client = None
        self._loop = None
        if client is None:
            return
        loop = _current_loop()
        if loop is not None and self._loop is not None and loop is not self._loop:
            _close_soon(client, self._loop)
            return
        await client.aclose()


__all__ = ["LoopBoundClient"]
