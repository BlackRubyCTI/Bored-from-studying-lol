"""HTTP client used by the head node to talk to mesh workers.

Worker protocol (implemented by ``worker/worker_agent.py`` in milestone 2)::

    GET  /health            -> {"status": "ok", "worker_id": ..., "model_loaded": ...}
    POST /v1/chat           -> Server-Sent Events, one TokenChunk JSON per event
    POST /v1/chat/sync      -> a single ChatCompletion JSON object
    POST /v1/model/load     -> ModelLoadResult
    POST /v1/admin/drain    -> {"draining": true}

The client is deliberately defensive: a worker that dies mid-stream is turned
into a typed exception the orchestrator can retry on another node.
"""

from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator

import httpx

from core.config import Settings
from core.errors import WorkerUnreachableError
from core.logging_setup import get_logger
from core.protocol import (
    ChatCompletion,
    ChatRequest,
    ModelLoadResult,
    ModelSpec,
    TokenChunk,
)
from head_node.worker_registry import WorkerRecord

LOGGER = get_logger("head_node.worker_client")


async def parse_sse_lines(lines: AsyncIterator[str]) -> AsyncIterator[str]:
    """Normalise an SSE stream into payload strings.

    ``lines`` is expected to yield *already split* lines (as produced by
    ``httpx``'s ``aiter_lines``).  The parser understands ``data:`` fields,
    comments, multi-line data fields and the ``[DONE]`` sentinel, and it also
    tolerates workers that emit bare JSON lines.
    """
    data_lines: list[str] = []
    async for raw in lines:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        line = raw.rstrip("\r")
        if not line:
            if data_lines:
                payload = "\n".join(data_lines)
                data_lines = []
                if payload == "[DONE]":
                    return
                yield payload
            continue
        if line.startswith(":"):  # comment / keep-alive
            continue
        if line.startswith("event:"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].strip())
            continue
        # Tolerate workers that emit bare JSON lines.
        data_lines.append(line.strip())
    if data_lines:
        payload = "\n".join(data_lines)
        if payload != "[DONE]":
            yield payload


class WorkerClient:
    """Thin async HTTP client for worker endpoints."""

    def __init__(self, settings: Settings, *, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=settings.request_timeout_seconds, write=10.0, pool=10.0),
            limits=httpx.Limits(max_connections=256, max_keepalive_connections=64),
            headers={"user-agent": "hyper-cluster-head/1.0"},
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # ------------------------------------------------------------------ #
    def _url(self, record: WorkerRecord, path: str) -> str:
        return f"{record.base_url()}{path}"

    def _headers(self, record: WorkerRecord) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        token = self.settings.mesh_token
        if token:
            headers["x-mesh-token"] = token
        return headers

    # ------------------------------------------------------------------ #
    async def health(self, record: WorkerRecord) -> dict[str, Any]:
        """Fetch a worker's health document."""
        try:
            response = await self._client.get(self._url(record, "/health"), headers=self._headers(record))
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise WorkerUnreachableError(
                f"worker {record.worker_id} unreachable: {exc}",
                details={"worker_id": record.worker_id, "address": record.base_url()},
            ) from exc

    async def probe_latency(self, record: WorkerRecord) -> float:
        """Measure the mesh round-trip time to a worker in milliseconds."""
        started = time.perf_counter()
        try:
            await self.health(record)
        except WorkerUnreachableError:
            return -1.0
        return (time.perf_counter() - started) * 1000.0

    async def stream_chat(
        self, record: WorkerRecord, request: ChatRequest
    ) -> AsyncIterator[TokenChunk]:
        """Stream tokens from a worker.

        Raises:
            WorkerUnreachableError: when the worker refuses or dies mid-stream.
        """
        payload = request.model_dump(mode="json")
        payload["stream"] = True
        try:
            async with self._client.stream(
                "POST",
                self._url(record, "/v1/chat"),
                json=payload,
                headers=self._headers(record),
                timeout=httpx.Timeout(self.settings.request_timeout_seconds, connect=5.0),
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    raise WorkerUnreachableError(
                        f"worker {record.worker_id} returned HTTP {response.status_code}",
                        details={"worker_id": record.worker_id, "body": body[:512].decode("utf-8", "replace")},
                    )
                async for data in parse_sse_lines(response.aiter_lines()):
                    try:
                        chunk = TokenChunk(**json.loads(data))
                    except (json.JSONDecodeError, ValueError) as exc:
                        LOGGER.warning(
                            "discarding malformed worker chunk",
                            extra={"hc_worker": record.worker_id, "hc_error": str(exc)},
                        )
                        continue
                    yield chunk
        except httpx.HTTPError as exc:
            raise WorkerUnreachableError(
                f"worker {record.worker_id} stream failed: {exc}",
                details={"worker_id": record.worker_id},
            ) from exc

    async def chat(self, record: WorkerRecord, request: ChatRequest) -> ChatCompletion:
        """Run a blocking (non-streaming) completion on a worker."""
        payload = request.model_dump(mode="json")
        payload["stream"] = False
        try:
            response = await self._client.post(
                self._url(record, "/v1/chat/sync"),
                json=payload,
                headers=self._headers(record),
                timeout=httpx.Timeout(self.settings.request_timeout_seconds, connect=5.0),
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            raise WorkerUnreachableError(
                f"worker {record.worker_id} failed: {exc}",
                details={"worker_id": record.worker_id},
            ) from exc
        except json.JSONDecodeError as exc:
            raise WorkerUnreachableError(
                f"worker {record.worker_id} returned invalid JSON: {exc}",
                details={"worker_id": record.worker_id},
            ) from exc
        try:
            return ChatCompletion(**data)
        except Exception as exc:  # noqa: BLE001 - tolerate worker schema drift
            raise WorkerUnreachableError(
                f"worker {record.worker_id} returned an invalid completion: {exc}",
                details={"worker_id": record.worker_id},
            ) from exc

    async def load_model(self, record: WorkerRecord, spec: ModelSpec, *, force: bool = False) -> ModelLoadResult:
        """Ask a worker to hot-load a model."""
        body = {
            "worker_id": record.worker_id,
            "model": spec.name,
            "family": spec.family,
            "params_b": spec.params_b,
            "vram_required_gb": spec.vram_required_gb,
            "force": force,
        }
        try:
            response = await self._client.post(
                self._url(record, "/v1/model/load"),
                json=body,
                headers=self._headers(record),
                timeout=httpx.Timeout(600.0, connect=5.0),
            )
            response.raise_for_status()
            return ModelLoadResult(**response.json())
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise WorkerUnreachableError(
                f"worker {record.worker_id} failed to load {spec.name}: {exc}",
                details={"worker_id": record.worker_id, "model": spec.name},
            ) from exc

    async def drain(self, record: WorkerRecord) -> bool:
        """Ask a worker to stop accepting new work."""
        try:
            response = await self._client.post(
                self._url(record, "/v1/admin/drain"),
                json={"worker_id": record.worker_id},
                headers=self._headers(record),
                timeout=httpx.Timeout(10.0, connect=3.0),
            )
            return response.status_code < 400
        except httpx.HTTPError as exc:
            LOGGER.warning(
                "drain request failed",
                extra={"hc_worker": record.worker_id, "hc_error": str(exc)},
            )
            return False


__all__ = ["WorkerClient", "parse_sse_lines"]
