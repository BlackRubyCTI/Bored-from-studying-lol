"""Harnesses that run a *real* head node and *real* worker agents in-process.

Both are uvicorn servers on loopback ports, so the tests exercise genuine
HTTP + Server-Sent Events, genuine registration and genuine heartbeats.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from typing import Any

import uvicorn

from core.config import Settings
from head_node.app import create_app
from worker.config import WorkerConfig
from worker.inference_engine import InferenceEngine, ModelRuntime
from worker.worker_agent import WorkerAgent


def free_port() -> int:
    """Find an unused TCP port on the loopback interface."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class ThreadedServer:
    """Runs an ASGI app with uvicorn inside a background thread."""

    def __init__(self, app: Any, port: int) -> None:
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", access_log=False)
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"server-{port}")

    def _run(self) -> None:
        asyncio.run(self._server.serve())

    def start(self, timeout: float = 15.0) -> ThreadedServer:
        """Start the server and wait until it accepts connections."""
        self._thread.start()
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.2):
                    return self
            except OSError:
                time.sleep(0.03)
        raise RuntimeError(f"server on port {self.port} never started")

    def stop(self) -> None:
        """Stop the server."""
        self._server.should_exit = True
        self._thread.join(timeout=5.0)


class HeadServer(ThreadedServer):
    """The control plane, served over real HTTP."""

    def __init__(self, settings: Settings, orchestrator: Any | None = None) -> None:
        super().__init__(create_app(settings, orchestrator=orchestrator), free_port())


class WorkerHarness(ThreadedServer):
    """A mesh worker agent, served over real HTTP."""

    def __init__(
        self,
        *,
        head_url: str,
        mesh_token: str = "",
        model: str = "Qwen2.5-7B-Instruct-GGUF",
        family: str = "qwen2",
        params_b: float = 7.0,
        provider: str = "local",
        region: str = "test",
        port: int | None = None,
        worker_id: str = "",
        vram_total_gb: float = 24.0,
        vram_free_gb: float = 20.0,
        device: str = "cuda",
        max_concurrency: int = 2,
        mesh_latency_ms: float = 5.0,
        session_ends_in: float | None = None,
    ) -> None:
        self.config = WorkerConfig(
            worker_id=worker_id,
            provider=provider,
            region=region,
            head_url=head_url,
            mesh_token=mesh_token,
            bind_host="127.0.0.1",
            bind_port=port or free_port(),
            backend="simulator",
            model_filename=f"{model}.gguf",
            model_family=family,
            model_params_b=params_b,
            context_size=4096,
            max_concurrency=max_concurrency,
            heartbeat_seconds=2,
            session_ends_at=(time.time() + session_ends_in) if session_ends_in else 0.0,
        )
        self.engine = InferenceEngine(
            backend="simulator",
            default_runtime=ModelRuntime(
                name=model,
                filename=f"{model}.gguf",
                family=family,
                params_b=params_b,
                context=4096,
            ),
        )
        self.agent = WorkerAgent(self.config, engine=self.engine)
        super().__init__(self.agent.create_app(), self.config.bind_port)
        # Advertised hardware (the sandbox host has no GPU, so we inject it).
        self.vram_total_gb = vram_total_gb
        self.vram_free_gb = vram_free_gb
        self.device = device
        self.mesh_latency_ms = mesh_latency_ms

    async def boot(self) -> None:
        """Load the model, benchmark and register with the head node.

        The agent registers in the background, so this waits until the head
        node actually knows the worker -- otherwise a test could assert on a
        registration that has not happened yet.
        """
        await self.agent.start()
        await self.wait_registered()

    async def wait_registered(self, timeout: float = 20.0) -> bool:
        """Wait until the head node lists this worker."""
        import httpx

        deadline = time.time() + timeout
        async with httpx.AsyncClient(timeout=5.0) as client:
            while time.time() < deadline:
                try:
                    response = await client.get(f"{self.config.head_url}/v1/workers")
                    if response.status_code == 200:
                        for worker in response.json()["workers"]:
                            if worker["worker_id"] == self.agent.worker_id:
                                return True
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.1)
        return False

    async def shutdown(self) -> None:
        """Stop the agent."""
        await self.agent.stop()

    async def crash(self) -> None:
        """Simulate a hard crash: the server dies and the heartbeats stop.

        Unlike :meth:`shutdown` this does *not* drain, so the head node has to
        notice the death on its own.  The background tasks are cancelled *and
        awaited* first, so no registration can be in flight when the worker
        "dies" -- exactly like a process that is killed.
        """
        import contextlib

        tasks = (self.agent._heartbeat_task, self.agent._register_task)  # noqa: SLF001
        for task in tasks:
            if task is not None:
                task.cancel()
        for task in tasks:
            if task is not None:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        self.agent._heartbeat_task = None  # noqa: SLF001
        self.agent._register_task = None  # noqa: SLF001
        self.stop()
        with contextlib.suppress(Exception):
            await self.agent._client.aclose()  # noqa: SLF001
        self.agent.draining = False


def patch_capabilities(harness: WorkerHarness) -> None:
    """Make the agent advertise the injected hardware instead of the host's."""
    original = harness.agent._capabilities  # noqa: SLF001 - test seam

    def _capabilities() -> Any:
        caps = original()
        caps.vram_total_gb = harness.vram_total_gb
        caps.vram_free_gb = harness.vram_free_gb
        caps.compute_tflops = 60.0
        caps.device = harness.device
        caps.mesh_latency_ms = harness.mesh_latency_ms
        caps.supported_families = ["qwen2", "llama", "gemma2", "phi3"]
        return caps

    harness.agent._capabilities = _capabilities  # type: ignore[method-assign]  # noqa: SLF001


__all__ = ["ThreadedServer", "HeadServer", "WorkerHarness", "free_port", "patch_capabilities"]
