"""Shared pytest fixtures.

Every test runs against an isolated data directory, a fresh worker registry
and a *real* orchestrator (with the background capacity/maintenance loops
running) so behaviour matches production.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterator

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import Settings  # noqa: E402
from core.credentials import CredentialVault  # noqa: E402
from core.protocol import Capabilities, WorkerHello  # noqa: E402
from head_node.events import EventBus  # noqa: E402
from head_node.orchestrator import Orchestrator  # noqa: E402
from head_node.request_store import RequestLedger  # noqa: E402
from head_node.scheduler import Scheduler  # noqa: E402
from head_node.worker_registry import WorkerRegistry  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Keep every test away from the developer's real environment."""
    for key in list(os.environ):
        if key.startswith("HYPER_"):
            monkeypatch.delenv(key, raising=False)
    for key in ("GROQ_API_KEY", "CEREBRAS_API_KEY", "SAMBANOVA_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HYPER_DATA_DIR", str(tmp_path / "data"))
    yield


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Isolated settings for the test-suite."""
    return Settings(
        data_dir=str(tmp_path / "data"),
        log_level="WARNING",
        log_json=False,
        api_key="",
        mesh_token="",
        admin_token="",
        worker_ttl_seconds=60,
        heartbeat_interval_seconds=5,
        hold_request_ttl_seconds=30,
        request_timeout_seconds=20,
        speculative_enabled=True,
        speculative_edge_providers=["groq"],
        council_size=3,
        council_critique_rounds=0,
    )


@pytest.fixture
def vault(tmp_path: Path) -> CredentialVault:
    """Empty, isolated credential vault."""
    return CredentialVault(tmp_path / "credentials.json").load()


@pytest.fixture
def registry(settings: Settings) -> WorkerRegistry:
    """A fresh worker registry."""
    return WorkerRegistry(
        ttl_seconds=settings.worker_ttl_seconds,
        max_concurrency_per_worker=settings.max_concurrency_per_worker,
    )


@pytest.fixture
def scheduler(settings: Settings, registry: WorkerRegistry) -> Scheduler:
    """Latency-aware scheduler bound to the registry."""
    return Scheduler(settings, registry)


@pytest.fixture
async def orchestrator(
    settings: Settings,
    registry: WorkerRegistry,
    scheduler: Scheduler,
    vault: CredentialVault,
) -> Iterator[Orchestrator]:
    """A fully wired orchestrator with its background loops running."""
    engine = Orchestrator(
        settings,
        registry=registry,
        scheduler=scheduler,
        vault=vault,
        events=EventBus(),
        ledger=RequestLedger(),
    )
    await engine.startup()
    try:
        yield engine
    finally:
        await engine.shutdown()


def make_hello(
    worker_id: str,
    port: int,
    *,
    host: str = "127.0.0.1",
    provider: str = "local",
    vram: float = 24.0,
    free_vram: float = 20.0,
    model: str = "Qwen2.5-7B-Instruct-GGUF",
    family: str = "qwen2",
    params_b: float = 7.0,
    latency_ms: float = 5.0,
    tflops: float = 60.0,
    device: str = "cuda",
    max_concurrency: int = 2,
) -> WorkerHello:
    """Build a registration payload for a worker."""
    return WorkerHello(
        worker_id=worker_id,
        host=host,
        port=port,
        model_loaded=model,
        model_family=family,
        model_params_b=params_b,
        capabilities=Capabilities(
            vram_total_gb=vram,
            vram_free_gb=free_vram,
            compute_tflops=tflops,
            cpu_cores=8,
            ram_gb=32.0,
            provider=provider,
            region="test",
            backend="llama.cpp",
            device=device,
            supported_families=[family, "llama", "gemma2", "phi3"],
            supported_context=32768,
            max_concurrency=max_concurrency,
            mesh_latency_ms=latency_ms,
        ),
    )


__all__ = ["make_hello"]
