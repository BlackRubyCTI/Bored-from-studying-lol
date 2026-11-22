"""Self-benchmarking telemetry for mesh workers.

When a worker joins the mesh it measures *itself* instead of guessing:

* **GPU** -- name, total/used/free VRAM, compute capability, utilisation and
  temperature (via ``nvidia-smi``, with a ``torch`` fallback).
* **CPU / RAM** -- cores, physical cores, memory (via ``psutil``).
* **FLOPS** -- a real floating-point throughput benchmark (numpy on the CPU,
  ``torch`` on the GPU when available).
* **Mesh latency** -- round-trip time to the head node and to peer workers,
  reported as min/p50/p95/max.
* **Decode throughput** -- measured tokens/second of the resident model.

The result is published to the head node as :class:`core.protocol.Capabilities`
and is what the scheduler uses to place work.
"""

from __future__ import annotations

import asyncio
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

from core.ids import now_s
from core.logging_setup import get_logger

LOGGER = get_logger("worker.telemetry")

_NVIDIA_QUERY = (
    "index,name,memory.total,memory.used,memory.free,compute_cap,temperature.gpu,utilization.gpu"
)


@dataclass
class GPUInfo:
    """A single accelerator."""

    index: int = 0
    name: str = "unknown"
    vram_total_gb: float = 0.0
    vram_used_gb: float = 0.0
    vram_free_gb: float = 0.0
    compute_cap: str = ""
    temperature_c: float = 0.0
    utilization_pct: float = 0.0
    driver: str = ""
    source: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "vram_total_gb": round(self.vram_total_gb, 3),
            "vram_used_gb": round(self.vram_used_gb, 3),
            "vram_free_gb": round(self.vram_free_gb, 3),
            "compute_cap": self.compute_cap,
            "temperature_c": self.temperature_c,
            "utilization_pct": self.utilization_pct,
            "driver": self.driver,
            "source": self.source,
        }


@dataclass
class CPUInfo:
    """Host CPU and memory."""

    model: str = "unknown"
    cores: int = 0
    physical_cores: int = 0
    ram_total_gb: float = 0.0
    ram_free_gb: float = 0.0
    platform: str = ""
    python: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "cores": self.cores,
            "physical_cores": self.physical_cores,
            "ram_total_gb": round(self.ram_total_gb, 3),
            "ram_free_gb": round(self.ram_free_gb, 3),
            "platform": self.platform,
            "python": self.python,
        }


@dataclass
class FLOPSResult:
    """Floating-point throughput."""

    tflops: float = 0.0
    method: str = "unavailable"
    seconds: float = 0.0
    iterations: int = 0
    matrix_size: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tflops": round(self.tflops, 3),
            "method": self.method,
            "seconds": round(self.seconds, 4),
            "iterations": self.iterations,
            "matrix_size": self.matrix_size,
        }


@dataclass
class LatencyStats:
    """Round-trip time distribution towards a peer."""

    samples: int = 0
    min_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    max_ms: float = 0.0
    mean_ms: float = 0.0
    failures: int = 0
    target: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "samples": self.samples,
            "min_ms": round(self.min_ms, 2),
            "p50_ms": round(self.p50_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "mean_ms": round(self.mean_ms, 2),
            "failures": self.failures,
        }


@dataclass
class BenchmarkReport:
    """Everything the worker measured about itself."""

    worker_id: str = ""
    ts: float = field(default_factory=now_s)
    gpu: GPUInfo | None = None
    cpu: CPUInfo = field(default_factory=CPUInfo)
    flops: FLOPSResult = field(default_factory=FLOPSResult)
    head_latency: LatencyStats = field(default_factory=LatencyStats)
    peer_latency: list[LatencyStats] = field(default_factory=list)
    tokens_per_sec: float = 0.0
    backend: str = ""
    model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "ts": self.ts,
            "gpu": self.gpu.to_dict() if self.gpu else None,
            "cpu": self.cpu.to_dict(),
            "flops": self.flops.to_dict(),
            "head_latency": self.head_latency.to_dict(),
            "peer_latency": [p.to_dict() for p in self.peer_latency],
            "tokens_per_sec": round(self.tokens_per_sec, 2),
            "backend": self.backend,
            "model": self.model,
        }


# --------------------------------------------------------------------------- #
# Hardware probes
# --------------------------------------------------------------------------- #
class HardwareProbe:
    """Detects the accelerators and the CPU of the host."""

    @staticmethod
    def detect_gpu() -> GPUInfo | None:
        """Detect an NVIDIA GPU (nvidia-smi first, torch second)."""
        info = HardwareProbe._detect_gpu_nvidia_smi()
        if info is not None:
            return info
        return HardwareProbe._detect_gpu_torch()

    @staticmethod
    def _detect_gpu_nvidia_smi() -> GPUInfo | None:
        binary = shutil.which("nvidia-smi")
        if not binary:
            return None
        try:
            completed = subprocess.run(
                [binary, f"--query-gpu={_NVIDIA_QUERY}", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            LOGGER.warning("nvidia-smi probe failed", extra={"hc_error": str(exc)})
            return None
        if completed.returncode != 0 or not completed.stdout.strip():
            return None
        line = completed.stdout.strip().splitlines()[0]
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 8:
            return None

        def to_float(value: str) -> float:
            try:
                return float(value)
            except ValueError:
                return 0.0

        def to_int(value: str) -> int:
            try:
                return int(float(value))
            except ValueError:
                return 0

        total_mb = to_float(parts[2])
        used_mb = to_float(parts[3])
        free_mb = to_float(parts[4])
        driver = ""
        try:
            drv = subprocess.run(
                [binary, "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            driver = drv.stdout.strip().splitlines()[0] if drv.stdout.strip() else ""
        except (OSError, subprocess.SubprocessError):
            driver = ""
        return GPUInfo(
            index=to_int(parts[0]),
            name=parts[1] or "NVIDIA GPU",
            vram_total_gb=round(total_mb / 1024.0, 3),
            vram_used_gb=round(used_mb / 1024.0, 3),
            vram_free_gb=round(free_mb / 1024.0, 3),
            compute_cap=parts[5],
            temperature_c=to_float(parts[6]),
            utilization_pct=to_float(parts[7]),
            driver=driver,
            source="nvidia-smi",
        )

    @staticmethod
    def _detect_gpu_torch() -> GPUInfo | None:
        try:
            import torch  # type: ignore[import-not-found]
        except Exception:  # noqa: BLE001 - torch is optional
            return None
        try:
            if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
                return None
            props = torch.cuda.get_device_properties(0)
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            return GPUInfo(
                index=0,
                name=props.name,
                vram_total_gb=round(total_bytes / (1024**3), 3),
                vram_used_gb=round((total_bytes - free_bytes) / (1024**3), 3),
                vram_free_gb=round(free_bytes / (1024**3), 3),
                compute_cap=f"{props.major}.{props.minor}",
                source="torch",
            )
        except Exception:  # noqa: BLE001 - never break startup on a probe error
            return None

    @staticmethod
    def detect_cpu() -> CPUInfo:
        """Detect CPU and RAM."""
        info = CPUInfo(platform=platform.platform(), python=platform.python_version())
        try:
            import psutil

            info.model = platform.processor() or platform.machine()
            info.cores = psutil.cpu_count(logical=True) or 0
            info.physical_cores = psutil.cpu_count(logical=False) or 0
            memory = psutil.virtual_memory()
            info.ram_total_gb = round(memory.total / (1024**3), 3)
            info.ram_free_gb = round(memory.available / (1024**3), 3)
        except Exception:  # noqa: BLE001 - psutil is optional
            info.model = platform.machine()
            info.cores = __import__("os").cpu_count() or 0
            info.physical_cores = info.cores
        return info


class FLOPSBenchmark:
    """Measures real floating-point throughput."""

    def __init__(self, *, matrix_size: int = 1024, iterations: int = 12) -> None:
        self.matrix_size = matrix_size
        self.iterations = iterations

    def run(self) -> FLOPSResult:
        """Run the benchmark (GPU via torch when possible, else numpy on the CPU)."""
        result = self._run_torch()
        if result.tflops > 0:
            return result
        return self._run_numpy()

    def _run_torch(self) -> FLOPSResult:
        try:
            import torch  # type: ignore[import-not-found]
        except Exception:  # noqa: BLE001
            return FLOPSResult()
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type == "cpu":
                return FLOPSResult()
            a = torch.randn(self.matrix_size, self.matrix_size, device=device, dtype=torch.float32)
            b = torch.randn(self.matrix_size, self.matrix_size, device=device, dtype=torch.float32)
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(self.iterations):
                a @ b
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            flops = 2.0 * self.matrix_size**3 * self.iterations
            return FLOPSResult(
                tflops=flops / elapsed / 1e12,
                method="torch-cuda-fp32",
                seconds=elapsed,
                iterations=self.iterations,
                matrix_size=self.matrix_size,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("torch FLOPS benchmark failed", extra={"hc_error": str(exc)})
            return FLOPSResult()

    def _run_numpy(self) -> FLOPSResult:
        try:
            import numpy as np
        except Exception:  # noqa: BLE001 - numpy is optional
            return FLOPSResult()
        try:
            a = np.random.rand(self.matrix_size, self.matrix_size).astype(np.float32)
            b = np.random.rand(self.matrix_size, self.matrix_size).astype(np.float32)
            started = time.perf_counter()
            for _ in range(self.iterations):
                _ = a @ b
            elapsed = time.perf_counter() - started
            flops = 2.0 * self.matrix_size**3 * self.iterations
            return FLOPSResult(
                tflops=flops / elapsed / 1e12,
                method="numpy-fp32",
                seconds=elapsed,
                iterations=self.iterations,
                matrix_size=self.matrix_size,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("numpy FLOPS benchmark failed", extra={"hc_error": str(exc)})
            return FLOPSResult()


# --------------------------------------------------------------------------- #
# Network probes
# --------------------------------------------------------------------------- #
class LatencyProbe:
    """Measures round-trip time towards an HTTP peer."""

    @staticmethod
    async def measure(
        url: str,
        *,
        samples: int = 5,
        timeout: float = 5.0,
        path: str = "/health",
    ) -> LatencyStats:
        """Probe ``url`` and return the latency distribution."""
        import httpx

        stats = LatencyStats(target=url)
        target = url.rstrip("/") + path
        timings: list[float] = []
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                for _ in range(max(1, samples)):
                    started = time.perf_counter()
                    try:
                        response = await client.get(target)
                        elapsed = (time.perf_counter() - started) * 1000.0
                        if response.status_code < 500:
                            timings.append(elapsed)
                        else:
                            stats.failures += 1
                    except httpx.HTTPError:
                        stats.failures += 1
                    await asyncio.sleep(0.02)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("latency probe failed", extra={"hc_target": url, "hc_error": str(exc)})
            stats.failures += max(1, samples)
        stats.samples = len(timings)
        if timings:
            ordered = sorted(timings)
            stats.min_ms = ordered[0]
            stats.max_ms = ordered[-1]
            stats.mean_ms = sum(ordered) / len(ordered)
            stats.p50_ms = ordered[max(0, int(len(ordered) * 0.5) - 1)]
            stats.p95_ms = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
        return stats


class ThroughputProbe:
    """Measures the decode throughput of the resident model."""

    @staticmethod
    async def measure(engine: Any, *, prompt: str = "Benchmark: write a short paragraph about GPUs.") -> float:
        """Generate a short completion and report tokens/second."""
        if engine is None or not getattr(engine, "loaded", False):
            return 0.0
        from worker.inference_engine import GenerationParams

        started = time.perf_counter()
        produced = 0
        try:
            for piece in engine.stream(prompt, GenerationParams(max_tokens=96, temperature=0.2)):
                produced += len(piece)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("throughput probe failed", extra={"hc_error": str(exc)})
            return 0.0
        elapsed = time.perf_counter() - started
        if elapsed <= 0:
            return 0.0
        # Rough token estimate: ~4 characters per token for English text.
        return round((produced / 4.0) / elapsed, 2)


# --------------------------------------------------------------------------- #
# Aggregate benchmark
# --------------------------------------------------------------------------- #
class Benchmarker:
    """Runs the full self-benchmark and produces a report."""

    def __init__(
        self,
        *,
        worker_id: str = "",
        head_url: str = "",
        peer_urls: list[str] | None = None,
        flops_matrix: int = 1024,
    ) -> None:
        self.worker_id = worker_id
        self.head_url = head_url
        self.peer_urls = list(peer_urls or [])
        self.flops_matrix = flops_matrix

    async def run(self, engine: Any = None) -> BenchmarkReport:
        """Execute every probe and return the aggregated report."""
        report = BenchmarkReport(worker_id=self.worker_id)
        report.gpu = HardwareProbe.detect_gpu()
        report.cpu = HardwareProbe.detect_cpu()
        report.flops = FLOPSBenchmark(matrix_size=self.flops_matrix).run()
        if self.head_url:
            report.head_latency = await LatencyProbe.measure(self.head_url, samples=5)
        for peer in self.peer_urls:
            report.peer_latency.append(await LatencyProbe.measure(peer, samples=3))
        if engine is not None:
            report.tokens_per_sec = await ThroughputProbe.measure(engine)
            info = engine.info()
            report.backend = info.backend
            report.model = info.model
        return report


def capabilities_from_report(
    report: BenchmarkReport,
    *,
    provider: str,
    region: str,
    families: list[str],
    context: int = 8192,
    max_concurrency: int = 2,
    device: str | None = None,
    model_params_b: float = 0.0,
) -> Any:
    """Convert a benchmark report into the wire-protocol capability payload."""
    from core.protocol import Capabilities

    gpu = report.gpu
    if device is None:
        device = "cuda" if gpu else "cpu"
    vram_total = gpu.vram_total_gb if gpu else 0.0
    vram_free = gpu.vram_free_gb if gpu else 0.0
    tflops = report.flops.tflops if report.flops.tflops > 0 else report.flops.tflops
    if gpu and tflops <= 0:
        # No FP32 benchmark available (e.g. no torch/numpy): estimate from the
        # accelerator class so the scheduler still has a usable number.
        name = gpu.name.lower()
        if "h100" in name or "h200" in name:
            tflops = 700.0
        elif "a100" in name:
            tflops = 312.0
        elif "a10" in name:
            tflops = 125.0
        elif "t4" in name:
            tflops = 65.0
        elif "l4" in name:
            tflops = 121.0
        elif "v100" in name:
            tflops = 125.0
        elif "p100" in name:
            tflops = 18.0
        else:
            tflops = 30.0
    return Capabilities(
        vram_total_gb=vram_total,
        vram_free_gb=vram_free,
        compute_tflops=tflops,
        cpu_cores=report.cpu.cores,
        ram_gb=report.cpu.ram_total_gb,
        provider=provider,
        region=region,
        backend=report.backend or "llama.cpp",
        device=device,
        supported_families=families,
        supported_context=context,
        max_concurrency=max_concurrency,
        mesh_latency_ms=report.head_latency.p50_ms,
        extra={
            "gpu": report.gpu.to_dict() if report.gpu else None,
            "cpu": report.cpu.to_dict(),
            "flops": report.flops.to_dict(),
            "head_latency": report.head_latency.to_dict(),
            # The head node needs the throughput *and* the size it was
            # measured with: ``tokens/s * B`` is the node's hardware factor
            # and stays valid after a hot-swap to a different model.
            "throughput_tps": round(report.tokens_per_sec, 3),
            "benchmark_params_b": model_params_b,
        },
    )


__all__ = [
    "GPUInfo",
    "CPUInfo",
    "FLOPSResult",
    "LatencyStats",
    "BenchmarkReport",
    "HardwareProbe",
    "FLOPSBenchmark",
    "LatencyProbe",
    "ThroughputProbe",
    "Benchmarker",
    "capabilities_from_report",
]
