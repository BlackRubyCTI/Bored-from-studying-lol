"""The worker's inference engine.

Three interchangeable backends, one interface:

``llama.cpp``   the production path -- a locally hosted GGUF on the GPU.
``upstream``    any OpenAI-compatible server (vLLM, Ollama, LM Studio, TGI)
                reached over HTTP; useful when the node already runs a server.
``simulator``   a deterministic, dependency-free generator.  It exists so the
                worker protocol can be exercised end-to-end on machines without
                a GPU or a GGUF (CI, laptops, the head node's dev mode).  It is
                always reported as ``backend="simulator"`` so it can never be
                mistaken for a real model.

The engine also owns model *hot-swapping*: loading a new model builds the new
backend first and only then unloads the old one, so a failed load never takes
the worker offline.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from core.errors import ModelNotReadyError, WorkerUnreachableError
from core.ids import now_s
from core.logging_setup import get_logger

LOGGER = get_logger("worker.inference_engine")

BACKEND_LLAMA = "llama.cpp"
BACKEND_UPSTREAM = "upstream"
BACKEND_SIMULATOR = "simulator"


@dataclass(slots=True)
class GenerationParams:
    """Sampling parameters for one generation."""

    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 512
    stop: list[str] = field(default_factory=list)
    seed: int | None = None
    stream: bool = True


@dataclass
class ModelRuntime:
    """The model a backend should serve."""

    name: str = "unknown"
    path: str = ""
    repo: str = ""
    filename: str = ""
    family: str = "unknown"
    params_b: float = 7.0
    context: int = 8192
    gpu_layers: int = -1
    threads: int = 0
    sha256: str = ""

    def resolved_name(self) -> str:
        if self.name and self.name != "unknown":
            return self.name
        if self.filename:
            return self.filename.replace(".gguf", "")
        if self.path:
            return Path(self.path).stem
        return "unknown"


@dataclass
class EngineInfo:
    """Public description of the engine's current state."""

    backend: str = BACKEND_SIMULATOR
    model: str = "none"
    loaded: bool = False
    context: int = 0
    gpu_layers: int = 0
    vram_used_gb: float = 0.0
    model_path: str = ""
    loaded_at: float = 0.0
    tokens_generated: int = 0
    generations: int = 0
    last_tokens_per_sec: float = 0.0
    last_ttft_ms: float = 0.0
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model,
            "loaded": self.loaded,
            "context": self.context,
            "gpu_layers": self.gpu_layers,
            "vram_used_gb": round(self.vram_used_gb, 3),
            "model_path": self.model_path,
            "loaded_at": self.loaded_at,
            "tokens_generated": self.tokens_generated,
            "generations": self.generations,
            "last_tokens_per_sec": round(self.last_tokens_per_sec, 2),
            "last_ttft_ms": round(self.last_ttft_ms, 2),
            "detail": self.detail,
        }


class EngineBackend(Protocol):
    """Interface every inference backend implements."""

    name: str

    def load(self, runtime: ModelRuntime) -> None: ...

    def unload(self) -> None: ...

    def stream(self, prompt: str, params: GenerationParams) -> Iterator[str]: ...

    def info(self) -> EngineInfo: ...


# --------------------------------------------------------------------------- #
# Backend: llama.cpp
# --------------------------------------------------------------------------- #
class LlamaCppBackend:
    """Local GGUF inference through ``llama-cpp-python``."""

    name = BACKEND_LLAMA

    def __init__(self, runtime: ModelRuntime) -> None:
        self.runtime = runtime
        self._llama: Any = None
        self._info = EngineInfo(backend=self.name, model=runtime.resolved_name(), model_path=runtime.path)

    def load(self, runtime: ModelRuntime) -> None:
        """Instantiate the model (raises when the file or the library is missing)."""
        try:
            from llama_cpp import Llama  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on the host
            raise ModelNotReadyError(
                "llama-cpp-python is not installed on this worker",
                details={"hint": "pip install llama-cpp-python"},
            ) from exc
        path = Path(runtime.path).expanduser()
        if not path.is_file():
            raise ModelNotReadyError(
                f"model file not found: {path}",
                details={"model": runtime.resolved_name()},
            )
        kwargs: dict[str, Any] = {
            "model_path": str(path),
            "n_ctx": runtime.context,
            "n_gpu_layers": runtime.gpu_layers,
            "verbose": False,
        }
        if runtime.threads:
            kwargs["n_threads"] = runtime.threads
        started = time.perf_counter()
        self._llama = Llama(**kwargs)
        load_ms = (time.perf_counter() - started) * 1000.0
        self._info = EngineInfo(
            backend=self.name,
            model=runtime.resolved_name(),
            loaded=True,
            context=runtime.context,
            gpu_layers=runtime.gpu_layers,
            vram_used_gb=self._vram_used(),
            model_path=str(path),
            loaded_at=now_s(),
            detail=f"loaded in {load_ms:.0f} ms",
        )
        LOGGER.info(
            "llama.cpp model loaded",
            extra={"hc_model": self._info.model, "hc_ctx": runtime.context, "hc_gpu_layers": runtime.gpu_layers},
        )

    def unload(self) -> None:
        self._llama = None
        self._info.loaded = False

    def stream(self, prompt: str, params: GenerationParams) -> Iterator[str]:
        """Yield decoded fragments."""
        if self._llama is None:
            raise ModelNotReadyError("no model is loaded")
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "stream": True,
        }
        if params.stop:
            kwargs["stop"] = params.stop
        if params.seed is not None:
            kwargs["seed"] = params.seed
        for chunk in self._llama.create_completion(**kwargs):
            choices = chunk.get("choices") or []
            if not choices:
                continue
            text = choices[0].get("text") or ""
            if text:
                yield text

    def info(self) -> EngineInfo:
        return self._info

    def _vram_used(self) -> float:
        try:
            from worker.telemetry import HardwareProbe  # local import to avoid a cycle

            gpu = HardwareProbe.detect_gpu()
            return gpu.vram_used_gb if gpu else 0.0
        except Exception:  # noqa: BLE001 - telemetry must never break loading
            return 0.0


# --------------------------------------------------------------------------- #
# Backend: OpenAI-compatible upstream
# --------------------------------------------------------------------------- #
class UpstreamBackend:
    """Proxy an OpenAI-compatible inference server (vLLM / Ollama / TGI)."""

    name = BACKEND_UPSTREAM

    def __init__(self, runtime: ModelRuntime, *, base_url: str, model: str, api_key: str = "") -> None:
        self.runtime = runtime
        self.base_url = base_url.rstrip("/")
        self.model = model or runtime.resolved_name()
        self.api_key = api_key
        self._info = EngineInfo(backend=self.name, model=self.model, model_path=runtime.path)

    def load(self, runtime: ModelRuntime) -> None:
        if not self.base_url:
            raise ModelNotReadyError("upstream backend needs an upstream URL")
        self._info = EngineInfo(
            backend=self.name,
            model=self.model,
            loaded=True,
            context=runtime.context,
            model_path=runtime.path,
            loaded_at=now_s(),
            detail=f"upstream {self.base_url}",
        )

    def unload(self) -> None:
        self._info.loaded = False

    def stream(self, prompt: str, params: GenerationParams) -> Iterator[str]:
        import httpx

        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        body = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "stream": True,
        }
        if params.stop:
            body["stop"] = params.stop
        try:
            with httpx.Client(timeout=300.0) as client:
                with client.stream("POST", f"{self.base_url}/completions", json=body, headers=headers) as response:
                    if response.status_code >= 400:
                        detail = response.read().decode("utf-8", "replace")[:200]
                        raise WorkerUnreachableError(
                            f"upstream returned HTTP {response.status_code}: {detail}",
                            details={"upstream": self.base_url},
                        )
                    for line in response.iter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload in ("", "[DONE]"):
                            if payload == "[DONE]":
                                break
                            continue
                        try:
                            parsed = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        for choice in parsed.get("choices", []):
                            text = choice.get("text") or (choice.get("delta") or {}).get("content") or ""
                            if text:
                                yield text
        except httpx.HTTPError as exc:
            raise WorkerUnreachableError(
                f"upstream unreachable: {exc}",
                details={"upstream": self.base_url},
            ) from exc

    def info(self) -> EngineInfo:
        return self._info


# --------------------------------------------------------------------------- #
# Backend: deterministic simulator
# --------------------------------------------------------------------------- #
_TOOL_DIRECTIVE = "[[TOOL_PROTOCOL]]"
_CALL_MARKER = re.compile(r"\[\[call:([a-zA-Z_][a-zA-Z0-9_]*)\]\]\s*(\{.*?\})", re.DOTALL)


class SimulatorBackend:
    """Deterministic generator used when no real model is available.

    It is *not* a language model: it produces a structured, reproducible
    answer from the prompt so that every downstream system (the worker
    protocol, the orchestrator, the tool loop, the dashboard) can be exercised
    deterministically.  ``EngineInfo.backend`` always reports ``simulator``.
    """

    name = BACKEND_SIMULATOR

    def __init__(self, runtime: ModelRuntime) -> None:
        self.runtime = runtime
        self._info = EngineInfo(backend=self.name, model=runtime.resolved_name())

    def load(self, runtime: ModelRuntime) -> None:
        self._info = EngineInfo(
            backend=self.name,
            model=runtime.resolved_name(),
            loaded=True,
            context=runtime.context,
            model_path=runtime.path,
            loaded_at=now_s(),
            detail="deterministic simulator (no GGUF / no llama.cpp on this host)",
        )

    def unload(self) -> None:
        self._info.loaded = False

    # -- helpers -------------------------------------------------------- #
    @staticmethod
    def _extract(prompt: str, role: str) -> str:
        """Pull one role's content out of a rendered prompt.

        Supports the ChatML template (``<|im_start|>role``), the Llama-3
        template (``<|start_header_id|>role``) and the plain template
        (``Role: ...``).  Returns an empty string when the role is absent --
        never a fragment of another role.
        """
        patterns = (
            rf"<\|im_start\|>{re.escape(role)}\n(.*?)<\|im_end\|>",
            rf"<\|start_header_id\|>{re.escape(role)}<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>",
        )
        for pattern in patterns:
            matches = re.findall(pattern, prompt, re.DOTALL)
            if matches:
                return matches[-1].strip()
        plain = re.findall(
            rf"^{re.escape(role.capitalize())}:\s*(.*?)(?=^\w+:|\Z)",
            prompt,
            re.DOTALL | re.MULTILINE,
        )
        if plain:
            return plain[-1].strip()
        return ""

    def _answer_for(self, prompt: str) -> str:
        """Produce a deterministic, useful-shaped answer."""
        system = self._extract(prompt, "system")
        user = self._extract(prompt, "user")

        # Tool protocol: honour an explicit call marker, otherwise answer.
        if _TOOL_DIRECTIVE in system or _TOOL_DIRECTIVE in prompt:
            tool_result = self._extract(prompt, "tool")
            if tool_result:
                # The tool already ran: answer from its output instead of
                # calling the same tool again (mirrors a real agent loop).
                return json.dumps({"final": self._tool_narrative(user, tool_result)})
            match = _CALL_MARKER.search(user)
            if match:
                tool = match.group(1)
                raw_args = match.group(2)
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = {"raw": raw_args}
                return json.dumps({"tool": tool, "args": args})
            return json.dumps({"final": self._narrative(user)})

        return self._narrative(user)

    def _tool_narrative(self, user: str, tool_result: str) -> str:
        """Build the final answer once a tool has returned its output."""
        digest = hashlib.sha256(f"{user}|{tool_result}".encode()).hexdigest()[:8]
        excerpt = " ".join(tool_result.strip().split())[:600]
        return (
            f"[simulator:{digest}] The tool returned: {excerpt} "
            f"This answer was synthesised from the live tool output for the request "
            f"\"{' '.join(user.split())[:120]}\"."
        )

    def _narrative(self, user: str) -> str:
        digest = hashlib.sha256(user.encode("utf-8")).hexdigest()[:8]
        topic = user.strip().replace("\n", " ")[:180] or "(empty prompt)"
        words = topic.split(" ")
        return (
            f"[simulator:{digest}] Analysed request: {topic} "
            f"({len(words)} words). The deterministic backend returns a reproducible "
            f"answer so the pipeline can be verified without a GPU; request hash {digest}. "
            f"Swap in a real GGUF via bootstrap.sh to get model-grade output."
        )

    # -- backend API ---------------------------------------------------- #
    def stream(self, prompt: str, params: GenerationParams) -> Iterator[str]:
        text = self._answer_for(prompt)
        for stop in params.stop:
            index = text.find(stop)
            if index != -1:
                text = text[:index]
        budget = max(1, params.max_tokens)
        tokens = text.split(" ")
        for index in range(0, len(tokens), 3):
            piece = " ".join(tokens[index : index + 3])
            if index:
                piece = " " + piece
            if index // 3 >= budget:
                break
            yield piece
            time.sleep(0.001)

    def info(self) -> EngineInfo:
        return self._info


# --------------------------------------------------------------------------- #
# Facade
# --------------------------------------------------------------------------- #
def llama_cpp_available() -> bool:
    """True when ``llama-cpp-python`` can be imported."""
    import importlib.util

    return importlib.util.find_spec("llama_cpp") is not None


class InferenceEngine:
    """Owns the active backend and performs hot-swaps."""

    def __init__(
        self,
        *,
        backend: str = "auto",
        upstream_url: str = "",
        upstream_model: str = "",
        upstream_api_key: str = "",
        default_runtime: ModelRuntime | None = None,
    ) -> None:
        self.requested_backend = backend
        self.upstream_url = upstream_url
        self.upstream_model = upstream_model
        self.upstream_api_key = upstream_api_key
        self.runtime = default_runtime or ModelRuntime()
        self._backend: EngineBackend | None = None
        self._lock = __import__("threading").RLock()
        self._tokens_generated = 0
        self._generations = 0

    # ------------------------------------------------------------------ #
    def available_backends(self) -> list[str]:
        """Backends that could be used on this host."""
        available = [BACKEND_SIMULATOR]
        if llama_cpp_available():
            available.insert(0, BACKEND_LLAMA)
        if self.upstream_url:
            available.insert(0, BACKEND_UPSTREAM)
        return available

    def _build_backend(self, runtime: ModelRuntime) -> EngineBackend:
        choice = self.requested_backend
        if choice == "auto":
            if self.upstream_url:
                choice = BACKEND_UPSTREAM
            elif llama_cpp_available() and runtime.path and Path(runtime.path).is_file():
                choice = BACKEND_LLAMA
            else:
                choice = BACKEND_SIMULATOR
        if choice == BACKEND_UPSTREAM:
            return UpstreamBackend(
                runtime,
                base_url=self.upstream_url,
                model=self.upstream_model,
                api_key=self.upstream_api_key,
            )
        if choice == BACKEND_LLAMA:
            return LlamaCppBackend(runtime)
        return SimulatorBackend(runtime)

    # ------------------------------------------------------------------ #
    def load(self, runtime: ModelRuntime | None = None, **overrides: Any) -> EngineInfo:
        """Load (or hot-swap to) a model.

        The new backend is built *before* the old one is released, so a failed
        load leaves the worker serving the previous model.
        """
        with self._lock:
            target = runtime or self.runtime
            for key, value in overrides.items():
                if value is not None and hasattr(target, key):
                    setattr(target, key, value)
            self.runtime = target
            backend = self._build_backend(target)
            backend.load(target)
            previous = self._backend
            self._backend = backend
            if previous is not None:
                try:
                    previous.unload()
                except Exception:  # noqa: BLE001 - unloading must never raise
                    LOGGER.warning("failed to unload previous backend")
            return backend.info()

    def unload(self) -> None:
        with self._lock:
            if self._backend is not None:
                self._backend.unload()
                self._backend = None

    @property
    def loaded(self) -> bool:
        return self._backend is not None and self._backend.info().loaded

    # ------------------------------------------------------------------ #
    def stream(self, prompt: str, params: GenerationParams | None = None) -> Iterator[str]:
        """Stream generated text, measuring TTFT and throughput."""
        if self._backend is None:
            raise ModelNotReadyError("no model is loaded on this worker")
        resolved = params or GenerationParams()
        started = time.perf_counter()
        first_at: float | None = None
        produced = 0
        for piece in self._backend.stream(prompt, resolved):
            if not piece:
                continue
            if first_at is None:
                first_at = time.perf_counter()
            produced += 1
            yield piece
        elapsed = time.perf_counter() - started
        info = self._backend.info()
        self._tokens_generated += produced
        self._generations += 1
        info.tokens_generated = self._tokens_generated
        info.generations = self._generations
        if elapsed > 0:
            info.last_tokens_per_sec = produced / elapsed
        if first_at is not None:
            info.last_ttft_ms = (first_at - started) * 1000.0

    def generate(self, prompt: str, params: GenerationParams | None = None) -> str:
        """Blocking generation (returns the whole text)."""
        return "".join(self.stream(prompt, params))

    def info(self) -> EngineInfo:
        if self._backend is None:
            return EngineInfo(backend=self.requested_backend, model="none", loaded=False)
        return self._backend.info()


__all__ = [
    "BACKEND_LLAMA",
    "BACKEND_UPSTREAM",
    "BACKEND_SIMULATOR",
    "EngineInfo",
    "EngineBackend",
    "GenerationParams",
    "InferenceEngine",
    "LlamaCppBackend",
    "ModelRuntime",
    "SimulatorBackend",
    "UpstreamBackend",
    "llama_cpp_available",
]
