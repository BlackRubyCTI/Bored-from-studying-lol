"""Default model catalogue for the cluster.

The catalogue is the contract between the *dynamic VRAM auto-scaler* (which
decides how large a model the live pool can host) and the *scheduler* (which
decides which worker can serve a request right now).

``vram_required_gb`` values are conservative, real-world figures for the listed
quantisation (weights + a working KV-cache headroom).  ``intelligence`` is a
normalised 0..1 capability score used to break ties when several workers can
serve a request equally fast.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.ids import now_s
from core.protocol import ModelCatalog, ModelSpec

DEFAULT_MODELS: tuple[ModelSpec, ...] = (
    # ---------------- tiny / edge of the pool ---------------- #
    ModelSpec(
        name="Qwen2.5-0.5B-Instruct-GGUF",
        family="qwen2",
        params_b=0.5,
        quant="Q8_0",
        vram_required_gb=1.2,
        intelligence=0.18,
        context=32768,
        repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        filename="qwen2.5-0.5b-instruct-q8_0.gguf",
        license="apache-2.0",
        tags=["fast", "speculative-draft", "cpu"],
    ),
    ModelSpec(
        name="Qwen2.5-1.5B-Instruct-GGUF",
        family="qwen2",
        params_b=1.5,
        quant="Q4_K_M",
        vram_required_gb=2.0,
        intelligence=0.28,
        context=32768,
        repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
        license="apache-2.0",
        tags=["fast", "cpu", "speculative-draft"],
    ),
    # ---------------- main working set ---------------- #
    ModelSpec(
        name="Qwen2.5-7B-Instruct-GGUF",
        family="qwen2",
        params_b=7.0,
        quant="Q4_K_M",
        vram_required_gb=5.5,
        intelligence=0.55,
        context=32768,
        repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
        filename="qwen2.5-7b-instruct-q4_k_m.gguf",
        license="apache-2.0",
        tags=["balanced"],
    ),
    ModelSpec(
        name="Qwen2.5-14B-Instruct-GGUF",
        family="qwen2",
        params_b=14.0,
        quant="Q4_K_M",
        vram_required_gb=9.5,
        intelligence=0.68,
        context=32768,
        repo_id="Qwen/Qwen2.5-14B-Instruct-GGUF",
        filename="qwen2.5-14b-instruct-q4_k_m.gguf",
        license="apache-2.0",
        tags=["deep", "balanced"],
    ),
    ModelSpec(
        name="Qwen2.5-32B-Instruct-GGUF",
        family="qwen2",
        params_b=32.0,
        quant="Q4_K_M",
        vram_required_gb=20.0,
        intelligence=0.79,
        context=32768,
        repo_id="Qwen/Qwen2.5-32B-Instruct-GGUF",
        filename="qwen2.5-32b-instruct-q4_k_m.gguf",
        license="apache-2.0",
        tags=["deep", "cluster"],
    ),
    ModelSpec(
        name="Qwen2.5-72B-Instruct-GGUF",
        family="qwen2",
        params_b=72.0,
        quant="Q4_K_M",
        vram_required_gb=42.0,
        intelligence=0.88,
        context=32768,
        repo_id="Qwen/Qwen2.5-72B-Instruct-GGUF",
        filename="qwen2.5-72b-instruct-q4_k_m.gguf",
        license="apache-2.0",
        tags=["frontier", "cluster"],
    ),
    # ---------------- reasoning specialists ---------------- #
    ModelSpec(
        name="DeepSeek-R1-Distill-Qwen-7B-GGUF",
        family="qwen2",
        params_b=7.0,
        quant="Q4_K_M",
        vram_required_gb=5.8,
        intelligence=0.6,
        context=32768,
        repo_id="bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF",
        filename="DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
        license="mit",
        tags=["reasoning"],
    ),
    ModelSpec(
        name="DeepSeek-R1-Distill-Qwen-14B-GGUF",
        family="qwen2",
        params_b=14.0,
        quant="Q4_K_M",
        vram_required_gb=9.8,
        intelligence=0.72,
        context=32768,
        repo_id="bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF",
        filename="DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
        license="mit",
        tags=["reasoning", "deep"],
    ),
    ModelSpec(
        name="DeepSeek-R1-Distill-Llama-70B-GGUF",
        family="llama",
        params_b=70.0,
        quant="Q4_K_M",
        vram_required_gb=41.0,
        intelligence=0.9,
        context=32768,
        repo_id="bartowski/DeepSeek-R1-Distill-Llama-70B-GGUF",
        filename="DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf",
        license="mit",
        tags=["reasoning", "frontier", "cluster"],
    ),
    ModelSpec(
        name="Llama-3.3-70B-Instruct-GGUF",
        family="llama",
        params_b=70.0,
        quant="Q4_K_M",
        vram_required_gb=42.0,
        intelligence=0.86,
        context=131072,
        repo_id="bartowski/Llama-3.3-70B-Instruct-GGUF",
        filename="Llama-3.3-70B-Instruct-Q4_K_M.gguf",
        license="llama3.3",
        tags=["frontier", "cluster"],
    ),
    ModelSpec(
        name="Llama-3.1-8B-Instruct-GGUF",
        family="llama",
        params_b=8.0,
        quant="Q4_K_M",
        vram_required_gb=6.0,
        intelligence=0.5,
        context=131072,
        repo_id="bartowski/Llama-3.1-8B-Instruct-GGUF",
        filename="Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        license="llama3.1",
        tags=["fast", "cluster"],
    ),
    ModelSpec(
        name="gemma-2-9b-it-GGUF",
        family="gemma2",
        params_b=9.0,
        quant="Q4_K_M",
        vram_required_gb=6.8,
        intelligence=0.52,
        context=8192,
        repo_id="bartowski/gemma-2-9b-it-GGUF",
        filename="gemma-2-9b-it-Q4_K_M.gguf",
        license="gemma",
        tags=["balanced"],
    ),
    ModelSpec(
        name="Phi-4-mini-instruct-GGUF",
        family="phi3",
        params_b=3.8,
        quant="Q4_K_M",
        vram_required_gb=3.4,
        intelligence=0.46,
        context=16384,
        repo_id="bartowski/phi-4-mini-instruct-GGUF",
        filename="Phi-4-mini-instruct-Q4_K_M.gguf",
        license="mit",
        tags=["fast", "cpu"],
    ),
)

#: Models that are excellent "draft" models for speculative decoding.
DRAFT_CANDIDATES: tuple[str, ...] = (
    "Qwen2.5-0.5B-Instruct-GGUF",
    "Qwen2.5-1.5B-Instruct-GGUF",
    "Phi-4-mini-instruct-GGUF",
)


def default_catalog() -> ModelCatalog:
    """A fresh catalogue containing the built-in model table."""
    return ModelCatalog(models=list(DEFAULT_MODELS), updated_at=now_s())


def load_catalog(path: str | Path) -> ModelCatalog:
    """Load a catalogue from a JSON file (falls back to the default table)."""
    catalog_path = Path(path).expanduser()
    if not catalog_path.is_file():
        return default_catalog()
    try:
        raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default_catalog()
    entries = raw.get("models", raw) if isinstance(raw, dict) else raw
    models: list[ModelSpec] = []
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            try:
                models.append(ModelSpec(**entry))
            except Exception:  # noqa: BLE001 - a bad entry must not kill the cluster
                continue
    if not models:
        return default_catalog()
    return ModelCatalog(models=models, updated_at=now_s(), active_model=raw.get("active_model") if isinstance(raw, dict) else None)


def save_catalog(catalog: ModelCatalog, path: str | Path) -> None:
    """Persist a catalogue to disk."""
    catalog_path = Path(path).expanduser()
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "active_model": catalog.active_model,
        "updated_at": catalog.updated_at,
        "models": [m.model_dump() for m in catalog.models],
    }
    catalog_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def find_model(catalog: ModelCatalog, name: str | None) -> ModelSpec | None:
    """Case-insensitive lookup of a model by name."""
    if not name:
        return None
    lowered = name.strip().lower()
    for spec in catalog.models:
        if spec.name.lower() == lowered:
            return spec
    for spec in catalog.models:
        if lowered in spec.name.lower() or lowered in spec.filename.lower():
            return spec
    return None


def best_models_for_vram(catalog: ModelCatalog, vram_gb: float, *, limit: int = 5) -> list[ModelSpec]:
    """Highest-intelligence models that fit inside ``vram_gb``."""
    fits = [m for m in catalog.models if m.vram_required_gb <= vram_gb]
    fits.sort(key=lambda m: (m.intelligence, m.params_b), reverse=True)
    return fits[:limit]


__all__ = [
    "DEFAULT_MODELS",
    "DRAFT_CANDIDATES",
    "default_catalog",
    "load_catalog",
    "save_catalog",
    "find_model",
    "best_models_for_vram",
]
