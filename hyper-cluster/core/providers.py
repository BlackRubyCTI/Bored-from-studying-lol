"""Canonical registry of every external provider Hyper-Cluster can drive.

Two provider *kinds* exist:

``inference``
    Sub-second, pay-per-token (but free-tier) APIs used as the edge tier for
    speculative drafting and as a zero-downtime fallback brain.

``compute``
    Headless compute pools (Kaggle, Hugging Face Spaces, GitHub Actions,
    Lightning AI, Oracle Cloud, Android Termux) that provide real GPUs and
    therefore join the *cluster* tier.

Keeping every endpoint, auth style, default model and rate limit in one table
means the credential vault, the edge bridge and the auto-spawner all agree on
the truth instead of hard-coding URLs in five different places.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

INFERENCE = "inference"
COMPUTE = "compute"


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    """Static description of one external provider."""

    name: str
    kind: str
    base_url: str
    api_key_env: str
    default_model: str = ""
    models: tuple[str, ...] = ()
    chat_path: str = "/chat/completions"
    health_path: str = "/models"
    auth_header: str = "Authorization"
    auth_prefix: str = "Bearer "
    rpm_limit: int = 30
    tpm_limit: int = 100_000
    supports_streaming: bool = True
    openai_compatible: bool = True
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def header_for(self, token: str) -> dict[str, str]:
        """Build the authenticated header mapping for this provider."""
        if not token:
            return {}
        if self.auth_header.lower() == "x-api-key":
            return {self.auth_header: token}
        return {self.auth_header: f"{self.auth_prefix}{token}"}

    def health_url(self) -> str:
        return f"{self.base_url.rstrip('/')}{self.health_path}"


PROVIDERS: dict[str, ProviderSpec] = {
    # ----------------------------------------------------------------- #
    # Inference (edge tier)
    # ----------------------------------------------------------------- #
    "groq": ProviderSpec(
        name="groq",
        kind=INFERENCE,
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        default_model="llama-3.3-70b-versatile",
        models=(
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "qwen-2.5-32b",
            "deepseek-r1-distill-llama-70b",
            "gemma2-9b-it",
        ),
        rpm_limit=30,
        tpm_limit=12_000,
        notes="LPU inference; ~500 tok/s on 70B. Primary speculative drafter.",
    ),
    "cerebras": ProviderSpec(
        name="cerebras",
        kind=INFERENCE,
        base_url="https://api.cerebras.ai/v1",
        api_key_env="CEREBRAS_API_KEY",
        default_model="llama-3.3-70b",
        models=("llama-3.3-70b", "llama3.1-8b", "qwen-3-32b", "qwen-2.5-32b"),
        rpm_limit=60,
        tpm_limit=60_000,
        notes="Wafer-scale; second drafter.",
    ),
    "sambanova": ProviderSpec(
        name="sambanova",
        kind=INFERENCE,
        base_url="https://api.sambanova.ai/v1",
        api_key_env="SAMBANOVA_API_KEY",
        default_model="Meta-Llama-3.3-70B-Instruct",
        models=("Meta-Llama-3.3-70B-Instruct", "Meta-Llama-3.1-8B-Instruct", "DeepSeek-R1-Distill-Llama-70B"),
        rpm_limit=20,
        tpm_limit=40_000,
        notes="Redundant topology across clouds.",
    ),
    "openrouter": ProviderSpec(
        name="openrouter",
        kind=INFERENCE,
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        default_model="deepseek/deepseek-chat-v3.1",
        models=(
            "deepseek/deepseek-chat-v3.1",
            "qwen/qwen-2.5-72b-instruct",
            "meta-llama/llama-3.3-70b-instruct",
            "mistralai/mistral-large",
        ),
        rpm_limit=20,
        tpm_limit=80_000,
        notes="Aggregator: one key reaches dozens of upstream models.",
    ),
    "google_ai_studio": ProviderSpec(
        name="google_ai_studio",
        kind=INFERENCE,
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key_env="GOOGLE_AI_STUDIO_KEY",
        default_model="gemini-2.0-flash",
        models=("gemini-2.0-flash", "gemini-1.5-pro", "gemini-2.5-flash"),
        chat_path="/models/{model}:streamGenerateContent",
        health_path="/models",
        auth_header="x-goog-api-key",
        auth_prefix="",
        openai_compatible=False,
        rpm_limit=15,
        tpm_limit=1_000_000,
        notes="Very large free RPM window; good for bursts.",
    ),
    "fireworks": ProviderSpec(
        name="fireworks",
        kind=INFERENCE,
        base_url="https://api.fireworks.ai/inference/v1",
        api_key_env="FIREWORKS_API_KEY",
        default_model="accounts/fireworks/models/llama-v3p3-70b-instruct",
        models=(
            "accounts/fireworks/models/llama-v3p3-70b-instruct",
            "accounts/fireworks/models/qwen2p5-72b-instruct",
        ),
        rpm_limit=30,
        tpm_limit=100_000,
        notes="Optional extra edge node.",
    ),
    # ----------------------------------------------------------------- #
    # Compute (cluster tier)
    # ----------------------------------------------------------------- #
    "kaggle": ProviderSpec(
        name="kaggle",
        kind=COMPUTE,
        base_url="https://www.kaggle.com/api/v1",
        api_key_env="KAGGLE_USERNAME",
        default_model="Qwen2.5-32B-Instruct-AWQ",
        rpm_limit=5,
        notes="T4x2 / P100 kernels, 30h/week quota, no browser captcha.",
        extra={"secondary_env": "KAGGLE_KEY"},
    ),
    "hf_spaces": ProviderSpec(
        name="hf_spaces",
        kind=COMPUTE,
        base_url="https://huggingface.co/api/spaces",
        api_key_env="HF_TOKEN",
        default_model="Qwen2.5-7B-Instruct",
        rpm_limit=20,
        notes="ZeroGPU Spaces; dynamic container spin-up/down.",
    ),
    "github_actions": ProviderSpec(
        name="github_actions",
        kind=COMPUTE,
        base_url="https://api.github.com",
        api_key_env="GITHUB_TOKEN",
        default_model="Qwen2.5-7B-Instruct",
        rpm_limit=30,
        notes="Matrix runners; billed minutes burn fast, use for short bursts.",
    ),
    "lightning": ProviderSpec(
        name="lightning",
        kind=COMPUTE,
        base_url="https://api.lightning.ai",
        api_key_env="LIGHTNING_API_TOKEN",
        default_model="Qwen2.5-7B-Instruct",
        rpm_limit=20,
        notes="Free GPU studios with SSH-able instances.",
    ),
    "oracle": ProviderSpec(
        name="oracle",
        kind=COMPUTE,
        base_url="https://api.oracle.com",
        api_key_env="OCI_API_KEY",
        default_model="Qwen2.5-7B-Instruct",
        rpm_limit=10,
        notes="Always-free ARM Ampere A1 nodes (up to 4 OCPU / 24GB).",
        extra={"secondary_env": "OCI_TENANCY", "tertiary_env": "OCI_USER"},
    ),
    "termux": ProviderSpec(
        name="termux",
        kind=COMPUTE,
        base_url="http://127.0.0.1:8081",
        api_key_env="TERMUX_MESH_TOKEN",
        default_model="Qwen2.5-0.5B-Instruct",
        rpm_limit=60,
        notes="Local Android node (CPU inference + relay).",
    ),
    "local": ProviderSpec(
        name="local",
        kind=COMPUTE,
        base_url="http://127.0.0.1:8081",
        api_key_env="HYPER_MESH_TOKEN",
        default_model="Qwen2.5-1.5B-Instruct",
        rpm_limit=120,
        notes="Loopback worker used for development and tests.",
    ),
}

#: Providers that participate in the sub-second edge tier.
EDGE_PROVIDERS: tuple[str, ...] = ("groq", "cerebras", "sambanova", "openrouter", "google_ai_studio", "fireworks")

#: Providers that can be provisioned headlessly to host a GPU worker.
COMPUTE_PROVIDERS: tuple[str, ...] = (
    "kaggle",
    "hf_spaces",
    "github_actions",
    "lightning",
    "oracle",
    "termux",
    "local",
)


def get_provider(name: str) -> ProviderSpec | None:
    """Look up a provider by name (case-insensitive)."""
    return PROVIDERS.get(name.strip().lower())


def require_provider(name: str) -> ProviderSpec:
    """Look up a provider or raise :class:`KeyError`."""
    spec = get_provider(name)
    if spec is None:
        raise KeyError(f"unknown provider: {name}")
    return spec


def edge_provider_names() -> list[str]:
    """Names of every provider usable as an edge drafter."""
    return list(EDGE_PROVIDERS)


def compute_provider_names() -> list[str]:
    """Names of every provider able to host a cluster worker."""
    return list(COMPUTE_PROVIDERS)


__all__ = [
    "INFERENCE",
    "COMPUTE",
    "ProviderSpec",
    "PROVIDERS",
    "EDGE_PROVIDERS",
    "COMPUTE_PROVIDERS",
    "get_provider",
    "require_provider",
    "edge_provider_names",
    "compute_provider_names",
]
