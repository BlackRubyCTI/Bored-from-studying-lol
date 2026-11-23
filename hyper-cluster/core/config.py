"""Layered configuration for Hyper-Cluster Omega.

Resolution order (later wins):

1. Hard-coded defaults in :class:`Settings` (always safe, never secret).
2. The ``.env`` file next to the project root (or ``HYPER_ENV_FILE``).
3. Real process environment variables prefixed with ``HYPER_``.

Every field can therefore be overridden without touching code, which is what
lets ``install_all.sh`` generate a working ``.env`` without the operator
editing a single Python file.

The module intentionally avoids third-party settings libraries so the control
plane can boot even in a degraded environment where only the standard library
plus FastAPI is installed.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from core.errors import ConfigError

ENV_PREFIX = "HYPER_"

#: Root of the project (the directory that contains ``core/``).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Default on-disk location for runtime state (databases, indexes, checkpoints).
DEFAULT_DATA_DIR = Path(os.environ.get("HYPER_DATA_DIR", Path.home() / ".hypercluster"))


def project_env_file() -> Path:
    """Path of the ``.env`` file used for configuration."""
    override = os.environ.get("HYPER_ENV_FILE")
    if override:
        return Path(override).expanduser()
    return PROJECT_ROOT / ".env"


def parse_dotenv(text: str) -> dict[str, str]:
    """Parse ``.env`` content into a mapping.

    Supports ``export KEY=value``, ``# comments``, blank lines, single/double
    quoted values and inline comments after unquoted values.
    """
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
            if value.startswith("!"):  # docker-compose style literal
                value = value[1:]
        else:
            hash_index = value.find(" #")
            if hash_index != -1:
                value = value[:hash_index].rstrip()
        values[key] = value
    return values


def load_dotenv(path: Path | str | None = None, *, override: bool = False) -> dict[str, str]:
    """Load a ``.env`` file into ``os.environ`` and return the parsed mapping."""
    env_path = Path(path) if path is not None else project_env_file()
    if not env_path.is_file():
        return {}
    try:
        raw = env_path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - unreadable file is a setup issue
        raise ConfigError(f"Unable to read env file {env_path}: {exc}") from exc
    parsed = parse_dotenv(raw)
    for key, value in parsed.items():
        if override or key not in os.environ:
            os.environ[key] = value
    return parsed


def _as_bool(raw: str | bool | None, default: bool) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    normalized = raw.strip().lower()
    if normalized in ("1", "true", "yes", "on", "y"):
        return True
    if normalized in ("0", "false", "no", "off", "n"):
        return False
    return default


def _as_int(raw: str | int | None, default: int) -> int:
    if raw is None or raw == "":
        return default
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


def _as_float(raw: str | float | None, default: float) -> float:
    if raw is None or raw == "":
        return default
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return default


def _as_list(raw: str | list[str] | None, default: list[str]) -> list[str]:
    if raw is None or raw == "":
        return list(default)
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    text = str(raw).strip()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass
    return [part.strip() for part in text.split(",") if part.strip()]


def _as_dict(raw: str | dict[str, Any] | None, default: dict[str, Any]) -> dict[str, Any]:
    if raw is None or raw == "":
        return dict(default)
    if isinstance(raw, dict):
        return dict(raw)
    text = str(raw).strip()
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Malformed JSON mapping in configuration: {exc}") from exc
    return dict(default)


@dataclass(slots=True)
class Settings:
    """Fully resolved runtime configuration.

    Instances are treated as immutable; use :meth:`with_overrides` to derive a
    modified copy (used heavily by the test-suite).
    """

    # --- Identity -----------------------------------------------------------
    cluster_name: str = "hyper-cluster-omega"
    instance_id: str = ""

    # --- Control plane bind -------------------------------------------------
    head_host: str = "0.0.0.0"
    head_port: int = 8080
    public_base_url: str = ""
    cors_origins: list[str] = field(default_factory=lambda: ["*"])

    # --- Authentication -----------------------------------------------------
    api_key: str = ""
    mesh_token: str = ""
    admin_token: str = ""

    # --- Filesystem ---------------------------------------------------------
    data_dir: str = str(DEFAULT_DATA_DIR)
    state_dir: str = ""
    logs_dir: str = ""
    env_file: str = ""

    # --- Worker lifecycle ---------------------------------------------------
    worker_ttl_seconds: int = 3600
    heartbeat_interval_seconds: int = 20
    handoff_lead_seconds: int = 600
    drain_grace_seconds: int = 30
    max_concurrency_per_worker: int = 2

    # --- Scheduling ---------------------------------------------------------
    request_timeout_seconds: int = 180
    scheduler_latency_weight: float = 1.0
    scheduler_queue_weight: float = 0.85
    scheduler_load_weight: float = 0.5
    scheduler_cold_start_penalty_ms: float = 8000.0
    scheduler_mesh_probe_timeout_ms: float = 1200

    # --- Hold & requeue -----------------------------------------------------
    hold_enabled: bool = True
    hold_queue_max_size: int = 512
    hold_request_ttl_seconds: int = 900
    hold_min_wait_seconds: float = 0.0

    # --- Reasoning modes ----------------------------------------------------
    default_mode: str = "auto"
    council_size: int = 3
    council_critique_rounds: int = 1
    council_max_tokens: int = 1400
    speculative_enabled: bool = True
    speculative_edge_providers: list[str] = field(default_factory=lambda: ["groq", "cerebras", "sambanova"])
    speculative_min_edge_tps: float = 40.0

    # --- tool use -----------------------------------------------------------
    sandbox_enabled: bool = True
    max_tool_rounds: int = 4
    sandbox_timeout_seconds: int = 20
    sandbox_memory_limit_mb: int = 512

    # --- auto-scaler (VRAM-aware model up/down-shift) -----------------------
    autoscaler_enabled: bool = True
    autoscaler_safety_factor: float = 1.15
    autoscaler_min_gain: float = 0.05
    autoscaler_min_dwell_seconds: int = 120
    autoscaler_max_changes_per_tick: int = 1

    # --- auto-spawner (headless compute provisioning) ------------------------
    # Provisioning external compute is opt-in: it spends the user's quota.
    spawner_enabled: bool = False
    spawner_providers: list[str] = field(
        default_factory=lambda: ["kaggle", "hf_spaces", "github_actions", "oracle"]
    )
    spawner_min_workers: int = 1
    spawner_tick_seconds: int = 15
    spawner_registration_timeout_seconds: int = 900
    spawner_default_ttl_seconds: int = 3600
    spawner_max_parallel_provisions: int = 2

    # --- persistent memory (milestone 4) ------------------------------------
    memory_enabled: bool = True
    memory_dir: str = ""
    memory_chunk_size: int = 800
    memory_chunk_overlap: int = 120
    memory_max_chunks: int = 20000
    memory_context_chars: int = 2400
    memory_context_top_k: int = 4

    # --- sleep-time self-evolution (milestone 4) -----------------------------
    #: The cluster learns while it is idle.
    evolution_enabled: bool = False
    #: How long the cluster must be quiet before it starts training.
    evolution_idle_seconds: int = 300
    #: The budget a user request has to preempt the training.
    evolution_preempt_budget_seconds: float = 2.0
    #: How many synthetic examples one cycle trains on.
    evolution_dataset_size: int = 16
    #: How hard the synthetic prompts are (0..1).
    evolution_difficulty: float = 0.8
    #: How many workers draft one prompt.
    evolution_drafts_per_prompt: int = 2
    #: The maximum number of training shards (one per worker).
    evolution_max_shards: int = 4
    #: Whether the system prompt is evolved as well.
    evolution_prompt_enabled: bool = True
    #: The training hyper-parameters.
    evolution_train_steps: int = 8
    evolution_train_learning_rate: float = 0.5
    evolution_train_rank: int = 8
    evolution_train_dim: int = 128
    evolution_train_vocab: int = 128
    evolution_checkpoint_dir: str = ""

    # --- phone alerts (milestone 5) ----------------------------------------
    #: Whether the cluster may push to the phone.
    notifier_enabled: bool = False
    #: The Telegram bot and the chat it writes to.
    notifier_telegram_token: str = ""
    notifier_telegram_chat_id: str = ""
    #: A generic webhook (ntfy, a push relay, ...).
    notifier_webhook_url: str = ""
    notifier_webhook_secret: str = ""
    #: Which events are pushed (empty = the defaults).
    notifier_events: list[str] = field(default_factory=list)
    #: The minimum delay between two pushes.
    notifier_min_interval_seconds: float = 0.0

    # --- mobile dashboard (milestone 5) -------------------------------------
    #: Whether the dashboard is served next to the control plane.
    dashboard_enabled: bool = True
    #: The address the dashboard binds to.
    dashboard_host: str = "0.0.0.0"
    #: The port of the dashboard (0 = the same port as the control plane).
    dashboard_port: int = 8082
    #: The public base URL of the dashboard (used by the QR code).
    dashboard_public_url: str = ""
    #: The head node the dashboard proxies to.
    dashboard_head_url: str = "http://127.0.0.1:8080"

    # --- cloud sync (milestone 4) -------------------------------------------
    #: The token of the Google Drive service account / OAuth client.
    drive_token: str = ""
    drive_api_base: str = "https://www.googleapis.com/drive/v3"
    drive_upload_base: str = "https://www.googleapis.com/upload/drive/v3"
    drive_folder: str = "Hyper-Cluster Memory"
    #: The Hugging Face Hub token and repository.
    hub_token: str = ""
    hub_base_url: str = "https://huggingface.co"
    hub_repo: str = ""
    sync_enabled: bool = False

    # --- External providers -------------------------------------------------
    edge_timeout_seconds: float = 30.0
    edge_max_retries: int = 2
    credential_file: str = ""
    provider_probe_interval_seconds: int = 300

    # --- Observability ------------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = True
    log_file: str = ""
    log_include_payload: bool = False

    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        if not self.instance_id:
            import uuid

            self.instance_id = uuid.uuid4().hex[:12]
        if not self.state_dir:
            self.state_dir = str(Path(self.data_dir) / "state")
        if not self.logs_dir:
            self.logs_dir = str(Path(self.data_dir) / "logs")
        if not self.credential_file:
            self.credential_file = str(Path(self.data_dir) / "credentials.json")
        if not self.memory_dir:
            self.memory_dir = str(Path(self.data_dir) / "memory")
        if not self.evolution_checkpoint_dir:
            self.evolution_checkpoint_dir = str(Path(self.data_dir) / "checkpoints")

    # ------------------------------------------------------------------ #
    @property
    def data_path(self) -> Path:
        return Path(self.data_dir).expanduser()

    @property
    def state_path(self) -> Path:
        return Path(self.state_dir).expanduser()

    @property
    def logs_path(self) -> Path:
        return Path(self.logs_dir).expanduser()

    def ensure_dirs(self) -> None:
        """Create the directories the runtime expects to exist."""
        for directory in (self.data_path, self.state_path, Path(self.logs_dir).expanduser()):
            directory.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Settings:
        """Build settings from an explicit environment mapping."""
        env = dict(os.environ if env is None else env)

        def pick(name: str, default: Any = None) -> str | None:
            return env.get(f"{ENV_PREFIX}{name.upper()}", default)

        raw_log_level = (pick("log_level", "INFO") or "INFO").upper()
        defaults = cls()
        settings = cls(
            cluster_name=pick("cluster_name", defaults.cluster_name) or defaults.cluster_name,
            instance_id=pick("instance_id", "") or "",
            head_host=pick("head_host", defaults.head_host) or defaults.head_host,
            head_port=_as_int(pick("head_port"), defaults.head_port),
            public_base_url=pick("public_base_url", "") or "",
            cors_origins=_as_list(pick("cors_origins"), defaults.cors_origins),
            api_key=pick("api_key", "") or "",
            mesh_token=pick("mesh_token", "") or "",
            admin_token=pick("admin_token", "") or "",
            data_dir=pick("data_dir", defaults.data_dir) or defaults.data_dir,
            state_dir=pick("state_dir", "") or "",
            logs_dir=pick("logs_dir", "") or "",
            env_file=pick("env_file", "") or "",
            worker_ttl_seconds=_as_int(pick("worker_ttl_seconds"), defaults.worker_ttl_seconds),
            heartbeat_interval_seconds=_as_int(pick("heartbeat_interval_seconds"), defaults.heartbeat_interval_seconds),
            handoff_lead_seconds=_as_int(pick("handoff_lead_seconds"), defaults.handoff_lead_seconds),
            drain_grace_seconds=_as_int(pick("drain_grace_seconds"), defaults.drain_grace_seconds),
            max_concurrency_per_worker=_as_int(pick("max_concurrency_per_worker"), defaults.max_concurrency_per_worker),
            request_timeout_seconds=_as_int(pick("request_timeout_seconds"), defaults.request_timeout_seconds),
            scheduler_latency_weight=_as_float(pick("scheduler_latency_weight"), defaults.scheduler_latency_weight),
            scheduler_queue_weight=_as_float(pick("scheduler_queue_weight"), defaults.scheduler_queue_weight),
            scheduler_load_weight=_as_float(pick("scheduler_load_weight"), defaults.scheduler_load_weight),
            scheduler_cold_start_penalty_ms=_as_float(
                pick("scheduler_cold_start_penalty_ms"), defaults.scheduler_cold_start_penalty_ms
            ),
            scheduler_mesh_probe_timeout_ms=_as_float(
                pick("scheduler_mesh_probe_timeout_ms"), defaults.scheduler_mesh_probe_timeout_ms
            ),
            hold_enabled=_as_bool(pick("hold_enabled"), defaults.hold_enabled),
            hold_queue_max_size=_as_int(pick("hold_queue_max_size"), defaults.hold_queue_max_size),
            hold_request_ttl_seconds=_as_int(pick("hold_request_ttl_seconds"), defaults.hold_request_ttl_seconds),
            hold_min_wait_seconds=_as_float(pick("hold_min_wait_seconds"), defaults.hold_min_wait_seconds),
            default_mode=(pick("default_mode", defaults.default_mode) or defaults.default_mode).lower(),
            council_size=_as_int(pick("council_size"), defaults.council_size),
            council_critique_rounds=_as_int(pick("council_critique_rounds"), defaults.council_critique_rounds),
            council_max_tokens=_as_int(pick("council_max_tokens"), defaults.council_max_tokens),
            speculative_enabled=_as_bool(pick("speculative_enabled"), defaults.speculative_enabled),
            speculative_edge_providers=_as_list(pick("speculative_edge_providers"), defaults.speculative_edge_providers),
            speculative_min_edge_tps=_as_float(pick("speculative_min_edge_tps"), defaults.speculative_min_edge_tps),
            sandbox_enabled=_as_bool(pick("sandbox_enabled"), defaults.sandbox_enabled),
            max_tool_rounds=_as_int(pick("max_tool_rounds"), defaults.max_tool_rounds),
            sandbox_timeout_seconds=_as_int(pick("sandbox_timeout_seconds"), defaults.sandbox_timeout_seconds),
            sandbox_memory_limit_mb=_as_int(pick("sandbox_memory_limit_mb"), defaults.sandbox_memory_limit_mb),
            autoscaler_enabled=_as_bool(pick("autoscaler_enabled"), defaults.autoscaler_enabled),
            autoscaler_safety_factor=_as_float(pick("autoscaler_safety_factor"), defaults.autoscaler_safety_factor),
            autoscaler_min_gain=_as_float(pick("autoscaler_min_gain"), defaults.autoscaler_min_gain),
            autoscaler_min_dwell_seconds=_as_int(
                pick("autoscaler_min_dwell_seconds"), defaults.autoscaler_min_dwell_seconds
            ),
            autoscaler_max_changes_per_tick=_as_int(
                pick("autoscaler_max_changes_per_tick"), defaults.autoscaler_max_changes_per_tick
            ),
            spawner_enabled=_as_bool(pick("spawner_enabled"), defaults.spawner_enabled),
            spawner_providers=_as_list(pick("spawner_providers"), defaults.spawner_providers),
            spawner_min_workers=_as_int(pick("spawner_min_workers"), defaults.spawner_min_workers),
            spawner_tick_seconds=_as_int(pick("spawner_tick_seconds"), defaults.spawner_tick_seconds),
            spawner_registration_timeout_seconds=_as_int(
                pick("spawner_registration_timeout_seconds"), defaults.spawner_registration_timeout_seconds
            ),
            spawner_default_ttl_seconds=_as_int(
                pick("spawner_default_ttl_seconds"), defaults.spawner_default_ttl_seconds
            ),
            spawner_max_parallel_provisions=_as_int(
                pick("spawner_max_parallel_provisions"), defaults.spawner_max_parallel_provisions
            ),
            memory_enabled=_as_bool(pick("memory_enabled"), defaults.memory_enabled),
            memory_dir=pick("memory_dir", "") or "",
            memory_chunk_size=_as_int(pick("memory_chunk_size"), defaults.memory_chunk_size),
            memory_chunk_overlap=_as_int(pick("memory_chunk_overlap"), defaults.memory_chunk_overlap),
            memory_max_chunks=_as_int(pick("memory_max_chunks"), defaults.memory_max_chunks),
            memory_context_chars=_as_int(pick("memory_context_chars"), defaults.memory_context_chars),
            memory_context_top_k=_as_int(pick("memory_context_top_k"), defaults.memory_context_top_k),
            evolution_enabled=_as_bool(pick("evolution_enabled"), defaults.evolution_enabled),
            evolution_idle_seconds=_as_int(pick("evolution_idle_seconds"), defaults.evolution_idle_seconds),
            evolution_preempt_budget_seconds=_as_float(
                pick("evolution_preempt_budget_seconds"), defaults.evolution_preempt_budget_seconds
            ),
            evolution_dataset_size=_as_int(pick("evolution_dataset_size"), defaults.evolution_dataset_size),
            evolution_difficulty=_as_float(pick("evolution_difficulty"), defaults.evolution_difficulty),
            evolution_drafts_per_prompt=_as_int(
                pick("evolution_drafts_per_prompt"), defaults.evolution_drafts_per_prompt
            ),
            evolution_max_shards=_as_int(pick("evolution_max_shards"), defaults.evolution_max_shards),
            evolution_prompt_enabled=_as_bool(
                pick("evolution_prompt_enabled"), defaults.evolution_prompt_enabled
            ),
            evolution_train_steps=_as_int(pick("evolution_train_steps"), defaults.evolution_train_steps),
            evolution_train_learning_rate=_as_float(
                pick("evolution_train_learning_rate"), defaults.evolution_train_learning_rate
            ),
            evolution_train_rank=_as_int(pick("evolution_train_rank"), defaults.evolution_train_rank),
            evolution_train_dim=_as_int(pick("evolution_train_dim"), defaults.evolution_train_dim),
            evolution_train_vocab=_as_int(pick("evolution_train_vocab"), defaults.evolution_train_vocab),
            evolution_checkpoint_dir=pick("evolution_checkpoint_dir", "") or "",
            notifier_enabled=_as_bool(pick("notifier_enabled"), defaults.notifier_enabled),
            notifier_telegram_token=pick("notifier_telegram_token", "") or "",
            notifier_telegram_chat_id=pick("notifier_telegram_chat_id", "") or "",
            notifier_webhook_url=pick("notifier_webhook_url", "") or "",
            notifier_webhook_secret=pick("notifier_webhook_secret", "") or "",
            notifier_events=list(pick("notifier_events", []) or []),
            notifier_min_interval_seconds=_as_float(
                pick("notifier_min_interval_seconds"), defaults.notifier_min_interval_seconds
            ),
            dashboard_enabled=_as_bool(pick("dashboard_enabled"), defaults.dashboard_enabled),
            dashboard_host=pick("dashboard_host", defaults.dashboard_host) or defaults.dashboard_host,
            dashboard_port=_as_int(pick("dashboard_port"), defaults.dashboard_port),
            dashboard_public_url=pick("dashboard_public_url", "") or "",
            dashboard_head_url=pick("dashboard_head_url", "") or defaults.dashboard_head_url,
            drive_token=pick("drive_token", "") or "",
            drive_api_base=pick("drive_api_base", defaults.drive_api_base),
            drive_upload_base=pick("drive_upload_base", defaults.drive_upload_base),
            drive_folder=pick("drive_folder", defaults.drive_folder),
            hub_token=pick("hub_token", "") or "",
            hub_base_url=pick("hub_base_url", defaults.hub_base_url),
            hub_repo=pick("hub_repo", "") or "",
            sync_enabled=_as_bool(pick("sync_enabled"), defaults.sync_enabled),
            edge_timeout_seconds=_as_float(pick("edge_timeout_seconds"), defaults.edge_timeout_seconds),
            edge_max_retries=_as_int(pick("edge_max_retries"), defaults.edge_max_retries),
            credential_file=pick("credential_file", "") or "",
            provider_probe_interval_seconds=_as_int(
                pick("provider_probe_interval_seconds"), defaults.provider_probe_interval_seconds
            ),
            log_level=raw_log_level,
            log_json=_as_bool(pick("log_json"), defaults.log_json),
            log_file=pick("log_file", "") or "",
            log_include_payload=_as_bool(pick("log_include_payload"), defaults.log_include_payload),
        )
        return settings

    # ------------------------------------------------------------------ #
    def with_overrides(self, **overrides: Any) -> Settings:
        """Return a copy of these settings with ``overrides`` applied."""
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        data.update({k: v for k, v in overrides.items() if v is not None})
        return Settings(**data)

    def redacted(self) -> dict[str, Any]:
        """Dictionary view with every secret masked (safe for logs and the API)."""
        secret_keys = {"api_key", "mesh_token", "admin_token"}
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name in secret_keys:
                out[f.name] = "***" if value else ""
            elif isinstance(value, Path):
                out[f.name] = str(value)
            else:
                out[f.name] = value
        return out

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Settings(cluster={self.cluster_name!r}, port={self.head_port}, instance={self.instance_id})"


_settings_lock = threading.Lock()
_settings: Settings | None = None


def get_settings(*, reload: bool = False, env: dict[str, str] | None = None) -> Settings:
    """Return the process-wide settings singleton.

    Args:
        reload: Rebuild from the environment even if a cached instance exists.
        env: Optional explicit environment mapping (used by tests).
    """
    global _settings
    with _settings_lock:
        if _settings is None or reload:
            if env is None and not reload:
                load_dotenv()
            _settings = Settings.from_env(env)
        return _settings


def reset_settings() -> None:
    """Drop the cached settings (used by the test-suite between cases)."""
    global _settings
    with _settings_lock:
        _settings = None


__all__ = [
    "ENV_PREFIX",
    "PROJECT_ROOT",
    "DEFAULT_DATA_DIR",
    "Settings",
    "get_settings",
    "reset_settings",
    "load_dotenv",
    "parse_dotenv",
    "project_env_file",
]
