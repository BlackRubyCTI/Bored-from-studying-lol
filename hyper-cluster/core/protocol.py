"""The Hyper-Cluster wire protocol (version ``1.0``).

Every message that crosses a process boundary -- head node <-> worker,
head node <-> phone, head node <-> external provider -- is described here.
The models are Pydantic v2 so they validate on arrival, serialise to JSON for
transmission and can be re-used directly as FastAPI request/response bodies.

Design rules
------------
* Every message carries ``protocol_version``; peers reject unknown majors.
* Every message that can be retried carries an idempotency ``request_id``.
* Enums are plain ``str`` subclasses so JSON stays human readable.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.ids import new_request_id

PROTOCOL_VERSION = "1.0"
SUPPORTED_PROTOCOL_MAJORS = (1,)


class StrictModel(BaseModel):
    """Base model with sane defaults for network payloads."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True, validate_assignment=False)


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
class WorkerStatus(str, Enum):
    """Lifecycle state of a mesh worker."""

    BOOTING = "booting"
    LOADING_MODEL = "loading_model"
    IDLE = "idle"
    BUSY = "busy"
    DRAINING = "draining"
    DEAD = "dead"


class RequestMode(str, Enum):
    """Reasoning mode requested by the client."""

    AUTO = "auto"          # head node decides (fast / deep / council by heuristics)
    FAST = "fast"          # edge-first, lowest latency, shortest answer
    DEEP = "deep"          # strongest pooled worker, long budget
    COUNCIL = "council"    # multi-model deliberation with cross-critique
    SANDBOX = "sandbox"    # tool-use enabled (sandbox arrives in milestone 2)


class WorkerTier(str, Enum):
    """Where a piece of inference is executed."""

    EDGE = "edge"          # sub-second free inference API (Groq/Cerebras/...)
    CLUSTER = "cluster"    # pooled self-hosted GPU worker


class ChunkSource(str, Enum):
    """Provenance of a streamed token."""

    EDGE = "edge"
    DRAFT = "draft"
    CLUSTER = "cluster"
    SYNTHESIS = "synthesis"


class EventType(str, Enum):
    """Server-sent event types emitted by the head node."""

    WORKER_REGISTERED = "worker.registered"
    WORKER_UPDATED = "worker.updated"
    WORKER_DRAINING = "worker.draining"
    WORKER_EXPIRED = "worker.expired"
    WORKER_LOST = "worker.lost"
    MODEL_CHANGED = "model.changed"
    QUEUE_UPDATED = "queue.updated"
    CAPACITY = "capacity.updated"
    REQUEST_STATE = "request.state"
    SYSTEM = "system"


# --------------------------------------------------------------------------- #
# Worker <-> head node
# --------------------------------------------------------------------------- #
class Capabilities(StrictModel):
    """Hardware/software capabilities advertised by a worker at registration."""

    vram_total_gb: float = Field(default=0.0, ge=0.0)
    vram_free_gb: float = Field(default=0.0, ge=0.0)
    compute_tflops: float = Field(default=0.0, ge=0.0)
    cpu_cores: int = Field(default=0, ge=0)
    ram_gb: float = Field(default=0.0, ge=0.0)
    provider: str = "local"
    region: str = "unknown"
    backend: str = "llama.cpp"
    device: str = "cpu"
    supported_families: list[str] = Field(default_factory=list)
    supported_context: int = Field(default=8192, ge=0)
    max_concurrency: int = Field(default=2, ge=1)
    supports_streaming: bool = True
    supports_tool_use: bool = False
    mesh_latency_ms: float = Field(default=0.0, ge=0.0)
    extra: dict[str, Any] = Field(default_factory=dict)


class WorkerHello(StrictModel):
    """Registration payload sent by a worker to the head node."""

    protocol_version: str = PROTOCOL_VERSION
    worker_id: str
    host: str = "127.0.0.1"
    port: int = Field(default=8081, ge=1, le=65535)
    capabilities: Capabilities = Field(default_factory=Capabilities)
    model_loaded: str | None = None
    model_family: str | None = None
    model_params_b: float | None = None
    started_at: float = Field(default=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("protocol_version")
    @classmethod
    def _check_version(cls, value: str) -> str:
        major = value.split(".")[0]
        if not major.isdigit() or int(major) not in SUPPORTED_PROTOCOL_MAJORS:
            raise ValueError(f"unsupported protocol version: {value}")
        return value


class WorkerHeartbeat(StrictModel):
    """Periodic liveness + telemetry beacon sent by a worker."""

    worker_id: str
    status: WorkerStatus = WorkerStatus.IDLE
    inflight: int = Field(default=0, ge=0)
    queue_depth: int = Field(default=0, ge=0)
    tokens_per_sec: float = Field(default=0.0, ge=0.0)
    ttft_ms: float = Field(default=0.0, ge=0.0)
    vram_free_gb: float = Field(default=0.0, ge=0.0)
    vram_total_gb: float = Field(default=0.0, ge=0.0)
    model_loaded: str | None = None
    mesh_latency_ms: float = Field(default=0.0, ge=0.0)
    ts: float = Field(default=0.0)


class WorkerAck(StrictModel):
    """Acknowledgement returned by the head node after registration."""

    accepted: bool = True
    worker_id: str
    assigned_worker_id: str | None = None
    ttl_seconds: int = 3600
    refresh_before_seconds: int = 300
    drain_before_seconds: int = 600
    head_url: str = ""
    protocol_version: str = PROTOCOL_VERSION
    message: str = ""


class ModelLoadRequest(StrictModel):
    """Ask a worker to load (or hot-swap) a GGUF model."""

    worker_id: str
    model: str
    family: str | None = None
    params_b: float | None = None
    vram_required_gb: float | None = None
    force: bool = False


class ModelLoadResult(StrictModel):
    worker_id: str
    model: str
    loaded: bool
    load_ms: float = 0.0
    vram_used_gb: float = 0.0
    error: str | None = None


# --------------------------------------------------------------------------- #
# Chat / inference
# --------------------------------------------------------------------------- #
class ChatMessage(StrictModel):
    """A single conversation turn."""

    role: Literal["system", "user", "assistant", "tool"] = "user"
    content: str
    name: str | None = None
    tool_call_id: str | None = None


class ChatRequest(StrictModel):
    """Inference request issued by the phone (or by the orchestrator itself)."""

    request_id: str = Field(default_factory=new_request_id)
    trace_id: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list)
    mode: RequestMode = RequestMode.AUTO
    model: str | None = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1024, ge=1, le=32768)
    stream: bool = True
    stop: list[str] = Field(default_factory=list)
    priority: int = Field(default=0, ge=0, le=100)
    deadline_seconds: float | None = Field(default=None, gt=0)
    seed: int | None = None
    tools: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("messages")
    @classmethod
    def _non_empty(cls, value: list[ChatMessage]) -> list[ChatMessage]:
        if not value:
            raise ValueError("at least one message is required")
        return value

    def to_prompt(self) -> str:
        """Flatten the conversation into a single prompt (worker fallback path)."""
        parts = []
        for message in self.messages:
            role = message.role.capitalize()
            parts.append(f"{role}: {message.content}")
        return "\n\n".join(parts)


class TokenChunk(StrictModel):
    """A streamed token fragment."""

    request_id: str
    text: str = ""
    index: int = 0
    source: ChunkSource = ChunkSource.CLUSTER
    draft: bool = False
    finish_reason: str | None = None


class Usage(StrictModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletion(StrictModel):
    """Non-streaming inference result."""

    request_id: str
    text: str = ""
    model: str | None = None
    worker_id: str | None = None
    tier: WorkerTier = WorkerTier.CLUSTER
    source: ChunkSource = ChunkSource.CLUSTER
    mode: RequestMode = RequestMode.AUTO
    tokens: int = 0
    usage: Usage = Field(default_factory=Usage)
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    tokens_per_sec: float = 0.0
    held_ms: float = 0.0
    verified: bool = True
    verified_by: str | None = None
    finish_reason: str = "stop"
    error: str | None = None
    council: dict[str, Any] | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Model catalogue
# --------------------------------------------------------------------------- #
class ModelSpec(StrictModel):
    """A model the cluster knows how to serve."""

    name: str
    family: str = "unknown"
    params_b: float = 7.0
    quant: str = "Q4_K_M"
    vram_required_gb: float = 6.0
    intelligence: float = 0.5
    context: int = 8192
    repo_id: str = ""
    filename: str = ""
    license: str = "unknown"
    edge_capable: bool = False
    tags: list[str] = Field(default_factory=list)

    @property
    def tier_hint(self) -> str:
        """Coarse size bucket used by the scheduler for capability matching."""
        if self.params_b < 3:
            return "nano"
        if self.params_b < 10:
            return "small"
        if self.params_b < 25:
            return "medium"
        if self.params_b < 45:
            return "large"
        return "xlarge"


class ModelCatalog(StrictModel):
    """The set of models the cluster is allowed to serve."""

    models: list[ModelSpec] = Field(default_factory=list)
    updated_at: float = 0.0
    active_model: str | None = None
    edge_models: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Cluster telemetry
# --------------------------------------------------------------------------- #
class WorkerSnapshot(StrictModel):
    """Immutable projection of a worker, safe to serialise to the phone."""

    worker_id: str
    host: str
    port: int
    provider: str
    region: str
    status: WorkerStatus
    tier: WorkerTier = WorkerTier.CLUSTER
    model_loaded: str | None = None
    model_family: str | None = None
    vram_total_gb: float = 0.0
    vram_free_gb: float = 0.0
    tokens_per_sec: float = 0.0
    mesh_latency_ms: float = 0.0
    inflight: int = 0
    queue_depth: int = 0
    registered_at: float = 0.0
    last_heartbeat: float = 0.0
    expires_at: float = 0.0
    seconds_to_expiry: float = 0.0
    draining: bool = False
    healthy: bool = False
    compute_tflops: float = 0.0
    supported_families: list[str] = Field(default_factory=list)


class ClusterStatus(StrictModel):
    """Aggregate view of the whole cluster, served to the mobile dashboard."""

    cluster_name: str
    instance_id: str
    ts: float
    healthy: bool
    worker_count: int = 0
    healthy_workers: int = 0
    draining_workers: int = 0
    total_vram_gb: float = 0.0
    free_vram_gb: float = 0.0
    total_tflops: float = 0.0
    active_model: str | None = None
    model_catalog_size: int = 0
    queued_requests: int = 0
    held_requests: int = 0
    in_flight: int = 0
    tokens_per_sec_aggregate: float = 0.0
    edge_providers: list[str] = Field(default_factory=list)
    edge_healthy: list[str] = Field(default_factory=list)
    workers: list[WorkerSnapshot] = Field(default_factory=list)
    mode: RequestMode = RequestMode.AUTO
    autopilot: bool = False
    uptime_seconds: float = 0.0


# --------------------------------------------------------------------------- #
# Streaming envelope
# --------------------------------------------------------------------------- #
class StreamEvent(StrictModel):
    """Envelope for every server-sent event produced by the orchestrator."""

    event: str = "message"
    request_id: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class RequestState(str, Enum):
    """Observable lifecycle of a request inside the control plane."""

    QUEUED = "queued"
    HELD = "held"
    DISPATCHED = "dispatched"
    STREAMING = "streaming"
    VERIFYING = "verifying"
    COUNCIL = "council"
    COMPLETED = "completed"
    FAILED = "failed"
    DEFERRED = "deferred"


class RequestStatus(StrictModel):
    """Snapshot of one request, queryable from the phone after a hold."""

    request_id: str
    state: RequestState = RequestState.QUEUED
    mode: RequestMode = RequestMode.AUTO
    created_at: float = 0.0
    updated_at: float = 0.0
    held_ms: float = 0.0
    attempts: int = 0
    worker_id: str | None = None
    text: str | None = None
    error: str | None = None
    replayed: bool = False


__all__ = [
    "PROTOCOL_VERSION",
    "SUPPORTED_PROTOCOL_MAJORS",
    "StrictModel",
    "WorkerStatus",
    "RequestMode",
    "WorkerTier",
    "ChunkSource",
    "EventType",
    "Capabilities",
    "WorkerHello",
    "WorkerHeartbeat",
    "WorkerAck",
    "ModelLoadRequest",
    "ModelLoadResult",
    "ChatMessage",
    "ChatRequest",
    "TokenChunk",
    "Usage",
    "ChatCompletion",
    "ModelSpec",
    "ModelCatalog",
    "WorkerSnapshot",
    "ClusterStatus",
    "StreamEvent",
    "RequestState",
    "RequestStatus",
]
