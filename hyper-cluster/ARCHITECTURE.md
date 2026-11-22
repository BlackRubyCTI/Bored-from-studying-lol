# HYPER-CLUSTER OMEGA — System Blueprint

> Zero-limit autonomous AI operating system.
> Headless GPU compute pool · sub-second inference · mobile mission control.

---

## 0. Design laws (non-negotiable, enforced in code)

| # | Law | Where it is enforced |
|---|-----|----------------------|
| L1 | **No user-visible failure.** A request is never answered with a raw 5xx. Capacity problems become *holds*; holds become *deferred*; deferred work is replayed automatically. | `head_node/orchestrator.py` → `_await_capacity`, `head_node/hold_queue.py`, `head_node/request_store.py` |
| L2 | **No manual operations.** Provisioning, credential rotation, model swaps, TTL rotation and recovery are all driven by daemons, never by a human. | `head_node/auto_spawner.py` *(M3)*, `training/autopilot_evolver.py` *(M4)* |
| L3 | **Every component is a protocol citizen.** All cross-process messages are versioned Pydantic models. | `core/protocol.py` (`PROTOCOL_VERSION = "1.0"`) |
| L4 | **Secrets never leak.** Tokens are redacted in logs, in `/v1/status` and in every API surface. | `core/logging_setup.py` (`_SECRET_HINTS`), `core/credentials.py` (`to_dict(redact=True)`) |
| L5 | **Latency is a first-class metric.** The scheduler optimises *perceived* latency (TTFT + decode), corrected continuously by measured EWMA. | `head_node/scheduler.py` |
| L6 | **Failure is normal.** Workers die, quotas vanish, networks flap. Every path has a retry/failover/degrade branch. | failover in `_execute_cluster`, `WorkerClient`, `EdgeBridge` |

---

## 1. System topology

```
                        ┌──────────────────────────────────────┐
                        │            YOUR PHONE (PWA)          │
                        │  chat · voice · upload · telemetry  │
                        └───────────────┬──────────────────────┘
                                        │ HTTPS / SSE
                        ┌───────────────▼──────────────────────┐
                        │        HEAD NODE (FastAPI)           │
                        │  ┌──────────────────────────────┐    │
                        │  │ orchestrator (the brain)     │    │
                        │  │  · mode router               │    │
                        │  │  · speculative cascade       │    │
                        │  │  · council deliberation      │    │
                        │  │  · hold & requeue            │    │
                        │  └───────┬──────────────┬───────┘    │
                        │          │              │            │
                        │   ┌──────▼─────┐  ┌─────▼──────┐     │
                        │   │  scheduler  │  │ edge bridge│     │
                        │   │ (latency)   │  │ (Groq/Cer.)│    │
                        │   └──────┬─────┘  └─────┬──────┘     │
                        │          │              │            │
                        │   ┌──────▼──────────────▼───────┐    │
                        │   │  worker registry + hold queue│    │
                        │   └──────┬──────────────┬───────┘    │
                        │          │              │            │
                        │  ┌───────▼──────┐ ┌─────▼────────┐   │
                        │  │ request store│ │  event bus   │   │
                        │  │ (ledger)     │ │  (SSE fanout)│   │
                        │  └──────────────┘ └──────────────┘   │
                        └───────────────┬──────────────────────┘
                                        │ mesh (HTTP + SSE)
        ┌───────────────┬───────────────┼───────────────┬───────────────┐
        ▼               ▼               ▼               ▼               ▼
   ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐   ┌──────────┐
   │  T4 x2  │    │   P100   │    │ A100 40G │    │ Termux   │   │  CPU dev │
   │ Kaggle  │    │  Kaggle  │    │ HF Space │    │  Android │   │  local   │
   └─────────┘    └──────────┘    └──────────┘    └──────────┘   └──────────┘
```

**Two inference tiers, one protocol:**

* **Edge tier** — free inference APIs (Groq, Cerebras, SambaNova, OpenRouter, Google AI Studio, Fireworks).
  Sub-second TTFT, no GPUs required, multi-account with automatic failover.
* **Cluster tier** — pooled, self-hosted GPU workers (llama.cpp). Slower to reach, but hosts the
  large models, is hot-swappable and survives provider outages.

---

## 2. Milestone map

| Milestone | Deliverables | Status |
|-----------|--------------|--------|
| **1 — Control plane & resilient orchestrator** | `core/*`, `head_node/{app,serve,orchestrator,scheduler,worker_registry,hold_queue,request_store,worker_client,edge_bridge,council,events}.py` | ✅ **complete** |
| **2 — Self-benchmarking worker, mesh & tool sandbox** | `worker/{bootstrap.sh,worker_agent.py,telemetry.py,inference_engine.py,mesh.py}`, `tools/sandbox.py`, Tailscale mesh | ✅ **complete** |
| 3 — Headless API compute spawner & VRAM auto-scaler | `head_node/{auto_spawner.py,model_selector.py}` | ⏳ |
| 4 — Infinite memory & sleep-time self-evolution | `memory/vector_memory.py`, `training/{autopilot_evolver,self_play,synthetic_data,prompt_evolver,distributed_trainer,checkpoint_streamer}.py` | ⏳ |
| 5 — Mobile PWA mission control, phone alerts, one-command installer | `mobile/{dashboard.py,cli.py,notifier.py}`, `install_all.sh` | ⏳ |

---

## 3. Milestone 1 — deep dive

### 3.1 Module map

```
core/                      shared by head node, workers and the phone
├── config.py              layered settings (defaults → .env → HYPER_* env)
├── logging_setup.py       JSON + console formatters, ambient request context, secret redaction
├── protocol.py            the versioned wire protocol (Pydantic v2)
├── providers.py           canonical registry of every external provider
├── credentials.py         multi-account credential vault (0600 file + env overlay)
├── catalog.py             default GGUF model catalogue (VRAM + intelligence per model)
├── errors.py              typed exception hierarchy
└── ids.py                 id + clock helpers

head_node/                 the control plane
├── app.py                 FastAPI factory: chat, stream, workers, status, events, credentials
├── serve.py               CLI entry point (uvicorn, --check, --print-config)
├── orchestrator.py        THE BRAIN: routing, speculation, council, hold/requeue, failover
├── scheduler.py           latency-aware selection (TTFT + queue + load + cold-start model)
├── worker_registry.py     authoritative mesh view (TTL countdowns, EWMA latency, capacity claims)
├── hold_queue.py          in-memory hold-and-requeue buffer (FIFO, deadlines, deferred pool)
├── request_store.py       request ledger — deferred work stays recoverable
├── worker_client.py       HTTP/SSE client for workers (with a correct SSE parser)
├── edge_bridge.py         multi-account bridge to the free inference tier
├── council.py             Mixture-of-Agents deliberation (fan-out → cross-critique → synthesis)
└── events.py              async event bus feeding the live dashboard
```

### 3.2 Request lifecycle

```
phone ──POST /v1/chat/stream──▶ orchestrator
                                 │
                    classify() ──┴──▶ mode = fast | deep | council
                                 │
                 ┌───────────────┴────────────────┐
                 ▼                                ▼
        FAST (speculative)                DEEP / COUNCIL
                 │                                │
      edge draft ──┐                    scheduler.select()
      (Groq etc.)  │                            │
            ┌──────┴──────┐            no capacity?
            ▼             ▼              ┌────┴────┐
      cluster verify   (edge only)      yes       no
            │                            │         │
       similarity ≥ 0.55?            HOLD in    dispatch to
        │        │                   memory    best worker
     confirm    revise                     │
        │        │                    capacity arrives?
        └────────┴──────────────▶ resume & replay
                                 │
                            ledger (recoverable)
```

### 3.3 The scheduler's cost model

For every candidate worker:

```
expected_tps   = ewma_tokens_per_sec           (measured, EWMA α=0.3)
               | tokens_per_sec                 (reported)
               | f(vram, params, device)        (cold-start estimate)

service_ms     = expected_tokens / expected_tps * 1000
queue_delay_ms = service_ms * (inflight / max_concurrency)
load_penalty   = (inflight / max_concurrency) * 180 ms
cold_start_ms  = 8000 * min(2, model_vram / 8)   (only when the model must be loaded)

ttft_ms        = 2*mesh_rtt + 25 + queue_delay + load_penalty + cold_start
decode_ms      = expected_tokens / expected_tps * 1000

score          = 1.00*ttft + 0.85*queue + 0.50*load + 1.00*cold_start
score         *= (1 - 0.07*intelligence)   in deep/council modes
score         *= (1 + 0.04*intelligence)   in fast mode
score         *= 0.92                      when the model is already resident
```

The score is recalculated on every request from *observed* numbers, so a worker
that silently degrades loses its position within a few requests.

### 3.4 Hold & requeue semantics

1. No worker with capacity → the request is parked in `HoldQueue` (FIFO, bounded, TTL).
2. The phone is told `event: hold` — it keeps waiting, it never sees an error.
3. A worker frees up → the capacity loop wakes the waiter → `event: resume` → dispatch.
4. Deadline reached → `event: deferred`, HTTP `202`, request kept in the deferred pool.
5. The background replayer picks deferred work up automatically once capacity returns and
   writes the result into the ledger → the phone fetches `GET /v1/requests/{id}`.

### 3.5 Council mode

1. **Fan-out** — N workers (distinct models preferred) answer the same question in parallel.
2. **Cross-critique** — each member reviews the others for errors, contradictions and gaps.
3. **Synthesis** — the strongest member merges everything into one definitive answer.
4. **Consensus** — mean pairwise semantic similarity is measured and reported to the dashboard.

### 3.6 Speculative cascading

The edge tier drafts the answer in ~200 ms while the GPU cluster verifies it in parallel.
If the cluster's answer agrees (Jaccard ≥ 0.55) the draft is *confirmed*; if it diverges the
client receives `event: revise` with the authoritative text. Either way the phone renders
something within a fraction of a second.

### 3.7 API surface (Milestone 1)

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| GET | `/` | – | service banner |
| GET | `/health` | – | liveness + coarse health |
| GET | `/v1/status` | client | full telemetry (workers, VRAM, queue, edge, ledger) |
| GET | `/v1/workers` | client | worker inventory |
| POST | `/v1/workers/register` | mesh | worker self-registration |
| POST | `/v1/workers/{id}/heartbeat` | mesh | liveness beacon |
| POST | `/v1/workers/{id}/drain` | admin | graceful retirement |
| DELETE | `/v1/workers/{id}` | admin | hard removal |
| POST | `/v1/chat` | client | completion (JSON, or SSE when `stream=true`) |
| POST | `/v1/chat/stream` | client | SSE completion |
| GET | `/v1/events` | client | live telemetry stream |
| GET | `/v1/models` | client | model catalogue + active model |
| GET | `/v1/requests/{id}` | client | request status + result |
| GET | `/v1/queue` | client | hold-and-requeue buffer |
| GET/POST | `/v1/credentials` | admin | credential vault |

### 3.8 SSE event vocabulary

| event | payload |
|-------|---------|
| `start` | resolved mode, model, budget |
| `dispatch` | chosen worker, provider, predicted TTFT, score |
| `hold` / `resume` | queue position, deadline / held duration |
| `token` | streamed text + source (`edge`/`draft`/`cluster`/`synthesis`) |
| `edge` | which edge provider drafted |
| `verify` | verdict (`confirmed` / `unverified`) + similarity |
| `revise` | authoritative replacement text |
| `council` | members, critiques, consensus |
| `done` | the authoritative `ChatCompletion` |
| `error` / `deferred` | terminal failure / queued-for-replay |

---

## 4. Milestone 2 — worker + sandbox (implemented)

### 4.1 Worker runtime

```
┌──────────────────────── worker node ────────────────────────┐
│ bootstrap.sh                                                │
│  ├─ platform detect (linux / darwin / termux / android)     │
│  ├─ system deps (apt / dnf / apk / pkg)                     │
│  ├─ venv + pinned requirements                              │
│  ├─ pre-compiled llama.cpp  → ~/.hypercluster/cache (SHA)   │
│  ├─ GGUF                  → ~/.hypercluster/models (SHA)    │
│  ├─ systemd unit / Termux service (generated from env)      │
│  └─ health probe + auto-restart                             │
│                                                             │
│ worker_agent.py                                             │
│  ├─ load model → benchmark → register → heartbeat           │
│  ├─ /v1/chat (sync)  /v1/chat/stream (SSE)  /v1/models     │
│  ├─ /v1/telemetry  /v1/mesh  /health                        │
│  ├─ hot-load a different GGUF without restarting             │
│  └─ drain (stop new work, finish in flight, exit)           │
│                                                             │
│ telemetry.py                                                 │
│  ├─ GPU: nvidia-smi → name, VRAM total/free, driver         │
│  ├─ CPU/RAM: cores, load, available memory                  │
│  ├─ FLOPS: real FP32 GEMM (torch CUDA, else numpy)         │
│  ├─ latency: head-to-head RTT to the control plane          │
│  ├─ decode: tokens/s measured on the running model          │
│  └─ → core.protocol.Capabilities (measured, never guessed)  │
└─────────────────────────────────────────────────────────────┘
```

The backends are explicit and never mislabelled:

| backend | what it is | reported as |
|---|---|---|
| `llama.cpp` | local GGUF through `llama-cpp-python` | `llama.cpp` |
| `upstream` | any OpenAI-compatible HTTP endpoint | `upstream` |
| `simulator` | deterministic generator for hosts without a GPU | `simulator` |

Hot-loading builds the replacement backend *before* releasing the active one,
so a failed load can never leave the worker without a model.

### 4.2 Tailscale mesh

* `tailscale up --authkey=… --advertise-tags=tag:hypercluster-worker`
* `tailscale serve` (tailnet-only) and `tailscale funnel` (public) advertise the
  worker's own port; `tailscale status --json` is parsed for peer discovery.
* A tag-scoped ACL (`generate_acl`) keeps workers in their own lane: they may
  reach the head node and their peers, never the internet.
* Every call degrades gracefully: without the binary the mesh reports
  `available=False` and the worker joins over the LAN instead.

### 4.3 Tool sandbox

```
model ──{"tool":"python","args":{...}}──▶ head node
                                             │
                              ┌──────────────┴───────────────┐
                              ▼                              ▼
                    PythonSandbox (subprocess)        WebTools / FileAnalyzer
                    ├ separate interpreter             ├ live HTTP fetch
                    ├ rlimit CPU/AS/FSIZE/NPROC/NOFILE   ├ live search (DDG → Wikipedia)
                    ├ socket + getaddrinfo guard        └ file stats, entropy, text
                    └ subprocess / os.system guard
                              │
                    stdout / stderr / value ──▶ back into the conversation
```

The model answers with exactly one JSON object per round
(`{"tool": …}` or `{"final": …}`); the head node executes it, appends the
result and repeats until the model finalises or `HYPER_MAX_TOOL_ROUNDS` is
reached. Tool activity is streamed to the client as a `tool` SSE event.

---

## 5. Planned architecture (Milestones 3–5)

### Milestone 3 — spawner + auto-scaler
* `head_node/auto_spawner.py` — headless provisioning for Kaggle kernels, HF Spaces, GitHub Actions,
  Lightning AI, Oracle ARM and Termux; multi-account credential vault; **10-minute pre-emptive
  TTL rotator** that provisions a replacement before the current session dies and shifts traffic
  after the replacement registers.
* `head_node/model_selector.py` — sums live cluster VRAM, picks the highest-intelligence GGUF that
  fits, hot-loads it, and pairs small fast nodes as speculative draft models.

### Milestone 4 — memory + self-evolution
* `memory/vector_memory.py` — FAISS/Chroma index synced to Google Drive + Hugging Face Hub.
* `training/autopilot_evolver.py` — after 5 idle minutes the cluster trains; any incoming message
  preempts it within the 2 s budget.
* `training/self_play.py` / `synthetic_data.py` — mine chats, generate hard prompts, draft with
  many workers, keep only critic-verified traces.
* `training/distributed_trainer.py` / `checkpoint_streamer.py` — checkpointed QLoRA, auto-merge,
  push to Drive/HF, hot-reload the upgraded brain.
* `training/prompt_evolver.py` — mutate, benchmark and promote system prompts.

### Milestone 5 — mission control
* `mobile/dashboard.py` — dark-mode, touch-first PWA (streaming chat, voice in/out, drag-and-drop
  RAG upload, council toggle, autopilot switch, live GPU/VRAM/tok-s telemetry).
* `mobile/notifier.py` — Telegram/webhook push when a better model ships or a key needs attention.
* `install_all.sh` — one command: `.env` wizard, Drive + HF wiring, systemd/Termux autostart,
  LAN URL + terminal QR code.

---

## 6. Repository layout

```
hyper-cluster/
├── core/                    shared foundation
├── head_node/               control plane  (M1 ✅, M3 ⏳)
├── worker/                  mesh worker    (M2 ✅)
│   ├── worker_agent.py      runtime + HTTP API + CLI
│   ├── inference_engine.py  llama.cpp / upstream / simulator
│   ├── telemetry.py         self-benchmarking
│   ├── mesh.py              Tailscale automation
│   └── bootstrap.sh         zero-touch provisioning
├── tools/                   tool-use sandbox (M2 ✅)
│   └── sandbox.py           python isolation, web, files, tool protocol
├── memory/                  vector memory   (M4 ⏳)
├── training/                self-evolution  (M4 ⏳)
├── mobile/                  PWA + notifier  (M5 ⏳)
├── tests/                   live test-suite (141 tests, real HTTP + SSE)
├── scripts/                 smoke harnesses (M1 + M2)
├── .env.example             configuration template
├── ARCHITECTURE.md          this document
└── pyproject.toml
```
