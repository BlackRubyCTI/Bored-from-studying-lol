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
| **3 — Headless API compute spawner & VRAM auto-scaler** | `head_node/{auto_spawner.py,model_selector.py}` | ✅ **complete** |
| **4 — Infinite memory & sleep-time self-evolution** | `memory/{vector_memory,sync}.py`, `training/{autopilot_evolver,self_play,synthetic_data,prompt_evolver,distributed_trainer,checkpoint_streamer,qlora}.py`, `worker/brain.py` | ✅ **complete** |
| **5 — Mobile PWA mission control, phone alerts, one-command installer** | `mobile/{dashboard.py,cli.py,notifier.py,qr.py}`, `mobile/static/*`, `memory/pdf.py`, `install_all.sh` | ✅ **complete** |

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

## 5. Milestone 3 — spawner + auto-scaler (implemented)

### 5.1 The headless compute spawner

`head_node/auto_spawner.py` turns a credential into a running worker. Every provider is a
`Provisioner` that speaks that provider's REST API headlessly — no browser, no manual click:

| Provider | What the provisioner does |
|---|---|
| `kaggle` | creates a kernel, uploads the bootstrap payload, starts it, waits for the session deadline |
| `hf_spaces` | creates a ZeroGPU Space, uploads the Dockerfile, starts it, reads the runtime stage |
| `github_actions` | dispatches a reusable workflow (`workflow_dispatch`) with the head URL and mesh token |
| `oracle` | boots an ARM/Ampere instance through the OCI API — the request is **signed** (RSA, `cryptography`) with the instance principal's key |
| `edge_api` | registers a free API (Groq / Cerebras / SambaNova / OpenRouter) as a synthetic tier |

Every provisioner returns a `SpawnResult` with the instance id, the endpoint, the account that
paid for it and a redacted detail string. The **multi-account credential vault** picks the account:
`core/credentials.py` holds one entry per account (`provider`, `label`, `token`, `metadata`), reads
`HYPER_CRED_<PROVIDER>_<NAME>` plus `HYPER_CRED_<PROVIDER>_<NAME>_META` (JSON) from the
environment, keeps the file at `0600`, and **fails over** to the next account when a provider
rejects the current one (rate limit, expired key, quota).

### 5.2 The pre-emptive TTL hot-swapper

The liveness TTL is refreshed by every heartbeat, so it never *approaches* — it cannot be used to
predict a death. The real deadline is the **session deadline**: a Kaggle kernel's runtime limit, a
GitHub job's timeout, an instance's lifetime. The worker reports it in its registration metadata
(`session_ends_at`), and the rotator acts when it comes within `HYPER_HANDOFF_LEAD_SECONDS`
(default 600 s = 10 minutes):

```
session_ends_at ──(t-10 min)──▶ provision replacement ──▶ replacement registers ──▶ drain old ──▶ release
        │                              │                            │                      │           │
        └── the old worker keeps serving traffic the whole time ───┘                      └───────────┘
```

`AutoSpawner.tick()` performs three steps per pass:

1. **rotate** — every worker whose session ends within the lead time gets a replacement provisioned
   (a worker that went silent is treated the same way);
2. **handoff** — as soon as a replacement registers, the old worker is drained and the traffic moves;
3. **scale out** — if the pool drops below `HYPER_SPAWNER_MIN_WORKERS`, the spawner provisions the
   missing nodes itself.

Because the replacement is provisioned *before* the session dies, the hand-over is invisible to
the client: `scripts/smoke_m3.py` keeps a request loop running during the whole procedure and
records **zero failures**.

### 5.3 The VRAM auto-scaler

`head_node/model_selector.py` works on the *measured* VRAM of the live mesh, not on guesses:

* **pooling** — every worker's advertised VRAM, the VRAM its resident model occupies (catalogue
  estimate) and its free VRAM are summed; the pool exposes total / free / reserved / capacity,
  per provider and per family;
* **demand** — the orchestrator records every served request into a `DemandProfile`
  (`fast` / `deep` / `council`), which yields the current demand level;
* **target** — the level maps to a target model: the best GGUF that fits a healthy worker with the
  safety factor applied (`autoscaler_safety_factor`), or a small-but-competent model for fast
  traffic;
* **plan** — a worker that hosts a *weaker* model than the target is upshifted, a worker that hosts
  a *stronger* one is downshifted; `autoscaler_max_changes_per_tick` bounds the churn and
  `autoscaler_min_dwell_seconds` keeps a node stable after a swap;
* **apply** — the orchestrator hot-loads the model on the worker (`POST /v1/model/load`), updates
  the registry and emits a `model.changed` event. A failed load leaves the previous model in place.

**Honest throughput.** A worker that hot-swaps its model invalidates its last measurement. The
registry therefore keeps the node's *hardware factor* (`tokens/s × B`, measured once in the
benchmark) and re-derives the throughput for the model the node hosts now — so a 32B model on a
node that measured 500 tok/s with a 0.5B model is correctly estimated as slower.

### 5.4 Speculative draft / verifier pairing

The selector pairs a small resident model (the **draft**) with a large one (the **verifier**) when
the draft is at least 4× smaller *and* faster. The expected cost of speculative decoding is

```
E[t] = (1 - a)·t_target + a·(t_draft + e·t_target)      speed-up = t_target / E[t]
```

with an acceptance rate `a = 0.7` and a verification overhead `e = 0.1`. The orchestrator runs the
draft on its own worker and verifies the tokens on the verifier worker (`ChunkSource.DRAFT` marks
the speculative tokens in the SSE stream); the edge tier is used as the draft when the cluster has
no small node.

### 5.5 New control-plane routes

| Route | Purpose |
|---|---|
| `GET /v1/cluster/vram` | the live VRAM pool and the demand profile |
| `GET /v1/models/plan` | what the auto-scaler *would* do (no side effects) |
| `POST /v1/cluster/autoscale` | compute the plan and execute it (hot-swap) |
| `GET /v1/speculative/pair` | the current draft / verifier pair and its speed-up |
| `GET /v1/spawner` | the spawner state: instances, accounts, pending hand-overs |
| `POST /v1/spawner/provision` | provision one instance on demand |
| `POST /v1/spawner/rotate` | rotate one worker now |

---

## 6. Milestone 4 — memory + self-evolution (implemented)

### 6.1 The memory

```
memory/
├── vector_memory.py   the store: chunks, hybrid retrieval, facts, snapshots
└── sync.py            the cloud: Drive + Hub clients, resumable uploads, checksums
```

* **Chunking & indexing** — a document is split into overlapping chunks; every chunk gets a
  deterministic hashed embedding (`dim` configurable) *and* a keyword vector, so the retrieval
  works without a GPU and without a model download.
* **Hybrid retrieval** — the score is `0.6 × cosine + 0.4 × keyword overlap`; a fact always
  outranks the raw conversation it came from (`importance`).
* **Durable facts** — the conversations are distilled into short statements
  (`my name is Ada`, `stack: kubernetes with three node pools`). The distiller stops at the first
  conjunction, so one sentence never becomes two facts.
* **Persistence** — the store is written atomically (temp file + rename) after every mutation and
  reloaded on boot; the revision counter tells the caller whether anything changed.
* **Snapshots** — a portable JSON document with a SHA-256 of its content; `restore()` refuses a
  snapshot whose checksum does not match.
* **The cloud** — `MemorySync` pushes the snapshot to Drive (resumable session upload) and to the
  Hugging Face Hub (pre-signed upload + commit), stores the checksum in the Drive
  `appProperties`, and pulls the newest snapshot back. A corrupted download is rejected, never
  restored.

### 6.2 The sleep-time engine

```
idle ──▶ mine ──▶ synthesise ──▶ self-play ──▶ prompt ──▶ train ──▶ merge ──▶ push ──▶ hot-reload
```

| Stage | Module | What it does |
|---|---|---|
| idle | `training/autopilot_evolver.py` | waits for `evolution_idle_seconds` of silence, then starts a cycle |
| mine | `training/synthetic_data.py` | the conversations are mined into question/answer seeds |
| synthesise | `training/synthetic_data.py` | the seeds are mutated (audience, domain, format, risk, counterfactual) into harder prompts — deterministic for a fixed seed |
| self-play | `training/self_play.py` | the prompts are drafted by several workers, the critic scores them, only the verified traces survive |
| prompt | `training/prompt_evolver.py` | the system prompt is mutated, benchmarked on held-out questions and promoted **only** on a real gain |
| train | `training/distributed_trainer.py` | federated QLoRA: the dataset is split into shards, every shard is computed by a worker, the gradients are averaged by shard weight |
| merge | `training/qlora.py` | the dense update is factorised (truncated SVD) into a rank-`r` LoRA pair, int8-quantised and merged into the base matrix (`B @ A`) |
| push | `training/checkpoint_streamer.py` | the checkpoint (matrix + metadata) goes to Drive and the Hub, SHA-256 verified |
| hot-reload | `worker/brain.py` | the new artifact is decoded and validated *before* it replaces the running brain — a broken artifact never takes a worker down |

Every step writes a checkpoint, so an interrupted (or preempted) run resumes where it stopped.

### 6.3 The preemption contract

> A chat that arrives during the training must be served in **less than two seconds**.

The contract is enforced by a single synchronous flip on the request path:

1. `Orchestrator._user_activity()` runs on **every** incoming request (before any routing).
2. It calls `AutopilotEvolver.notify_activity()`, which flips the `PreemptionToken`
   (an `asyncio.Event`) and resets the idle timer. Both are O(1) — no lock, no I/O.
3. Every stage of the pipeline checks the token: the self-play between prompts, the trainer
   between steps *and* between shards.
4. The in-flight shard call is **raced against the token** (`asyncio.wait`), so a slow worker
   cannot hold the user hostage: the call is cancelled the moment the token flips.
5. The token records `requested_at` and `acknowledged_at`; the difference is the measured
   preemption latency, which is exposed on `/v1/evolution/status` and asserted by the tests.

Measured in the live smoke test: **~50 ms** (budget: 2000 ms).

The training never claims worker capacity, so a preempted run cannot leave a worker busy.

### 6.4 New control-plane routes

| Route | Purpose |
|---|---|
| `GET /v1/memory` | the state of the memory (chunks, documents, revision) |
| `GET /v1/memory/documents` | the indexed documents |
| `POST /v1/memory/documents` | upload a file — it is indexed immediately |
| `POST /v1/memory/search` | hybrid retrieval over the memory |
| `POST /v1/memory/sync` | push the snapshot to Drive / the Hub, or pull it back |
| `GET /v1/evolution/status` | the state of the engine, the phase, the preemption latency |
| `POST /v1/evolution/run` | run one self-evolution cycle now |
| `POST /v1/evolution/prompt` | set the system prompt of the cluster |
| `POST /v1/evolution/toggle` | the phone's autopilot switch (on/off, effective at once) |

### 6.5 Milestone 5 — the mobile mission control (implemented)

The phone is the only screen the user is guaranteed to have.  Milestone 5
turns the cluster into something you drive from a phone: a dark, touch-first
PWA that streams the answers, a terminal that works in Termux, alerts that
push to the pocket and an installer that wires the whole thing up in one
command.

#### 6.5.1 The dashboard (`mobile/dashboard.py`)

A small FastAPI app with two jobs:

1. **The shell** — `index.html`, `styles.css`, `app.js`, `manifest.webmanifest`,
   `sw.js` and `icon.svg` are served from `mobile/static/`.  No framework, no
   build step, no CDN: the phone may be offline and the shell still opens.
2. **The proxy** — everything under `/api/` is forwarded to the head node
   (`HYPER_DASHBOARD_HEAD_URL`) with the method, the body, the query and the
   authentication headers.  `text/event-stream` answers are relayed chunk by
   chunk, so the chat really streams *through* the dashboard.

The catch-all route that serves the shell is registered **after** the proxy,
so `/api/...` can never be shadowed by the static files.

```
phone ──HTTP/SSE──▶ dashboard (:8082) ──HTTP/SSE──▶ head node (:8080)
        │                   │                            │
        │                   └── /api/* proxy ───────────┘
        └── /, /app.js, /sw.js  (the shell)
```

The PWA itself:

| Element | Implementation |
|---|---|
| Chat | `fetch` + a `ReadableStream` reader that appends every `token` event |
| Voice in | `SpeechRecognition` / `webkitSpeechRecognition` — the transcript lands in the composer |
| Voice out | `speechSynthesis` with the voice chosen in the settings sheet |
| PDF upload | a file picker **and** a drag-and-drop target; the file goes to `POST /v1/memory/documents` |
| Council | `mode: "council"` on the next request |
| Autopilot | `POST /v1/evolution/toggle` (admin token) — the engine stops *immediately* |
| Telemetry | `GET /v1/status` every three seconds: workers, free VRAM, tok/s, model, queue |
| Offline | a service worker caches the shell only — an API answer is never cached |

#### 6.5.2 The PDF reader (`memory/pdf.py`)

`pypdf`, `PyPDF2` and `fitz` are **not** dependencies of this project (and
Termux would need another package), so the text layer is extracted with the
standard library only:

* the objects are scanned out of the file (`N 0 obj … endobj`),
* the streams are inflated (`zlib`, with the raw-deflate fallback),
* the PNG predictors are undone (the `/Predictor` of the filters),
* the fonts are resolved (the resource names of the page are mapped to the
  font objects),
* the `ToUnicode` CMaps are parsed (including the `bfchar`/`bfrange`
  sections) so a subset font decodes to real characters,
* the text operators (`Tj`, `TJ`, `'`, `"`) are executed and the escapes are
  undone.

A PDF that is encrypted, scanned or broken never raises into the request
path: the upload answers **415** with the reason ("the PDF has no readable
text: …").

#### 6.5.3 The notifier (`mobile/notifier.py`)

The notifier subscribes to the head node's event bus and pushes the
interesting events to the pocket:

* **Telegram** — the bot API (`sendMessage`), so the phone gets a real push.
* **A webhook** — any URL that accepts JSON (ntfy, a relay, your own server),
  optionally signed with `x-hyper-secret`.

Both transports are best effort: a failed push is counted, logged and
dropped — it never blocks the event bus or the request path.  A throttle
(`HYPER_NOTIFIER_MIN_INTERVAL_SECONDS`) keeps a burst of events from turning
into a flood.

#### 6.5.4 The Termux CLI (`mobile/cli.py`)

`hc` is the same cluster from a terminal (Termux, SSH, `adb shell`):

```
hc status | workers | watch | memory | evolution | qr | config
hc chat "…" | ask | upload paper.pdf
```

The settings live in `~/.hypercluster/client.json` (`0600`).

#### 6.5.5 The QR encoder (`mobile/qr.py`)

`qrencode` is not everywhere (and Termux would need a package), so the
installer ships its own encoder: byte mode, error correction level M,
versions 1–10, the eight masks with the standard penalty rules and a
terminal renderer that uses the half-block characters (`▀`).  The encoder is
verified against the ISO/IEC 18004 layout (finder, timing and alignment
patterns, the format information, the zigzag placement) and it round-trips
through its own reader.

#### 6.5.6 The installer (`install_all.sh`)

One command, seven steps:

1. detect the platform (Linux / macOS / Termux on Android),
2. check Python (≥ 3.10),
3. create the virtualenv and install the dependencies,
4. write a complete `.env` (atomic, mode `0600`; the previous one is backed
   up; the secrets are generated when they are missing),
5. wire the Google Drive and the Hugging Face sync,
6. install the auto-boot service (a systemd unit on Linux, a
   `~/.termux/boot` script on Android, a launcher everywhere else),
7. print the QR code of the dashboard and the launch command.

It is idempotent: a second run only refreshes what is missing.

#### 6.5.7 The dashboard routes

| Route | Purpose |
|---|---|
| `GET /` | the PWA shell |
| `GET /styles.css`, `/app.js`, `/sw.js`, `/icon.svg`, `/manifest.webmanifest` | the assets of the shell |
| `GET /health` | is the head node reachable? |
| `GET /api/config` | what the phone needs to know (the head node, the features) |
| `* /api/*` | the proxy to the head node (every method, streaming included) |

---

## 7. Repository layout

```
hyper-cluster/
├── core/                    shared foundation
├── head_node/               control plane  (M1 ✅, M3 ✅, M4 ✅)
├── worker/                  mesh worker    (M2 ✅, M4 ✅)
│   ├── worker_agent.py      runtime + HTTP API + CLI
│   ├── inference_engine.py  llama.cpp / upstream / simulator
│   ├── telemetry.py         self-benchmarking
│   ├── mesh.py              Tailscale automation
│   ├── brain.py             the hot-reloadable LoRA brain
│   └── bootstrap.sh         zero-touch provisioning
├── tools/                   tool-use sandbox (M2 ✅)
│   └── sandbox.py           python isolation, web, files, tool protocol
├── memory/                  vector memory   (M4 ✅, M5 ✅)
│   ├── vector_memory.py     store, hybrid retrieval, facts, snapshots
│   ├── sync.py              Drive + Hub sync, resumable uploads, checksums
│   └── pdf.py               the standard-library PDF text extractor (M5 ✅)
├── training/                self-evolution  (M4 ✅)
│   ├── autopilot_evolver.py the idle watchdog + the preemption contract
│   ├── synthetic_data.py    mining + deterministic mutations
│   ├── self_play.py         fan-out drafting + critic scoring
│   ├── distributed_trainer.py federated, checkpointed, preemptible
│   ├── qlora.py             the maths (ridge, int8, SVD, merge)
│   ├── checkpoint_streamer.py push/pull the training state
│   └── prompt_evolver.py    mutate, benchmark, promote
├── mobile/                  the mission control  (M5 ✅)
│   ├── dashboard.py         the PWA + the API proxy
│   ├── notifier.py          Telegram + webhook alerts
│   ├── cli.py               the Termux command line (hc)
│   ├── qr.py                the dependency-free QR encoder
│   └── static/              the shell (HTML, CSS, JS, manifest, service worker, icon)
├── tests/                   live test-suite (331 tests, real HTTP + SSE)
├── scripts/                 smoke harnesses (M1 + M2 + M3 + M4 + M5)
├── install_all.sh           the one-command installer (M5 ✅)
├── .env.example             configuration template
├── ARCHITECTURE.md          this document
└── pyproject.toml
```
