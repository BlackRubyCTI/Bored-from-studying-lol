# ⚡ HYPER-CLUSTER OMEGA

**Zero-limit autonomous AI operating system.**
A headless GPU compute pool, a sub-second inference path, and a phone-first mission control.

> All five milestones are complete and verified:
> **M1** — control plane & resilient orchestrator · **M2** — self-benchmarking worker, mesh & tool sandbox ·
> **M3** — headless compute spawner, VRAM auto-scaler & speculative pairing ·
> **M4** — infinite memory & sleep-time self-evolution ·
> **M5** — the mobile mission control (PWA, phone alerts, Termux CLI, one-command installer) ·
> **M6** — truth, hygiene and a real jail (no silent model, no fake head URL, capability truth, kernel isolation).

---

## What exists right now

### Milestone 1 — control plane & resilient orchestrator

| Capability | Where |
|---|---|
| FastAPI control plane with SSE streaming | `head_node/app.py`, `head_node/serve.py` |
| Request brain: mode routing, speculative cascade, council, hold & requeue, failover | `head_node/orchestrator.py` |
| Latency-aware scheduling (TTFT + queue + load + cold-start, EWMA-corrected) | `head_node/scheduler.py` |
| Authoritative mesh registry with TTL countdowns and capacity claims | `head_node/worker_registry.py` |
| Hold-and-requeue buffer + request ledger (deferred work is never lost) | `head_node/hold_queue.py`, `head_node/request_store.py` |
| Multi-account credential vault (0600 file + env overlay, silent failover) | `core/credentials.py` |
| Free-API edge bridge (Groq / Cerebras / SambaNova / OpenRouter / Google AI Studio) | `head_node/edge_bridge.py` |
| Mixture-of-Agents council (fan-out → cross-critique → synthesis) | `head_node/council.py` |
| Versioned wire protocol shared by head node and workers | `core/protocol.py` |

### Milestone 2 — self-benchmarking worker, mesh & tool sandbox

| Capability | Where |
|---|---|
| Worker agent: model lifecycle, registration, heartbeats, hot-swap, drain, CLI | `worker/worker_agent.py` |
| Three real backends (llama.cpp / upstream OpenAI-compatible / deterministic simulator) | `worker/inference_engine.py` |
| **Self-benchmarking**: GPU, CPU, RAM, measured FP32 FLOPS, TTFT, decode throughput | `worker/telemetry.py` |
| Tailscale mesh automation: join, serve/funnel, peer discovery, reconnect, tag ACL | `worker/mesh.py` |
| Zero-touch provisioning: binary + model caching, SHA-256 verification, systemd/Termux service | `worker/bootstrap.sh` |
| Isolated Python execution (separate process, rlimits, network + process guards) | `tools/sandbox.py` |
| Live web search / fetch, file analysis, model-facing tool protocol | `tools/sandbox.py` |
| Agentic tool loop wired into the orchestrator (tool events are streamed) | `head_node/orchestrator.py` |
| 141 tests, including **real HTTP + SSE worker servers and live failover** | `tests/` |

### Milestone 3 — headless compute spawner, VRAM auto-scaler & speculation

| Capability | Where |
|---|---|
| Headless API provisioners: Kaggle kernels, HF Spaces, GitHub Actions, OCI Compute, edge APIs | `head_node/auto_spawner.py` |
| Multi-account credential vault with provider defaults, per-account metadata and failover | `core/credentials.py` |
| Live VRAM pooling across the whole worker mesh (measured, not guessed) | `head_node/model_selector.py` |
| Demand-aware GGUF upshift / downshift with hysteresis and a safety factor | `head_node/model_selector.py` |
| Speculative draft / verifier pairing with a cost model for the speed-up | `head_node/model_selector.py` |
| Pre-emptive TTL hot-swapper: replacement is provisioned *before* the session dies | `head_node/auto_spawner.py` |
| Worker session deadline reported in the registration metadata | `worker/worker_agent.py` |
| Node hardware factor (`tokens/s × B`) so a hot-swap keeps the throughput estimate honest | `head_node/worker_registry.py` |
| Seven new control-plane routes (`/v1/cluster/vram`, `/v1/models/plan`, `/v1/cluster/autoscale`, `/v1/speculative/pair`, `/v1/spawner`, `/v1/spawner/provision`, `/v1/spawner/rotate`) | `head_node/app.py` |
| 181 tests, including **live auto-scaling and zero-downtime hand-overs** | `tests/` |

### Milestone 4 — infinite memory & sleep-time self-evolution

| Capability | Where |
|---|---|
| Persistent hybrid (vector + keyword) memory with atomic snapshots and restore | `memory/vector_memory.py` |
| Durable-fact distillation from the conversations (`my name is …`, `we use …`, …) | `memory/vector_memory.py` |
| Google Drive + Hugging Face Hub sync: resumable uploads, SHA-256 verification, newest-snapshot pull | `memory/sync.py` |
| Automatic indexing of uploaded files (the API or a folder on disk) | `memory/vector_memory.py` |
| Retrieval context injected into every request, exchanges written back to the memory | `head_node/orchestrator.py` |
| **Sleep-time engine**: idle → mine → synthesise → self-play → train → merge → push → hot-reload | `training/autopilot_evolver.py` |
| Deterministic synthetic data (audience / domain / format / risk / counterfactual mutations) | `training/synthetic_data.py` |
| Self-play: fan-out drafting, critic scoring, similarity fallback, verified traces | `training/self_play.py` |
| Federated, checkpointed QLoRA training with a weighted gradient average and an auto-merge | `training/distributed_trainer.py` |
| The maths: ridge loss, int8 quantisation, truncated-SVD LoRA factorisation, `B @ A` merge | `training/qlora.py` |
| Checkpoint streaming to Drive and the Hub (the training state outlives the disk) | `training/checkpoint_streamer.py` |
| Prompt evolution: mutate, benchmark on held-out questions, promote only on a real gain | `training/prompt_evolver.py` |
| Atomic brain hot-reload on every worker (a broken artifact never takes a worker down) | `worker/brain.py` |
| **Strict preemption**: a chat interrupts the training and is served in **< 2 s** (measured: ~50 ms) | `training/autopilot_evolver.py` |
| Eight new control-plane routes (`/v1/memory*`, `/v1/evolution/*`) | `head_node/app.py` |
| 232 tests, including **live preemption of a running training** | `tests/` |

### Milestone 5 — the mobile mission control

| Capability | Where |
|---|---|
| **Dark-mode, touch-first PWA** (no framework, no build step, works offline) | `mobile/static/index.html`, `mobile/static/styles.css`, `mobile/static/app.js` |
| The dashboard server: the PWA **and** an API proxy (one origin for the phone) | `mobile/dashboard.py` |
| Streaming chat (Server-Sent Events proxied token by token) | `mobile/dashboard.py`, `mobile/static/app.js` |
| Voice input (Web Speech API) and voice output (`speechSynthesis`) | `mobile/static/app.js` |
| Drag-and-drop **PDF RAG upload** — the text layer is extracted on the fly | `memory/pdf.py`, `head_node/app.py` |
| A dependency-free PDF reader (Flate, PNG predictors, `ToUnicode` CMaps, Type0) | `memory/pdf.py` |
| Council Mode and Autopilot switches (the autopilot is stopped **immediately**) | `mobile/static/app.js`, `head_node/app.py` |
| Live GPU, VRAM and tokens-per-second telemetry | `mobile/static/app.js`, `head_node/app.py` |
| **Phone alerts**: Telegram and webhook pushes of the cluster events | `mobile/notifier.py` |
| The **Termux CLI** (`hc status`, `hc chat`, `hc upload`, `hc watch`, `hc qr`, …) | `mobile/cli.py` |
| A dependency-free **QR encoder** (the phone scans the dashboard address) | `mobile/qr.py` |
| The **one-command installer**: `.env` wizard, Drive + Hub wiring, systemd/Termux autostart, QR code | `install_all.sh` |
| 331 tests, including a **live** dashboard, a real PDF upload and the preemption race | `tests/` |

### Milestone 6 — truth, hygiene and a real jail

Four lies are gone. Each one was a real defect found by running the system, and each
fix is covered by a test that fails on the pre-M6 code.

| Capability | Where |
|---|---|
| **One canonical model env name.** The spawner exported `HYPER_WORKER_MODEL_FILENAME` while `bootstrap.sh` read `HYPER_WORKER_MODEL_FILE`, so every spawned node silently downloaded the default 7B. The canonical name is now read first, the legacy alias still works, and the generated bootstrap exports **both**. | `worker/bootstrap.sh`, `worker/config.py`, `head_node/auto_spawner.py` |
| **No silent substitution.** A requested model with no repo is a hard error; a simulator node downloads no weights; the 7B default is logged, never assumed. | `worker/bootstrap.sh` (`select_model`) |
| **No fake head URL.** With `HYPER_PUBLIC_BASE_URL` empty the spawner refuses to provision and returns a 4xx that names the variable and the reason — instead of handing a remote node `http://0.0.0.0:8080` and burning quota on a node that never checks in. | `core/config.py` (`public_head_url`), `core/errors.py` (`PublicURLUnsetError`), `head_node/auto_spawner.py`, `head_node/app.py` |
| **Capability truth on the wire.** `/v1/status` reports each worker's real `backend` plus top-level `answers_are_real`, `backends` and a `capability` block spelling out that simulator answers are **not model output**. | `core/protocol.py`, `head_node/worker_registry.py`, `head_node/orchestrator.py`, `head_node/app.py` |
| **`/v1/capacity/plan`** — reports mechanical capacity (VRAM pool, virtual cores from advertised `max_concurrency`) and attaches a model-grade claim **only** when a real backend is measured serving. Otherwise `claim: null` plus a `why_no_claim` explaining exactly what is missing. | `head_node/app.py` |
| **PWA capability badge** — "simulator — not model output" or "live model", driven by `/v1/status`, visible on the phone. | `mobile/static/index.html`, `mobile/static/app.js`, `mobile/static/styles.css` |
| **A real jail.** User + mount + PID + network namespaces, a fresh `/proc`, masked homes, a read-only root, and a seccomp syscall denylist installed inside the runner. Tiers are probed (`nsjail` → `bwrap` → `unshare` → `process`); the tier that actually ran is reported as `sandbox_isolation` on every result. | `tools/jail.py`, `tools/sandbox.py` |
| **Escape tests that prove the kernel holds** with the monkeypatched guards **disabled**: `ctypes` `socket(2)`/`execve(2)`, `importlib` reload of `os`, `/proc`, symlinks, read-only root, host homes. | `tests/test_m6_truth_and_jail.py` |

Honest limits, measured on this host: `nsjail` and `bwrap` are not installed here, so
the tier exercised is `unshare`. The read-only root remount is **not** available under
Docker's default seccomp profile; when the kernel refuses, the result says so in
`isolation_notes` rather than pretending. See `REVIEW.md` for the measurements.

---

## Quick start

```bash
cd /home/user/hyper-cluster

# 0. the whole thing (venv, .env, Drive + Hub, autostart, QR code)
./install_all.sh

# 1. environment (or the wizard above)
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env && nano .env        # set HYPER_API_KEY / HYPER_MESH_TOKEN

# 2. validate the installation
python -m head_node.serve --check

# 3. run the control plane *and* the mission control
python -m head_node.serve --host 0.0.0.0 --port 8080   # the API
python -m mobile.dashboard --host 0.0.0.0 --port 8082   # the phone
```

```bash
# 4. talk to it
curl -s localhost:8080/health | jq
curl -s localhost:8080/v1/status | jq '.workers, .edge, .hold_queue'
curl -s localhost:8080/v1/tools | jq '.tools[].function.name'
curl -N -X POST localhost:8080/v1/chat/stream \
     -H 'content-type: application/json' \
     -d '{"messages":[{"role":"user","content":"hello"}],"mode":"auto"}'
```

### Provision a worker (zero manual steps)

On any Linux box, Termux phone, Kaggle kernel or CI runner:

```bash
export HYPER_WORKER_HEAD_URL=https://your-head-node:8080
export HYPER_WORKER_MESH_TOKEN=…            # must match the head node
export HYPER_WORKER_TAILSCALE_AUTH_KEY=tskey-…   # optional, for the mesh

bash worker/bootstrap.sh --check      # validate the host, change nothing
bash worker/bootstrap.sh --install    # provision: deps → venv → binaries → model → service → verify
```

The script is idempotent: binaries and GGUF weights are cached under
`~/.hypercluster/cache`, every download is SHA-256 verified, the systemd unit
(or the Termux service) is (re)written from the resolved environment, and the
worker is health-checked before the script exits. A re-run of the same command
downloads **zero** bytes.

### Run a worker manually (no bootstrap)

```bash
HYPER_WORKER_HEAD_URL=http://127.0.0.1:8080 \
HYPER_WORKER_BACKEND=simulator \
python -m worker.worker_agent
```

The worker registers itself, benchmarks its own hardware, advertises the
measured capabilities and heartbeats until it is told to drain.

### Use the tool sandbox

```bash
python - <<'PY'
from tools.sandbox import ToolExecutor
executor = ToolExecutor()
print(executor.execute("python", {"code": "print(6*7)"}).output)
PY
```

A request that carries a `tools` array is routed into the agentic tool loop
automatically; the model emits one JSON object per round
(`{"tool": ..., "args": {...}}` or `{"final": ...}`), the head node executes it
in the sandbox and feeds the output back.

---

## The phone (Mission Control)

The dashboard is a PWA: dark by default, thumb-sized targets, no framework
and no build step.  It is served on its own port and proxies the API, so
the phone only ever talks to one origin.

```bash
python -m mobile.dashboard --host 0.0.0.0 --port 8082
# → open http://<your-ip>:8082 on the phone
# → or scan the QR code that install_all.sh prints
```

| On the phone | What it does |
|---|---|
| The chat | The answer streams in token by token (SSE through the proxy) |
| The 🎤 button | Voice input (Web Speech API) — the words land in the composer |
| The 🔊 switch | The answer is read out loud (`speechSynthesis`, any installed voice) |
| The 📎 button / a drop | A PDF is uploaded, its text layer is extracted and indexed at once |
| **Council** | The next answer is written by a panel (`mode: council`) |
| **Autopilot** | The sleep-time engine is switched on or off — off means the training stops *now* |
| The telemetry strip | Workers, free VRAM, tokens/second, the live model, the queue |

The terminal is a first-class citizen too — inside Termux, over SSH or
through `adb shell`:

```bash
hc status                       # the cluster at a glance
hc chat "what can you do?"      # one answer
hc ask                          # a conversation
hc workers                      # the mesh
hc watch                        # live telemetry
hc upload paper.pdf             # teach the cluster (the PDF is extracted)
hc memory "who is on call"      # what does it remember
hc evolution                    # the sleep-time engine
hc qr                           # the QR code of the dashboard
```

And the cluster can call the phone back:

```bash
HYPER_NOTIFIER_ENABLED=1
HYPER_NOTIFIER_TELEGRAM_TOKEN=…          # from @BotFather
HYPER_NOTIFIER_TELEGRAM_CHAT_ID=…
HYPER_NOTIFIER_WEBHOOK_URL=https://ntfy.sh/hyper-cluster
```

---

## Configuration

Everything is environment driven (`HYPER_*` prefix for the control plane,
`HYPER_WORKER_*` for workers) with a `.env` file — see `.env.example`.

| Key | Meaning |
|---|---|
| `HYPER_API_KEY` | bearer token the phone presents |
| `HYPER_MESH_TOKEN` | token workers present when registering |
| `HYPER_ADMIN_TOKEN` | token for drain / credential management |
| `HYPER_WORKER_TTL_SECONDS` | worker session lifetime (the rotator replaces nodes 10 min early) |
| `HYPER_HOLD_REQUEST_TTL_SECONDS` | how long a request may wait before it is deferred |
| `HYPER_SPECULATIVE_EDGE_PROVIDERS` | which free APIs act as the edge tier |
| `HYPER_COUNCIL_SIZE` | panel size for council mode |
| `HYPER_SANDBOX_ENABLED` / `HYPER_MAX_TOOL_ROUNDS` | tool sandbox on/off and the maximum tool rounds |
| `HYPER_WORKER_MODEL_REPO` / `_FILE` / `_SHA256` | which GGUF the bootstrap fetches and verifies |
| `HYPER_WORKER_BACKEND` | `auto`, `llama.cpp`, `upstream` or `simulator` |
| `HYPER_WORKER_TAILSCALE_AUTH_KEY` / `_TAGS` | tailnet join credentials and ACL tags |
| `HYPER_HANDOFF_LEAD_SECONDS` | how long before a session ends the replacement is provisioned |
| `HYPER_AUTOSCALER_ENABLED` / `_SAFETY_FACTOR` / `_MIN_DWELL_SECONDS` | the VRAM auto-scaler and its anti-thrash timer |
| `HYPER_SPAWNER_ENABLED` / `_PROVIDERS` / `_MIN_WORKERS` | the headless compute spawner and the provider order |
| `HYPER_SPAWNER_DEFAULT_TTL_SECONDS` | session lifetime the provisioner asks the provider for |
| `HYPER_CRED_KAGGLE_META` / `_HF_META` / `_GITHUB_META` / `_ORACLE_META` | per-account JSON metadata (owner, region, tenancy, compartment) |
| `HYPER_MEMORY_ENABLED` / `HYPER_MEMORY_DIR` | the persistent memory on/off and where it lives |
| `HYPER_DRIVE_FOLDER` / `HYPER_HUB_REPO` | where the memory snapshots and the checkpoints are mirrored |
| `HYPER_DRIVE_TOKEN` / `HYPER_HUB_TOKEN` | credentials for the Drive / Hub sync |
| `HYPER_EVOLUTION_ENABLED` / `_IDLE_SECONDS` / `_COOLDOWN_SECONDS` | the sleep-time engine, how long the cluster must be idle, and the gap between two cycles |
| `HYPER_EVOLUTION_PREEMPT_BUDGET_SECONDS` | the preemption budget (default 2 s) |
| `HYPER_EVOLUTION_DATASET_SIZE` / `_TRAIN_STEPS` / `_MAX_SHARDS` | how much the cluster learns per idle window |

Provider keys (`GROQ_API_KEY`, `CEREBRAS_API_KEY`, …) can be exported directly or registered
through `POST /v1/credentials`; the vault stores them with `0600` permissions and rotates
between accounts automatically.

---

## Verification

```bash
make test        # 386 tests: live HTTP/SSE servers, failover, sandbox, mesh, bootstrap, auto-scaler, memory, self-evolution, mobile, M6 truth + jail
make lint        # ruff over core/head_node/worker/tools/memory/training/mobile/tests/scripts
make smoke       # Milestone 1 smoke test (23 checks)
make smoke-m2    # Milestone 2 smoke test (31 checks)
make smoke-m3    # Milestone 3 smoke test (27 checks: hot-swap, scaling, zero-downtime hand-over)
make smoke-m4    # Milestone 4 smoke test (40 checks: memory, cloud sync, self-evolution, instant preemption)
make smoke-m5    # Milestone 5 smoke test (39 checks: the PWA, the proxy, the PDF, the alerts, the QR)
make smoke-m6    # Milestone 6 smoke test (62 checks: model hygiene, head URL, capability truth, the jail)
```

---

## Roadmap

1. ✅ **Control plane & resilient orchestrator**
2. ✅ **Self-benchmarking worker, mesh & tool sandbox**
3. ✅ **Headless API compute spawner & VRAM auto-scaler**
4. ✅ **Infinite memory & sleep-time self-evolution**
5. ✅ **Mobile PWA mission control, phone alerts & one-command installer**
6. ✅ **Truth, hygiene and a real jail** — capability truth, a refusing spawner, kernel-enforced isolation

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full blueprint.
