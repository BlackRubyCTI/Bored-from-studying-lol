# ⚡ HYPER-CLUSTER OMEGA

**Zero-limit autonomous AI operating system.**
A headless GPU compute pool, a sub-second inference path, and a phone-first mission control.

> Milestones 1 and 2 of 5 are complete and verified:
> **M1** — control plane & resilient orchestrator · **M2** — self-benchmarking worker, mesh & tool sandbox.

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

---

## Quick start

```bash
cd /home/user/hyper-cluster

# 1. environment
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env && nano .env        # set HYPER_API_KEY / HYPER_MESH_TOKEN

# 2. validate the installation
python -m head_node.serve --check

# 3. run the control plane
python -m head_node.serve --host 0.0.0.0 --port 8080
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

Provider keys (`GROQ_API_KEY`, `CEREBRAS_API_KEY`, …) can be exported directly or registered
through `POST /v1/credentials`; the vault stores them with `0600` permissions and rotates
between accounts automatically.

---

## Verification

```bash
make test        # 141 tests: live HTTP/SSE servers, failover, sandbox, mesh, bootstrap
make lint        # ruff over core/head_node/worker/tools/tests
make smoke       # Milestone 1 smoke test (23 checks)
make smoke-m2    # Milestone 2 smoke test (31 checks)
```

---

## Roadmap

1. ✅ **Control plane & resilient orchestrator**
2. ✅ **Self-benchmarking worker, mesh & tool sandbox**
3. ⏳ Headless API compute spawner & VRAM auto-scaler
4. ⏳ Infinite memory & sleep-time self-evolution
5. ⏳ Mobile PWA mission control, phone alerts & one-command installer

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full blueprint.
