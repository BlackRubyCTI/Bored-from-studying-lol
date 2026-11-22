# ⚡ HYPER-CLUSTER OMEGA

**Zero-limit autonomous AI operating system.**
A headless GPU compute pool, a sub-second inference path, and a phone-first mission control.

> Milestone 1 of 5 — *Control Plane & Resilient Orchestrator* — is complete and verified.

---

## What exists right now (Milestone 1)

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
| 70 tests, including **real HTTP + SSE worker servers** | `tests/` |

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
curl -N -X POST localhost:8080/v1/chat/stream \
     -H 'content-type: application/json' \
     -d '{"messages":[{"role":"user","content":"hello"}],"mode":"auto"}'
```

Run the test-suite (spins up real worker servers):

```bash
.venv/bin/python -m pytest tests/ -q
```

---

## Configuration

Everything is environment driven (`HYPER_*` prefix) with a `.env` file — see `.env.example`.
The most important keys:

| Key | Meaning |
|---|---|
| `HYPER_API_KEY` | bearer token the phone presents |
| `HYPER_MESH_TOKEN` | token workers present when registering |
| `HYPER_ADMIN_TOKEN` | token for drain / credential management |
| `HYPER_WORKER_TTL_SECONDS` | worker session lifetime (the rotator replaces nodes 10 min early) |
| `HYPER_HOLD_REQUEST_TTL_SECONDS` | how long a request may wait before it is deferred |
| `HYPER_SPECULATIVE_EDGE_PROVIDERS` | which free APIs act as the edge tier |
| `HYPER_COUNCIL_SIZE` | panel size for council mode |

Provider keys (`GROQ_API_KEY`, `CEREBRAS_API_KEY`, …) can be exported directly or registered
through `POST /v1/credentials`; the vault stores them with `0600` permissions and rotates
between accounts automatically.

---

## Roadmap

1. ✅ **Control plane & resilient orchestrator**
2. ⏳ Self-benchmarking worker, mesh & tool sandbox
3. ⏳ Headless API compute spawner & VRAM auto-scaler
4. ⏳ Infinite memory & sleep-time self-evolution
5. ⏳ Mobile PWA mission control, phone alerts & one-command installer

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full blueprint.
