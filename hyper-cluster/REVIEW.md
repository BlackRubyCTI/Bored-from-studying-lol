# Review — hyper-cluster (virtual "supercomputer")

Reviewed at commit `c04d8b9` by actually running it: head node on :8080, a mesh worker on :8081,
the phone dashboard on :8082, on this machine. Everything below is observed, not inferred.

## Verdict

**Not a supercomputer, but a genuinely well-built, working system.** ~27k lines of Python,
no filler: it boots, it routes, it streams, it remembers, it recovers. The README is unusually
honest (it says explicitly that free-tier GPU runners — Kaggle, HF Spaces, Oracle, Colab — are
against those providers' ToS, and that the operator takes the risk).
What it is *not*: a machine with more compute than one box. It is an **orchestrator** that pools
machines you already have and adds a free-API tier on top.

## What verified clean

| Area | Evidence |
|---|---|
| Test suite | 334 tests → **341 passed, 1 skipped** (skip = no internet here) |
| Smoke suites M1–M5 | **160/160 checks** |
| Lint | ruff clean over all 10 packages |
| End-to-end chat | SSE streams token by token through head node *and* through the phone proxy |
| Worker mesh | registers, self-benchmarks (FP32 0.216 TFLOPS, latency to head ~3 ms), heartbeats |
| Backpressure | 10 concurrent chats on a 2-slot worker → holds, then **202-style deferral, zero 5xx** (law L1 holds) |
| Memory / RAG | uploaded a real PDF via the dashboard proxy → text extracted, secret word retrieved by hybrid search |
| Self-evolution | preemption of a running training measured at ~26 ms against a 2 s budget; brain hot-reloaded into the worker |
| QR codes | OpenCV decoded all four dashboard URLs correctly |
| Auth | API/mesh/admin tiers behave; dashboard proxy passes auth through and rejects without a key (401) |
| Secrets | redacted in `/v1/status`; credential vault 0600 |

## Defects found and fixed (see PR #3)

1. **Hold queue never drained** — `/v1/status` reported a growing queue after bursts; entries could be replayed twice.
2. **Tool sandbox read the host** — `open()` bypassed the guards: read the head node's `.env` (all API keys), wrote to `/tmp`.
3. **Autopilot never slept** — full train + hot-reload every ~2 s on an idle box.
4. **Training stopped learning after cycle 1** — `steps: 0`, identical artifact hash forever.

## Things to be clear-eyed about (not bugs, but the "supercomputer" claim)

* **The answers you see here come from a *simulator*.** Without a GGUF + `llama-cpp-python` on the
  box, `HYPER_WORKER_BACKEND=auto` falls back to a deterministic generator. Everything downstream
  (routing, tools, memory, training) is real; the prose is not. Point it at a real GGUF or an
  OpenAI-compatible server (Ollama/vLLM) for real output.
* **The sandbox is not a jail.** It is in-process Python: POSIX rlimits (CPU/AS/FSIZE/NPROC/NOFILE)
  plus monkeypatched `socket`/`os`/`subprocess`/`open`. Network and subprocesses are genuinely
  blocked; getattr/property tricks can still reach import machinery. For untrusted input, run it in
  a container/gVisor/nsjail.
* **The "training" is real but small**: a ridge-regression LoRA on hashed features (dim 128, rank 8,
  vocab 64), not a transformer fine-tune. It genuinely improves its own loss (0.0156 → 6.1e-05), but
  it will not make a 7B model smarter.
* **Memory embeddings are hashed bag-of-features**, not a neural embedder. Retrieval works; semantics
  are shallow.
* **Headless "compute spawner" defaults off**, needs provider accounts, and depends on providers that
  forbid this in their ToS.
* The tailnet/mesh automation needs a real Tailscale auth key to exercise; here it correctly
  degraded ("no tailnet") instead of crashing.

## Suggested next steps

1. Wire a real backend (`HYPER_WORKER_BACKEND=llama.cpp` or `upstream`) and re-run M2/M4 against real tokens.
2. Move the tool sandbox into a container / gVisor; keep the current guards as the fast path.
3. Add a load test: 50 concurrent chats, 3 workers, assert p95 TTFT and zero dropped requests.
4. Replace the hashed embedder with a real one (the interface is already pluggable) and measure recall.
