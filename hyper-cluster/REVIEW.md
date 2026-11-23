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
| Test suite | **386 passed, 1 skipped** after M6 (341 passed at the M5 baseline; skip = no internet here) |
| Smoke suites M1–M6 | **222/222 checks** (160/160 at the M5 baseline) |
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
  OpenAI-compatible server (Ollama/vLLM) for real output. **As of M6 the system says this out loud:**
  `/v1/status` returns `answers_are_real: false`, `/v1/capacity/plan` refuses to attach a
  model-grade claim, and the PWA shows a "simulator — not model output" badge.
* ~~**The sandbox is not a jail.**~~ **Fixed in M6** — see the M6 section below. The kernel
  boundary (namespaces + seccomp + read-only root) is now the primary defence, and the old
  monkeypatched guards are defence in depth with an explicit `process` tier that labels itself.
* **The "training" is real but small**: a ridge-regression LoRA on hashed features (dim 128, rank 8,
  vocab 64), not a transformer fine-tune. It genuinely improves its own loss (0.0156 → 6.1e-05), but
  it will not make a 7B model smarter.
* **Memory embeddings are hashed bag-of-features**, not a neural embedder. Retrieval works; semantics
  are shallow.
* **Headless "compute spawner" defaults off**, needs provider accounts, and depends on providers that
  forbid this in their ToS.
* The tailnet/mesh automation needs a real Tailscale auth key to exercise; here it correctly
  degraded ("no tailnet") instead of crashing.

## Milestone 6 — truth, hygiene and a real jail (measured)

Method: everything below was produced by running the code on this machine, not by reading it.

```
$ make test          → 386 passed, 1 skipped in 124 s      (was 341 passed before M6; +45 new)
$ make lint          → All checks passed!
$ make smoke         → 23/23
$ make smoke-m2      → 31/31
$ make smoke-m3      → 27/27
$ make smoke-m4      → 40/40
$ make smoke-m5      → 39/39
$ make smoke-m6      → 62/62                               → 222/222 smoke checks total
```

The skip is unchanged from the pre-M6 baseline (no outbound internet in this environment).
No test was deleted or weakened: the three spawner tests that had encoded the bug by running
with the default `0.0.0.0` bind were given a routable `public_base_url`, and the *refusal* they
had been masking now has its own dedicated tests.

### Defect 1 — silent 7B spawns (fixed)

| Measurement | Value | Method |
|---|---|---|
| Model actually loaded for a requested 3B model, pre-M6 | `qwen2.5-7b-instruct-q4_k_m.gguf` | revert `select_model`, source `bootstrap.sh` with `HYPER_WORKER_MODEL_FILENAME` set |
| Model actually loaded post-M6 | `llama-3.2-3b-instruct-q4_k_m.gguf` | same command, M6 bootstrap |
| M6 tests that fail when the fix is reverted | **5** | `pytest -k "canonical or legacy or seven_b or simulator_backend or refused"` against the pre-M6 `bootstrap.sh` |

Regression proof, verbatim from the run:

```
FAILED test_bootstrap_reads_the_canonical_filename
FAILED test_canonical_name_wins_over_the_legacy_alias
FAILED test_no_model_requested_is_not_a_silent_seven_b
FAILED test_simulator_backend_downloads_no_model
FAILED test_bootstrap_writes_the_canonical_name_into_the_env_file
```

### Defect 2 — the `http://0.0.0.0:8080` head URL (fixed)

`Settings(head_host="0.0.0.0", public_base_url="")` now raises `PublicURLUnsetError`
(`http_status=409`, `code="public_base_url_unset"`) instead of returning a URL.
Measured effect on scale-out: with the floor at 2 workers and one usable credential, the
spawner emits **1 refused action** and provisions **0 instances** (`spawner.instances == {}`),
so no provider quota is spent. Before the fix the same path provisioned nodes pointing at an
unreachable address.

### Defect 3 — capability truth (added)

With two registered `simulator` workers and no real model, through the *phone* path
(`dashboard → /api → head`, i.e. the URLs the PWA actually uses):

```
GET /api/v1/status
  answers_are_real: False
  backends: ['simulator']
  capability.label: 'simulator'
  capability.detail: '... Answers are structurally valid but are NOT model output ...'

GET /api/v1/capacity/plan
  model_grade: False      claim: None
  virtual_vram_gb: {'total': 48.0, 'free': 36.0}
  virtual_cores.concurrent_slots: 4
  why_no_claim: 'No measurement behind a model-grade claim: the answering backend is
                 simulator and aggregate tokens/sec is 0.00.'
```

Registering one healthy `llama.cpp` worker with `max_concurrency=4` flips
`answers_are_real` to `true`, `backends` to `["llama.cpp"]`, the badge label to `"live model"`,
and the plan's `virtual_cores.concurrent_slots` to `4` — i.e. the numbers follow the measurement,
and the claim appears only when there is one.

### Defect 4 — the sandbox was not a jail (fixed)

Tier availability on this host, **probed** rather than assumed:

```
available_tiers() → ['unshare', 'process']
detect_isolation() → 'unshare'
```

`nsjail`, `bwrap`, `firejail`, `docker` and `podman` are not installed here, so the tier
actually exercised is `unshare`: user + mount + PID + network namespaces, fresh `/proc`,
masked homes, read-only root, plus a seccomp BPF denylist installed by the runner.

The decisive comparison — **the same escape attempts with the monkeypatched guards disabled**,
so only the kernel can stop them:

| Escape attempt (guards OFF) | `process` tier (pre-M6 design) | `unshare` tier (M6) |
|---|---|---|
| `ctypes` → `socket(2)` | **fd `3` returned — a real socket** | `-1`, `errno 1` (EPERM) |
| `ctypes` → `execve(2)` `/bin/sh -c touch …` | **file created on the host** | `-1`, `errno 1` (EPERM), no file |
| `open('/etc/shadow')` | `PermissionError` | `PermissionError` |
| `open('/etc/x','w')` | `PermissionError` | `OSError: [Errno 30] Read-only file system` |
| `/proc` visible PIDs | host process table | **1** (the jail itself); `HYPER_API_KEY` absent from `/proc/1/environ` |
| symlink `→ /etc` then read `shadow` | reachable path | `PermissionError` |

So the pre-M6 "sandbox" genuinely leaked: a `ctypes` handle walked past the `socket` monkeypatch
and got a working socket, and an `execve` from `ctypes` executed on the host. Both are now denied
by the kernel.

Cost of the boundary (same machine, `print(1+1)`):

| Metric | Value |
|---|---|
| Cold start (first run, includes the namespace + seccomp setup) | **100 ms** |
| Warm median over 5 runs | **99 ms** |
| Read-only root achieved | **yes** (`read_only_root: true`) |

### What is still not real (honest limits)

* **No `nsjail` or `bwrap` here.** The `container` tier is implemented and probed but untested on
  this machine — there is no binary to run. On a host with `nsjail` installed it is selected first.
  The reported string is therefore `unshare`, never `container`.
* **Seccomp is a denylist, not an allowlist.** It blocks the syscalls that turn a Python-level
  escape into a real one (exec/mount/ptrace/socket/bpf/module loading/keyring), and everything else
  is allowed so CPython keeps working. An allowlist would be stronger and is the obvious next step;
  a denylist is what could be validated honestly in this milestone. Also note `io_uring` is not
  denied — `io_uring_setup` can bypass seccomp on some kernels.
* **The read-only root is host-dependent.** Docker's default profile refuses the locked
  `remount,ro,bind /`, and the code then records
  `"root filesystem was not remounted read-only on this host"` in `isolation_notes` and reports
  `read_only_root: false`. It succeeded here; it will not everywhere.
* **A network namespace is not a firewall.** Egress is cut by having no interfaces; DNS and
  `socket(2)` fail. That is containment, but it is not egress policy.
* **The jail still runs CPython, not a micro-VM.** A CPython memory-safety bug is still a CPython
  memory-safety bug. gVisor/Firecracker remain strictly stronger.
* **`/v1/capacity/plan` is capacity, not a SLA.** It reports measured VRAM and advertised
  concurrency slots. It does **not** publish a p95 TTFT promise — that is M12's load-tested claim,
  and inventing one here would be exactly the kind of number this milestone exists to remove.

## Suggested next steps

1. Wire a real backend (`HYPER_WORKER_BACKEND=llama.cpp` or `upstream`) and re-run M2/M4 against real tokens.
2. ~~Move the tool sandbox into a container / gVisor~~ — done in M6 (`unshare` + seccomp + RO root,
   with the old guards kept as the fast path). Remaining: run it under `nsjail` on a host that has it,
   and move the denylist to an allowlist.
3. Add a load test: 50 concurrent chats, 3 workers, assert p95 TTFT and zero dropped requests.
4. Replace the hashed embedder with a real one (the interface is already pluggable) and measure recall.
