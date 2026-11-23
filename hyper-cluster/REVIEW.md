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

## Milestone 7 — real learning, measured (not asserted)

The M4 "trainer" was a numpy ridge regression. M7 replaces it with real QLoRA and
keeps the ridge trainer as a capability-detected fallback. Nothing below is a
projection: every number came from a run in this environment.

### Environment (the limits that shape every claim)

| Fact | Value | Consequence |
|---|---|---|
| `torch` | 2.15.0+cu132 | installed in `.venv` only (the `[training]` extra) |
| `transformers` / `peft` / `accelerate` | 5.24.0 / 0.22.0 / 1.16.0 | the QLoRA stack runs |
| `bitsandbytes` | 0.50.2 | **imports, but unusable** |
| CUDA | **`torch.cuda.is_available() == False`** | 4-bit NF4 **cannot run**; `quantized_available()` is `False` |
| `huggingface.co` | **unreachable** (PyPI is fine) | **no pretrained weights**; base model is a config-built 115k model |

### What was measured

| Claim | Measurement |
|---|---|
| A real adapter is produced | 4 files, **24,951 bytes**, sha256 `b296eaa8b5c3…`, incl. `adapter_model.safetensors` + `adapter_config.json` |
| The LoRA tensors are really trained | 8 LoRA tensors, **6/8 non-zero** after training |
| The adapter reloads | `PeftModel.from_pretrained` into a fresh `LlamaForCausalLM` restores the trained values exactly (a `lora_B` filled with 1.0 reads back `mean=1.000`) |
| The adapter fits the serving path | bridged to `BrainArtifact`, **`worker/brain.py::Brain.load()` → `loaded: True`**, revision 1 |
| Held-out learning is measured | loss **5.4765 → 4.7572** (Δ **+0.7193**), 3 held-out examples, prompt-grouped and deterministic |
| The delta is stored with the artifact | `hyper_metrics.json` next to the adapter, `eval.loss_delta` read back from disk |
| Preemption still holds | a chat stopped a QLoRA run in **205 ms** (budget **2,000 ms**) |
| Torch is optional | `make test`: **414 passed, 1 skipped** with torch; **402 passed, 13 skipped** without (skip reasons name the missing extra) |

Trainable parameters on the self-test model: **4,096 of 118,xxx (3.4 %)** with
r=8 / α=16 on `q_proj` and `v_proj`.

### Defects found and fixed while building M7

1. **A silent no-op in the wrapper.** `AutoTrainer` delegated nothing, so
   `trainer.shard_fn = ...` (what the autopilot and the M4/M7 tests do) wrote an
   attribute the real trainer never saw. The preemption test caught it — training
   never slowed down, so the interrupt had nothing to interrupt. `AutoTrainer` now
   forwards attribute reads *and* writes to the trainer it selected.
2. **The chooser could crash the control plane.** A broken torch install raised
   out of `torch_available()`/`quantized_available()` during startup. Probing is
   now fully guarded and degrades to the math backend.
3. **A real adapter was invisible to the cloud.** The checkpoint streamer only
   listed `.npz` files and only pushed numpy matrices, so every QLoRA run's
   adapter silently never left the machine. Added `push_bytes()`, `pack()`
   (one `.tar.gz`) and a listing filter that accepts adapter archives.
4. **Checkpoints were under-counted.** The torch backend saved a real adapter but
   never incremented `report.checkpoints`, so the cloud/checkpoint bookkeeping
   reported zero for a run that had, in fact, checkpointed. The M4 smoke caught it.
5. **A tiny model with the wrong vocabulary** (256 ids vs the 259-id byte
   tokenizer) crashed in `F.embedding` with `IndexError: index out of range`. The
   config is now sized from `BYTE_VOCAB_SIZE` and asserted against the tokenizer.

### Clear-eyed about the claim

* **This is not evidence that a production model improved.** The base model is a
  115k-parameter self-test model built from a config. It proves the *pipeline* is
  real — adapter produced, measured, reloaded, served, preempted — not that
  7B-model quality rose. Every such report carries that sentence in `notes`, and
  `test_the_report_labels_the_offline_self_test_model` fails if it is removed.
* **A loss drop on 3 held-out examples is a weak signal.** It is a real
  measurement of a real split, and it is reported with its sample size; it is not
  a benchmark. `exact_match` and `token_f1` stayed at `0.0`, which is the honest
  result for a byte-level 115k model, and the report shows it rather than showing
  only the loss.
* **4-bit is unverified.** The code path is selected automatically on a CUDA host,
  but it was not executed here. REVIEW would be lying if M7 claimed QLoRA-4bit
  throughput numbers.
* **The fallback is not a toy that gets deleted.** On any machine without the
  training extra, `AutoTrainer` selects the numpy trainer and says so with a
  reason string. That path is covered by tests in both directions.

## Suggested next steps

1. Wire a real backend (`HYPER_WORKER_BACKEND=llama.cpp` or `upstream`) and re-run M2/M4 against real tokens.
2. ~~Move the tool sandbox into a container / gVisor~~ — done in M6 (`unshare` + seccomp + RO root,
   with the old guards kept as the fast path). Remaining: run it under `nsjail` on a host that has it,
   and move the denylist to an allowlist.
3. Add a load test: 50 concurrent chats, 3 workers, assert p95 TTFT and zero dropped requests.
4. Replace the hashed embedder with a real one (the interface is already pluggable) and measure recall.
5. Point `HYPER_EVOLUTION_BASE_MODEL` at a cached 7B on a CUDA host and re-run
   `make smoke-m7` — that is the first run whose eval delta says something about a
   *production* model, and the first that exercises 4-bit NF4 for real.
6. Grow the eval set beyond 12 hand-written pairs before trusting `exact_match`;
   the harness (grouped split, deterministic seed) is ready for it.
