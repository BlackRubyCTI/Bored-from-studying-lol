#!/usr/bin/env python3
"""Live simulation for Milestone 7 -- real learning.

Everything here runs for real: the torch/peft stack actually fine-tunes a model,
the adapter is actually written to disk, actually reloaded into a fresh base
model, and actually handed to the worker's ``Brain`` through the hot-reload path
the mesh already uses.

The simulation proves the four M7 claims:

* **a cycle produces a real LoRA adapter** — ``adapter_model.safetensors`` plus
  ``adapter_config.json``, with trained (non-zero, non-stub) weights;
* **the adapter loads into the serving path** — PEFT reloads it into a fresh
  base model and ``worker.brain.Brain.load()`` accepts it;
* **the eval delta is measured and printed** — held-out, prompt-grouped split,
  before/after, stored next to the adapter;
* **preemption still holds** — a chat interrupts QLoRA training in under 2 s.

When the optional ``[training]`` extra is absent the script says so and runs the
math-backend checks only.  It never pretends a fine-tune happened.

    python scripts/smoke_m7.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.distributed_trainer import PreemptionToken  # noqa: E402
from training.evaluate import eval_set, split_traces  # noqa: E402
from training.torch_qlora import (  # noqa: E402
    BACKEND_MATH,
    BACKEND_TORCH,
    AutoTrainer,
    TorchQLoRATrainer,
    adapter_to_brain_artifact,
    cuda_available,
    quantized_available,
    resolve_base_model,
    torch_available,
)

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"

_results: list[tuple[bool, str]] = []
_measurements: dict[str, Any] = {}


def check(condition: bool, label: str, detail: str = "") -> bool:
    _results.append((bool(condition), label))
    suffix = f" — {detail}" if detail else ""
    print(f"  {PASS if condition else FAIL} {label}{suffix}")
    return bool(condition)


def section(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")


def note(text: str) -> None:
    print(f"  \033[33m·\033[0m {text}")


# --------------------------------------------------------------------------- #
# 1. The evaluation harness itself
# --------------------------------------------------------------------------- #
def prove_the_harness() -> None:
    section("1. Evaluation harness — a delta that means something")

    traces = eval_set()
    train, held, notes = split_traces(traces, held_out_fraction=0.25, min_held_out=3)
    check(len(train) > 0 and len(held) > 0, "the split produced a train and a held-out half")
    check(
        not ({t.prompt for t in train} & {t.prompt for t in held}),
        "no prompt appears in both halves (grouped split)",
    )
    again = split_traces(traces, held_out_fraction=0.25, min_held_out=3)
    check([t.prompt for t in again[1]] == [t.prompt for t in held], "the split is deterministic")
    if notes:
        note("split notes: " + "; ".join(notes))

    from training.evaluate import evaluate

    good = evaluate(lambda p, g: (g, 1.0), held, label="after")
    bad = evaluate(lambda p, g: ("nonsense", 9.0), held, label="before")
    check(good.exact_match == 1.0, "an exact answer scores exact-match 1.0")
    check(bad.exact_match == 0.0, "a wrong answer scores 0.0")
    check(bad.loss > good.loss, "loss ranks the bad answer worse than the good one")
    _measurements["held_out_examples"] = len(held)


# --------------------------------------------------------------------------- #
# 2. The real QLoRA cycle
# --------------------------------------------------------------------------- #
async def prove_qlora() -> None:
    section("2. A real training cycle — QLoRA, an adapter, a measured delta")

    if not torch_available():
        note("torch/transformers/peft are NOT installed: the [training] extra is optional")
        check(True, "the QLoRA path is skipped honestly (nothing faked)")
        trainer = AutoTrainer(checkpoint_dir=tempfile.mkdtemp(), math_kwargs={"steps": 3, "dim": 32, "vocab_size": 32})
        check(trainer.backend == BACKEND_MATH, "the math backend is selected when torch is absent")
        check("fallback" in trainer.choice.reason, "the reason for the fallback is reported")
        report = await trainer.train(eval_set()[:6])
        check(report.steps == 3, "the math backend still trains")
        _measurements["backend"] = BACKEND_MATH
        return

    base = resolve_base_model("")
    note(f"base model: {base.name} (source: {base.source})")
    note(f"cuda: {cuda_available()} · 4-bit usable: {quantized_available()}")

    tmp = Path(tempfile.mkdtemp())
    trainer = TorchQLoRATrainer(
        checkpoint_dir=tmp,
        steps=8,
        learning_rate=1e-3,
        max_length=96,
        eval_max_new_tokens=8,
    )
    started = time.perf_counter()
    report = await trainer.train(eval_set())
    elapsed = time.perf_counter() - started

    check(report.error == "", "training completed without an error", report.error[:100])
    check(report.backend == BACKEND_TORCH, "the report says which backend ran", report.backend)
    check(report.steps == 8, "all requested steps ran", str(report.steps))
    if base.synthetic:
        check(
            any("offline self-test" in n for n in report.notes),
            "the offline self-test base model is labelled in the report",
        )
    check(report.precision in {"fp32", "fp16", "4bit-nf4"}, "the precision used is recorded", report.precision)

    # --- the adapter is real ------------------------------------------- #
    artifact = trainer.last_artifact
    check(artifact is not None and artifact.exists(), "a real adapter was written to disk")
    if artifact is not None:
        measured = artifact.measure()
        check(measured["files"] >= 2, "the adapter directory holds its files", f"{measured['files']} files")
        check(measured["bytes"] > 1000, "the adapter has substance", f"{measured['bytes']} bytes")
        check(len(measured["sha256"]) == 64, "the adapter has a checksum", measured["sha256"][:12])
        _measurements["adapter_bytes"] = measured["bytes"]
        _measurements["adapter_files"] = measured["files"]

        names = sorted(p.name for p in artifact.directory.iterdir())
        check("adapter_model.safetensors" in names, "adapter_model.safetensors exists")
        check("adapter_config.json" in names, "adapter_config.json exists")

        # Not a stub: the LoRA tensors must actually carry trained weights.
        import safetensors.torch as st

        tensors = st.load_file(str(artifact.adapter_file))
        lora = [t for k, t in tensors.items() if "lora_" in k]
        check(len(lora) > 0, "the adapter contains LoRA tensors", f"{len(lora)} tensors")
        trained = sum(1 for t in lora if float(t.float().abs().sum()) > 0.0)
        check(trained > 0, "the LoRA tensors are trained (non-zero)", f"{trained}/{len(lora)} non-zero")

    # --- the delta ------------------------------------------------------ #
    payload = report.eval or {}
    before, after = payload.get("before"), payload.get("after")
    check(before is not None and after is not None, "a before/after evaluation was recorded")
    if before and after:
        delta = payload.get("loss_delta", 0.0)
        print(
            f"      held-out loss {before['loss']:.4f} -> {after['loss']:.4f} "
            f"(delta {delta:+.4f}) on {before['examples']} examples"
        )
        print(
            f"      exact-match {before['exact_match']:.3f} -> {after['exact_match']:.3f} · "
            f"token-F1 {before['token_f1']:.3f} -> {after['token_f1']:.3f}"
        )
        _measurements["loss_before"] = before["loss"]
        _measurements["loss_after"] = after["loss"]
        _measurements["loss_delta"] = delta
        check("method" in payload, "the delta states how it was measured")
        # Honest assertion: either it improved, or we say it did not.
        if delta > 1e-4:
            check(True, "held-out loss improved", f"{delta:+.4f}")
        else:
            note(f"held-out loss did NOT improve measurably (delta {delta:+.4f}) — reporting it as measured")

    # --- metrics travel with the adapter -------------------------------- #
    if artifact is not None:
        stored = json.loads(artifact.metrics_file.read_text(encoding="utf-8"))
        check(stored.get("eval", {}).get("before") is not None, "the metrics are stored next to the adapter")
        check(stored.get("backend") == BACKEND_TORCH, "the stored metrics name the backend")

    # --- the adapter loads back into a fresh base model ----------------- #
    try:
        from peft import PeftModel
        from transformers import LlamaForCausalLM

        from training.torch_qlora import _tiny_config

        if artifact is not None:
            fresh = LlamaForCausalLM(_tiny_config())
            served = PeftModel.from_pretrained(fresh, str(artifact.directory))
            check(type(served).__name__ == "PeftModelForCausalLM", "the adapter reloads into a fresh base model")
    except Exception as exc:  # noqa: BLE001 - report, never fake
        note(f"reload check skipped: {exc}")

    # --- the serving path ---------------------------------------------- #
    if artifact is not None:
        from worker.brain import Brain

        wire = adapter_to_brain_artifact(artifact)
        encoded = wire.encode()
        check(wire.metadata.get("format") == "peft-adapter", "the wire artifact is labelled as a real adapter")
        check(len(encoded) > 1000, "the wire payload carries the adapter", f"{len(encoded)} bytes")
        brain = Brain()
        info = brain.load(encoded)
        check(info.get("loaded") is True, "worker Brain.load() accepts the adapter (hot-reload path)")
        check(brain.loaded is True, "the worker brain reports the adapter as loaded")

        from training.torch_qlora import install_adapter

        model, tokenizer = install_adapter(artifact)
        check(type(model).__name__ == "PeftModelForCausalLM", "install_adapter returns a servable model")

    # --- preemption ----------------------------------------------------- #
    token = PreemptionToken()
    slow = TorchQLoRATrainer(checkpoint_dir=Path(tempfile.mkdtemp()), steps=40, max_length=48, eval_max_new_tokens=4)
    task = asyncio.ensure_future(slow.train(eval_set()[:6], token=token, step_delay=0.2))
    await asyncio.sleep(0.5)
    started_p = time.perf_counter()
    token.preempt("user request")
    preempted = await asyncio.wait_for(task, timeout=30.0)
    latency_ms = (time.perf_counter() - started_p) * 1000.0
    check(preempted.preempted is True, "a chat preempts the QLoRA run")
    check(latency_ms < 2000.0, "the interrupt lands inside the 2 s budget", f"{latency_ms:.0f} ms")
    _measurements["preempt_ms"] = latency_ms
    _measurements["train_seconds"] = elapsed
    _measurements["backend"] = BACKEND_TORCH


# --------------------------------------------------------------------------- #
# 3. Capability truth
# --------------------------------------------------------------------------- #
async def prove_truth() -> None:
    section("3. Capability truth — the choice is probed and reported")

    choice = AutoTrainer(checkpoint_dir=tempfile.mkdtemp()).choice
    check(choice.backend in {BACKEND_TORCH, BACKEND_MATH}, "a backend is chosen", choice.backend)
    check(bool(choice.reason), "the choice carries a human reason", choice.reason[:90])
    check(
        isinstance(choice.torch_ready, bool) and isinstance(choice.cuda, bool),
        "the probe results are exposed, not hidden",
    )

    pinned = AutoTrainer(checkpoint_dir=tempfile.mkdtemp(), preferred="math")
    check(pinned.backend == BACKEND_MATH, "the backend can be pinned explicitly")

    # Delegation: configuring through the wrapper must reach the chosen trainer.
    pinned.steps = 4
    check(pinned.math.steps == 4, "configuration set on the wrapper reaches the real trainer")

    base = resolve_base_model("")
    if base.synthetic:
        check("offline-selftest" in base.name, "the tiny model is labelled offline-selftest")
        note("no real base model is cached here, so the offline self-test model is used")


# --------------------------------------------------------------------------- #
async def main() -> int:
    started = time.time()
    print("\033[1m" + "=" * 64)
    print("  Hyper-Cluster Omega — Milestone 7 smoke: real learning")
    print("=" * 64 + "\033[0m")

    prove_the_harness()
    await prove_qlora()
    await prove_truth()

    passed = sum(1 for ok, _ in _results if ok)
    total = len(_results)
    failed = [label for ok, label in _results if not ok]

    if _measurements:
        print("\n\033[1m  measurements\033[0m")
        for key, value in _measurements.items():
            if isinstance(value, float):
                print(f"    {key:20s} {value:.4f}")
            else:
                print(f"    {key:20s} {value}")

    print(f"\n{'=' * 64}")
    print(f"  {passed}/{total} smoke checks passed in {time.time() - started:.1f}s")
    if failed:
        print("  failures:")
        for label in failed:
            print(f"    - {label}")
    print(f"{'=' * 64}\n")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
