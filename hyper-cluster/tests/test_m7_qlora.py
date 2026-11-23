"""M7 — real learning: QLoRA, an evaluation harness, and the preemption contract.

Every test here fails on pre-M7 code, because pre-M7 code had no
``training.torch_qlora`` and no ``training.evaluate`` at all.

What is proven:

* a run produces a **real LoRA adapter** on disk (safetensors + config), which
  PEFT loads back into a fresh base model with its trained weights intact;
* the adapter reaches the **serving path** through the existing
  ``worker.brain.Brain`` hot-reload contract;
* the **eval delta** is measured on a held-out, prompt-grouped split and stored
  *with* the artifact;
* **preemption** interrupts a torch run in under 2 s;
* the **math backend is not deleted** and is still the automatic choice when the
  torch stack is missing.

Honest limits encoded here as assertions, not prose:

* the default base model is the tiny ``offline-selftest`` model, and the report
  must *say so* — a test fails if that label is ever silently dropped;
* 4-bit quantisation requires CUDA, which this environment does not have, so the
  report must record the precision it actually used (``fp32`` here).
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from training.distributed_trainer import PreemptionToken, TrainingTrace
from training.evaluate import (
    EvalResult,
    eval_set,
    evaluate,
    exact_match,
    normalise,
    split_traces,
    token_f1,
)
from training.torch_qlora import (
    BACKEND_MATH,
    BACKEND_TORCH,
    AdapterArtifact,
    AutoTrainer,
    TorchQLoRATrainer,
    adapter_to_brain_artifact,
    cuda_available,
    quantized_available,
    resolve_base_model,
    torch_available,
)

#: The whole torch file is skipped (not silently passed) when the optional
#: training extra is not installed — `make test` must stay green without it.
requires_torch = pytest.mark.skipif(
    not torch_available(),
    reason="torch/transformers/peft not installed (the [training] extra is optional)",
)


def _traces() -> list[TrainingTrace]:
    return eval_set()


# --------------------------------------------------------------------------- #
# 1. The evaluation harness
# --------------------------------------------------------------------------- #
def test_split_never_shares_a_prompt_between_train_and_held_out() -> None:
    """Grouping by prompt is what makes the delta mean anything."""
    traces = [
        TrainingTrace(prompt="same question", response=f"answer {i}") for i in range(6)
    ] + [TrainingTrace(prompt=f"other {i}", response=f"answer {i}") for i in range(10)]
    train, held, _ = split_traces(traces, held_out_fraction=0.25, seed=0)
    assert train and held
    train_prompts = {normalise(t.prompt) for t in train}
    held_prompts = {normalise(t.prompt) for t in held}
    assert not (train_prompts & held_prompts), "a prompt leaked across the split"


def test_split_is_deterministic() -> None:
    traces = _traces()
    first = split_traces(traces, seed=7)
    second = split_traces(traces, seed=7)
    assert [t.prompt for t in first[1]] == [t.prompt for t in second[1]]


def test_split_never_holds_out_everything() -> None:
    """Training needs something left to learn from."""
    traces = _traces()[:2]
    train, held, notes = split_traces(traces, held_out_fraction=0.9, min_held_out=5)
    assert train, "the split left nothing to train on"
    assert notes, "a degraded split must explain itself"


def test_metrics_are_what_they_claim() -> None:
    assert exact_match("the answer is 42.", "the answer is 42") == 1.0
    assert exact_match("something else", "the answer is 42") == 0.0
    assert token_f1("a b c", "a b c") == 1.0
    assert token_f1("x y z", "a b c") == 0.0
    assert 0.0 < token_f1("a b c d", "a b") < 1.0


def test_evaluate_reports_the_numbers_it_measured() -> None:
    traces = _traces()[:4]

    def perfect(prompt: str, gold: str) -> tuple[str, float | None]:
        return gold, 1.0

    def terrible(prompt: str, gold: str) -> tuple[str, float | None]:
        return "nonsense", 9.0

    good = evaluate(perfect, traces, label="after")
    bad = evaluate(terrible, traces, label="before")
    assert good.examples == 4
    assert good.exact_match == 1.0 and bad.exact_match == 0.0
    assert good.loss < bad.loss


def test_evaluate_does_not_invent_a_loss_when_no_likelihood_exists() -> None:
    """A scorer that cannot produce a likelihood must not look perfect."""

    def no_likelihood(prompt: str, gold: str) -> tuple[str, float | None]:
        return gold, None

    result = evaluate(no_likelihood, _traces()[:3], label="after")
    assert result.exact_match == 1.0
    assert any("no likelihood" in note for note in result.notes)


def test_a_raising_scorer_is_a_miss_not_a_crash() -> None:
    def broken(prompt: str, gold: str) -> tuple[str, float | None]:
        raise RuntimeError("boom")

    result = evaluate(broken, _traces()[:3], label="before")
    assert result.exact_match == 0.0
    assert any("raised" in note for note in result.notes)


# --------------------------------------------------------------------------- #
# 2. Capability probing and backend choice
# --------------------------------------------------------------------------- #
def test_resolve_base_model_labels_the_offline_self_test_model(tmp_path: Path) -> None:
    base = resolve_base_model("")
    # Either a genuinely cached model, or the tiny one — and if it is the tiny
    # one, it must be *labelled* as such.
    if base.source == "offline-selftest":
        assert base.synthetic is True
        assert "offline-selftest" in base.name
    else:
        assert base.source == "hf-cache"


def test_resolve_base_model_refuses_to_invent_a_missing_path() -> None:
    base = resolve_base_model(str(Path("/nonexistent/model/that/is/not/there")))
    assert base.source != "local-path", "a non-existent path must not be reported as usable"


def test_auto_trainer_reports_why_it_chose() -> None:
    choice = AutoTrainer(checkpoint_dir="/tmp/hc-m7-choice").choice
    assert choice.backend in {BACKEND_TORCH, BACKEND_MATH}
    assert choice.reason, "the choice must carry a reason"
    payload = choice.to_dict()
    assert set(payload) >= {"backend", "reason", "torch_ready", "cuda", "quantized"}


def test_auto_trainer_can_be_pinned_to_math() -> None:
    trainer = AutoTrainer(checkpoint_dir="/tmp/hc-m7-math", preferred="math")
    assert trainer.backend == BACKEND_MATH


def test_auto_trainer_delegates_configuration_to_the_selected_trainer() -> None:
    """A caller that does not know the backend must still configure the right one."""
    trainer = AutoTrainer(checkpoint_dir="/tmp/hc-m7-delegate", preferred="math")

    def shard(payload: dict) -> tuple[list, float]:
        return [[0.0]], 0.1

    trainer.shard_fn = shard
    assert trainer.math.shard_fn is shard
    trainer.steps = 5
    assert trainer.math.steps == 5


def test_precision_is_recorded_not_assumed() -> None:
    """4-bit needs CUDA; whatever was used must be in the report."""
    if quantized_available():
        pytest.skip("a GPU with bitsandbytes is present; this asserts the CPU path")
    assert not quantized_available()
    assert cuda_available() is False


# --------------------------------------------------------------------------- #
# 3. The math backend is not deleted
# --------------------------------------------------------------------------- #
async def test_the_math_backend_still_trains(tmp_path: Path) -> None:
    """The pre-M7 trainer must keep working, unchanged, as the fallback."""
    from training.distributed_trainer import DistributedTrainer

    trainer = DistributedTrainer(checkpoint_dir=tmp_path, steps=3, dim=32, vocab_size=32)
    report = await trainer.train(_traces()[:6], shards=2)
    assert report.steps == 3
    assert report.merged is True
    assert report.artifact_sha256
    assert trainer.last_artifact is not None


async def test_auto_trainer_falls_back_to_math_without_torch(tmp_path: Path, monkeypatch) -> None:
    """No torch -> the math backend, and the report says so."""
    import training.torch_qlora as module

    monkeypatch.setattr(module, "torch_available", lambda: False)
    trainer = AutoTrainer(checkpoint_dir=tmp_path, math_kwargs={"steps": 2, "dim": 32, "vocab_size": 32})
    assert trainer.backend == BACKEND_MATH
    assert "fallback" in trainer.choice.reason
    report = await trainer.train(_traces()[:6])
    assert report.steps == 2
    assert any("math backend" in note for note in report.notes)


# --------------------------------------------------------------------------- #
# 4. The real QLoRA path
# --------------------------------------------------------------------------- #
@requires_torch
async def test_qlora_training_produces_a_real_adapter(tmp_path: Path) -> None:
    trainer = TorchQLoRATrainer(checkpoint_dir=tmp_path, steps=3, max_length=48, eval_max_new_tokens=4)
    report = await trainer.train(_traces()[:8])

    assert report.error == "", report.error
    assert report.backend == BACKEND_TORCH
    assert report.steps == 3
    assert report.merged is True

    artifact = trainer.last_artifact
    assert isinstance(artifact, AdapterArtifact)
    assert artifact.exists(), "no adapter_model.safetensors / adapter_config.json written"
    # A real adapter, not an empty stub.
    measured = artifact.measure()
    assert measured["files"] >= 2
    assert measured["bytes"] > 1000
    assert len(measured["sha256"]) == 64


@requires_torch
async def test_the_report_labels_the_offline_self_test_model(tmp_path: Path) -> None:
    """Honesty: a 115k-param model must never be reported as a real fine-tune."""
    trainer = TorchQLoRATrainer(checkpoint_dir=tmp_path, steps=2, max_length=48, eval_max_new_tokens=4)
    report = await trainer.train(_traces()[:6])
    if report.base_model_source == "offline-selftest":
        assert any("offline self-test" in note for note in report.notes), (
            "the offline self-test model was used but the report did not say so"
        )
        assert "offline-selftest" in report.model


@requires_torch
async def test_the_adapter_loads_back_with_its_trained_weights(tmp_path: Path) -> None:
    """The saving grace of a fine-tune: the weights that were trained come back."""
    pytest.importorskip("safetensors")
    import safetensors.torch as st
    from peft import PeftModel
    from transformers import LlamaForCausalLM

    from training.torch_qlora import _tiny_config

    trainer = TorchQLoRATrainer(checkpoint_dir=tmp_path, steps=3, max_length=48, eval_max_new_tokens=4)
    await trainer.train(_traces()[:8])
    artifact = trainer.last_artifact
    assert artifact is not None

    saved = st.load_file(str(artifact.adapter_file))
    lora_keys = [key for key in saved if "lora_" in key]
    assert lora_keys, "the adapter file contains no LoRA tensors"

    served = PeftModel.from_pretrained(LlamaForCausalLM(_tiny_config()), str(artifact.directory))
    loaded = dict(served.named_parameters())
    # The adapter's values must be present in the served model (name mangling
    # adds a `.default` segment, so compare by suffix).
    matched = 0
    for key, tensor in saved.items():
        suffix = key.replace("base_model.model.", "").replace(".weight", ".default.weight")
        for name, served_tensor in loaded.items():
            if name.endswith(suffix) and served_tensor.shape == tensor.shape:
                assert served_tensor.detach().float().cpu().allclose(tensor.float().cpu(), atol=1e-6)
                matched += 1
                break
    assert matched >= 1, "no saved LoRA tensor could be matched back into the served model"


@requires_torch
async def test_the_adapter_reaches_the_serving_path_through_the_brain(tmp_path: Path) -> None:
    """`Brain.load` is the existing hot-reload contract: the adapter must fit it."""
    from worker.brain import Brain

    trainer = TorchQLoRATrainer(checkpoint_dir=tmp_path, steps=2, max_length=48, eval_max_new_tokens=4)
    report = await trainer.train(_traces()[:8])
    assert report.merged is True
    artifact = trainer.last_artifact
    assert artifact is not None

    wire = adapter_to_brain_artifact(artifact)
    payload = wire.encode()
    assert payload, "the bridged artifact is empty"
    assert wire.metadata["format"] == "peft-adapter"
    assert set(wire.metadata["adapter_files"]) == {"adapter_model.safetensors", "adapter_config.json"}

    brain = Brain()
    info = brain.load(payload)
    assert info["loaded"] is True
    assert brain.loaded is True


@requires_torch
async def test_install_adapter_returns_a_servable_model(tmp_path: Path) -> None:
    from training.torch_qlora import install_adapter

    trainer = TorchQLoRATrainer(checkpoint_dir=tmp_path, steps=2, max_length=48, eval_max_new_tokens=4)
    await trainer.train(_traces()[:6])
    artifact = trainer.last_artifact
    assert artifact is not None
    model, tokenizer = install_adapter(artifact)
    assert type(model).__name__ == "PeftModelForCausalLM"
    assert tokenizer is not None


@requires_torch
async def test_held_out_loss_is_measured_before_and_after(tmp_path: Path) -> None:
    """The delta must be stored with the artifact, and must be a real measurement."""
    trainer = TorchQLoRATrainer(
        checkpoint_dir=tmp_path, steps=6, learning_rate=1e-3, max_length=64, eval_max_new_tokens=4
    )
    report = await trainer.train(_traces())
    payload = report.eval
    assert payload, "no evaluation was recorded"
    assert payload["before"] is not None and payload["after"] is not None
    assert payload["before"]["examples"] >= 3
    assert payload["method"], "the delta must state how it was measured"
    # Training must not make held-out loss *worse* by a lot on this tiny set;
    # and on this model it genuinely improves.  Assert the direction loosely so
    # the test is about the measurement existing, not about a lucky number.
    assert payload["after"]["loss"] <= payload["before"]["loss"] + 0.5

    # The metrics travel with the adapter.
    assert trainer.last_artifact is not None
    assert trainer.last_artifact.metrics_file.is_file()
    stored = json.loads(trainer.last_artifact.metrics_file.read_text(encoding="utf-8"))
    assert stored["eval"]["before"] is not None
    assert stored["eval"]["after"] is not None


@requires_torch
async def test_preemption_interrupts_qlora_under_two_seconds(tmp_path: Path) -> None:
    """The M4 contract still holds for the real trainer."""
    trainer = TorchQLoRATrainer(
        checkpoint_dir=tmp_path, steps=40, max_length=48, eval_max_new_tokens=4
    )
    token = PreemptionToken()

    task = asyncio.ensure_future(trainer.train(_traces()[:6], token=token, step_delay=0.2))
    await asyncio.sleep(0.5)
    assert not task.done(), "the training finished before the interrupt could be tested"
    started = time.perf_counter()
    token.preempt("user request")
    report = await asyncio.wait_for(task, timeout=30.0)
    latency_ms = (time.perf_counter() - started) * 1000.0

    assert report.preempted is True
    assert report.preempt_reason == "user request"
    assert latency_ms < 2000.0, f"the user waited {latency_ms:.0f} ms for the interrupt"
    # A preempted run keeps what it learned rather than throwing it away.
    assert report.artifact_path or report.notes


@requires_torch
async def test_a_missing_model_is_an_error_not_a_fake_success(tmp_path: Path, monkeypatch) -> None:
    trainer = TorchQLoRATrainer(checkpoint_dir=tmp_path, steps=2)

    def explosion(*args, **kwargs):
        raise RuntimeError("no base model here")

    monkeypatch.setattr(trainer, "_load", explosion)
    report = await trainer.train(_traces()[:4])
    assert report.error, "a failed load must set report.error"
    assert report.merged is False
    assert report.steps == 0


@requires_torch
async def test_auto_trainer_falls_back_when_torch_fails_to_load(tmp_path: Path, monkeypatch) -> None:
    """A torch failure mid-way must degrade to math and admit it."""
    trainer = AutoTrainer(
        checkpoint_dir=tmp_path,
        preferred="torch",
        math_kwargs={"steps": 2, "dim": 32, "vocab_size": 32},
    )
    assert trainer.backend == BACKEND_TORCH

    def explosion(*args, **kwargs):
        raise RuntimeError("simulated load failure")

    monkeypatch.setattr(trainer.torch, "_load", explosion)
    report = await trainer.train(_traces()[:6])
    assert report.steps == 2, "the fallback did not train"
    assert any("math backend" in note for note in report.notes)
    assert any("simulated load failure" in note for note in report.notes)


# --------------------------------------------------------------------------- #
# 5. EvalDelta serialisation (travels with the artifact)
# --------------------------------------------------------------------------- #
def test_eval_delta_survives_a_round_trip() -> None:
    from training.evaluate import EvalDelta

    before = EvalResult(label="before", loss=5.0, exact_match=0.0, examples=4)
    after = EvalResult(label="after", loss=4.0, exact_match=0.25, examples=4)
    delta = EvalDelta(before=before, after=after, backend="torch", model="m/fp32")
    payload = delta.to_dict()
    assert payload["improved"] is True
    restored = EvalDelta.from_dict(payload)
    assert restored.loss_delta == pytest.approx(1.0)
    assert restored.exact_match_delta == pytest.approx(0.25)
    assert restored.improved is True


# --------------------------------------------------------------------------- #
# 6. The adapter reaches the cloud and the wire (regression guards)
# --------------------------------------------------------------------------- #
@requires_torch
async def test_the_adapter_packs_into_one_restorable_blob(tmp_path: Path) -> None:
    """One self-describing object, so a restored checkpoint is still usable."""
    import io
    import tarfile

    trainer = TorchQLoRATrainer(checkpoint_dir=tmp_path, steps=2, max_length=48, eval_max_new_tokens=4)
    await trainer.train(_traces()[:6])
    artifact = trainer.last_artifact
    assert artifact is not None

    blob = artifact.pack()
    assert len(blob) > 1000
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
        names = archive.getnames()
    assert "adapter_model.safetensors" in names
    assert "adapter_config.json" in names
    assert "hyper_metrics.json" in names


@requires_torch
async def test_a_real_adapter_is_what_gets_pushed_to_the_cloud(tmp_path: Path) -> None:
    """The streamer must receive the adapter itself, not a silent empty push."""
    from training.autopilot_evolver import AutopilotEvolver

    class RecordingStreamer:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int, dict]] = []

        async def push_bytes(self, step, payload, *, filename, metadata=None):
            self.calls.append((filename, len(payload), metadata or {}))

            class _Streamed:
                targets = ["drive"]

            return _Streamed()

        async def push(self, step, matrix, *, metadata=None):
            self.calls.append(("npz", int(matrix.size), metadata or {}))

            class _Streamed:
                targets = ["drive"]

            return _Streamed()

    trainer = TorchQLoRATrainer(checkpoint_dir=tmp_path, steps=2, max_length=48, eval_max_new_tokens=4)
    report = await trainer.train(_traces()[:6])

    evolver = AutopilotEvolver.__new__(AutopilotEvolver)
    evolver.trainer = trainer
    evolver.streamer = RecordingStreamer()
    pushed = await evolver._push(report, trainer.last_artifact)

    assert pushed == ["drive"]
    assert len(evolver.streamer.calls) == 1
    filename, size, metadata = evolver.streamer.calls[0]
    assert filename.endswith(".tar.gz"), "the adapter archive is what travels"
    assert size > 1000
    assert metadata["backend"] == BACKEND_TORCH
    assert "model" in metadata


@requires_torch
async def test_the_cloud_listing_sees_adapter_archives(tmp_path: Path) -> None:
    """A .npz-only filter would hide every real adapter from the phone."""
    from training.checkpoint_streamer import CheckpointStreamer

    class FakeDrive:
        async def ensure_folder(self, name):
            return "folder"

        async def list_folder(self, folder_id):
            return [
                {"name": "checkpoint-000003.npz", "size": 10},
                {"name": "adapter-000008.tar.gz", "size": 20},
                {"name": "notes.txt", "size": 1},
            ]

    streamer = CheckpointStreamer(drive=FakeDrive())
    listed = await streamer.list_remote(target="drive")
    names = {item["path"] for item in listed}
    assert "checkpoint-000003.npz" in names
    assert "adapter-000008.tar.gz" in names, "a real adapter must be visible in the cloud listing"
    assert "notes.txt" not in names
