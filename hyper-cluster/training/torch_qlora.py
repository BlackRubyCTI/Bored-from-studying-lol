"""Real QLoRA fine-tuning (milestone 7).

This replaces the ridge-regression toy as the *preferred* trainer while keeping
it as the honest CPU fallback.  Two trainers, one interface:

``TorchQLoRATrainer``
    ``transformers`` + ``peft`` LoRA (with 4-bit quantisation when
    ``bitsandbytes`` and a GPU are both present) on a small instruct model.
    Produces a real adapter — ``adapter_model.safetensors`` plus
    ``adapter_config.json`` — that PEFT loads back into the serving path.
``math`` backend
    ``training.distributed_trainer.DistributedTrainer`` (numpy ridge + SVD LoRA).
    Not deleted, not deprecated, and still the default when torch is absent.

:class:`AutoTrainer` chooses between them **at runtime, by capability**, and
always reports which one ran.  The choice is never inferred from a config flag
alone: a flag that says ``torch`` on a box with no torch would otherwise produce
a silent downgrade, which is the exact failure mode milestone 6 removed
elsewhere.

Base model resolution, in order — and every step is reported, never assumed:

1. an explicit local path (``HYPER_EVOLUTION_BASE_MODEL`` pointing at a dir);
2. a Hugging Face id **only if it is already in the local cache** (no surprise
   network calls from a training loop);
3. a tiny LLaMA architecture built from a config — ``offline-selftest``.

Option 3 exists so this code path is testable in an offline environment, and it
is labelled as such everywhere it surfaces.  Fine-tuning a 115k-parameter model
is not fine-tuning a 7B model, and the artifacts say which one they came from.

Preemption is inherited unchanged: :class:`PreemptionToken` is checked between
every step and raced against every step, so a user request stops training in
well under the 2 s budget.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.ids import mono_ms, now_s
from core.logging_setup import get_logger
from training.distributed_trainer import (
    PreemptedError,
    PreemptionToken,
    TrainingReport,
    TrainingTrace,
)
from training.evaluate import EvalDelta, EvalResult, compare, evaluate, split_traces

LOGGER = get_logger("training.torch_qlora")

#: Backend identifiers (these strings are part of the API contract).
BACKEND_TORCH = "torch"
BACKEND_MATH = "math"

#: The architectural default used when no real model is available.
TINY_MODEL_NAME = "offline-selftest"


# --------------------------------------------------------------------------- #
# Capability probing
# --------------------------------------------------------------------------- #
def torch_available() -> bool:
    """True when the whole torch training stack imports.

    Deliberately defensive: a *broken* install (a half-installed torch, an ABI
    mismatch, a CUDA driver that raises on import) must be reported as "not
    available" so the node falls back to the math trainer, rather than raising
    during startup and taking the control plane down with it.
    """
    import importlib.util

    for name in ("torch", "transformers", "peft"):
        try:
            if importlib.util.find_spec(name) is None:
                return False
        except (ImportError, ValueError, AttributeError):
            return False
    return True


def cuda_available() -> bool:
    if not torch_available():
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - a broken driver is "no GPU", not a crash
        return False


def quantized_available() -> bool:
    """True when 4-bit quantisation can actually be *used* (bnb + CUDA).

    Every step is guarded: on a machine where ``bitsandbytes`` is absent (or
    half-installed, or importable but unable to see a GPU) this must answer
    ``False`` so the trainer silently *and correctly* selects full precision.
    """
    import importlib.util

    try:
        if importlib.util.find_spec("bitsandbytes") is None:
            return False
    except (ImportError, ValueError, AttributeError):
        return False
    if not cuda_available():
        return False
    try:
        import bitsandbytes  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# Model resolution
# --------------------------------------------------------------------------- #
@dataclass
class BaseModel:
    """The model a run actually trains, and how it was found."""

    #: What to pass to ``from_pretrained`` / the config builder.
    target: str
    #: ``local-path`` | ``hf-cache`` | ``offline-selftest``.
    source: str
    #: Human-readable identity stored with the artifact.
    name: str
    #: True for the tiny config-built model (so callers can say so).
    synthetic: bool = False

    def describe(self, precision: str) -> str:
        return f"{self.name}/{precision}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "source": self.source,
            "name": self.name,
            "synthetic": self.synthetic,
        }


#: The byte tokenizer's vocabulary: 256 byte ids + <pad>/<eos>/<unk>.
BYTE_VOCAB_SIZE = 259


def _tiny_config():
    from transformers import LlamaConfig

    return LlamaConfig(
        vocab_size=BYTE_VOCAB_SIZE,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
    )


def resolve_base_model(explicit: str = "") -> BaseModel:
    """Resolve the base model, preferring something real, always reporting which."""
    candidate = (explicit or "").strip()
    if candidate:
        path = Path(candidate).expanduser()
        if path.is_dir() and (path / "config.json").is_file():
            return BaseModel(target=str(path), source="local-path", name=path.name)
        if path.is_file():
            return BaseModel(target=str(path.parent), source="local-path", name=path.name)
        # An HF id: only usable if it is already cached (no surprise downloads).
        if _hf_cached(candidate):
            return BaseModel(target=candidate, source="hf-cache", name=candidate)
        LOGGER.info(
            "base model not available locally; using the offline self-test model",
            extra={"hc_requested": candidate},
        )

    for name in _cached_models():
        return BaseModel(target=name, source="hf-cache", name=name)

    return BaseModel(
        target=TINY_MODEL_NAME,
        source="offline-selftest",
        name=f"tiny-llama-115k/{TINY_MODEL_NAME}",
        synthetic=True,
    )


def _hf_cached(model_id: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return try_to_load_from_cache(model_id, "config.json") is not None
    except Exception:  # noqa: BLE001 - no hub cache is simply "not cached"
        return False


def _cached_models() -> list[str]:
    """Every HF model already in the local cache (deterministic order)."""
    try:
        from huggingface_hub import scan_cache_dir

        info = scan_cache_dir()
        return sorted({repo.repo_id for repo in info.repos})
    except Exception:  # noqa: BLE001
        return []


# --------------------------------------------------------------------------- #
# Artifact
# --------------------------------------------------------------------------- #
@dataclass
class AdapterArtifact:
    """A real LoRA adapter plus everything needed to interpret it."""

    directory: Path
    backend: str = BACKEND_TORCH
    model: str = ""
    steps: int = 0
    train_loss_start: float = 0.0
    train_loss_end: float = 0.0
    examples: int = 0
    precision: str = ""
    eval: EvalDelta | None = None
    created_at: float = field(default_factory=now_s)
    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    @property
    def adapter_file(self) -> Path:
        return self.directory / "adapter_model.safetensors"

    @property
    def config_file(self) -> Path:
        return self.directory / "adapter_config.json"

    @property
    def metrics_file(self) -> Path:
        return self.directory / "hyper_metrics.json"

    def exists(self) -> bool:
        return self.adapter_file.is_file() and self.config_file.is_file()

    def measure(self) -> dict[str, Any]:
        """Size and checksum of the adapter (for REVIEW numbers)."""
        import hashlib

        if not self.adapter_file.is_file():
            return {"files": 0, "bytes": 0, "sha256": ""}
        digest = hashlib.sha256(self.adapter_file.read_bytes()).hexdigest()
        files = [p for p in self.directory.rglob("*") if p.is_file()]
        return {
            "files": len(files),
            "bytes": sum(p.stat().st_size for p in files),
            "sha256": digest,
        }

    def pack(self) -> bytes:
        """The whole adapter directory as a single ``.tar.gz`` blob.

        This is what travels to the cloud: one self-describing object that PEFT
        can load after a plain ``tar -xzf``, with the metrics included so a
        restored checkpoint still knows how good it was.
        """
        import io
        import tarfile

        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            for path in sorted(self.directory.rglob("*")):
                if path.is_file():
                    archive.add(path, arcname=path.relative_to(self.directory).as_posix())
        return buffer.getvalue()

    def to_dict(self) -> dict[str, Any]:
        return {
            "directory": str(self.directory),
            "backend": self.backend,
            "model": self.model,
            "steps": self.steps,
            "train_loss_start": round(self.train_loss_start, 6),
            "train_loss_end": round(self.train_loss_end, 6),
            "examples": self.examples,
            "precision": self.precision,
            "eval": self.eval.to_dict() if self.eval else None,
            "created_at": self.created_at,
            "notes": list(self.notes),
            **self.measure(),
        }

    def write_metrics(self) -> Path:
        """Persist metrics **next to the adapter**, so they travel together."""
        payload = self.to_dict()
        payload.pop("files", None)
        payload.pop("bytes", None)
        self.metrics_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return self.metrics_file


# --------------------------------------------------------------------------- #
# The trainer
# --------------------------------------------------------------------------- #
class TorchQLoRATrainer:
    """QLoRA fine-tuning with the same contract as :class:`DistributedTrainer`."""

    def __init__(
        self,
        *,
        checkpoint_dir: str | Path,
        base_model: str = "",
        steps: int = 8,
        learning_rate: float = 2e-4,
        rank: int = 8,
        lora_alpha: int = 16,
        max_length: int = 128,
        batch_size: int = 2,
        seed: int = 0,
        held_out_fraction: float = 0.25,
        eval_max_new_tokens: int = 24,
        use_4bit: bool = True,
        target_modules: Sequence[str] = ("q_proj", "v_proj"),
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir).expanduser()
        self.base_model_request = base_model or os.environ.get("HYPER_EVOLUTION_BASE_MODEL", "")
        self.steps = max(1, int(steps))
        self.learning_rate = float(learning_rate)
        self.rank = max(1, int(rank))
        self.lora_alpha = int(lora_alpha)
        self.max_length = max(16, int(max_length))
        self.batch_size = max(1, int(batch_size))
        self.seed = int(seed)
        self.held_out_fraction = float(held_out_fraction)
        self.eval_max_new_tokens = max(1, int(eval_max_new_tokens))
        self.use_4bit = bool(use_4bit)
        self.target_modules = list(target_modules)

        self.last_artifact: AdapterArtifact | None = None
        self.last_eval: EvalDelta | None = None
        self._model: Any = None
        self._tokenizer: Any = None
        self._base: BaseModel | None = None
        self._precision: str = ""

    # ------------------------------------------------------------------ #
    @property
    def backend(self) -> str:
        return BACKEND_TORCH

    # ------------------------------------------------------------------ #
    def _build_tokenizer(self):
        """A byte-level tokenizer that needs no downloaded merges.

        ``PreTrainedTokenizerFast`` with no ``tokenizer_file`` works offline and
        gives a deterministic id for any byte string, which is what the
        offline-self-test path needs.  When a real model ships its own
        tokenizer, it is used instead.
        """
        from transformers import PreTrainedTokenizerFast

        if self._base is not None and not self._base.synthetic:
            from transformers import AutoTokenizer

            try:
                return AutoTokenizer.from_pretrained(self._base.target, local_files_only=True)
            except Exception as exc:  # noqa: BLE001 - fall back to bytes
                LOGGER.warning("tokenizer unavailable; using a byte tokenizer", extra={"hc_error": str(exc)})

        from tokenizers import Tokenizer, models, pre_tokenizers

        vocab = {f"b{index}": index for index in range(256)}
        vocab["<pad>"] = 256
        vocab["<eos>"] = 257
        vocab["<unk>"] = 258
        assert len(vocab) == BYTE_VOCAB_SIZE == _tiny_config().vocab_size
        raw = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
        raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        return PreTrainedTokenizerFast(
            tokenizer_object=raw,
            pad_token="<pad>",
            eos_token="<eos>",
            unk_token="<unk>",
        )

    # ------------------------------------------------------------------ #
    def _load(self) -> tuple[Any, Any, str]:
        """Load (model, tokenizer, precision).  Raises when torch is absent."""
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, LlamaForCausalLM

        base = resolve_base_model(self.base_model_request)
        self._base = base
        torch.manual_seed(self.seed)

        quant_kwargs: dict[str, Any] = {}
        precision = "fp32"
        if self.use_4bit and quantized_available():
            from transformers import BitsAndBytesConfig

            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            quant_kwargs["device_map"] = {"": 0}
            precision = "4bit-nf4"

        if base.synthetic:
            model = LlamaForCausalLM(_tiny_config())
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base.target,
                local_files_only=True,
                torch_dtype=torch.float32 if precision == "fp32" else "auto",
                **quant_kwargs,
            )

        lora = LoraConfig(
            r=self.rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self.target_modules,
        )
        model = get_peft_model(model, lora)
        tokenizer = self._build_tokenizer()
        model.train()
        # A newly attached LoRA head starts at B = 0, so the very first forward
        # pass is exactly the base model.  Nothing is "already trained".
        self._precision = precision
        return model, tokenizer, precision

    # ------------------------------------------------------------------ #
    def _encode(self, tokenizer: Any, prompt: str, answer: str) -> tuple[list[int], int]:
        """Tokenise ``prompt+answer`` and return (ids, prompt_len)."""
        prompt_ids = tokenizer(f"Q: {prompt}\nA:", add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(f" {answer}", add_special_tokens=False)["input_ids"]
        ids = list(prompt_ids) + list(answer_ids)
        return ids[: self.max_length], min(len(prompt_ids), self.max_length - 1)

    # ------------------------------------------------------------------ #
    def _batch(self, tokenizer: Any, traces: Sequence[TrainingTrace]):
        """Build padded (input_ids, attention_mask, labels) tensors."""
        import torch

        rows = [self._encode(tokenizer, t.prompt, t.response) for t in traces]
        if not rows:
            return None
        width = max(len(ids) for ids, _ in rows)
        pad = tokenizer.pad_token_id or 0
        input_ids, attention, labels = [], [], []
        for ids, prompt_len in rows:
            padding = width - len(ids)
            input_ids.append(ids + [pad] * padding)
            attention.append([1] * len(ids) + [0] * padding)
            # Only the *answer* tokens carry loss: predicting the question is
            # not the skill being taught.
            row = [-100] * prompt_len + ids[prompt_len:] + [-100] * padding
            labels.append(row)
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )

    # ------------------------------------------------------------------ #
    def _step_loss(self, model: Any, batch) -> float:
        """Forward once and return the scalar loss (no backward)."""
        input_ids, attention, labels = batch
        out = model(input_ids=input_ids, attention_mask=attention, labels=labels)
        return float(out.loss.detach().item())

    # ------------------------------------------------------------------ #
    def _generate(self, model: Any, tokenizer: Any, prompt: str) -> str:
        import torch

        model.eval()
        try:
            ids = tokenizer(f"Q: {prompt}\nA:", return_tensors="pt", add_special_tokens=False)["input_ids"]
            with torch.no_grad():
                out = model.generate(
                    input_ids=ids,
                    max_new_tokens=self.eval_max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            return text.strip()
        finally:
            model.train()

    def _answer_nll(self, model: Any, tokenizer: Any, prompt: str, answer: str) -> float | None:
        """Mean per-token NLL the model assigns to the gold answer tokens."""
        import torch

        ids, prompt_len = self._encode(tokenizer, prompt, answer)
        if len(ids) <= prompt_len:
            return None
        model.eval()
        try:
            tensor = torch.tensor([ids], dtype=torch.long)
            labels = tensor.clone()
            labels[:, :prompt_len] = -100
            with torch.no_grad():
                out = model(input_ids=tensor, labels=labels)
            value = float(out.loss.detach().item())
        finally:
            model.train()
        return value if value == value else None  # NaN guard

    # ------------------------------------------------------------------ #
    def evaluate_now(self, traces: Sequence[TrainingTrace], *, label: str) -> EvalResult:
        """Score the *current* model on ``traces`` (used before and after)."""
        if self._model is None or self._tokenizer is None:
            return EvalResult(label=label, notes=["model not loaded; nothing measured"])

        def scorer(prompt: str, gold: str) -> tuple[str, float | None]:
            generated = self._generate(self._model, self._tokenizer, prompt)
            return generated, self._answer_nll(self._model, self._tokenizer, prompt, gold)

        return evaluate(scorer, traces, label=label)

    # ------------------------------------------------------------------ #
    async def train(
        self,
        traces: Sequence[TrainingTrace],
        *,
        shards: int = 1,
        token: PreemptionToken | None = None,
        resume: bool = True,
        on_step: Callable[[int, float], Any] | None = None,
        eval_traces: Sequence[TrainingTrace] | None = None,
        step_delay: float = 0.0,
    ) -> TrainingReport:
        """Fine-tune and return a :class:`TrainingReport`.

        ``shards`` is accepted for interface compatibility with the federated
        trainer and is ignored: a LoRA update is not shard-separable the way a
        gradient average is, and pretending otherwise would be a lie in the
        report.  ``grade_shards`` is reported as 1.
        """
        import asyncio

        started = mono_ms()
        report = TrainingReport(examples=len(traces), shards=1)
        if not traces:
            report.duration_ms = mono_ms() - started
            report.backend = BACKEND_TORCH
            report.notes.append("no training traces: nothing was trained")
            return report

        train_traces, held_out, split_notes = split_traces(
            traces, held_out_fraction=self.held_out_fraction, min_held_out=3, seed=self.seed
        )
        if eval_traces:
            held_out = list(eval_traces)
            split_notes.append("held-out set supplied explicitly by the caller")
        report.notes.extend(split_notes)

        try:
            model, tokenizer, precision = self._load()
        except Exception as exc:  # noqa: BLE001 - a missing stack must not kill the node
            report.backend = BACKEND_TORCH
            report.error = f"torch stack unavailable: {type(exc).__name__}: {exc}"
            report.duration_ms = mono_ms() - started
            report.notes.append("trainer could not load a model; no artifact produced")
            return report

        base = self._base
        assert base is not None
        self._model, self._tokenizer = model, tokenizer
        report.backend = BACKEND_TORCH
        report.model = base.describe(precision)
        report.precision = precision
        report.base_model_source = base.source
        if base.synthetic:
            report.notes.append(
                "base model is the offline self-test model (115k params): this proves the "
                "QLoRA path runs, not that a production model was improved"
            )

        # --- before ---------------------------------------------------- #
        before: EvalResult | None = None
        if held_out:
            report.phase = "evaluate-before"
            before = self.evaluate_now(held_out, label="before")
        else:
            report.notes.append("no held-out examples: no before/after delta can be measured")

        # --- train ----------------------------------------------------- #
        import torch

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=self.learning_rate
        )
        losses: list[float] = []
        report.phase = "training"
        try:
            for step in range(self.steps):
                if token is not None:
                    token.guard()
                if step_delay:
                    # A deliberately slow step: lets the preemption test prove the
                    # interrupt latency without waiting on a real model.
                    await asyncio.sleep(step_delay)
                    if token is not None:
                        token.guard()
                batch = self._batch(tokenizer, train_traces)
                if batch is None:
                    break
                optimizer.zero_grad(set_to_none=True)
                input_ids, attention, labels = batch
                out = model(input_ids=input_ids, attention_mask=attention, labels=labels)
                loss = float(out.loss.detach().item())
                out.loss.backward()
                optimizer.step()
                losses.append(loss)
                report.steps = step + 1
                report.loss_end = loss
                if report.loss_start == 0.0:
                    report.loss_start = loss
                if on_step is not None:
                    result = on_step(report.steps, loss)
                    if asyncio.iscoroutine(result):
                        await result
        except PreemptedError as exc:
            if token is not None:
                token.acknowledge()
            report.preempted = True
            report.preempt_reason = str(exc)
            report.phase = "preempted"
            report.duration_ms = mono_ms() - started
            LOGGER.info("torch training preempted", extra={"hc_steps": report.steps})
            # A preempted run still saves what it learned: the adapter is the
            # checkpoint.
            self._save(report, held_out, before, losses, base, precision)
            return report

        # --- after ----------------------------------------------------- #
        after: EvalResult | None = None
        if held_out:
            report.phase = "evaluate-after"
            after = self.evaluate_now(held_out, label="after")
        delta = compare(before, after, backend=BACKEND_TORCH, model=base.describe(precision))
        self.last_eval = delta
        report.eval = delta.to_dict()

        artifact = self._save(report, held_out, before, losses, base, precision)
        report.artifact_sha256 = artifact.measure()["sha256"] if artifact else ""
        report.merged = bool(artifact and artifact.exists())
        report.phase = "done"
        report.duration_ms = mono_ms() - started
        LOGGER.info(
            "torch training finished",
            extra={
                "hc_steps": report.steps,
                "hc_loss": round(report.loss_end, 6),
                "hc_model": report.model,
                "hc_loss_delta": round(delta.loss_delta, 6),
            },
        )
        return report

    # ------------------------------------------------------------------ #
    def _save(
        self,
        report: TrainingReport,
        held_out: Sequence[TrainingTrace],
        before: EvalResult | None,
        losses: Sequence[float],
        base: BaseModel,
        precision: str,
    ) -> AdapterArtifact | None:
        """Write the adapter + metrics atomically into the checkpoint dir."""
        if self._model is None:
            return None
        stamp = f"{int(now_s())}"
        directory = self.checkpoint_dir / f"adapter-{stamp}" / "final"
        directory.mkdir(parents=True, exist_ok=True)
        try:
            self._model.save_pretrained(str(directory))
        except Exception as exc:  # noqa: BLE001 - report, never fake the artifact
            LOGGER.warning("adapter save failed", extra={"hc_error": str(exc)})
            report.notes.append(f"adapter could not be saved: {exc}")
            return None

        after = self.evaluate_now(held_out, label="after") if held_out else None
        if before is not None and after is not None:
            delta = compare(before, after, backend=BACKEND_TORCH, model=base.describe(precision))
            self.last_eval = delta
            report.eval = delta.to_dict()
        else:
            delta = compare(None, None, backend=BACKEND_TORCH, model=base.describe(precision))

        # The adapter *is* the checkpoint here: a real, loadable artifact on
        # disk.  Counting it keeps the checkpoint/cloud-sync bookkeeping that
        # the autopilot and the M4 checks rely on truthful for both backends.
        report.checkpoints += 1

        artifact = AdapterArtifact(
            directory=directory,
            backend=BACKEND_TORCH,
            model=base.describe(precision),
            steps=report.steps,
            train_loss_start=report.loss_start,
            train_loss_end=report.loss_end,
            examples=report.examples,
            precision=precision,
            eval=delta,
            notes=[*report.notes, f"base source: {base.source}"],
        )
        artifact.write_metrics()
        self.last_artifact = artifact
        if artifact.exists():
            report.artifact_path = str(directory)
        else:
            report.notes.append("adapter directory written but adapter files are missing")
        return artifact

    # ------------------------------------------------------------------ #
    def checkpoints(self) -> list[dict[str, Any]]:
        """Every adapter written by this trainer, newest first."""
        if not self.checkpoint_dir.is_dir():
            return []
        found: list[dict[str, Any]] = []
        for path in sorted(self.checkpoint_dir.glob("adapter-*/final/hyper_metrics.json")):
            try:
                found.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return list(reversed(found))


# --------------------------------------------------------------------------- #
# Serving path: hand a real adapter to the worker's brain
# --------------------------------------------------------------------------- #
def adapter_to_brain_artifact(artifact: AdapterArtifact):
    """Wrap a real LoRA adapter in a :class:`BrainArtifact` the mesh can ship.

    ``BrainArtifact`` is the wire format the workers already hot-reload
    (``worker/brain.py``).  It is an ``.npz`` blob, which cannot carry a
    safetensors adapter — so the adapter is embedded as a compressed base64
    payload in its ``metadata``, and the matrices the legacy hint path needs are
    built from the adapter's own ``lora_A``/``lora_B`` so the brain still has
    something meaningful to say even before a worker has PEFT installed.

    The returned artifact keeps ``metadata["format"] = "peft-adapter"`` so the
    consumer knows to install the real adapter rather than only use the hint.
    """
    import base64
    import zlib

    import numpy as np

    from training.qlora import BrainArtifact

    files: dict[str, str] = {}
    for path in sorted(artifact.directory.rglob("*")):
        if path.is_file() and path.name in {"adapter_model.safetensors", "adapter_config.json"}:
            files[path.name] = base64.b64encode(zlib.compress(path.read_bytes(), 9)).decode("ascii")

    metadata = {
        "format": "peft-adapter",
        "backend": artifact.backend,
        "model": artifact.model,
        "precision": artifact.precision,
        "adapter_files": files,
        "eval": artifact.eval.to_dict() if artifact.eval else None,
        "notes": list(artifact.notes),
    }
    brain = BrainArtifact(
        dim=0,
        rank=0,
        vocab=[],
        steps=artifact.steps,
        loss=artifact.train_loss_end,
        metadata=metadata,
    )
    # An empty matrix set keeps the legacy hint path inert instead of
    # inventing numbers the adapter never produced.
    brain.base_quantized = _pack_np(np.zeros((0, 0), np.int8))
    brain.base_scale = _pack_np(np.zeros((0,), np.float32))
    brain.A = _pack_np(np.zeros((0, 0), np.float32))
    brain.B = _pack_np(np.zeros((0, 0), np.float32))
    brain.merged = _pack_np(np.zeros((0, 0), np.float32))
    return brain


def _pack_np(array):
    import io

    import numpy as np

    buffer = io.BytesIO()
    np.savez_compressed(buffer, payload=array)
    return buffer.getvalue()


def install_adapter(artifact: AdapterArtifact, *, base_model: str = "") -> Any:
    """Load a trained adapter for serving (the worker side of the bridge).

    Returns a ``(model, tokenizer)`` pair, or raises with a clear reason.  The
    caller is expected to keep the process alive and hot-swappable, exactly like
    the legacy brain: build, then swap.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, LlamaForCausalLM

    if not artifact.exists():
        raise FileNotFoundError(f"no adapter files in {artifact.directory}")
    base = resolve_base_model(base_model or os.environ.get("HYPER_EVOLUTION_BASE_MODEL", ""))
    model = LlamaForCausalLM(_tiny_config()) if base.synthetic else AutoModelForCausalLM.from_pretrained(
        base.target, local_files_only=True
    )
    served = PeftModel.from_pretrained(model, str(artifact.directory))
    served.eval()
    trainer = TorchQLoRATrainer(checkpoint_dir=artifact.directory)
    trainer._base = base  # noqa: SLF001 - the tokenizer depends on this choice
    tokenizer = trainer._build_tokenizer()  # noqa: SLF001
    return served, tokenizer


# --------------------------------------------------------------------------- #
# Chooser
# --------------------------------------------------------------------------- #
@dataclass
class TrainerChoice:
    """Which trainer was selected, and why — reported, never inferred."""

    backend: str
    reason: str
    torch_ready: bool
    cuda: bool
    quantized: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "reason": self.reason,
            "torch_ready": self.torch_ready,
            "cuda": self.cuda,
            "quantized": self.quantized,
        }


class AutoTrainer:
    """Picks the strongest trainer the machine can actually run.

    The interface is the union of both trainers, so the autopilot does not need
    to know which one it is holding.  ``backend`` and ``reason`` always say.
    """

    def __init__(
        self,
        *,
        checkpoint_dir: str | Path,
        preferred: str = "auto",
        torch_kwargs: dict[str, Any] | None = None,
        math_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.preferred = preferred
        self.torch_kwargs = dict(torch_kwargs or {})
        self.math_kwargs = dict(math_kwargs or {})
        object.__setattr__(self, "_math", None)
        object.__setattr__(self, "_torch", None)
        object.__setattr__(self, "choice", self._choose())

    # ------------------------------------------------------------------ #
    def _choose(self) -> TrainerChoice:
        # Probing must never raise: a broken install is "not available", not a
        # reason for the control plane to fail to start.
        try:
            ready = torch_available()
            cuda = cuda_available()
            quant = quantized_available()
        except Exception:  # noqa: BLE001 - defensive, see above
            return TrainerChoice(
                BACKEND_MATH,
                "probing the torch stack raised; the numpy ridge trainer is the honest fallback",
                False,
                False,
                False,
            )
        wanted = (self.preferred or "auto").strip().lower()

        if wanted in {BACKEND_MATH, "ridge", "numpy", "cpu"}:
            return TrainerChoice(BACKEND_MATH, "configured explicitly", ready, cuda, quant)
        if not ready:
            return TrainerChoice(
                BACKEND_MATH,
                "torch/transformers/peft not importable; the numpy ridge trainer is the honest fallback",
                ready,
                cuda,
                quant,
            )
        if wanted == BACKEND_TORCH:
            return TrainerChoice(BACKEND_TORCH, "configured explicitly and the stack is present", ready, cuda, quant)
        try:
            base = resolve_base_model(
                self.torch_kwargs.get("base_model", "") or os.environ.get("HYPER_EVOLUTION_BASE_MODEL", "")
            )
        except Exception:  # noqa: BLE001 - a broken hub cache is "nothing cached"
            return TrainerChoice(
                BACKEND_TORCH,
                "torch stack present but the model cache could not be read; the offline self-test model will be used",
                ready,
                cuda,
                quant,
            )
        if base.source == "offline-selftest":
            return TrainerChoice(
                BACKEND_TORCH,
                "torch stack present; no real base model is cached so the offline self-test model is used",
                ready,
                cuda,
                quant,
            )
        return TrainerChoice(BACKEND_TORCH, f"torch stack present with base model from {base.source}", ready, cuda, quant)

    # ------------------------------------------------------------------ #
    @property
    def backend(self) -> str:
        return self.choice.backend

    # -- Delegation -------------------------------------------------------- #
    # The autopilot and the tests configure the trainer through the *math*
    # trainer's attributes (``shard_fn``, ``steps`` ...).  ``AutoTrainer``
    # therefore forwards attribute reads/writes to the trainer it selected, so
    # a caller that does not know which backend won still configures the right
    # object — and gets no silent no-op when it sets one.
    def __getattr__(self, name: str) -> Any:
        # ``__getattr__`` only runs for attributes not found normally, so this
        # cannot shadow the real properties above.
        inner = self._selected()
        if inner is not None and hasattr(inner, name):
            return getattr(inner, name)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_") or name in {
            "checkpoint_dir",
            "preferred",
            "torch_kwargs",
            "math_kwargs",
            "choice",
        }:
            object.__setattr__(self, name, value)
            return
        inner = self._selected()
        if inner is not None and hasattr(inner, name):
            setattr(inner, name, value)
            return
        object.__setattr__(self, name, value)

    def _selected(self) -> Any:
        """The live trainer instance for the chosen backend (or None)."""
        backend = getattr(self, "choice", None)
        if backend is None:
            return None
        if backend.backend == BACKEND_TORCH:
            return self.__dict__.get("_torch") or self.torch
        return self.__dict__.get("_math") or self.math

    @property
    def math(self) -> Any:
        instance = self.__dict__.get("_math")
        if instance is None:
            from training.distributed_trainer import DistributedTrainer

            instance = DistributedTrainer(checkpoint_dir=self.checkpoint_dir, **self.math_kwargs)
            object.__setattr__(self, "_math", instance)
        return instance

    @property
    def torch(self) -> TorchQLoRATrainer:
        instance = self.__dict__.get("_torch")
        if instance is None:
            instance = TorchQLoRATrainer(checkpoint_dir=self.checkpoint_dir, **self.torch_kwargs)
            object.__setattr__(self, "_torch", instance)
        return instance

    @property
    def last_artifact(self) -> Any:
        if self.backend == BACKEND_TORCH and self._torch is not None:
            return self._torch.last_artifact
        return self._math.last_artifact

    @property
    def last_eval(self) -> EvalDelta | None:
        if self.backend == BACKEND_TORCH and self._torch is not None:
            return self._torch.last_eval
        return getattr(self._math, "last_eval", None)

    # ------------------------------------------------------------------ #
    async def train(self, traces: Sequence[TrainingTrace], **kwargs: Any) -> TrainingReport:
        if self.backend == BACKEND_TORCH:
            report = await self.torch.train(traces, **kwargs)
            # If the torch run could not load at all, fall back rather than
            # report a fake success (and say that we did).
            if report.error and not report.steps:
                LOGGER.warning("torch training failed to start; falling back to the math trainer")
                fallback = await self.math.train(traces, **{k: v for k, v in kwargs.items() if k != "step_delay"})
                fallback.notes.append("torch backend failed to start; this run used the math backend")
                fallback.notes.append(f"torch error: {report.error}")
                return fallback
            return report
        report = await self.math.train(traces, **{k: v for k, v in kwargs.items() if k != "step_delay"})
        if not report.notes:
            report.notes = []
        report.notes.append(f"trained with the math backend ({self.choice.reason})")
        return report

    def status(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            **self.choice.to_dict(),
            "checkpoint_dir": str(self.checkpoint_dir),
            "last_eval": self.last_eval.to_dict() if self.last_eval else None,
        }


__all__ = [
    "BACKEND_MATH",
    "BACKEND_TORCH",
    "AdapterArtifact",
    "AutoTrainer",
    "BaseModel",
    "BYTE_VOCAB_SIZE",
    "TINY_MODEL_NAME",
    "TorchQLoRATrainer",
    "TrainerChoice",
    "adapter_to_brain_artifact",
    "cuda_available",
    "install_adapter",
    "quantized_available",
    "resolve_base_model",
    "torch_available",
]
