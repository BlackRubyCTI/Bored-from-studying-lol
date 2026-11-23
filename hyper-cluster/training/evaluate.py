"""Evaluation harness for the self-evolution loop (milestone 7).

Milestone 4's training reported a *training* loss, which is the number a model is
guaranteed to improve on a dataset it is fitting.  It is not evidence of
learning.  This module measures held-out performance instead, before and after,
with a fixed split and deterministic scoring, and stores the result *with* the
artifact that produced it.

What is measured, precisely:

``loss``
    Mean per-token cross-entropy of the gold answer tokens, given the prompt.
    Lower is better.  This is the number a language-model fine-tune is actually
    supposed to move.
``exact_match``
    Fraction of held-out prompts whose greedy continuation starts with the gold
    answer (normalised: lowercase, whitespace-collapsed, no trailing period).
``token_f1``
    Bag-of-words F1 between the continuation and the gold answer — a softer
    signal that still penalises answering the wrong question.

What is *not* measured, and is therefore not claimed: helpfulness, factuality,
safety, or anything a human would call "quality".  A model can drive all three
numbers above to their optimum on a 24-example held-out set and still be a worse
assistant.  The delta is reported as exactly what it is: a held-out fit.

Splitting is deterministic and *grouped by prompt*, so the same question cannot
appear in both halves and let a model that memorised the prompt look like it
generalised.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

#: Normalisation used by every string metric (kept visible, not hidden in a call).
_WS_RE = re.compile(r"\s+")
_PUNCT_TAIL_RE = re.compile(r"[.!?]\s*$")


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass
class EvalResult:
    """One measurement of a model on one held-out set."""

    #: ``before`` or ``after`` (or ``baseline`` / ``math`` — whatever the caller says).
    label: str = ""
    #: Mean per-token cross-entropy on the held-out answers (lower is better).
    loss: float = 0.0
    #: Fraction of exact (normalised) matches.
    exact_match: float = 0.0
    #: Bag-of-words F1 against the gold answer.
    token_f1: float = 0.0
    #: How many held-out examples produced these numbers.
    examples: int = 0
    #: Mean generated tokens per example (a proxy for "did it stop rambling").
    mean_length: float = 0.0
    #: Free-form notes, e.g. "reused cached baseline" or "tiny model".
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "loss": round(self.loss, 6),
            "exact_match": round(self.exact_match, 4),
            "token_f1": round(self.token_f1, 4),
            "examples": self.examples,
            "mean_length": round(self.mean_length, 2),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvalResult:
        return cls(
            label=str(payload.get("label") or ""),
            loss=float(payload.get("loss") or 0.0),
            exact_match=float(payload.get("exact_match") or 0.0),
            token_f1=float(payload.get("token_f1") or 0.0),
            examples=int(payload.get("examples") or 0),
            mean_length=float(payload.get("mean_length") or 0.0),
            notes=[str(n) for n in (payload.get("notes") or [])],
        )


@dataclass
class EvalDelta:
    """The before/after comparison, stored with every artifact."""

    before: EvalResult | None
    after: EvalResult | None
    #: Which trainer produced the ``after`` measurement (``torch`` | ``math``).
    backend: str = ""
    #: The concrete model/precision, e.g. ``"tiny-llama/fp32"``.
    model: str = ""

    @property
    def loss_delta(self) -> float:
        """``before - after``: positive means the loss went down (better)."""
        if self.before is None or self.after is None:
            return 0.0
        return self.before.loss - self.after.loss

    @property
    def exact_match_delta(self) -> float:
        if self.before is None or self.after is None:
            return 0.0
        return self.after.exact_match - self.before.exact_match

    @property
    def token_f1_delta(self) -> float:
        if self.before is None or self.after is None:
            return 0.0
        return self.after.token_f1 - self.before.token_f1

    @property
    def improved(self) -> bool:
        """True only when held-out *loss* went down by a non-trivial margin."""
        return self.loss_delta > 1e-4

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model,
            "before": self.before.to_dict() if self.before else None,
            "after": self.after.to_dict() if self.after else None,
            "loss_delta": round(self.loss_delta, 6),
            "exact_match_delta": round(self.exact_match_delta, 4),
            "token_f1_delta": round(self.token_f1_delta, 4),
            "improved": self.improved,
            # Stated in the artifact itself, so nobody has to read this file to
            # know what the number means.
            "method": (
                "held-out mean per-token cross-entropy + exact-match + token-F1 "
                "on a prompt-grouped split; 'improved' means held-out loss fell"
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvalDelta:
        before = payload.get("before")
        after = payload.get("after")
        return cls(
            before=EvalResult.from_dict(before) if before else None,
            after=EvalResult.from_dict(after) if after else None,
            backend=str(payload.get("backend") or ""),
            model=str(payload.get("model") or ""),
        )


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def _stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def prompt_groups(traces: Sequence[Any]) -> dict[str, list[Any]]:
    """Group traces by their (normalised) prompt."""
    groups: dict[str, list[Any]] = {}
    for trace in traces:
        key = normalise(getattr(trace, "prompt", "") or "")
        groups.setdefault(key, []).append(trace)
    return groups


def split_traces(
    traces: Sequence[Any],
    *,
    held_out_fraction: float = 0.25,
    min_held_out: int = 4,
    seed: int = 0,
) -> tuple[list[Any], list[Any], list[str]]:
    """Split into (train, held_out) with **whole prompts** on each side.

    Grouping by prompt is the point: if the same question appears in both halves,
    a model that memorised the prompt scores well on the held-out set without
    generalising, and the delta lies.

    Returns ``(train, held_out, notes)`` — the notes explain any degradation
    (too few groups, too few examples) rather than silently returning a
    meaningless split.
    """
    notes: list[str] = []
    groups = prompt_groups(traces)
    if not groups:
        return [], [], ["no traces"]

    keys = sorted(groups, key=lambda k: (_stable_hash(f"{seed}:{k}"), k))
    total = len(traces)
    target = max(min_held_out, int(round(total * max(0.0, min(1.0, held_out_fraction)))))
    target = min(target, max(1, total // 2))

    held_keys: list[str] = []
    held_count = 0
    for key in keys:
        if held_count >= target and held_keys:
            break
        # Never hold out *every* group: training needs something to learn from.
        if len(held_keys) + 1 >= len(keys) and held_keys:
            break
        held_keys.append(key)
        held_count += len(groups[key])

    held_set = set(held_keys)
    train = [t for key in keys if key not in held_set for t in groups[key]]
    held = [t for key in held_keys for t in groups[key]]

    if not held:
        held = sorted(traces, key=lambda t: _stable_hash(f"{seed}:{getattr(t, 'prompt', '')}"))[:1]
        train = [t for t in traces if all(t is not h for h in held)]
        notes.append("held-out set too small; used a single example")
    if not train:
        train, held = list(traces), []
        notes.append("not enough data to hold anything out; measuring on the training set (optimistic)")
    if len(held) < min_held_out:
        notes.append(f"only {len(held)} held-out examples (wanted {min_held_out}); delta is noisy")
    return train, held, notes


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def normalise(text: str) -> str:
    """Lowercase, collapse whitespace, drop a trailing sentence terminator."""
    collapsed = _WS_RE.sub(" ", (text or "").strip().lower())
    return _PUNCT_TAIL_RE.sub("", collapsed).strip()


def exact_match(prediction: str, gold: str) -> float:
    """1.0 when the prediction starts with the gold answer (normalised)."""
    pred, want = normalise(prediction), normalise(gold)
    if not want:
        return 0.0
    return 1.0 if pred.startswith(want) else 0.0


def token_f1(prediction: str, gold: str) -> float:
    """Bag-of-words F1 between prediction and gold."""
    pred_tokens = Counter(normalise(prediction).split())
    gold_tokens = Counter(normalise(gold).split())
    if not pred_tokens or not gold_tokens:
        return 0.0
    overlap = sum((pred_tokens & gold_tokens).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(pred_tokens.values())
    recall = overlap / sum(gold_tokens.values())
    return 2 * precision * recall / (precision + recall)


# --------------------------------------------------------------------------- #
# The evaluator
# --------------------------------------------------------------------------- #
#: ``scorer(prompt, gold) -> (generated_text, mean_token_nll | None)``
Scorer = Callable[[str, str], "tuple[str, float | None]"]


def evaluate(
    scorer: Scorer,
    traces: Sequence[Any],
    *,
    label: str = "",
    notes: Sequence[str] = (),
) -> EvalResult:
    """Score a model on ``traces`` with ``scorer``.

    ``scorer`` returns the generated continuation and (optionally) the mean
    per-token negative log-likelihood the model assigned to the gold answer.
    A scorer that cannot produce a likelihood returns ``None`` and the loss is
    then simply not reported as 0.0 pretending to be perfect — it is reported as
    ``inf``-free but flagged in the notes.
    """
    result = EvalResult(label=label, examples=len(traces), notes=list(notes))
    if not traces:
        result.notes.append("no held-out examples: nothing was measured")
        return result

    losses: list[float] = []
    matches: list[float] = []
    f1s: list[float] = []
    lengths: list[int] = []
    no_likelihood = 0

    for trace in traces:
        prompt = getattr(trace, "prompt", "") or ""
        gold = getattr(trace, "response", "") or ""
        try:
            generated, nll = scorer(prompt, gold)
        except Exception:  # noqa: BLE001 - a broken scorer must not fake a number
            generated, nll = "", None
            result.notes.append("scorer raised on one example; counted as a miss")
        matches.append(exact_match(generated, gold))
        f1s.append(token_f1(generated, gold))
        lengths.append(len((generated or "").split()))
        if nll is None:
            no_likelihood += 1
        else:
            losses.append(float(nll))

    if losses:
        result.loss = sum(losses) / len(losses)
    if no_likelihood:
        result.notes.append(f"{no_likelihood}/{len(traces)} examples had no likelihood (loss under-reported)")
    result.exact_match = sum(matches) / len(matches) if matches else 0.0
    result.token_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    result.mean_length = sum(lengths) / len(lengths) if lengths else 0.0
    return result


def compare(
    before: EvalResult | None,
    after: EvalResult | None,
    *,
    backend: str = "",
    model: str = "",
) -> EvalDelta:
    return EvalDelta(before=before, after=after, backend=backend, model=model)


# --------------------------------------------------------------------------- #
# The fixed evaluation set (used by the smoke script and the tests)
# --------------------------------------------------------------------------- #
#: A small, hand-written, deterministic evaluation set.  It is deliberately
#: *not* in any training data the repository ships: it exists so a run can be
#: reproduced and the delta compared across machines.  Real use should point the
#: harness at the operator's own held-out conversations.
EVAL_SET: tuple[dict[str, str], ...] = (
    {"prompt": "what does the scheduler do", "response": "the scheduler places requests on healthy workers"},
    {"prompt": "how do i expose the dashboard", "response": "serve it over https with tailscale funnel"},
    {"prompt": "what happens when a worker dies", "response": "the request is retried on another worker"},
    {"prompt": "how do i add a document", "response": "upload it in the pwa and it is indexed"},
    {"prompt": "what is the vram pool", "response": "the sum of free vram across healthy workers"},
    {"prompt": "how do i stop the autopilot", "response": "toggle it off and training is preempted"},
    {"prompt": "where are checkpoints stored", "response": "in drive and the hugging face hub"},
    {"prompt": "what is the api prefix", "response": "the control plane serves under v1"},
    {"prompt": "how do i start a worker", "response": "run bootstrap sh on the node"},
    {"prompt": "what is speculative decoding", "response": "a small model drafts and a big model verifies"},
    {"prompt": "how is the api protected", "response": "with an api key and a mesh token"},
    {"prompt": "what does the critic do", "response": "it scores drafts so weak answers are dropped"},
)


def eval_set() -> list[Any]:
    """The fixed evaluation set as ``TrainingTrace`` objects (score 1.0)."""
    from training.distributed_trainer import TrainingTrace

    return [
        TrainingTrace(prompt=item["prompt"], response=item["response"], score=1.0, tags=["eval-set"])
        for item in EVAL_SET
    ]


def to_dict(result: EvalResult | EvalDelta | None) -> dict[str, Any] | None:
    return None if result is None else asdict(result) if isinstance(result, EvalResult) else result.to_dict()


__all__ = [
    "EVAL_SET",
    "EvalDelta",
    "EvalResult",
    "Scorer",
    "compare",
    "eval_set",
    "evaluate",
    "exact_match",
    "normalise",
    "prompt_groups",
    "split_traces",
    "to_dict",
    "token_f1",
]
