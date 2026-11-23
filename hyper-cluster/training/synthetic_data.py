"""Synthetic data generation for the sleep-time self-evolution engine.

The cluster learns from its own conversations.  This module turns the mined
history into a *training set*:

1. **mine** -- the questions and answers the cluster has already seen are
   extracted from the memory (every exchange is stored there);
2. **mutate** -- each seed is turned into harder variants (constraints, a
   role, a counterfactual, a format, a second step).  A model that only
   memorises the seed answer fails the mutated prompt;
3. **assemble** -- the mutations are filtered for novelty, labelled with a
   difficulty and paired with a reference answer.

Everything is deterministic: the same seed and the same RNG seed always
produce the same dataset, which is what makes a training run reproducible.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from core.logging_setup import get_logger

LOGGER = get_logger("training.synthetic_data")

#: The audiences a mutated prompt can be re-framed for.
AUDIENCES = (
    "a five year old",
    "a systems architect",
    "a board member",
    "a new joiner on the team",
    "a reviewer who has never seen the project",
)

#: The domains a mutated prompt can be moved to.
DOMAINS = (
    "in the context of a distributed system",
    "in the context of a regulated industry",
    "in the context of a two week deadline",
    "in the context of a failed migration",
)

#: The formats a mutated prompt can demand.
FORMATS = (
    "as a numbered list of at most three items",
    "as a single paragraph of at most four sentences",
    "as a table with a decision and a consequence column",
    "as a short brief with a risk section",
)


@dataclass
class PromptSeed:
    """A mined question/answer pair."""

    prompt: str
    answer: str = ""
    source_id: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "answer": self.answer,
            "source_id": self.source_id,
            "tags": list(self.tags),
        }


@dataclass
class SyntheticExample:
    """One training example.

    An example carries *content* only -- no wall-clock time -- so a dataset
    built from the same seeds with the same seed is byte-for-byte
    identical.  The build time lives on the dataset itself.
    """

    prompt: str
    reference: str = ""
    difficulty: float = 0.0
    origin: str = "mutation"
    tags: list[str] = field(default_factory=list)
    seed_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "reference": self.reference,
            "difficulty": round(self.difficulty, 3),
            "origin": self.origin,
            "tags": list(self.tags),
            "seed_id": self.seed_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SyntheticExample:
        return cls(
            prompt=str(data.get("prompt") or ""),
            reference=str(data.get("reference") or ""),
            difficulty=float(data.get("difficulty") or 0.0),
            origin=str(data.get("origin") or "mutation"),
            tags=list(data.get("tags") or []),
            seed_id=str(data.get("seed_id") or ""),
        )


# --------------------------------------------------------------------------- #
# Mining
# --------------------------------------------------------------------------- #
_QA_PATTERN = re.compile(r"Q:\s*(?P<prompt>.*?)\nA:\s*(?P<answer>.*)", re.DOTALL)


def mine_seeds(chunks: Iterable[Any]) -> list[PromptSeed]:
    """Extract the question/answer pairs from memory chunks.

    ``chunks`` may be :class:`memory.vector_memory.MemoryChunk` objects or
    plain dictionaries with ``text``/``id`` fields.
    """
    seeds: list[PromptSeed] = []
    for chunk in chunks:
        text = getattr(chunk, "text", None)
        chunk_id = getattr(chunk, "id", "")
        if text is None and isinstance(chunk, dict):
            text = chunk.get("text")
            chunk_id = chunk.get("id", "")
        if not text:
            continue
        match = _QA_PATTERN.search(text)
        if not match:
            continue
        prompt = match.group("prompt").strip()
        answer = match.group("answer").strip()
        if len(prompt) < 8 or len(answer) < 8:
            continue
        seeds.append(PromptSeed(prompt=prompt, answer=answer, source_id=str(chunk_id)))
    return seeds


# --------------------------------------------------------------------------- #
# Mutation
# --------------------------------------------------------------------------- #
def mutate_prompt(prompt: str, rng: random.Random, *, difficulty: float = 1.0) -> tuple[str, str]:
    """Mutate a prompt into a harder variant.

    Returns ``(mutated_prompt, tag)``.  The difficulty controls how many
    mutations are stacked: a difficulty of 0 keeps the prompt as it is.
    """
    base = prompt.strip().rstrip("?")
    if not base:
        return prompt, "identity"
    layers: list[str] = []
    tags: list[str] = []

    count = 0
    if difficulty >= 0.34:
        audience = rng.choice(AUDIENCES)
        layers.append(f"Explain this to {audience}.")
        tags.append("audience")
        count += 1
    if difficulty >= 0.5:
        domain = rng.choice(DOMAINS)
        layers.append(f"Answer {domain}.")
        tags.append("domain")
        count += 1
    if difficulty >= 0.67:
        fmt = rng.choice(FORMATS)
        layers.append(f"Give the answer {fmt}.")
        tags.append("format")
        count += 1
    if difficulty >= 0.84:
        layers.append("State the assumption you are making and the risk of being wrong.")
        tags.append("risk")
        count += 1
    if difficulty >= 0.95:
        layers.append("Then give the opposite answer and say why it is wrong.")
        tags.append("counterfactual")
        count += 1
    if not layers:
        return prompt, "identity"
    mutated = f"{base}? {' '.join(layers)}"
    return mutated, "+".join(tags)


def score_difficulty(prompt: str) -> float:
    """A cheap difficulty estimate (0..1)."""
    words = len(prompt.split())
    questions = prompt.count("?")
    constraints = sum(
        marker in prompt.lower()
        for marker in ("at most", "first", "then", "assume", "risk", "critique", "state", "list")
    )
    score = 0.25 * min(words / 90.0, 1.0) + 0.1 * min(questions, 3) + 0.08 * constraints
    return min(1.0, score)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class SyntheticDataset:
    """A deterministic, de-duplicated training set."""

    def __init__(self, examples: Sequence[SyntheticExample] | None = None) -> None:
        self.examples: list[SyntheticExample] = list(examples or [])

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)

    def to_list(self) -> list[dict[str, Any]]:
        return [example.to_dict() for example in self.examples]

    @classmethod
    def from_list(cls, data: Iterable[dict[str, Any]]) -> SyntheticDataset:
        return cls([SyntheticExample.from_dict(item) for item in data])

    def stats(self) -> dict[str, Any]:
        if not self.examples:
            return {"examples": 0, "average_difficulty": 0.0, "tags": {}}
        tags: dict[str, int] = {}
        for example in self.examples:
            for tag in example.tags:
                tags[tag] = tags.get(tag, 0) + 1
        return {
            "examples": len(self.examples),
            "average_difficulty": round(
                sum(example.difficulty for example in self.examples) / len(self.examples), 3
            ),
            "tags": tags,
        }


def build_dataset(
    seeds: Sequence[PromptSeed],
    *,
    count: int = 32,
    difficulty: float = 0.8,
    seed: int = 1337,
) -> SyntheticDataset:
    """Build a dataset from the mined seeds.

    The seeds are shuffled deterministically, mutated and de-duplicated.  A
    mutation keeps the reference answer of its seed: the reference is what
    the *critic* compares against, not the expected answer of the mutated
    question.
    """
    rng = random.Random(seed)
    pool = list(seeds)
    if not pool:
        return SyntheticDataset()
    rng.shuffle(pool)
    examples: list[SyntheticExample] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = max(count * 12, 64)
    while len(examples) < count and attempts < max_attempts:
        attempts += 1
        origin = pool[attempts % len(pool)]
        mutated, tag = mutate_prompt(origin.prompt, rng, difficulty=difficulty)
        key = re.sub(r"\s+", " ", mutated.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        examples.append(
            SyntheticExample(
                prompt=mutated,
                reference=origin.answer,
                difficulty=score_difficulty(mutated),
                origin="mutation" if tag != "identity" else "seed",
                tags=[tag] if tag != "identity" else ["seed"],
                seed_id=origin.source_id,
            )
        )
    LOGGER.info(
        "synthetic dataset built",
        extra={"hc_examples": len(examples), "hc_seeds": len(seeds)},
    )
    return SyntheticDataset(examples)


__all__ = [
    "AUDIENCES",
    "DOMAINS",
    "FORMATS",
    "PromptSeed",
    "SyntheticDataset",
    "SyntheticExample",
    "build_dataset",
    "mine_seeds",
    "mutate_prompt",
    "score_difficulty",
]
