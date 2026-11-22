"""The QLoRA math, shared by the head node and the workers.

The "brain" of the cluster is a small linear model that maps the *features*
of a prompt onto the *tokens* of an answer:

```
x  = features(prompt)            (hashed bag of words, L2 normalised)
y  = features(answer, vocab)     (token counts over the answer vocabulary)
ŷ  = x · W                       (W is dim × |vocab|)
```

Training is a ridge regression on ``W`` -- a real gradient, computed on the
workers, averaged on the head node (federated averaging).  The dense update
``ΔW`` is then factorised with a truncated SVD into a rank-``r`` LoRA pair
``(A, B)`` and merged back into the (int8-quantised) base matrix:

```
ΔW  ≈ B · A                        (rank r ≪ dim)
W'  = dequant(base) + B · A         (the merged brain)
```

Everything here is plain ``numpy`` so it runs on a CPU-only worker, is
reproducible bit for bit and can be shipped over HTTP as a small ``.npz``.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: The size of the hashed prompt feature space.
DEFAULT_DIM = 128

#: The size of the answer vocabulary.
DEFAULT_VOCAB_SIZE = 128

#: The rank of the LoRA adapter.
DEFAULT_RANK = 8

#: The regularisation of the ridge regression.
DEFAULT_L2 = 1e-3

#: How the token vocabulary is built.
_TOKEN_PATTERN = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall((text or "").lower())


def hash_features(text: str, dim: int = DEFAULT_DIM) -> np.ndarray:
    """The hashed, L2-normalised feature vector of a text."""
    vector = np.zeros(dim, dtype=np.float32)
    tokens = _tokens(text)
    if not tokens:
        return vector
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    for token, count in counts.items():
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign * (1.0 + np.log1p(count))
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector


def build_vocabulary(texts: Iterable[str], size: int = DEFAULT_VOCAB_SIZE) -> list[str]:
    """The most frequent tokens of the answers (the readout vocabulary)."""
    counts: dict[str, int] = {}
    for text in texts:
        for token in _tokens(text):
            if len(token) < 3:
                continue
            counts[token] = counts.get(token, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [token for token, _ in ordered[: max(8, int(size))]]


def answer_features(text: str, vocab: Sequence[str]) -> np.ndarray:
    """The token-count vector of an answer over the vocabulary."""
    vector = np.zeros(len(vocab), dtype=np.float32)
    index = {token: position for position, token in enumerate(vocab)}
    for token in _tokens(text):
        position = index.get(token)
        if position is not None:
            vector[position] += 1.0
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector


def featurize(
    prompts: Sequence[str],
    answers: Sequence[str],
    vocab: Sequence[str],
    dim: int = DEFAULT_DIM,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the ``(X, Y)`` matrices of a shard."""
    if not prompts:
        return np.zeros((0, dim), dtype=np.float32), np.zeros((0, len(vocab)), dtype=np.float32)
    X = np.stack([hash_features(prompt, dim) for prompt in prompts])
    Y = np.stack([answer_features(answer, vocab) for answer in answers])
    return X, Y


def local_gradient(
    X: np.ndarray,
    Y: np.ndarray,
    W: np.ndarray,
    *,
    l2: float = DEFAULT_L2,
) -> tuple[np.ndarray, float]:
    """The gradient of the ridge loss and the loss itself.

    ``L = 1/n · ‖XW − Y‖² + λ‖W‖²``  ->  ``∇W = 2/n · Xᵀ(XW − Y) + 2λW``
    """
    if X.size == 0:
        return np.zeros_like(W), 0.0
    residual = X @ W - Y
    loss = float(np.mean(residual**2) + l2 * np.mean(W**2))
    gradient = (2.0 / X.shape[0]) * (X.T @ residual) + 2.0 * l2 * W
    return gradient.astype(np.float32), loss


def apply_update(W: np.ndarray, gradient: np.ndarray, learning_rate: float) -> np.ndarray:
    """One gradient-descent step."""
    return (W - learning_rate * gradient).astype(np.float32)


def average_gradients(
    gradients: Sequence[np.ndarray],
    weights: Sequence[float] | None = None,
) -> np.ndarray:
    """The (weighted) average of the shard gradients -- federated averaging."""
    if not gradients:
        raise ValueError("no gradients to average")
    if weights is None:
        weights = [1.0] * len(gradients)
    total = float(sum(weights))
    if total <= 0:
        weights = [1.0] * len(gradients)
        total = float(len(gradients))
    merged = np.zeros_like(gradients[0], dtype=np.float32)
    for gradient, weight in zip(gradients, weights, strict=False):
        merged += gradient.astype(np.float32) * (float(weight) / total)
    return merged


def lora_factorize(delta: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Split a dense update into a rank-``r`` LoRA pair ``(A, B)``.

    ``ΔW ≈ B · A`` with ``A`` of shape ``(r, dim)`` and ``B`` of shape
    ``(|vocab|, r)``.
    """
    if delta.size == 0:
        raise ValueError("cannot factorize an empty update")
    rank = max(1, min(int(rank), min(delta.shape)))
    U, S, Vt = np.linalg.svd(delta, full_matrices=False)
    root = np.sqrt(S[:rank])
    A = (root[:, None] * Vt[:rank, :]).astype(np.float32)
    B = (U[:, :rank] * root[None, :]).astype(np.float32)
    return A, B


def quantize(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row int8 quantisation (the "Q" of QLoRA)."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.size == 0:
        return matrix.astype(np.int8), np.ones((matrix.shape[0] or 1,), dtype=np.float32)
    scale = np.max(np.abs(matrix), axis=1)
    scale = np.where(scale > 0, scale / 127.0, 1.0).astype(np.float32)
    quantized = np.round(matrix / scale[:, None]).astype(np.int8)
    return quantized, scale


def dequantize(quantized: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """The inverse of :func:`quantize`."""
    return (np.asarray(quantized, dtype=np.float32) * np.asarray(scale, dtype=np.float32)[:, None]).astype(
        np.float32
    )


def merge(base_quantized: np.ndarray, base_scale: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Merge a LoRA adapter into the dequantised base matrix."""
    base = dequantize(base_quantized, base_scale)
    return (base + B @ A).astype(np.float32)


def top_tokens(
    prompt: str,
    W: np.ndarray,
    vocab: Sequence[str],
    *,
    dim: int = DEFAULT_DIM,
    k: int = 5,
) -> list[str]:
    """The tokens the brain associates with a prompt."""
    if W.size == 0 or not vocab:
        return []
    x = hash_features(prompt, dim)
    scores = x @ W
    order = np.argsort(scores)[::-1][: max(1, int(k))]
    return [vocab[int(position)] for position in order if scores[int(position)] > 0]


@dataclass
class BrainArtifact:
    """The trained brain, ready to be shipped to a worker."""

    dim: int = DEFAULT_DIM
    rank: int = DEFAULT_RANK
    vocab: list[str] = field(default_factory=list)
    base_quantized: bytes = b""
    base_scale: bytes = b""
    A: bytes = b""
    B: bytes = b""
    merged: bytes = b""
    steps: int = 0
    loss: float = 0.0
    revision: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(base, scale, A, B)`` as numpy arrays."""
        base = unpack(self.base_quantized) if self.base_quantized else np.zeros((0, 0), np.int8)
        scale = unpack(self.base_scale) if self.base_scale else np.zeros((0,), np.float32)
        A = unpack(self.A) if self.A else np.zeros((0, 0), np.float32)
        B = unpack(self.B) if self.B else np.zeros((0, 0), np.float32)
        return base, scale, A, B

    def merged_matrix(self) -> np.ndarray:
        """The merged brain matrix."""
        if self.merged:
            return unpack(self.merged)
        base, scale, A, B = self.matrices()
        return merge(base, scale, A, B)

    def to_bytes(self) -> bytes:
        """The artifact as a single ``.npz`` blob."""
        buffer = io.BytesIO()
        base, scale, A, B = self.matrices()
        merged = self.merged_matrix()
        np.savez_compressed(
            buffer,
            base_q=base,
            base_scale=scale,
            A=A,
            B=B,
            W=merged,
            meta=np.array(
                json.dumps(
                    {
                        "dim": self.dim,
                        "rank": self.rank,
                        "vocab": self.vocab,
                        "steps": self.steps,
                        "loss": self.loss,
                        "revision": self.revision,
                        "metadata": self.metadata,
                    }
                )
            ),
        )
        return buffer.getvalue()

    @classmethod
    def from_bytes(cls, payload: bytes) -> BrainArtifact:
        """Rebuild an artifact from its ``.npz`` blob."""
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            artifact = cls(
                dim=int(meta.get("dim") or DEFAULT_DIM),
                rank=int(meta.get("rank") or DEFAULT_RANK),
                vocab=list(meta.get("vocab") or []),
                steps=int(meta.get("steps") or 0),
                loss=float(meta.get("loss") or 0.0),
                revision=int(meta.get("revision") or 0),
                metadata=dict(meta.get("metadata") or {}),
            )
            artifact.base_quantized = _pack(data["base_q"])
            artifact.base_scale = _pack(data["base_scale"])
            artifact.A = _pack(data["A"])
            artifact.B = _pack(data["B"])
            artifact.merged = _pack(data["W"])
        return artifact

    def encode(self) -> str:
        """The artifact, base64 encoded (what travels over HTTP)."""
        return base64.b64encode(self.to_bytes()).decode("ascii")

    @classmethod
    def decode(cls, payload: str) -> BrainArtifact:
        """Rebuild an artifact from its base64 representation."""
        return cls.from_bytes(base64.b64decode(payload))

    def hint(self, prompt: str, *, k: int = 5) -> list[str]:
        """The tokens this brain associates with a prompt."""
        return top_tokens(prompt, self.merged_matrix(), self.vocab, dim=self.dim, k=k)

    def sha256(self) -> str:
        """The checksum of the artifact."""
        return hashlib.sha256(self.to_bytes()).hexdigest()


def _pack(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **{"payload": array})
    return buffer.getvalue()


def unpack(payload: bytes, key: str = "payload") -> np.ndarray:
    """The inverse of :func:`_pack`."""
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        return np.array(data[key])


__all__ = [
    "DEFAULT_DIM",
    "DEFAULT_L2",
    "DEFAULT_RANK",
    "DEFAULT_VOCAB_SIZE",
    "BrainArtifact",
    "answer_features",
    "apply_update",
    "average_gradients",
    "build_vocabulary",
    "dequantize",
    "featurize",
    "hash_features",
    "local_gradient",
    "lora_factorize",
    "merge",
    "quantize",
    "top_tokens",
    "unpack",
]
