"""The trained brain of a worker.

The sleep-time engine trains a small LoRA adapter (see
``training/qlora.py``) and ships it to the workers.  A worker keeps the
artifact in memory and uses it to *prime* every request: the tokens the
brain associates with the prompt are added to the system prompt, so the
model (or the simulator) starts from what the cluster learned.

The hot-reload is atomic: the new artifact is decoded and validated first,
and only then does it replace the running brain.  A broken artifact
therefore never takes a worker down.
"""

from __future__ import annotations

import threading
from typing import Any

from core.logging_setup import get_logger

LOGGER = get_logger("worker.brain")

#: The marker the brain uses in the system prompt.
BRAIN_MARKER = "BRAIN CONTEXT:"


class Brain:
    """Holds the LoRA artifact of a worker."""

    def __init__(self) -> None:
        self._artifact: Any | None = None
        #: How many times a brain was hot-reloaded (0 = the base weights).
        self._revision = 0
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    @property
    def loaded(self) -> bool:
        with self._lock:
            return self._artifact is not None

    def load(self, payload: str | bytes) -> dict[str, Any]:
        """Hot-reload the brain from a base64 payload (or raw bytes)."""
        from training.qlora import BrainArtifact

        try:
            if isinstance(payload, bytes):
                artifact = BrainArtifact.from_bytes(payload)
            else:
                artifact = BrainArtifact.decode(payload)
        except Exception as exc:  # noqa: BLE001 - a broken artifact must not kill the worker
            LOGGER.warning("brain load failed", extra={"hc_error": str(exc)})
            raise
        with self._lock:
            self._artifact = artifact
            self._revision += 1
        LOGGER.info(
            "brain loaded",
            extra={
                "hc_steps": artifact.steps,
                "hc_vocab": len(artifact.vocab),
                "hc_sha": artifact.sha256()[:12],
            },
        )
        return self.info()

    def unload(self) -> None:
        """Forget the brain (the worker goes back to its base weights)."""
        with self._lock:
            self._artifact = None
            self._revision += 1

    # ------------------------------------------------------------------ #
    def hint(self, prompt: str, *, k: int = 5) -> list[str]:
        """The tokens the brain associates with a prompt."""
        with self._lock:
            artifact = self._artifact
        if artifact is None:
            return []
        try:
            return artifact.hint(prompt, k=k)
        except Exception:  # noqa: BLE001 - the hint is a nicety, never a blocker
            return []

    def context(self, prompt: str, *, k: int = 5) -> str:
        """The system-prompt fragment the brain contributes."""
        tokens = self.hint(prompt, k=k)
        if not tokens:
            return ""
        return f"{BRAIN_MARKER} {', '.join(tokens)}"

    def info(self) -> dict[str, Any]:
        """A JSON-safe view of the brain."""
        with self._lock:
            artifact = self._artifact
            if artifact is None:
                return {"loaded": False, "revision": self._revision}
            return {
                "loaded": True,
                "dim": artifact.dim,
                "rank": artifact.rank,
                "vocab_size": len(artifact.vocab),
                "steps": artifact.steps,
                "loss": round(artifact.loss, 6),
                "sha256": artifact.sha256(),
                # how many brains this worker has loaded (1 = the first one)
                "revision": self._revision,
                "artifact_revision": artifact.revision,
            }


__all__ = ["BRAIN_MARKER", "Brain"]
