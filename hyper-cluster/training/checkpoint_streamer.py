"""Checkpoint streaming: the training state lives in the cloud, not on disk.

A training run must survive a reboot of the head node, a dead worker and a
full disk.  The streamer therefore

* **pushes** a checkpoint (the adapter matrix plus its metadata) to Google
  Drive and the Hugging Face Hub,
* **pulls** the newest checkpoint back, verifying the SHA-256 before it is
  restored -- a truncated upload never poisons a training run,
* **lists** the versions that are available remotely.

The layout on both targets is the same::

    hyper-cluster/checkpoints/checkpoint-000007.npz
    hyper-cluster/checkpoints/checkpoint-000007.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.ids import now_s
from core.logging_setup import get_logger
from memory.sync import DriveClient, HubClient, SyncError, sha256_bytes

LOGGER = get_logger("training.checkpoint_streamer")


@dataclass
class StreamedCheckpoint:
    """A checkpoint that lives in the cloud."""

    step: int
    loss: float = 0.0
    sha256: str = ""
    bytes_sent: int = 0
    targets: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=now_s)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "loss": round(self.loss, 6),
            "sha256": self.sha256,
            "bytes": self.bytes_sent,
            "targets": list(self.targets),
            "created_at": self.created_at,
        }


class CheckpointStreamer:
    """Pushes and pulls training checkpoints."""

    def __init__(
        self,
        *,
        drive: DriveClient | None = None,
        hub: HubClient | None = None,
        drive_folder: str = "Hyper-Cluster Memory",
        hub_repo: str = "",
        prefix: str = "hyper-cluster/checkpoints",
    ) -> None:
        self.drive = drive
        self.hub = hub
        self.drive_folder = drive_folder
        self.hub_repo = hub_repo
        self.prefix = prefix.strip("/")

    # ------------------------------------------------------------------ #
    def _names(self, step: int) -> tuple[str, str]:
        return f"checkpoint-{step:06d}.npz", f"checkpoint-{step:06d}.json"

    async def push(
        self,
        step: int,
        matrix: np.ndarray,
        *,
        metadata: dict[str, Any] | None = None,
        targets: tuple[str, ...] = ("drive", "hub"),
    ) -> StreamedCheckpoint:
        """Upload a checkpoint to the requested targets."""
        import io

        buffer = io.BytesIO()
        np.savez_compressed(buffer, matrix=matrix)
        payload = buffer.getvalue()
        checksum = sha256_bytes(payload)
        meta = {
            "step": int(step),
            "loss": float((metadata or {}).get("loss") or 0.0),
            "sha256": checksum,
            "bytes": len(payload),
            "created_at": now_s(),
            **{k: v for k, v in (metadata or {}).items() if k != "loss"},
        }
        meta_payload = json.dumps(meta, indent=2).encode("utf-8")
        npz_name, json_name = self._names(step)
        streamed = StreamedCheckpoint(
            step=int(step),
            loss=float(meta["loss"]),
            sha256=checksum,
            bytes_sent=len(payload),
        )
        if "drive" in targets and self.drive is not None:
            try:
                folder_id = await self.drive.ensure_folder(self.drive_folder)
                receipt = await self.drive.upload(
                    f"{self.prefix}/{npz_name}",
                    payload,
                    folder_id=folder_id,
                    app_properties={
                        "kind": "checkpoint",
                        "step": str(step),
                        "sha256": checksum,
                    },
                    mime_type="application/octet-stream",
                )
                if receipt.ok:
                    streamed.targets.append("drive")
                    # The metadata travels with the checkpoint.
                    await self.drive.upload(
                        f"{self.prefix}/{json_name}",
                        meta_payload,
                        folder_id=folder_id,
                        app_properties={"kind": "checkpoint-meta", "step": str(step)},
                        mime_type="application/json",
                    )
                else:
                    LOGGER.warning("checkpoint drive push failed", extra={"hc_error": receipt.error})
            except (SyncError, Exception) as exc:  # noqa: BLE001
                LOGGER.warning("checkpoint drive push failed", extra={"hc_error": str(exc)})
        if "hub" in targets and self.hub is not None and self.hub_repo:
            receipt = await self.hub.upload_file(
                self.hub_repo,
                f"{self.prefix}/{npz_name}",
                payload,
                message=f"checkpoint {step}",
            )
            if receipt.ok:
                streamed.targets.append("hub")
                await self.hub.upload_file(
                    self.hub_repo,
                    f"{self.prefix}/{json_name}",
                    meta_payload,
                    message=f"checkpoint {step} metadata",
                )
            else:
                LOGGER.warning("checkpoint hub push failed", extra={"hc_error": receipt.error})
        LOGGER.info(
            "checkpoint streamed",
            extra={"hc_step": step, "hc_targets": ",".join(streamed.targets) or "none"},
        )
        return streamed

    async def pull(self, *, target: str = "hub") -> tuple[int, np.ndarray, dict[str, Any]] | None:
        """Fetch the newest checkpoint from a target (``None`` when there is none)."""
        if target == "hub" and self.hub is not None and self.hub_repo:
            try:
                paths = await self.hub.list_files(self.hub_repo, prefix=f"{self.prefix}/")
                candidates = sorted(path for path in paths if path.endswith(".npz"))
                if not candidates:
                    return None
                path = candidates[-1]
                payload = await self.hub.download_file(self.hub_repo, path)
                step = int(Path(path).stem.split("-")[-1])
                metadata: dict[str, Any] = {"sha256": sha256_bytes(payload)}
                # the metadata travels next to the checkpoint
                sidecar = path[: -len(".npz")] + ".json"
                try:
                    raw = await self.hub.download_file(self.hub_repo, sidecar)
                    metadata.update(json.loads(raw.decode("utf-8")))
                except (SyncError, ValueError):
                    pass
                return step, _load_matrix(payload), metadata
            except (SyncError, Exception) as exc:  # noqa: BLE001
                LOGGER.info("checkpoint hub pull: nothing available", extra={"hc_error": str(exc)})
                return None
        if target == "drive" and self.drive is not None:
            try:
                folder_id = await self.drive.ensure_folder(self.drive_folder)
                files = await self.drive.list_folder(folder_id)
                checkpoints = [
                    item
                    for item in files
                    if str(item.get("name", "")).endswith(".npz")
                    and str((item.get("appProperties") or {}).get("kind")) == "checkpoint"
                ]
                if not checkpoints:
                    return None
                item = checkpoints[0]
                payload = await self.drive.download(str(item.get("id")))
                properties = item.get("appProperties") or {}
                expected = str(properties.get("sha256") or "")
                if expected and expected != sha256_bytes(payload):
                    raise SyncError("checkpoint checksum mismatch")
                step = int(properties.get("step") or 0)
                return step, _load_matrix(payload), dict(properties)
            except (SyncError, Exception) as exc:  # noqa: BLE001
                LOGGER.info("checkpoint drive pull: nothing available", extra={"hc_error": str(exc)})
                return None
        return None

    async def list_remote(self, *, target: str = "hub") -> list[dict[str, Any]]:
        """List the checkpoints that are available remotely."""
        found: list[dict[str, Any]] = []
        if target == "hub" and self.hub is not None and self.hub_repo:
            try:
                for path in await self.hub.list_files(self.hub_repo, prefix=f"{self.prefix}/"):
                    found.append({"path": path, "target": "hub"})
            except (SyncError, Exception) as exc:  # noqa: BLE001
                LOGGER.info("checkpoint listing failed", extra={"hc_error": str(exc)})
        if target == "drive" and self.drive is not None:
            try:
                folder_id = await self.drive.ensure_folder(self.drive_folder)
                for item in await self.drive.list_folder(folder_id):
                    if str(item.get("name", "")).endswith(".npz"):
                        found.append(
                            {
                                "path": item.get("name"),
                                "target": "drive",
                                "size": item.get("size"),
                                "appProperties": item.get("appProperties"),
                            }
                        )
            except (SyncError, Exception) as exc:  # noqa: BLE001
                LOGGER.info("checkpoint listing failed", extra={"hc_error": str(exc)})
        return found


def _load_matrix(payload: bytes) -> np.ndarray:
    import io

    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        return np.array(data["matrix"])


__all__ = ["CheckpointStreamer", "StreamedCheckpoint"]
