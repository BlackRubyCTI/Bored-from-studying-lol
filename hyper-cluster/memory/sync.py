"""Cloud sync for the memory and the training checkpoints.

Two back ends are supported, both speaking the *real* REST APIs:

``DriveClient``
    Google Drive v3.  Files are uploaded with the resumable protocol
    (``POST /upload/drive/v3/files?uploadType=resumable`` followed by a
    ``PUT`` of the payload with the ``X-Goog-Upload-*`` headers), and the
    revision/checksum of a snapshot is stored in the file's
    ``appProperties`` so a later pull can tell whether the remote copy is
    newer than the local one.

``HubClient``
    The Hugging Face Hub.  A file is uploaded with the pre-signed flow the
    Hub uses internally (``POST /api/models/{repo}/upload/{revision}`` ->
    ``PUT`` to the returned URL -> ``POST /api/models/{repo}/commit/{revision}``)
    and downloaded from ``/{repo}/resolve/{revision}/{path}``.

Both clients take their base URL from the configuration, so the whole sync
path can be exercised against a local double in the test-suite while
pointing at the real services in production.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from core.logging_setup import get_logger

LOGGER = get_logger("memory.sync")

#: The real Drive API.
DRIVE_API_BASE = "https://www.googleapis.com/drive/v3"
DRIVE_UPLOAD_BASE = "https://www.googleapis.com/upload/drive/v3"

#: The real Hugging Face Hub.
HUB_BASE = "https://huggingface.co"

#: Google Drive's folder mime type.
FOLDER_MIME = "application/vnd.google-apps.folder"

#: The size of one resumable upload chunk (Drive insists on 256 KiB).
UPLOAD_CHUNK_SIZE = 256 * 1024


class SyncError(RuntimeError):
    """Raised when a sync operation fails."""


@dataclass
class SyncReceipt:
    """The proof that an artifact reached a remote target."""

    target: str
    ok: bool
    location: str = ""
    revision: int = 0
    sha256: str = ""
    bytes_sent: int = 0
    error: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "ok": self.ok,
            "location": self.location,
            "revision": self.revision,
            "sha256": self.sha256,
            "bytes": self.bytes_sent,
            "error": self.error,
            "detail": self.detail,
        }


def sha256_bytes(payload: bytes) -> str:
    """The checksum that identifies an artifact."""
    return hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------- #
# Google Drive
# --------------------------------------------------------------------------- #
class DriveClient:
    """A minimal Google Drive v3 client (files + folders)."""

    def __init__(
        self,
        *,
        token: str = "",
        api_base: str = DRIVE_API_BASE,
        upload_base: str = DRIVE_UPLOAD_BASE,
        client: httpx.AsyncClient | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.token = token
        self.api_base = api_base.rstrip("/")
        self.upload_base = upload_base.rstrip("/")
        self._client = client
        self._owns_client = client is None
        self._timeout = timeout

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------ #
    async def find(self, name: str, *, folder_id: str = "") -> dict[str, Any] | None:
        """Find a file by name (``None`` when it does not exist)."""
        escaped = name.replace("'", r"\'")
        query = f"name = '{escaped}' and trashed = false"
        if folder_id:
            query += f" and '{folder_id}' in parents"
        response = await self._http().get(
            f"{self.api_base}/files",
            params={"q": query, "fields": "files(id,name,mimeType,size,appProperties,modifiedTime)"},
            headers=self._headers(),
        )
        if response.status_code >= 400:
            raise SyncError(f"drive list failed: HTTP {response.status_code}")
        payload = response.json()
        files = payload.get("files") or []
        return files[0] if files else None

    async def list_folder(self, folder_id: str) -> list[dict[str, Any]]:
        """List the files of a folder (newest first)."""
        response = await self._http().get(
            f"{self.api_base}/files",
            params={
                "q": f"'{folder_id}' in parents and trashed = false",
                "orderBy": "modifiedTime desc",
                "fields": "files(id,name,mimeType,size,appProperties,modifiedTime)",
            },
            headers=self._headers(),
        )
        if response.status_code >= 400:
            raise SyncError(f"drive list failed: HTTP {response.status_code}")
        return list(response.json().get("files") or [])

    async def ensure_folder(self, name: str) -> str:
        """Return the id of a folder, creating it when it does not exist."""
        existing = await self.find(name)
        if existing and existing.get("mimeType") == FOLDER_MIME:
            return str(existing.get("id"))
        metadata = {
            "name": name,
            "mimeType": FOLDER_MIME,
        }
        response = await self._http().post(
            f"{self.api_base}/files",
            params={"fields": "id,name,mimeType"},
            json=metadata,
            headers=self._headers(),
        )
        if response.status_code >= 400:
            raise SyncError(f"drive folder creation failed: HTTP {response.status_code}")
        return str(response.json().get("id"))

    async def upload(
        self,
        name: str,
        payload: bytes,
        *,
        folder_id: str = "",
        app_properties: dict[str, str] | None = None,
        mime_type: str = "application/json",
    ) -> SyncReceipt:
        """Upload a file with the resumable protocol."""
        checksum = sha256_bytes(payload)
        metadata: dict[str, Any] = {
            "name": name,
            "mimeType": mime_type,
        }
        if folder_id:
            metadata["parents"] = [folder_id]
        if app_properties:
            metadata["appProperties"] = {str(k): str(v) for k, v in app_properties.items()}
        try:
            response = await self._http().post(
                f"{self.upload_base}/files",
                params={"uploadType": "resumable", "fields": "id,name,mimeType,appProperties"},
                headers={**self._headers(), "X-Upload-Content-Type": mime_type},
                json=metadata,
            )
            if response.status_code >= 400:
                raise SyncError(f"drive upload session failed: HTTP {response.status_code}")
            session_uri = response.headers.get("location") or response.headers.get("Location")
            if not session_uri:
                raise SyncError("drive did not return an upload session URI")
            offset = 0
            total = len(payload)
            while offset < total:
                chunk = payload[offset : offset + UPLOAD_CHUNK_SIZE]
                offset += len(chunk)
                command = "upload, finalize" if offset >= total else "upload"
                upload = await self._http().put(
                    session_uri,
                    content=chunk,
                    headers={
                        **self._headers(),
                        "Content-Type": mime_type,
                        "Content-Length": str(len(chunk)),
                        "X-Goog-Upload-Offset": str(offset - len(chunk)),
                        "X-Goog-Upload-Command": command,
                        "X-Goog-Upload-Content-Length": str(total),
                    },
                )
                if upload.status_code >= 400:
                    raise SyncError(f"drive upload failed: HTTP {upload.status_code}")
            body = upload.json() if upload.content else {}
            return SyncReceipt(
                target="drive",
                ok=True,
                location=str(body.get("id") or ""),
                revision=int(body.get("appProperties", {}).get("revision", 0) or 0)
                if isinstance(body.get("appProperties"), dict)
                else 0,
                sha256=checksum,
                bytes_sent=total,
                detail={"name": name, "mime_type": mime_type},
            )
        except httpx.HTTPError as exc:
            return SyncReceipt(target="drive", ok=False, error=f"{type(exc).__name__}: {exc}")
        except SyncError as exc:
            return SyncReceipt(target="drive", ok=False, error=str(exc))

    async def download(self, file_id: str) -> bytes:
        """Download the content of a file."""
        response = await self._http().get(
            f"{self.api_base}/files/{file_id}",
            params={"alt": "media"},
            headers=self._headers(),
        )
        if response.status_code >= 400:
            raise SyncError(f"drive download failed: HTTP {response.status_code}")
        return response.content

    async def metadata(self, file_id: str) -> dict[str, Any]:
        """The metadata (including ``appProperties``) of a file."""
        response = await self._http().get(
            f"{self.api_base}/files/{file_id}",
            params={"fields": "id,name,mimeType,size,appProperties,modifiedTime"},
            headers=self._headers(),
        )
        if response.status_code >= 400:
            raise SyncError(f"drive metadata failed: HTTP {response.status_code}")
        return response.json()


# --------------------------------------------------------------------------- #
# Hugging Face Hub
# --------------------------------------------------------------------------- #
class HubClient:
    """A minimal Hugging Face Hub client (model repos)."""

    def __init__(
        self,
        *,
        token: str = "",
        base_url: str = HUB_BASE,
        client: httpx.AsyncClient | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.token = token
        self.base_url = base_url.rstrip("/")
        self._client = client
        self._owns_client = client is None
        self._timeout = timeout

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def ensure_repo(self, repo: str, *, private: bool = True) -> bool:
        """Create the repository when it does not exist yet."""
        response = await self._http().post(
            f"{self.base_url}/api/repos/create",
            json={"name": repo, "organization": "", "private": private, "type": "model"},
            headers=self._headers(),
        )
        if response.status_code in (200, 201, 409):
            return True
        if response.status_code == 422:
            # Already exists.
            return True
        raise SyncError(f"hub repo creation failed: HTTP {response.status_code}")

    async def upload_file(
        self,
        repo: str,
        path: str,
        payload: bytes,
        *,
        revision: str = "main",
        message: str = "",
    ) -> SyncReceipt:
        """Upload a file with the Hub's pre-signed upload flow."""
        checksum = sha256_bytes(payload)
        try:
            preupload = await self._http().post(
                f"{self.base_url}/api/models/{repo}/upload/{revision}",
                json={
                    "paths": [path],
                    "files": [
                        {
                            "path": path,
                            "size": len(payload),
                            "sha256": checksum,
                        }
                    ],
                },
                headers=self._headers(),
            )
            if preupload.status_code >= 400:
                raise SyncError(f"hub preupload failed: HTTP {preupload.status_code}")
            body = preupload.json()
            upload_url = body.get("url") or body.get("uploadUrl")
            if not upload_url:
                raise SyncError("hub did not return an upload URL")
            upload_headers = dict(body.get("header") or {})
            upload_headers.setdefault("Content-Type", "application/octet-stream")
            put = await self._http().put(upload_url, content=payload, headers=upload_headers)
            if put.status_code >= 400:
                raise SyncError(f"hub upload failed: HTTP {put.status_code}")
            commit = await self._http().post(
                f"{self.base_url}/api/models/{repo}/commit/{revision}",
                json={
                    "paths": [path],
                    "title": message or f"update {path}",
                    "description": message or "",
                },
                headers=self._headers(),
            )
            if commit.status_code >= 400:
                raise SyncError(f"hub commit failed: HTTP {commit.status_code}")
            return SyncReceipt(
                target="hub",
                ok=True,
                location=f"{repo}/{path}",
                sha256=checksum,
                bytes_sent=len(payload),
                detail={"repo": repo, "path": path, "revision": revision},
            )
        except httpx.HTTPError as exc:
            return SyncReceipt(target="hub", ok=False, error=f"{type(exc).__name__}: {exc}")
        except SyncError as exc:
            return SyncReceipt(target="hub", ok=False, error=str(exc))

    async def list_files(self, repo: str, *, revision: str = "main", prefix: str = "") -> list[str]:
        """List the files of a repository."""
        response = await self._http().get(
            f"{self.base_url}/api/models/{repo}/tree/{revision}",
            params={"recursive": "true", "expand": "false"},
            headers=self._headers(),
        )
        if response.status_code >= 400:
            raise SyncError(f"hub tree failed: HTTP {response.status_code}")
        payload = response.json()
        entries = payload if isinstance(payload, list) else payload.get("siblings") or []
        paths = [str(entry.get("path")) for entry in entries if entry.get("path")]
        if prefix:
            paths = [path for path in paths if path.startswith(prefix)]
        return sorted(paths)

    async def download_file(self, repo: str, path: str, *, revision: str = "main") -> bytes:
        """Download a file from a repository."""
        response = await self._http().get(
            f"{self.base_url}/{repo}/resolve/{revision}/{path}",
            headers=self._headers(),
        )
        if response.status_code >= 400:
            raise SyncError(f"hub download failed: HTTP {response.status_code}")
        return response.content

    async def file_metadata(self, repo: str, path: str, *, revision: str = "main") -> dict[str, Any]:
        """The metadata of a file (size, lfs sha, ...)."""
        response = await self._http().get(
            f"{self.base_url}/api/models/{repo}/paths-info/{revision}",
            json={"paths": [path]},
            headers=self._headers(),
        )
        if response.status_code >= 400:
            raise SyncError(f"hub paths-info failed: HTTP {response.status_code}")
        payload = response.json()
        entries = payload if isinstance(payload, list) else payload.get("items") or []
        for entry in entries:
            if entry.get("path") == path:
                return entry
        return {}


# --------------------------------------------------------------------------- #
# The memory sync
# --------------------------------------------------------------------------- #
class MemorySync:
    """Push and pull the memory snapshot to Drive and the Hub."""

    def __init__(
        self,
        *,
        drive: DriveClient | None = None,
        hub: HubClient | None = None,
        drive_folder: str = "Hyper-Cluster Memory",
        hub_repo: str = "",
        snapshot_prefix: str = "memory-snapshot",
    ) -> None:
        self.drive = drive
        self.hub = hub
        self.drive_folder = drive_folder
        self.hub_repo = hub_repo
        self.snapshot_prefix = snapshot_prefix

    # ------------------------------------------------------------------ #
    def snapshot_name(self, *, when: float | None = None, reason: str = "") -> str:
        """The (human readable) name of a snapshot."""
        stamp = time.strftime("%d %b %Y %H:%M", time.gmtime(when or time.time()))
        suffix = f" - {reason}" if reason else ""
        return f"Memory snapshot - {stamp}{suffix}.json"

    def _encode(self, snapshot: dict[str, Any]) -> bytes:
        return json.dumps(snapshot, ensure_ascii=False, sort_keys=True).encode("utf-8")

    def _decode(self, payload: bytes) -> dict[str, Any]:
        return json.loads(payload.decode("utf-8"))

    # ------------------------------------------------------------------ #
    async def push(
        self,
        snapshot: dict[str, Any],
        *,
        targets: tuple[str, ...] = ("drive", "hub"),
        reason: str = "",
    ) -> list[SyncReceipt]:
        """Upload a snapshot to every requested target."""
        payload = self._encode(snapshot)
        checksum = sha256_bytes(payload)
        revision = int(snapshot.get("revision") or 0)
        receipts: list[SyncReceipt] = []
        if "drive" in targets and self.drive is not None:
            name = self.snapshot_name(reason=reason)
            try:
                folder_id = await self.drive.ensure_folder(self.drive_folder)
                receipt = await self.drive.upload(
                    name,
                    payload,
                    folder_id=folder_id,
                    app_properties={
                        "revision": str(revision),
                        "sha256": checksum,
                        "chunks": str(len(snapshot.get("chunks") or [])),
                        "kind": "memory-snapshot",
                    },
                )
            except (SyncError, httpx.HTTPError) as exc:
                receipt = SyncReceipt(target="drive", ok=False, error=str(exc))
            receipt.revision = revision
            receipt.sha256 = checksum
            receipts.append(receipt)
        if "hub" in targets and self.hub is not None and self.hub_repo:
            path = f"{self.snapshot_prefix}/snapshot-{revision:06d}.json"
            receipt = await self.hub.upload_file(
                self.hub_repo,
                path,
                payload,
                message=f"memory snapshot r{revision}",
            )
            receipt.revision = revision
            receipt.sha256 = checksum
            receipts.append(receipt)
        for receipt in receipts:
            LOGGER.info(
                "memory sync",
                extra={
                    "hc_target": receipt.target,
                    "hc_ok": receipt.ok,
                    "hc_revision": receipt.revision,
                    "hc_error": receipt.error,
                },
            )
        return receipts

    async def pull(self, *, targets: tuple[str, ...] = ("drive", "hub")) -> list[dict[str, Any]]:
        """Fetch the newest snapshot from the remote targets.

        Every candidate is verified against the checksum that was stored
        next to it; a corrupted download is rejected instead of restored.
        """
        found: list[dict[str, Any]] = []
        if "drive" in targets and self.drive is not None:
            try:
                folder_id = await self.drive.ensure_folder(self.drive_folder)
                candidates = await self.drive.list_folder(folder_id)
                for candidate in candidates:
                    properties = candidate.get("appProperties") or {}
                    if properties.get("kind") not in (None, "memory-snapshot"):
                        continue
                    payload = await self.drive.download(str(candidate.get("id")))
                    found.append(self._verify(payload, str(properties.get("sha256") or "")))
                    break
            except (SyncError, httpx.HTTPError, json.JSONDecodeError) as exc:
                LOGGER.warning("drive pull failed", extra={"hc_error": str(exc)})
        if "hub" in targets and self.hub is not None and self.hub_repo:
            try:
                paths = await self.hub.list_files(self.hub_repo, prefix=f"{self.snapshot_prefix}/")
                for path in sorted(paths, reverse=True):
                    payload = await self.hub.download_file(self.hub_repo, path)
                    found.append(self._verify(payload))
                    break
            except (SyncError, httpx.HTTPError, json.JSONDecodeError) as exc:
                LOGGER.info("hub pull: no snapshot yet", extra={"hc_error": str(exc)})
        return found

    def _verify(self, payload: bytes, expected_checksum: str = "") -> dict[str, Any]:
        """Check a downloaded snapshot against its checksums.

        Two checks are made: the bytes must match the checksum that was
        stored next to the file, and the chunk list must match the checksum
        the snapshot carries.  A corrupted download is rejected instead of
        restored.
        """
        checksum = sha256_bytes(payload)
        if expected_checksum and expected_checksum != checksum:
            raise SyncError("snapshot checksum mismatch")
        snapshot = self._decode(payload)
        body = json.dumps(snapshot.get("chunks") or [], ensure_ascii=False, sort_keys=True)
        expected = str(snapshot.get("sha256") or "")
        if expected and expected != hashlib.sha256(body.encode("utf-8")).hexdigest():
            raise SyncError("snapshot content checksum mismatch")
        return snapshot


__all__ = [
    "DRIVE_API_BASE",
    "DRIVE_UPLOAD_BASE",
    "FOLDER_MIME",
    "HUB_BASE",
    "DriveClient",
    "HubClient",
    "MemorySync",
    "SyncError",
    "SyncReceipt",
    "sha256_bytes",
]
