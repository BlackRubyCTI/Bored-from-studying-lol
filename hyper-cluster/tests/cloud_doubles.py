"""Local stand-ins for Google Drive and the Hugging Face Hub.

The doubles speak the *real* REST contracts:

* Drive: ``POST /drive/v3/files`` creates a folder, ``GET /drive/v3/files``
  lists with the ``q`` query syntax, ``POST /upload/drive/v3/files`` opens a
  resumable session (the ``Location`` header) and the ``PUT`` of the payload
  finishes it with the ``X-Goog-Upload-*`` headers;
* Hub: ``POST /api/models/{repo}/upload/{revision}`` returns the upload URL,
  the ``PUT`` stores the bytes and ``POST /api/models/{repo}/commit/{rev}``
  commits them; ``GET /{repo}/resolve/{rev}/{path}`` serves the file.

Everything is kept in memory, so the tests can assert on the exact bytes
that crossed the wire.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse


class _CloudHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # ------------------------------------------------------------------ #
    def log_message(self, *_args: Any) -> None:
        return

    def _body(self) -> bytes:
        length = int(self.headers.get("content-length") or 0)
        return self.rfile.read(length) if length else b""

    def _json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _raw(self, payload: bytes, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    # ------------------------------------------------------------------ #
    def do_POST(self) -> None:  # noqa: N802 - http.server API
        parsed = urlparse(self.path)
        path = parsed.path
        body = self._body()
        state = self.server.state  # type: ignore[attr-defined]

        if path.startswith("/upload/drive/v3/files"):
            metadata = json.loads(body or b"{}")
            file_id = f"drive_{len(state['files']) + 1:04d}"
            state["files"][file_id] = {
                "id": file_id,
                "name": metadata.get("name", ""),
                "mimeType": metadata.get("mimeType", "application/octet-stream"),
                "parents": metadata.get("parents") or [],
                "appProperties": metadata.get("appProperties") or {},
                "content": b"",
                "size": 0,
            }
            session = f"{self.server.base_url}/upload/session/{file_id}"  # type: ignore[attr-defined]
            self.send_response(200)
            self.send_header("Location", session)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        if path.startswith("/upload/session/"):
            file_id = path.rsplit("/", 1)[-1]
            offset = int(self.headers.get("X-Goog-Upload-Offset") or 0)
            record = state["files"].get(file_id)
            if record is None:
                self._json({"error": "unknown session"}, status=404)
                return
            current = record["content"]
            if offset != len(current):
                # The client must continue where it stopped.
                self.send_response(308)
                self.send_header("X-Goog-Upload-Status", "active")
                self.send_header("X-Goog-Upload-Offset", str(len(current)))
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            record["content"] = current + body
            record["size"] = len(record["content"])
            self._json(
                {
                    "id": record["id"],
                    "name": record["name"],
                    "mimeType": record["mimeType"],
                    "appProperties": record["appProperties"],
                    "size": str(record["size"]),
                }
            )
            return

        if path.startswith("/drive/v3/files"):
            metadata = json.loads(body or b"{}")
            file_id = f"drive_{len(state['files']) + 1:04d}"
            state["files"][file_id] = {
                "id": file_id,
                "name": metadata.get("name", ""),
                "mimeType": metadata.get("mimeType", "application/octet-stream"),
                "parents": metadata.get("parents") or [],
                "appProperties": metadata.get("appProperties") or {},
                "content": b"",
                "size": 0,
            }
            self._json({"id": file_id, "name": state["files"][file_id]["name"]})
            return

        if "/upload/" in path and path.startswith("/api/models/"):
            # preupload
            payload = json.loads(body or b"{}")
            repo = path.split("/api/models/", 1)[1].split("/upload/")[0]
            files = payload.get("files") or []
            upload_id = f"hub_{len(state['hub']) + 1:04d}"
            for item in files:
                state["hub"][upload_id] = {
                    "repo": repo,
                    "path": item.get("path", ""),
                    "content": b"",
                    "sha256": item.get("sha256", ""),
                }
            self._json(
                {
                    "url": f"{self.server.base_url}/hub/upload/{upload_id}",  # type: ignore[attr-defined]
                    "header": {},
                    "uploadLfsUrl": "",
                }
            )
            return

        if path.startswith("/hub/upload/"):
            upload_id = path.rsplit("/", 1)[-1]
            record = state["hub"].get(upload_id)
            if record is None:
                self._json({"error": "unknown upload"}, status=404)
                return
            record["content"] = body
            self._json({"ok": True})
            return

        if "/commit/" in path and path.startswith("/api/models/"):
            repo = path.split("/api/models/", 1)[1].split("/commit/")[0]
            revision = path.rsplit("/", 1)[-1]
            for record in state["hub"].values():
                if record["repo"] == repo:
                    record["revision"] = revision
            self._json({"ok": True, "commit": f"{repo}@{revision}"})
            return

        if path.startswith("/api/repos/create"):
            self._json({"ok": True})
            return

        self._json({"error": "not found", "path": path}, status=404)

    # ------------------------------------------------------------------ #
    def do_PUT(self) -> None:  # noqa: N802 - http.server API
        parsed = urlparse(self.path)
        path = parsed.path
        body = self._body()
        state = self.server.state  # type: ignore[attr-defined]

        if path.startswith("/upload/session/"):
            file_id = path.rsplit("/", 1)[-1]
            record = state["files"].get(file_id)
            if record is None:
                self._json({"error": "unknown session"}, status=404)
                return
            offset = int(self.headers.get("X-Goog-Upload-Offset") or 0)
            current = record["content"]
            if offset != len(current):
                self.send_response(308)
                self.send_header("X-Goog-Upload-Status", "active")
                self.send_header("X-Goog-Upload-Offset", str(len(current)))
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            record["content"] = current + body
            record["size"] = len(record["content"])
            self._json(
                {
                    "id": record["id"],
                    "name": record["name"],
                    "mimeType": record["mimeType"],
                    "appProperties": record["appProperties"],
                    "size": str(record["size"]),
                }
            )
            return

        if path.startswith("/hub/upload/"):
            upload_id = path.rsplit("/", 1)[-1]
            record = state["hub"].get(upload_id)
            if record is None:
                self._json({"error": "unknown upload"}, status=404)
                return
            record["content"] = body
            self._json({"ok": True})
            return

        self._json({"error": "not found", "path": path}, status=404)

    # ------------------------------------------------------------------ #
    def do_GET(self) -> None:  # noqa: N802 - http.server API
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        state = self.server.state  # type: ignore[attr-defined]

        if path == "/drive/v3/files":
            items = list(state["files"].values())
            raw_query = (query.get("q") or [""])[0]
            name = _extract_name(raw_query)
            folder = _extract_parent(raw_query)
            if name:
                items = [item for item in items if item["name"] == name]
            if folder:
                items = [item for item in items if folder in item["parents"]]
            if "trashed = false" in raw_query or "trashed=false" in raw_query:
                items = [item for item in items if not item.get("trashed")]
            self._json(
                {
                    "files": [
                        {
                            "id": item["id"],
                            "name": item["name"],
                            "mimeType": item["mimeType"],
                            "size": str(item["size"]),
                            "appProperties": item["appProperties"],
                        }
                        for item in items
                    ]
                }
            )
            return

        if path.startswith("/drive/v3/files/"):
            file_id = path.rsplit("/", 1)[-1]
            record = state["files"].get(file_id)
            if record is None:
                self._json({"error": "not found"}, status=404)
                return
            if (query.get("alt") or [""])[0] == "media":
                self._raw(record["content"])
                return
            self._json(
                {
                    "id": record["id"],
                    "name": record["name"],
                    "mimeType": record["mimeType"],
                    "size": str(record["size"]),
                    "appProperties": record["appProperties"],
                }
            )
            return

        if path.startswith("/api/models/") and "/tree/" in path:
            repo = path.split("/api/models/", 1)[1].split("/tree/")[0]
            entries = [
                {"path": record["path"], "size": len(record["content"])}
                for record in state["hub"].values()
                if record["repo"] == repo
            ]
            self._json(entries)
            return

        if "/resolve/" in path:
            # /{repo}/resolve/{revision}/{path} -- the repo itself contains
            # a slash ("someone/hyper-cluster"), so split on the marker.
            left, _, right = path.lstrip("/").partition("/resolve/")
            repo = left
            _, _, file_path = right.partition("/")
            for record in state["hub"].values():
                if record["repo"] == repo and record["path"] == file_path:
                    self._raw(record["content"])
                    return
            self._json({"error": "not found"}, status=404)
            return

        self._json({"error": "not found", "path": path}, status=404)


def _extract_name(query: str) -> str:
    """Pull ``name = 'x'`` out of a Drive query."""
    if "name = '" not in query:
        return ""
    start = query.index("name = '") + len("name = '")
    end = query.index("'", start)
    return query[start:end].replace("\\'", "'")


def _extract_parent(query: str) -> str:
    """Pull ``'FOLDER' in parents`` out of a Drive query."""
    if "in parents" not in query:
        return ""
    start = query.index("'") + 1
    end = query.index("'", start)
    return query[start:end]


class CloudDouble:
    """A local Drive + Hub server."""

    def __init__(self) -> None:
        self.state: dict[str, Any] = {"files": {}, "hub": {}}
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _CloudHandler)
        self.server.state = self.state  # type: ignore[attr-defined]
        host, port = self.server.server_address[0], self.server.server_address[1]
        self.base_url = f"http://{host}:{port}"
        self.server.base_url = self.base_url  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    # ------------------------------------------------------------------ #
    @property
    def drive_api_base(self) -> str:
        return f"{self.base_url}/drive/v3"

    @property
    def drive_upload_base(self) -> str:
        return f"{self.base_url}/upload/drive/v3"

    @property
    def hub_base_url(self) -> str:
        return self.base_url

    def files(self) -> list[dict[str, Any]]:
        return list(self.state["files"].values())

    def hub_files(self) -> list[dict[str, Any]]:
        return list(self.state["hub"].values())

    def corrupt(self, file_id: str) -> None:
        """Damage a stored file (to prove the checksum is verified)."""
        record = self.state["files"].get(file_id)
        if record is not None:
            record["content"] = b"corrupted"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


__all__ = ["CloudDouble"]
