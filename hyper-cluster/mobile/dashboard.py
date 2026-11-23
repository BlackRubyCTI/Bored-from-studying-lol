"""The Mobile Mission Control.

The dashboard is a tiny FastAPI app that

* serves the PWA (the shell, the stylesheet, the script, the manifest and
  the service worker), and
* proxies the head node's API under ``/api`` -- including the streaming
  endpoints, so the phone talks to exactly one origin.

It can sit in the same process as the control plane (``head_app=``) or next
to it (``head_url=``).  The proxy is deliberately thin: it forwards the
method, the body, the query and the authentication headers, and streams the
answer back.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse

from core.config import Settings
from core.logging_setup import get_logger

LOGGER = get_logger("mobile.dashboard")

#: Where the PWA lives.
STATIC_DIR = Path(__file__).resolve().parent / "static"

#: The files that are served (and cached by the service worker).
SHELL_FILES: dict[str, str] = {
    "/": "index.html",
    "/index.html": "index.html",
    "/styles.css": "styles.css",
    "/app.js": "app.js",
    "/manifest.webmanifest": "manifest.webmanifest",
    "/sw.js": "sw.js",
    "/icon.svg": "icon.svg",
}

#: The headers that are forwarded to the head node.
FORWARDED_HEADERS = (
    "x-api-key",
    "authorization",
    "x-mesh-token",
    "x-admin-token",
    "content-type",
    "accept",
)

#: The methods the proxy forwards.
PROXY_METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE")

#: The timeout of a proxied call.  A stream never times out on the read
#: side: the chat may stay silent while the cluster holds the request.
STREAM_TIMEOUT = httpx.Timeout(connect=5.0, read=None, write=30.0, pool=30.0)


def _is_shell(path: str) -> bool:
    return path in SHELL_FILES


def create_dashboard(
    settings: Settings | None = None,
    *,
    head_url: str = "",
    head_app: Any | None = None,
    client: httpx.AsyncClient | None = None,
) -> FastAPI:
    """Build the dashboard app.

    ``head_url`` points at a head node that runs somewhere else;
    ``head_app`` mounts an in-process head node (the fast path).
    """
    resolved = settings if settings is not None else Settings.from_env()
    target = (head_url or resolved.dashboard_head_url or "http://127.0.0.1:8080").rstrip("/")
    owns_client = client is None
    http = client if client is not None else httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0))


    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        if owns_client:
            await http.aclose()

    app = FastAPI(
        title="Hyper-Cluster Omega — Mission Control",
        description="The touch-first dashboard of the cluster.",
        version="1.0.0",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    app.state.head_url = target
    app.state.head_app = head_app
    app.state.http = http
    app.state.settings = resolved

    # ------------------------------------------------------------------ #
    async def require_client_auth(
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> str:
        """The phone presents the API key of the cluster (if it has one)."""
        expected = resolved.api_key
        if not expected:
            return "anonymous"
        presented = x_api_key
        if not presented and authorization and authorization.lower().startswith("bearer "):
            presented = authorization[7:].strip()
        if not presented or presented != expected:
            raise HTTPException(status_code=401, detail="invalid or missing API key")
        return "client"

    # ------------------------------------------------------------------ #
    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Is the cluster reachable?"""
        try:
            response = await http.get(f"{target}/health", timeout=5.0)
            return {"dashboard": "ok", "head": target, "head_status": response.status_code}
        except httpx.HTTPError as exc:
            return {"dashboard": "ok", "head": target, "head_status": 0, "error": str(exc)}

    @app.get("/api/config")
    async def config(_: str = Depends(require_client_auth)) -> dict[str, Any]:
        """What the phone needs to know about this cluster."""
        return {
            "head": target,
            "requires_key": bool(resolved.api_key),
            "cluster_name": resolved.cluster_name,
            "features": {
                "memory": bool(resolved.memory_enabled),
                "evolution": bool(resolved.evolution_enabled),
                "council": True,
                "voice": True,
                "rag": True,
            },
        }

    # ------------------------------------------------------------------ #
    @app.api_route("/api/{path:path}", methods=list(PROXY_METHODS))
    async def proxy(
        request: Request,
        path: str,
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
        x_mesh_token: str | None = Header(default=None),
        x_admin_token: str | None = Header(default=None),
        _: str = Depends(require_client_auth),
    ) -> Any:
        """Forward a call to the head node (streaming included)."""
        url = f"{target}/{path}"
        if request.url.query:
            url = f"{url}?{request.url.query}"
        headers = {
            name: value
            for name, value in request.headers.items()
            if name.lower() in FORWARDED_HEADERS
        }
        body = await request.body()
        # The answer is *streamed*: the upstream response is opened in
        # streaming mode, otherwise httpx would read the whole body before
        # the first byte reaches the phone (and a long stream would time
        # out instead of trickling).
        upstream_request = http.build_request(
            request.method,
            url,
            headers=headers,
            content=body or None,
        )
        upstream_request.extensions = {
            **upstream_request.extensions,
            "timeout": STREAM_TIMEOUT.as_dict(),
        }
        upstream = await http.send(upstream_request, stream=True)
        content_type = upstream.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            return StreamingResponse(
                _stream(upstream),
                media_type="text/event-stream",
                headers={
                    "cache-control": "no-cache",
                    "connection": "keep-alive",
                    "x-accel-buffering": "no",
                },
            )
        try:
            payload = await upstream.aread()
        finally:
            await upstream.aclose()
        return Response(
            content=payload,
            status_code=upstream.status_code,
            media_type=content_type or "application/json",
            headers={
                key: value
                for key, value in upstream.headers.items()
                if key.lower() in ("cache-control", "x-request-id")
            },
        )

    # ------------------------------------------------------------------ #
    # The catch-all comes last: it must not shadow /api/...
    @app.get("/{path:path}")
    async def shell(path: str) -> Any:
        """The PWA shell."""
        if _is_shell("/" + path) or _is_shell(path):
            name = SHELL_FILES.get("/" + path) or SHELL_FILES.get(path)
            file_path = STATIC_DIR / name
            if file_path.is_file():
                media = "application/manifest+json" if name.endswith("webmanifest") else None
                return FileResponse(
                    file_path,
                    media_type=media,
                    headers={"cache-control": "no-cache"},
                )
        raise HTTPException(status_code=404, detail="not found")

    return app


async def _stream(upstream: httpx.Response) -> AsyncIterator[bytes]:
    """Relay an upstream response body (the SSE stream)."""
    try:
        async for chunk in upstream.aiter_bytes():
            yield chunk
    finally:
        await upstream.aclose()


def build_parser() -> argparse.ArgumentParser:
    """The command line of the dashboard."""
    parser = argparse.ArgumentParser(
        prog="mobile.dashboard",
        description="Hyper-Cluster Omega -- the mobile Mission Control.",
    )
    parser.add_argument("--host", default=None, help="the address to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="the port to bind (default: 8082)")
    parser.add_argument("--head-url", default="", help="the head node to proxy to")
    parser.add_argument("--env-file", default=None, help="the .env file to read")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the dashboard."""
    import uvicorn

    from core.config import load_dotenv

    args = build_parser().parse_args(argv)
    if args.env_file:
        os.environ["HYPER_ENV_FILE"] = args.env_file
    load_dotenv(args.env_file)
    resolved = Settings.from_env()
    host = args.host or resolved.dashboard_host
    port = args.port or resolved.dashboard_port
    app = create_dashboard(resolved, head_url=args.head_url or resolved.dashboard_head_url)
    LOGGER.info(
        "mission control starting",
        extra={"hc_host": host, "hc_port": port, "hc_head": resolved.dashboard_head_url},
    )
    uvicorn.run(app, host=host, port=port, log_level=resolved.log_level.lower())
    return 0


def serve(
    settings: Settings | None = None,
    *,
    host: str = "0.0.0.0",
    port: int = 8082,
    head_url: str = "",
) -> None:
    """Run the dashboard with uvicorn."""
    import uvicorn

    resolved = settings if settings is not None else Settings.from_env()
    app = create_dashboard(resolved, head_url=head_url or resolved.dashboard_head_url)
    LOGGER.info("mission control starting", extra={"hc_host": host, "hc_port": port})
    uvicorn.run(app, host=host, port=port, log_level=resolved.log_level.lower())


__all__ = ["SHELL_FILES", "STATIC_DIR", "create_dashboard", "main", "serve"]


if __name__ == "__main__":
    raise SystemExit(main())
