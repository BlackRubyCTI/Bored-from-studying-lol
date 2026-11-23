"""Head node entry point.

Usage::

    python -m head_node.serve --host 0.0.0.0 --port 8080
    python -m head_node.serve --check          # validate the environment
    python -m head_node.serve --print-config   # show the resolved configuration
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from typing import Any

from core.config import Settings, get_settings, load_dotenv
from core.logging_setup import get_logger, setup_logging


def build_parser() -> argparse.ArgumentParser:
    """Command line interface of the control plane."""
    parser = argparse.ArgumentParser(
        prog="head_node.serve",
        description="Hyper-Cluster Omega head node (FastAPI control plane).",
    )
    parser.add_argument("--host", default=None, help="bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="bind port (default: 8080)")
    parser.add_argument("--reload", action="store_true", help="auto-reload on code changes (development)")
    parser.add_argument("--log-level", default=None, help="DEBUG/INFO/WARNING/ERROR")
    parser.add_argument("--log-text", action="store_true", help="human readable logs instead of JSON")
    parser.add_argument("--data-dir", default=None, help="runtime state directory")
    parser.add_argument("--workers", type=int, default=1, help="number of uvicorn workers")
    parser.add_argument("--check", action="store_true", help="validate configuration and exit")
    parser.add_argument("--print-config", action="store_true", help="print the resolved configuration and exit")
    parser.add_argument("--env-file", default=None, help="path to a .env file")
    return parser


def lan_ip() -> str:
    """Best-effort discovery of the machine's LAN address."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(0.5)
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def resolve_settings(args: argparse.Namespace) -> Settings:
    """Build the effective settings from CLI flags + environment."""
    if args.env_file:
        os.environ["HYPER_ENV_FILE"] = args.env_file
    load_dotenv(args.env_file)
    overrides: dict[str, Any] = {}
    if args.host:
        overrides["head_host"] = args.host
    if args.port:
        overrides["head_port"] = args.port
    if args.data_dir:
        overrides["data_dir"] = args.data_dir
    if args.log_level:
        overrides["log_level"] = args.log_level.upper()
    if args.log_text:
        overrides["log_json"] = False
    settings = get_settings(reload=True)
    return settings.with_overrides(**overrides) if overrides else settings


def check_environment(settings: Settings) -> int:
    """Validate that the control plane can start; returns a process exit code."""
    problems: list[str] = []
    settings.ensure_dirs()
    if not os.access(settings.data_path, os.W_OK):
        problems.append(f"data directory is not writable: {settings.data_path}")
    if settings.api_key and len(settings.api_key) < 16:
        problems.append("HYPER_API_KEY is set but shorter than 16 characters")
    if not settings.mesh_token:
        problems.append("HYPER_MESH_TOKEN is unset -- workers cannot authenticate")
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError as exc:
        problems.append(f"missing dependency: {exc}")
    report = {
        "cluster": settings.cluster_name,
        "instance": settings.instance_id,
        "host": settings.head_host,
        "port": settings.head_port,
        "data_dir": settings.data_dir,
        "api_key": "set" if settings.api_key else "unset",
        "mesh_token": "set" if settings.mesh_token else "unset",
        "admin_token": "set" if settings.admin_token else "unset",
        "edge_providers": settings.speculative_edge_providers,
        "problems": problems,
    }
    print(json.dumps(report, indent=2))
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    """Start the control plane."""
    args = build_parser().parse_args(argv)
    settings = resolve_settings(args)
    setup_logging(settings.log_level, json_mode=settings.log_json, log_file=settings.log_file or None)
    logger = get_logger("head_node.serve")

    if args.check:
        return check_environment(settings)
    if args.print_config:
        print(json.dumps(settings.redacted(), indent=2, default=str))
        return 0

    import uvicorn

    from head_node.app import create_app

    logger.info(
        "starting head node",
        extra={
            "hc_cluster": settings.cluster_name,
            "hc_bind": f"{settings.head_host}:{settings.head_port}",
            "hc_data_dir": settings.data_dir,
        },
    )
    banner = (
        f"\n  Hyper-Cluster Omega :: {settings.cluster_name}\n"
        f"  local    http://127.0.0.1:{settings.head_port}\n"
        f"  network  http://{lan_ip()}:{settings.head_port}\n"
        f"  docs     http://127.0.0.1:{settings.head_port}/docs\n"
    )
    print(banner, flush=True)

    uvicorn.run(
        create_app(settings),
        host=settings.head_host,
        port=settings.head_port,
        reload=args.reload,
        log_level=settings.log_level.lower(),
        access_log=False,
        workers=args.workers if args.workers > 1 else None,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
