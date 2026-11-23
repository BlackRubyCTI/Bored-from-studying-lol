"""The Termux command line.

The phone is a terminal as well as a browser: ``hc`` talks to the head node
over HTTP and works over SSH, over ``adb shell`` and inside Termux.

    hc status                 the cluster at a glance
    hc chat "hello"           one answer
    hc ask                    a conversation (REPL)
    hc workers                the mesh
    hc watch                  live telemetry
    hc upload paper.pdf       teach the cluster (RAG)
    hc memory "question"      what does it remember
    hc evolution              the sleep-time engine
    hc evolution run          learn right now
    hc qr                     the QR code of the dashboard
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

from core.logging_setup import get_logger

LOGGER = get_logger("mobile.cli")

#: Where the client keeps its settings.
CONFIG_PATH = Path(
    os.environ.get("HYPER_CLI_CONFIG")
    or Path(os.environ.get("HOME", "~")).expanduser() / ".hypercluster" / "client.json"
)

#: The colours (disabled when the output is not a terminal).
_COLOURS = {
    "reset": "\033[0m",
    "dim": "\033[2m",
    "bold": "\033[1m",
    "green": "\033[32m",
    "cyan": "\033[36m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "violet": "\033[35m",
}


def _use_colour() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def colour(name: str, text: str) -> str:
    """Wrap a text in an ANSI colour (when the terminal allows it)."""
    if not _use_colour():
        return text
    return f"{_COLOURS.get(name, '')}{text}{_COLOURS['reset']}"


def _symbol(value: bool) -> str:
    return colour("green", "●") if value else colour("red", "○")


class Client:
    """A tiny HTTP client for the head node."""

    def __init__(self, head_url: str = "", api_key: str = "", timeout: float = 60.0) -> None:
        config = _read_config()
        self.head_url = (head_url or config.get("head_url") or os.environ.get("HYPER_HEAD_URL") or "").rstrip("/")
        self.api_key = api_key or config.get("api_key") or os.environ.get("HYPER_API_KEY") or ""
        if not self.head_url:
            self.head_url = "http://127.0.0.1:8080"
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    # ------------------------------------------------------------------ #
    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def get(self, path: str) -> Any:
        response = self._client.get(f"{self.head_url}{path}", headers=self._headers())
        return self._unwrap(response)

    def post(self, path: str, payload: dict[str, Any] | None = None) -> Any:
        response = self._client.post(
            f"{self.head_url}{path}",
            headers=self._headers(),
            json=payload or {},
        )
        return self._unwrap(response)

    def upload(self, path: str, file_path: Path) -> Any:
        with file_path.open("rb") as handle:
            response = self._client.post(
                f"{self.head_url}{path}",
                headers={"x-api-key": self.api_key} if self.api_key else {},
                files={"file": (file_path.name, handle, "application/octet-stream")},
            )
        return self._unwrap(response)

    @staticmethod
    def _unwrap(response: httpx.Response) -> Any:
        if response.status_code >= 400:
            raise SystemExit(colour("red", f"the cluster refused the call: HTTP {response.status_code}"))
        try:
            return response.json()
        except ValueError:
            return response.text

    def stream(self, prompt: str, *, mode: str = "auto") -> str:
        """Ask a question and print the answer as it arrives."""
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "mode": mode,
            "stream": True,
            "max_tokens": 512,
        }
        answer: list[str] = []
        with self._client.stream(
            "POST",
            f"{self.head_url}/v1/chat/stream",
            headers=self._headers(),
            json=payload,
        ) as response:
            if response.status_code >= 400:
                raise SystemExit(colour("red", f"HTTP {response.status_code}"))
            kind = ""
            for line in response.iter_lines():
                if line.startswith("event:"):
                    kind = line[6:].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if not raw or raw == "[DONE]":
                    continue
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if kind == "token":
                    piece = str(data.get("text") or "")
                    answer.append(piece)
                    sys.stdout.write(piece)
                    sys.stdout.flush()
                elif kind == "error":
                    sys.stdout.write("\n")
                    raise SystemExit(colour("red", str(data.get("message") or data.get("error") or "failed")))
        sys.stdout.write("\n")
        return "".join(answer)

    def close(self) -> None:
        self._client.close()


def _read_config() -> dict[str, Any]:
    try:
        if CONFIG_PATH.is_file():
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _write_config(config: dict[str, Any]) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
    try:
        os.chmod(CONFIG_PATH, 0o600)
    except OSError:  # pragma: no cover - exotic file systems
        pass


# --------------------------------------------------------------------------- #
# The commands
# --------------------------------------------------------------------------- #
def cmd_status(client: Client, args: argparse.Namespace) -> int:
    """The cluster at a glance."""
    status = client.get("/v1/status")
    healthy = bool(status.get("healthy"))
    print(colour("bold", f"● {status.get('cluster_name', 'cluster')}"))
    print(f"  health      {_symbol(healthy)} {'ok' if healthy else 'degraded'}")
    print(
        f"  workers     {status.get('healthy_workers', 0)}/{status.get('worker_count', 0)}"
        f" ({status.get('draining_workers', 0)} draining)"
    )
    print(
        f"  vram        {status.get('free_vram_gb', 0):.1f} GB free"
        f" of {status.get('total_vram_gb', 0):.1f} GB"
    )
    print(f"  throughput  {status.get('tokens_per_sec_aggregate', 0):.1f} tok/s")
    print(f"  model       {status.get('active_model') or '—'}")
    print(f"  queue       {status.get('queued_requests', 0)} waiting, {status.get('in_flight', 0)} in flight")
    edge = status.get("edge_healthy") or []
    if edge:
        print(f"  edge        {', '.join(str(name) for name in edge)}")
    return 0


def cmd_workers(client: Client, args: argparse.Namespace) -> int:
    """The mesh."""
    payload = client.get("/v1/workers")
    workers = payload.get("workers") or []
    if not workers:
        print(colour("dim", "no worker is registered"))
        return 0
    print(colour("bold", f"{'worker':<22}{'model':<28}{'vram':>10}{'tok/s':>9}  state"))
    for worker in workers:
        state = "draining" if worker.get("draining") else ("healthy" if worker.get("healthy") else "stale")
        print(
            f"{str(worker.get('worker_id'))[:21]:<22}"
            f"{str(worker.get('model_loaded'))[:27]:<28}"
            f"{float(worker.get('vram_free_gb') or 0):>8.1f}G"
            f"{float(worker.get('tokens_per_sec') or 0):>9.1f}  "
            f"{colour('yellow', state) if state == 'draining' else state}"
        )
    return 0


def cmd_chat(client: Client, args: argparse.Namespace) -> int:
    """One question, one answer."""
    prompt = " ".join(args.message).strip()
    if not prompt:
        raise SystemExit(colour("red", "say something"))
    client.stream(prompt, mode=args.mode)
    return 0


def cmd_ask(client: Client, args: argparse.Namespace) -> int:
    """A conversation."""
    print(colour("dim", "ctrl-d or /quit to leave"))
    history: list[dict[str, str]] = []
    while True:
        try:
            prompt = input(colour("cyan", "you › ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not prompt:
            continue
        if prompt in ("/quit", "/exit"):
            return 0
        history.append({"role": "user", "content": prompt})
        answer = client.stream(prompt, mode=args.mode)
        history.append({"role": "assistant", "content": answer})
    return 0


def cmd_watch(client: Client, args: argparse.Namespace) -> int:
    """Live telemetry."""
    try:
        while True:
            status = client.get("/v1/status")
            workers = status.get("worker_count", 0)
            healthy = status.get("healthy_workers", 0)
            sys.stdout.write(
                "\r"
                + f" {healthy}/{workers} workers"
                + f"  {status.get('free_vram_gb', 0):.0f}GB vram"
                + f"  {status.get('tokens_per_sec_aggregate', 0):.0f} tok/s"
                + f"  {status.get('queued_requests', 0)} queued"
                + "   "
            )
            sys.stdout.flush()
            import time

            time.sleep(max(0.5, float(args.interval)))
    except KeyboardInterrupt:
        print()
        return 0


def cmd_upload(client: Client, args: argparse.Namespace) -> int:
    """Teach the cluster (RAG)."""
    path = Path(args.path).expanduser()
    if not path.is_file():
        raise SystemExit(colour("red", f"no such file: {path}"))
    result = client.upload("/v1/memory/documents", path)
    document = result.get("document") or {}
    print(colour("green", "✓ indexed ") + f"{document.get('name')} ({document.get('chunks')} chunks)")
    if result.get("pdf_pages"):
        print(colour("dim", f"  {result['pdf_pages']} pages extracted"))
    return 0


def cmd_memory(client: Client, args: argparse.Namespace) -> int:
    """What does the cluster remember?"""
    if not args.query:
        status = client.get("/v1/memory")
        print(colour("bold", "memory"))
        print(f"  chunks     {status.get('chunks', 0)}")
        print(f"  documents  {status.get('documents', 0)}")
        print(f"  by source  {status.get('by_source', {})}")
        return 0
    result = client.post("/v1/memory/search", {"query": args.query, "k": args.k})
    hits = result.get("hits") or []
    if not hits:
        print(colour("dim", "nothing found"))
        return 0
    for hit in hits:
        print(colour("cyan", f"● {hit.get('source')} ") + f"(score {hit.get('score')})")
        print("  " + str(hit.get("text", "")).replace("\n", "\n  ")[:600])
    return 0


def cmd_evolution(client: Client, args: argparse.Namespace) -> int:
    """The sleep-time engine."""
    if args.action == "run":
        result = client.post("/v1/evolution/run", {})
        print(colour("green", "✓ cycle finished"))
        print(f"  seeds      {result.get('seeds', 0)}")
        print(f"  verified   {result.get('verified', 0)}")
        training = result.get("training") or {}
        print(f"  loss       {training.get('loss_start', 0):.5f} → {training.get('loss_end', 0):.5f}")
        return 0
    status = client.get("/v1/evolution/status")
    print(colour("bold", "sleep-time engine"))
    print(f"  state      {status.get('state')} ({status.get('phase', 'idle')})")
    print(f"  enabled    {status.get('enabled')}")
    print(f"  idle for   {status.get('idle_for', 0):.0f}s of {status.get('idle_seconds', 0):.0f}s")
    print(f"  preempt    {status.get('preempt_latency_ms', 0):.0f}ms of {status.get('preempt_budget_ms', 2000):.0f}ms")
    print(f"  cycles     {status.get('cycles', 0)}")
    return 0


def cmd_qr(client: Client, args: argparse.Namespace) -> int:
    """The QR code of the dashboard."""
    from mobile.qr import render_terminal

    url = args.url or client.head_url
    print(render_terminal(url))
    return 0


def cmd_config(client: Client, args: argparse.Namespace) -> int:
    """Remember the head node."""
    config = _read_config()
    config["head_url"] = args.url or client.head_url
    if args.api_key is not None:
        config["api_key"] = args.api_key
    _write_config(config)
    print(colour("green", "✓ saved ") + str(CONFIG_PATH))
    return 0


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """The command line."""
    parser = argparse.ArgumentParser(
        prog="hc",
        description="Hyper-Cluster Omega — the phone command line.",
    )
    parser.add_argument("--head-url", default="", help="the head node (default: the saved one)")
    parser.add_argument("--api-key", default="", help="the API key of the cluster")
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="the cluster at a glance")
    status.set_defaults(func=cmd_status)

    workers = sub.add_parser("workers", help="the mesh")
    workers.set_defaults(func=cmd_workers)

    chat = sub.add_parser("chat", help="ask one question")
    chat.add_argument("message", nargs="+")
    chat.add_argument("--mode", default="auto", choices=("auto", "fast", "deep", "council"))
    chat.set_defaults(func=cmd_chat)

    ask = sub.add_parser("ask", help="a conversation")
    ask.add_argument("--mode", default="auto", choices=("auto", "fast", "deep", "council"))
    ask.set_defaults(func=cmd_ask)

    watch = sub.add_parser("watch", help="live telemetry")
    watch.add_argument("--interval", type=float, default=2.0)
    watch.set_defaults(func=cmd_watch)

    upload = sub.add_parser("upload", help="teach the cluster (RAG)")
    upload.add_argument("path")
    upload.set_defaults(func=cmd_upload)

    memory = sub.add_parser("memory", help="what does the cluster remember")
    memory.add_argument("query", nargs="*")
    memory.add_argument("-k", type=int, default=5)
    memory.set_defaults(func=cmd_memory)

    evolution = sub.add_parser("evolution", help="the sleep-time engine")
    evolution.add_argument("action", nargs="?", default="status", choices=("status", "run"))
    evolution.set_defaults(func=cmd_evolution)

    qr = sub.add_parser("qr", help="the QR code of the dashboard")
    qr.add_argument("url", nargs="?", default="")
    qr.set_defaults(func=cmd_qr)

    config = sub.add_parser("config", help="remember the head node")
    config.add_argument("url", nargs="?", default="")
    config.add_argument("--api-key", default=None)
    config.set_defaults(func=cmd_config)

    return parser


def main(argv: list[str] | None = None) -> int:
    """The entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    client = Client(head_url=args.head_url, api_key=args.api_key)
    try:
        return int(args.func(client, args) or 0)
    except SystemExit as exc:
        if isinstance(exc.code, str):
            print(exc.code, file=sys.stderr)
            return 1
        raise
    except KeyboardInterrupt:
        print()
        return 0
    except httpx.HTTPError as exc:
        print(colour("red", f"the cluster is unreachable: {exc}"), file=sys.stderr)
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
