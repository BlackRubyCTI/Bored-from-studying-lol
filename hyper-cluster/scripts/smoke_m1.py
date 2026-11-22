#!/usr/bin/env python3
"""Live smoke test for Milestone 1.

Boots the real control plane (uvicorn) plus a throw-away worker, then exercises
the full path: registration -> streaming chat -> non-streaming chat -> status ->
hold-and-requeue -> shutdown.  Exits non-zero on the first failed expectation.

    python scripts/smoke_m1.py
"""

from __future__ import annotations

import asyncio
import json
import socket
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import httpx  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402

from core.config import Settings  # noqa: E402
from head_node.app import create_app  # noqa: E402

PASS, FAIL = "✅", "❌"
_results: list[tuple[bool, str]] = []


def check(condition: bool, label: str, detail: str = "") -> bool:
    """Record and print a check."""
    _results.append((condition, label))
    suffix = f" — {detail}" if detail else ""
    print(f"  {PASS if condition else FAIL} {label}{suffix}")
    return condition


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


# --------------------------------------------------------------------------- #
# Throw-away worker
# --------------------------------------------------------------------------- #
def build_worker(worker_id: str, model: str) -> FastAPI:
    app = FastAPI(title=f"smoke-worker-{worker_id}")

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "worker_id": worker_id, "model_loaded": model, "protocol_version": "1.0"}

    @app.post("/v1/chat")
    async def chat(payload: dict) -> StreamingResponse:
        async def _stream():
            prompt = " ".join(str(m.get("content", "")) for m in payload.get("messages", []))[:120]
            for index, word in enumerate(f"[{worker_id}] {prompt} done".split(" ")):
                await asyncio.sleep(0.002)
                frame = {
                    "request_id": payload.get("request_id", ""),
                    "text": word + " ",
                    "index": index,
                    "source": "cluster",
                    "draft": False,
                }
                yield f"data: {json.dumps(frame)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_stream(), media_type="text/event-stream")

    @app.post("/v1/chat/sync")
    async def chat_sync(payload: dict) -> dict:
        prompt = " ".join(str(m.get("content", "")) for m in payload.get("messages", []))[:120]
        text = f"[{worker_id}] {prompt} done"
        return {
            "request_id": payload.get("request_id", ""),
            "text": text,
            "model": model,
            "worker_id": worker_id,
            "tier": "cluster",
            "source": "cluster",
            "tokens": len(text.split()),
            "ttft_ms": 12.0,
            "total_ms": 40.0,
            "tokens_per_sec": 120.0,
            "finish_reason": "stop",
        }

    @app.post("/v1/model/load")
    async def load_model(payload: dict) -> dict:
        return {
            "worker_id": worker_id,
            "model": payload.get("model", model),
            "loaded": True,
            "load_ms": 5.0,
            "vram_used_gb": 6.0,
            "error": None,
        }

    @app.post("/v1/admin/drain")
    async def drain(payload: dict) -> dict:
        return {"draining": True, "worker_id": worker_id}

    return app


def run_server(app: FastAPI, port: int) -> uvicorn.Server:
    """Start a uvicorn server in a background thread and wait until it listens."""
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", access_log=False)
    server = uvicorn.Server(config)

    def _run() -> None:
        asyncio.run(server.serve())

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return server
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(f"server on port {port} never came up")


# --------------------------------------------------------------------------- #

def fresh_dir(path: str) -> Path:
    """A clean data directory -- every run of a smoke starts from scratch.

    Without it the state of the previous run leaks into the next one (the
    memory grows, the ranking drifts) and the smoke stops being repeatable.
    """
    import shutil

    target = Path(path)
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)
    return target



def main() -> int:
    """Run the smoke test."""
    head_port = free_port()
    worker_port = free_port()
    data_dir = fresh_dir("/tmp/hyper-cluster-smoke")
    data_dir.mkdir(parents=True, exist_ok=True)

    print("\n⚡ Hyper-Cluster Omega — Milestone 1 smoke test\n")

    worker_server = run_server(build_worker("wk-smoke", "Qwen2.5-7B-Instruct-GGUF"), worker_port)
    settings = Settings(
        data_dir=str(data_dir),
        head_host="127.0.0.1",
        head_port=head_port,
        log_level="WARNING",
        log_json=False,
        api_key="",
        mesh_token="",
        admin_token="",
        worker_ttl_seconds=300,
    )
    head_server = run_server(create_app(settings), head_port)
    base = f"http://127.0.0.1:{head_port}"

    try:
        with httpx.Client(timeout=30.0) as client:
            print("• control plane")
            root = client.get(f"{base}/")
            check(root.status_code == 200, "GET / returns the service banner")
            health = client.get(f"{base}/health")
            check(health.status_code == 200, "GET /health answers", health.json().get("status", "?"))

            print("• worker mesh")
            registration = {
                "worker_id": "wk-smoke",
                "host": "127.0.0.1",
                "port": worker_port,
                "model_loaded": "Qwen2.5-7B-Instruct-GGUF",
                "model_family": "qwen2",
                "model_params_b": 7.0,
                "capabilities": {
                    "vram_total_gb": 24.0,
                    "vram_free_gb": 20.0,
                    "compute_tflops": 60.0,
                    "provider": "local",
                    "region": "smoke",
                    "device": "cuda",
                    "supported_families": ["qwen2", "llama"],
                    "max_concurrency": 2,
                    "mesh_latency_ms": 4.0,
                },
            }
            ack = client.post(f"{base}/v1/workers/register", json=registration)
            check(ack.status_code == 200, "POST /v1/workers/register accepted", ack.json().get("message", ""))
            check(ack.json().get("ttl_seconds", 0) > 0, "registration returns a TTL")
            inventory = client.get(f"{base}/v1/workers").json()
            check(inventory["totals"]["workers"] == 1, "worker appears in the inventory")
            check(inventory["totals"]["total_vram_gb"] == 24.0, "VRAM is aggregated")

            print("• inference")
            completion = client.post(
                f"{base}/v1/chat",
                json={
                    "messages": [{"role": "user", "content": "hello cluster"}],
                    "mode": "deep",
                    "max_tokens": 32,
                    "stream": False,
                },
            ).json()
            check(completion.get("error") is None, "POST /v1/chat completed", completion.get("error") or "")
            check("wk-smoke" in completion.get("text", ""), "answer came from the registered worker")
            check(completion.get("ttft_ms", 0) > 0, "time-to-first-token was measured")

            print("• streaming")
            events: list[str] = []
            token_text = ""
            with client.stream(
                "POST",
                f"{base}/v1/chat/stream",
                json={"messages": [{"role": "user", "content": "stream me"}], "mode": "deep", "max_tokens": 32},
            ) as response:
                check(response.headers["content-type"].startswith("text/event-stream"), "SSE content-type")
                current = ""
                for line in response.iter_lines():
                    if line.startswith("event:"):
                        current = line[6:].strip()
                    elif line.startswith("data:"):
                        events.append(current)
                        if current == "token":
                            token_text += json.loads(line[5:].strip()).get("text", "")
            check("start" in events, "stream emitted a start event")
            check("token" in events, "stream emitted token events")
            check(events[-1] == "done", "stream ended with done", events[-1] if events else "none")
            check(len(token_text) > 0, "streamed text is non-empty")

            print("• telemetry")
            status = client.get(f"{base}/v1/status").json()
            check(status["worker_count"] == 1, "status reports the worker")
            check("hold_queue" in status and "edge" in status, "status includes queue + edge telemetry")
            check(status["cluster_name"] == settings.cluster_name, "status carries the cluster name")

            print("• hold & requeue")
            drained = client.post(f"{base}/v1/workers/wk-smoke/drain")
            check(drained.status_code == 200, "worker can be drained for retirement")
            deferred = client.post(
                f"{base}/v1/chat",
                json={
                    "messages": [{"role": "user", "content": "queue me"}],
                    "mode": "deep",
                    "max_tokens": 16,
                    "stream": False,
                    "deadline_seconds": 0.5,
                },
            )
            check(deferred.status_code == 202, "saturated request answers 202 (not 5xx)", str(deferred.status_code))
            check(deferred.json().get("error") == "deferred", "the request is deferred, not failed")
            queue = client.get(f"{base}/v1/queue").json()
            check(queue["stats"]["deferred"] >= 1, "the deferred request is parked in the hold queue")

            print("• request ledger")
            request_id = completion["request_id"]
            ledger = client.get(f"{base}/v1/requests/{request_id}").json()
            check(ledger["status"]["state"] == "completed", "ledger records the completion")
            check(bool(ledger.get("result", {}).get("text")), "ledger stores the answer")
    finally:
        head_server.should_exit = True
        worker_server.should_exit = True
        time.sleep(0.6)

    passed = sum(1 for ok, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 58}\n  {passed}/{total} checks passed\n{'=' * 58}\n")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
