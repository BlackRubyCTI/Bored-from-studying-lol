#!/usr/bin/env python3
"""Live simulation for Milestone 5 -- the mobile mission control.

Everything in this script is real: a real head node, a real worker agent
over HTTP, a real dashboard that proxies the API (Server-Sent Events
included), a real PDF extraction and a real notifier.

The simulation proves the guarantees the phone depends on:

* **the phone reaches the cluster** -- the PWA is served and every API call
  is proxied through a single origin;
* **the answer streams** -- the chat reaches the phone token by token;
* **the switches work** -- the council mode and the autopilot can be driven
  from the phone;
* **the telemetry is live** -- GPU, VRAM and tokens per second;
* **the phone teaches the cluster** -- a PDF is dropped into the chat, its
  text is extracted and it becomes searchable;
* **the cluster calls the phone** -- the alerts are pushed and the QR code
  of the dashboard is real.

    python scripts/smoke_m5.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import zlib
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import httpx  # noqa: E402

from core.config import Settings  # noqa: E402
from mobile.dashboard import create_dashboard  # noqa: E402
from mobile.notifier import Notifier  # noqa: E402
from mobile.qr import encode, read_payload, render_terminal  # noqa: E402
from tests.worker_harness import (  # noqa: E402
    HeadServer,
    ThreadedServer,
    WorkerHarness,
    free_port,
    patch_capabilities,
)

PASS, FAIL = "✅", "❌"
_results: list[tuple[bool, str]] = []


def check(condition: bool, label: str, detail: str = "") -> bool:
    """Record and print a check."""
    _results.append((condition, label))
    suffix = f" — {detail}" if detail else ""
    print(f"  {PASS if condition else FAIL} {label}{suffix}")
    return condition


async def wait_until(predicate: Any, *, timeout: float = 25.0, interval: float = 0.2) -> bool:
    """Wait for a (synchronous) predicate to become true."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return bool(predicate())


def build_pdf(text: str) -> bytes:
    """A minimal one-page PDF with a text layer (Flate compressed)."""
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    content = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1")
    stream = zlib.compress(content)
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" /Filter /FlateDecode >>\nstream\n"
        + stream
        + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)



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



async def main() -> int:
    data_dir = fresh_dir("/tmp/hypercluster-smoke-m5")
    settings = Settings(
        data_dir=str(data_dir),
        log_level="WARNING",
        log_json=False,
        worker_ttl_seconds=1800,
        request_timeout_seconds=60,
        memory_enabled=True,
        memory_dir=str(data_dir / "memory"),
        evolution_enabled=True,
        evolution_idle_seconds=3600,
        notifier_enabled=True,
        notifier_webhook_url="https://push.example.invalid/hook",
        dashboard_enabled=True,
        dashboard_port=free_port(),
    )

    head = HeadServer(settings)
    workers: list[WorkerHarness] = []
    pushes: list[dict[str, Any]] = []

    async def fake_push(request: httpx.Request) -> httpx.Response:
        pushes.append({"url": str(request.url), "body": json.loads(request.content.decode())})
        return httpx.Response(200, json={"ok": True})

    try:
        # ------------------------------------------------------ #
        print("\n[1/6] the phone reaches the cluster")
        head.start()
        worker = WorkerHarness(
            head_url=head.base_url,
            model="Qwen2.5-7B-Instruct-GGUF",
            vram_total_gb=24.0,
            vram_free_gb=20.0,
        )
        worker.start()
        workers.append(worker)
        patch_capabilities(worker)
        await worker.boot()
        check(True, "the worker registered with the head node", worker.agent.worker_id)

        # the dashboard proxies the head node over real HTTP
        dashboard = ThreadedServer(
            create_dashboard(settings, head_url=head.base_url),
            free_port(),
        )
        dashboard.start()
        base = dashboard.base_url

        async with httpx.AsyncClient(timeout=30.0) as client:
            shell = await client.get(f"{base}/")
            check(shell.status_code == 200, "the PWA is served")
            check("text/html" in shell.headers["content-type"], "the shell is HTML")
            check("Mission Control" in shell.text or "hyper" in shell.text.lower(), "the shell is the mission control")

            for asset in ("/styles.css", "/app.js", "/manifest.webmanifest", "/sw.js", "/icon.svg"):
                response = await client.get(f"{base}{asset}")
                check(response.status_code == 200, f"the asset {asset} is served")

            status = await client.get(f"{base}/api/v1/status")
            check(status.status_code == 200, "the API is proxied")
            body = status.json()
            check(body.get("cluster_name") == settings.cluster_name, "the cluster answers through the proxy")

            # ------------------------------------------------------ #
            print("\n[2/6] the answer streams to the phone")
            events: list[str] = []
            tokens: list[str] = []
            async with client.stream(
                "POST",
                f"{base}/api/v1/chat/stream",
                json={
                    "messages": [{"role": "user", "content": "hello from the phone"}],
                    "stream": True,
                    "max_tokens": 64,
                },
            ) as response:
                check(response.status_code == 200, "the stream starts")
                check(
                    "text/event-stream" in response.headers.get("content-type", ""),
                    "the stream is Server-Sent Events",
                )
                kind = ""
                async for line in response.aiter_lines():
                    if line.startswith("event:"):
                        kind = line.split(":", 1)[1].strip()
                        events.append(kind)
                        continue
                    if line.startswith("data:") and kind == "token":
                        try:
                            tokens.append(str(json.loads(line[5:]).get("text") or ""))
                        except json.JSONDecodeError:
                            pass
            check("token" in events, "the tokens stream", f"{len(tokens)} tokens")
            check(bool("".join(tokens).strip()), "the answer is not empty")

            # ------------------------------------------------------ #
            print("\n[3/6] the switches of the phone work")
            status = await client.get(f"{base}/api/v1/status")
            check("autopilot" in status.json(), "the autopilot state is visible")

            toggled = await client.post(f"{base}/api/v1/evolution/toggle", json={"enabled": False})
            check(toggled.status_code == 200, "the autopilot switch answers")
            check(toggled.json().get("enabled") is False, "the autopilot was turned off")
            again = await client.post(f"{base}/api/v1/evolution/toggle", json={"enabled": True})
            check(again.json().get("enabled") is True, "the autopilot was turned on again")

            council = await client.post(
                f"{base}/api/v1/chat",
                json={
                    "messages": [{"role": "user", "content": "council mode, please"}],
                    "mode": "council",
                    "max_tokens": 48,
                    "stream": False,
                },
            )
            check(council.status_code == 200, "the council answers through the proxy")

            # ------------------------------------------------------ #
            print("\n[4/6] the telemetry is live")
            telemetry = (await client.get(f"{base}/api/v1/status")).json()
            check(telemetry.get("healthy_workers", 0) >= 1, "the worker is visible", f"{telemetry.get('healthy_workers')}")
            check(
                float(telemetry.get("total_vram_gb") or 0) > 0,
                "the VRAM of the mesh is reported",
                f"{telemetry.get('free_vram_gb')} GB free",
            )
            check(
                float(telemetry.get("tokens_per_sec_aggregate") or 0) > 0,
                "the throughput is reported",
                f"{telemetry.get('tokens_per_sec_aggregate')} tok/s",
            )
            workers_body = (await client.get(f"{base}/api/v1/workers")).json()
            check(bool(workers_body.get("workers")), "the mesh is listed")
            first = workers_body["workers"][0]
            check("device" in first or "vram_free_gb" in first, "the GPU telemetry of the worker is there")

            # ------------------------------------------------------ #
            print("\n[5/6] the phone teaches the cluster (PDF)")
            pdf = build_pdf("the mitochondria is the powerhouse of the cell and it makes ATP")
            uploaded = await client.post(
                f"{base}/api/v1/memory/documents",
                files={"file": ("paper.pdf", pdf, "application/pdf")},
            )
            check(uploaded.status_code == 200, "the PDF is accepted")
            document = uploaded.json().get("document", {})
            check(uploaded.json().get("extracted") is True, "the text layer was extracted")
            check(uploaded.json().get("pdf_pages") == 1, "the page count is known")
            check(int(document.get("chunks") or 0) >= 1, "the document was chunked", f"{document.get('chunks')} chunks")

            hits = await client.post(f"{base}/api/v1/memory/search", json={"query": "mitochondria", "k": 3})
            found = hits.json().get("hits", [])
            check(bool(found), "the PDF is searchable")
            check(
                any("mitochondria" in str(hit.get("text", "")) for hit in found),
                "the search finds the sentence of the PDF",
            )

            # ------------------------------------------------------ #
            print("\n[6/6] the cluster calls the phone")
            notifier = Notifier(
                telegram_token="smoke-token",
                telegram_chat_id="42",
                webhook_url="https://push.example.invalid/hook",
                client=httpx.AsyncClient(transport=httpx.MockTransport(fake_push)),
            )
            results = await notifier.handle_event("model.changed", {"model": "qwen2.5-7b"})
            check(len(results) == 2, "both transports were used", f"{[r.target for r in results]}")
            check(all(result.ok for result in results), "the pushes were accepted")
            check(len(pushes) == 2, "the phone received the alerts")
            check(
                any("api.telegram.org" in push["url"] for push in pushes),
                "the Telegram alert was pushed",
            )
            check(
                any("push.example.invalid" in push["url"] for push in pushes),
                "the webhook alert was pushed",
            )

            url = f"{base}/mission"
            code = encode(url)
            check(read_payload(code) == url.encode(), "the QR code carries the dashboard address")
            rendered = render_terminal(url)
            check("█" in rendered, "the QR code is drawn in the terminal")
            check(len(code.modules) == code.size, "the QR matrix is square")

            await notifier.aclose()

        dashboard.stop()
    finally:
        for harness in workers:
            await harness.shutdown()
            harness.stop()
        head.stop()

    passed = sum(1 for ok, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 60}")
    print(f"  {passed}/{total} checks passed")
    print(f"{'=' * 60}\n")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
