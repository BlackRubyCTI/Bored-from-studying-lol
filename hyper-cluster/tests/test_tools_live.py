"""Live tests for the tool layer: sandbox isolation, web tools, file analysis
and the end-to-end agentic tool loop through the head node."""

from __future__ import annotations

import asyncio
import json
import os
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest

from core.config import Settings
from tests.worker_harness import HeadServer, WorkerHarness, patch_capabilities
from tools.sandbox import (
    TOOL_SCHEMAS,
    FileAnalyzer,
    PythonSandbox,
    ToolExecutor,
    WebTools,
    build_tool_prompt,
    extract_final_answer,
    parse_tool_call,
)


@pytest.fixture
def sandbox() -> PythonSandbox:
    return PythonSandbox(timeout=15, memory_limit_mb=256)


# --------------------------------------------------------------------------- #
# Sandbox
# --------------------------------------------------------------------------- #
def test_sandbox_executes_python(sandbox: PythonSandbox) -> None:
    result = sandbox.run("print(sum(range(10)))\nprint('side effect')")
    assert result.ok is True
    assert "45" in result.stdout
    assert "side effect" in result.stdout
    assert result.error == ""
    assert result.duration_ms >= 0


def test_sandbox_captures_stderr_and_failure(sandbox: PythonSandbox) -> None:
    result = sandbox.run("import sys; sys.stderr.write('boom'); raise ValueError('nope')")
    assert result.ok is False
    assert "boom" in result.stderr
    assert "nope" in result.error


def test_sandbox_blocks_network(sandbox: PythonSandbox) -> None:
    result = sandbox.run(
        "import socket\n"
        "s = socket.socket()\n"
        "s.settimeout(5)\n"
        "s.connect(('1.1.1.1', 80))\n"
    )
    assert result.ok is False
    # The denial may come from the in-process guard ("network ... blocked") or,
    # under the M6 kernel jail, from seccomp/the empty network namespace
    # ("Operation not permitted").  Either is a pass: what must never happen is
    # a successful connect.
    message = (result.error + result.stderr).lower()
    assert "network" in message or "blocked" in message or "not permitted" in message


def test_sandbox_blocks_dns(sandbox: PythonSandbox) -> None:
    result = sandbox.run("import socket; socket.getaddrinfo('example.com', 80)")
    assert result.ok is False


def test_sandbox_blocks_process_spawning(sandbox: PythonSandbox) -> None:
    result = sandbox.run("import os; os.system('echo pwned')")
    assert result.ok is False
    assert "pwned" not in result.stdout


def test_sandbox_blocks_subprocess(sandbox: PythonSandbox) -> None:
    result = sandbox.run("import subprocess; subprocess.run(['/bin/echo','pwned'])")
    assert result.ok is False


def test_sandbox_enforces_the_cpu_limit(sandbox: PythonSandbox) -> None:
    result = sandbox.run("while True:\n    pass")
    assert result.ok is False
    assert result.timed_out is True


def test_sandbox_enforces_the_memory_limit(sandbox: PythonSandbox) -> None:
    result = sandbox.run("x = bytearray(1024 * 1024 * 1024)")
    assert result.ok is False


def test_sandbox_runs_in_a_separate_process(sandbox: PythonSandbox) -> None:
    """The sandbox must not share the interpreter's memory."""
    result = sandbox.run("import os; print(os.getpid())")
    assert result.ok is True
    child_pid = int(result.stdout.strip())
    assert child_pid != os.getpid()


def test_sandbox_state_does_not_leak_between_runs(sandbox: PythonSandbox) -> None:
    sandbox.run("leaked = 12345")
    result = sandbox.run("print('leaked' in dir())")
    assert result.ok is True
    assert "False" in result.stdout


def test_sandbox_writes_only_inside_its_workspace(sandbox: PythonSandbox) -> None:
    result = sandbox.run("open('/etc/hypercluster-should-not-exist','w').write('x')")
    assert result.ok is False


def test_sandbox_cannot_read_host_secrets(sandbox: PythonSandbox, tmp_path: Path) -> None:
    """Regression: the sandbox must not be able to read the head node's .env."""
    secret = tmp_path / "credentials.env"
    secret.write_text("HYPER_API_KEY=super-secret\n", encoding="utf-8")
    result = sandbox.run(f"print(open({str(secret)!r}).read())")
    assert result.ok is False
    assert "permission" in (result.stderr + result.error).lower()
    assert "super-secret" not in result.stdout


def test_sandbox_cannot_write_outside_its_workspace(sandbox: PythonSandbox, tmp_path: Path) -> None:
    target = tmp_path / "sandbox-escape.txt"
    result = sandbox.run(f"open({str(target)!r}, 'w').write('x')")
    assert result.ok is False
    assert not target.exists()


def test_sandbox_can_write_inside_its_workspace(sandbox: PythonSandbox) -> None:
    result = sandbox.run("open('notes.txt','w').write('hello'); print(open('notes.txt').read())")
    assert result.ok is True
    assert "hello" in result.stdout


def test_sandbox_can_still_import_the_standard_library(sandbox: PythonSandbox) -> None:
    result = sandbox.run("import json, os, re, collections; print(sorted(json.loads('{\"b\":1}')))")
    assert result.ok is True
    assert "b" in result.stdout


def test_sandbox_captures_the_last_expression(sandbox: PythonSandbox) -> None:
    result = sandbox.run("2 ** 10")
    assert result.ok is True
    assert result.value == "1024"  # the repr, exactly what the model sees


def test_sandbox_rejects_a_timeout_that_is_too_short(sandbox: PythonSandbox) -> None:
    result = sandbox.run("import time; time.sleep(5)", timeout=0.5)
    assert result.ok is False
    assert result.timed_out is True


# --------------------------------------------------------------------------- #
# File analysis
# --------------------------------------------------------------------------- #
def test_file_analyzer_reads_text(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("hyper cluster omega\n" * 100, encoding="utf-8")
    analyzer = FileAnalyzer(allowed_roots=[str(tmp_path)])
    result = analyzer.analyze(str(target))
    assert result.ok is True
    assert "hyper cluster omega" in result.output
    assert result.data["size_bytes"] > 0


def test_file_analyzer_reports_entropy(tmp_path: Path) -> None:
    target = tmp_path / "blob.bin"
    target.write_bytes(bytes(range(256)) * 8)
    analyzer = FileAnalyzer(allowed_roots=[str(tmp_path)])
    result = analyzer.analyze(str(target))
    assert result.ok is True
    assert "entropy" in result.data


def test_file_analyzer_refuses_paths_outside_its_roots(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-secret.txt"
    outside.write_text("top secret", encoding="utf-8")
    try:
        analyzer = FileAnalyzer(allowed_roots=[str(tmp_path / "allowed")])
        result = analyzer.analyze(str(outside))
        assert result.ok is False
        assert "not allowed" in result.error.lower() or "outside" in result.error.lower()
    finally:
        outside.unlink(missing_ok=True)


def test_file_analyzer_handles_a_missing_file(tmp_path: Path) -> None:
    analyzer = FileAnalyzer(allowed_roots=[str(tmp_path)])
    result = analyzer.analyze(str(tmp_path / "missing.txt"))
    assert result.ok is False


# --------------------------------------------------------------------------- #
# Tool protocol
# --------------------------------------------------------------------------- #
def test_tool_schemas_are_well_formed() -> None:
    assert TOOL_SCHEMAS
    for schema in TOOL_SCHEMAS:
        assert schema["type"] == "function"
        assert schema["function"]["name"]
        assert schema["function"]["description"]
        assert schema["function"]["parameters"]["type"] == "object"


def test_parse_tool_call_handles_plain_and_fenced_json() -> None:
    assert parse_tool_call('{"tool": "python", "args": {"code": "print(1)"}}') == {
        "tool": "python",
        "args": {"code": "print(1)"},
    }
    fenced = '```json\n{"tool": "web_search", "args": {"query": "omega"}}\n```'
    assert parse_tool_call(fenced)["tool"] == "web_search"
    assert parse_tool_call('{"final": "done"}') == {"final": "done"}
    assert parse_tool_call("not json at all") is None


def test_parse_tool_call_accepts_string_arguments() -> None:
    call = parse_tool_call('{"tool": "python", "args": "{\\"code\\": \\"print(1)\\"}"}')
    assert call is not None
    assert call["args"]["code"] == "print(1)"


def test_extract_final_answer() -> None:
    assert extract_final_answer('{"final": "42"}') == "42"
    assert extract_final_answer("plain text") == "plain text"


def test_build_tool_prompt_lists_the_tools() -> None:
    prompt = build_tool_prompt(TOOL_SCHEMAS, "what is 2+2?")
    assert "web_search" in prompt
    assert "what is 2+2?" in prompt


# --------------------------------------------------------------------------- #
# Executor
# --------------------------------------------------------------------------- #
def test_executor_dispatches_python() -> None:
    executor = ToolExecutor()
    result = executor.execute("python", {"code": "print(6*7)"})
    assert result.ok is True
    assert "42" in result.output


def test_executor_rejects_unknown_tools() -> None:
    executor = ToolExecutor()
    result = executor.execute("rm_rf", {"path": "/"})
    assert result.ok is False
    assert "unknown tool" in result.error


def test_executor_rejects_empty_code() -> None:
    executor = ToolExecutor()
    result = executor.execute("python", {"code": "  "})
    assert result.ok is False


def test_executor_only_advertises_enabled_tools() -> None:
    executor = ToolExecutor(enabled_tools=["python"])
    assert [s["function"]["name"] for s in executor.list_tools()] == ["python"]
    assert executor.execute("web_search", {"query": "x"}).ok is False


# --------------------------------------------------------------------------- #
# Web tools (live HTTP against a real server)
# --------------------------------------------------------------------------- #
class _Handler(BaseHTTPRequestHandler):
    """A tiny but genuine HTTP origin used to exercise the web tools."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *_args: Any) -> None:  # keep the test output clean
        return

    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        path = self.path
        if path.startswith("/page"):
            html = (
                "<html><head><title>Omega</title>"
                "<style>body{color:red}</style></head>"
                "<body><h1>Hyper Cluster Omega</h1><p>Distributed control plane.</p>"
                "<script>console.log('nope')</script></body></html>"
            )
            self._send(html.encode("utf-8"), "text/html; charset=utf-8")
        elif path.startswith("/json"):
            self._send(b'{"answer": 42}', "application/json")
        elif path.startswith("/search"):
            html = (
                '<html><body>'
                '<a class="result__a" href="https://example.org/omega">Omega</a>'
                '<div class="result__snippet">The distributed control plane.</div>'
                '<a class="result__a" href="https://example.org/omega2">Omega two</a>'
                '<div class="result__snippet">Second result.</div>'
                "</body></html>"
            )
            self._send(html.encode("utf-8"), "text/html; charset=utf-8")
        elif path.startswith("/wiki"):
            payload = {
                "query": {
                    "search": [
                        {"title": "Hyper cluster", "snippet": "A <b>cluster</b> of workers."},
                    ]
                }
            }
            self._send(json.dumps(payload).encode("utf-8"), "application/json")
        elif path.startswith("/boom"):
            self._send(b"server error", "text/plain", status=500)
        else:
            self._send(b"not found", "text/plain", status=404)


@pytest.fixture(scope="module")
def origin() -> Iterator[str]:
    """Run a real HTTP origin on the loopback interface."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[0], server.server_address[1]
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def web(origin: str) -> WebTools:
    return WebTools(
        timeout=15,
        search_endpoint=origin + "/search?q={query}",
        wiki_endpoint=origin + "/wiki?srsearch={query}&srlimit={limit}",
    )


async def test_web_fetch_reads_a_real_page(web: WebTools, origin: str) -> None:
    result = await asyncio.to_thread(web.fetch, f"{origin}/page", max_chars=4000)
    assert result.ok is True, result.error
    assert "Hyper Cluster Omega" in result.output
    # The HTML must be reduced to readable text (no tags, no scripts).
    assert "<h1>" not in result.output
    assert "console.log" not in result.output
    assert result.data["status"] == 200


async def test_web_fetch_returns_json_verbatim(web: WebTools, origin: str) -> None:
    result = await asyncio.to_thread(web.fetch, f"{origin}/json")
    assert result.ok is True
    assert '"answer": 42' in result.output


async def test_web_fetch_reports_an_http_error(web: WebTools, origin: str) -> None:
    result = await asyncio.to_thread(web.fetch, f"{origin}/boom")
    assert result.ok is False
    assert result.data["status"] == 500


async def test_web_fetch_rejects_a_bad_scheme(web: WebTools) -> None:
    result = await asyncio.to_thread(web.fetch, "file:///etc/passwd")
    assert result.ok is False
    assert "http" in result.error.lower()


async def test_web_fetch_handles_a_dead_host(web: WebTools) -> None:
    result = await asyncio.to_thread(web.fetch, "http://127.0.0.1:1/nothing")
    assert result.ok is False


async def test_web_search_returns_results(web: WebTools) -> None:
    result = await asyncio.to_thread(web.search, "hyper cluster omega", limit=3)
    assert result.ok is True, result.error
    results = result.data["results"]
    assert results
    assert results[0]["title"] == "Omega"
    assert results[0]["url"] == "https://example.org/omega"
    assert "control plane" in results[0]["snippet"]


async def test_web_search_falls_back_to_the_second_engine(origin: str) -> None:
    tools = WebTools(
        timeout=15,
        search_endpoint=origin + "/nothing?q={query}",
        wiki_endpoint=origin + "/wiki?srsearch={query}&srlimit={limit}",
    )
    result = await asyncio.to_thread(tools.search, "hyper cluster", limit=3)
    assert result.ok is True, result.error
    assert result.data["engine"] == "wikipedia"
    assert result.data["results"][0]["title"] == "Hyper cluster"


async def test_web_search_reports_an_empty_result_set(origin: str) -> None:
    tools = WebTools(
        timeout=15,
        search_endpoint=origin + "/nothing?q={query}",
        wiki_endpoint=origin + "/nothing?srsearch={query}&srlimit={limit}",
    )
    result = await asyncio.to_thread(tools.search, "hyper cluster", limit=3)
    assert result.ok is False
    assert "no search engine" in result.error


def _internet_available() -> bool:
    import httpx

    try:
        response = httpx.get("https://example.com/", timeout=8.0)
        return response.status_code < 400
    except Exception:
        return False


@pytest.mark.skipif(not _internet_available(), reason="the general internet is blocked in this sandbox")
async def test_web_fetch_reads_a_public_page() -> None:
    tools = WebTools(timeout=20)
    result = await asyncio.to_thread(tools.fetch, "https://example.com/", max_chars=4000)
    assert result.ok is True, result.error
    assert "Example Domain" in result.output


# --------------------------------------------------------------------------- #
# End-to-end agentic tool loop through the head node
# --------------------------------------------------------------------------- #
@pytest.fixture
def head(tmp_path) -> HeadServer:
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        log_level="WARNING",
        log_json=False,
        worker_ttl_seconds=120,
        request_timeout_seconds=60,
        sandbox_enabled=True,
        max_tool_rounds=4,
    )
    return HeadServer(settings).start()


async def test_tool_loop_executes_python_and_answers(head: HeadServer) -> None:
    worker = WorkerHarness(head_url=head.base_url, model="Qwen2.5-7B-Instruct-GGUF")
    patch_capabilities(worker)
    worker.start()
    try:
        await worker.boot()
        async with httpx.AsyncClient(timeout=90.0) as client:
            question = "compute the answer [[call:python]] {\"code\": \"print(6*7)\"}"
            response = await client.post(
                f"{head.base_url}/v1/chat",
                json={
                    "messages": [{"role": "user", "content": question}],
                    "tools": [{"type": "function", "function": {"name": "python"}}],
                    "mode": "auto",
                    "max_tokens": 128,
                    "stream": False,
                },
            )
            assert response.status_code == 200, response.text
            payload = response.json()
            assert payload["error"] is None
            transcript = payload["raw"]["tools"]
            assert transcript, "no tool was executed"
            assert transcript[0]["tool"] == "python"
            assert transcript[0]["ok"] is True
            assert "42" in transcript[0]["output"]
            # The final answer must be synthesised from the tool output.
            assert "42" in payload["text"]
    finally:
        await worker.shutdown()
        worker.stop()
        head.stop()


async def test_tool_catalogue_endpoint_lists_the_tools(head: HeadServer) -> None:
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(f"{head.base_url}/v1/tools")
        assert response.status_code == 200
        payload = response.json()
        names = [schema["function"]["name"] for schema in payload["tools"]]
        assert "python" in names
        assert "web_search" in names
        assert payload["max_rounds"] >= 1


async def test_tool_loop_streams_tool_events(head: HeadServer) -> None:
    worker = WorkerHarness(head_url=head.base_url, model="Qwen2.5-7B-Instruct-GGUF")
    patch_capabilities(worker)
    worker.start()
    try:
        await worker.boot()
        async with httpx.AsyncClient(timeout=90.0) as client:
            events: list[str] = []
            question = "compute [[call:python]] {\"code\": \"print('hello from the sandbox')\"}"
            async with client.stream(
                "POST",
                f"{head.base_url}/v1/chat/stream",
                json={
                    "messages": [{"role": "user", "content": question}],
                    "tools": [{"type": "function", "function": {"name": "python"}}],
                    "max_tokens": 128,
                },
            ) as response:
                assert response.status_code == 200
                current = ""
                async for line in response.aiter_lines():
                    if line.startswith("event:"):
                        current = line[6:].strip()
                    elif line.startswith("data:"):
                        events.append(current)
            assert "tool" in events, events
            assert events[-1] == "done"
    finally:
        await worker.shutdown()
        worker.stop()
        head.stop()


async def test_sandbox_rejects_a_dangerous_request_from_the_model(head: HeadServer) -> None:
    """A model-authored escape attempt must be stopped by the sandbox."""
    worker = WorkerHarness(head_url=head.base_url, model="Qwen2.5-7B-Instruct-GGUF")
    patch_capabilities(worker)
    worker.start()
    try:
        await worker.boot()
        async with httpx.AsyncClient(timeout=90.0) as client:
            question = (
                "escape [[call:python]] "
                + json.dumps({"code": "import os; os.system('echo pwned > /tmp/pwned-by-model')"})
            )
            response = await client.post(
                f"{head.base_url}/v1/chat",
                json={
                    "messages": [{"role": "user", "content": question}],
                    "tools": [{"type": "function", "function": {"name": "python"}}],
                    "max_tokens": 128,
                    "stream": False,
                },
            )
            assert response.status_code == 200
            transcript = response.json()["raw"]["tools"]
            assert transcript
            assert transcript[0]["ok"] is False
            assert not Path("/tmp/pwned-by-model").exists()
    finally:
        await worker.shutdown()
        worker.stop()
        head.stop()
