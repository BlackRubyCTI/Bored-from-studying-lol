"""Isolated tool-use sandbox for the AI.

Three tool families, one executor:

``python``      runs model-authored code in a *separate* process with hard
                resource limits (CPU seconds, address space, file size, file
                descriptors, processes), no environment inheritance, no
                network by default and no process spawning.
``web_fetch``   fetches a URL (size-capped, HTML reduced to text).
``web_search``  live search across several engines with graceful degradation.
``analyze_file`` inspects a local file (text stats, entropy, PDF extraction).

The model talks to the tools with a tiny JSON protocol::

    {"tool": "python", "args": {"code": "print(2**10)"}}
    {"final": "the answer"}

:func:`build_tool_prompt` renders that protocol into a system prompt and
:func:`parse_tool_call` parses the model's reply.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.ids import now_s
from core.logging_setup import get_logger
from tools.jail import (
    KERNEL_TIERS,
    JailConfig,
    build_jail_command,
    detect_isolation,
    jail_script,
    python_executable,
)

LOGGER = get_logger("tools.sandbox")

TOOL_PROTOCOL_MARKER = "[[TOOL_PROTOCOL]]"

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "python",
            "description": (
                "Execute Python 3 code in an isolated sandbox and return stdout/stderr. "
                "No network access, no subprocesses, hard CPU/memory limits, and the "
                "filesystem is limited to the workspace plus read-only system libraries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "The Python source to execute."}
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the live web and return ranked results (title, url, snippet).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "limit": {"type": "integer", "description": "Maximum number of results (default 5)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch a URL and return its readable text content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Absolute http(s) URL."},
                    "max_chars": {"type": "integer", "description": "Maximum characters to return (default 20000)."},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_file",
            "description": "Analyse a local file: size, type, text statistics, entropy and (for PDFs) extracted text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path of the file to analyse."},
                    "max_chars": {"type": "integer", "description": "Maximum characters of extracted text to return."},
                },
                "required": ["path"],
            },
        },
    },
]


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass
class SandboxResult:
    """Outcome of a sandboxed execution."""

    ok: bool = False
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    duration_ms: float = 0.0
    timed_out: bool = False
    error: str = ""
    truncated: bool = False
    value: str = ""
    #: Which isolation tier actually ran: ``container`` / ``bwrap`` /
    #: ``unshare`` / ``process``.  Reported, never assumed (M6).
    sandbox_isolation: str = "process"
    #: True when the child ran against a read-only root filesystem.
    read_only_root: bool = False
    #: Human-readable notes about degraded guarantees (empty when full).
    isolation_notes: list[str] = field(default_factory=list)

    @property
    def kernel_isolated(self) -> bool:
        """True when containment came from the kernel, not monkeypatched guards."""
        return self.sandbox_isolation in KERNEL_TIERS

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "returncode": self.returncode,
            "duration_ms": round(self.duration_ms, 2),
            "timed_out": self.timed_out,
            "error": self.error,
            "truncated": self.truncated,
            "value": self.value,
            "sandbox_isolation": self.sandbox_isolation,
            "read_only_root": self.read_only_root,
            "isolation_notes": list(self.isolation_notes),
        }

    def render(self, limit: int = 4000) -> str:
        """Human/model readable rendering of the result."""
        if self.ok:
            body = self.stdout.strip()
            if not body:
                body = self.value.strip() or "(no output)"
        else:
            body = self.stderr.strip() or self.error or "execution failed"
        if len(body) > limit:
            body = body[:limit] + f"\n... [{len(body) - limit} more characters]"
        return body


@dataclass
class ToolResult:
    """Outcome of a tool call."""

    tool: str = ""
    ok: bool = False
    output: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "ok": self.ok,
            "output": self.output,
            "data": self.data,
            "duration_ms": round(self.duration_ms, 2),
            "error": self.error,
        }


# --------------------------------------------------------------------------- #
# The in-process runner executed by the sandboxed child
# --------------------------------------------------------------------------- #
_RUNNER = r'''
import builtins
import io
import json
import os
import resource
import sys
import traceback

CONFIG = json.loads(os.environ.get("SANDBOX_CONFIG", "{}"))
ALLOWED_HOSTS = set(CONFIG.get("allowed_hosts") or [])
MAX_OUTPUT = int(CONFIG.get("max_output") or 200000)


def _apply_limits():
    cpu = int(CONFIG.get("cpu_seconds") or 10)
    memory = int(CONFIG.get("memory_bytes") or 0)
    fsize = int(CONFIG.get("file_size") or 0)
    nofile = int(CONFIG.get("open_files") or 64)
    npids = int(CONFIG.get("processes") or 0)
    for name, limit in (
        ("RLIMIT_CPU", cpu),
        ("RLIMIT_FSIZE", fsize),
        ("RLIMIT_NOFILE", nofile),
        ("RLIMIT_NPROC", npids),
    ):
        if limit <= 0:
            continue
        res = getattr(resource, name, None)
        if res is None:
            continue
        try:
            resource.setrlimit(res, (limit, limit))
        except (ValueError, OSError):
            pass
    if memory > 0:
        try:
            resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        except (ValueError, OSError):
            pass


def _install_network_guard():
    try:
        import socket
    except Exception:
        return

    original_connect = socket.socket.connect

    def _host_of(address):
        if isinstance(address, (tuple, list)) and address:
            return str(address[0])
        return str(address)

    def guarded_connect(self, address):
        host = _host_of(address)
        if host not in ALLOWED_HOSTS:
            raise PermissionError(
                "network access is disabled in the sandbox (host %r is not allowed)" % host
            )
        return original_connect(self, address)

    socket.socket.connect = guarded_connect

    original_getaddrinfo = socket.getaddrinfo

    def guarded_getaddrinfo(host, *args, **kwargs):
        if host not in ALLOWED_HOSTS:
            raise PermissionError("DNS resolution is disabled in the sandbox")
        return original_getaddrinfo(host, *args, **kwargs)

    socket.getaddrinfo = guarded_getaddrinfo


def _install_process_guard():
    import os as _os

    def _blocked(*_args, **_kwargs):
        raise PermissionError("process spawning is disabled in the sandbox")

    for name in ("system", "fork", "forkpty", "popen", "execv", "execve", "spawnv", "spawnve"):
        if hasattr(_os, name):
            setattr(_os, name, _blocked)
    try:
        import subprocess as _subprocess

        for name in ("run", "call", "check_call", "check_output", "Popen"):
            setattr(_subprocess, name, _blocked)
    except Exception:
        pass
    try:
        import ctypes as _ctypes

        setattr(_ctypes, "CDLL", _blocked)
        setattr(_ctypes, "windll", _blocked)
    except Exception:
        pass
    builtins.input = _blocked
    builtins.exit = _blocked
    builtins.quit = _blocked


def _emit(payload):
    sys.__stdout__.write("\n__SANDBOX_RESULT__" + json.dumps(payload) + "\n")
    sys.__stdout__.flush()


def _split_last_expression(source):
    """Split ``source`` into (statements, trailing expression).

    Mirrors a REPL: ``2 ** 10`` evaluates to ``1024`` instead of doing
    nothing.  Returns ``(source, None)`` when there is no trailing
    expression.
    """
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source, None
    body = tree.body
    if not body or not isinstance(body[-1], ast.Expr):
        return source, None
    head = ast.Module(body=body[:-1], type_ignores=[])
    ast.fix_missing_locations(head)
    try:
        statements = ast.unparse(head)
        expression = ast.unparse(body[-1].value)
    except Exception:
        return source, None
    return statements, expression


def _install_path_guard():
    """Confine file access to the sandbox workspace.

    The process guards stop ``subprocess``/``os.system``, but plain ``open()``
    would still read the head node's ``.env`` (every API key) or write into the
    host filesystem.  Every path-opening builtin is therefore routed through a
    check: only the workspace, the system's read-only library paths and
    ``/dev/null``-style devices stay reachable.  Writes are allowed *inside the
    workspace only*.
    """
    workspace = os.path.realpath(str(CONFIG.get("workspace") or os.getcwd()))
    readable_roots = [workspace]
    for prefix in ("/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc/ssl", "/etc/localtime", "/proc/self", "/dev"):
        readable_roots.append(prefix)
    for prefix in (sys.prefix, sys.base_prefix):
        readable_roots.append(prefix)
    writable_roots = [workspace]
    devices = ("/dev/null", "/dev/zero", "/dev/full", "/dev/random", "/dev/urandom", "/dev/stdout", "/dev/stderr")

    def _under(path, roots):
        for root in roots:
            if path == root or path.startswith(root.rstrip("/") + "/"):
                return True
        return False

    def _check(file, mode):
        if isinstance(file, int):
            return  # an already-open descriptor (never created by user code)
        try:
            raw = os.fspath(file)
        except TypeError:
            return
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        target = os.path.realpath(os.path.join(workspace, raw))
        writing = any(flag in mode for flag in ("w", "a", "x", "+"))
        if writing:
            if not _under(target, writable_roots):
                raise PermissionError("the sandbox only writes inside its workspace (%s)" % target)
            return
        if target in devices:
            return
        if _under(target, readable_roots):
            return
        raise PermissionError("the sandbox cannot read %s (outside its workspace)" % target)

    real_open = builtins.open
    real_io_open = io.open

    def guarded_open(file, mode="r", *args, **kwargs):
        _check(file, mode)
        return real_open(file, mode, *args, **kwargs)

    def guarded_io_open(file, mode="r", *args, **kwargs):
        _check(file, mode)
        return real_io_open(file, mode, *args, **kwargs)

    builtins.open = guarded_open
    io.open = guarded_io_open
    os.open = guarded_open

    real_listdir, real_scandir, real_stat = os.listdir, os.scandir, os.stat
    real_path_exists, real_is_file, real_is_dir = os.path.exists, os.path.isfile, os.path.isdir

    def guarded_listdir(path="."):
        _check(path, "r")
        return real_listdir(path)

    def guarded_scandir(path="."):
        _check(path, "r")
        return real_scandir(path)

    def guarded_stat(path, *args, **kwargs):
        _check(path, "r")
        return real_stat(path, *args, **kwargs)

    def guarded_exists(path):
        try:
            _check(path, "r")
        except PermissionError:
            return False
        return real_path_exists(path)

    def guarded_isfile(path):
        try:
            _check(path, "r")
        except PermissionError:
            return False
        return real_is_file(path)

    def guarded_isdir(path):
        try:
            _check(path, "r")
        except PermissionError:
            return False
        return real_is_dir(path)

    os.listdir = guarded_listdir
    os.scandir = guarded_scandir
    os.stat = guarded_stat
    os.path.exists = guarded_exists
    os.path.isfile = guarded_isfile
    os.path.isdir = guarded_isdir


def _install_seccomp_denylist():
    """Kernel-enforced syscall denylist (M6).

    The outer jail script cannot install this itself: it must still ``execve``
    the interpreter.  So the filter goes in here, as the very first thing the
    runner does and before any untrusted code (or any monkeypatched guard) is
    touched.  Denied syscalls return EPERM instead of killing the process.
    """
    import ctypes
    import ctypes.util

    table = json.loads(os.environ.get("JAIL_ARCH_TABLE", "{}") or "{}")
    spec = table.get(os.uname().machine)
    if not spec:
        return False

    class SockFilter(ctypes.Structure):
        _fields_ = [("code", ctypes.c_uint16), ("jt", ctypes.c_uint8),
                    ("jf", ctypes.c_uint8), ("k", ctypes.c_uint32)]

    class SockFprog(ctypes.Structure):
        _fields_ = [("len", ctypes.c_uint16), ("filter", ctypes.POINTER(SockFilter))]

    deny = ("execve", "execveat", "socket", "mount", "umount2", "pivot_root",
            "chroot", "ptrace", "process_vm_readv", "process_vm_writev", "bpf",
            "perf_event_open", "kexec_load", "init_module", "finit_module",
            "delete_module", "unshare", "setns", "reboot", "swapon", "swapoff",
            "add_key", "request_key", "keyctl")
    LD_ABS_W, JEQ_K, RET_K = 0x20, 0x15, 0x06
    RET_ALLOW, RET_ERRNO, KILL, EPERM = 0x7FFF0000, 0x00050000, 0x80000000, 1

    def ret(k):
        return SockFilter(RET_K, 0, 0, k)

    prog = [SockFilter(LD_ABS_W, 0, 0, 4)]
    prog += [SockFilter(JEQ_K, 1, 0, spec["audit"]), ret(KILL)]
    prog.append(SockFilter(LD_ABS_W, 0, 0, 0))
    for name in deny:
        number = spec.get(name)
        if number is None:
            continue
        prog += [SockFilter(JEQ_K, 0, 1, number), ret(RET_ERRNO | EPERM)]
    prog.append(ret(RET_ALLOW))

    array = (SockFilter * len(prog))(*prog)
    fprog = SockFprog(len(prog), array)
    libc = ctypes.CDLL(ctypes.util.find_library("c") or None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
        return False
    return libc.prctl(22, 2, ctypes.byref(fprog), 0, 0) == 0  # PR_SET_SECCOMP


def main():
    _apply_limits()
    if os.environ.get("JAIL_IN_KERNEL") == "1":
        try:
            _install_seccomp_denylist()
        except Exception:  # noqa: BLE001 - a missing filter must not abort the run
            pass
    # ``guards_off`` exists to *test the kernel boundary alone*: when it is set
    # the monkeypatched guards are skipped, so an escape attempt can only be
    # stopped by namespaces/seccomp/rlimits.  It is never set in production.
    if not CONFIG.get("guards_off"):
        _install_network_guard()
        _install_process_guard()
        _install_path_guard()

    source = ""
    try:
        with open(CONFIG.get("code_file", ""), "r", encoding="utf-8") as handle:
            source = handle.read()
    except OSError as exc:
        _emit({"ok": False, "stdout": "", "stderr": "cannot read code file: %s" % exc,
               "returncode": 70, "error": "io"})
        return

    statements, expression = _split_last_expression(source)
    out_buffer = io.StringIO()
    err_buffer = io.StringIO()
    result = {"ok": True, "stdout": "", "stderr": "", "returncode": 0, "error": "", "value": ""}
    real_stdout, real_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out_buffer, err_buffer
    try:
        namespace = {"__name__": "__main__", "__builtins__": builtins}
        exec(compile(statements, "<sandbox>", "exec"), namespace)  # noqa: S102 - the whole point
        if expression is not None:
            value = eval(compile(expression, "<sandbox>", "eval"), namespace)  # noqa: S307
            result["value"] = repr(value)
    except SystemExit as exc:
        result["returncode"] = int(exc.code or 0)
    except BaseException:  # noqa: BLE001 - every failure is reported, not raised
        result["ok"] = False
        result["returncode"] = 1
        result["error"] = traceback.format_exc(limit=8)
    finally:
        sys.stdout, sys.stderr = real_stdout, real_stderr

    stdout = out_buffer.getvalue()
    stderr = err_buffer.getvalue()
    if len(stdout) > MAX_OUTPUT:
        stdout = stdout[:MAX_OUTPUT]
        result["truncated"] = True
    if len(stderr) > MAX_OUTPUT:
        stderr = stderr[:MAX_OUTPUT]
    result["stdout"] = stdout
    result["stderr"] = stderr
    _emit(result)


if __name__ == "__main__":
    main()
'''


class PythonSandbox:
    """Executes untrusted Python with hard isolation."""

    def __init__(
        self,
        *,
        workdir: str | Path | None = None,
        timeout: float = 20.0,
        memory_limit_mb: int = 512,
        cpu_seconds: int | None = None,
        file_size_mb: int = 32,
        open_files: int = 64,
        processes: int = 0,
        allowed_hosts: list[str] | None = None,
        max_output_chars: int = 200_000,
        isolation: str = "auto",
        seccomp: bool = True,
        mask_paths: list[str] | None = None,
        guards_off: bool = False,
    ) -> None:
        self.workdir = Path(workdir).expanduser() if workdir else Path(tempfile.gettempdir()) / "hypercluster-sandbox"
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.memory_limit_mb = memory_limit_mb
        self.cpu_seconds = cpu_seconds
        self.file_size_mb = file_size_mb
        self.open_files = open_files
        self.processes = processes
        self.allowed_hosts = list(allowed_hosts or [])
        self.max_output_chars = max_output_chars
        #: Test-only: run with the monkeypatched guards disabled so that the
        #: kernel boundary can be exercised on its own.
        self.guards_off = bool(guards_off)
        self.jail = JailConfig(
            preferred=isolation,
            seccomp=seccomp,
            mask_paths=tuple(mask_paths) if mask_paths is not None else JailConfig.mask_paths,
        )

    # ------------------------------------------------------------------ #
    @property
    def isolation(self) -> str:
        """The tier this sandbox would use right now (probed, not assumed)."""
        return detect_isolation(self.jail)

    # ------------------------------------------------------------------ #
    def run(self, code: str, *, timeout: float | None = None) -> SandboxResult:
        """Execute ``code`` and return the captured result.

        Containment is layered: the kernel jail (namespaces, read-only root,
        seccomp) is the boundary, and the monkeypatched guards inside the
        runner remain as defence in depth.  When the kernel refuses every jail
        tier we still run — and report ``sandbox_isolation: "process"`` so
        nobody mistakes the guards for a security boundary.
        """
        limit = timeout if timeout is not None else self.timeout
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(dir=str(self.workdir)) as tmp:
            tmp_path = Path(tmp)
            code_file = tmp_path / "code.py"
            code_file.write_text(code or "", encoding="utf-8")
            runner_file = tmp_path / "runner.py"
            runner_file.write_text(_RUNNER, encoding="utf-8")

            config = {
                "code_file": str(code_file),
                "cpu_seconds": self.cpu_seconds if self.cpu_seconds is not None else max(1, int(limit) + 1),
                "memory_bytes": self.memory_limit_mb * 1024 * 1024,
                "file_size": self.file_size_mb * 1024 * 1024,
                "open_files": self.open_files,
                "processes": self.processes,
                "allowed_hosts": self.allowed_hosts,
                "workspace": str(tmp_path),
                "max_output": self.max_output_chars,
                "guards_off": bool(self.guards_off),
            }
            python = python_executable()
            env = {
                "SANDBOX_CONFIG": json.dumps(config),
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "HOME": str(tmp_path),
                "TMPDIR": str(tmp_path),
                "LANG": "C.UTF-8",
                "PYTHONHASHSEED": "0",
                "PYTHONIOENCODING": "utf-8",
            }
            jail_script_path = tmp_path / "jail.py"
            limits = {
                "cpu_seconds": config["cpu_seconds"],
                "memory_bytes": config["memory_bytes"],
                "file_size": config["file_size"],
                "open_files": config["open_files"],
                "processes": max(1, self.processes or 1),
            }
            # A *stable* scratch path is needed because the jail hides /tmp.
            scratch = tempfile.mkdtemp(prefix="hc-jail-")
            try:
                jail_script_path.write_text(jail_script(), encoding="utf-8")
                jail_cmd = build_jail_command(
                    python=python,
                    runner_path=str(runner_file),
                    workspace=str(tmp_path),
                    jail_script_path=str(jail_script_path),
                    scratch_dir=scratch,
                    config=self.jail,
                    limits=limits,
                    child_env=env,
                )
                for key, value in jail_cmd.env.items():
                    env[key] = value
                if jail_cmd.tier in KERNEL_TIERS and self.jail.seccomp:
                    env["JAIL_IN_KERNEL"] = "1"
                try:
                    completed = subprocess.run(
                        jail_cmd.argv,
                        capture_output=True,
                        text=True,
                        timeout=limit,
                        cwd=str(tmp_path),
                        env=env,
                        check=False,
                    )
                except subprocess.TimeoutExpired:
                    return SandboxResult(
                        ok=False,
                        timed_out=True,
                        error=f"execution exceeded the {limit:.0f}s limit",
                        duration_ms=(time.perf_counter() - started) * 1000.0,
                        sandbox_isolation=jail_cmd.tier,
                        read_only_root=jail_cmd.read_only_root,
                        isolation_notes=jail_cmd.notes,
                    )
                except OSError as exc:
                    return SandboxResult(
                        ok=False,
                        error=f"could not start the sandbox process: {exc}",
                        duration_ms=(time.perf_counter() - started) * 1000.0,
                        sandbox_isolation=jail_cmd.tier,
                        read_only_root=jail_cmd.read_only_root,
                        isolation_notes=jail_cmd.notes,
                    )
            finally:
                _rmtree(scratch)

        payload = _extract_result(completed.stdout)
        duration = (time.perf_counter() - started) * 1000.0
        notes = list(jail_cmd.notes)
        if jail_cmd.tier in KERNEL_TIERS and not jail_cmd.read_only_root:
            notes.append("root filesystem was not remounted read-only on this host")
        if payload is None:
            return SandboxResult(
                ok=False,
                stdout=completed.stdout[-4000:],
                stderr=completed.stderr[-4000:],
                returncode=completed.returncode,
                error="the sandbox process died before reporting a result",
                duration_ms=duration,
                sandbox_isolation=jail_cmd.tier,
                read_only_root=jail_cmd.read_only_root,
                isolation_notes=notes,
            )
        return SandboxResult(
            ok=bool(payload.get("ok")),
            stdout=str(payload.get("stdout") or ""),
            stderr=str(payload.get("stderr") or "") or str(payload.get("error") or ""),
            returncode=int(payload.get("returncode") or 0),
            timed_out=False,
            error="" if payload.get("ok") else str(payload.get("error") or "execution failed"),
            truncated=bool(payload.get("truncated")),
            value=str(payload.get("value") or ""),
            duration_ms=duration,
            sandbox_isolation=jail_cmd.tier,
            read_only_root=jail_cmd.read_only_root,
            isolation_notes=notes,
        )


def _rmtree(path: str) -> None:
    """Best-effort recursive removal (the jail may have tightened modes)."""
    import shutil

    subprocess.run(["chmod", "-R", "u+rwX", path], capture_output=True, check=False)
    shutil.rmtree(path, ignore_errors=True)


def _extract_result(stdout: str) -> dict[str, Any] | None:
    """Pull the JSON payload out of the runner's stdout."""
    marker = "__SANDBOX_RESULT__"
    if marker not in stdout:
        return None
    tail = stdout.rsplit(marker, 1)[1].strip()
    if not tail:
        return None
    try:
        return json.loads(tail)
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------- #
# Web tools
# --------------------------------------------------------------------------- #
_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_WS_RE = re.compile(r"\n{3,}")


def html_to_text(html: str) -> str:
    """Reduce an HTML document to readable text."""
    without_scripts = _SCRIPT_RE.sub(" ", html)
    text = _TAG_RE.sub(" ", without_scripts)
    import html as html_module

    text = html_module.unescape(text)
    lines = [line.strip() for line in text.splitlines()]
    return _WS_RE.sub("\n\n", "\n".join(line for line in lines if line))


class WebTools:
    """Live web search and scraping."""

    def __init__(
        self,
        *,
        timeout: float = 20.0,
        max_bytes: int = 3_000_000,
        user_agent: str = "HyperClusterOmega/1.0 (+https://github.com/hyper-cluster-omega)",
        language: str = "en",
        search_endpoint: str = "https://html.duckduckgo.com/html/?q={query}",
        wiki_endpoint: str = "https://{lang}.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json&srlimit={limit}",
    ) -> None:
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.user_agent = user_agent
        self.language = language
        self.search_endpoint = search_endpoint
        self.wiki_endpoint = wiki_endpoint

    # ------------------------------------------------------------------ #
    def fetch(self, url: str, *, max_chars: int = 20000) -> ToolResult:
        """Fetch a URL and return readable text."""
        import httpx

        started = time.perf_counter()
        if not url.lower().startswith(("http://", "https://")):
            return ToolResult(
                tool="web_fetch",
                ok=False,
                error="only http(s) URLs can be fetched",
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )
        try:
            with httpx.Client(
                timeout=self.timeout,
                follow_redirects=True,
                headers={"user-agent": self.user_agent},
            ) as client:
                with client.stream("GET", url) as response:
                    content_type = response.headers.get("content-type", "")
                    chunks: list[bytes] = []
                    size = 0
                    for chunk in response.iter_bytes():
                        chunks.append(chunk)
                        size += len(chunk)
                        if size >= self.max_bytes:
                            break
                    raw = b"".join(chunks)
                    status = response.status_code
        except httpx.HTTPError as exc:
            return ToolResult(
                tool="web_fetch",
                ok=False,
                error=f"fetch failed: {exc}",
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )

        encoding = "utf-8"
        if "charset=" in content_type.lower():
            encoding = content_type.lower().split("charset=")[-1].split(";")[0].strip() or "utf-8"
        try:
            body = raw.decode(encoding, errors="replace")
        except LookupError:
            body = raw.decode("utf-8", errors="replace")

        if "html" in content_type.lower() or body.lstrip()[:200].lower().startswith(("<!doctype html", "<html")):
            text = html_to_text(body)
            kind = "html"
        elif "json" in content_type.lower():
            text = body
            kind = "json"
        else:
            text = body
            kind = content_type or "text"

        clipped = text[:max_chars]
        return ToolResult(
            tool="web_fetch",
            ok=status < 400,
            output=clipped,
            data={
                "url": url,
                "status": status,
                "content_type": content_type,
                "kind": kind,
                "bytes": size,
                "truncated": len(text) > max_chars,
                "fetched_at": now_s(),
            },
            duration_ms=(time.perf_counter() - started) * 1000.0,
            error="" if status < 400 else f"HTTP {status}",
        )

    # ------------------------------------------------------------------ #
    def search(self, query: str, *, limit: int = 5) -> ToolResult:
        """Search the live web (DuckDuckGo HTML, Wikipedia fallback)."""
        started = time.perf_counter()
        results = self._search_duckduckgo(query, limit=limit)
        engine = "duckduckgo"
        if not results:
            results = self._search_wikipedia(query, limit=limit)
            engine = "wikipedia"
        rendered = "\n\n".join(
            f"{index + 1}. {item['title']}\n   {item['url']}\n   {item['snippet'][:280]}"
            for index, item in enumerate(results)
        )
        return ToolResult(
            tool="web_search",
            ok=bool(results),
            output=rendered or "(no results)",
            data={"query": query, "engine": engine, "results": results, "count": len(results)},
            duration_ms=(time.perf_counter() - started) * 1000.0,
            error="" if results else "no search engine returned results",
        )

    def _search_duckduckgo(self, query: str, *, limit: int) -> list[dict[str, Any]]:
        from urllib.parse import parse_qs, quote_plus, unquote, urlparse

        import httpx

        url = self.search_endpoint.format(query=quote_plus(query))
        try:
            with httpx.Client(
                timeout=self.timeout,
                follow_redirects=True,
                headers={"user-agent": self.user_agent},
            ) as client:
                response = client.get(url)
            if response.status_code >= 400:
                return []
            html = response.text
        except httpx.HTTPError as exc:
            LOGGER.warning("duckduckgo search failed", extra={"hc_error": str(exc)})
            return []

        results: list[dict[str, Any]] = []
        blocks = re.findall(
            r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>(.*?)(?=class="result__a"|$)',
            html,
            re.DOTALL,
        )
        for href, title_html, tail in blocks[: limit * 2]:
            title = html_to_text(title_html).strip()
            parsed = urlparse(href)
            if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
                params = parse_qs(parsed.query)
                target = params.get("uddg", [""])[0]
                if target:
                    href = unquote(target)
            snippet = html_to_text(tail).strip()[:400]
            if title and href.startswith("http"):
                results.append({"title": title, "url": href, "snippet": snippet})
            if len(results) >= limit:
                break
        return results[:limit]

    def _search_wikipedia(self, query: str, *, limit: int) -> list[dict[str, Any]]:
        from urllib.parse import quote_plus

        import httpx

        url = self.wiki_endpoint.format(
            query=quote_plus(query),
            limit=max(1, limit),
            lang=self.language,
        )
        try:
            with httpx.Client(timeout=self.timeout, headers={"user-agent": self.user_agent}) as client:
                response = client.get(url)
            if response.status_code >= 400:
                return []
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            LOGGER.warning("wikipedia search failed", extra={"hc_error": str(exc)})
            return []
        hits = (payload.get("query") or {}).get("search") or []
        results: list[dict[str, Any]] = []
        for hit in hits[:limit]:
            title = str(hit.get("title", ""))
            snippet = html_to_text(str(hit.get("snippet", ""))).strip()
            link = f"https://{self.language}.wikipedia.org/wiki/{title.replace(' ', '_')}"
            results.append({"title": title, "url": link, "snippet": snippet})
        return results


# --------------------------------------------------------------------------- #
# File analyzer
# --------------------------------------------------------------------------- #
class FileAnalyzer:
    """Inspects local files."""

    def __init__(self, *, max_text_chars: int = 20000, allowed_roots: list[str] | None = None) -> None:
        self.max_text_chars = max_text_chars
        self.allowed_roots = [Path(root).expanduser() for root in (allowed_roots or [])]

    def _allowed(self, path: Path) -> bool:
        if not self.allowed_roots:
            return True
        resolved = path.resolve()
        return any(str(resolved).startswith(str(root.resolve())) for root in self.allowed_roots)

    def analyze(self, path: str, *, max_chars: int | None = None) -> ToolResult:
        """Return statistics and (where possible) the text of a file."""
        started = time.perf_counter()
        target = Path(path).expanduser()
        if not target.is_file():
            return ToolResult(
                tool="analyze_file",
                ok=False,
                error=f"file not found: {target}",
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )
        if not self._allowed(target):
            return ToolResult(
                tool="analyze_file",
                ok=False,
                error=f"path is outside the allowed roots: {target}",
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )
        limit = max_chars or self.max_text_chars
        size = target.stat().st_size
        data = target.read_bytes()
        entropy = _shannon_entropy(data[:1_000_000])
        text, kind = _extract_text(target, data)
        clipped = text[:limit]
        stats: dict[str, Any] = {
            "path": str(target),
            "name": target.name,
            "size_bytes": size,
            "entropy": round(entropy, 4),
            "text_kind": kind,
            "truncated": len(text) > limit,
        }
        if kind == "text":
            words = clipped.split()
            stats.update(
                {
                    "lines": clipped.count("\n") + (1 if clipped else 0),
                    "words": len(words),
                    "characters": len(clipped),
                    "top_terms": [term for term, _ in Counter(w.lower() for w in words if len(w) > 3).most_common(10)],
                }
            )
        return ToolResult(
            tool="analyze_file",
            ok=True,
            output=clipped,
            data=stats,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )


def _shannon_entropy(data: bytes) -> float:
    """Shannon entropy of a byte string (bits per byte)."""
    if not data:
        return 0.0
    counts = Counter(data)
    total = len(data)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _extract_text(path: Path, data: bytes) -> tuple[str, str]:
    """Best-effort text extraction (plain text, PDF, DOCX-ish)."""
    suffix = path.suffix.lower()
    if suffix == ".pdf" or data[:5] == b"%PDF-":
        text = _extract_pdf_text(path, data)
        if text:
            return text, "pdf"
        return "", "pdf-binary"
    if suffix in (".txt", ".md", ".py", ".json", ".csv", ".log", ".yaml", ".yml", ".html", ".xml", ".sh"):
        return data.decode("utf-8", errors="replace"), "text"
    try:
        return data.decode("utf-8"), "text"
    except UnicodeDecodeError:
        return "", "binary"


def _extract_pdf_text(path: Path, data: bytes) -> str:
    """Extract the text of a PDF using pypdf when it is installed."""
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 - optional dependency
        try:
            from PyPDF2 import PdfReader  # type: ignore[import-not-found]  # noqa: N813
        except Exception:  # noqa: BLE001
            return ""
    try:
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages[:200]:
            pages.append(page.extract_text() or "")
        return "\n\n".join(pages)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("pdf extraction failed", extra={"hc_error": str(exc)})
        return ""


# --------------------------------------------------------------------------- #
# Executor
# --------------------------------------------------------------------------- #
class ToolExecutor:
    """Dispatches tool calls coming from the model."""

    def __init__(
        self,
        *,
        sandbox: PythonSandbox | None = None,
        web: WebTools | None = None,
        files: FileAnalyzer | None = None,
        enabled_tools: list[str] | None = None,
    ) -> None:
        self.sandbox = sandbox or PythonSandbox()
        self.web = web or WebTools()
        self.files = files or FileAnalyzer()
        self.enabled_tools = enabled_tools or ["python", "web_search", "web_fetch", "analyze_file"]

    # ------------------------------------------------------------------ #
    def list_tools(self) -> list[dict[str, Any]]:
        """The tool schemas advertised to the model."""
        return [schema for schema in TOOL_SCHEMAS if schema["function"]["name"] in self.enabled_tools]

    def execute(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        """Run one tool call and return its result."""
        args = arguments or {}
        if name not in self.enabled_tools:
            return ToolResult(tool=name, ok=False, error=f"unknown tool: {name}")
        if name == "python":
            code = str(args.get("code") or "")
            if not code.strip():
                return ToolResult(tool=name, ok=False, error="no code supplied")
            result = self.sandbox.run(code)
            return ToolResult(
                tool=name,
                ok=result.ok,
                output=result.render(),
                data=result.to_dict(),
                duration_ms=result.duration_ms,
                error=result.error,
            )
        if name == "web_search":
            return self.web.search(str(args.get("query") or ""), limit=int(args.get("limit") or 5))
        if name == "web_fetch":
            return self.web.fetch(
                str(args.get("url") or ""),
                max_chars=int(args.get("max_chars") or 20000),
            )
        if name == "analyze_file":
            return self.files.analyze(str(args.get("path") or ""), max_chars=args.get("max_chars"))
        return ToolResult(tool=name, ok=False, error=f"tool '{name}' has no implementation")

    def execute_call(self, call: dict[str, Any]) -> ToolResult:
        """Execute a parsed tool call dictionary."""
        name = str(call.get("tool") or call.get("name") or "")
        arguments = call.get("args") or call.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"raw": arguments}
        return self.execute(name, arguments if isinstance(arguments, dict) else {})


# --------------------------------------------------------------------------- #
# Model-facing protocol
# --------------------------------------------------------------------------- #
def build_tool_prompt(tools: list[dict[str, Any]], question: str) -> str:
    """Render the tool-call protocol as a system prompt."""
    names = ", ".join(schema["function"]["name"] for schema in tools)
    catalogue = json.dumps(tools, indent=2)
    return (
        f"{TOOL_PROTOCOL_MARKER}\n"
        "You are connected to a live tool environment. To use a tool, reply with EXACTLY ONE "
        "JSON object and nothing else:\n"
        '{"tool": "<tool name>", "args": {...}}\n'
        "When you can answer without a tool, or when the tools already gave you everything you "
        'need, reply with EXACTLY ONE JSON object:\n'
        '{"final": "<your complete answer>"}\n\n'
        f"Available tools: {names}\n\n"
        f"Tool schemas:\n{catalogue}\n\n"
        f"Request:\n{question}\n"
    )


def parse_tool_call(text: str) -> dict[str, Any] | None:
    """Parse the model's reply into ``{"tool": ..., "args": ...}`` or ``{"final": ...}``.

    Returns ``None`` when the text is not a tool call at all.
    """
    if not text:
        return None
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end > start:
            candidate = candidate[start : end + 1]
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if "tool" in payload or "name" in payload:
        name = str(payload.get("tool") or payload.get("name") or "")
        arguments = payload.get("args")
        if arguments is None:
            arguments = payload.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"raw": arguments}
        if not name:
            return None
        return {"tool": name, "args": arguments if isinstance(arguments, dict) else {}}
    if "final" in payload:
        return {"final": str(payload.get("final") or "")}
    return None


def extract_final_answer(text: str) -> str:
    """Pull the ``final`` payload out of a model reply (or return the text)."""
    call = parse_tool_call(text)
    if call and "final" in call:
        return call["final"]
    return (text or "").strip()


__all__ = [
    "TOOL_SCHEMAS",
    "TOOL_PROTOCOL_MARKER",
    "SandboxResult",
    "ToolResult",
    "PythonSandbox",
    "WebTools",
    "FileAnalyzer",
    "ToolExecutor",
    "html_to_text",
    "build_tool_prompt",
    "parse_tool_call",
    "extract_final_answer",
]
