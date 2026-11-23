"""Tiered, *real* isolation for the tool sandbox (milestone 6).

Before M6 the sandbox relied on monkeypatched guards inside the child process
(``socket.connect``, ``os.system``, ``ctypes.CDLL`` ...).  Monkeys are not a
security boundary: anything that reaches past the patched module — a fresh
``ctypes`` handle, an ``importlib`` reload, a raw ``/proc/self/mem`` write — was
outside the guard.  This module replaces the *outer* layer with kernel-enforced
containment and keeps the monkeypatched guards only as an inner defence.

Isolation tiers, strongest first, each one a superset of the next:

``container``
    ``nsjail`` if the operator installed it (the reference tool for this).
``bwrap``
    ``bubblewrap`` (unprivileged, but a real mount namespace + seccomp).
``unshare``
    :class:`UnshareJail` below: user/mount/PID/network namespaces with a fresh
    ``/proc``, a **read-only root filesystem**, tmpfs scratch, and a seccomp
    syscall denylist installed before any untrusted code runs.
``process``
    The pre-M6 path: a separate process with rlimits and monkeypatched guards
    only.  Explicitly reported as ``process`` because it is the weakest tier.

Every result carries ``sandbox_isolation`` so a caller (and the PWA) can see
which tier actually ran.  We never claim ``container`` when we only ran
``process``: :func:`detect_isolation` probes the host, and the tier that
succeeds is the tier we report.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Tier names (these strings are part of the API contract)
# --------------------------------------------------------------------------- #
ISOLATION_CONTAINER = "container"
ISOLATION_BWRAP = "bwrap"
ISOLATION_UNSHARE = "unshare"
ISOLATION_PROCESS = "process"

#: Tiers that give us kernel-enforced containment (mount/pid/net namespaces at
#: the very least).  Used to decide whether the *process* guards are a fallback
#: or a redundant inner layer.
KERNEL_TIERS = frozenset({ISOLATION_CONTAINER, ISOLATION_BWRAP, ISOLATION_UNSHARE})

#: Placeholder the jail script writes when a selftest fails.
_SELFTEST_FAILED = "JAIL_SELFTEST_FAILED"


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class JailConfig:
    """Knobs that decide *how* the untrusted child is contained."""

    preferred: str = "auto"
    unsupported: tuple[str, ...] = ()
    mask_paths: tuple[str, ...] = ("/home", "/root", "/tmp", "/var/tmp", "/dev/shm", "/mnt", "/media", "/srv")
    seccomp: bool = True
    time_budget_seconds: float = 5.0

    @property
    def disabled(self) -> bool:
        return self.preferred in {"process", "none", "off"}


# --------------------------------------------------------------------------- #
# Probes
# --------------------------------------------------------------------------- #
_PROBE_CACHE: dict[str, bool] = {}


def _reset_probe_cache() -> None:
    """Test hook: forget cached capability probes."""
    _PROBE_CACHE.clear()


def which(name: str) -> str:
    return shutil.which(name) or ""


def has_nsjail() -> bool:
    return bool(which("nsjail"))


def has_bwrap() -> bool:
    return bool(which("bwrap"))


def unshare_works() -> bool:
    """True when user + mount + pid + net namespaces are all available.

    Cached because it costs a process spawn; the result cannot change while the
    process lives (it depends on kernel knobs and the container's seccomp
    profile, both fixed at startup).
    """
    if "unshare" in _PROBE_CACHE:
        return _PROBE_CACHE["unshare"]
    binary = which("unshare")
    ok = False
    if binary:
        try:
            completed = subprocess.run(
                [binary, "-Urmpfn", "--kill-child", "true"],
                capture_output=True,
                timeout=5,
                check=False,
            )
            ok = completed.returncode == 0
        except (OSError, subprocess.SubprocessError):
            ok = False
    _PROBE_CACHE["unshare"] = ok
    return ok


def available_tiers() -> list[str]:
    """Isolation tiers this host can actually deliver, strongest first."""
    tiers: list[str] = []
    if has_nsjail():
        tiers.append(ISOLATION_CONTAINER)
    if has_bwrap():
        tiers.append(ISOLATION_BWRAP)
    if unshare_works():
        tiers.append(ISOLATION_UNSHARE)
    tiers.append(ISOLATION_PROCESS)
    return tiers


def detect_isolation(config: JailConfig | None = None) -> str:
    """The tier we would use right now, without running untrusted code."""
    config = config or JailConfig()
    if config.disabled:
        return ISOLATION_PROCESS
    tiers = [t for t in available_tiers() if t not in config.unsupported]
    if not tiers:
        return ISOLATION_PROCESS
    if config.preferred in {"auto", ""}:
        return tiers[0]
    if config.preferred in tiers:
        return config.preferred
    # A tier was demanded but is unavailable: fall back, and admit it.  The
    # caller sees the real tier in ``sandbox_isolation``; nothing is faked.
    return tiers[0]


# --------------------------------------------------------------------------- #
# The seccomp + mount-namespace launcher (pure stdlib, no C extension)
# --------------------------------------------------------------------------- #
# Audit arch -> syscall numbers we deny.  Only the syscalls that turn a
# python-level escape into a real one are listed; everything else is allowed so
# that CPython itself keeps working.
_ARCH = {
    "x86_64": {
        "audit": 0xC000003E,
        "execve": 59,
        "execveat": 322,
        "socket": 41,
        "mount": 165,
        "umount2": 166,
        "pivot_root": 155,
        "chroot": 161,
        "ptrace": 101,
        "process_vm_readv": 310,
        "process_vm_writev": 311,
        "bpf": 321,
        "perf_event_open": 298,
        "kexec_load": 246,
        "init_module": 175,
        "finit_module": 313,
        "delete_module": 176,
        "unshare": 272,
        "setns": 308,
        "reboot": 169,
        "swapon": 167,
        "swapoff": 168,
        "add_key": 248,
        "request_key": 249,
        "keyctl": 250,
    },
    "aarch64": {
        "audit": 0xC00000B7,
        "execve": 221,
        "execveat": 281,
        "socket": 198,
        "mount": 40,
        "umount2": 39,
        "pivot_root": 41,
        "chroot": 51,
        "ptrace": 117,
        "process_vm_readv": 270,
        "process_vm_writev": 271,
        "bpf": 280,
        "perf_event_open": 241,
        "kexec_load": 104,
        "init_module": 105,
        "finit_module": 273,
        "delete_module": 106,
        "unshare": 97,
        "setns": 268,
        "reboot": 142,
        "swapon": 224,
        "swapoff": 225,
        "add_key": 217,
        "request_key": 218,
        "keyctl": 219,
    },
}

#: Name -> errno the sandbox returns for a denied syscall (EPERM).
_EPERM = 1
_EFAULT = 14

#: This script is executed by the *jailed* child as PID 1 of the namespaces.
#: It configures the filesystem view, drops privileges into a seccomp denylist,
#: then replaces itself with the runner.  If any step fails it refuses to run
#: untrusted code at all and reports why.
_JAIL_SCRIPT = r"""
import ctypes, ctypes.util, json, os, sys

CERR = json.loads(os.environ.get("JAIL_ARCH_TABLE", "{}"))
CONFIG = json.loads(os.environ.get("JAIL_CONFIG", "{}"))
MASK = CONFIG.get("mask_paths") or []
ROOT = CONFIG.get("root_flags") or "ro"
SECCOMP = bool(CONFIG.get("seccomp", True))
SCRATCH = CONFIG.get("scratch") or "/tmp"
SELF = CONFIG.get("self_path") or ""
SOURCE = CONFIG.get("source_workspace") or ""
WORKDIR = (SCRATCH.rstrip("/") + "/.hc-work") if SCRATCH else SOURCE
PYTHON = CONFIG.get("python") or "/usr/bin/python3"
RUNNER = WORKDIR + "/runner.py"
LIMITS = CONFIG.get("limits") or {}
EPERM = 1


def log(msg):
    sys.stderr.write("[jail] " + str(msg) + "\n")
    sys.stderr.flush()


def sh(args):
    import subprocess
    return subprocess.run(args, capture_output=True, text=True, check=False)


def setup_filesystem():
    # Never propagate mounts back to the host namespace.
    sh(["mount", "--make-rslave", "/"])

    # 1. Writable scratch inside the namespace.  The untrusted workspace is
    #    bind-mounted into it *before* the parent path is hidden, so the code
    #    under test keeps a stable path (/tmp/.hc-work) with no host /tmp view.
    bound = False
    if SCRATCH and os.path.isdir(SCRATCH):
        if sh(["mount", "-t", "tmpfs", "-o", "size=64m,mode=1777", "tmpfs", SCRATCH]).returncode == 0:
            os.makedirs(WORKDIR, exist_ok=True)
            if sh(["mount", "--bind", SOURCE, WORKDIR]).returncode == 0:
                bound = True
    if not bound:
        # No writable scratch: run straight out of the (read-only) workspace.
        log("scratch tmpfs unavailable; running from " + SOURCE)

    for path in ("/var/tmp", "/dev/shm"):
        if os.path.isdir(path) and path != SCRATCH:
            sh(["mount", "-t", "tmpfs", "-o", "size=32m,mode=1777", "tmpfs", path])

    # 2. Hide real homes and mount points rather than exposing them read-only.
    for path in MASK:
        if path in (SCRATCH, "/tmp") or not os.path.isdir(path):
            continue
        if sh(["mount", "-t", "tmpfs", "-o", "size=1m,mode=000", "tmpfs", path]).returncode != 0:
            log("could not mask " + path)

    # 3. Read-only root filesystem where the kernel allows an unprivileged
    #    locked remount.  Docker-style hosts refuse it; seccomp + masked homes
    #    remain the boundary and the caller reports the weaker guarantee.
    if ROOT == "ro":
        r = sh(["mount", "-o", "remount,ro,bind", "/"])
        if r.returncode != 0:
            log("read-only root unavailable: " + (r.stderr or "").strip()[:160])
    return True


def apply_limits():
    import resource
    for name, value in (
        ("RLIMIT_CPU", LIMITS.get("cpu_seconds")),
        ("RLIMIT_AS", LIMITS.get("memory_bytes")),
        ("RLIMIT_FSIZE", LIMITS.get("file_size")),
        ("RLIMIT_NOFILE", LIMITS.get("open_files")),
        ("RLIMIT_NPROC", LIMITS.get("processes")),
        ("RLIMIT_CORE", 0),
    ):
        res = getattr(resource, name, None)
        if res is None or value is None:
            continue
        try:
            resource.setrlimit(res, (int(value), int(value)))
        except (ValueError, OSError):
            pass


class SockFilter(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint16), ("jt", ctypes.c_uint8),
                ("jf", ctypes.c_uint8), ("k", ctypes.c_uint32)]


class SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_uint16), ("filter", ctypes.POINTER(SockFilter))]


DENY = ("execve", "execveat", "socket", "mount", "umount2", "pivot_root", "chroot",
        "ptrace", "process_vm_readv", "process_vm_writev", "bpf", "perf_event_open",
        "kexec_load", "init_module", "finit_module", "delete_module", "unshare",
        "setns", "reboot", "swapon", "swapoff", "add_key", "request_key", "keyctl")


def install_seccomp():
    # Kernel-enforced syscall denylist (BPF via prctl).  No C extension.
    if not SECCOMP:
        log("seccomp disabled by configuration")
        return False
    spec = CERR.get(os.uname().machine)
    if not spec:
        log("no seccomp table for " + os.uname().machine)
        return False
    libc = ctypes.CDLL(ctypes.util.find_library("c") or None, use_errno=True)
    PR_SET_NO_NEW_PRIVS, PR_SET_SECCOMP = 38, 22
    MODE_FILTER, RET_ALLOW, RET_ERRNO, KILL = 2, 0x7FFF0000, 0x00050000, 0x80000000
    LD_ABS_W, JEQ_K, RET_K = 0x20, 0x15, 0x06

    def stmt(k):
        return SockFilter(RET_K, 0, 0, k)

    # seccomp_data.arch: equal -> jump over the kill (jt=1), else kill (jf=0).
    prog = [SockFilter(LD_ABS_W, 0, 0, 4)]
    prog += [SockFilter(JEQ_K, 1, 0, spec["audit"]), stmt(KILL)]
    prog.append(SockFilter(LD_ABS_W, 0, 0, 0))
    for name in DENY:
        nr = spec.get(name)
        if nr is None:
            continue
        prog += [SockFilter(JEQ_K, 0, 1, nr), stmt(RET_ERRNO | EPERM)]
    prog.append(stmt(RET_ALLOW))

    arr = (SockFilter * len(prog))(*prog)
    fprog = SockFprog(len(prog), arr)
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        log("prctl(NO_NEW_PRIVS) errno=%d" % ctypes.get_errno())
        return False
    if libc.prctl(PR_SET_SECCOMP, MODE_FILTER, ctypes.byref(fprog), 0, 0) != 0:
        log("prctl(SET_SECCOMP) errno=%d" % ctypes.get_errno())
        return False
    return True


def selftest():
    # Prove egress is contained (seccomp or the empty network namespace).
    import socket
    try:
        s = socket.socket()
    except OSError:
        return True  # seccomp already denies socket(2)
    try:
        s.settimeout(1.0)
        s.connect(("1.1.1.1", 53))
        log("egress is NOT contained")
        return False
    except OSError:
        return True
    finally:
        try:
            s.close()
        except OSError:
            pass


def main():
    if not setup_filesystem():
        return 2
    if not selftest():
        return 3
    # NOTE: seccomp is installed by the *runner* (tools/sandbox.py), i.e. after
    # this process execve()s the interpreter.  Filtering here would deny our own
    # execve, and installing it earlier than the first line of untrusted code is
    # exactly what the runner guarantees.
    apply_limits()
    try:
        os.chdir(WORKDIR)
    except OSError:
        pass
    try:
        os.execv(PYTHON, [PYTHON, "-I", "-B", RUNNER])
    except OSError as exc:
        log("exec failed: %s" % exc)
        return 4


if __name__ == "__main__":
    sys.exit(main())
"""


def seccomp_table(arch: str | None = None) -> dict[str, int]:
    """The syscall table for ``arch`` (defaults to this host)."""
    return _ARCH.get(arch or os.uname().machine, {})


def jail_script() -> str:
    return _JAIL_SCRIPT.replace("CERRS = {srcdir!r}", f"CERRS = {_ARCH!r}")


# --------------------------------------------------------------------------- #
# Invocation
# --------------------------------------------------------------------------- #
@dataclass
class JailCommand:
    """A prepared command line that runs the runner inside a jail."""

    argv: list[str]
    tier: str
    env: dict[str, str] = field(default_factory=dict)
    #: True when the filesystem guarantee is "read-only root"; False when the
    #: kernel refused it and only homes+mounts are contained.
    read_only_root: bool = False
    notes: list[str] = field(default_factory=list)


def build_jail_command(
    *,
    python: str,
    runner_path: str,
    workspace: str,
    jail_script_path: str,
    scratch_dir: str,
    config: JailConfig,
    limits: dict[str, object],
    child_env: dict[str, str],
    arch: str | None = None,
) -> JailCommand:
    """Compose the jail invocation for one run.

    ``workspace`` is the host directory holding the code under test; the jail
    bind-mounts it into ``scratch_dir/.hc-work`` before hiding ``/tmp``, so the
    child sees a stable path with no view of the host's temporary files.
    """
    tier = detect_isolation(config)
    if tier == ISOLATION_PROCESS:
        return JailCommand(
            argv=[python, "-I", "-B", runner_path],
            tier=tier,
            env=dict(child_env),
            read_only_root=False,
            notes=["kernel isolation unavailable; process guards only"],
        )

    jail_config = {
        "mask_paths": list(config.mask_paths),
        "root_flags": "ro",
        "seccomp": bool(config.seccomp and arch_supported(arch)),
        "scratch": scratch_dir,
        "source_workspace": workspace,
        "python": python,
        "limits": limits,
    }
    env = dict(child_env)
    env["JAIL_CONFIG"] = json_dumps(jail_config)
    env["JAIL_ARCH_TABLE"] = json_dumps(_ARCH)
    env["PATH"] = "/usr/local/bin:/usr/bin:/bin"

    if tier == ISOLATION_UNSHARE:
        argv = [which("unshare"), "-Urmpfn", "--mount-proc", "--kill-child"]
    elif tier == ISOLATION_BWRAP:
        argv = [
            which("bwrap"),
            "--unshare-all",
            "--die-with-parent",
            "--ro-bind", "/", "/",
            "--bind", workspace, workspace,
            "--proc", "/proc",
            "--dev", "/dev",
            "--chdir", workspace,
        ]
    else:  # nsjail
        argv = [
            which("nsjail"),
            "--quiet",
            "--mode", "o",
            "--user", "65534",
            "--group", "65534",
            "--bindmount_ro", "/:/",
            "--bindmount", f"{workspace}:{workspace}",
            "--disable_proc",
            "--cwd", workspace,
        ]
    argv += [python, "-I", "-B", jail_script_path]
    return JailCommand(
        argv=argv,
        tier=tier,
        env=env,
        read_only_root=(tier in {ISOLATION_CONTAINER, ISOLATION_BWRAP}) or _ro_root_supported(),
        notes=[] if tier == ISOLATION_UNSHARE else ["read-only root delegated to the tool"],
    )


def arch_supported(arch: str | None = None) -> bool:
    return (arch or os.uname().machine) in _ARCH


def _ro_root_supported() -> bool:
    """Whether an unprivileged namespace may remount ``/`` read-only here.

    Docker's default profile blocks the ``mount`` bitmap for locked remounts, so
    this is genuinely host-dependent — we ask the kernel once instead of
    guessing, and report the answer.
    """
    if "ro_root" in _PROBE_CACHE:
        return _PROBE_CACHE["ro_root"]
    script = (
        'mount --make-rslave / >/dev/null 2>&1; '
        'mount -o remount,ro,bind / >/dev/null 2>&1'
    )
    ok = False
    binary = which("unshare")
    if binary:
        try:
            completed = subprocess.run(
                [binary, "-Urmpfn", "--kill-child", "sh", "-c", script],
                capture_output=True,
                timeout=5,
                check=False,
            )
            ok = completed.returncode == 0
        except (OSError, subprocess.SubprocessError):
            ok = False
    _PROBE_CACHE["ro_root"] = ok
    return ok


def json_dumps(payload: dict[str, object]) -> str:
    import json

    return json.dumps(payload, separators=(",", ":"))


def python_executable() -> str:
    """A python path that survives the jail (masked homes, no $PATH trust)."""
    return os.path.realpath(sys.executable)


__all__ = [
    "ISOLATION_BWRAP",
    "ISOLATION_CONTAINER",
    "ISOLATION_PROCESS",
    "ISOLATION_UNSHARE",
    "KERNEL_TIERS",
    "JailCommand",
    "JailConfig",
    "arch_supported",
    "available_tiers",
    "build_jail_command",
    "detect_isolation",
    "has_bwrap",
    "has_nsjail",
    "jail_script",
    "python_executable",
    "seccomp_table",
    "unshare_works",
    "which",
]
