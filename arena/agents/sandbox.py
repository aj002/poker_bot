"""Process confinement applied by a worker to itself before it loads bot code.

Layers, outermost first (see ``docs/agent-sandbox.md`` for what each one does and
does not cover):

1. ``unshare`` (util-linux) starts the worker in fresh user, network, IPC, UTS,
   mount and PID namespaces as the unprivileged invoking user (no root mapping).
   The network namespace has only a down loopback interface. ``/proc`` is
   remounted so the worker sees only its own namespace's processes.
2. Resource limits (``setrlimit``): address space, CPU seconds, file size 0,
   open files, no core dumps, no locked memory, no POSIX message queues.
3. ``PR_SET_NO_NEW_PRIVS``.
4. Landlock (ABI >= 1): read-only access to an allowlist of directories (the
   Python runtime, site-packages and the staged bot bundle). Nothing is writable
   or executable. With ABI >= 4 TCP bind/connect are denied, and with ABI >= 6
   signals and abstract UNIX sockets outside the worker's domain are denied.
5. seccomp-BPF: an x86-64 deny list (socket creation, process creation, exec,
   ptrace, namespace and mount changes, kernel keyrings, BPF, io_uring, ...).

Everything here uses only the standard library and ``ctypes`` so it can run in
the worker before any untrusted import. Each function raises
:class:`SandboxError` rather than continuing with a weaker sandbox.
"""

from __future__ import annotations

import ctypes
import errno
import os
import platform
import struct
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field



class SandboxError(RuntimeError):
    pass


class _LazyLibc:
    """libc, loaded on first use so importing this module works where there is none (native Windows)."""

    _lib: ctypes.CDLL | None = None

    def __getattr__(self, name: str):
        if self._lib is None:
            if not sys.platform.startswith("linux"):
                raise SandboxError(
                    "the sandbox needs Linux (Landlock and seccomp); on Windows run the sandboxed "
                    "paths inside WSL2")
            lib = ctypes.CDLL(None, use_errno=True)
            lib.syscall.restype = ctypes.c_long
            lib.prctl.restype = ctypes.c_int
            self._lib = lib
        return getattr(self._lib, name)


_libc = _LazyLibc()


# --------------------------------------------------------------------- limits


@dataclass(frozen=True)
class Limits:
    address_space_bytes: int = 1 << 30
    cpu_seconds: int = 300
    open_files: int = 64
    threads: int = 8
    """Upper bound on threads per worker, enforced by the parent's NPROC limit (see docs)."""

    def as_dict(self) -> dict[str, int]:
        return {
            "address_space_bytes": self.address_space_bytes,
            "cpu_seconds": self.cpu_seconds,
            "open_files": self.open_files,
            "threads": self.threads,
            "file_size_bytes": 0,
            "core_bytes": 0,
            "memlock_bytes": 0,
            "msgqueue_bytes": 0,
        }


def apply_rlimits(limits: Limits) -> None:
    import resource

    pairs = [
        (resource.RLIMIT_AS, limits.address_space_bytes),
        (resource.RLIMIT_CPU, limits.cpu_seconds),
        (resource.RLIMIT_FSIZE, 0),
        (resource.RLIMIT_NOFILE, limits.open_files),
        (resource.RLIMIT_CORE, 0),
        (resource.RLIMIT_MEMLOCK, 0),
        (resource.RLIMIT_MSGQUEUE, 0),
    ]
    for res, value in pairs:
        try:
            resource.setrlimit(res, (value, value))
        except (ValueError, OSError) as exc:
            raise SandboxError(f"setrlimit({res}) failed: {exc}") from None


# ----------------------------------------------------------------- prctl bits

PR_SET_PDEATHSIG = 1
PR_SET_DUMPABLE = 4
PR_SET_NO_NEW_PRIVS = 38
PR_SET_TIMERSLACK = 29
PR_SET_SECCOMP = 22
SECCOMP_MODE_FILTER = 2


def _prctl(option: int, arg2: int = 0) -> None:
    if _libc.prctl(option, ctypes.c_ulong(arg2), 0, 0, 0) != 0:
        err = ctypes.get_errno()
        raise SandboxError(f"prctl({option}) failed: {os.strerror(err)}")


def set_no_new_privs() -> None:
    _prctl(PR_SET_NO_NEW_PRIVS, 1)


def set_not_dumpable() -> None:
    _prctl(PR_SET_DUMPABLE, 0)


def set_timer_slack(ns: int) -> None:
    """Shorter timer slack makes ``select`` wake closer to the requested time."""
    _prctl(PR_SET_TIMERSLACK, ns)


# ------------------------------------------------------------------- landlock

_SYS_LANDLOCK_CREATE_RULESET = 444
_SYS_LANDLOCK_ADD_RULE = 445
_SYS_LANDLOCK_RESTRICT_SELF = 446
_LANDLOCK_CREATE_RULESET_VERSION = 1
_LANDLOCK_RULE_PATH_BENEATH = 1

FS_EXECUTE = 1 << 0
FS_WRITE_FILE = 1 << 1
FS_READ_FILE = 1 << 2
FS_READ_DIR = 1 << 3
_FS_BY_ABI = {1: (1 << 13) - 1, 2: (1 << 14) - 1, 3: (1 << 15) - 1, 4: (1 << 15) - 1, 5: (1 << 16) - 1}
_NET_ALL = (1 << 0) | (1 << 1)  # BIND_TCP | CONNECT_TCP, ABI >= 4
_SCOPE_ALL = (1 << 0) | (1 << 1)  # ABSTRACT_UNIX_SOCKET | SIGNAL, ABI >= 6
_FILE_ONLY_ACCESS = FS_EXECUTE | FS_WRITE_FILE | FS_READ_FILE | (1 << 14) | (1 << 15)


def landlock_abi() -> int:
    v = _libc.syscall(_SYS_LANDLOCK_CREATE_RULESET, None, ctypes.c_size_t(0), _LANDLOCK_CREATE_RULESET_VERSION)
    return int(v) if v > 0 else 0


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64), ("handled_access_net", ctypes.c_uint64),
                ("scoped", ctypes.c_uint64)]


class _PathBeneathAttr(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


@dataclass
class LandlockReport:
    abi: int
    fs: bool
    net: bool
    scoped: bool
    read_paths: list[str] = field(default_factory=list)


def apply_landlock(read_paths: Iterable[str]) -> LandlockReport:
    """Deny every filesystem access except reading beneath ``read_paths``."""
    abi = landlock_abi()
    if abi < 1:
        raise SandboxError("Landlock is not available on this kernel")
    handled_fs = _FS_BY_ABI.get(abi, (1 << 16) - 1)
    attr = _RulesetAttr(handled_fs, _NET_ALL if abi >= 4 else 0, _SCOPE_ALL if abi >= 6 else 0)
    size = ctypes.sizeof(_RulesetAttr) if abi >= 6 else (16 if abi >= 4 else 8)
    fd = _libc.syscall(_SYS_LANDLOCK_CREATE_RULESET, ctypes.byref(attr), ctypes.c_size_t(size), 0)
    if fd < 0:
        raise SandboxError(f"landlock_create_ruleset failed: {os.strerror(ctypes.get_errno())}")
    used: list[str] = []
    try:
        for path in read_paths:
            if not os.path.exists(path):
                continue
            access = FS_READ_FILE | FS_READ_DIR
            if not os.path.isdir(path):
                access &= _FILE_ONLY_ACCESS
            pfd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = _PathBeneathAttr(access, pfd)
                if _libc.syscall(_SYS_LANDLOCK_ADD_RULE, fd, _LANDLOCK_RULE_PATH_BENEATH, ctypes.byref(rule), 0) != 0:
                    raise SandboxError(f"landlock_add_rule({path}) failed: {os.strerror(ctypes.get_errno())}")
            finally:
                os.close(pfd)
            used.append(path)
        if _libc.syscall(_SYS_LANDLOCK_RESTRICT_SELF, fd, 0) != 0:
            raise SandboxError(f"landlock_restrict_self failed: {os.strerror(ctypes.get_errno())}")
    finally:
        os.close(fd)
    return LandlockReport(abi=abi, fs=True, net=abi >= 4, scoped=abi >= 6, read_paths=used)


# -------------------------------------------------------------------- seccomp

_AUDIT_ARCH_X86_64 = 0xC000003E
_X32_SYSCALL_BIT = 0x40000000
_RET_ALLOW = 0x7FFF0000
_RET_KILL_PROCESS = 0x80000000
_RET_ERRNO = 0x00050000
_LD_W_ABS = 0x20
_JEQ_K = 0x15
_JSET_K = 0x45
_JGE_K = 0x35
_RET_K = 0x06
_CLONE_THREAD = 0x00010000

# x86-64 syscall numbers denied with EPERM (ENOSYS for clone3 so glibc falls back
# to clone, which is then allowed only for threads).
DENIED_SYSCALLS: dict[str, int] = {
    "socket": 41, "socketpair": 53, "connect": 42, "bind": 49, "listen": 50, "accept": 43, "accept4": 288,
    "fork": 57, "vfork": 58, "execve": 59, "execveat": 322,
    "ptrace": 101, "process_vm_readv": 310, "process_vm_writev": 311, "kcmp": 312, "pidfd_getfd": 438,
    "pidfd_open": 434,
    "mount": 165, "umount2": 166, "pivot_root": 155, "chroot": 161, "unshare": 272, "setns": 308,
    "fsopen": 430, "fsmount": 432, "move_mount": 429, "open_tree": 428, "mount_setattr": 442,
    "bpf": 321, "perf_event_open": 298, "userfaultfd": 323,
    "io_uring_setup": 425, "io_uring_enter": 426, "io_uring_register": 427,
    "keyctl": 250, "add_key": 248, "request_key": 249,
    "kexec_load": 246, "kexec_file_load": 320, "init_module": 175, "finit_module": 313, "delete_module": 176,
    "reboot": 169, "swapon": 167, "swapoff": 168, "syslog": 103, "acct": 163,
    "name_to_handle_at": 303, "open_by_handle_at": 304, "personality": 135,
    "mq_open": 240, "msgget": 68, "semget": 64, "shmget": 29,
}
_SYS_CLONE = 56
_SYS_CLONE3 = 435


def _insn(code: int, jt: int, jf: int, k: int) -> bytes:
    return struct.pack("HBBI", code, jt, jf, k)


def seccomp_program() -> bytes:
    eperm = _RET_ERRNO | errno.EPERM
    prog = [
        _insn(_LD_W_ABS, 0, 0, 4),  # arch
        _insn(_JEQ_K, 1, 0, _AUDIT_ARCH_X86_64),
        _insn(_RET_K, 0, 0, _RET_KILL_PROCESS),
        _insn(_LD_W_ABS, 0, 0, 0),  # nr
        _insn(_JGE_K, 0, 1, _X32_SYSCALL_BIT),
        _insn(_RET_K, 0, 0, _RET_KILL_PROCESS),
        _insn(_JEQ_K, 0, 1, _SYS_CLONE3),
        _insn(_RET_K, 0, 0, _RET_ERRNO | errno.ENOSYS),
        _insn(_JEQ_K, 0, 4, _SYS_CLONE),
        _insn(_LD_W_ABS, 0, 0, 16),  # args[0] low word: clone flags
        _insn(_JSET_K, 0, 1, _CLONE_THREAD),
        _insn(_RET_K, 0, 0, _RET_ALLOW),
        _insn(_RET_K, 0, 0, eperm),
    ]
    for nr in sorted(DENIED_SYSCALLS.values()):
        prog.append(_insn(_JEQ_K, 0, 1, nr))
        prog.append(_insn(_RET_K, 0, 0, eperm))
    prog.append(_insn(_RET_K, 0, 0, _RET_ALLOW))
    return b"".join(prog)


class _SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]


def apply_seccomp() -> None:
    if platform.machine() != "x86_64":
        raise SandboxError(f"the seccomp filter is written for x86_64, not {platform.machine()}")
    program = seccomp_program()
    buf = ctypes.create_string_buffer(program, len(program))
    fprog = _SockFprog(len(program) // 8, ctypes.cast(buf, ctypes.c_void_p))
    if _libc.prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, ctypes.byref(fprog), 0, 0) != 0:
        raise SandboxError(f"seccomp filter failed: {os.strerror(ctypes.get_errno())}")


# ------------------------------------------------------------------ namespaces

UNSHARE_ARGS: tuple[str, ...] = (
    "--map-current-user", "--net", "--ipc", "--uts", "--pid", "--fork", "--kill-child",
    "--mount", "--mount-proc",
)


def namespace_report() -> dict[str, object]:
    """What the worker can observe about its own namespaces (diagnostics only)."""
    out: dict[str, object] = {"uid": os.getuid(), "pid": os.getpid()}
    try:
        with open("/proc/self/net/dev", encoding="ascii") as f:
            out["net_interfaces"] = sorted(line.split(":")[0].strip() for line in f.readlines()[2:])
    except OSError as exc:
        out["net_interfaces"] = f"unavailable: {exc.strerror}"
    try:
        out["visible_pids"] = len([p for p in os.listdir("/proc") if p.isdigit()])
    except OSError as exc:
        out["visible_pids"] = f"unavailable: {exc.strerror}"
    return out
