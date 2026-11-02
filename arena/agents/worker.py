"""Parent side of the agent runtime: deadlines, fallbacks, pooled sandboxed workers.

Decision deadline (spec sections 6.3 and 7)
    The clock starts immediately before the observation is serialized and sent. A
    reply is on time when the parent has *completely* received it at
    ``received_ns <= dispatched_ns + budget_ns`` (default budget 1,000,000 ns), so
    serialization and IPC count against the bot. ``received_ns`` is taken right
    after the read that completed the frame, which is conservative: if the parent
    wakes late, a reply that was already queued is judged late. An on-time reply
    is accepted even when it is validated and applied after the deadline.

Fallback
    A missing, late, malformed, oversized, crashed, raising or illegal reply
    consumes the decision once with check, otherwise fold
    (:meth:`arena.table.service.Table.apply_fallback`), with a reason code. There is
    no retry. Late replies are drained and discarded. A worker that still has not
    answered when its seat is next asked (after ``stale_grace_ns``) is killed and
    replaced by a fresh worker restored from the last committed state.

Commit rule
    A bot's returned state becomes the committed state only after the parent has
    accepted the action that came with it (on time, valid schema, legal). The
    worker is told which reply was committed in the next request; restores receive
    the parent's copy of the committed state.

Timeout records
    Every timeout appends a :class:`TimeoutRecord` to ``SeatStats.timeouts``: the
    parent-observed round trip and, once the late reply has been drained, the
    worker's own parse, decide and CPU times for that request. With
    ``timeout_inputs=N`` an explicit-state seat also keeps the request's observation
    and committed state for its first N timeouts, so :func:`replay_decision` can
    replay the exact decision in a fresh worker and tell a slow bot from a stalled
    host.

Parent garbage collection
    The parent's collector is disabled from dispatch until the reply is read (or the
    deadline passes), so a collection in the harness process never counts against a
    bot; :func:`play_table` also freezes the heap that exists when a table starts, so
    collections during the table only scan objects the table created. It unfreezes
    on return, which releases everything frozen, including anything frozen before
    the call.

Everything that touches time goes through :class:`Clock`, and the seat logic talks
to workers through :class:`Channel`, so tests drive it with :class:`FakeClock`
and scripted channels.
"""

from __future__ import annotations

import fcntl
import gc
import hashlib
import json
import os
import queue
import select
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from arena.agents import sandbox
from arena.agents.protocol import DETAIL_CHARS, FrameError, FrameReader, encode_frame
from arena.agents.sdk import (
    MAX_STATE_BYTES,
    BotInfo,
    StateError,
    TableInfo,
    decode_state,
    encode_state,
    require_supported,
    split_note,
)
from arena.domain.observations import Observation

DEFAULT_BUDGET_NS = 1_000_000
DEFAULT_STALE_GRACE_NS = 20_000_000
DEFAULT_LOAD_TIMEOUT_NS = 10_000_000_000
DEFAULT_MAX_RESTARTS = 25
LOG_KEEP_BYTES = 4096
WAKE_FRAME = encode_frame({"t": "wake"})
_F_SETPIPE_SZ = 1031

# Reason codes recorded with ActionFallback. Public: the code. Private: the payload.
FALLBACK_REASONS = (
    "timeout",            # no complete reply by the deadline (including late replies)
    "worker_crash",       # the worker exited or closed its pipe
    "bot_exception",      # decide() raised
    "bot_memory_error",   # decide() raised MemoryError (address-space limit)
    "invalid_output",     # reply is not a valid frame, or the action is not JSON
    "oversized_output",   # reply frame over the size cap
    "state_invalid",      # returned state is not JSON
    "state_too_large",    # returned state over 64 KiB
    "invalid_schema",     # action fails the Action schema
    "illegal_action",     # action is not in the legal set
    "worker_unavailable", # no worker could be started or restored for this decision
)


# ---------------------------------------------------------------------- clocks


class Clock(Protocol):
    def now_ns(self) -> int: ...


class MonotonicClock:
    def now_ns(self) -> int:
        return time.perf_counter_ns()


class FakeClock:
    """Manually advanced clock for deterministic deadline tests."""

    def __init__(self, start_ns: int = 1_000_000_000) -> None:
        self.t = start_ns

    def now_ns(self) -> int:
        return self.t

    def advance(self, ns: int) -> None:
        self.t += ns

    def set(self, t: int) -> None:
        self.t = max(self.t, t)


def on_time(received_ns: int, deadline_ns: int) -> bool:
    """The only deadline rule: a completely received reply at the deadline is on time."""
    return received_ns <= deadline_ns


# -------------------------------------------------------------------- channels


class ChannelClosed(Exception):
    pass


@dataclass
class ReceivedFrame:
    meta: dict[str, Any]
    blob: bytes
    received_ns: int


class Channel(Protocol):
    """One worker connection, as the seat logic sees it."""

    def send(self, frame: bytes) -> None: ...

    def recv(self, deadline_ns: int | None, spin_ns: int = 0) -> ReceivedFrame | None:
        """The next complete frame, or None once ``deadline_ns`` passes without one.

        Raises :class:`ChannelClosed` on EOF and ``FrameError`` on a bad frame.
        """

    def kill(self) -> None: ...

    def log_excerpt(self) -> str: ...


class WorkerProcess:
    """A pre-started, sandboxed worker process (implements :class:`Channel`)."""

    def __init__(self, proc: subprocess.Popen, to_fd: int, from_fd: int, log_fd: int, clock: Clock) -> None:
        self.proc = proc
        self.to_fd = to_fd
        self.from_fd = from_fd
        self.log_fd = log_fd
        self.clock = clock
        self.reader = FrameReader()
        self.last_read_ns = 0
        self.sandbox_report: dict[str, Any] = {}
        self.log = bytearray()
        self.closed = False
        os.set_blocking(from_fd, False)
        os.set_blocking(log_fd, False)

    def send(self, frame: bytes) -> None:
        view = memoryview(frame)
        try:
            while view:
                n = os.write(self.to_fd, view)
                view = view[n:]
        except (BrokenPipeError, OSError):
            raise ChannelClosed("worker pipe closed") from None

    def recv(self, deadline_ns: int | None, spin_ns: int = 0) -> ReceivedFrame | None:
        """Wait for a frame. With ``spin_ns`` the parent polls without sleeping for up to
        that long first, so the reply does not also pay for waking the parent's CPU."""
        spin_until = self.clock.now_ns() + spin_ns if spin_ns else 0
        while True:
            frame = self.reader.next_frame()
            if frame is not None:
                return ReceivedFrame(frame.meta, frame.blob, self.last_read_ns)
            now = self.clock.now_ns()
            if now < spin_until and (deadline_ns is None or now < deadline_ns):
                ready = True
            else:
                timeout = None if deadline_ns is None else max(0, deadline_ns - now) / 1e9
                ready = bool(select.select([self.from_fd], [], [], timeout)[0])
            if not ready:
                if deadline_ns is not None and self.clock.now_ns() >= deadline_ns:
                    return None
                continue
            try:
                data = os.read(self.from_fd, 1 << 20)
            except BlockingIOError:
                continue
            except OSError:
                raise ChannelClosed("worker pipe error") from None
            self.last_read_ns = self.clock.now_ns()
            if not data:
                raise ChannelClosed("worker closed its pipe")
            self.reader.feed(data)

    def drain_log(self) -> None:
        try:
            while True:
                data = os.read(self.log_fd, 65536)
                if not data:
                    return
                if len(self.log) < LOG_KEEP_BYTES:
                    self.log += data[: LOG_KEEP_BYTES - len(self.log)]
        except (BlockingIOError, OSError):
            return

    def log_excerpt(self) -> str:
        self.drain_log()
        return self.log.decode("utf-8", "replace")

    def kill(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            os.killpg(self.proc.pid, 9)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        self.drain_log()
        for fd in (self.to_fd, self.from_fd, self.log_fd):
            try:
                os.close(fd)
            except OSError:
                pass


# ------------------------------------------------------------------- artifacts


@dataclass(frozen=True)
class Artifact:
    """A staged, read-only bot bundle: one file, or a directory with an entry file."""

    name: str
    entry: Path
    root: Path
    sha256: str


def artifact_hash(src: Path) -> str:
    h = hashlib.sha256(b"arena-bot-artifact-v1")
    files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    base = src.parent if src.is_file() else src
    for p in files:
        rel = p.relative_to(base).as_posix().encode()
        data = p.read_bytes()
        h.update(len(rel).to_bytes(4, "big") + rel + len(data).to_bytes(8, "big") + data)
    return h.hexdigest()


def stage_artifact(path: str | Path, staging_root: Path, entry: str = "bot.py") -> Artifact:
    """Copy a bot into ``staging_root/<sha256>/`` and make the copy read-only."""
    src = Path(path).resolve()
    if not src.exists():
        raise FileNotFoundError(src)
    digest = artifact_hash(src)
    dest = staging_root / digest
    entry_name = src.name if src.is_file() else entry
    if not dest.exists():
        tmp = Path(tempfile.mkdtemp(dir=staging_root, prefix=".stage-"))
        if src.is_file():
            shutil.copy2(src, tmp / src.name)
        else:
            shutil.copytree(src, tmp, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for p in [tmp, *tmp.rglob("*")]:
            mode = stat.S_IMODE(p.stat().st_mode)
            p.chmod(mode & ~0o222)
        try:
            tmp.rename(dest)
        except OSError:
            shutil.rmtree(tmp, ignore_errors=True)
    if not (dest / entry_name).is_file():
        raise FileNotFoundError(f"artifact entry {entry_name} missing in {src}")
    return Artifact(name=src.stem if src.is_file() else src.name, entry=dest / entry_name, root=dest, sha256=digest)


def runtime_read_paths() -> list[str]:
    """Read-only roots a worker needs: the Python installation and site-packages."""
    roots = {os.path.realpath(p) for p in (sys.base_prefix, sys.prefix, sys.base_exec_prefix)}
    for p in sys.path:
        if p and "site-packages" in p and os.path.isdir(p):
            roots.add(os.path.realpath(p))
    roots.add(os.path.realpath(os.path.dirname(os.path.realpath(sys.executable))))
    repo = os.path.realpath(Path(__file__).resolve().parents[2])
    return sorted(r for r in roots if not (repo == r or repo.startswith(r.rstrip("/") + "/")))


# ------------------------------------------------------------------------ pool


class WorkerStartError(RuntimeError):
    pass


class WorkerPool:
    """Pre-started sandboxed workers. Each worker serves one bot context for one table.

    A table takes ready workers from the pool; a background thread starts
    replacements so process start-up and Python imports stay off the table's
    critical path. Workers are never reused across tables (fresh context per table).
    """

    def __init__(
        self,
        size: int = 8,
        *,
        limits: sandbox.Limits | None = None,
        clock: Clock | None = None,
        start_timeout_s: float = 20.0,
        python: str | None = None,
    ) -> None:
        if shutil.which("unshare") is None:
            raise WorkerStartError("util-linux 'unshare' is required for sandboxed workers")
        self.size = size
        self.limits = limits or sandbox.Limits()
        self.clock = clock or MonotonicClock()
        self.start_timeout_s = start_timeout_s
        self.python = python or sys.executable
        self.staging_root = Path(tempfile.mkdtemp(prefix="arena-bots-"))
        self.runtime_paths = runtime_read_paths()
        self._ready: queue.Queue[WorkerProcess | Exception] = queue.Queue()
        self._lock = threading.Lock()
        self._starting = 0
        self._closed = False
        self._wake = threading.Event()
        self.started = 0
        self._filler = threading.Thread(target=self._fill, name="arena-worker-pool", daemon=True)
        self._filler.start()

    # -- staging
    def stage(self, path: str | Path) -> Artifact:
        return stage_artifact(path, self.staging_root)

    # -- process start
    def spawn(self) -> WorkerProcess:
        to_r, to_w = os.pipe()
        from_r, from_w = os.pipe()
        log_r, log_w = os.pipe()
        for fd in (to_w, from_w):
            try:
                fcntl.fcntl(fd, _F_SETPIPE_SZ, 1 << 20)
            except OSError:
                pass
        cfg = json.dumps({
            "runtime_paths": self.runtime_paths,
            "staging_root": str(self.staging_root),
            "limits": {"address_space_bytes": self.limits.address_space_bytes,
                       "cpu_seconds": self.limits.cpu_seconds,
                       "open_files": self.limits.open_files,
                       "threads": self.limits.threads},
        })
        argv = ["unshare", *sandbox.UNSHARE_ARGS, self.python, "-s", "-P", "-B", "-m", "arena.agents._worker_main",
                "--in-fd", str(to_r), "--out-fd", str(from_w), "--config", cfg]
        env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONHASHSEED": "0"}
        try:
            proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log_w, stderr=log_w, env=env, cwd="/",
                                    pass_fds=(to_r, from_w), start_new_session=True, close_fds=True)
        finally:
            for fd in (to_r, from_w, log_w):
                os.close(fd)
        worker = WorkerProcess(proc, to_w, from_r, log_r, self.clock)
        try:
            frame = worker.recv(self.clock.now_ns() + int(self.start_timeout_s * 1e9))
        except (ChannelClosed, FrameError) as exc:
            worker.kill()
            raise WorkerStartError(f"worker failed to start: {exc}; log: {worker.log_excerpt()[:500]}") from None
        if frame is None or frame.meta.get("t") != "ready":
            worker.kill()
            detail = frame.meta if frame is not None else "timeout"
            raise WorkerStartError(f"worker did not become ready: {detail}; log: {worker.log_excerpt()[:500]}")
        worker.sandbox_report = frame.meta.get("sandbox", {})
        self.started += 1
        return worker

    def _fill(self) -> None:
        while not self._closed:
            with self._lock:
                need = self.size - self._ready.qsize() - self._starting
                if need > 0:
                    self._starting += 1
            if need <= 0:
                self._wake.wait(0.05)
                self._wake.clear()
                continue
            try:
                item: WorkerProcess | Exception = self.spawn()
            except Exception as exc:  # surfaced to the next acquire()
                item = exc
            with self._lock:
                self._starting -= 1
            if self._closed and isinstance(item, WorkerProcess):
                item.kill()
                return
            self._ready.put(item)

    def acquire(self) -> WorkerProcess:
        try:
            item = self._ready.get(timeout=self.start_timeout_s * 2)
        except queue.Empty:
            raise WorkerStartError("no worker became ready in time") from None
        self._wake.set()
        if isinstance(item, Exception):
            raise item
        return item

    def close(self) -> None:
        self._closed = True
        self._wake.set()
        self._filler.join(timeout=self.start_timeout_s + 5)
        while True:
            try:
                item = self._ready.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, WorkerProcess):
                item.kill()
        shutil.rmtree(self.staging_root, ignore_errors=True)

    def __enter__(self) -> WorkerPool:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# --------------------------------------------------------------- seat runtime


@dataclass
class Decision:
    action: Any | None
    fallback: str | None
    rejected: Any = None
    e2e_ns: int | None = None
    worker_wall_ns: int | None = None
    worker_cpu_ns: int | None = None
    worker_parse_ns: int | None = None
    worker_parse_gc: int | None = None
    state_bytes: int | None = None
    note: str | None = None
    """The bot's private reasoning note (:func:`arena.agents.sdk.with_note`), stripped from ``action``."""


@dataclass
class TimeoutRecord:
    """One timed-out decision, with whatever the parent learned about it afterwards."""

    decision_index: int
    seq: int
    budget_ns: int
    e2e_ns: int
    """Parent-observed time from dispatch to the timeout verdict."""
    outcome: str = "unsettled"
    """``late_reply`` (the reply arrived after the deadline), ``unresponsive`` (no reply
    within the stale grace; the worker was replaced), ``worker_crash``, or
    ``unsettled`` (the table ended before the seat was asked again)."""
    late_by_ns: int | None = None
    """How late the reply arrived; only when it was read while the parent was still waiting."""
    drained_after_ns: int | None = None
    """For a late reply drained at the seat's next turn: deadline to the read, an upper
    bound on its lateness (the reply may have sat in the pipe meanwhile)."""
    reply_kind: str | None = None
    """``action`` or ``error`` for a late reply."""
    worker_parse_ns: int | None = None
    worker_wall_ns: int | None = None
    worker_cpu_ns: int | None = None
    worker_parse_cpu_ns: int | None = None
    worker_parse_gc: int | None = None
    """Garbage collections that ran inside the observation parse."""
    worker_threads: int | None = None
    """Python threads alive in the worker when it replied (1: the bot started none)."""
    observation: bytes | None = None
    """The request's observation JSON (only with ``timeout_inputs``, explicit state)."""
    committed: bytes | None = None
    """The committed state the worker decided from (with ``observation``)."""
    deadline_ns: int = 0
    """Clock value of the deadline (not reported)."""

    @property
    def replayable(self) -> bool:
        return self.observation is not None and self.committed is not None

    def as_dict(self) -> dict[str, Any]:
        return {"decision_index": self.decision_index, "seq": self.seq, "budget_ns": self.budget_ns,
                "e2e_ns": self.e2e_ns, "outcome": self.outcome, "late_by_ns": self.late_by_ns,
                "drained_after_ns": self.drained_after_ns,
                "reply_kind": self.reply_kind, "worker_parse_ns": self.worker_parse_ns,
                "worker_wall_ns": self.worker_wall_ns, "worker_cpu_ns": self.worker_cpu_ns,
                "worker_parse_cpu_ns": self.worker_parse_cpu_ns, "worker_parse_gc": self.worker_parse_gc,
                "worker_threads": self.worker_threads}


@dataclass
class Restart:
    decision_index: int
    cause: str
    restored: bool
    """True when the new worker resumed from the committed explicit state."""


@dataclass
class SeatStats:
    decisions: int = 0
    fallbacks: Counter = field(default_factory=Counter)
    restarts: list[Restart] = field(default_factory=list)
    late_discarded: int = 0
    max_state_bytes: int = 0
    e2e_ns: list[int] = field(default_factory=list)
    worker_wall_ns: list[int] = field(default_factory=list)
    worker_cpu_ns: list[int] = field(default_factory=list)
    worker_parse_ns: list[int] = field(default_factory=list)
    """One entry per decision, in order (index = ``TimeoutRecord.decision_index``); -1
    when the worker reported no parse time (timeouts, crashes) or a garbage
    collection ran inside the parse."""
    timeouts: list[TimeoutRecord] = field(default_factory=list)
    infrastructure: list[tuple[int, str]] = field(default_factory=list)
    """(decision index, event) for failures that were not the bot's doing:
    ``worker_start_failed`` (no worker could be started) or the cause given to
    :meth:`WorkerSeat.kill_worker`."""

    @property
    def restorable(self) -> bool:
        return all(r.restored for r in self.restarts)


class SeatAgent(Protocol):
    seat: int
    info: BotInfo
    artifact_sha256: str
    stats: SeatStats

    def start(self) -> BotInfo: ...

    def restore(self, state: bytes) -> None: ...

    def prepare(self) -> None: ...

    def decide(self, observation: Observation) -> Decision: ...

    def accept(self) -> None: ...

    def reject(self) -> None: ...

    def close(self) -> str: ...


class AgentLoadError(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class WorkerSeat:
    """One seat's bot behind a worker channel, with deadline, fallback and restore."""

    def __init__(
        self,
        seat: int,
        artifact: Artifact,
        table_info: TableInfo,
        spawn: Callable[[], Channel],
        *,
        clock: Clock | None = None,
        budget_ns: int = DEFAULT_BUDGET_NS,
        stale_grace_ns: int = DEFAULT_STALE_GRACE_NS,
        load_timeout_ns: int = DEFAULT_LOAD_TIMEOUT_NS,
        max_restarts: int = DEFAULT_MAX_RESTARTS,
        spin_ns: int | None = None,
        timeout_inputs: int = 0,
    ) -> None:
        self.seat = seat
        self.timeout_inputs = timeout_inputs
        self._open_timeout: TimeoutRecord | None = None
        self.artifact = artifact
        self.artifact_sha256 = artifact.sha256
        self.table_info = table_info
        self.spawn = spawn
        self.clock = clock or MonotonicClock()
        self.budget_ns = budget_ns
        self.stale_grace_ns = stale_grace_ns
        self.load_timeout_ns = load_timeout_ns
        self.max_restarts = max_restarts
        self.spin_ns = budget_ns if spin_ns is None else spin_ns
        self.stats = SeatStats()
        self.channel: Channel | None = None
        self.info: BotInfo | None = None  # type: ignore[assignment]
        self.committed: bytes = b"null"
        self.committed_history: list[bytes] = []
        self.seq = 0
        self.commit_seq: int | None = None
        self.pending: tuple[int, bytes] | None = None
        self.stale_seq: int | None = None
        self.lost_cause = "worker_crash"
        self.logs: list[str] = []

    # -- lifecycle
    def _load(self, restore: bool) -> Channel:
        channel = self.spawn()
        meta = {"t": "load", "bot_path": str(self.artifact.entry), "bot_dir": str(self.artifact.root),
                "table": self.table_info.model_dump(mode="json"), "restore": restore}
        try:
            channel.send(encode_frame(meta, self.committed if restore else b""))
            frame = channel.recv(self.clock.now_ns() + self.load_timeout_ns)
        except (ChannelClosed, FrameError) as exc:
            self._discard(channel)
            raise AgentLoadError("worker_crash", str(exc)[:DETAIL_CHARS]) from None
        if frame is None:
            self._discard(channel)
            raise AgentLoadError("load_timeout", "bot did not finish loading in time")
        if frame.meta.get("t") != "loaded":
            self._discard(channel)
            raise AgentLoadError(str(frame.meta.get("code", "bot_load_error")), str(frame.meta.get("detail", ""))[:DETAIL_CHARS])
        try:
            info = BotInfo.from_dict(frame.meta["info"])
        except (KeyError, TypeError, ValueError):
            self._discard(channel)
            raise AgentLoadError("bot_load_error", "malformed bot description") from None
        if self.info is not None and info != self.info:
            self._discard(channel)
            raise AgentLoadError("bot_load_error", "bot description changed on restart")
        self.info = info
        if info.state_mode == "explicit" and not restore:
            try:
                self.committed = _checked_state(frame.blob)
            except StateError as exc:
                self._discard(channel)
                raise AgentLoadError(exc.code, str(exc)) from None
            self.committed_history.append(self.committed)
        return channel

    def start(self) -> BotInfo:
        """Load the bot and check its ruleset before the table starts."""
        self.channel = self._load(restore=False)
        assert self.info is not None
        try:
            require_supported(self.info, self.table_info.ruleset_id)
        except Exception:
            self.close()
            raise
        return self.info

    def _discard(self, channel: Channel) -> None:
        excerpt = channel.log_excerpt()
        if excerpt:
            self.logs.append(excerpt)
        channel.kill()

    def _replace(self, cause: str) -> bool:
        self.lost_cause = "worker_crash"
        if self.channel is not None:
            self._discard(self.channel)
            self.channel = None
        self.pending = None
        self.stale_seq = None
        if len(self.stats.restarts) >= self.max_restarts:
            return False
        restored = self.info is not None and self.info.state_mode == "explicit"
        self.stats.restarts.append(Restart(self.stats.decisions, cause, restored))
        try:
            self.channel = self._load(restore=True)
        except WorkerStartError:
            self.stats.infrastructure.append((self.stats.decisions, "worker_start_failed"))
            self.channel = None
            return False
        except AgentLoadError:
            self.channel = None
            return False
        self.commit_seq = None
        return True

    def restore(self, state: bytes) -> None:
        """Continue from a checkpointed explicit state, e.g. after the server restarted."""
        self.committed = _checked_state(state)
        self.committed_history.append(self.committed)
        if self.channel is not None:
            self._discard(self.channel)
        self.channel = self._load(restore=True)
        self.commit_seq = None

    def kill_worker(self, cause: str = "infrastructure_kill") -> None:
        """Kill the worker between decisions, as an infrastructure failure would."""
        self.stats.infrastructure.append((self.stats.decisions, cause))
        if self.channel is not None:
            self._discard(self.channel)
            self.channel = None
            self.lost_cause = cause

    # -- decisions
    def _settle_stale(self) -> None:
        if self.stale_seq is None or self.channel is None:
            return
        record, self._open_timeout = self._open_timeout, None
        deadline = self.clock.now_ns() + self.stale_grace_ns
        try:
            while True:
                frame = self.channel.recv(deadline)
                if frame is None:
                    if record is not None:
                        record.outcome = "unresponsive"
                    self._replace("unresponsive")
                    return
                self.stats.late_discarded += 1
                if frame.meta.get("seq") == self.stale_seq:
                    if record is not None:
                        _note_reply(record, frame, drained=True)
                    self.stale_seq = None
                    return
        except (ChannelClosed, FrameError):
            if record is not None:
                record.outcome = "worker_crash"
            self._replace("worker_crash")

    def _record_timeout(self, seq: int, e2e_ns: int, deadline_ns: int, payload: bytes,
                        committed: bytes) -> TimeoutRecord:
        record = TimeoutRecord(self.stats.decisions - 1, seq, self.budget_ns, e2e_ns, deadline_ns=deadline_ns)
        kept = sum(1 for r in self.stats.timeouts if r.replayable)
        if kept < self.timeout_inputs and self.info is not None and self.info.state_mode == "explicit":
            record.observation, record.committed = payload, committed
        self.stats.timeouts.append(record)
        return record

    def prepare(self) -> None:
        """Called when this seat is to act, before the observation is built.

        Settles any late reply, restores a dead worker, and sends a wake-up notice so
        the worker's CPU is awake when the request arrives. The deadline clock has
        not started yet and the notice carries no game information.
        """
        self._settle_stale()
        if self.channel is None:
            return
        try:
            self.channel.send(WAKE_FRAME)
        except ChannelClosed:
            self._replace_later()

    def decide(self, observation: Observation) -> Decision:
        self.stats.decisions += 1
        self.pending = None
        self._settle_stale()
        if self.channel is None and not self._replace(self.lost_cause):
            return Decision(None, "worker_unavailable")
        channel = self.channel
        assert channel is not None
        self.seq += 1
        seq = self.seq
        committed = self.committed
        gc_enabled = gc.isenabled()
        gc.disable()
        try:
            dispatched = self.clock.now_ns()
            deadline = dispatched + self.budget_ns
            payload = observation.model_dump_json().encode()
            meta = json.dumps({"t": "decide", "seq": seq, "commit": self.commit_seq}, separators=(",", ":")).encode()
            try:
                channel.send(encode_frame(meta, payload))
                while True:
                    frame = channel.recv(deadline, spin_ns=self.spin_ns)
                    if frame is None or frame.meta.get("seq") == seq:
                        break
                    self.stats.late_discarded += 1
            except ChannelClosed:
                self._replace_later()
                return Decision(None, "worker_crash", e2e_ns=self.clock.now_ns() - dispatched)
            except FrameError as exc:
                self._replace_later()
                return Decision(None, exc.code, e2e_ns=self.clock.now_ns() - dispatched)
        finally:
            if gc_enabled:
                gc.enable()
        if frame is None:
            self.stale_seq = seq
            e2e = self.clock.now_ns() - dispatched
            self._open_timeout = self._record_timeout(seq, e2e, deadline, payload, committed)
            return Decision(None, "timeout", e2e_ns=e2e)
        e2e = frame.received_ns - dispatched
        m = frame.meta
        timing = {k: m.get(k) if isinstance(m.get(k), int) else None
                  for k in ("wall_ns", "cpu_ns", "parse_ns", "parse_gc")}
        decision = Decision(None, None, e2e_ns=e2e, worker_wall_ns=timing["wall_ns"],
                            worker_cpu_ns=timing["cpu_ns"], worker_parse_ns=timing["parse_ns"],
                            worker_parse_gc=timing["parse_gc"])
        if not on_time(frame.received_ns, deadline):
            decision.fallback = "timeout"
            decision.rejected = {"late_by_ns": frame.received_ns - deadline}
            _note_reply(self._record_timeout(seq, e2e, deadline, payload, committed), frame)
            return decision
        if m.get("t") == "error":
            code = m.get("code")
            decision.fallback = code if code in FALLBACK_REASONS else "bot_exception"
            decision.rejected = {"detail": str(m.get("detail", ""))[:DETAIL_CHARS]}
            return decision
        if m.get("t") != "action" or "action" not in m:
            decision.fallback = "invalid_output"
            return decision
        decision.action, decision.note = split_note(m["action"])
        if self.info is not None and self.info.state_mode == "explicit":
            try:
                state = _checked_state(frame.blob)
            except StateError as exc:
                decision.fallback, decision.action = exc.code, None
                decision.rejected = {"action": _small(m["action"])}
                return decision
            decision.state_bytes = len(state)
            self.pending = (seq, state)
        return decision

    def _replace_later(self) -> None:
        if self.channel is not None:
            self._discard(self.channel)
            self.channel = None

    def accept(self) -> None:
        if self.pending is not None:
            self.commit_seq, self.committed = self.pending
            self.committed_history.append(self.committed)
            self.stats.max_state_bytes = max(self.stats.max_state_bytes, len(self.committed))
        self.pending = None

    def reject(self) -> None:
        self.pending = None

    def close(self) -> str:
        if self.channel is not None:
            self._discard(self.channel)
            self.channel = None
        return "".join(self.logs)[:LOG_KEEP_BYTES]


def _note_reply(record: TimeoutRecord, frame: ReceivedFrame, drained: bool = False) -> None:
    m = frame.meta
    record.outcome = "late_reply"
    if drained:
        record.drained_after_ns = frame.received_ns - record.deadline_ns
    else:
        record.late_by_ns = frame.received_ns - record.deadline_ns
    record.reply_kind = str(m.get("t"))
    (record.worker_parse_ns, record.worker_wall_ns, record.worker_cpu_ns, record.worker_parse_cpu_ns,
     record.worker_parse_gc, record.worker_threads) = (
        m.get(k) if isinstance(m.get(k), int) else None
        for k in ("parse_ns", "wall_ns", "cpu_ns", "parse_cpu_ns", "parse_gc", "threads"))


REPLAY_BUDGET_NS = 100_000_000


def replay_decision(spawn: Callable[[], Channel], artifact: Artifact, table_info: TableInfo, observation: bytes,
                    committed: bytes, *, repeats: int = 3, budget_ns: int = REPLAY_BUDGET_NS,
                    clock: Clock | None = None) -> list[dict[str, Any]]:
    """Replay one explicit-state decision ``repeats`` times in a fresh worker.

    The worker is restored from ``committed`` and asked the same request each time;
    nothing is committed between repeats, so every repeat decides from the same
    state. Returns the worker-reported times per repeat (``fallback`` is ``timeout``
    when even ``budget_ns`` was not enough).
    """
    seat = WorkerSeat(table_info.hero_seat, artifact, table_info, spawn, clock=clock, budget_ns=budget_ns)
    seat.committed = _checked_state(committed)
    obs = Observation.model_validate_json(observation)
    out: list[dict[str, Any]] = []
    try:
        seat.channel = seat._load(restore=True)
        for _ in range(repeats):
            seat.prepare()
            d = seat.decide(obs)
            seat.reject()
            out.append({"fallback": d.fallback, "e2e_ns": d.e2e_ns, "worker_parse_ns": d.worker_parse_ns,
                        "worker_wall_ns": d.worker_wall_ns, "worker_cpu_ns": d.worker_cpu_ns})
    finally:
        seat.close()
    return out


def _checked_state(blob: bytes) -> bytes:
    if len(blob) > MAX_STATE_BYTES:
        raise StateError("state_too_large", "state over 64 KiB")
    try:
        decode_state(blob)
    except (UnicodeDecodeError, ValueError):
        raise StateError("state_invalid", "state is not valid JSON") from None
    return blob


def _small(value: Any) -> Any:
    text = json.dumps(value, default=str)
    return value if len(text) <= DETAIL_CHARS else text[:DETAIL_CHARS]


class InProcessSeat:
    """A trusted bot called directly, without isolation or an enforced deadline.

    For built-in baselines in bulk runs and for diagnostics. It uses the same
    commit rule (explicit state is re-decoded from committed canonical JSON) so
    results match the sandboxed path, but timings are not competitive numbers.
    """

    def __init__(self, seat: int, bot: object, info: BotInfo, table_info: TableInfo, artifact_sha256: str) -> None:
        self.seat = seat
        self.bot = bot
        self.info = info
        self.table_info = table_info
        self.artifact_sha256 = artifact_sha256
        self.stats = SeatStats()
        self.committed = b"null"
        self.committed_history: list[bytes] = []
        self.pending: bytes | None = None

    def start(self) -> BotInfo:
        require_supported(self.info, self.table_info.ruleset_id)
        if self.info.state_mode == "explicit":
            self.committed = encode_state(self.bot.initial_state(self.table_info))  # type: ignore[attr-defined]
            self.committed_history.append(self.committed)
        else:
            self.bot.start_table(self.table_info)  # type: ignore[attr-defined]
        return self.info

    def prepare(self) -> None:
        return None

    def restore(self, state: bytes) -> None:
        """Continue from a checkpointed explicit state, e.g. after the server restarted."""
        decode_state(state)
        self.committed = state
        self.committed_history.append(self.committed)

    def decide(self, observation: Observation) -> Decision:
        self.stats.decisions += 1
        self.pending = None
        t0 = time.perf_counter_ns()
        c0 = time.thread_time_ns()
        try:
            if self.info.state_mode == "explicit":
                action, state = self.bot.decide(observation, decode_state(self.committed))  # type: ignore[attr-defined]
                self.pending = encode_state(state)
            else:
                action = self.bot.decide(observation)  # type: ignore[attr-defined]
        except StateError as exc:
            return Decision(None, exc.code)
        except Exception as exc:
            return Decision(None, "bot_exception", rejected={"detail": f"{type(exc).__name__}: {exc}"[:DETAIL_CHARS]})
        wall, cpu = time.perf_counter_ns() - t0, time.thread_time_ns() - c0
        payload, note = split_note(action.model_dump(exclude_none=True) if hasattr(action, "model_dump") else action)
        return Decision(payload, None, e2e_ns=wall, worker_wall_ns=wall, worker_cpu_ns=cpu,
                        state_bytes=len(self.pending) if self.pending is not None else None, note=note)

    def accept(self) -> None:
        if self.pending is not None:
            self.committed = self.pending
            self.committed_history.append(self.committed)
            self.stats.max_state_bytes = max(self.stats.max_state_bytes, len(self.committed))
        self.pending = None

    def reject(self) -> None:
        self.pending = None

    def close(self) -> str:
        return ""


# ------------------------------------------------------------------ table loop


@dataclass
class TableRun:
    table: Any
    seats: dict[int, SeatAgent]
    wall_s: float

    def fallback_counts(self) -> dict[int, dict[str, int]]:
        return {s: dict(a.stats.fallbacks) for s, a in self.seats.items()}


def play_table(
    table: Any,
    agents: Mapping[int, SeatAgent],
    *,
    on_decision: Callable[[int, Decision, Any], None] | None = None,
    before_decision: Callable[[int, int], None] | None = None,
) -> TableRun:
    """Drive ``table`` to completion, asking each seat's agent for its decisions.

    Agents must already be started (and their rulesets checked) before the table
    was created. Every decision is consumed exactly once: by the agent's accepted
    action or by a fallback.
    """
    from arena.table.projections import observation
    from arena.table.service import ActionRejected

    try:
        sandbox.set_timer_slack(1000)
    except sandbox.SandboxError:
        pass
    t0 = time.perf_counter()
    index = 0
    gc.freeze()
    try:
        while not table.complete:
            seat = table.actor_seat
            agent = agents[seat]
            if before_decision is not None:
                before_decision(index, seat)
            agent.prepare()
            obs = observation(table, seat)
            decision = agent.decide(obs)
            receipt = None
            if decision.fallback is None:
                try:
                    receipt = table.submit(seat, decision.action, hand_id=obs.hand_id, decision_id=obs.decision_id,
                                           expected_state_version=obs.state_version)
                except ActionRejected as exc:
                    decision.fallback = exc.code
                    decision.rejected = {"action": _small(decision.action)}
            if receipt is not None:
                agent.accept()
            else:
                agent.reject()
                agent.stats.fallbacks[decision.fallback] += 1
                table.apply_fallback(seat, decision.fallback, decision.rejected)
            st = agent.stats
            if decision.e2e_ns is not None:
                st.e2e_ns.append(decision.e2e_ns)
            if decision.worker_wall_ns is not None:
                st.worker_wall_ns.append(decision.worker_wall_ns)
            if decision.worker_cpu_ns is not None:
                st.worker_cpu_ns.append(decision.worker_cpu_ns)
            st.worker_parse_ns.append(-1 if decision.worker_parse_ns is None or decision.worker_parse_gc
                                      else decision.worker_parse_ns)
            if on_decision is not None:
                on_decision(seat, decision, table)
            index += 1
    finally:
        gc.unfreeze()
    return TableRun(table=table, seats=dict(agents), wall_s=time.perf_counter() - t0)
