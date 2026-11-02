"""Worker process entry point. Started by :mod:`arena.agents.worker`, never by hand.

Order matters: trusted modules are imported first, then the sandbox is applied
(:mod:`arena.agents.sandbox`), and only then is bot code imported. Requests and
replies use :mod:`arena.agents.protocol` frames on two inherited pipe fds; fd 1 and
fd 2 go to a non-blocking, size-capped log pipe so bot output can neither corrupt
the protocol nor block the worker.
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import os
import resource
import select
import sys
import time
import traceback

from arena.agents import sandbox
from arena.agents.protocol import DETAIL_CHARS, encode_frame, read_frame_blocking, write_all
from arena.agents.sdk import (
    BotLoadError,
    StateError,
    TableInfo,
    decode_state,
    encode_state,
    load_bot,
)
from arena.domain.models import Action
from arena.domain.observations import Observation

# Trusted library code bots may import; it must be loaded before the sandbox hides the repository.
import arena.agents.styles.engine  # noqa: E402, F401, I001
import arena.table.player_stats  # noqa: E402, F401

LOG_CAP_BYTES = 64 * 1024
WARMUP_OBSERVATION = os.path.join(os.path.dirname(os.path.abspath(__file__)), "warmup_observation.json")
WARMUP_ROUNDS = 200
WAKE_SPIN_NS = 3_000_000


def _warm_up() -> None:
    """Exercise the request/reply code paths before the worker reports ready.

    The first validations in a fresh interpreter are several times slower than
    later ones; doing them here keeps that cost off a bot's first real deadline.
    The bot is not loaded yet, so no bot code runs on this synthetic observation.
    """
    with open(WARMUP_OBSERVATION, "rb") as f:
        raw = f.read()
    for _ in range(WARMUP_ROUNDS):
        obs = Observation.model_validate_json(raw)
        encode_state({"warm": obs.decision_id, "n": [1, 2, 3]})
        json.dumps({"t": "action", "seq": 1, "action": {"type": "check"}, "wall_ns": 1}, separators=(",", ":"))


class _CappedLog(io.TextIOBase):
    def __init__(self, fd: int, cap: int) -> None:
        self.fd = fd
        self.left = cap

    def writable(self) -> bool:
        return True

    def write(self, s: str) -> int:
        if self.left > 0 and s:
            data = s.encode("utf-8", "replace")[: self.left]
            try:
                self.left -= os.write(self.fd, data)
            except OSError:
                self.left = 0
        return len(s)

    def flush(self) -> None:
        return None


def _detail(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:DETAIL_CHARS]


class Worker:
    def __init__(self, in_fd: int, out_fd: int, runtime_paths: list[str]) -> None:
        self.in_fd = in_fd
        self.out_fd = out_fd
        self.runtime_paths = runtime_paths
        self.bot: object | None = None
        self.mode: str | None = None
        self.cached: bytes = b"null"
        self.pending: tuple[int, bytes] | None = None

    def send(self, meta: dict, blob: bytes = b"") -> None:
        write_all(self.out_fd, encode_frame(meta, blob))

    def load(self, meta: dict, blob: bytes) -> None:
        try:
            sandbox.apply_landlock([*self.runtime_paths, meta["bot_dir"]])
            sys.path.insert(0, meta["bot_dir"])
            table = TableInfo.model_validate(meta["table"])
            bot, info = load_bot(meta["bot_path"], module_name="arena_bot")
            state = b""
            if info.state_mode == "explicit":
                if meta.get("restore"):
                    self.cached = blob
                else:
                    self.cached = state = encode_state(bot.initial_state(table))
            else:
                bot.start_table(table)
        except StateError as exc:
            self.send({"t": "load_error", "code": exc.code, "detail": _detail(exc)})
            return
        except BotLoadError as exc:
            self.send({"t": "load_error", "code": "bot_load_error", "detail": _detail(exc)})
            return
        except BaseException as exc:  # bot import or factory failed
            self.send({"t": "load_error", "code": "bot_exception", "detail": _detail(exc)})
            return
        self.bot, self.mode = bot, info.state_mode
        gc.collect()
        gc.freeze()
        self.send({"t": "loaded", "info": info.as_dict()}, state)

    def decide(self, meta: dict, blob: bytes) -> None:
        seq = meta["seq"]
        if self.pending is not None and self.pending[0] == meta.get("commit"):
            self.cached = self.pending[1]
        self.pending = None
        t0, p0 = time.perf_counter_ns(), time.thread_time_ns()
        obs = Observation.model_validate_json(blob)
        t1 = time.perf_counter_ns()
        c0 = time.thread_time_ns()
        bot = self.bot
        try:
            if self.mode == "explicit":
                action, new_state = bot.decide(obs, decode_state(self.cached))  # type: ignore[union-attr]
            else:
                action, new_state = bot.decide(obs), None  # type: ignore[union-attr]
        except BaseException as exc:  # the bot's fault: consumed by fallback, worker keeps running
            c1, t2 = time.thread_time_ns(), time.perf_counter_ns()
            code = "bot_memory_error" if isinstance(exc, MemoryError) else "bot_exception"
            self.send({"t": "error", "seq": seq, "code": code, "detail": _detail(exc),
                       "parse_ns": t1 - t0, "parse_cpu_ns": c0 - p0, "wall_ns": t2 - t1, "cpu_ns": c1 - c0})
            return
        c1, t2 = time.thread_time_ns(), time.perf_counter_ns()
        timing = {"parse_ns": t1 - t0, "parse_cpu_ns": c0 - p0, "wall_ns": t2 - t1, "cpu_ns": c1 - c0}
        try:
            state = encode_state(new_state) if self.mode == "explicit" else b""
            payload = action.model_dump(exclude_none=True) if isinstance(action, Action) else action
            meta_out = json.dumps({"t": "action", "seq": seq, "action": payload, **timing},
                                  separators=(",", ":"), allow_nan=False).encode()
        except StateError as exc:
            self.send({"t": "error", "seq": seq, "code": exc.code, "detail": _detail(exc), **timing})
            return
        except (TypeError, ValueError) as exc:
            self.send({"t": "error", "seq": seq, "code": "invalid_output", "detail": _detail(exc), **timing})
            return
        if self.mode == "explicit":
            self.pending = (seq, state)
        write_all(self.out_fd, encode_frame(meta_out, state))

    def _spin_for_request(self) -> None:
        """After a wake-up notice, poll for the request instead of sleeping in read()."""
        until = time.perf_counter_ns() + WAKE_SPIN_NS
        while time.perf_counter_ns() < until:
            if select.select([self.in_fd], [], [], 0)[0]:
                return

    def serve(self) -> None:
        while True:
            try:
                meta, blob = read_frame_blocking(self.in_fd)
            except EOFError:
                return
            kind = meta.get("t")
            if kind == "wake":
                self._spin_for_request()
            elif kind == "decide" and self.bot is not None:
                self.decide(meta, blob)
            elif kind == "load" and self.bot is None:
                self.load(meta, blob)
            elif kind == "exit":
                return
            else:
                self.send({"t": "error", "seq": meta.get("seq"), "code": "protocol", "detail": f"unexpected {kind}"})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-fd", type=int, required=True)
    parser.add_argument("--out-fd", type=int, required=True)
    parser.add_argument("--config", required=True, help="JSON: runtime_paths, staging_root, limits")
    args = parser.parse_args(argv)
    cfg = json.loads(args.config)

    os.set_blocking(2, False)
    os.dup2(2, 1)
    sys.stdout = sys.stderr = _CappedLog(2, LOG_CAP_BYTES)

    report: dict = {"namespaces": sandbox.namespace_report()}
    _warm_up()
    try:
        limits = sandbox.Limits(**cfg["limits"])
        sandbox.apply_rlimits(limits)
        # Counted per user namespace: the unshare helper, this thread, and bot threads.
        resource.setrlimit(resource.RLIMIT_NPROC, (limits.threads + 2, limits.threads + 2))
        sandbox.set_no_new_privs()
        sandbox.set_not_dumpable()
        ll = sandbox.apply_landlock([*cfg["runtime_paths"], cfg["staging_root"]])
        sandbox.apply_seccomp()
    except sandbox.SandboxError as exc:
        write_all(args.out_fd, encode_frame({"t": "sandbox_error", "detail": str(exc)[:DETAIL_CHARS]}))
        return 3
    report.update(landlock_abi=ll.abi, landlock_net=ll.net, landlock_scoped=ll.scoped, seccomp=True,
                  limits=limits.as_dict())
    sys.path[:] = [p for p in sys.path if p and os.path.realpath(p) in _readable(cfg["runtime_paths"])]

    worker = Worker(args.in_fd, args.out_fd, cfg["runtime_paths"])
    worker.send({"t": "ready", "pid": os.getpid(), "sandbox": report})
    try:
        worker.serve()
    except Exception:
        traceback.print_exc()
        return 2
    return 0


def _readable(roots: list[str]) -> set[str]:
    keep = set()
    for p in sys.path:
        rp = os.path.realpath(p) if p else ""
        if any(rp == r or rp.startswith(r.rstrip("/") + "/") for r in roots):
            keep.add(rp)
    return keep


if __name__ == "__main__":
    sys.exit(main())
