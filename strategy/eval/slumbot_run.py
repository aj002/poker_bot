"""Play bot bundles against Slumbot's public API through the arena's normal decide path.

    uv run python -m strategy.eval.slumbot_run play \\
        --bundle on=artifacts/slumbot/bundle-on --bundle off=artifacts/slumbot/bundle-off \\
        --hands 2000 --out artifacts/slumbot/pilot.jsonl

Each bundle is staged into a sandboxed worker and asked for every decision exactly as the
arena asks (observation built by the projector, 1 ms budget, state committed only with an
accepted action, check-or-fold fallback on a missing, late, illegal or failing reply). Slumbot's
action strings are replayed through the arena engine (:mod:`strategy.eval.slumbot`) so every
action, ours and Slumbot's, is validated by the rules code.

Being a polite client of a free public service: one session, one request at a time, a minimum
interval between requests, exponential backoff with a cap on errors, and a stop file.
Bundles take turns in blocks, so server-side drift hits them equally.
"""

from __future__ import annotations

import argparse
import json
import signal
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from arena.agents.sdk import TableInfo
from arena.domain.models import Action
from arena.table.projections import observation
from arena.table.service import ActionRejected
from strategy.eval import slumbot as S

API = "https://slumbot.com/slumbot/api/"
USER_AGENT = "poker-bot-arena-eval/1 (research; contact via repo owner)"


class ApiError(RuntimeError):
    """Slumbot answered with an error message (a protocol problem, never retried)."""


class Api:
    """Slumbot's ``new_hand`` and ``act`` calls with pacing and backoff."""

    def __init__(self, base: str = API, *, min_interval: float = 0.2, timeout: float = 20.0,
                 max_retries: int = 8, max_backoff: float = 120.0, sleep=time.sleep, clock=time.monotonic) -> None:
        self.base, self.min_interval, self.timeout = base, min_interval, timeout
        self.max_retries, self.max_backoff = max_retries, max_backoff
        self._sleep, self._clock = sleep, clock
        self._last = -1e9
        self.requests = 0
        self.retries = 0
        self.token: str | None = None
        self.slow = 0
        self.max_latency = 0.0

    def _post(self, path: str, body: dict) -> dict:
        wait = self.min_interval - (self._clock() - self._last)
        if wait > 0:
            self._sleep(wait)
        req = urllib.request.Request(self.base + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json", "User-Agent": USER_AGENT})
        self._last = self._clock()
        self.requests += 1
        started = self._clock()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        finally:
            took = self._clock() - started
            self.max_latency = max(self.max_latency, took)
            self.slow += took > 5.0

    def call(self, path: str, body: dict) -> dict:
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            try:
                r = self._post(path, body)
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504):
                    raise ApiError(f"HTTP {exc.code} from {path}") from None
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                delay = max(delay, float(retry_after)) if retry_after and retry_after.isdigit() else delay
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError, OSError):
                pass
            else:
                if "error_msg" in r:
                    raise ApiError(f"{path}: {r['error_msg']} (old_action={r.get('old_action')!r})")
                if r.get("token"):
                    self.token = r["token"]
                return r
            if attempt == self.max_retries:
                break
            self.retries += 1
            self._sleep(min(delay, self.max_backoff))
            delay *= 2
        raise ApiError(f"{path}: gave up after {self.max_retries} retries")

    def new_hand(self) -> dict:
        return self.call("new_hand", {"token": self.token} if self.token else {})

    def act(self, incr: str) -> dict:
        return self.call("act", {"token": self.token, "incr": incr})


# ------------------------------------------------------------------------------ bot seat


class BotSeat:
    """One bundle behind a sandboxed worker, plus per-hand serial numbers."""

    def __init__(self, name: str, bundle: Path, pool: Any) -> None:
        from arena.agents.worker import WorkerSeat

        self.name = name
        cfg = S.config()
        info = TableInfo(table_id=S.TABLE_ID + "_" + name, hero_seat=S.HERO, ruleset_id="practice_nlhe",
                         config=cfg.resolved(), config_hash=cfg.config_hash(), seats=cfg.resolved_seats,
                         decision_budget_ns=S.BUDGET_NS, policy_seed=f"slumbot-{name}")
        self.art = pool.stage(bundle)
        self.agent = WorkerSeat(S.HERO, self.art, info, pool.acquire, budget_ns=S.BUDGET_NS, spin_ns=S.BUDGET_NS)
        self.agent.start()
        self.serial = 0
        self.table_id = info.table_id

    def decide(self, st: S.HandState) -> tuple[Action, Any]:
        table = S.replay(st.client_pos, st.hole, st.board, st.action, first_hand=self.serial)
        if table.complete or table.actor_seat != S.HERO:
            raise S.ProtocolMismatch(f"hero is not to act after {st.action!r}")
        street = S._street_index(table)
        if len(st.board) != S.expected_board(street):
            raise S.ProtocolMismatch(f"{len(st.board)} board cards on street {street} in {st.action!r}")
        # As in play_table: wake the worker first, build the observation while it wakes, then ask.
        self.agent.prepare()
        obs = observation(table, S.HERO)
        d = self.agent.decide(obs)
        action = None
        if d.fallback is None:
            try:
                action = Action.model_validate(d.action, strict=True)
                table.submit(S.HERO, action)
            except (ActionRejected, ValueError):
                d.fallback = "illegal_action"
        if d.fallback is None:
            self.agent.accept()
            return action, d
        self.agent.reject()
        self.agent.stats.fallbacks[d.fallback] += 1
        fresh = S.replay(st.client_pos, st.hole, st.board, st.action, first_hand=self.serial)
        return fresh.fallback_action(), d

    def state_summary(self) -> dict:
        """The committed explicit state's belief string and counters (what survives between hands)."""
        from arena.agents.sdk import decode_state

        st = decode_state(self.agent.committed)
        return {k: st.get(k) for k in ("x", "sp", "fb", "h") if isinstance(st, dict)}

    def close(self) -> None:
        self.agent.close()


def note_class(note: str | None) -> str:
    if not note:
        return "none"
    if "x[" in note:
        return "exploit"
    if note.startswith("push/fold"):
        return "pushfold"
    if "miss" in note:
        return "miss"
    return "blueprint"


def play_hand(api: Api, seat: BotSeat, n: int, cfg_name: str) -> dict:
    seat.serial += 1
    r = api.new_hand()
    decisions: list[dict] = []
    while True:
        st = S.HandState.from_response(r)
        if st.winnings is not None:
            break
        action, d = seat.decide(st)
        decisions.append({"e2e": d.e2e_ns, "wall": d.worker_wall_ns, "fb": d.fallback, "kind": note_class(d.note),
                          "act": S.format_action(action)})
        r = api.act(S.format_action(action))
    rec: dict[str, Any] = {
        "n": n, "cfg": cfg_name, "serial": seat.serial, "client_pos": st.client_pos, "hole": list(st.hole),
        "bot_hole": list(st.villain_hole) if st.villain_hole else None, "board": list(st.board),
        "action": st.action, "winnings": st.winnings, "decisions": decisions, "t": round(time.time(), 1), "m": round(time.monotonic(), 1),
    }
    if seat.serial % 50 == 0:
        rec["bot_state"] = seat.state_summary()
    try:
        t = S.replay(st.client_pos, st.hole, st.board, st.action, first_hand=seat.serial, villain_hole=st.villain_hole)
        net = S.hero_net(t)
        rec["engine_net"] = net
        rec["reconciled"] = net == st.winnings
    except S.ProtocolMismatch as exc:
        rec["engine_net"], rec["reconciled"], rec["mismatch"] = None, False, str(exc)
    return rec


# ------------------------------------------------------------------------------ session


class Stop:
    def __init__(self, path: Path | None) -> None:
        self.path, self.flag = path, False
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._handle)

    def _handle(self, *_: object) -> None:
        self.flag = True

    def __bool__(self) -> bool:
        return self.flag or bool(self.path and self.path.exists())


def count_done(out: Path) -> Counter:
    done: Counter = Counter()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                done[json.loads(line)["cfg"]] += 1
    return done


def play(bundles: dict[str, Path], hands: int, out: Path, *, block: int = 50, min_interval: float = 0.2,
         stop_file: Path | None = None, api: Api | None = None, log=print) -> dict:
    """Play ``hands`` per bundle, alternating blocks of ``block`` hands, appending JSONL to ``out``.

    Resumable: hands already in ``out`` count toward the total (a resumed bot starts with fresh
    beliefs, its hand serial continues).
    """
    from arena.agents.worker import WorkerPool

    out.parent.mkdir(parents=True, exist_ok=True)
    done = count_done(out)
    api = api or Api(min_interval=min_interval)
    stop = Stop(stop_file)
    t0 = time.time()
    with WorkerPool(size=max(2, len(bundles))) as pool, out.open("a") as fh:
        seats = {name: BotSeat(name, path, pool) for name, path in bundles.items()}
        for name, seat in seats.items():
            seat.serial = done[name]
        try:
            while not stop and any(done[n] < hands for n in seats):
                for name, seat in seats.items():
                    for _ in range(min(block, hands - done[name])):
                        if stop:
                            break
                        total = sum(done.values())
                        w0, m0 = time.time(), time.monotonic()
                        rec = play_hand(api, seat, total, name)
                        dw, dm = time.time() - w0, time.monotonic() - m0
                        if dw > 60:
                            log(f"long hand {rec['serial']}: wall={dw:.0f}s monotonic={dm:.0f}s "
                                f"max_latency={getattr(api, 'max_latency', 0.0):.1f}s", file=sys.stderr, flush=True)
                        fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
                        fh.flush()
                        done[name] += 1
                        if sum(done.values()) % 100 == 0:
                            log(f"{dict(done)} requests={api.requests} retries={api.retries} "
                                f"slow={getattr(api, 'slow', 0)} max_latency={getattr(api, 'max_latency', 0.0):.1f}s elapsed={time.time() - t0:.0f}s", file=sys.stderr, flush=True)
        finally:
            for seat in seats.values():
                seat.close()
    return {"done": dict(done), "requests": api.requests, "retries": api.retries,
            "stopped": bool(stop), "seconds": round(time.time() - t0, 1)}


def build_off_bundle(src: Path, dst: Path) -> None:
    """Copy a bundle and zero the exploit weights, leaving the blueprint alone."""
    import shutil

    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    bot = dst / "bot.py"
    text = bot.read_text()
    needle = 'exploit_pack=DATA / "exploit-practice_nlhe.npz")'
    if needle not in text:
        raise SystemExit("unexpected bundle bot.py; cannot patch exploit weights")
    bot.write_text(text.replace(needle, 'exploit_pack=DATA / "exploit-practice_nlhe.npz",\n'
                                       '                    overrides={"lam_scale": 0.0})'))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("play")
    p.add_argument("--bundle", action="append", required=True, metavar="NAME=DIR")
    p.add_argument("--hands", type=int, required=True, help="hands per bundle (total, including any already in --out)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--block", type=int, default=50)
    p.add_argument("--min-interval", type=float, default=0.2, help="seconds between request starts")
    p.add_argument("--stop-file", type=Path)
    o = sub.add_parser("off-bundle", help="derive the exploitation-off bundle from an exploitation-on bundle")
    o.add_argument("src", type=Path)
    o.add_argument("dst", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "off-bundle":
        build_off_bundle(a.src, a.dst)
        return 0
    bundles = {}
    for item in a.bundle:
        name, _, path = item.partition("=")
        bundles[name] = Path(path)
    res = play(bundles, a.hands, a.out, block=a.block, min_interval=a.min_interval, stop_file=a.stop_file)
    print(json.dumps(res))
    return 0


def summarize_latency(records: list[dict]) -> dict:
    e2e = sorted(d["e2e"] for r in records for d in r["decisions"] if d.get("e2e") is not None)
    if not e2e:
        return {}
    return {"decisions": len(e2e), "p50_us": e2e[len(e2e) // 2] / 1e3,
            "p99_us": e2e[int(len(e2e) * 0.99)] / 1e3, "max_us": e2e[-1] / 1e3,
            "mean_us": statistics.fmean(e2e) / 1e3}


if __name__ == "__main__":
    raise SystemExit(main())
