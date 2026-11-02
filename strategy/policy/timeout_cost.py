"""Points cost of deadline timeouts: paired in-process replays with forced timeout fallbacks.

Each evaluation table is played in-process three ways with the same seeds, seats,
opponents and policy streams (common random numbers): as is, and with the
candidate's decisions forced to time out at ``rate`` and at ``10 * rate``. A forced
timeout is what the arena does with a late reply: the bot's decision is never
taken, the table applies its timeout fallback (check if legal, otherwise fold), and
an explicit-state bot keeps its last committed state.

Whether decision ``i`` of the candidate's seat times out is a fixed draw
``u(table seed, seat, i) < rate``, shared by every arm, so the 10x arm's forced
decisions include the 1x arm's for as long as the tables have not diverged. A table
with no forced decision replays its baseline exactly, which the report checks
(``unforced_tables_changed``, expected 0).

The cost is ``points(baseline) - points(forced)`` per table: its mean with a 95%
interval over tables (tables are independent tasks), and the cost per forced
timeout (ratio estimator, ``sum cost / sum forced``).

    python -m strategy.policy.timeout_cost --bundle artifacts/bots/baseline-candidate --rate 0.0003
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from arena.agents.worker import Decision, InProcessSeat

MULTIPLIERS = (1, 10)
DEFAULT_TABLES = 1000
Z95 = 1.959964

_C: dict[str, Any] = {}


def draw(seed: str, seat: int, index: int) -> float:
    """The fixed uniform draw for decision ``index`` of ``seat`` at table ``seed``."""
    h = hashlib.sha256(f"strategy-timeout-cost-v1|{seed}|{seat}|{index}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


class ForcedTimeoutSeat(InProcessSeat):
    """An in-process seat whose decisions time out where ``draw(...) < rate``."""

    def __init__(self, *args: Any, rate: float, seed: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.rate = rate
        self.seed = seed
        self.forced = 0

    def decide(self, observation: Any) -> Decision:
        if self.rate > 0 and draw(self.seed, self.seat, self.stats.decisions) < self.rate:
            self.stats.decisions += 1
            self.pending = None
            self.forced += 1
            return Decision(None, "timeout")
        return super().decide(observation)


def _init_worker(config_json: str, bundle: str) -> None:
    from arena.agents.sdk import load_bot
    from strategy.search.evaluate import _init

    _init(config_json)
    path = Path(bundle)
    entry = path / "bot.py" if path.is_dir() else path
    _, info = load_bot(entry, module_name="timeout_cost_candidate")
    _C["module"] = sys.modules["timeout_cost_candidate"]
    _C["info"] = info
    _C["bot"] = _C["module"].make_bot() if info.state_mode == "explicit" else None


def _play(task: Any, rate: float) -> dict[str, Any]:
    from arena.agents.worker import play_table
    from strategy.search.evaluate import _W, seat_agents

    def candidate(seat: int, info: Any) -> Any:
        bot = _C["bot"] if _C["bot"] is not None else _C["module"].make_bot()
        return ForcedTimeoutSeat(seat, bot, _C["info"], info, "timeout-cost", rate=rate, seed=task.seed)

    agents, cs, table = seat_agents(_W["config"], task, candidate)
    play_table(table, agents)
    seat = agents[cs]
    return {"points": float(table.points()[cs]), "decisions": seat.stats.decisions, "forced": seat.forced,
            "other_fallbacks": sum(v for k, v in seat.stats.fallbacks.items() if k != "timeout")}


def _job(args: tuple[int, Any, tuple[float, ...]]) -> tuple[int, list[dict[str, Any]]]:
    i, task, rates = args
    return i, [_play(task, r) for r in rates]


def _mean_ci(xs: Sequence[float]) -> dict[str, float]:
    n = len(xs)
    mean = sum(xs) / n
    se = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1) / n) if n > 1 else 0.0
    return {"mean": mean, "se": se, "ci95": [mean - Z95 * se, mean + Z95 * se]}


def summarize(base: Sequence[dict[str, Any]], forced: Sequence[dict[str, Any]], rate: float,
              multiplier: int) -> dict[str, Any]:
    cost = [b["points"] - f["points"] for b, f in zip(base, forced)]
    k = [f["forced"] for f in forced]
    total = sum(k)
    per_forced = sum(cost) / total if total else None
    se_forced = (math.sqrt(sum((c - per_forced * ki) ** 2 for c, ki in zip(cost, k))) / total
                 if total and per_forced is not None else None)
    decisions = sum(f["decisions"] for f in forced)
    return {"multiplier": multiplier, "rate": rate, "tables": len(cost), "decisions": decisions,
            "forced": total, "forced_rate": total / decisions if decisions else None,
            "tables_with_forced": sum(1 for x in k if x),
            "unforced_tables_changed": sum(1 for c, ki in zip(cost, k) if ki == 0 and c != 0),
            "cost_per_table": _mean_ci(cost),
            "cost_per_forced_timeout": None if per_forced is None else {
                "mean": per_forced, "se": se_forced,
                "ci95": [per_forced - Z95 * se_forced, per_forced + Z95 * se_forced]}}


def points_cost(bundle: Path, tasks: Sequence[Any], config: Any, rate: float, *,
                multipliers: Sequence[int] = MULTIPLIERS, jobs: int = 1, log: Any = None) -> dict[str, Any]:
    """Paired points cost of forced timeouts at ``rate * m`` for each multiplier ``m``."""
    rates = (0.0, *(min(1.0, rate * m) for m in multipliers))
    work = [(i, t, rates) for i, t in enumerate(tasks)]
    out: list[list[dict[str, Any]] | None] = [None] * len(tasks)
    args = (config.model_dump_json(), str(bundle))
    if jobs <= 1:
        _init_worker(*args)
        results: Any = map(_job, work)
        pool = None
    else:
        pool = ProcessPoolExecutor(jobs, mp_context=get_context("spawn"), initializer=_init_worker, initargs=args)
        results = pool.map(_job, work, chunksize=4)
    try:
        for done, (i, r) in enumerate(results, start=1):
            out[i] = r
            if log is not None and done % 200 == 0:
                log(f"points cost: {done}/{len(tasks)} tables")
    finally:
        if pool is not None:
            pool.shutdown(cancel_futures=True)
    base = [r[0] for r in out if r is not None]
    pts = [b["points"] for b in base]
    return {"bundle": str(bundle), "measured_rate": rate, "tables": len(base),
            "baseline_points_per_table": _mean_ci(pts) if pts else None,
            "fallback": "the arena's timeout fallback: check if legal, otherwise fold",
            "arms": [summarize(base, [r[j + 1] for r in out if r is not None], rates[j + 1], m)
                     for j, m in enumerate(multipliers)]}


def main(argv: list[str] | None = None) -> int:
    from arena.domain.models import TournamentConfig
    from strategy.policy.gate import gate_tasks
    from strategy.search.population import load_population

    p = argparse.ArgumentParser(prog="python -m strategy.policy.timeout_cost")
    p.add_argument("--bundle", required=True)
    p.add_argument("--rate", type=float, required=True, help="measured timeout rate per decision")
    p.add_argument("--ruleset", default="practice_nlhe")
    p.add_argument("--tables", type=int, default=DEFAULT_TABLES)
    p.add_argument("--seed", default="deployed-gate-v1", help="the gate's seed: replays the gate's tables first")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--out")
    args = p.parse_args(argv)
    config = TournamentConfig(ruleset_id=args.ruleset, initial_player_count=8, hand_cap=200)
    tasks = list(itertools.islice(gate_tasks(load_population(), config, seed=args.seed, ruleset_id=args.ruleset,
                                             partition="train"), args.tables))
    rep = points_cost(Path(args.bundle), tasks, config, args.rate, jobs=args.jobs,
                      log=lambda s: print(s, file=sys.stderr, flush=True))
    text = json.dumps(rep, indent=1, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
