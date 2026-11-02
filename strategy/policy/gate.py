"""Deployed-path gate: the packaged bot through the real SDK sandbox worker.

The bundle from :mod:`strategy.policy.package` is staged and played by
:class:`arena.agents.worker.WorkerSeat` exactly as in a sandboxed tournament
(Landlock, seccomp, rlimits, the 1 ms deadline, explicit state round-trips through
the protocol). Only the candidate's seat is sandboxed; opponents (train-partition
lineups) run in-process so they do not compete with the worker for CPU, and tables run
one at a time: parallel sandboxed play on a small VM produces infrastructure timeouts
that say nothing about the bot.

Host control
    Every candidate table is followed by a control table: the same task (seed,
    seat, opponents) with a workload-matched burner (:mod:`strategy.eval.burn`)
    sandboxed in the candidate's seat. The burner spends a fixed CPU time per
    decision, calibrated before the first attempt to the candidate's median decide
    time over ``CALIBRATION_TABLES`` tasks that no attempt plays, so it meets host
    events at the candidate's rate per decision. Its timeouts measure the host
    while the gate runs.

Timeouts
    Each timeout, candidate and control alike, is classified
    (:mod:`strategy.eval.stalls`) from the worker's own times in the late reply
    (decide wall and thread CPU time, parse time) and three replays of the
    decision in a fresh worker from its exact observation and committed state. A
    decide that ran past the deadline with CPU ~ wall is bot-caused, as is a slow
    replay; bot-caused timeouts fail the gate outright. Because a host stall on
    this VM is charged to the running thread's CPU clock, the control also collects
    "bot-caused" timeouts: they are reported as the false-attribution floor of the
    rule. Stall-like ones are tested against the control's stall-like ones (C-test,
    p >= 0.01), and the control must not run significantly above the 0.05% host
    ceiling. A host-invalid or stall-excess attempt is rerun with the same seeds up
    to ``max_reruns`` (2) times; every attempt is in the report and the first
    passing one decides.

Pass criteria (all must hold, in the attempt used):

* at least ``min_decisions`` candidate decisions (default 50,000);
* worker decision wall time p99 below ``p99_limit_us`` (default 500 us) - this is the
  bot's own time (state decode + decide) as the worker measures it;
* zero arena fallbacks other than timeouts (illegal action, crash, invalid output,
  state errors) and zero worker restarts;
* zero bot-caused timeouts, and stall-like timeouts within the control's rate on a
  host within the ceiling;
* zero internal fallbacks (``state["fb"]`` in the committed state).

The harness process's collector is off from dispatch until each reply is read and
its heap is frozen per table (:func:`arena.agents.worker.play_table`), so harness GC
never lands inside a decision window. GC inside the bot is attributed by
:mod:`strategy.policy.bench`, which replays the same decision path.

    python -m strategy.policy.gate --bundle artifacts/bots/baseline-candidate --decisions 50000
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

from strategy.eval import stalls
from strategy.eval.burn import burn_bundle, matched_burn_ns
from strategy.search.evaluate import _init, make_tasks, seat_agents
from strategy.search.population import load_population

REPLAYS_PER_TABLE = stalls.TIMEOUT_REPLAYS_PER_SEAT
CALIBRATION_TABLES = 2


def quantiles(xs: list[int]) -> dict[str, float]:
    if not xs:
        return {}
    s = sorted(xs)
    n = len(s)

    def q(p: float) -> float:
        return s[min(n - 1, int(p * n))] / 1000.0

    return {"n": n, "p50_us": q(0.5), "p95_us": q(0.95), "p99_us": q(0.99), "p999_us": q(0.999),
            "max_us": s[-1] / 1000.0, "over_500us": sum(1 for x in s if x > 500_000),
            "over_1ms": sum(1 for x in s if x > 1_000_000)}


class _Side:
    """Everything one seat role (candidate or control) did over an attempt."""

    def __init__(self) -> None:
        self.decisions = self.restarts = self.late = self.internal = self.tables = self.max_state = 0
        self.wall: list[int] = []
        self.e2e: list[int] = []
        self.fallbacks: Counter[str] = Counter()
        self.records: list[tuple[str, dict[str, Any]]] = []

    def add(self, table_id: str, seat: Any, records: list[dict[str, Any]]) -> None:
        st = seat.stats
        self.tables += 1
        self.decisions += st.decisions
        self.wall += st.worker_wall_ns
        self.e2e += st.e2e_ns
        self.fallbacks.update(st.fallbacks)
        self.restarts += len(st.restarts)
        self.late += st.late_discarded
        self.max_state = max(self.max_state, st.max_state_bytes)
        self.records += [(table_id, r) for r in records]

    def tally(self, budget_ns: int) -> dict[str, Any]:
        return stalls.tally_samples(decisions=self.decisions, timeouts=self.fallbacks.get("timeout", 0),
                                    e2e=self.e2e, wall=self.wall, records=self.records, budget_ns=budget_ns)

    def summary(self) -> dict[str, Any]:
        return {"tables": self.tables, "decisions": self.decisions, "worker_wall": quantiles(self.wall),
                "e2e": quantiles(self.e2e), "fallbacks": dict(self.fallbacks), "restarts": self.restarts,
                "late_replies": self.late, "max_state_bytes": self.max_state}


def _records(seat: Any, pool: Any, artifact: Any) -> list[dict[str, Any]]:
    from arena.agents.worker import REPLAY_BUDGET_NS, replay_decision

    out = []
    for rec in seat.stats.timeouts:
        row = rec.as_dict()
        if rec.replayable:
            try:
                runs = replay_decision(pool.acquire, artifact, seat.table_info, rec.observation, rec.committed)
                row["replay"] = {"budget_ns": REPLAY_BUDGET_NS, "runs": runs}
            except Exception as exc:  # the timeout stays unclassified
                row["replay"] = {"error": f"{type(exc).__name__}: {exc}"[:500]}
        out.append(row)
    return out


def _play(pool: Any, artifact: Any, config: Any, task: Any, timeout_inputs: int) -> tuple[Any, Any]:
    """Play one task with ``artifact`` sandboxed in the candidate seat; ``(table, seat)``."""
    from arena.agents.worker import WorkerSeat, play_table

    agents, cs, table = seat_agents(config, task, lambda seat, info: WorkerSeat(
        seat, artifact, info, pool.acquire, budget_ns=config.bot_budget_ns, timeout_inputs=timeout_inputs))
    seat = agents[cs]
    try:
        play_table(table, agents)
    finally:
        seat.close()
    return table, seat


def calibrate(bundle: Path, config: Any, source: Any, *, ruleset_id: str, seed: str, partition: str,
              tables: int = CALIBRATION_TABLES) -> list[int]:
    """The candidate's sandboxed decide wall times over ``tables`` tasks drawn from
    their own namespace (never played by an attempt)."""
    from arena.agents.worker import WorkerPool

    tasks = make_tasks(source, partition, ruleset_id=ruleset_id, players=config.initial_player_count,
                       count=tables, namespace=f"{seed}|{ruleset_id}|calibration")
    walls: list[int] = []
    with WorkerPool(size=2) as pool:
        art = pool.stage(bundle)
        for task in tasks:
            walls += _play(pool, art, config, task, 0)[1].stats.worker_wall_ns
    return walls


def _attempt(bundle: Path, control: Path, config: Any, source: Any, *, ruleset_id: str, seed: str, partition: str,
             min_decisions: int, max_tables: int, log: Any) -> tuple[_Side, _Side, int]:
    from arena.agents.sdk import decode_state
    from arena.agents.worker import WorkerPool

    cand, ctrl = _Side(), _Side()
    pauses = [0]

    def on_gc(phase: str, info: dict[str, Any]) -> None:
        if phase == "start":
            pauses[0] += 1

    gc.callbacks.append(on_gc)
    t0 = time.time()
    try:
        with WorkerPool(size=2) as pool:
            artifacts = {"candidate": pool.stage(bundle), "control": pool.stage(control)}
            batch = 0
            while cand.decisions < min_decisions and cand.tables < max_tables:
                tasks = make_tasks(source, partition, ruleset_id=ruleset_id, players=config.initial_player_count,
                                   count=50, namespace=f"{seed}|{ruleset_id}|{batch}")
                batch += 1
                for task in tasks:
                    for role, side in (("candidate", cand), ("control", ctrl)):
                        art = artifacts[role]
                        table, seat = _play(pool, art, config, task, REPLAYS_PER_TABLE)
                        side.add(table.table_id, seat, _records(seat, pool, art))
                        if role == "candidate":
                            state = decode_state(seat.committed)
                            side.internal += int(state.get("fb", 0)) if isinstance(state, dict) else 0
                    if cand.tables % 25 == 0:
                        log(f"{cand.tables} tables, {cand.decisions} decisions, timeouts "
                            f"{cand.fallbacks.get('timeout', 0)} (control {ctrl.fallbacks.get('timeout', 0)}/"
                            f"{ctrl.decisions}), internal {cand.internal}, {time.time() - t0:.0f}s")
                    if cand.decisions >= min_decisions:
                        break
    finally:
        gc.callbacks.remove(on_gc)
    return cand, ctrl, pauses[0]


def run_gate(bundle: Path, *, ruleset_id: str = "practice_nlhe", min_decisions: int = 50_000,
             p99_limit_us: float = 500.0, seed: str = "deployed-gate-v1", partition: str = "train",
             max_tables: int = 2000, players: int = 8, hand_cap: int = 200, max_reruns: int = stalls.MAX_RERUNS,
             max_control_timeout_rate: float = stalls.MAX_CONTROL_TIMEOUT_RATE,
             stall_p_min: float = stalls.STALL_P_MIN, control_burn_ns: int | None = None,
             log: Any = print) -> dict[str, Any]:
    from arena.agents.baselines import BASELINES
    from arena.domain.models import TournamentConfig

    config = TournamentConfig(ruleset_id=ruleset_id, initial_player_count=players, hand_cap=hand_cap)
    budget = config.bot_budget_ns
    _init(config.model_dump_json())
    source = load_population()
    t0 = time.time()
    control_info: dict[str, Any] = {"burn_ns": control_burn_ns, "calibration_decisions": 0,
                                    "calibration_median_ns": None}
    if control_burn_ns is None:
        walls = calibrate(bundle, config, source, ruleset_id=ruleset_id, seed=seed, partition=partition)
        control_burn_ns = matched_burn_ns(walls)
        control_info |= {"burn_ns": control_burn_ns, "calibration_decisions": len(walls),
                         "calibration_median_ns": int(statistics.median(walls)) if walls else None}
    with tempfile.TemporaryDirectory(prefix="gate-control-") as tmp:
        if control_burn_ns is None:
            control = Path(BASELINES["calling_station"])
            control_info["policy"] = stalls.CONTROL_POLICY
        else:
            burner = burn_bundle(Path(tmp) / "cpu_burner", control_burn_ns)
            control = burner.path
            control_info |= {"policy": burner.name, "artifact_sha256": burner.artifact_sha256}
        log(f"control: {control_info}")
        attempts: list[dict[str, Any]] = []
        verdicts: list[dict[str, Any]] = []
        while True:
            t1 = time.time()
            cand, ctrl, gc_runs = _attempt(bundle, control, config, source, ruleset_id=ruleset_id, seed=seed,
                                           partition=partition, min_decisions=min_decisions, max_tables=max_tables,
                                           log=log)
            w = quantiles(cand.wall)
            other = {k: v for k, v in cand.fallbacks.items() if k != "timeout"}
            k_tally, c_tally = cand.tally(budget), ctrl.tally(budget)
            assessment = stalls.assess(k_tally, c_tally, strict=True, max_bot_timeouts=0,
                                       max_control_timeout_rate=max_control_timeout_rate, stall_p_min=stall_p_min)
            checks = {
                "decisions": cand.decisions >= min_decisions,
                "p99": bool(w) and w["p99_us"] < p99_limit_us,
                "arena_fallbacks_other_than_timeout": not other,
                "restarts": cand.restarts == 0,
                "internal_fallbacks": cand.internal == 0,
                "bot_caused_timeouts": k_tally["classes"]["bot_caused"] == 0,
                "timeouts_within_host_control": assessment["verdict"] == "pass",
            }
            hard = [k for k, ok in checks.items() if not ok and k != "timeouts_within_host_control"]
            verdict = ({"verdict": "fail", "reason": "failed: " + ", ".join(hard)} if hard
                       else {"verdict": assessment["verdict"], "reason": assessment["reason"]})
            verdicts.append(verdict)
            attempts.append({
                "attempt": len(attempts) + 1, **verdict, "checks": checks, "wall_s": round(time.time() - t1, 1),
                "candidate": {**cand.summary(), "internal_fallbacks": cand.internal, "timeouts": k_tally},
                "control": {**ctrl.summary(), "timeouts": c_tally},
                "assessment": assessment, "parent_gc_collections": gc_runs,
            })
            log(f"attempt {len(attempts)}: {verdict['verdict']} - {verdict['reason']}")
            selection = stalls.select_attempt(verdicts, max_reruns)
            if selection["action"] != "rerun":
                break
    used = attempts[selection["used"]] if selection["used"] is not None else attempts[-1]
    c = used["candidate"]
    mf = Path(bundle) / "manifest.json"
    return {"bundle": str(bundle), "manifest": json.loads(mf.read_text()) if mf.exists() else None,
            "ruleset_id": ruleset_id, "partition": partition, "seed": seed, "tables": c["tables"],
            "decisions": c["decisions"], "worker_wall": c["worker_wall"], "e2e": c["e2e"],
            "fallbacks": c["fallbacks"], "restarts": c["restarts"], "late_replies": c["late_replies"],
            "internal_fallbacks": c["internal_fallbacks"], "max_state_bytes": c["max_state_bytes"],
            "timeout_classes": c["timeouts"]["classes"], "timeout_rules": c["timeouts"]["rules"],
            "control": control_info, "control_timeouts": used["control"]["timeouts"],
            "p99_limit_us": p99_limit_us, "checks": used["checks"], "selection": {**selection, "max_reruns": max_reruns},
            "attempts": attempts, "passed": selection["action"] == "use", "wall_s": round(time.time() - t0, 1)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m strategy.policy.gate")
    p.add_argument("--bundle", required=True)
    p.add_argument("--ruleset", default="practice_nlhe")
    p.add_argument("--decisions", type=int, default=50_000)
    p.add_argument("--p99-us", type=float, default=500.0)
    p.add_argument("--seed", default="deployed-gate-v1")
    p.add_argument("--max-reruns", type=int, default=stalls.MAX_RERUNS,
                   help="reruns of a host-invalid or stall-excess attempt (same seeds; every attempt reported)")
    p.add_argument("--control-burn-us", type=float,
                   help="CPU time per decision of the matched control (default: calibrated to the bundle's median)")
    p.add_argument("--out")
    args = p.parse_args(argv)
    burn_ns = None if args.control_burn_us is None else int(args.control_burn_us * 1000)
    rep = run_gate(Path(args.bundle), ruleset_id=args.ruleset, min_decisions=args.decisions,
                   p99_limit_us=args.p99_us, seed=args.seed, max_reruns=args.max_reruns, control_burn_ns=burn_ns,
                   log=lambda s: print(s, file=sys.stderr, flush=True))
    rep.pop("manifest")
    text = json.dumps(rep, indent=1, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
