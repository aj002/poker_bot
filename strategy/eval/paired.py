"""Paired A/B tournament blocks with full seat rotation, on :func:`arena.benchmark.runner.run_benchmark`.

Design
    A *cell* is one ruleset, tournament config and opponent lineup. In each cell the
    candidate seat (entrant 0) is played by arm A and, separately, by arm B, against
    the same opponents, over the same seed blocks. In rotation ``r`` of a block the
    candidate sits at seat ``r``; every rotation of a block shares its decks and first
    button. Opponents' policy seeds derive from the table seed and seat, so the only
    difference between the arms is the candidate policy (common random numbers).

Unit
    The block. ``U_i(arm)`` is the candidate's mean terminal points over the block's
    rotations and ``D_i = U_i(B) - U_i(A)``. A block counts only when every rotation
    is valid in both arms; excluded blocks are listed with their reason, never
    dropped silently. Hands and decisions are never treated as independent.
"""

from __future__ import annotations

import shutil
from array import array
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from arena.benchmark.runner import BenchmarkRun, RetryPolicy, run_benchmark
from arena.domain.models import TournamentConfig
from strategy.eval import burn, stalls
from strategy.eval.execution import entrant_execution, seat_timeouts
from strategy.eval.outcomes import TableRecord, table_record
from strategy.eval.policies import ResolvedPolicy, resolve_policy, salted_bundle
from strategy.eval.population import Lineup

CANDIDATE = 0
STRICT_ARM = "B"


@dataclass(frozen=True)
class Arm:
    label: str
    policy: str
    salt: str | None = None


@dataclass(frozen=True)
class Cell:
    cell_id: str
    config: TournamentConfig
    lineup: Lineup
    blocks: int
    seed: str
    """Private base seed for this cell's blocks."""

    @property
    def ruleset_id(self) -> str:
        return self.config.ruleset_id


@dataclass
class ArmRun:
    arm: Arm
    policy: ResolvedPolicy
    records: list[TableRecord]
    execution: dict[str, Any]
    benchmark_manifest_hash: str
    raw_latency: dict[str, list[int]] = field(default_factory=dict)
    seat_timeouts: dict[str, dict[str, int]] = field(default_factory=dict)
    """Decisions and timeouts of every entrant, candidate and opponents."""
    timeout_classes: dict[str, Any] = field(default_factory=dict)
    """Classified timeouts (:func:`strategy.eval.stalls.tally`) of the candidate, the
    opponents and the control, plus ``assessment`` and ``selection``; empty when the
    run does not classify (in-process, or no control)."""


@dataclass
class CellResult:
    cell: Cell
    runs: dict[str, ArmRun]


@dataclass(frozen=True)
class BlockPair:
    cell_id: str
    ruleset_id: str
    lineup_id: str
    families: tuple[str, ...]
    block: int
    u_a: float
    u_b: float
    seat_diffs: Mapping[int, float]

    @property
    def d(self) -> float:
        return self.u_b - self.u_a

    @property
    def cluster(self) -> str:
        return f"{self.cell_id}/{self.block}"


def resolve_arm(arm: Arm, work_dir: Path) -> ResolvedPolicy:
    base = resolve_policy(arm.policy)
    if arm.salt is None:
        return base
    return salted_bundle(base, arm.salt, work_dir / "arms" / arm.label)


def run_arm(cell: Cell, arm: Arm, policy: ResolvedPolicy, *, isolation: str, out_dir: Path, jobs: int = 1,
            budget_ns: int | None = None, retry: RetryPolicy | None = None, keep_histories: bool = False,
            progress: Callable[[str, int, int], None] | None = None, control_every: int = 0,
            timeout_inputs: int = 0, control_policy: ResolvedPolicy | None = None) -> ArmRun:
    """``control_every`` > 0 (sandbox only) interleaves control tables of ``control_policy``
    (default the calling station) and classifies timeouts; ``timeout_inputs`` replays
    that many timeouts per seat per table."""
    classify = isolation == "sandbox" and control_every > 0
    control = None
    if classify:
        control = (control_policy or resolve_policy(stalls.CONTROL_POLICY)).entrant(isolation, "host control")
    entrants = [policy.entrant(isolation, f"{arm.label}:{policy.name}")]
    entrants += [resolve_policy(o.policy).entrant(isolation) for o in cell.lineup.opponents]
    if len(entrants) != cell.config.initial_player_count:
        raise ValueError(f"cell {cell.cell_id}: lineup fills {len(entrants)} seats, config has "
                         f"{cell.config.initial_player_count}")
    records: list[TableRecord] = []
    tables_dir = out_dir / "tables"

    def collect(done: int, total: int, r: Any) -> None:
        path = tables_dir / f"{r.table_id}.json" if r.valid else None
        records.append(table_record(r, path, focus=(CANDIDATE,), delete=not keep_histories))
        if progress is not None:
            progress(f"{cell.cell_id}/{arm.label}", done, total)

    run: BenchmarkRun = run_benchmark(config=cell.config, entrants=entrants, blocks=cell.blocks, base_seed=cell.seed,
                                      isolation=isolation, budget_ns=budget_ns, jobs=jobs, out_dir=out_dir,
                                      save_histories=True, progress=collect, retry_policy=retry,
                                      control=control, control_every=control_every if classify else 0,
                                      timeout_inputs=timeout_inputs if classify else 0)
    if not keep_histories and tables_dir.exists():
        shutil.rmtree(tables_dir)
    s = run.manifest.summary
    budget = run.manifest.budget_ns
    execution = entrant_execution(run.attempts, CANDIDATE, budget_ns=budget, isolation=isolation,
                                  competitive=run.manifest.competitive, artifact_bytes=policy.artifact_bytes,
                                  failed_tables=len(s["tables_failed"]), invalid_tables=len(s["tables_invalid"]),
                                  reruns=s["reruns"])
    raw = {
        "latency_end_to_end_s": [x for r in run.attempts for x in array("q", r.e2e_ns.get(CANDIDATE, b""))],
        "latency_worker_wall_s": [x for r in run.attempts for x in array("q", r.worker_wall_ns.get(CANDIDATE, b""))],
        "latency_worker_cpu_s": [x for r in run.attempts for x in array("q", r.worker_cpu_ns.get(CANDIDATE, b""))],
    }
    labels = {CANDIDATE: "candidate"}
    labels |= {i: f"opponent {i} ({o.member_id})" for i, o in enumerate(cell.lineup.opponents, start=1)}
    records.sort(key=lambda r: (r.block, r.rotation))
    classes: dict[str, Any] = {}
    if classify:
        classes = {
            "candidate": stalls.tally(run.attempts, CANDIDATE, budget),
            "opponents": {labels[i]: stalls.tally(run.attempts, i, budget) for i in range(1, len(entrants))},
            "control": stalls.tally(run.control, len(entrants), budget),
            "control_every": control_every,
            "control_policy": (control_policy or resolve_policy(stalls.CONTROL_POLICY)).name,
        }
    return ArmRun(arm, policy, records, execution, run.manifest.content_hash(), raw,
                  seat_timeouts(run.attempts, labels), classes)


def _attempt_row(n: int, out_dir: Path, run: ArmRun) -> dict[str, Any]:
    c = run.timeout_classes
    return {"attempt": n, "out_dir": str(out_dir), "benchmark_manifest_hash": run.benchmark_manifest_hash,
            "verdict": c["assessment"]["verdict"], "reason": c["assessment"]["reason"],
            "candidate": {k: c["candidate"][k] for k in ("decisions", "timeouts", "classes", "exposure_s")},
            "control": {k: c["control"][k] for k in ("decisions", "timeouts", "timeout_rate", "exposure_s", "classes")}}


def matched_control(cell: Cell, policy: ResolvedPolicy, *, out_dir: Path, budget_ns: int | None = None,
                    iterations: int | None = None) -> tuple[ResolvedPolicy, dict[str, Any]]:
    """The workload-matched burner control for ``policy`` in ``cell``
    (:mod:`strategy.eval.burn`): the candidate's median decide time over one
    sandboxed probe table, matched by two burner probe tables, unless ``iterations``
    is given; the calling station when the probe saw no candidate decision."""
    info: dict[str, Any] | None = {"iterations": iterations}
    if iterations is None:
        entrants = [policy.entrant("sandbox")] + [resolve_policy(o.policy).entrant("sandbox")
                                                  for o in cell.lineup.opponents]
        walls = burn.calibrate(cell.config, entrants, seed=cell.seed, budget_ns=budget_ns)
        info = burn.matched_iterations(walls, ruleset_id=cell.ruleset_id, seed=cell.seed)
    if info is None:
        return resolve_policy(stalls.CONTROL_POLICY), {"policy": stalls.CONTROL_POLICY, "iterations": None}
    control = burn.burn_bundle(out_dir / "cpu_burner", info["iterations"])
    return control, {**info, "policy": control.name, "artifact_sha256": control.artifact_sha256}


def run_arm_gated(cell: Cell, arm: Arm, policy: ResolvedPolicy, *, out_dir: Path,
                  gates: Mapping[str, Any] | None = None, **kwargs: Any) -> ArmRun:
    """:func:`run_arm` under the classified timeout gate and its rerun policy, against
    the workload-matched control (one calibration per arm, shared by its reruns)."""
    g = dict(gates or {})
    control_every = int(g.get("control_every", stalls.CONTROL_EVERY) or 0)
    inputs = int(g.get("timeout_replays_per_seat", stalls.TIMEOUT_REPLAYS_PER_SEAT))
    max_reruns = int(g.get("max_reruns", stalls.MAX_RERUNS))
    control, calibration = None, None
    if kwargs.get("isolation") == "sandbox" and control_every > 0:
        control, calibration = matched_control(cell, policy, out_dir=out_dir, budget_ns=kwargs.get("budget_ns"),
                                               iterations=g.get("control_iterations"))
    attempts: list[ArmRun] = []
    rows: list[dict[str, Any]] = []
    while True:
        target = out_dir if not attempts else out_dir / f"rerun-{len(attempts)}"
        run = run_arm(cell, arm, policy, out_dir=target, control_every=control_every, timeout_inputs=inputs,
                      control_policy=control, **kwargs)
        attempts.append(run)
        if not run.timeout_classes:
            return run
        run.timeout_classes["control_calibration"] = calibration
        run.timeout_classes["assessment"] = stalls.assess(
            run.timeout_classes["candidate"], run.timeout_classes["control"], strict=arm.label == STRICT_ARM,
            max_bot_timeouts=g.get("max_bot_timeouts", 0),
            max_control_timeout_rate=g.get("max_control_timeout_rate", stalls.MAX_CONTROL_TIMEOUT_RATE),
            stall_p_min=g.get("stall_p_min", stalls.STALL_P_MIN))
        rows.append(_attempt_row(len(attempts), target, run))
        selection = stalls.select_attempt([a.timeout_classes["assessment"] for a in attempts], max_reruns)
        if selection["action"] != "rerun":
            break
    chosen = attempts[selection["used"]] if selection["used"] is not None else attempts[-1]
    chosen.timeout_classes["selection"] = {**selection, "max_reruns": max_reruns, "attempts": rows}
    return chosen


def run_cell(cell: Cell, arms: Sequence[Arm], policies: Mapping[str, ResolvedPolicy], *, isolation: str,
             work_dir: Path, gates: Mapping[str, Any] | None = None, **kwargs: Any) -> CellResult:
    runs = {arm.label: run_arm_gated(cell, arm, policies[arm.label], isolation=isolation, gates=gates,
                                     out_dir=work_dir / "cells" / cell.cell_id / arm.label, **kwargs) for arm in arms}
    return CellResult(cell, runs)


def pair_blocks(result: CellResult, a: str = "A", b: str = "B") -> tuple[list[BlockPair], list[dict[str, Any]]]:
    """Block pairs plus the excluded blocks with reasons."""
    cell = result.cell
    rotations = cell.config.initial_player_count
    seats = cell.config.resolved_seats
    by_arm = {label: {(r.block, r.rotation): r for r in run.records} for label, run in result.runs.items()}
    pairs, excluded = [], []
    for block in range(cell.blocks):
        rows_a = [by_arm[a].get((block, r)) for r in range(rotations)]
        rows_b = [by_arm[b].get((block, r)) for r in range(rotations)]
        bad = [f"{lbl} rotation {r}: {row.error if row else 'missing'}" for lbl, rows in ((a, rows_a), (b, rows_b))
               for r, row in enumerate(rows) if row is None or not row.valid]
        if bad:
            excluded.append({"cell_id": cell.cell_id, "block": block, "reasons": bad})
            continue
        pa = [row.entrants[CANDIDATE].points for row in rows_a if row]
        pb = [row.entrants[CANDIDATE].points for row in rows_b if row]
        seat_diffs = {seats[r]: pb[r] - pa[r] for r in range(rotations)}
        pairs.append(BlockPair(cell.cell_id, cell.ruleset_id, cell.lineup.lineup_id, cell.lineup.families, block,
                               sum(pa) / rotations, sum(pb) / rotations, seat_diffs))
    return pairs, excluded
