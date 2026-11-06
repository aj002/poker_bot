"""Promotion check: a searched vector against the hand-written control on the validation partition.

Both vectors are packaged (:mod:`strategy.policy.package`) and compared by the
experiment harness (:func:`strategy.experiments.runner.run_experiment`): paired seed
blocks with every seat rotation, official terminal points, a paired 95% interval and a
``min_effect`` declared in the spec before the run. The report's built-in slices give
results by opponent family and candidate seat; this module adds the **phase** split.

Phase split: the candidate's points-equity after hand ``h`` is what the table's own
scorer would award if the tournament stopped there (finished places keep their points,
survivors get the ICM share of the rest; :func:`strategy.objective.points.rescore`).
It starts at ``100 / players`` and ends at the official points, so the per-phase
changes add up exactly to the primary metric. Phases follow the blind schedule:
levels 1-2 (hands 1-50), 3-5 (51-125) and 6-8 (126-200, including the cap's ICM).

    python -m strategy.search.validate baseline-validation-nlhe --candidate artifacts/search/nlhe-cem-v1/incumbent.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from unittest import mock

from strategy.policy.package import build_bundle, load_params

REPO = Path(__file__).resolve().parents[2]
PHASES = (("levels 1-2", 0, 50), ("levels 3-5", 50, 125), ("levels 6-8", 125, 10 ** 9))
CONTROL_DIR = "artifacts/bots/baseline-control"
CANDIDATE_DIR = "artifacts/bots/baseline-candidate"


def equity_path(events: Sequence[Mapping[str, Any]], seat: int, boundaries: Sequence[int]) -> list[float]:
    """The seat's points-equity after each boundary hand (clipped to the last hand played)."""
    from strategy.objective.points import rescore

    payouts: list[int] = []
    seats: list[int] = []
    start: dict[int, int] = {}
    cap = 0
    hands: list[tuple[int, dict[int, int], dict[int, int]]] = []
    current: tuple[int, dict[int, int]] | None = None
    for e in events:
        kind, data = e["type"], e["data"]
        if kind == "TableStarted":
            payouts, seats = list(data["config"]["payout_points"]), list(data["seat_ring"])
            cap = int(data["config"]["hand_cap"])
            start = {int(s): int(v) for s, v in data["start_stacks"].items()}
        elif kind == "HandStarted":
            current = (int(data["hand_number"]), {int(s): int(v) for s, v in data["start_stacks"].items()})
        elif kind == "HandCompleted" and current is not None:
            hands.append((current[0], current[1], {int(s): int(v) for s, v in data["final_stacks"].items()}))
            current = None
    return [rescore(payouts, seats, start, [h for h in hands if h[0] <= b], cap).points(seat) for b in boundaries]


class PhaseCollector:
    """Wraps ``strategy.eval.paired.table_record`` to read each history once more before it is deleted."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str, int], list[list[float]]] = defaultdict(list)

    def wrap(self, original: Any) -> Any:
        def table_record(result: Any, history_path: Path | None, focus: Sequence[int] = (0,), **kw: Any) -> Any:
            if result.valid and history_path is not None and history_path.exists():
                events = json.loads(history_path.read_text(encoding="ascii"))["events"]
                seat = next(s for s, e in result.seats.items() if e == 0)
                bounds = [0, *(hi for _, _, hi in PHASES)]
                eq = equity_path(events, seat, bounds)
                arm_dir = history_path.parent.parent
                key = (arm_dir.parent.name, arm_dir.name, int(result.block))
                self.rows[key].append([b - a for a, b in zip(eq, eq[1:])])
            return original(result, history_path, focus, **kw)

        return table_record

    def summary(self) -> dict[str, Any]:
        from strategy.eval.stats import paired_t

        blocks = sorted({(c, b) for c, _, b in self.rows})
        out: dict[str, Any] = {"phases": [p for p, _, _ in PHASES], "blocks": len(blocks), "by_phase": {}}
        for i, (name, lo, hi) in enumerate(PHASES):
            diffs, ua, ub = [], [], []
            for c, b in blocks:
                ra, rb = self.rows.get((c, "A", b)), self.rows.get((c, "B", b))
                if not ra or not rb:
                    continue
                a = sum(r[i] for r in ra) / len(ra)
                bb = sum(r[i] for r in rb) / len(rb)
                ua.append(a)
                ub.append(bb)
                diffs.append(bb - a)
            if len(diffs) >= 2:
                t = paired_t(diffs)
                out["by_phase"][name] = {"hands": [lo + 1, min(hi, 200)], "blocks": len(diffs),
                                         "mean_equity_change_A": sum(ua) / len(ua),
                                         "mean_equity_change_B": sum(ub) / len(ub), "mean_difference": t.mean,
                                         "ci95": [t.lo, t.hi], "p_value": t.p_value}
        return out


def build_spec(template: str | Path, *, overrides: Mapping[str, Any] | None = None) -> Any:
    from strategy.experiments.schema import SPEC_DIR, ExperimentSpec

    path = Path(template)
    if not path.exists():
        path = SPEC_DIR / f"{template}.json"
    raw = json.loads(path.read_text(encoding="utf-8")) | dict(overrides or {})
    return ExperimentSpec.model_validate(raw)


def validate(template: str, candidate: Mapping[str, float], control: Mapping[str, float] | None = None, *,
             jobs: int = 1, overrides: Mapping[str, Any] | None = None) -> tuple[Path, dict[str, Any]]:
    from strategy.experiments.runner import run_experiment
    from strategy.search.cem import _git_commit
    import strategy.eval.paired as paired

    spec = build_spec(template, overrides=overrides)
    commit = _git_commit()
    build_bundle(REPO / spec.arms.A.policy, control, name="baseline_control", commit=commit)
    build_bundle(REPO / spec.arms.B.policy, candidate, name="strategy_baseline", commit=commit)
    phases = PhaseCollector()
    with mock.patch.object(paired, "table_record", phases.wrap(paired.table_record)):
        run_dir, report = run_experiment(spec, jobs=jobs, keep_histories=False,
                                         progress=lambda label, done, total: done % 200 == 0 and print(
                                             f"{label}: {done}/{total}", file=sys.stderr, flush=True))
    ph = phases.summary()
    (run_dir / "phases.json").write_text(json.dumps(ph, indent=1, sort_keys=True) + "\n", encoding="ascii")
    return run_dir, {"report": report, "phases": ph}


def brief(result: Mapping[str, Any]) -> str:
    a = result["report"]["analysis"]
    p, d = a["primary"], a["decision"]
    lines = [f"blocks {p['blocks']}  A {p['mean_points_A']:.2f}  B {p['mean_points_B']:.2f}  "
             f"diff {p['mean_difference']:+.2f}  95% CI [{p['chosen_interval'][0]:+.2f}, {p['chosen_interval'][1]:+.2f}]"
             f"  min_effect {p['min_effect']}  -> {d.get('decision', d)}"]
    v = a.get("variance_reduced")
    if v and v.get("available"):
        lines.append(f"  luck-adjusted diff {v['mean_difference']:+.2f}  95% CI [{v['lo']:+.2f}, {v['hi']:+.2f}]  "
                     f"variance reduction x{v['variance_reduction_factor']:.2f}"
                     + ("  (decision interval)" if v.get("primary") else ""))
    for group in ("family", "seat"):
        for k, v in sorted(a["slices"][group].items()):
            if v.get("n", 0) < 2:
                continue
            lines.append(f"  {group:<6} {k:<22} n {v['n']:>4}  diff {v['mean_difference']:+6.2f}  "
                         f"CI [{v['lo']:+.2f}, {v['hi']:+.2f}]  p_holm {v.get('p_holm', math.nan):.3f}")
    for k, v in result["phases"]["by_phase"].items():
        lines.append(f"  phase  {k:<22} A {v['mean_equity_change_A']:+6.2f}  B {v['mean_equity_change_B']:+6.2f}  "
                     f"diff {v['mean_difference']:+6.2f}  CI [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m strategy.search.validate")
    p.add_argument("spec", help="spec id or path (arms A/B name the control/candidate bundle directories)")
    p.add_argument("--candidate", required=True, help="vector JSON (incumbent.json or params.json)")
    p.add_argument("--control", help="vector JSON (default: the hand-written control)")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--lineups", type=int, help="override population.lineups_per_variant (smoke runs)")
    p.add_argument("--blocks", type=int, help="override blocks_per_lineup (smoke runs)")
    p.add_argument("--experiment-id", help="override experiment_id (smoke runs)")
    args = p.parse_args(argv)
    over: dict[str, Any] = {}
    if args.blocks:
        over["blocks_per_lineup"] = args.blocks
    if args.experiment_id:
        over["experiment_id"] = args.experiment_id
    if args.lineups:
        raw = build_spec(args.spec).model_dump(mode="json")
        over["population"] = raw["population"] | {"lineups_per_variant": args.lineups}
    run_dir, res = validate(args.spec, load_params(args.candidate),
                            load_params(args.control) if args.control else None, jobs=args.jobs, overrides=over)
    print(brief(res))
    print(f"run directory: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
