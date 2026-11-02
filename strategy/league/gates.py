"""Promotion gates of the league on official points (real engine, validation partition).

A league candidate is promoted only through paired experiments run by the strategy lab
(:mod:`strategy.experiments.runner`): full eight-seat tournaments on the authoritative
engine, every seat rotation, both arms on the same seeds, terminal points as the only
metric. Every gate spec is generated from the league config before it runs, written to
the run directory and pinned by its manifest; the holdout partition is never used.

``improve``
    Candidate (B) against the incumbent (A) in validation-population lineups.
``regress-<j>``
    Candidate (B) against an earlier pool member ``j`` (A) in ``j``-rich lineups: ``j`` in
    :data:`J_SEATS` opponent seats and validation members in the rest. B - A < 0 would mean
    the candidate does worse in ``j``'s world than ``j`` itself, i.e. it lost to an older
    policy (a cycle) or forgot how to play against it.

The decision rule (:func:`decide`) is declared in the league config before any gate runs.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from strategy.eval.stats import holm, t_cdf
from strategy.experiments.runner import AlreadyRun, default_run_dir, materialize, run_experiment
from strategy.experiments.schema import ExperimentSpec

J_SEATS = 3
"""Opponent seats an earlier pool member fills in its anti-regression lineups."""
GATE_EXECUTION = {"max_fallback_rate": 0.0, "max_illegal_actions": 0, "max_errors": 0, "max_failed_tables": 0,
                  "max_artifact_bytes": 100_000_000}


@dataclass(frozen=True)
class Rule:
    """The promotion rule, fixed in the league config before any gate runs."""

    improve_lo: float = 0.0
    """``improve``: the 95% paired-t lower bound must exceed this (points per tournament)."""
    worst_family: float = -5.0
    """``improve``: the worst validation family's mean difference must be at least this. A
    family slice holds about 1/15 of the blocks (standard error near 2 points at 288 blocks),
    so this is a catastrophe guard, not a test."""
    regress_mean: float = -2.0
    """Every ``regress-<j>``: point estimate above this."""
    regress_lo: float = -5.0
    """Every ``regress-<j>``: lower bound above this (no catastrophic regression)."""
    pooled_lo: float = -1.5
    """Inverse-variance pooled ``regress-*`` lower bound above this (non-inferiority margin)."""
    improve_lineups: int = 24
    improve_blocks: int = 12
    regress_lineups: int = 12
    regress_blocks: int = 8


def _spec(experiment_id: str, title: str, description: str, a: str, b: str, *, base_seed: str, stage: str,
          lineups: int | None = None, explicit: list[list[str]] | None = None, blocks: int,
          min_effect: float = 0.0) -> dict[str, Any]:
    population: dict[str, Any] = {"source": "strategy.population:PopulationSource", "partition": "validation"}
    if explicit is not None:
        population["explicit_lineups"] = explicit
        population["lineups_per_variant"] = len(explicit)
    else:
        population["lineups_per_variant"] = lineups
    return {"experiment_id": experiment_id, "title": title, "description": description, "stage": stage,
            "min_effect": min_effect, "interval": "paired_t", "arms": {"A": {"policy": a}, "B": {"policy": b}},
            "variants": ["practice_nlhe"], "table": {"players": 8, "starting_stack": 2000, "hand_cap": 200},
            "population": population, "blocks_per_lineup": blocks, "base_seed": base_seed,
            "isolation": "in_process", "gates": GATE_EXECUTION, "bootstrap_resamples": 2000}


def validation_fill(count: int, seats: int, seed: str) -> list[list[str]]:
    """``count`` partial lineups of ``seats`` validation members (each keeps a validation-only pair)."""
    from strategy.population import PopulationSource

    source = PopulationSource()
    part = source.partition("validation")
    lus = source.sample_lineups(part, ruleset_id="practice_nlhe", opponents=seats, count=count, seed=seed)
    return [[o.policy for o in lu.opponents] for lu in lus]


def improve_spec(run: str, k: int, incumbent: tuple[str, str], candidate: tuple[str, str], rule: Rule) -> dict:
    return _spec(f"league-{run}-it{k}-improve", f"League {run} iteration {k}: candidate vs incumbent (validation)",
                 f"Promotion gate. A = incumbent {incumbent[0]}, B = candidate {candidate[0]}; validation population.",
                 incumbent[1], candidate[1], base_seed=f"league-{run}-it{k}-improve", stage="discovery",
                 lineups=rule.improve_lineups, blocks=rule.improve_blocks)


def regress_spec(run: str, k: int, earlier: tuple[str, str], candidate: tuple[str, str], rule: Rule,
                 *, tag: str = "", stage: str = "discovery", blocks: int | None = None,
                 lineups: int | None = None, min_effect: float = 0.0) -> dict:
    name, policy = earlier
    n = lineups or rule.regress_lineups
    fill = validation_fill(n, 7 - J_SEATS, f"league-{run}-it{k}-regress-{name}{tag}")
    explicit = [[policy] * J_SEATS + f for f in fill]
    return _spec(f"league-{run}-it{k}-regress-{name}{tag}",
                 f"League {run} iteration {k}: candidate vs {name} in {name}-rich lineups",
                 f"Anti-regression gate. A = {name}, B = candidate {candidate[0]}; {name} fills {J_SEATS} opponent "
                 "seats, validation members the rest.", policy, candidate[1],
                 base_seed=f"league-{run}-it{k}-regress-{name}{tag}", stage=stage, explicit=explicit,
                 blocks=blocks or rule.regress_blocks, min_effect=min_effect)


def write_spec(spec: dict, directory: Path) -> Path:
    path = directory / f"{spec['experiment_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(spec, indent=2) + "\n"
    if path.exists() and path.read_text() != text:
        raise FileExistsError(f"{path} already holds a different spec; gate specs are never edited")
    path.write_text(text)
    return path


def run(spec_path: Path, *, jobs: int | None = None, root: Path = Path("artifacts/lab")) -> dict[str, Any]:
    """Run a spec (or read the finished run of the same manifest) and summarise it."""
    spec = ExperimentSpec.model_validate_json(spec_path.read_text())
    try:
        run_dir, report = run_experiment(spec, root=root, jobs=jobs)
    except AlreadyRun:
        plan = materialize(spec, root / spec.experiment_id / "_arms")
        run_dir = default_run_dir(plan.manifest, root)
        report = json.loads((run_dir / "report.json").read_text())
    return summarise(run_dir, report)


def summarise(run_dir: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    a = report["analysis"]
    p = a["primary"]
    if a.get("withheld"):
        return {"run_dir": str(run_dir), "withheld": True, "verdict": a["decision"]["verdict"],
                "reason": a["decision"]["reason"]}
    t = p.get("paired_t", {})
    rob = a.get("robustness", {})
    wf = rob.get("worst_family") or {}
    ex = a["execution"]
    return {
        "run_dir": str(run_dir), "manifest_id": report["manifest"]["manifest_id"], "withheld": False,
        "blocks": p["blocks"], "excluded": len(p["blocks_excluded"]),
        "mean_A": p.get("mean_points_A"), "mean_B": p.get("mean_points_B"),
        "diff": p.get("mean_difference"), "sd": p.get("sd_difference"), "lo": t.get("lo"), "hi": t.get("hi"),
        "p_value": t.get("p_value"), "mde": p.get("minimum_detectable_effect"),
        "wins": [p.get("blocks_B_better"), p.get("blocks_tied"), p.get("blocks_A_better")],
        "worst_family": {"family": wf.get("family"), "mean": wf.get("mean_difference"), "n": wf.get("n")},
        "lineup_p5": rob.get("lineup_p5_mean_difference"),
        "families": {k: v.get("mean_difference") for k, v in a["slices"]["family"].items()},
        "verdict": a["decision"]["verdict"],
        "gates": ex["gates"], "gates_ok": all(g["passed"] for g in ex["gates"]),
        "execution_B": {k: ex["B"].get(k) for k in ("decisions", "fallbacks", "fallback_rate", "illegal_actions",
                                                     "errors", "timeouts", "artifact_bytes")} if ex.get("B") else {},
    }


def pooled(results: Sequence[Mapping[str, Any]]) -> dict[str, float] | None:
    """Inverse-variance weighted mean of several gate differences, with a normal 95% interval."""
    rows = [(r["diff"], (r["sd"] / math.sqrt(r["blocks"])) ** 2) for r in results
            if not r.get("withheld") and r.get("blocks", 0) > 1 and r.get("sd")]
    if not rows:
        return None
    w = [1.0 / v for _, v in rows]
    m = sum(wi * d for wi, (d, _) in zip(w, rows)) / sum(w)
    se = math.sqrt(1.0 / sum(w))
    return {"mean": m, "se": se, "lo": m - 1.96 * se, "hi": m + 1.96 * se, "k": len(rows)}


def decide(rule: Rule, improve: Mapping[str, Any], regress: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Promotion decision with every check listed."""
    checks = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    def valid(r: Mapping[str, Any]) -> bool:
        return not r.get("withheld") and r.get("lo") is not None and not r.get("excluded")

    ok = valid(improve)
    add("improve: lower bound", ok and improve["lo"] > rule.improve_lo,
        f"{improve.get('diff', math.nan):+.3f} [{improve.get('lo', math.nan):+.3f}, {improve.get('hi', math.nan):+.3f}]"
        f" > {rule.improve_lo:+g}" if ok else f"invalid: {improve.get('reason', 'withheld or incomplete')}")
    wf = improve.get("worst_family", {}) if ok else {}
    add("improve: worst family", ok and wf.get("mean") is not None and wf["mean"] >= rule.worst_family,
        f"{wf.get('family')} {wf.get('mean', math.nan):+.3f} >= {rule.worst_family:+g}")
    add("improve: execution", bool(improve.get("gates_ok")), "illegal, errors, fallbacks, failed tables, bytes")
    for name, r in regress.items():
        okr = valid(r)
        add(f"regress-{name}: mean", okr and r["diff"] > rule.regress_mean,
            f"{r.get('diff', math.nan):+.3f} > {rule.regress_mean:+g}" if okr else "invalid")
        add(f"regress-{name}: lower bound", okr and r["lo"] > rule.regress_lo,
            f"{r.get('diff', math.nan):+.3f} [{r.get('lo', math.nan):+.3f}, {r.get('hi', math.nan):+.3f}]"
            f" > {rule.regress_lo:+g}" if okr else "invalid")
        add(f"regress-{name}: execution", bool(r.get("gates_ok")), "illegal, errors, fallbacks, failed tables")
    pool = pooled(list(regress.values()))
    if pool is not None:
        add("regress pooled: non-inferiority", pool["lo"] > rule.pooled_lo,
            f"{pool['mean']:+.3f} [{pool['lo']:+.3f}, {pool['hi']:+.3f}] > {rule.pooled_lo:+g} over {pool['k']}")
    return {"promote": all(c["passed"] for c in checks), "checks": checks, "pooled_regress": pool}


# ------------------------------------------------------------------ confirmations


def one_sided_p(diff: float, sd: float, n: int, margin: float = 0.0) -> float:
    """One-sided paired-t p-value of ``mean > margin``."""
    if n < 2 or not sd:
        return math.nan
    t = (diff - margin) / (sd / math.sqrt(n))
    return 1.0 - t_cdf(t, n - 1)


def holm_family(rows: Mapping[str, Mapping[str, Any]], alpha: float = 0.05) -> dict[str, dict[str, Any]]:
    """Holm step-down over pre-registered one-sided tests ``{name: {diff, sd, blocks, margin}}``."""
    ps = {k: one_sided_p(r["diff"], r["sd"], r["blocks"], r.get("margin", 0.0)) for k, r in rows.items()}
    adj = holm({k: p for k, p in ps.items() if not math.isnan(p)})
    out = {}
    for k, r in rows.items():
        out[k] = {**r, "p_one_sided": ps[k], "p_holm": adj.get(k, math.nan),
                  "confirmed": bool(adj.get(k, 1.0) < alpha)}
    return out
