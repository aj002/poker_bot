"""Summarise a Slumbot match log (``strategy.eval.slumbot_run play`` JSONL) into one report.

    uv run python -m strategy.eval.slumbot_report artifacts/slumbot/main.jsonl [--json OUT]

Per configuration: hands, raw and all-in-adjusted mbb/hand with 95% intervals, by seat, with
decision-time, fallback and exploitation counters; then the difference between configurations.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from strategy.eval import slumbot as S
from strategy.eval import slumbot_stats as ST


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def adjusted_chips(rec: dict) -> float:
    return ST.adjusted_winnings(rec)


def config_report(recs: list[dict], terms: dict[int, tuple[float, list[float]]] | None = None,
                  beta: list[float] | None = None) -> dict:
    raw = [ST.mbb(r["winnings"]) for r in recs]
    adj = [ST.mbb(adjusted_chips(r)) for r in recs]
    cv = None
    if terms is not None:
        cv = ST.regression_adjust([terms[r["n"]][0] for r in recs], [terms[r["n"]][1] for r in recs], beta)
    by_pos = {}
    for pos, name in ((1, "small_blind"), (0, "big_blind")):
        sub = [r for r in recs if r["client_pos"] == pos]
        by_pos[name] = {"raw": ST.summarize([ST.mbb(r["winnings"]) for r in sub]),
                        "adjusted": ST.summarize([ST.mbb(adjusted_chips(r)) for r in sub])}
    decisions = [d for r in recs for d in r["decisions"]]
    e2e = sorted(d["e2e"] for d in decisions if d.get("e2e") is not None)
    first = Counter(r["decisions"][0]["act"][0] for r in recs if r["decisions"] and r["client_pos"] == 1)
    allin = [r for r in recs if ST.all_in_street(r["action"]) is not None]
    return {
        "hands": len(recs),
        "raw_mbb": ST.summarize(raw),
        "adjusted_mbb": ST.summarize(adj),
        "cv_mbb": cv,
        "by_position": by_pos,
        "reconciled": sum(1 for r in recs if r.get("reconciled")),
        "all_in_calls": len(allin),
        "all_in_by_street": dict(Counter(ST.all_in_street(r["action"]) for r in allin)),
        "showdowns": sum(1 for r in recs if r.get("bot_hole")),
        "decisions": len(decisions),
        "decision_kinds": dict(Counter(d["kind"] for d in decisions)),
        "fallbacks": dict(Counter(d["fb"] for d in decisions if d["fb"])),
        "e2e_us": ({"p50": e2e[len(e2e) // 2] / 1e3, "p99": e2e[int(len(e2e) * 0.99)] / 1e3, "max": e2e[-1] / 1e3,
                    "over_1ms": sum(1 for x in e2e if x > S.BUDGET_NS)} if e2e else {}),
        "sb_first_action": dict(first),
        "mean_decisions_per_hand": len(decisions) / max(1, len(recs)),
    }


def report(recs: list[dict], *, control_variates: bool = True) -> dict:
    """Per-configuration and between-configuration statistics.

    ``control_variates`` adds the luck-term regression estimate (slow: it evaluates equities for
    every hand). Its coefficients are fitted once on all hands of all configurations, so the
    configurations are corrected identically.
    """
    cfgs = sorted({r["cfg"] for r in recs})
    terms = beta = None
    if control_variates:
        terms = {r["n"]: ST.luck_terms(r) for r in recs}
        import numpy as np

        y = np.array([terms[r["n"]][0] for r in recs])
        x = np.array([terms[r["n"]][1] for r in recs])
        beta = [float(b) for b in np.linalg.lstsq(x, y, rcond=None)[0]]
    out: dict = {"beta": beta, "configs": {c: config_report([r for r in recs if r["cfg"] == c], terms, beta) for c in cfgs}}
    if len(cfgs) == 2:
        a, b = cfgs
        ra = [r for r in recs if r["cfg"] == a]
        rb = [r for r in recs if r["cfg"] == b]
        diff = {
            "a": a, "b": b,
            "raw": ST.difference([ST.mbb(r["winnings"]) for r in ra], [ST.mbb(r["winnings"]) for r in rb]),
            "adjusted": ST.difference([ST.mbb(adjusted_chips(r)) for r in ra], [ST.mbb(adjusted_chips(r)) for r in rb]),
        }
        if terms is not None:
            def resid(rs):
                return [ST.mbb(terms[r["n"]][0] - sum(b_ * t for b_, t in zip(beta, terms[r["n"]][1]))) for r in rs]
            diff["cv"] = ST.difference(resid(ra), resid(rb))
        out["difference"] = diff
    return out


def fmt(s: dict) -> str:
    return f"{s['mean']:+.0f} [{s['lo']:+.0f}, {s['hi']:+.0f}]"


def markdown(rep: dict) -> str:
    lines = ["| config | hands | raw mbb/hand [95% CI] | all-in adjusted [95% CI] | + luck control [95% CI] | SB adjusted | "
             "BB adjusted | all-in calls | fallbacks | decisions over 1 ms |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c, r in rep["configs"].items():
        cv = fmt(r["cv_mbb"]) if r.get("cv_mbb") else "n/a"
        lines.append(f"| {c} | {r['hands']} | {fmt(r['raw_mbb'])} | {fmt(r['adjusted_mbb'])} | {cv} | "
                     f"{fmt(r['by_position']['small_blind']['adjusted'])} | {fmt(r['by_position']['big_blind']['adjusted'])} | "
                     f"{r['all_in_calls']} | {sum(r['fallbacks'].values())} | {r['e2e_us'].get('over_1ms', 0)} |")
    if "difference" in rep:
        d = rep["difference"]
        lines.append("")
        lines.append(f"{d['a']} minus {d['b']}: raw {d['raw']['diff']:+.0f} [{d['raw']['lo']:+.0f}, {d['raw']['hi']:+.0f}], "
                     f"adjusted {d['adjusted']['diff']:+.0f} [{d['adjusted']['lo']:+.0f}, {d['adjusted']['hi']:+.0f}]"
                     + (f", + luck control {d['cv']['diff']:+.0f} [{d['cv']['lo']:+.0f}, {d['cv']['hi']:+.0f}]" if "cv" in d else "")
                     + " mbb/hand")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--no-cv", action="store_true", help="skip the (slow) luck control variates")
    a = ap.parse_args(argv)
    rep = report(load(a.log), control_variates=not a.no_cv)
    if a.json:
        a.json.write_text(json.dumps(rep, indent=1, default=float))
    print(markdown(rep))
    for c, r in rep["configs"].items():
        print(f"\n{c}: kinds={r['decision_kinds']} fallbacks={r['fallbacks']} e2e_us={r['e2e_us']} "
              f"reconciled={r['reconciled']}/{r['hands']} showdowns={r['showdowns']} all_in_by_street={r['all_in_by_street']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
