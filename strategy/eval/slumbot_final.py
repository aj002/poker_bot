"""Final-analysis helpers for a Slumbot log: fallbacks per block and fallback sensitivity.

Usage: ``python -m strategy.eval.slumbot_final LOG.jsonl [--terms terms.pkl] [--json OUT]``.
The luck terms are slow to compute, so they are cached in the pickle named by ``--terms``.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

from strategy.eval import slumbot_stats as ST
from strategy.eval.slumbot import BUDGET_NS
from strategy.eval.slumbot_report import adjusted_chips, load


def fallback_blocks(recs: list[dict], block: int = 1000) -> list[dict]:
    rows = []
    for b in range(0, len(recs), block):
        dec = [d for r in recs[b:b + block] for d in r["decisions"]]
        ok = sorted(d["e2e"] for d in dec if not d["fb"] and d.get("e2e") is not None)
        fb = [d for d in dec if d["fb"]]
        rows.append({
            "first_hand": b + 1, "hands": len(recs[b:b + block]), "decisions": len(dec), "fallbacks": len(fb),
            "fallback_pct": 100 * len(fb) / max(1, len(dec)), "reasons": dict(Counter(d["fb"] for d in fb)),
            "over_1ms": sum(1 for d in dec if (d.get("e2e") or 0) > BUDGET_NS),
            "ok_e2e_p50_us": ok[len(ok) // 2] / 1e3 if ok else None,
            "ok_e2e_p99_us": ok[int(len(ok) * 0.99)] / 1e3 if ok else None,
            "ok_bot_wall_p50_us": _p50([d["wall"] for d in dec if not d["fb"] and d.get("wall")]),
        })
    return rows


def _p50(xs: list[float]) -> float | None:
    xs = sorted(xs)
    return xs[len(xs) // 2] / 1e3 if xs else None


def gap_audit(recs: list[dict], freeze_s: float = 5.0) -> dict:
    """Wall-clock jumps that the monotonic clock did not see (VM pauses), from records carrying ``m``."""
    stamped = [r for r in recs if "m" in r]
    freezes = []
    for a, b in zip(stamped, stamped[1:]):
        dw, dm = b["t"] - a["t"], b["m"] - a["m"]
        if dw - dm > freeze_s:
            freezes.append({"serial": b["serial"], "wall_s": round(dw, 1), "monotonic_s": round(dm, 1)})
    fb_e2e = sorted(d["e2e"] for r in recs for d in r["decisions"] if d["fb"] and d.get("e2e"))
    return {"vm_freezes": freezes, "fallback_e2e_us_max": fb_e2e[-1] / 1e3 if fb_e2e else None,
            "fallback_e2e_us_p50": fb_e2e[len(fb_e2e) // 2] / 1e3 if fb_e2e else None}


def _estimate(recs: list[dict], terms: list) -> dict:
    cv = ST.regression_adjust([t[0] for t in terms], [t[1] for t in terms])
    return {"hands": len(recs), "raw": ST.summarize([ST.mbb(r["winnings"]) for r in recs]),
            "all_in_adjusted": ST.summarize([ST.mbb(adjusted_chips(r)) for r in recs]), "luck_controlled": cv}


def sensitivity(recs: list[dict], terms: list) -> dict:
    """All hands, hands without any fallback, and a regression estimate of the fallback cost."""
    fb = np.array([any(d["fb"] for d in r["decisions"]) for r in recs])
    idx = np.arange(len(recs))
    out = {"all": _estimate(recs, terms),
           "without_fallback_hands": _estimate([recs[i] for i in idx[~fb]], [terms[i] for i in idx[~fb]]),
           "fallback_hands_only": _estimate([recs[i] for i in idx[fb]], [terms[i] for i in idx[fb]])}
    beta = out["all"]["luck_controlled"]["beta"]
    y = np.array([t[0] - float(np.dot(beta, t[1])) for t in terms])
    length = np.array([len(r["decisions"]) for r in recs], dtype=float)
    x = np.column_stack([np.ones(len(y)), fb.astype(float), length])
    coef = np.linalg.lstsq(x, y, rcond=None)[0]
    resid = y - x @ coef
    se = math.sqrt((np.linalg.inv(x.T @ x) * resid.var(ddof=3))[1, 1])
    share = float(fb.mean())
    out["fallback_cost"] = {
        "hands_with_fallback": int(fb.sum()), "share": share,
        "per_hand_mbb": float(coef[1] * 10), "per_hand_mbb_ci95": [float((coef[1] - 1.96 * se) * 10), float((coef[1] + 1.96 * se) * 10)],
        "bias_on_overall_mbb": float(coef[1] * 10 * share),
        "bias_on_overall_mbb_ci95": [float((coef[1] - 1.96 * se) * 10 * share), float((coef[1] + 1.96 * se) * 10 * share)],
        "controls": "hand length (decisions per hand)"}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--terms", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    recs = load(args.log)
    if args.terms and args.terms.exists():
        terms = pickle.loads(args.terms.read_bytes())
    else:
        terms = [ST.luck_terms(r) for r in recs]
        if args.terms:
            args.terms.write_bytes(pickle.dumps(terms))
    kinds = Counter(d["kind"] for r in recs for d in r["decisions"])
    res = {"hands": len(recs), "decision_kinds": dict(kinds), "fallback_blocks": fallback_blocks(recs),
           "gap_audit": gap_audit(recs), "sensitivity": sensitivity(recs, terms)}
    text = json.dumps(res, indent=1)
    if args.json:
        args.json.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
