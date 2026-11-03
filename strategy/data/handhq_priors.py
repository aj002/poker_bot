"""Population priors fitted on real six-max players (HandHQ), and the style bots measured against them.

Input: :mod:`strategy.data.handhq_stats` output (per-player counts and belief-layer counts).
A *player* is one obfuscated name at one stake, and only players with at least
:data:`MIN_HANDS` counted hands enter the per-player fits, so every player counts once
however much they played.

**Frequencies.** Per stake and pooled: hand-weighted rates (the population as a table
meets it) and the 5/25/50/75/95th percentiles over players (how much players differ).

**Dirichlet priors** (empirical Bayes, Minka's fixed point on the Dirichlet-multinomial):

* per opportunity, the per-player category counts give the prior mean ``m`` (the mean
  player, not the hand-weighted mean) and the concentration ``kappa_player``: how many
  pseudo-actions a new opponent's prior is worth before their own actions move it;
* per opportunity and depth bucket, the context cells of the pooled population give
  ``kappa_cell``: how far a cell departs from its opportunity x depth rate;
* per opportunity, the three depth buckets of each player give ``kappa_depth``.

These map onto :class:`strategy.beliefs.posterior.ActionPosterior` (``prior`` = ``m``;
``kappa`` = (table pool, seat, seat x cell) = (``5 kappa_player + 4``, ``kappa_player``,
``kappa_cell``), the pool being the mean of five opponents) and onto the exploit
likelihood table's hierarchy (:data:`strategy.exploit.models.TABLE_PRIOR`: cell ->
opportunity x depth -> opportunity -> population, with the population base measure).

    python -m strategy.data.handhq_priors fit --stats artifacts/handhq/stats --out strategy/population/data/handhq-priors.json
    python -m strategy.data.handhq_priors census --tables 600 --jobs 7 --out artifacts/handhq/census-styles.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from strategy.data.handhq_stats import COLUMNS, DEPTH_RANGE, N_CAT, N_DEPTH, N_OPP, STATS

MIN_HANDS = 300
PERCENTILES = (5, 25, 50, 75, 95)
SIX_MAX_100BB = {"starting_stack": 2000, "hands_per_level": 200, "hand_cap": 200}
"""Census format closest to the HandHQ hands: six seats, 10/20 blinds that never rise (100 BB)."""
COMPARED = ("vpip", "pfr", "three_bet", "fold_to_reraise", "limp", "cold_call", "cbet", "aggression", "allin_share",
            "min_raise_share", "iso", "squeeze", "xr_share")
TYPES = {
    "nit": "VPIP under 14%",
    "regular": "VPIP 14-30% and PFR at least 60% of VPIP",
    "loose_aggressive": "VPIP over 30% and PFR at least 60% of VPIP",
    "passive": "VPIP 14% or more and PFR under 60% of VPIP",
}


# ------------------------------------------------------------------ per-player rates


def rates(counts: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``stat -> (hits, samples)`` columns of the per-player count matrix."""
    col = {k: counts[:, i] for i, k in enumerate(COLUMNS)}
    out = {"vpip": (col["vpip"], col["hands"]), "pfr": (col["pfr"], col["hands"]),
           "fold_to_reraise": (col["folded"], col["faced"]),
           "aggression": (col["bets_raises"], col["calls"])}
    for s in STATS:
        out[s] = (col[f"{s}.hit"], col[f"{s}.n"])
    return out


def player_type(vpip: np.ndarray, pfr: np.ndarray) -> np.ndarray:
    ratio = np.where(vpip > 0, pfr / np.maximum(vpip, 1e-9), 0.0)
    out = np.where(vpip < 0.14, "nit", np.where(ratio < 0.6, "passive",
                                                np.where(vpip > 0.30, "loose_aggressive", "regular")))
    return out


def _bands(hit: np.ndarray, n: np.ndarray, keep: np.ndarray) -> dict[str, Any]:
    """Pooled ``hit / n`` and its percentiles over the ``keep`` players."""
    v = hit[keep] / np.maximum(1, n[keep])
    return {"pooled": float(hit.sum() / max(1, n.sum())), "players": int(keep.sum()),
            "percentiles": {str(p): float(np.percentile(v, p)) for p in PERCENTILES} if v.size else {}}


def frequencies(z: Any, min_hands: int = MIN_HANDS) -> dict[str, Any]:
    counts, stake = z["counts"], z["stake"]
    stakes = [str(s) for s in z["stakes"]]
    hands = counts[:, COLUMNS.index("hands")]
    out: dict[str, Any] = {}
    for si, name in [*enumerate(stakes), (-1, "all")]:
        sel = np.ones(len(stake), bool) if si < 0 else stake == si
        r = rates(counts[sel])
        regular = hands[sel] >= min_hands
        body: dict[str, Any] = {"players": int(sel.sum()), "fitted_players": int(regular.sum()),
                                "hands_counted": int(hands[sel].sum()), "stats": {}}
        for s in COMPARED:
            hit, n = r[s]
            body["stats"][s] = _bands(hit, n, regular if s in ("vpip", "pfr", "aggression") else n >= 30)
        v, p = r["vpip"][0] / np.maximum(1, hands[sel]), r["pfr"][0] / np.maximum(1, hands[sel])
        types = player_type(v[regular], p[regular])
        vol = hands[sel][regular]
        body["types"] = {t: {"players": float((types == t).mean()) if types.size else 0.0,
                             "hands": float(vol[types == t].sum() / max(1, vol.sum()))} for t in TYPES}
        if name == "all":
            sub = counts[sel][regular]
            rr = rates(sub)
            body["by_type"] = {t: {s: _bands(rr[s][0] * (types == t), rr[s][1] * (types == t),
                                             (types == t) & (rr[s][1] >= (1 if s in ("vpip", "pfr", "aggression")
                                                                          else 30)))
                                   for s in COMPARED} for t in TYPES}
        out[name] = body
    return out


# ------------------------------------------------------------------ Dirichlet fits


def digamma(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).copy()
    out = np.zeros_like(x)
    small = x < 6
    while small.any():
        out[small] -= 1.0 / x[small]
        x[small] += 1.0
        small = x < 6
    f = 1.0 / (x * x)
    return out + np.log(x) - 0.5 / x - f * (1 / 12 - f * (1 / 120 - f * (1 / 252 - f * (1 / 240 - f / 132))))


def fit_dirichlet(counts: np.ndarray, *, iters: int = 500, tol: float = 1e-7) -> np.ndarray:
    """Dirichlet-multinomial maximum-likelihood ``alpha`` from ``(units, K)`` counts (Minka 2000)."""
    c = counts[counts.sum(axis=1) > 0].astype(np.float64)
    n = c.sum(axis=1)
    p = c / n[:, None]
    m = p.mean(axis=0)
    var = p.var(axis=0).sum()
    s = max(0.5, (m * (1 - m)).sum() / max(var, 1e-9) - 1.0)
    alpha = np.maximum(m * s, 1e-3)
    for _ in range(iters):
        a0 = alpha.sum()
        den = (digamma(n + a0) - digamma(a0)).sum()
        num = (digamma(c + alpha) - digamma(alpha)).sum(axis=0)
        new = np.maximum(alpha * num / den, 1e-6)
        if np.abs(new - alpha).max() < tol * new.max():
            alpha = new
            break
        alpha = new
    return alpha


def _lgamma(x: np.ndarray) -> np.ndarray:
    return np.vectorize(math.lgamma, otypes=[np.float64])(x)


def fit_concentration(counts: np.ndarray, mean: np.ndarray) -> float:
    """MLE of ``kappa`` for Dirichlet(kappa * mean) given ``(units, K)`` counts (golden-section on log kappa).

    ``mean`` is one ``K`` vector for every unit or a ``(units, K)`` array (each unit's own parent rate)."""
    keep = counts.sum(axis=1) > 0
    c = counts[keep].astype(np.float64)
    n = c.sum(axis=1)
    m = np.broadcast_to(np.maximum(mean, 1e-9), counts.shape)[keep]
    m = m / m.sum(axis=1, keepdims=True)

    def ll(logk: float) -> float:
        k = math.exp(logk)
        return float((math.lgamma(k) - _lgamma(n + k)).sum() + (_lgamma(c + k * m) - _lgamma(k * m)).sum())

    lo, hi = math.log(0.1), math.log(1e5)
    g = (math.sqrt(5) - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    fa, fb = ll(a), ll(b)
    for _ in range(60):
        if fa > fb:
            hi, b, fb = b, a, fa
            a = hi - g * (hi - lo)
            fa = ll(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + g * (hi - lo)
            fb = ll(b)
    return math.exp((lo + hi) / 2)


def fit(stats: Path, min_hands: int = MIN_HANDS) -> dict[str, Any]:
    from strategy.beliefs.posterior import CATEGORIES, OPPORTUNITIES

    z = np.load(stats / "players.npz")
    cz = np.load(stats / "cells.npz")
    summary = json.loads((stats / "summary.json").read_text())
    counts, belief = z["counts"], z["belief"]
    hands = counts[:, COLUMNS.index("hands")]
    regular = hands >= min_hands
    cells = cz["cells"].sum(axis=0)
    depth = (np.arange(cells.shape[1]) // 4) % 3
    out: dict[str, Any] = {
        "source": {"dataset": summary["dataset"], "commit": summary["commit"], "licence": summary["licence"],
                   "stakes": [str(s) for s in z["stakes"]], "depth_range_bb": list(DEPTH_RANGE),
                   "min_hands": min_hands, "players": int(len(hands)), "fitted_players": int(regular.sum()),
                   "hands_counted": int(hands.sum()),
                   "replayed_hands": {k: v.get("replayed", 0) for k, v in summary["stakes"].items()}},
        "frequencies": frequencies(z, min_hands),
        "opportunities": list(OPPORTUNITIES),
        "categories": {OPPORTUNITIES[o]: list(c) for o, c in CATEGORIES.items()},
    }
    per_opp: dict[str, Any] = {}
    for o, name in enumerate(OPPORTUNITIES):
        pc = belief[regular, o].sum(axis=1)
        alpha = fit_dirichlet(pc)
        pooled = belief[:, o].sum(axis=(0, 1)).astype(float)
        own = (belief[regular, o].sum(axis=1) + 1.0)
        own = np.repeat(own / own.sum(axis=1, keepdims=True), N_DEPTH, axis=0)
        units = belief[regular, o].reshape(-1, N_CAT)
        big = units.sum(axis=1) >= 20
        kd = [fit_concentration(units[big], own[big])]
        kc = []
        for d in range(N_DEPTH):
            sel = cells[o][depth == d]
            sel = sel[sel.sum(axis=1) >= 50]
            if len(sel) >= 3:
                kc.append((fit_concentration(sel, sel.sum(axis=0) / sel.sum()), int(sel.sum())))
        per_opp[name] = {
            "mean_player": (alpha / alpha.sum()).tolist(), "kappa_player": float(alpha.sum()),
            "pooled": (pooled / pooled.sum()).tolist(), "actions": int(pooled.sum()),
            "kappa_depth_within_player": kd[0],
            "kappa_cell": float(np.exp(sum(math.log(k) * w for k, w in kc) / sum(w for _, w in kc))) if kc else None,
            "by_depth_pooled": [(belief[:, o, d].sum(0) / max(1, belief[:, o, d].sum())).tolist()
                                for d in range(N_DEPTH)],
        }
    out["dirichlet"] = per_opp
    kp = float(np.exp(np.mean([math.log(v["kappa_player"]) for v in per_opp.values()])))
    kcell = float(np.exp(np.mean([math.log(v["kappa_cell"]) for v in per_opp.values() if v["kappa_cell"]])))
    kdep = float(np.exp(np.mean([math.log(v["kappa_depth_within_player"]) for v in per_opp.values()])))
    out["action_posterior"] = {"prior": {o: per_opp[n]["mean_player"] for o, n in enumerate(OPPORTUNITIES)},
                               "kappa": [5 * kp + 4, kp, kcell]}
    out["likelihood_table"] = {"table_prior": [kcell, kdep, kp],
                               "base_measure": [per_opp[n]["mean_player"] for n in OPPORTUNITIES],
                               "cells": cells.tolist()}
    return out


# ------------------------------------------------------------------ style bots vs real players


def _style(sid: str):
    from arena.agents.styles.engine import StyleBot

    return StyleBot(sid)


def census_styles(tables: int, *, jobs: int, seed: str = "handhq-census-v1", families: list[str] | None = None,
                  styles: bool = True) -> dict[str, Any]:
    """The six style bots and every family's train members (pop-v1 and HandHQ) at six-seat, 100 BB
    tables with flat blinds; ``families`` restricts the members."""
    import functools
    import multiprocessing

    from arena.agents.styles.engine import STYLE_IDS
    from strategy.population.handhq import handhq_registry
    from strategy.population.measure import Tally, _run_tables, seatings

    reg = handhq_registry()
    entries: list[Any] = [(f"style:{s}", functools.partial(_style, s)) for s in STYLE_IDS] if styles else []
    entries += [(f"family:{m.family}|{m.member_id}", m.spec) for m in sorted(reg.partition("train"),
                                                                           key=lambda m: (m.family, m.index))
                if families is None or m.family in families]
    plan = list(enumerate(seatings(len(entries), tables=tables, seed=seed, seats=6)))
    chunks = [(entries, plan[j::jobs], seed, "practice_nlhe", SIX_MAX_100BB) for j in range(jobs)]
    with multiprocessing.get_context("fork").Pool(jobs) as pool:
        parts = pool.map(_run_tables, chunks)
    by_key: dict[str, Tally] = {}
    for part in parts:
        for key, t in part.items():
            group = key.split("|")[0]
            by_key.setdefault(group, Tally()).merge(t)
    return {k: {s: list(v) for s, v in t.stats().items()} for k, t in sorted(by_key.items())}


def compare(census: dict[str, Any], priors: dict[str, Any]) -> dict[str, Any]:
    """Each bot's statistic placed in the real-player distribution (percentile band and pooled rate)."""
    real = priors["frequencies"]["all"]["stats"]
    out: dict[str, Any] = {}
    for key, stats in census.items():
        row = {}
        for s in COMPARED:
            v = stats.get(s, [None, 0])[0]
            pc = real[s].get("percentiles", {})
            if v is None or not pc:
                continue
            lo, hi = pc["5"], pc["95"]
            row[s] = {"bot": round(v, 4), "real_p5": round(lo, 4), "real_median": round(pc["50"], 4),
                      "real_p95": round(hi, 4), "inside": lo <= v <= hi}
        out[key] = row
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--stats", type=Path, required=True)
    f.add_argument("--min-hands", type=int, default=MIN_HANDS)
    f.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("census")
    c.add_argument("--tables", type=int, default=600)
    c.add_argument("--jobs", type=int, default=7)
    c.add_argument("--priors", type=Path, default=Path("strategy/population/data/handhq-priors.json"))
    c.add_argument("--families", nargs="*", default=None)
    c.add_argument("--no-styles", action="store_true")
    c.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "fit":
        body = fit(a.stats, a.min_hands)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(body, indent=1) + "\n")
        print(json.dumps({k: body[k] for k in ("source", "action_posterior")} | {
            "table_prior": body["likelihood_table"]["table_prior"],
            "dirichlet": {k: {x: v[x] for x in ("mean_player", "kappa_player", "kappa_cell",
                                                "kappa_depth_within_player")} for k, v in body["dirichlet"].items()}},
            indent=1))
    else:
        census = census_styles(a.tables, jobs=a.jobs, families=a.families, styles=not a.no_styles)
        priors = json.loads(a.priors.read_text())
        body = {"format": {"seats": 6, **SIX_MAX_100BB}, "tables": a.tables, "census": census,
                "compare": compare(census, priors)}
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(body, indent=1) + "\n")
        for key, row in body["compare"].items():
            print(f"{key:32s} " + " ".join(f"{s}={r['bot']:.2f}{'' if r['inside'] else '*'}" for s, r in row.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
