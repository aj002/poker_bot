"""Null A/B calibration: identical policies must not produce false gains above the declared rate.

Real tournaments (:func:`calibrate`)
    Arms A and B run the same artifact with differently salted policy seeds, so the
    true difference is zero while policy noise is independent. The pooled blocks are
    split into ``replicates`` disjoint experiments of ``blocks_per_replicate`` blocks per
    cell (block ``k`` of every cell goes to replicate ``k // blocks_per_replicate``). Each
    replicate is analysed exactly like a real A/B. A *false gain* is a replicate whose
    interval lies above zero; at a two-sided level ``1 - alpha`` its declared rate is
    ``alpha / 2``. A *false promotion* also clears the minimum effect. Calibration
    fails when the observed count is improbably high: ``P(X >= observed) < 0.01``
    under ``Binomial(replicates, alpha / 2)``.

    For contrast, the unpaired Welch interval on table-level points ignores pairing
    and clustering; its width shows what pairing buys.

Exact coupling (:func:`exact_coupling`)
    The same artifact with the same policy seed in both arms must give identical
    tables, so every block difference is exactly zero. This verifies that
    environment and policy random streams are separate and shared as intended.
    Sandboxed tables where either arm had a timeout are reported, not compared.

Synthetic (:func:`synthetic_calibration`)
    The same statistics on simulated paired tournaments with shared card luck and
    independent policy noise, cheap enough for thousands of replicates in tests.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping, Sequence
from typing import Any

from strategy.eval.paired import CANDIDATE, BlockPair, CellResult, pair_blocks
from strategy.eval.stats import binomial_sf, bootstrap_mean, clopper_pearson, mean, paired_t, welch_t
from strategy.eval.variance import DEFAULT_FOLDS, DEFAULT_TERMS, adjust

CALIBRATION_P = 0.01


def _method_rows(intervals: dict[str, list[tuple[float | None, float | None]]], replicates: int, alpha: float,
                 min_effect: float) -> dict[str, Any]:
    declared = alpha / 2
    out = {}
    for name, rows in intervals.items():
        gains = sum(1 for lo, _ in rows if lo is not None and lo > 0)
        losses = sum(1 for _, hi in rows if hi is not None and hi < 0)
        promos = sum(1 for lo, _ in rows if lo is not None and lo > min_effect)
        widths = [hi - lo for lo, hi in rows if lo is not None and hi is not None]
        p = binomial_sf(gains, replicates, declared)
        out[name] = {
            "false_gains": gains,
            "false_losses": losses,
            "false_promotions": promos,
            "false_gain_rate": gains / replicates,
            "false_gain_rate_ci": list(clopper_pearson(gains, replicates)),
            "two_sided_rejection_rate": (gains + losses) / replicates,
            "mean_interval_width": mean(widths) if widths else None,
            "binomial_p_vs_declared": p,
            "calibrated": p >= CALIBRATION_P,
        }
    return out


def calibrate(results: Sequence[CellResult], *, replicates: int, blocks_per_replicate: int, alpha: float,
              min_effect: float, resamples: int = 1000, seed: str = "null",
              variance_reduction: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """With ``variance_reduction`` (a spec's section) every replicate is also analysed with the
    luck-adjusted estimator (method ``luck_cv``), cross-fitted within the replicate."""
    pairs: list[BlockPair] = []
    tables: dict[tuple[str, int], tuple[list[float], list[float]]] = {}
    for c in results:
        p, _ = pair_blocks(c)
        pairs += p
        for arm, idx in (("A", 0), ("B", 1)):
            for r in c.runs[arm].records:
                if r.valid:
                    tables.setdefault((c.cell.cell_id, r.block), ([], []))[idx].append(r.entrants[CANDIDATE].points)
    level = 1 - alpha
    intervals: dict[str, list[tuple[float | None, float | None]]] = {"paired_t": [], "cluster_bootstrap": [],
                                                                       "unpaired_welch_tables": []}
    vr = variance_reduction
    if vr is not None and pairs and all(p.luck_a is not None and p.luck_b is not None for p in pairs):
        intervals["luck_cv"] = []
    estimates = []
    for k in range(replicates):
        rep = [p for p in pairs if p.block // blocks_per_replicate == k]
        if len(rep) < 2:
            for rows in intervals.values():
                rows.append((None, None))
            continue
        diffs = [p.d for p in rep]
        estimates.append(mean(diffs))
        t = paired_t(diffs, level)
        intervals["paired_t"].append((t.lo, t.hi))
        strata = [p.cell_id for p in rep]
        b = bootstrap_mean(diffs, seed=f"{seed}|null|{k}", strata=strata if len(set(strata)) > 1 else None,
                           resamples=resamples, level=level)
        intervals["cluster_bootstrap"].append((b.lo, b.hi))
        ta = [x for p in rep for x in tables[(p.cell_id, p.block)][0]]
        tb = [x for p in rep for x in tables[(p.cell_id, p.block)][1]]
        w = welch_t(ta, tb, level)
        intervals["unpaired_welch_tables"].append((w.lo, w.hi))
        if "luck_cv" in intervals:
            z = [{t: p.luck_b[t] - p.luck_a[t] for t in p.luck_a} for p in rep]
            v = adjust(diffs, z, strata, [p.block for p in rep], terms=vr.get("terms", DEFAULT_TERMS),
                       coefficients=vr.get("coefficients", "crossfit"), folds=int(vr.get("folds", DEFAULT_FOLDS)),
                       stratify=bool(vr.get("stratify", False)), alpha=alpha)
            intervals["luck_cv"].append((v["lo"], v["hi"]))
    return {
        "replicates": replicates,
        "blocks_per_replicate": blocks_per_replicate,
        "cells": len(results),
        "alpha": alpha,
        "declared_false_gain_rate": alpha / 2,
        "min_effect": min_effect,
        "replicate_mean_differences": estimates,
        "methods": _method_rows(intervals, replicates, alpha, min_effect),
    }


def exact_coupling(results: Sequence[CellResult]) -> dict[str, Any]:
    """Every entrant's points must match between arms, table for table.

    A timeout at any seat forces a fold in that arm only, after which the tables
    legitimately diverge, so sandboxed tables with a timeout in either arm are counted
    separately and excluded from the identity check.
    """
    blocks: set[tuple[str, int]] = set()
    clean, timed, diverged = [], 0, 0
    for c in results:
        by_b = {(r.block, r.rotation): r for r in c.runs["B"].records if r.valid}
        for ra in c.runs["A"].records:
            rb = by_b.get((ra.block, ra.rotation))
            if not ra.valid or rb is None:
                continue
            blocks.add((c.cell.cell_id, ra.block))
            diff = max((abs(rb.points[e] - pts) for e, pts in ra.points.items()), default=0.0)
            if ra.timeouts or rb.timeouts:
                timed += 1
                diverged += diff != 0.0
            else:
                clean.append(diff)
    worst = max(clean, default=0.0)
    return {"blocks": len(blocks), "tables": len(clean) + timed, "tables_with_timeouts": timed,
            "diverged_after_timeout": diverged, "max_abs_difference": worst,
            "identical": bool(clean) and worst == 0.0}


# ------------------------------------------------------------------ synthetic


def _simulate_block(rng: random.Random, players: int, payouts: Sequence[int], luck: float,
                    noise: float) -> tuple[list[float], list[float]]:
    """Candidate points per rotation for two arms sharing card luck, with independent policy noise."""
    out_a, out_b = [], []
    for _ in range(players):
        shared = [rng.gauss(0.0, luck) for _ in range(players)]
        for out in (out_a, out_b):
            score = [shared[i] + rng.gauss(0.0, noise) for i in range(players)]
            rank = sorted(range(players), key=lambda i: -score[i]).index(0)
            out.append(float(payouts[rank]))
    return out_a, out_b


def synthetic_calibration(*, replicates: int, blocks: int, players: int = 8,
                          payouts: Sequence[int] = (50, 30, 20, 0, 0, 0, 0, 0), alpha: float = 0.05,
                          min_effect: float = 0.0, luck: float = 1.0, noise: float = 1.0, resamples: int = 500,
                          seed: str = "synthetic-null", bootstrap: bool = True) -> dict[str, Any]:
    rng = random.Random(hashlib.sha256(seed.encode()).digest())
    level = 1 - alpha
    intervals: dict[str, list[tuple[float | None, float | None]]] = {"paired_t": [], "unpaired_welch_tables": []}
    if bootstrap:
        intervals["cluster_bootstrap"] = []
    for k in range(replicates):
        diffs, ta, tb = [], [], []
        for _ in range(blocks):
            a, b = _simulate_block(rng, players, payouts, luck, noise)
            diffs.append(mean(b) - mean(a))
            ta += a
            tb += b
        t = paired_t(diffs, level)
        intervals["paired_t"].append((t.lo, t.hi))
        if bootstrap:
            bi = bootstrap_mean(diffs, seed=f"{seed}|{k}", resamples=resamples, level=level)
            intervals["cluster_bootstrap"].append((bi.lo, bi.hi))
        w = welch_t(ta, tb, level)
        intervals["unpaired_welch_tables"].append((w.lo, w.hi))
    return {"replicates": replicates, "blocks_per_replicate": blocks, "alpha": alpha,
            "declared_false_gain_rate": alpha / 2, "min_effect": min_effect,
            "methods": _method_rows(intervals, replicates, alpha, min_effect)}
