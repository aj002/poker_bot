"""Variance-reduced paired estimate: block differences minus cross-fitted multiples of zero-mean luck terms.

Estimator
    For block ``i`` with points difference ``D_i`` and luck difference
    ``Z_i = X_i(B) - X_i(A)`` (each arm's :mod:`strategy.eval.luck` terms averaged over the
    block's rotations), ``Y_i = D_i - b . Z_i`` and the estimate is ``mean(Y)``. Every luck
    term has mean exactly zero whatever the bots do, so ``E[Y_i] = E[D_i]`` for any ``b``
    fixed independently of block ``i``:

    * ``coefficients = "crossfit"``: blocks are split into ``folds`` groups by block index
      and ``b`` for a fold is the least-squares fit of ``D`` on ``Z`` (with an intercept)
      over the other folds. The estimate is unbiased whatever the fit, and the interval
      is the Student t interval on the ``Y_i`` with one degree of freedom less per term.
    * fixed coefficients (``{term: b}``, for example from a pilot run) are unbiased trivially.

Stratification (opt-in)
    With ``stratify`` the variance is the within-lineup (cell) variance. That interval
    is conditional on the lineups sampled for the run: it drops the uncertainty of the
    lineup sample, so it answers a narrower question than the default and is never the
    default.

The report keeps the raw paired estimate next to the adjusted one, the variance
reduction factor ``(SE_raw / SE_adjusted)^2``, the minimum detectable effect of both and,
as an unbiasedness diagnostic, the mean and standard error of every luck difference,
which must be consistent with zero.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from strategy.eval.luck import TERMS
from strategy.eval.stats import minimum_detectable_effect, normal_ppf, t_cdf, t_ppf

DEFAULT_TERMS = ("allin", "flop_pot", "turn_pot", "river_pot")
DEFAULT_FOLDS = 10


def block_luck(records: Sequence[Any], entrant: int) -> dict[str, float] | None:
    """Mean of each luck term over a block's rotations; None when a record has no luck terms."""
    rows = [r.entrants[entrant].luck for r in records]
    if not rows or any(x is None for x in rows):
        return None
    return {t: math.fsum(x[t] for x in rows) / len(rows) for t in TERMS}


def _fit(d: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Least-squares slopes of ``d`` on ``z`` with an intercept (minimum-norm when rank deficient)."""
    x = np.column_stack([np.ones(len(d)), z])
    return np.linalg.lstsq(x, d, rcond=None)[0][1:]


def _se(y: np.ndarray, cells: Sequence[str], stratify: bool, terms: int) -> tuple[float, int]:
    n = len(y)
    if not stratify:
        return float(np.std(y, ddof=1) / math.sqrt(n)), n - 1 - terms
    groups: dict[str, list[float]] = {}
    for v, c in zip(y.tolist(), cells):
        groups.setdefault(c, []).append(v)
    var = sum(len(g) * float(np.var(g, ddof=1)) for g in groups.values() if len(g) > 1) / n ** 2
    return math.sqrt(var), n - len(groups) - terms


def _interval(est: float, se: float, df: int, level: float) -> dict[str, Any]:
    if df < 1 or se <= 0:
        return {"se": se, "df": df, "lo": None, "hi": None, "t_stat": None, "p_value": None}
    tc = t_ppf(0.5 + level / 2, df)
    t = est / se
    return {"se": se, "df": df, "lo": est - tc * se, "hi": est + tc * se, "t_stat": t,
            "p_value": 2.0 * (1.0 - t_cdf(abs(t), df))}


def adjust(d: Sequence[float], z: Sequence[Mapping[str, float]], cells: Sequence[str], blocks: Sequence[int], *,
           terms: Sequence[str] = DEFAULT_TERMS, coefficients: str | Mapping[str, float] = "crossfit",
           folds: int = DEFAULT_FOLDS, stratify: bool = False, alpha: float = 0.05,
           power: float = 0.80) -> dict[str, Any]:
    """The luck-adjusted estimate of ``mean(d)`` (see the module docstring)."""
    n = len(d)
    if n < 2:
        raise ValueError("need at least two blocks")
    dv = np.asarray(d, dtype=np.float64)
    zv = np.array([[row[t] for t in terms] for row in z], dtype=np.float64).reshape(n, len(terms))
    fold = np.asarray(blocks, dtype=np.int64) % folds
    if coefficients == "crossfit":
        betas = {}
        y = np.empty(n)
        for k in sorted(set(fold.tolist())):
            out = fold == k
            b = _fit(dv[~out], zv[~out]) if (~out).sum() > len(terms) + 1 else np.zeros(len(terms))
            betas[int(k)] = b
            y[out] = dv[out] - zv[out] @ b
        fitted = len(terms)
        beta = {"by_fold": {str(k): dict(zip(terms, b.tolist())) for k, b in betas.items()}}
    else:
        b = np.array([float(coefficients[t]) for t in terms])
        y = dv - zv @ b
        fitted = 0
        beta = {"fixed": dict(zip(terms, b.tolist()))}
    beta["full_sample"] = dict(zip(terms, _fit(dv, zv).tolist())) if n > len(terms) + 1 else None
    level = 1 - alpha
    est = float(y.mean())
    se, df = _se(y, cells, stratify, fitted)
    se_raw, df_raw = _se(dv, cells, stratify, 0)
    se_primary = float(np.std(dv, ddof=1) / math.sqrt(n))
    diag = {}
    for j, t in enumerate(terms):
        col = zv[:, j]
        s = float(np.std(col, ddof=1) / math.sqrt(n))
        diag[t] = {"mean_difference": float(col.mean()), "se": s, "t_stat": float(col.mean() / s) if s > 0 else None,
                   "corr_with_difference": float(np.corrcoef(col, dv)[0, 1]) if col.std() > 0 and dv.std() > 0
                   else None}
    corr = dv - y
    return {
        "method": "luck_cv",
        "terms": list(terms),
        "coefficients": "crossfit" if coefficients == "crossfit" else "fixed",
        "folds": folds if coefficients == "crossfit" else None,
        "stratify": stratify,
        "blocks": n,
        "mean_difference": est,
        "sd_adjusted": float(np.std(y, ddof=1)),
        **_interval(est, se, df, level),
        "raw": {"mean_difference": float(dv.mean()), "sd": float(np.std(dv, ddof=1)), "se_paired_t": se_primary,
                **({"se_stratified": se_raw, "df_stratified": df_raw} if stratify else {})},
        "variance_reduction_factor": (se_primary / se) ** 2 if se > 0 else None,
        "minimum_detectable_effect": minimum_detectable_effect(se * math.sqrt(n), n, alpha=alpha, power=power),
        "minimum_detectable_effect_raw": minimum_detectable_effect(se_primary * math.sqrt(n), n, alpha=alpha,
                                                                   power=power),
        "beta": beta,
        "luck_differences": diag,
        "correction": {"mean": float(corr.mean()), "se": float(np.std(corr, ddof=1) / math.sqrt(n))},
        "level": level,
    }


def blocks_needed(sd: float, delta: float, *, alpha: float = 0.05, power: float = 0.80) -> int:
    """Blocks for an 80%-power two-sided test of ``delta`` at block SD ``sd`` (normal approximation)."""
    return math.ceil(((normal_ppf(1 - alpha / 2) + normal_ppf(power)) * sd / delta) ** 2)
