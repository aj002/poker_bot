"""Meta-game over the league pool: the empirical mixture, its exploitability, cycles (numpy only).

The pool's empirical game is the matrix ``G[a][b]``: the gain, in points per player-hand,
of a seat playing ``a`` instead of ``b`` when every other seat plays ``b``
(:func:`strategy.league.learner.payoff_matrix`). The eight-seat game is symmetric, so a
meta-strategy is one mixture ``sigma`` from which every opponent seat is drawn. The meta
value of ``a`` against ``sigma`` is linearised over homogeneous tables,

    u(a, sigma) = sum_b sigma_b G[a][b],

which is exact when each table holds one policy type and an approximation for mixed
tables; :mod:`strategy.league.runner` checks the solved mixture against exact mixed-seat
simulation. The mixture is the time average of regret matching on this symmetric game
(a PSRO meta-solver: the average converges to a coarse correlated equilibrium of the
two-role game, and in practice to a symmetric equilibrium here); its exploitability is

    NashConv(sigma) = max_a u(a, sigma) - sum_c sigma_c u(c, sigma).

Cycles are read from the antisymmetric part ``H = (G - G^T) / 2`` (``H[a][b] > 0``: ``a``
does better in ``b``'s world than ``b`` does in ``a``'s): a triple ``a > b > c > a`` with
every edge beyond ``z`` standard errors is reported as an intransitive cycle.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import numpy as np


def solve(G: Sequence[Sequence[float]], *, iters: int = 20000, floor: float = 0.0) -> np.ndarray:
    """Average strategy of regret matching+ on the symmetric game ``G``; ``floor`` mixes in uniform."""
    M = np.asarray(G, float)
    k = M.shape[0]
    if k == 1:
        return np.ones(1)
    regret = np.zeros(k)
    x = np.full(k, 1.0 / k)
    avg = np.zeros(k)
    for t in range(1, iters + 1):
        u = M @ x
        regret = np.maximum(regret + u - x @ u, 0.0)
        s = regret.sum()
        x = regret / s if s > 0 else np.full(k, 1.0 / k)
        avg += t * x
    sigma = avg / avg.sum()
    if floor > 0:
        sigma = (1 - floor) * sigma + floor / k
    return sigma


def values(G: Sequence[Sequence[float]], sigma: Sequence[float]) -> np.ndarray:
    return np.asarray(G, float) @ np.asarray(sigma, float)


def nash_conv(G: Sequence[Sequence[float]], sigma: Sequence[float]) -> float:
    u = values(G, sigma)
    return float(u.max() - np.asarray(sigma, float) @ u)


def antisymmetric(G: Sequence[Sequence[float]]) -> np.ndarray:
    M = np.asarray(G, float)
    return (M - M.T) / 2


def cycles(G: Sequence[Sequence[float]], se: Sequence[Sequence[float]], names: Sequence[str], *,
           z: float = 2.0) -> list[list[str]]:
    """Intransitive triples ``a > b > c > a`` whose every ``H`` edge exceeds ``z`` standard errors."""
    H = antisymmetric(G)
    S = np.asarray(se, float)
    E = np.sqrt(S ** 2 + S.T ** 2) / 2
    beats = (H > z * np.where(E > 0, E, math.inf)) & (E > 0)
    out = []
    for a, b, c in itertools.permutations(range(len(names)), 3):
        if a < b and a < c and beats[a, b] and beats[b, c] and beats[c, a]:
            out.append([names[a], names[b], names[c]])
    return out


def ranking(G: Sequence[Sequence[float]], names: Sequence[str]) -> list[tuple[str, float]]:
    """Names by mean antisymmetric advantage over the rest of the pool."""
    H = antisymmetric(G)
    k = len(names)
    score = H.sum(axis=1) / max(1, k - 1)
    return sorted(zip(names, score.tolist()), key=lambda t: -t[1])
