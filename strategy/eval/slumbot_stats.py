"""Statistics for Slumbot matches: mbb/hand with confidence intervals and all-in EV adjustment.

Results are in milli-big-blinds per hand (mbb/hand): ``winnings / 100 * 1000``.

All-in EV adjustment. When a hand's last betting action is a call of an all-in, both players
have committed the whole 20,000 stack and the rest of the board is pure luck. Both hole cards
are shown at that showdown, so the hand's expected result given everything known when the last
chip went in is ``(2 * equity - 1) * 20000`` for the hero, with ties counted as half. Replacing
the realised result by that expectation keeps the estimator unbiased, because the board cards
still to come are independent of everything the players chose, and removes the luck of the
runout. It does not touch hands that did not end in an all-in call, and it is *not* AIVAT:
Slumbot's hole cards are visible only at showdown, so the full control variate is unavailable.
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Iterable, Sequence

import numpy as np

from arena.domain.cards import RANKS, SUITS
from strategy.eval.kernels import HAVE_NUMBA, _preflop_exact, _preflop_mc, score7
from strategy.eval.slumbot import BIG_BLIND, STACK, STREET_BOARD, tokenize

PREFLOP_SAMPLES = 20_000  # Monte Carlo boards when numba is missing

_CARD = {r + s: 4 * i + j for i, r in enumerate(RANKS) for j, s in enumerate(SUITS)}


def _eval_boards(hero: tuple[int, int], villain: tuple[int, int], boards: Iterable[Sequence[int]]) -> tuple[float, int]:
    wins = 0.0
    n = 0
    for b in boards:
        h = score7(np.array([*hero, *b], dtype=np.int64))
        v = score7(np.array([*villain, *b], dtype=np.int64))
        wins += 1.0 if h > v else 0.5 if h == v else 0.0
        n += 1
    return wins, n


def preflop_equity_mc(hero: Sequence[str], villain: Sequence[str], *, samples: int = 20_000, seed: int = 0) -> float:
    """Unbiased Monte Carlo preflop equity (a fixed ``seed`` makes it repeatable)."""
    h = [_CARD[c] for c in hero]
    v = [_CARD[c] for c in villain]
    rest = np.array([c for c in range(52) if c not in set(h) | set(v)], dtype=np.int64)
    if HAVE_NUMBA:
        return float(_preflop_mc(h[0], h[1], v[0], v[1], rest, samples, seed))
    rng = random.Random(seed)
    wins, n = _eval_boards(tuple(h), tuple(v), (rng.sample(list(rest), 5) for _ in range(min(samples, 2000))))
    return wins / n


def equity(hero: Sequence[str], villain: Sequence[str], board: Sequence[str], *, seed: int = 0) -> float:
    """The hero's share of the pot against ``villain`` given ``board`` (ties count half).

    Exact for the flop, turn and river; exact for a preflop all-in when numba is available,
    otherwise Monte Carlo over :data:`PREFLOP_SAMPLES` boards with a fixed ``seed``.
    """
    h = tuple(_CARD[c] for c in hero)
    v = tuple(_CARD[c] for c in villain)
    b = tuple(_CARD[c] for c in board)
    used = set(h) | set(v) | set(b)
    if len(used) != len(h) + len(v) + len(b):
        raise ValueError("duplicate cards")
    rest = [c for c in range(52) if c not in used]
    need = 5 - len(b)
    if need == 0:
        wins, n = _eval_boards(h, v, [b])
    elif need <= 2:
        wins, n = _eval_boards(h, v, ((*b, *extra) for extra in itertools.combinations(rest, need)))
    elif HAVE_NUMBA:
        wins, n = _preflop_exact(h[0], h[1], v[0], v[1], np.array(rest, dtype=np.int64))
    else:
        rng = random.Random(seed)
        wins, n = _eval_boards(h, v, (rng.sample(rest, 5) for _ in range(PREFLOP_SAMPLES)))
    return wins / n


def all_in_street(action: str) -> int | None:
    """The street on which a call of an all-in ended the betting, or None.

    True when the last action is a call and the bet it called took the bettor's total
    commitment (earlier streets plus this street's bet-to) to the whole stack.
    """
    tokens = tokenize(action)
    acts = [t for t in tokens if t != "/"]
    if not acts or acts[-1] != "c":
        return None
    prior, street_to, street = 0, BIG_BLIND, 0
    seen = 0
    for t in tokens:
        if t == "/":
            prior += street_to
            street_to = 0
            street += 1
            continue
        seen += 1
        if t[0] == "b":
            street_to = int(t[1:])
        if seen == len(acts):
            return street if prior + street_to >= STACK else None
    return None


def adjusted_winnings(rec: dict) -> float:
    """The hand's result in chips with the all-in runout replaced by its expectation.

    ``rec`` is a match record with ``action``, ``hole``, ``bot_hole``, ``board`` and
    ``winnings``. Hands that did not end in an all-in call return the realised result.
    """
    street = all_in_street(rec["action"])
    bot_hole = rec.get("bot_hole")
    if street is None or street >= 3 or not bot_hole:
        return float(rec["winnings"])
    board = rec["board"][: STREET_BOARD[street]]
    if len(board) != STREET_BOARD[street]:
        return float(rec["winnings"])
    return (2.0 * equity(rec["hole"], bot_hole, board, seed=rec.get("n", 0)) - 1.0) * STACK


def mbb(chips: float) -> float:
    return chips / BIG_BLIND * 1000.0


def summarize(values: Sequence[float]) -> dict[str, float]:
    """Mean, standard deviation and 95% normal-theory CI of per-hand ``mbb`` values."""
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "sd": float("nan"), "se": float("nan"),
                "lo": float("nan"), "hi": float("nan")}
    mean = math.fsum(values) / n
    var = math.fsum((x - mean) ** 2 for x in values) / (n - 1) if n > 1 else float("nan")
    sd = math.sqrt(var) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    return {"n": n, "mean": mean, "sd": sd, "se": se, "lo": mean - 1.96 * se, "hi": mean + 1.96 * se}


def difference(a: Sequence[float], b: Sequence[float]) -> dict[str, float]:
    """``mean(a) - mean(b)`` with a 95% Welch interval, for independent hand samples."""
    sa, sb = summarize(a), summarize(b)
    d = sa["mean"] - sb["mean"]
    se = math.sqrt(sa["se"] ** 2 + sb["se"] ** 2)
    return {"diff": d, "se": se, "lo": d - 1.96 * se, "hi": d + 1.96 * se}


# ------------------------------------------------------------- luck control variates


def luck_terms(rec: dict) -> tuple[float, list[float]]:
    """The hand's result in chips and its zero-mean chance terms ``2 * (eq_t - eq_{t-1})``.

    The deal is uniformly random and independent of both players' choices, and an equity
    against the opponent's actual cards is a martingale as cards arrive. So the first term
    (hole cards, ``2 * (eq_preflop - 1/2)``, zero mean by symmetry between the two seats) and each
    street's increment (zero mean given everything before the card) have expectation exactly
    zero, whatever the players do. They are included only for cards dealt before the hand ended,
    and, when the hand ends in an all-in call, only up to the street of the call, with the
    result replaced by its expectation there (see :func:`adjusted_winnings`).

    Returns ``(chips, [x_preflop, x_flop, x_turn, x_river])``; the opponent's hole cards are
    shown at the end of every hand in the API, so no hand is dropped.
    """
    terms = [0.0, 0.0, 0.0, 0.0]
    bot_hole = rec.get("bot_hole")
    chips = adjusted_winnings(rec)
    if not bot_hole:
        return chips, terms
    allin = all_in_street(rec["action"])
    dealt = min(3, rec["action"].count("/"))
    last = dealt if allin is None else min(dealt, allin)
    eq = preflop_equity_mc(rec["hole"], bot_hole, seed=int(rec.get("n", 0)))
    terms[0] = 2.0 * (eq - 0.5)
    prev = eq
    for street in range(1, last + 1):
        now = equity(rec["hole"], bot_hole, rec["board"][: STREET_BOARD[street]])
        terms[street] = 2.0 * (now - prev)
        prev = now
    return chips, terms


def regression_adjust(chips: Sequence[float], terms: Sequence[Sequence[float]],
                      beta: Sequence[float] | None = None) -> dict:
    """Control-variate estimate of the mean: ``mean(chips - beta . terms)``, ``beta`` by least squares.

    The terms have mean zero, so subtracting them leaves the expectation unchanged and the
    estimate is consistent; the standard error is that of the residuals, which treats ``beta`` as
    known (its estimation error is of order 1/n). Returns mbb/hand statistics and ``beta``.
    """
    y = np.asarray(chips, dtype=float)
    x = np.asarray(terms, dtype=float)
    if beta is None:
        beta_arr = np.linalg.lstsq(x, y, rcond=None)[0] if len(y) > x.shape[1] else np.zeros(x.shape[1])
    else:
        beta_arr = np.asarray(beta, dtype=float)
    resid = y - x @ beta_arr
    out = summarize([mbb(float(v)) for v in resid])
    out["beta"] = [float(b) for b in beta_arr]
    return out
