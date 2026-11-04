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
from strategy.eval.slumbot import BIG_BLIND, STACK, STREET_BOARD, tokenize

try:  # optional: exact preflop all-in equity needs a compiled evaluator
    from numba import njit
except ImportError:  # pragma: no cover - exercised when numba is absent
    def njit(*a, **k):  # type: ignore[no-redef]
        if len(a) == 1 and callable(a[0]):
            return a[0]
        return lambda f: f

    HAVE_NUMBA = False
else:
    HAVE_NUMBA = True

PREFLOP_SAMPLES = 20_000  # Monte Carlo boards when numba is missing

_CARD = {r + s: 4 * i + j for i, r in enumerate(RANKS) for j, s in enumerate(SUITS)}


@njit(cache=True)
def straight_high(m):  # type: ignore[no-untyped-def]
    for hi in range(12, 3, -1):
        need = 0x1F << (hi - 4)
        if m & need == need:
            return hi
    if m & 0x100F == 0x100F:
        return 3
    return -1


@njit(cache=True)
def score7(cards):  # type: ignore[no-untyped-def]
    """Strength of the best five of seven cards (ints 0..51: ``4 * rank + suit``); larger wins."""
    rc = np.zeros(13, dtype=np.int64)
    sc = np.zeros(4, dtype=np.int64)
    mask = 0
    smask = np.zeros(4, dtype=np.int64)
    for c in cards:
        r = c // 4
        s = c % 4
        rc[r] += 1
        sc[s] += 1
        mask |= 1 << r
        smask[s] |= 1 << r

    flush = -1
    for s in range(4):
        if sc[s] >= 5:
            flush = s
    if flush >= 0:
        sh = straight_high(smask[flush])
        if sh >= 0:
            return 8 * 13**5 + sh
    quad = -1
    trips = np.full(2, -1, dtype=np.int64)
    pairs = np.full(3, -1, dtype=np.int64)
    nt = 0
    npair = 0
    for r in range(12, -1, -1):
        n = rc[r]
        if n == 4:
            quad = r
        elif n == 3:
            if nt < 2:
                trips[nt] = r
            nt += 1
        elif n == 2:
            if npair < 3:
                pairs[npair] = r
            npair += 1
    if quad >= 0:
        kick = -1
        for r in range(12, -1, -1):
            if rc[r] > 0 and r != quad:
                kick = r
                break
        return 7 * 13**5 + quad * 13 + kick
    if nt >= 1 and (nt >= 2 or npair >= 1):
        pair = trips[1] if nt >= 2 else pairs[0]
        if nt >= 2 and npair >= 1 and pairs[0] > pair:
            pair = pairs[0]
        return 6 * 13**5 + trips[0] * 13 + pair
    if flush >= 0:
        v = 0
        k = 0
        for r in range(12, -1, -1):
            if smask[flush] >> r & 1 and k < 5:
                v = v * 13 + r
                k += 1
        return 5 * 13**5 + v
    sh = straight_high(mask)
    if sh >= 0:
        return 4 * 13**5 + sh
    if nt >= 1:
        v = trips[0]
        k = 0
        for r in range(12, -1, -1):
            if rc[r] == 1 and k < 2:
                v = v * 13 + r
                k += 1
        return 3 * 13**5 + v
    if npair >= 2:
        v = pairs[0] * 13 + pairs[1]
        for r in range(12, -1, -1):
            if rc[r] > 0 and r != pairs[0] and r != pairs[1]:
                v = v * 13 + r
                break
        return 2 * 13**5 + v
    if npair == 1:
        v = pairs[0]
        k = 0
        for r in range(12, -1, -1):
            if rc[r] == 1 and k < 3:
                v = v * 13 + r
                k += 1
        return 1 * 13**5 + v
    v = 0
    k = 0
    for r in range(12, -1, -1):
        if rc[r] == 1 and k < 5:
            v = v * 13 + r
            k += 1
    return v


def _eval_boards(hero: tuple[int, int], villain: tuple[int, int], boards: Iterable[Sequence[int]]) -> tuple[float, int]:
    wins = 0.0
    n = 0
    for b in boards:
        h = score7([*hero, *b])
        v = score7([*villain, *b])
        wins += 1.0 if h > v else 0.5 if h == v else 0.0
        n += 1
    return wins, n


@njit(cache=True)
def _preflop_exact(hero0, hero1, vil0, vil1, rest):  # type: ignore[no-untyped-def]
    n = len(rest)
    wins = 0.0
    total = 0
    cards = np.zeros(7, dtype=np.int64)
    cards2 = np.zeros(7, dtype=np.int64)
    cards[0] = hero0
    cards[1] = hero1
    cards2[0] = vil0
    cards2[1] = vil1
    for a in range(n - 4):
        for b in range(a + 1, n - 3):
            for c in range(b + 1, n - 2):
                for d in range(c + 1, n - 1):
                    for e in range(d + 1, n):
                        cards[2] = rest[a]
                        cards[3] = rest[b]
                        cards[4] = rest[c]
                        cards[5] = rest[d]
                        cards[6] = rest[e]
                        for i in range(2, 7):
                            cards2[i] = cards[i]
                        h = score7(cards)
                        v = score7(cards2)
                        if h > v:
                            wins += 1.0
                        elif h == v:
                            wins += 0.5
                        total += 1
    return wins, total


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
