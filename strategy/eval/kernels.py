"""Compiled kernels for Slumbot statistics and luck terms: a seven-card hand evaluator, preflop
equity, and batched board scoring (:mod:`strategy.eval.luck`).

numba is optional here (it lives in the ``train`` dependency group): without it the same code
runs as plain Python, and :mod:`strategy.eval.slumbot_stats` falls back to a smaller Monte Carlo.
"""

from __future__ import annotations

import numpy as np

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


@njit(cache=True)
def score_boards(holes, board, extra):  # type: ignore[no-untyped-def]
    """``(n, k)`` strengths of the ``k`` two-card ``holes`` on ``board`` completed by each row of ``extra``."""
    n = extra.shape[0]
    m = extra.shape[1]
    k = holes.shape[0]
    b = board.shape[0]
    out = np.zeros((n, k), dtype=np.int64)
    cards = np.zeros(7, dtype=np.int64)
    for i in range(b):
        cards[2 + i] = board[i]
    for r in range(n):
        for j in range(m):
            cards[2 + b + j] = extra[r, j]
        for p in range(k):
            cards[0] = holes[p, 0]
            cards[1] = holes[p, 1]
            out[r, p] = score7(cards)
    return out


@njit(cache=True)
def _preflop_mc(hero0, hero1, vil0, vil1, rest, samples, seed):  # type: ignore[no-untyped-def]
    np.random.seed(seed)
    n = len(rest)
    pool = rest.copy()
    wins = 0.0
    cards = np.zeros(7, dtype=np.int64)
    cards2 = np.zeros(7, dtype=np.int64)
    cards[0] = hero0
    cards[1] = hero1
    cards2[0] = vil0
    cards2[1] = vil1
    for _ in range(samples):
        for i in range(5):
            j = i + np.random.randint(0, n - i)
            tmp = pool[i]
            pool[i] = pool[j]
            pool[j] = tmp
            cards[2 + i] = pool[i]
            cards2[2 + i] = pool[i]
        h = score7(cards)
        v = score7(cards2)
        if h > v:
            wins += 1.0
        elif h == v:
            wins += 0.5
    return wins / samples
