"""Decision-time ICM: :func:`strategy.endgame.keys.icm_player` and
:func:`~strategy.endgame.keys.pressure`, bit for bit, without a reference cycle per call.

Decision-time safe: pure Python, no numpy. ``keys.py`` is an input of the committed
endgame tables, so this copy is separate; the tests check the two agree exactly.
"""

from __future__ import annotations

from collections.abc import Sequence


def icm_player(stacks: Sequence[float], hero: int, prizes: Sequence[float]) -> float:
    """Exact Malmuth-Harville ICM value of one player (zero stacks are out of the race)."""
    live = [i for i, s in enumerate(stacks) if s > 0]
    if stacks[hero] <= 0:
        return 0.0
    k = len(live)
    prize = list(prizes[:k]) + [0.0] * max(0, k - len(prizes))
    paid = max((i + 1 for i, p in enumerate(prize) if p), default=0)
    total = float(sum(stacks[i] for i in live))
    sh = float(stacks[hero])
    others = [float(stacks[i]) for i in live if i != hero]
    return _icm_rec(total, sh, prize, paid, others, 0, 0.0, 0, 1.0) if paid else 0.0


def _icm_rec(total: float, sh: float, prize: list[float], paid: int, others: list[float], place: int,
             removed: float, used: int, prob: float) -> float:
    """The hero's value from ``place`` on: ``prob`` that the places before went to the
    ``used`` others (``removed`` chips), times the race for the rest.

    The last two paid places are evaluated in loops with the same floating-point
    operations, in the same order, as the recursive calls for them would perform."""
    rest = total - removed
    value = prob * sh / rest * prize[place]
    nxt = place + 1
    if nxt >= paid:
        return value
    if nxt + 1 == paid:
        pz = prize[nxt]
        for j, sj in enumerate(others):
            if not used >> j & 1:
                value += prob * sj / rest * sh / (total - (removed + sj)) * pz
    elif nxt + 2 == paid:
        p1, p2 = prize[nxt], prize[nxt + 1]
        for j, sj in enumerate(others):
            if not used >> j & 1:
                r1 = removed + sj
                q = prob * sj / rest
                rest1 = total - r1
                v = q * sh / rest1 * p1
                u1 = used | 1 << j
                for k, sk in enumerate(others):
                    if not u1 >> k & 1:
                        v += q * sk / rest1 * sh / (total - (r1 + sk)) * p2
                value += v
    else:
        for j, sj in enumerate(others):
            if not used >> j & 1:
                value += _icm_rec(total, sh, prize, paid, others, nxt, removed + sj, used | 1 << j, prob * sj / rest)
    return value


def pressure(stacks: Sequence[int], hero: int, villain: int, prizes: Sequence[float]) -> float:
    """ICM risk premium of an all-in for ``min(hero, villain)`` chips between two players."""
    e = min(stacks[hero], stacks[villain])
    if e <= 0:
        return 0.0
    win = list(stacks)
    lose = list(stacks)
    win[hero] += e
    win[villain] -= e
    lose[hero] -= e
    lose[villain] += e
    u_fold = icm_player(stacks, hero, prizes)
    u_win = icm_player(win, hero, prizes)
    if lose[hero] > 0:
        u_lose = icm_player(lose, hero, prizes)
    else:
        live = sum(1 for s in stacks if s > 0)
        u_lose = float(prizes[live - 1]) if live <= len(prizes) else 0.0
    span = u_win - u_lose
    if span <= 1e-12:
        return 0.0
    return (u_fold - u_lose) / span - 0.5
