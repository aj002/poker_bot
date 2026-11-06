"""Ante push/fold subgames, their solve and harvest (offline).

:class:`AnteSubgame` is :class:`~strategy.endgame.subgame.Subgame` with each player's ante in
the pot from the start. A player whose blind and ante take its whole stack is all-in from
posting. Terminal chips settle as the arena settles them: classic antes are contribution
levels, and a big-blind ante is dead money added to the main pot (:func:`settle_dead`).
Points then follow exactly as in the ante-free subgame: ICM continuation, eliminations, and
cap ICM on the hand-cap hand.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from strategy.antes.pushfold import AnteSnapshot, ante_keys_for
from strategy.endgame.harvest import Record, Solved, decision_nodes, default_iterations
from strategy.endgame.points import settle
from strategy.endgame.solver import PushFoldSolver
from strategy.endgame.subgame import Subgame, SubgameTree, Terminal, icm_value, ordering_codes


def settle_dead(levels: Sequence[int], folded: Sequence[bool], strength: Sequence[int], odd_order: Sequence[int],
                dead: int) -> list[int]:
    """:func:`strategy.endgame.points.settle` over contribution ``levels``, with ``dead`` chips in
    the main pot (the lowest level), which is split with them as one pot."""
    if not dead:
        return settle(levels, folded, strength, odd_order)
    n = len(levels)
    awards = [0] * n
    rank = {p: i for i, p in enumerate(odd_order)}
    prev = 0
    last: list[int] | None = None
    carry = 0
    for i, level in enumerate(sorted({c for c in levels if c > 0})):
        amount = sum(min(max(c - prev, 0), level - prev) for c in levels) + (dead if i == 0 else 0)
        eligible = [p for p in range(n) if not folded[p] and levels[p] >= level]
        prev = level
        if not eligible:
            carry += amount
            continue
        best = max(strength[p] for p in eligible)
        last = sorted((p for p in eligible if strength[p] == best), key=lambda p: rank[p])
        share, rem = divmod(amount, len(last))
        for k, w in enumerate(last):
            awards[w] += share + (1 if k < rem else 0)
    if carry:
        if last is None:
            live = [p for p in range(n) if not folded[p]]
            best = max(strength[p] for p in live)
            last = sorted((p for p in live if strength[p] == best), key=lambda p: rank[p])
        share, rem = divmod(carry, len(last))
        for k, w in enumerate(last):
            awards[w] += share + (1 if k < rem else 0)
    return awards


@dataclass(frozen=True)
class AnteSubgame(Subgame):
    antes: tuple[int, ...] = ()
    dead_antes: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.antes) != self.n:
            raise ValueError("antes must list one ante per player")
        if any(p + a > s for p, a, s in zip(self.posts, self.antes, self.stacks)):
            raise ValueError("posts plus antes exceed a stack")

    @property
    def auto_all_in(self) -> frozenset[int]:
        return frozenset(i for i in range(self.n) if self.posts[i] + self.antes[i] == self.stacks[i])


class AnteSubgameTree(SubgameTree):
    def chip_outcomes(self, idx: int):
        """``(showdown, folded, contrib, [(code, better, end chips), ...])`` of terminal ``idx``."""
        g = self.game
        n = g.n
        nd = self.nodes[idx]
        dead = sum(g.antes) if g.dead_antes else 0
        live = set(nd.in_pot) | set(self._auto)
        if nd.shover is None:
            live.add(n - 1)
        showdown = tuple(sorted(live))
        folded = tuple(p for p in range(n) if p not in live)
        contrib = [g.stacks[p] if (p in nd.in_pot or p in self._auto) else g.posts[p] + g.antes[p]
                   for p in range(n)]
        bets = sorted(((c - g.antes[p], p) for p, c in enumerate(contrib)), reverse=True)
        if bets[0][0] > bets[1][0]:
            contrib[bets[0][1]] -= bets[0][0] - bets[1][0]
        contrib = tuple(contrib)
        levels = tuple(c - g.antes[p] for p, c in enumerate(contrib)) if dead else contrib
        fold_flags = [p not in live for p in range(n)]
        outs = []
        for code, better in ordering_codes(len(showdown)) if len(showdown) > 1 else [(0, (0,))]:
            strength = [0] * n
            for p, b in zip(showdown, better):
                strength[p] = -b
            awards = settle_dead(levels, fold_flags, strength, g.order_for_odd_chips, dead)
            end = [g.stacks[p] - contrib[p] + awards[p] for p in range(n)]
            if sum(end) != sum(g.stacks):
                raise AssertionError("settlement does not conserve chips")
            outs.append((code, better, end))
        return showdown, folded, contrib, outs

    def _utilities(self) -> None:
        pending = []
        ends_all: list[list[int]] = []
        for nd in self.nodes:
            if nd.player != -1:
                continue
            showdown, folded, contrib, outs = self.chip_outcomes(nd.idx)
            ends_all.extend(end for _, _, end in outs)
            pending.append((nd.idx, showdown, folded, contrib, tuple(c for c, _, _ in outs)))
        values = self._values(ends_all)
        at = 0
        for idx, showdown, folded, contrib, codes in pending:
            self.terminals[idx] = Terminal(idx, showdown, folded, contrib, codes, values[at:at + len(codes)])
            at += len(codes)


def ante_subgame_for(snap: AnteSnapshot, *, utility: str = "points", odd_order: Sequence[int] | None = None,
                     value_fn=None, max_in_pot: int = 3) -> AnteSubgame:
    return AnteSubgame(stacks=snap.stacks, posts=snap.posts, payouts=snap.payouts, bb=snap.bb,
                       odd_order=tuple(odd_order) if odd_order is not None else None,
                       final_hand=snap.hand_number >= snap.hand_cap, max_in_pot=max_in_pot, utility=utility,
                       value_fn=value_fn or icm_value, antes=snap.antes, dead_antes=snap.posting == "big_blind")


def solve_ante_snapshot(snap: AnteSnapshot, model, *, utility: str = "points", iterations: int | None = None,
                        odd_order: Sequence[int] | None = None, value_fn=None, with_gains: bool = True) -> Solved:
    tree = AnteSubgameTree(ante_subgame_for(snap, utility=utility, odd_order=odd_order, value_fn=value_fn))
    solver = PushFoldSolver(tree, model).run(iterations or default_iterations(snap.n))
    sigma = solver.average()
    gains = solver.deviation_gains(sigma) if with_gains else np.full(snap.n, np.nan)
    return Solved(snap, tree, solver, sigma, gains)


def ante_harvest(solved: Solved, *, max_depth_bb: float) -> list[Record]:
    """:func:`strategy.endgame.harvest.harvest` keyed by :func:`ante_keys_for`."""
    snap = solved.snap
    r = solved.solver.reach(solved.sigma)
    prior = solved.solver.prior
    out = []
    for kind, node, hero, shover, caller in decision_nodes(solved.tree):
        if kind == "open":
            risk = min(snap.stacks[hero], max(snap.stacks[p] for p in range(hero + 1, snap.n)))
        elif kind == "call":
            risk = min(snap.stacks[hero], snap.stacks[shover])
        else:
            risk = min(snap.stacks[hero], max(snap.stacks[shover], snap.stacks[caller]))
        if risk / snap.bb > max_depth_bb:
            continue
        rbar = r[node] @ prior
        weight = float(np.prod(np.delete(rbar, hero)))
        out.append(Record(kind, ante_keys_for(kind, snap, hero, shover, caller), hero, weight,
                          solved.sigma[node][1].copy()))
    return out
