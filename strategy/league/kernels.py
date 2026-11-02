"""Numba kernels of the league learner (offline only; ``--group train``).

The abstract hand simulator, settlement and point utilities are those of
:mod:`strategy.blueprint.kernels`; this module adds a *mixture* walk in which every seat
plays its own fixed policy table. Tables are dense ``uint8`` rows over the blueprint key
space (``tabs[t, key * NUM_ACTIONS + a]``, as :meth:`PolicyPack.resolved` would give them
times 255); an all-zero row is played uniformly over the legal actions.

One seat per traversal is the *learner*. In training it is a greedy Monte Carlo best
response gated at ``z`` standard errors (the rule of :mod:`strategy.blueprint.response`):
at each of its information sets it accumulates the sampled regret of every action
against what it played and the squares of those increments, plays the action with the
largest cumulative regret once that regret clears ``z`` times the root of its summed
squares, and otherwise plays its base table. The other seats sample their actions from
their tables. In evaluation the learner plays a fixed table and every seat's value is
computed exactly over its own strategy (all of its actions explored) with chance and
the other seats sampled from a stream keyed by ``(seed, deal, seat)``, so two
evaluations that differ only in the learner's table share samples until play diverges.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from strategy.blueprint import abstraction as A
from strategy.blueprint.kernels import (
    MAX_PLAYERS,
    ST_POS,
    ST_STREET,
    _sample,
    apply_action,
    copy_level,
    decision_context,
    infoset_key,
    init_hand,
    order_player,
    resolve,
    settle,
    utility,
    workspace,
)

NUM_ACTIONS = A.NUM_ACTIONS
POINTS_MODE = 1
MAX_STREET = 3

__all__ = ["MAX_PLAYERS", "train_batch", "seat_values", "workspace"]


@njit
def _sigma_u8(tabs, row, base, mask, sig):
    total = 0.0
    legal = 0
    for a in range(NUM_ACTIONS):
        if (mask >> a) & 1:
            legal += 1
            x = float(tabs[row, base + a])
            sig[a] = x
            total += x
        else:
            sig[a] = 0.0
    if total > 0.0:
        for a in range(NUM_ACTIONS):
            sig[a] /= total
    else:
        for a in range(NUM_ACTIONS):
            sig[a] = 1.0 / legal if (mask >> a) & 1 else 0.0


@njit
def _respond_u8(regret, sq, tabs, row, base, mask, sig, z):
    best = -1
    top = 0.0
    for a in range(NUM_ACTIONS):
        if (mask >> a) & 1 and regret[base + a] > top:
            top = regret[base + a]
            best = a
    if best >= 0 and top * top > z * z * sq[base + best]:
        for a in range(NUM_ACTIONS):
            sig[a] = 1.0 if a == best else 0.0
    else:
        _sigma_u8(tabs, row, base, mask, sig)


@njit
def mix_walk(d, trav, n, I, SIG, VAL, start, prizes, bb, post_order, pressure, buckets, ranks, classes, eq, offsets,
             regret, sq, tabs, seats, learn_row, train, z, f, alive, fin, counter):
    counter[0] += 1
    code = resolve(I, d, n, post_order, MAX_STREET, bb)
    if code != 0:
        settle(I, d, n, ranks, post_order, 0, eq, classes, fin)
        return utility(trav, n, start, fin, prizes, POINTS_MODE, bb, f, alive)
    street = I[d, 7, ST_STREET]
    p = order_player(street, I[d, 7, ST_POS], post_order)
    ctx, mask, t_min, t_a, t_b, t_all = decision_context(I, d, n, p, bb, pressure[p])
    key = infoset_key(street, buckets[p, street], ctx, offsets)
    base = key * NUM_ACTIONS
    sig = SIG[d]
    if p != trav:
        _sigma_u8(tabs, seats[p], base, mask, sig)
        a = _sample(sig, mask)
        copy_level(I, d)
        tgt = t_min if a == A.MIN_RAISE else t_a if a == A.RAISE_A else t_b if a == A.RAISE_B else t_all
        apply_action(I, d + 1, n, p, a, tgt)
        return mix_walk(d + 1, trav, n, I, SIG, VAL, start, prizes, bb, post_order, pressure, buckets, ranks, classes,
                        eq, offsets, regret, sq, tabs, seats, learn_row, train, z, f, alive, fin, counter)
    if train:
        _respond_u8(regret, sq, tabs, learn_row, base, mask, sig, z)
    else:
        _sigma_u8(tabs, learn_row, base, mask, sig)
    vals = VAL[d]
    v = 0.0
    for a in range(NUM_ACTIONS):
        vals[a] = 0.0
        if not (mask >> a) & 1:
            continue
        copy_level(I, d)
        tgt = t_min if a == A.MIN_RAISE else t_a if a == A.RAISE_A else t_b if a == A.RAISE_B else t_all
        apply_action(I, d + 1, n, p, a, tgt)
        va = mix_walk(d + 1, trav, n, I, SIG, VAL, start, prizes, bb, post_order, pressure, buckets, ranks, classes,
                      eq, offsets, regret, sq, tabs, seats, learn_row, train, z, f, alive, fin, counter)
        vals[a] = va
        v += sig[a] * va
    if train:
        for a in range(NUM_ACTIONS):
            if (mask >> a) & 1:
                g = vals[a] - v
                regret[base + a] += g
                sq[base + a] += g * g
    return v


@njit(parallel=True)
def train_batch(seed, threads, deal_state, seat_tabs, buckets, ranks, classes, st_n, st_stack, st_post, st_bb,
                st_prizes, st_order, st_press, eq, offsets, regret, sq, tabs, learn_row, z, I, SIG, VAL, f, alive, fin,
                counters):
    """One learner traversal per live seat of every deal; ``seat_tabs[j, p]`` is the table of seat ``p``."""
    B = deal_state.shape[0]
    for t in prange(threads):
        np.random.seed(seed + t)
        cnt = counters[t]
        for j in range(t, B, threads):
            si = deal_state[j]
            n = st_n[si]
            for trav in range(n):
                init_hand(I[t], n, st_stack[si], st_post[si], st_bb[si])
                mix_walk(0, trav, n, I[t], SIG[t], VAL[t], st_stack[si], st_prizes[si], st_bb[si], st_order[si],
                         st_press[si], buckets[j], ranks[j], classes[j], eq, offsets, regret, sq, tabs, seat_tabs[j],
                         learn_row, 1, z, f[t], alive[t], fin[t], cnt)


@njit(parallel=True)
def seat_values(seed, threads, deal_state, seat_tabs, buckets, ranks, classes, st_n, st_stack, st_post, st_bb,
                st_prizes, st_order, st_press, eq, offsets, tabs, learn_row, I, SIG, VAL, f, alive, fin, counters, out):
    """``out[j, p]``: points / first prize of seat ``p`` playing table ``learn_row`` against ``seat_tabs[j]``."""
    B = deal_state.shape[0]
    dummy = np.zeros(1, np.float32)
    for t in prange(threads):
        for j in range(t, B, threads):
            si = deal_state[j]
            n = st_n[si]
            for trav in range(n):
                np.random.seed(seed + j * MAX_PLAYERS + trav)
                init_hand(I[t], n, st_stack[si], st_post[si], st_bb[si])
                out[j, trav] = mix_walk(0, trav, n, I[t], SIG[t], VAL[t], st_stack[si], st_prizes[si], st_bb[si],
                                        st_order[si], st_press[si], buckets[j], ranks[j], classes[j], eq, offsets,
                                        dummy, dummy, tabs, seat_tabs[j], learn_row, 0, 0.0, f[t], alive[t], fin[t],
                                        counters[t])
