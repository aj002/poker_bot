"""Numba kernels of the blueprint trainer (offline only; ``--group train``).

An abstract no-limit hold'em hand is simulated on preallocated integer arrays, one
level per decision depth: ``I[d, channel, player]`` with channels

=====  ========================================================================
0      chips behind
1      chips committed this street
2      chips committed this hand
3      status: 0 can act, 1 folded, 2 all-in
4      acted since the last full raise
5      bet level when the player last checked, called or raised this street (-1: not yet)
6      acted this street
7      scalars by index: 0 street, 1 bet to match, 2 last full raise size, 3 raises this
       street, 4 position of the current actor in the street's order (-1 at street start),
       5 last raiser this street, 6 last raiser of the previous street, 7 the hand's antes
8      dead chips the player put in the main pot (a big-blind ante)
=====  ========================================================================

Antes: a state's ``posts`` row is either the 8 blind posts or 24 values, the blind posts,
then each player's live ante (a classic ante: committed this hand, not this street, so it
is a contribution level), then each player's dead chips (a big-blind ante: in the pot,
paid with the main pot, never a contribution level). Stacks are before both. The start
stacks of a decision context include both, as the arena's ``stack + hand_committed``
does; its pot includes them from the flop on, while preflop it leaves the antes out,
as :class:`~strategy.blueprint.runtime.BlueprintPolicy` does with ``ante_view =
"stripped"``, so an unopened ante pot keys and sizes as an unopened pot.

Betting rights follow the arena adapter: a player may raise when it has not acted this
street or the bet level rose by at least a full raise since it did (short all-ins
accumulate), when it has chips beyond a call, and when another live player could put
in more than the current level. Preflop the level never drops below the big blind.
The arena's optional ``short_bet_reopens_checker`` rule is off in every ruleset and not
modelled.

Players are indices in preflop action order (big blind last); ``post_order`` lists
them in postflop order. Keys and bet sizes come from :mod:`strategy.blueprint.abstraction`
compiled here from the same source. Utilities are chips in big blinds (mode 0) or
tournament points over the first prize (mode 1): eliminated players take their places
(ranked by start-of-hand stack, ties share) and survivors their ICM value.

The trainer is external-sampling MCCFR: per deal, each live player traverses once,
exploring all of its own legal actions and sampling chance (the deal is drawn up front)
and the other players' actions from their current regret-matching strategies, whose
running sums form the average strategy.
"""

from __future__ import annotations

import types

import numpy as np
from numba import njit, prange

from strategy.blueprint import abstraction as A

MAX_PLAYERS = 8
NUM_ACTIONS = A.NUM_ACTIONS
MAX_DEPTH = 512
CH = 9

ST_STREET, ST_BET, ST_LAST_RAISE, ST_RAISES, ST_POS, ST_AGGR, ST_PREV_AGGR, ST_ANTES = range(8)

# int params
P_MODE, P_MAX_STREET, P_EQ_SHOWDOWN, P_OPP_FIXED, P_TRAV_FIXED, P_UPDATE_AVG, P_PRUNE, P_FLOOR_ZERO, P_FREEZE, P_RESPOND = range(10)
NUM_IPARAMS = 10
# float params
F_PRUNE_THRESHOLD, F_REGRET_FLOOR, F_Z = range(3)
NUM_FPARAMS = 3


def _rebind(fn, **jitted):
    g = dict(fn.__globals__)
    g.update(jitted)
    return njit(types.FunctionType(fn.__code__, g, fn.__name__, fn.__defaults__))


_price = njit(A.price_bucket)
_depth = njit(A.depth_bucket)
_spr = njit(A.spr_bucket)
_commit = njit(A.commit_bucket)
context_key = _rebind(A.context_key, price_bucket=_price, depth_bucket=_depth, spr_bucket=_spr,
                      commit_bucket=_commit)
action_targets = njit(A.action_targets)
infoset_key = njit(A.infoset_key)


@njit
def order_player(street, pos, post_order):
    if street == 0:
        return pos
    return post_order[pos]


@njit
def find_actor(I, d, n, post_order, start):
    street = I[d, 7, ST_STREET]
    cur = I[d, 7, ST_BET]
    na = 0
    for q in range(n):
        if I[d, 3, q] == 0:
            na += 1
    for k in range(n):
        pos = (start + k) % n
        q = order_player(street, pos, post_order)
        if I[d, 3, q] != 0:
            continue
        if I[d, 4, q] == 1 and I[d, 1, q] >= cur:
            continue
        if na == 1 and I[d, 1, q] >= cur:
            continue
        return pos
    return -1


@njit
def init_hand(I, n, stacks, posts, bb):
    for c in range(CH):
        for p in range(MAX_PLAYERS):
            I[0, c, p] = 0
    cur = 0
    antes = 0
    for p in range(MAX_PLAYERS):
        if p >= n:
            I[0, 3, p] = 1
            continue
        live = 0
        dead = 0
        if posts.shape[0] >= 3 * MAX_PLAYERS:
            live = posts[MAX_PLAYERS + p]
            dead = posts[2 * MAX_PLAYERS + p]
        behind = stacks[p] - posts[p] - live - dead
        I[0, 0, p] = behind
        I[0, 1, p] = posts[p]
        I[0, 2, p] = posts[p] + live
        I[0, 8, p] = dead
        antes += live + dead
        I[0, 3, p] = 2 if behind <= 0 else 0
        I[0, 5, p] = -1
        if posts[p] > cur:
            cur = posts[p]
    I[0, 7, ST_STREET] = 0
    I[0, 7, ST_BET] = cur if cur > bb else bb
    I[0, 7, ST_LAST_RAISE] = bb
    I[0, 7, ST_RAISES] = 0
    I[0, 7, ST_POS] = -1
    I[0, 7, ST_AGGR] = -1
    I[0, 7, ST_PREV_AGGR] = -1
    I[0, 7, ST_ANTES] = antes


@njit
def copy_level(I, d):
    for c in range(CH):
        for p in range(MAX_PLAYERS):
            I[d + 1, c, p] = I[d, c, p]


@njit
def apply_action(I, d, n, p, a, target):
    I[d, 4, p] = 1
    I[d, 6, p] = 1
    if a == A.FOLD:
        I[d, 3, p] = 1
        return
    cur = I[d, 7, ST_BET]
    if a == A.CALL:
        amt = cur - I[d, 1, p]
        if amt > I[d, 0, p]:
            amt = I[d, 0, p]
        if amt < 0:
            amt = 0
        I[d, 5, p] = cur
    else:
        amt = target - I[d, 1, p]
        inc = target - cur
        if inc >= I[d, 7, ST_LAST_RAISE]:
            I[d, 7, ST_LAST_RAISE] = inc
            for q in range(n):
                if q != p:
                    I[d, 4, q] = 0
        I[d, 5, p] = target
        I[d, 7, ST_BET] = target
        I[d, 7, ST_RAISES] += 1
        I[d, 7, ST_AGGR] = p
    I[d, 0, p] -= amt
    I[d, 1, p] += amt
    I[d, 2, p] += amt
    if I[d, 0, p] == 0:
        I[d, 3, p] = 2


@njit
def resolve(I, d, n, post_order, max_street, bb):
    """0: a decision is pending (actor position set), 1: everyone else folded, 2: showdown."""
    nf = 0
    for q in range(n):
        if I[d, 3, q] != 1:
            nf += 1
    if nf <= 1:
        return 1
    last = I[d, 7, ST_POS]
    pos = find_actor(I, d, n, post_order, 0 if last < 0 else (last + 1) % n)
    if pos >= 0:
        I[d, 7, ST_POS] = pos
        return 0
    street = I[d, 7, ST_STREET]
    if street >= max_street:
        return 2
    na = 0
    for q in range(n):
        if I[d, 3, q] == 0:
            na += 1
    if na <= 1:
        return 2
    I[d, 7, ST_STREET] = street + 1
    for q in range(n):
        I[d, 1, q] = 0
        I[d, 4, q] = 0
        I[d, 5, q] = -1
        I[d, 6, q] = 0
    I[d, 7, ST_BET] = 0
    I[d, 7, ST_LAST_RAISE] = bb
    I[d, 7, ST_RAISES] = 0
    I[d, 7, ST_PREV_AGGR] = I[d, 7, ST_AGGR]
    I[d, 7, ST_AGGR] = -1
    pos = find_actor(I, d, n, post_order, 0)
    I[d, 7, ST_POS] = pos
    return 0 if pos >= 0 else 2


@njit
def decision_context(I, d, n, p, bb, pressure_b):
    """``(context, mask, t_min, t_a, t_b, t_all)`` of player ``p`` deciding at level ``d``."""
    street = I[d, 7, ST_STREET]
    cur = I[d, 7, ST_BET]
    stack = I[d, 0, p]
    to_call = cur - I[d, 1, p]
    if to_call > stack:
        to_call = stack
    if to_call < 0:
        to_call = 0
    pot = 0
    behind = 0
    active = 0
    other_max = 0
    other_max_start = 0
    other_live = 0
    for q in range(n):
        pot += I[d, 2, q] + I[d, 8, q]
        if I[d, 3, q] == 1:
            continue
        active += 1
        if q == p:
            continue
        if I[d, 3, q] == 0:
            if I[d, 0, q] + I[d, 1, q] > cur:
                other_live += 1
            if I[d, 4, q] == 0 or I[d, 1, q] < cur:
                behind += 1
        if I[d, 0, q] > other_max:
            other_max = I[d, 0, q]
        s = I[d, 0, q] + I[d, 2, q] + I[d, 8, q]
        if s > other_max_start:
            other_max_start = s
    if street == 0:
        pot -= I[d, 7, ST_ANTES]
    eff = stack if stack < other_max else other_max
    start = stack + I[d, 2, p] + I[d, 8, p]
    eff_start = start if start < other_max_start else other_max_start
    acted = I[d, 6, p]
    init = 1 if I[d, 7, ST_PREV_AGGR] == p else 0
    ctx = context_key(street, behind, active, I[d, 7, ST_RAISES], to_call, pot, eff, eff_start, bb, acted, init,
                      pressure_b, cur, I[d, 1, p])
    rights = I[d, 5, p] < 0 or cur - I[d, 5, p] >= I[d, 7, ST_LAST_RAISE]
    can_raise = 1 if (rights and stack > to_call and other_live > 0) else 0
    min_to = cur + I[d, 7, ST_LAST_RAISE]
    max_to = I[d, 1, p] + stack
    mask, t_min, t_a, t_b, t_all = action_targets(street, cur, to_call, pot, min_to, max_to, I[d, 7, ST_RAISES],
                                                  can_raise)
    if d >= MAX_DEPTH - 16:
        mask &= (1 << A.FOLD) | (1 << A.CALL)
    return ctx, mask, t_min, t_a, t_b, t_all


@njit
def settle(I, d, n, ranks, post_order, eq_mode, eq, classes, fin):
    """Final stacks (float) after the pot is paid out; ``eq_mode`` pays heads-up showdowns by equity."""
    total = 0
    dead = 0
    nf = 0
    last = -1
    for q in range(MAX_PLAYERS):
        fin[q] = 0.0
    for q in range(n):
        fin[q] = float(I[d, 0, q])
        total += I[d, 2, q]
        dead += I[d, 8, q]
        if I[d, 3, q] != 1:
            nf += 1
            last = q
    if nf == 1:
        fin[last] += float(total + dead)
        return
    if eq_mode:
        m = I[d, 2, 0] if I[d, 2, 0] < I[d, 2, 1] else I[d, 2, 1]
        for q in range(2):
            fin[q] += float(I[d, 2, q] - m)
        e = eq[classes[0], classes[1]]
        fin[0] += e * (2.0 * m + dead)
        fin[1] += (1.0 - e) * (2.0 * m + dead)
        return
    prev = 0
    big = 1 << 60
    while True:
        lev = big
        for q in range(n):
            c = I[d, 2, q]
            if I[d, 3, q] != 1 and c > prev and c < lev:
                lev = c
        if lev == big:
            break
        amount = dead if prev == 0 else 0
        best = -(1 << 62)
        for q in range(n):
            c = I[d, 2, q]
            hi = c if c < lev else lev
            lo = c if c < prev else prev
            amount += hi - lo
            if I[d, 3, q] != 1 and c >= lev and ranks[q] > best:
                best = ranks[q]
        k = 0
        for q in range(n):
            if I[d, 3, q] != 1 and I[d, 2, q] >= lev and ranks[q] == best:
                k += 1
        share = amount // k
        rem = amount - share * k
        for pos in range(n):
            q = post_order[pos]
            if I[d, 3, q] != 1 and I[d, 2, q] >= lev and ranks[q] == best:
                fin[q] += float(share)
                if rem > 0:
                    fin[q] += 1.0
                    rem -= 1
        prev = lev
    left = 0
    top = -1
    for q in range(n):
        if I[d, 2, q] > prev:
            left += I[d, 2, q] - prev
        if I[d, 3, q] != 1 and (top < 0 or I[d, 2, q] > I[d, 2, top]):
            top = q
    fin[top] += float(left)


@njit
def icm_one(stacks, alive, m, prizes, who, f):
    """Malmuth-Harville ICM value of survivor ``who`` (index into ``alive[:m]``)."""
    paid = 0
    for k in range(m):
        if prizes[k] > 0.0:
            paid += 1
    full = (1 << m) - 1
    for mask in range(full + 1):
        f[mask] = 0.0
    f[full] = 1.0
    value = 0.0
    for mask in range(full, 0, -1):
        fm = f[mask]
        if fm == 0.0 or not (mask >> who) & 1:
            continue
        size = 0
        tot = 0.0
        for i in range(m):
            if (mask >> i) & 1:
                size += 1
                tot += stacks[alive[i]]
        place = m - size
        if place >= paid:
            continue
        for i in range(m):
            if (mask >> i) & 1:
                pr = stacks[alive[i]] / tot
                if i == who:
                    value += fm * pr * prizes[place]
                f[mask ^ (1 << i)] += fm * pr
    return value


@njit
def utility(trav, n, start, fin, prizes, mode, bb, f, alive):
    if mode == 0:
        return (fin[trav] - float(start[trav])) / float(bb)
    m = 0
    who = -1
    for q in range(n):
        if fin[q] > 0.5:
            if q == trav:
                who = m
            alive[m] = q
            m += 1
    if who >= 0:
        if m == 1:
            return prizes[0]
        return icm_one(fin, alive, m, prizes, who, f)
    better = 0
    ties = 0
    for q in range(n):
        if fin[q] <= 0.5:
            if start[q] > start[trav]:
                better += 1
            elif start[q] == start[trav]:
                ties += 1
    s = 0.0
    for k in range(ties):
        s += prizes[m + better + k]
    return s / ties


@njit
def _sample(sig, mask):
    u = np.random.random()
    acc = 0.0
    last = 0
    for a in range(NUM_ACTIONS):
        if (mask >> a) & 1:
            acc += sig[a]
            last = a
            if u < acc:
                return a
    return last


@njit
def _sigma_from(table, base, mask, sig, positive):
    total = 0.0
    legal = 0
    for a in range(NUM_ACTIONS):
        if (mask >> a) & 1:
            legal += 1
            x = table[base + a]
            if positive and x < 0.0:
                x = 0.0
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
    return total


@njit
def _respond_sigma(regret, sq, fixed, base, mask, sig, z):
    """Greedy Monte Carlo response: the action with the largest cumulative regret when it
    clears ``z`` standard errors of its sampled regret increments, else the fixed row."""
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
        _sigma_from(fixed, base, mask, sig, False)


@njit
def es_walk(d, trav, n, I, SIG, VAL, start, prizes, bb, post_order, pressure, buckets, ranks, classes, eq,
            offsets, regret, strat, fixed, ip, fp, f, alive, fin, counter):
    counter[0] += 1
    code = resolve(I, d, n, post_order, ip[P_MAX_STREET], bb)
    if code != 0:
        settle(I, d, n, ranks, post_order, ip[P_EQ_SHOWDOWN], eq, classes, fin)
        return utility(trav, n, start, fin, prizes, ip[P_MODE], bb, f, alive)
    street = I[d, 7, ST_STREET]
    p = order_player(street, I[d, 7, ST_POS], post_order)
    ctx, mask, t_min, t_a, t_b, t_all = decision_context(I, d, n, p, bb, pressure[p])
    key = infoset_key(street, buckets[p, street], ctx, offsets)
    base = key * NUM_ACTIONS
    sig = SIG[d]
    use_fixed = ip[P_TRAV_FIXED] if p == trav else ip[P_OPP_FIXED]
    respond = ip[P_RESPOND] > 0 and p == trav and not use_fixed
    if respond and street >= ip[P_RESPOND]:
        respond = False
        use_fixed = 1
    if use_fixed:
        _sigma_from(fixed, base, mask, sig, False)
    elif respond:
        _respond_sigma(regret, strat, fixed, base, mask, sig, fp[F_Z])
    else:
        _sigma_from(regret, base, mask, sig, True)
    if p != trav:
        if ip[P_UPDATE_AVG] and not use_fixed:
            for a in range(NUM_ACTIONS):
                if (mask >> a) & 1:
                    strat[base + a] += sig[a]
        a = _sample(sig, mask)
        copy_level(I, d)
        tgt = t_min if a == A.MIN_RAISE else t_a if a == A.RAISE_A else t_b if a == A.RAISE_B else t_all
        apply_action(I, d + 1, n, p, a, tgt)
        return es_walk(d + 1, trav, n, I, SIG, VAL, start, prizes, bb, post_order, pressure, buckets, ranks, classes,
                       eq, offsets, regret, strat, fixed, ip, fp, f, alive, fin, counter)
    vals = VAL[d]
    prune = ip[P_PRUNE] and not use_fixed
    v = 0.0
    for a in range(NUM_ACTIONS):
        vals[a] = 0.0
        if not (mask >> a) & 1:
            continue
        if prune and sig[a] == 0.0 and regret[base + a] < fp[F_PRUNE_THRESHOLD]:
            continue
        copy_level(I, d)
        tgt = t_min if a == A.MIN_RAISE else t_a if a == A.RAISE_A else t_b if a == A.RAISE_B else t_all
        apply_action(I, d + 1, n, p, a, tgt)
        va = es_walk(d + 1, trav, n, I, SIG, VAL, start, prizes, bb, post_order, pressure, buckets, ranks, classes,
                     eq, offsets, regret, strat, fixed, ip, fp, f, alive, fin, counter)
        vals[a] = va
        v += sig[a] * va
    if not use_fixed and not ip[P_FREEZE]:
        floor = fp[F_REGRET_FLOOR]
        for a in range(NUM_ACTIONS):
            if not (mask >> a) & 1:
                continue
            if prune and sig[a] == 0.0 and regret[base + a] < fp[F_PRUNE_THRESHOLD]:
                continue
            r = regret[base + a] + (vals[a] - v)
            if respond:
                strat[base + a] += (vals[a] - v) ** 2
            if ip[P_FLOOR_ZERO] and r < 0.0:
                r = 0.0
            if r < floor:
                r = floor
            regret[base + a] = r
    return v


def workspace(threads: int = 1) -> dict[str, np.ndarray]:
    return {
        "I": np.zeros((threads, MAX_DEPTH, CH, MAX_PLAYERS), dtype=np.int64),
        "SIG": np.zeros((threads, MAX_DEPTH, NUM_ACTIONS)),
        "VAL": np.zeros((threads, MAX_DEPTH, NUM_ACTIONS)),
        "f": np.zeros((threads, 1 << MAX_PLAYERS)),
        "alive": np.zeros((threads, MAX_PLAYERS), dtype=np.int64),
        "fin": np.zeros((threads, MAX_PLAYERS)),
    }


@njit(parallel=True)
def train_batch(seed, threads, deal_state, buckets, ranks, classes, st_n, st_stack, st_post, st_bb, st_prizes,
                st_order, st_press, eq, offsets, regret, strat, fixed, ip, prune_mask, fp, I, SIG, VAL, f, alive, fin,
                counters, values):
    """One traversal per live player for every deal. ``prune_mask[j]`` enables pruning for deal ``j``.

    ``values[t]`` accumulates the traversers' values and ``counters[t]`` nodes touched
    per thread slot; with ``threads > 1`` slots run in parallel and share the tables
    (lock-free updates).
    """
    B = deal_state.shape[0]
    for t in prange(threads):
        np.random.seed(seed + t)
        ipt = ip.copy()
        cnt = counters[t]
        for j in range(t, B, threads):
            si = deal_state[j]
            n = st_n[si]
            ipt[P_PRUNE] = ip[P_PRUNE] and prune_mask[j]
            for trav in range(n):
                init_hand(I[t], n, st_stack[si], st_post[si], st_bb[si])
                v = es_walk(0, trav, n, I[t], SIG[t], VAL[t], st_stack[si], st_prizes[si], st_bb[si], st_order[si],
                            st_press[si], buckets[j], ranks[j], classes[j], eq, offsets, regret, strat, fixed, ipt, fp,
                            f[t], alive[t], fin[t], cnt)
                values[t, 0] += v
                values[t, 1] += 1.0


@njit
def seat_values(deal_state, buckets, ranks, classes, st_n, st_stack, st_post, st_bb, st_prizes, st_order, st_press,
                eq, offsets, regret, strat, fixed, ip, fp, I, SIG, VAL, f, alive, fin, counters, out):
    """Per deal and seat, the traversal value with the tables as given (no updates when ``ip`` says fixed)."""
    for j in range(deal_state.shape[0]):
        si = deal_state[j]
        n = st_n[si]
        for trav in range(n):
            init_hand(I, n, st_stack[si], st_post[si], st_bb[si])
            out[j, trav] = es_walk(0, trav, n, I, SIG, VAL, st_stack[si], st_prizes[si], st_bb[si], st_order[si],
                                   st_press[si], buckets[j], ranks[j], classes[j], eq, offsets, regret, strat, fixed,
                                   ip, fp, f, alive, fin, counters)


@njit(parallel=True)
def seat_values_par(seed, threads, deal_state, buckets, ranks, classes, st_n, st_stack, st_post, st_bb, st_prizes,
                    st_order, st_press, eq, offsets, regret, strat, fixed, ip, fp, I, SIG, VAL, f, alive, fin,
                    counters, out):
    """:func:`seat_values` over thread slots; run with ``P_FREEZE`` (or every player fixed).

    Each traversal reseeds from ``(seed, deal, seat)``, so two calls that differ only in
    one player's table share opponent and chance samples until their play diverges.
    """
    B = deal_state.shape[0]
    for t in prange(threads):
        for j in range(t, B, threads):
            si = deal_state[j]
            n = st_n[si]
            for trav in range(n):
                np.random.seed(seed + j * MAX_PLAYERS + trav)
                init_hand(I[t], n, st_stack[si], st_post[si], st_bb[si])
                out[j, trav] = es_walk(0, trav, n, I[t], SIG[t], VAL[t], st_stack[si], st_prizes[si], st_bb[si],
                                       st_order[si], st_press[si], buckets[j], ranks[j], classes[j], eq, offsets,
                                       regret, strat, fixed, ip, fp, f[t], alive[t], fin[t], counters[t])


@njit
def deal_classes(seed, count, n, deck, pair_class, class_bucket, out_bucket, out_class):
    """Hole hands only (reduced preflop games): class index and preflop bucket per player."""
    np.random.seed(seed)
    m = deck.shape[0]
    used = np.zeros(64, dtype=np.int64)
    for j in range(count):
        for c in range(64):
            used[c] = 0
        for p in range(n):
            cards = np.zeros(2, dtype=np.int64)
            for k in range(2):
                while True:
                    c = deck[np.random.randint(m)]
                    if used[c] == 0:
                        used[c] = 1
                        cards[k] = c
                        break
            cls = pair_class[cards[0], cards[1]]
            out_class[j, p] = cls
            out_bucket[j, p, 0] = class_bucket[cls]


@njit
def enumerate_tree(d, n, I, start, bb, post_order, pressure, max_street, offsets, nd_kind, nd_player, nd_ctx,
                   nd_mask, nd_child, nd_contrib, nd_folded, counter):
    """Every betting sequence from level ``d`` (no chance): returns the node index.

    ``nd_kind``: 0 decision, 1 everyone else folded, 2 showdown. ``nd_ctx`` holds the
    street offset plus context, so the key for bucket ``b`` is ``nd_ctx + b * CONTEXTS``.
    """
    idx = counter[0]
    counter[0] += 1
    code = resolve(I, d, n, post_order, max_street, bb)
    for q in range(n):
        nd_contrib[idx, q] = I[d, 2, q]
        nd_folded[idx, q] = 1 if I[d, 3, q] == 1 else 0
    if code != 0:
        nd_kind[idx] = code
        return idx
    street = I[d, 7, ST_STREET]
    p = order_player(street, I[d, 7, ST_POS], post_order)
    ctx, mask, t_min, t_a, t_b, t_all = decision_context(I, d, n, p, bb, pressure[p])
    nd_kind[idx] = 0
    nd_player[idx] = p
    nd_ctx[idx] = offsets[street] + ctx
    nd_mask[idx] = mask
    for a in range(NUM_ACTIONS):
        nd_child[idx, a] = -1
        if not (mask >> a) & 1:
            continue
        copy_level(I, d)
        tgt = t_min if a == A.MIN_RAISE else t_a if a == A.RAISE_A else t_b if a == A.RAISE_B else t_all
        apply_action(I, d + 1, n, p, a, tgt)
        nd_child[idx, a] = enumerate_tree(d + 1, n, I, start, bb, post_order, pressure, max_street, offsets, nd_kind,
                                          nd_player, nd_ctx, nd_mask, nd_child, nd_contrib, nd_folded, counter)
    return idx


TRACE_FIELDS = ("street", "player", "context", "mask", "action", "target", "t_min", "t_a", "t_b", "t_all")


@njit
def trace_hand(seed, n, I, start, posts, bb, post_order, pressure, max_street, weights, rec):
    """Play one hand choosing each legal action with probability proportional to ``weights``.

    Rows of ``rec`` follow :data:`TRACE_FIELDS`; returns ``(decisions, end code)``.
    """
    np.random.seed(seed)
    init_hand(I, n, start, posts, bb)
    d = 0
    k = 0
    while True:
        code = resolve(I, d, n, post_order, max_street, bb)
        if code != 0:
            return k, code
        street = I[d, 7, ST_STREET]
        p = order_player(street, I[d, 7, ST_POS], post_order)
        ctx, mask, t_min, t_a, t_b, t_all = decision_context(I, d, n, p, bb, pressure[p])
        total = 0.0
        for a in range(NUM_ACTIONS):
            if (mask >> a) & 1:
                total += weights[a]
        u = np.random.random() * total
        a = -1
        for b in range(NUM_ACTIONS):
            if (mask >> b) & 1:
                a = b
                u -= weights[b]
                if u < 0.0:
                    break
        tgt = t_min if a == A.MIN_RAISE else t_a if a == A.RAISE_A else t_b if a == A.RAISE_B else t_all
        rec[k, 0] = street
        rec[k, 1] = p
        rec[k, 2] = ctx
        rec[k, 3] = mask
        rec[k, 4] = a
        rec[k, 5] = tgt if a >= A.MIN_RAISE else 0
        rec[k, 6] = t_min
        rec[k, 7] = t_a
        rec[k, 8] = t_b
        rec[k, 9] = t_all
        k += 1
        copy_level(I, d)
        apply_action(I, d + 1, n, p, a, tgt)
        d += 1
