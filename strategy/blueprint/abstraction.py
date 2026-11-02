"""Action and information-set abstraction shared by the trainer and the runtime.

Everything here is plain Python over ints so the numba kernels can compile the very
same functions (``numba.njit(context_key)`` etc.): the trainer and the deployed bot
cannot disagree about a key or a bet size. No numpy, no numba imports.

Actions (slots of every information set):

========  ============================================================
0 fold    legal only when facing a bet
1 call    check or call
2 min     minimum full raise (min-bet postflop, 2 BB open preflop)
3 raise a pot-fraction raise ``cur + f * (pot + to_call)``: 0.6 preflop, 0.5 postflop
4 raise b pot-size raise (f = 1)
5 all-in
========  ============================================================

A sized raise is offered only strictly between the minimum raise and all-in and
when it differs from the other size. From the third raise of a street on only
all-in remains (``MAX_SIZED_RAISES``).

Information-set key: street, card bucket (:mod:`strategy.blueprint.cards`) and a
context of public and tournament features, mixed-radix packed:

==========  =====  ==========================================================
field       radix  meaning
==========  =====  ==========================================================
behind      6      players who still act after the hero if it calls (5 = 5+)
active      4      players not folded, 2..5+
raises      4      raises this street, 0..3+
price       6      0 nothing to call, else to_call / (pot + to_call) by PRICE_EDGES
depth       8      preflop unraised: effective start stack in BB (DEPTH_EDGES_BB);
                   preflop raised: bet level / effective start stack
                   (COMMIT_EDGES); postflop: effective stack / pot (SPR_EDGES;
                   7 values used)
acted       2      the hero already acted this street
initiative  2      postflop: the hero made the last raise of the previous street;
                   preflop: the hero has committed over 10% of the effective
                   start stack
pressure    4      ICM risk premium at hand start (PRESSURE_EDGES)
==========  =====  ==========================================================

``effective`` is ``min(hero, largest other player still in the hand)``. Keys are
imperfect recall: the betting history is summarised, not stored. The preflop
bet-to-stack fields were chosen by exact exploitability in heads-up preflop games
(:mod:`strategy.blueprint.exploit`): with the effective start stack alone, histories
of very different commitment shared a key and cost 60-290 mbb/hand at 10-50 BB.
"""

from __future__ import annotations

NUM_ACTIONS = 6
FOLD, CALL, MIN_RAISE, RAISE_A, RAISE_B, ALL_IN = range(NUM_ACTIONS)
ACTION_NAMES = ("fold", "call", "min_raise", "raise_a", "raise_b", "all_in")

PREFLOP_FRACTION_A = (3, 5)
POSTFLOP_FRACTION_A = (1, 2)
FRACTION_B = (1, 1)
MAX_SIZED_RAISES = 3

STREETS = 4
PREFLOP_BUCKETS = 15
POSTFLOP_BUCKETS = 32
RIVER_BUCKETS = 16
BUCKETS = (PREFLOP_BUCKETS, POSTFLOP_BUCKETS, POSTFLOP_BUCKETS, RIVER_BUCKETS)

# Rational edges (num, den): a value strictly above an edge moves to the next bucket.
PRICE_EDGES = ((3, 20), (1, 4), (1, 3), (2, 5))
DEPTH_EDGES_BB = (8, 12, 18, 25, 35, 50, 75)
SPR_EDGES = ((1, 2), (1, 1), (2, 1), (7, 2), (6, 1), (10, 1))
COMMIT_EDGES = ((1, 40), (9, 200), (7, 100), (3, 25), (1, 5), (7, 20), (7, 10))
"""Preflop raised pots: the bet level as a share of the effective start stack (a value at
or above an edge moves up). Chosen among four edge sets by heads-up exploitability at 10,
25, 50 and 100 BB; the others collapsed (230-316 mbb/hand) at some depth where a raise
size landed on an edge."""
PRESSURE_EDGES = (0.05, 0.08, 0.13)
"""Roughly the quartiles of the premium over training-rollout states (heads-up is 0)."""

R_BEHIND, R_ACTIVE, R_RAISES, R_PRICE, R_DEPTH, R_ACTED, R_INIT, R_PRESSURE = 6, 4, 4, 6, 8, 2, 2, 4
CONTEXTS = R_BEHIND * R_ACTIVE * R_RAISES * R_PRICE * R_DEPTH * R_ACTED * R_INIT * R_PRESSURE


def street_offsets(buckets: tuple[int, int, int, int]) -> tuple[int, int, int, int, int]:
    """First key of each street and the total key count for per-street bucket counts."""
    o1 = buckets[0] * CONTEXTS
    o2 = o1 + buckets[1] * CONTEXTS
    o3 = o2 + buckets[2] * CONTEXTS
    return 0, o1, o2, o3, o3 + buckets[3] * CONTEXTS


OFFSETS = street_offsets(BUCKETS)
NUM_KEYS = OFFSETS[4]


def price_bucket(to_call: int, pot: int) -> int:
    if to_call <= 0:
        return 0
    total = pot + to_call
    b = 1
    for num, den in PRICE_EDGES:
        if to_call * den > num * total:
            b += 1
    return b


def depth_bucket(effective: int, bb: int) -> int:
    b = 0
    for e in DEPTH_EDGES_BB:
        if effective > e * bb:
            b += 1
    return b


def spr_bucket(effective: int, pot: int) -> int:
    b = 0
    for num, den in SPR_EDGES:
        if effective * den > num * pot:
            b += 1
    return b


def pressure_bucket(premium: float) -> int:
    b = 0
    for e in PRESSURE_EDGES:
        if premium > e:
            b += 1
    return b


def commit_bucket(cur_bet: int, effective_start: int) -> int:
    b = 0
    for num, den in COMMIT_EDGES:
        if cur_bet * den >= num * effective_start:
            b += 1
    return b


def context_key(street: int, behind: int, active: int, raises: int, to_call: int, pot: int, effective: int,
                effective_start: int, bb: int, acted: int, initiative: int, pressure_b: int, cur_bet: int,
                committed: int) -> int:
    """Mixed-radix context index (``0 <= key < CONTEXTS``) of a decision.

    ``cur_bet`` is the street's bet level and ``committed`` the hero's chips in it.
    """
    if behind > R_BEHIND - 1:
        behind = R_BEHIND - 1
    a = active - 2
    if a < 0:
        a = 0
    if a > R_ACTIVE - 1:
        a = R_ACTIVE - 1
    if raises > R_RAISES - 1:
        raises = R_RAISES - 1
    if street == 0:
        if raises == 0:
            d = depth_bucket(effective_start, bb)
        else:
            d = commit_bucket(cur_bet, effective_start)
        initiative = 1 if committed * 10 > effective_start else 0
    else:
        d = spr_bucket(effective, pot)
    k = behind
    k = k * R_ACTIVE + a
    k = k * R_RAISES + raises
    k = k * R_PRICE + price_bucket(to_call, pot)
    k = k * R_DEPTH + d
    k = k * R_ACTED + (1 if acted else 0)
    k = k * R_INIT + (1 if initiative else 0)
    k = k * R_PRESSURE + pressure_b
    return k


def infoset_key(street: int, bucket: int, context: int, offsets) -> int:
    return offsets[street] + bucket * CONTEXTS + context


def split_key(key: int, offsets=OFFSETS) -> tuple[int, int, int]:
    """``(street, bucket, context)`` of a key."""
    s = 0
    while s < STREETS - 1 and key >= offsets[s + 1]:
        s += 1
    rest = key - offsets[s]
    return s, rest // CONTEXTS, rest % CONTEXTS


def context_fields(context: int) -> dict[str, int]:
    out = {}
    for name, radix in (("pressure", R_PRESSURE), ("initiative", R_INIT), ("acted", R_ACTED), ("depth", R_DEPTH),
                        ("price", R_PRICE), ("raises", R_RAISES), ("active", R_ACTIVE)):
        out[name] = context % radix
        context //= radix
    out["behind"] = context
    out["active"] += 2
    return out


def action_targets(street: int, cur_bet: int, to_call: int, pot: int, min_to: int, max_to: int, raises: int,
                   can_raise: int) -> tuple[int, int, int, int, int]:
    """``(mask, min_to, a_to, b_to, all_in_to)``: legal slots as bits and street-total raise targets.

    ``min_to`` is the minimum full raise total and ``max_to`` the hero's all-in total;
    ``pot`` counts every chip committed this hand, current street included.
    """
    mask = 1 << CALL
    if to_call > 0:
        mask |= 1 << FOLD
    t_min = 0
    t_a = 0
    t_b = 0
    t_all = 0
    if can_raise and max_to > cur_bet:
        mask |= 1 << ALL_IN
        t_all = max_to
        if raises < MAX_SIZED_RAISES and min_to < max_to:
            mask |= 1 << MIN_RAISE
            t_min = min_to
            base = pot + to_call
            if street == 0:
                a = cur_bet + (PREFLOP_FRACTION_A[0] * base) // PREFLOP_FRACTION_A[1]
            else:
                a = cur_bet + (POSTFLOP_FRACTION_A[0] * base) // POSTFLOP_FRACTION_A[1]
            if min_to < a < max_to:
                mask |= 1 << RAISE_A
                t_a = a
            b = cur_bet + (FRACTION_B[0] * base) // FRACTION_B[1]
            if min_to < b < max_to and b != a:
                mask |= 1 << RAISE_B
                t_b = b
    return mask, t_min, t_a, t_b, t_all


def describe() -> dict:
    """Bucket and action counts for provenance and the report."""
    return {
        "actions": list(ACTION_NAMES),
        "preflop_fraction_a": PREFLOP_FRACTION_A, "postflop_fraction_a": POSTFLOP_FRACTION_A,
        "fraction_b": FRACTION_B, "max_sized_raises": MAX_SIZED_RAISES,
        "buckets": list(BUCKETS), "contexts": CONTEXTS, "keys": NUM_KEYS,
        "radix": {"behind": R_BEHIND, "active": R_ACTIVE, "raises": R_RAISES, "price": R_PRICE, "depth": R_DEPTH,
                  "acted": R_ACTED, "initiative": R_INIT, "pressure": R_PRESSURE},
        "price_edges": [list(e) for e in PRICE_EDGES], "depth_edges_bb": list(DEPTH_EDGES_BB),
        "spr_edges": [list(e) for e in SPR_EDGES], "pressure_edges": list(PRESSURE_EDGES),
        "commit_edges": [list(e) for e in COMMIT_EDGES], "preflop_commit_flag": [1, 10],
    }
