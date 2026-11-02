"""Variant card adapters for the population bots: No-limit Hold'em, short deck and PLO4.

The style rules in :mod:`arena.agents.styles.engine` speak Hold'em: a 169-class starting
hand and a post-flop strength bucket (``monster``, ``strong``, ``medium``, ``weak``) with
draw outs. This module gives every variant the same vocabulary:

* :func:`preflop_top` places any variant's starting hand at a *top fraction* of that
  variant's own starting hands (0 is the best hand), from generated tables in
  ``data/preflop.json`` (see :mod:`strategy.population.gen_tables`);
* :func:`equivalent_class` maps that fraction to the No-limit Hold'em class at the same
  fraction, so position charts and range knobs apply to short deck and PLO unchanged;
* :func:`strength_for` returns the bucket function for a ruleset: Hold'em uses the style
  evaluator; short deck adds the A-6-7-8-9 straight and its flush-over-full-house order;
  PLO uses exactly two hole cards and three board cards through a prime-product lookup
  (about 15 microseconds at the river instead of 300 with the generic evaluator).

These are cheap opponent-model approximations, not equity calculators.
"""

from __future__ import annotations

import bisect
import itertools
import json
from collections.abc import Callable, Sequence
from functools import cache
from pathlib import Path

from arena.agents.styles.handeval import (
    FLUSH,
    FULL_HOUSE,
    HIGH_CARD,
    PAIR,
    STRAIGHT,
    STRAIGHT_FLUSH,
    TRIPS,
    TWO_PAIR,
    RV,
    evaluate,
    strength,
)

DATA = Path(__file__).resolve().parent / "data" / "preflop.json"
Strength = tuple[str, int, tuple[int, ...]]
StrengthFn = Callable[[Sequence[str], Sequence[str]], Strength]
VARIANTS = ("practice_nlhe", "uncertainty_shortdeck", "uncertainty_plo4")


# ------------------------------------------------------------------ preflop tables


@cache
def _tables() -> dict:
    return json.loads(DATA.read_text())


def _bands(rows: list[list]) -> tuple[dict[str, float], list[str], list[float]]:
    """Per class: the top fraction at the middle of its combos; plus the cumulative order."""
    total = sum(r[2] for r in rows)
    mid: dict[str, float] = {}
    order: list[str] = []
    cum: list[float] = []
    acc = 0
    for cls, _eq, combos in rows:
        mid[cls] = (acc + combos / 2) / total
        acc += combos
        order.append(cls)
        cum.append(acc / total)
    return mid, order, cum


@cache
def _nlhe() -> tuple[dict[str, float], list[str], list[float], dict[str, float]]:
    rows = _tables()["practice_nlhe"]
    mid, order, cum = _bands(rows)
    return mid, order, cum, {r[0]: r[1] for r in rows}


@cache
def _shortdeck() -> dict[str, float]:
    return _bands(_tables()["uncertainty_shortdeck"])[0]


@cache
def _plo_quantiles() -> list[float]:
    return _tables()["uncertainty_plo4"]["score_quantiles"]


def nlhe_class(cards: Sequence[str]) -> str:
    a, b = cards[0], cards[1]
    hi, lo = (a, b) if RV[a[0]] >= RV[b[0]] else (b, a)
    if hi[0] == lo[0]:
        return hi[0] + lo[0]
    return hi[0] + lo[0] + ("s" if hi[1] == lo[1] else "o")


def nlhe_equity(cls: str) -> float:
    """Heads-up all-in equity of a Hold'em class against a random hand (generated table)."""
    return _nlhe()[3][cls]


def plo_score(cards: Sequence[str]) -> float:
    """A PLO4 starting-hand score: the best three two-card Hold'em equities, plus structure.

    Double-suited and connected hands score higher; three of a rank and three of a suit
    score lower, because those cards work against each other.
    """
    eq = sorted((nlhe_equity(nlhe_class(p)) for p in itertools.combinations(cards, 2)), reverse=True)
    score = eq[0] + 0.5 * eq[1] + 0.25 * eq[2]
    ranks = [RV[c[0]] for c in cards]
    suits = [c[1] for c in cards]
    rank_counts = sorted((ranks.count(r) for r in set(ranks)), reverse=True)
    suit_counts = sorted((suits.count(s) for s in set(suits)), reverse=True)
    if suit_counts[:2] == [2, 2]:
        score += 0.06
    elif suit_counts[0] == 2:
        score += 0.025
    elif suit_counts[0] >= 3:
        score -= 0.02 * (suit_counts[0] - 2)
    if rank_counts[0] >= 3:
        score -= 0.08
    distinct = sorted(set(ranks))
    if len(distinct) == 4 and distinct[-1] - distinct[0] <= 4:
        score += 0.04
    elif len(distinct) >= 3 and distinct[-1] - distinct[0] <= 4:
        score += 0.015
    return score


@cache
def card_ruleset(ruleset: str) -> str:
    """The practice adapter that reads a ruleset's cards.

    A build-day reveal variant is read by the nearest practice adapter (exactly-two-hole
    showdowns by the PLO one, the 36-card deck by the short-deck one, anything else by
    Hold'em), so on a revealed variant the whole population plays as variant-noisy
    transfer bots: they misjudge whatever the reveal changed (a fifth hole card, a
    different category order).
    """
    if ruleset in VARIANTS:
        return ruleset
    from strategy.rules.reveal import get_variant

    v = get_variant(ruleset)
    if v.hand_selection == "two_hole_three_board":
        return "uncertainty_plo4"
    if v.ranks == "6789TJQKA":
        return "uncertainty_shortdeck"
    return "practice_nlhe"


def preflop_top(cards: Sequence[str], ruleset: str) -> float:
    """Top fraction of the variant's starting hands at this hand: 0 is the best, 1 the worst."""
    ruleset = card_ruleset(ruleset)
    if ruleset == "uncertainty_plo4":
        q = _plo_quantiles()
        stronger = len(q) - bisect.bisect_right(q, plo_score(cards))
        return min(1.0, max(0.0, (stronger + 0.5) / len(q)))
    cls = nlhe_class(cards)
    if ruleset == "uncertainty_shortdeck":
        return _shortdeck()[cls]
    return _nlhe()[0][cls]


def equivalent_class(top: float) -> str:
    """The No-limit Hold'em class whose combos cover the top fraction ``top``."""
    _mid, order, cum, _eq = _nlhe()
    return order[min(len(order) - 1, bisect.bisect_left(cum, top))]


@cache
def top_classes(q: float) -> frozenset[str]:
    """Hold'em classes whose combo band starts inside the top fraction ``q``."""
    if q <= 0:
        return frozenset()
    mid, order, cum, _eq = _nlhe()
    out = []
    prev = 0.0
    for cls, c in zip(order, cum):
        if prev < q:
            out.append(cls)
        prev = c
    return frozenset(out)


def top(q: float) -> frozenset[str]:
    """:func:`top_classes` rounded to a thousandth, so equal knobs share one cached set."""
    return top_classes(round(min(1.0, max(0.0, q)), 3))


# ------------------------------------------------------------------ short deck

def _sd_straight_high(ranks: set[int]) -> int | None:
    for high in range(12, 7, -1):
        if all(r in ranks for r in range(high - 4, high + 1)):
            return high
    if {12, 4, 5, 6, 7} <= ranks:  # A-6-7-8-9: the ace plays low
        return 7
    return None


def evaluate_shortdeck(cards: Sequence[str]) -> tuple[int, ...]:
    """Short-deck ranking: trips < straight < full house < flush < quads.

    The categories reuse the Hold'em constants but compare with the flush above the full
    house: the returned tuple's first element is the short-deck category order.
    """
    ranks = [RV[c[0]] for c in cards]
    counts = [0] * 13
    for r in ranks:
        counts[r] += 1
    by_suit: dict[str, list[int]] = {}
    for c in cards:
        by_suit.setdefault(c[1], []).append(RV[c[0]])
    flush = next((sorted(rs, reverse=True) for rs in by_suit.values() if len(rs) >= 5), None)
    if flush is not None:
        sf = _sd_straight_high(set(flush))
        if sf is not None:
            return (8, sf)
    quads = [r for r in range(12, -1, -1) if counts[r] == 4]
    if quads:
        q = quads[0]
        return (7, q, max(r for r in ranks if r != q))
    if flush is not None:
        return (6, *flush[:5])
    trips = [r for r in range(12, -1, -1) if counts[r] == 3]
    pairs = [r for r in range(12, -1, -1) if counts[r] == 2]
    if trips and (len(trips) > 1 or pairs):
        return (5, trips[0], max([*trips[1:], *pairs]))
    st = _sd_straight_high(set(ranks))
    if st is not None:
        return (4, st)
    singles = sorted((r for r in range(13) if counts[r] == 1), reverse=True)
    if trips:
        return (3, trips[0], *singles[:2])
    if len(pairs) >= 2:
        return (2, pairs[0], pairs[1], max([*pairs[2:], *singles], default=-1))
    if pairs:
        return (1, pairs[0], *singles[:3])
    return (0, *singles[:5])


def strength_shortdeck(hole: Sequence[str], board: Sequence[str]) -> Strength:
    """Hold'em buckets, with straights the ace-low short-deck way counted as monsters."""
    bucket, outs, value = strength(hole, board)
    if bucket != "monster":
        mine = {RV[c[0]] for c in (*hole, *board)}
        if _sd_straight_high(mine) is not None and _sd_straight_high({RV[c[0]] for c in board}) is None:
            return "monster", outs, value
    return bucket, outs, value


# ------------------------------------------------------------------ PLO4

_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)


@cache
def _five_tables() -> tuple[dict[int, tuple[int, ...]], dict[int, tuple[int, ...]]]:
    """Prime-product key of five ranks -> Hold'em value, without and with a flush."""
    plain: dict[int, tuple[int, ...]] = {}
    flush: dict[int, tuple[int, ...]] = {}
    for combo in itertools.combinations_with_replacement(range(13), 5):
        key = 1
        for r in combo:
            key *= _PRIMES[r]
        if max(combo.count(r) for r in combo) > 4:
            continue
        suits = "cdhsc"
        cards = ["23456789TJQKA"[r] + suits[i] for i, r in enumerate(combo)]
        plain[key] = evaluate(cards)
        if len(set(combo)) == 5:
            flush[key] = evaluate(["23456789TJQKA"[r] + "h" for r in combo])
    return plain, flush


def omaha_value(hole: Sequence[str], board: Sequence[str]) -> tuple[int, ...]:
    """Best Hold'em value using exactly two hole cards and three board cards."""
    plain, flush = _five_tables()
    pairs = []
    for a, b in itertools.combinations(hole, 2):
        pairs.append((_PRIMES[RV[a[0]]] * _PRIMES[RV[b[0]]], a[1] if a[1] == b[1] else None))
    best: tuple[int, ...] = (-1,)
    for x, y, z in itertools.combinations(board, 3):
        key = _PRIMES[RV[x[0]]] * _PRIMES[RV[y[0]]] * _PRIMES[RV[z[0]]]
        suit = x[1] if x[1] == y[1] == z[1] else None
        for pk, ps in pairs:
            k = key * pk
            v = flush[k] if suit is not None and ps == suit and k in flush else plain[k]
            if v > best:
                best = v
    return best


def _omaha_outs(hole: Sequence[str], board: Sequence[str]) -> int:
    if len(board) not in (3, 4):
        return 0
    outs = 0
    for s in "cdhs":
        if sum(1 for c in hole if c[1] == s) >= 2 and sum(1 for c in board if c[1] == s) == 2:
            outs += 9
            break
    board_ranks = {RV[c[0]] for c in board}
    hole_ranks = sorted({RV[c[0]] for c in hole})
    completing: set[int] = set()
    for low in range(-1, 9):
        window = {12 if r == -1 else r for r in range(low, low + 5)}
        mine = [r for r in hole_ranks if r in window]
        for a, b in itertools.combinations(mine, 2):
            missing = window - {a, b} - board_ranks
            if len(missing) == 1:
                completing |= missing
    return outs + 4 * len(completing)


def strength_plo(hole: Sequence[str], board: Sequence[str]) -> Strength:
    value = omaha_value(hole, board)
    cat = value[0]
    board_cat = evaluate(list(board))[0]
    outs = _omaha_outs(hole, board)
    top_rank = max(RV[c[0]] for c in board)
    hole_ranks = [RV[c[0]] for c in hole]
    if cat >= STRAIGHT and cat > board_cat:
        return "monster", outs, value
    if cat == TRIPS and board_cat < PAIR and hole_ranks.count(value[1]) >= 2:
        return "monster", outs, value
    if cat in (TRIPS, TWO_PAIR) and board_cat < cat:
        return ("strong" if value[1] >= top_rank else "medium"), outs, value
    if cat == PAIR and board_cat == HIGH_CARD:
        if value[1] > top_rank:
            return "medium", outs, value
        return ("medium" if value[1] == top_rank else "weak"), outs, value
    return "weak", outs, value


def strength_for(ruleset: str) -> StrengthFn:
    return {"practice_nlhe": strength, "uncertainty_shortdeck": strength_shortdeck,
            "uncertainty_plo4": strength_plo}[card_ruleset(ruleset)]


# ------------------------------------------------------------------ board texture

def cards_rank(card: str) -> int:
    return RV[card[0]]


def texture(board: Sequence[str]) -> str:
    """``monotone``, ``paired``, ``wet`` (two-tone or connected) or ``dry``, from the flop."""
    flop = board[:3]
    if len(flop) < 3:
        return "dry"
    suits = {c[1] for c in flop}
    ranks = sorted({RV[c[0]] for c in flop})
    if len(suits) == 1:
        return "monotone"
    if len(ranks) < 3:
        return "paired"
    connected = ranks[-1] - ranks[0] <= 4 or (ranks[-1] == 12 and ranks[1] <= 3)
    if len(suits) == 2 or connected:
        return "wet"
    return "dry"


def warm() -> None:
    """Load every table at import (bot load), so no decision pays for it against the deadline."""
    _nlhe()
    _shortdeck()
    _plo_quantiles()
    _five_tables()


warm()

BUCKET_SCORE = {"monster": 1.0, "strong": 0.75, "medium": 0.5, "weak": 0.15}
BUCKETS = ("weak", "medium", "strong", "monster")
__all__ = ["BUCKETS", "BUCKET_SCORE", "FLUSH", "FULL_HOUSE", "STRAIGHT_FLUSH", "TWO_PAIR", "VARIANTS",
           "equivalent_class", "evaluate_shortdeck", "nlhe_class", "nlhe_equity", "omaha_value", "plo_score",
           "preflop_top", "strength_for", "strength_plo", "strength_shortdeck", "texture", "top"]
