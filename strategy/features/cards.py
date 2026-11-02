"""Variant-aware card features for decision time.

Everything here is bounded integer work over bitmasks: no Monte Carlo, no search.
Equity comes from offline tables (:mod:`strategy.tables`), looked up by keys this
module computes, so the generators and the runtime share one definition of every key.

Cards are ints ``rank * 4 + suit`` with rank 0..12 for ``2..A`` and suit 0..3 for
``c, d, h, s``. A variant is described by :class:`CardCapabilities` (ranks in the deck,
hole-card count, how many hole cards a hand must use, and the category order), so short
deck and Omaha are metadata, not separate code paths, except for the exactly-two-hole
evaluator.

Evaluator values are comparable ints: ``strength << 20`` plus five 4-bit tiebreak ranks,
where ``strength`` is the category's position in the variant's ``category_order``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations, combinations_with_replacement

RANKS = "23456789TJQKA"
SUITS = "cdhs"
RANK_INDEX = {r: i for i, r in enumerate(RANKS)}
CARD_INDEX = {r + s: i * 4 + j for i, r in enumerate(RANKS) for j, s in enumerate(SUITS)}
CARD_NAMES = tuple(r + s for r in RANKS for s in SUITS)
PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)
_R_DESC = tuple(range(12, -1, -1))
TEN = RANK_INDEX["T"]

HIGH_CARD, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH = range(9)
CATEGORY_NAMES = ("high_card", "pair", "two_pair", "trips", "straight", "flush", "full_house", "quads",
                  "straight_flush")
STANDARD_ORDER = (HIGH_CARD, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH)
# PokerKit's native short-deck order (docs/assumptions.md): a flush beats a full house.
SHORT_DECK_ORDER = (HIGH_CARD, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FULL_HOUSE, FLUSH, QUADS, STRAIGHT_FLUSH)
SPEC_CATEGORIES = {"high_card": HIGH_CARD, "one_pair": PAIR, "two_pair": TWO_PAIR, "three_of_a_kind": TRIPS,
                   "straight": STRAIGHT, "flush": FLUSH, "full_house": FULL_HOUSE, "four_of_a_kind": QUADS,
                   "straight_flush": STRAIGHT_FLUSH}
"""Category names of :class:`~strategy.rules.variants.VariantSpec` to the evaluator's constants."""

MADE_LEVELS = 11
"""Made-hand levels: 0 nothing/plays the board, 1 weak pair, 2 middle pair, 3 top pair,
4 overpair, 5 two pair, 6 trips or set, 7 straight, 8 and 9 the next two categories in the
variant's order (flush/full house), 10 quads or better."""
DRAW_CLASSES = 5
TEXTURE_KEYS = 18
TURN_CLASSES = ("none", "low_blank", "overcard", "flush_completer", "straight_completer", "board_pair",
                "dual_change")


def card_ints(cards: Sequence[str]) -> tuple[int, ...]:
    return tuple(CARD_INDEX[c] for c in cards)


# ------------------------------------------------------------------ capabilities


@dataclass(frozen=True)
class CardCapabilities:
    """What the card layer needs to know about a variant."""

    ruleset_id: str
    ranks: str
    hole_count: int
    hole_cards_used: int | None
    """``None``: best five of hole plus board. ``2``: exactly two hole and three board cards."""
    category_order: tuple[int, ...] = STANDARD_ORDER
    board_count: int = 5

    rank_indices: tuple[int, ...] = field(init=False, repr=False, compare=False)
    strength: tuple[int, ...] = field(init=False, repr=False, compare=False)
    windows: tuple[tuple[int, int], ...] = field(init=False, repr=False, compare=False)
    straight_high: tuple[int, ...] = field(init=False, repr=False, compare=False)
    deck: tuple[int, ...] = field(init=False, repr=False, compare=False)
    top_rank: int = field(init=False, repr=False, compare=False)
    standard_priority: bool = field(init=False, repr=False, compare=False)
    """Non-flush categories rank as in hold'em (the evaluator's fast path)."""

    def __post_init__(self) -> None:
        idx = tuple(RANK_INDEX[r] for r in self.ranks)
        if sorted(self.category_order) != list(range(9)):
            raise ValueError("category_order must be a permutation of the nine categories")
        strength = [0] * 9
        for pos, cat in enumerate(self.category_order):
            strength[cat] = pos
        wins = [(sum(1 << r for r in idx[i:i + 5]), idx[i + 4]) for i in range(len(idx) - 4)]
        wheel = (1 << 12) | sum(1 << r for r in idx[:4])
        wins.append((wheel, idx[3]))
        wins.sort(key=lambda w: -w[1])
        table = [-1] * 8192
        for mask in range(8192):
            for w, high in wins:
                if mask & w == w:
                    table[mask] = high
                    break
        set_ = object.__setattr__
        set_(self, "rank_indices", idx)
        set_(self, "strength", tuple(strength))
        set_(self, "windows", tuple(wins))
        set_(self, "straight_high", tuple(table))
        set_(self, "deck", tuple(r * 4 + s for r in idx for s in range(4)))
        set_(self, "top_rank", idx[-1])
        pri = [strength[c] for c in (HIGH_CARD, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FULL_HOUSE, QUADS)]
        set_(self, "standard_priority", pri == sorted(pri))

    @property
    def deck_size(self) -> int:
        return len(self.deck)

    def category(self, value: int) -> int:
        return self.category_order[value >> 20]


@lru_cache(maxsize=None)
def capabilities(ruleset_id: str) -> CardCapabilities:
    """Capabilities for a ruleset, derived from its :class:`~strategy.rules.variants.VariantSpec`."""
    from strategy.rules.reveal import get_variant

    v = get_variant(ruleset_id)
    try:
        order = tuple(SPEC_CATEGORIES[c] for c in v.category_order)
    except KeyError:
        raise ValueError(f"{ruleset_id}: unsupported category order {v.category_order}") from None
    if v.suits != SUITS or v.board_streets != (3, 1, 1) or v.lowest_straight != v.ranks[-1] + v.ranks[:4]:
        raise ValueError(f"{ruleset_id}: the card layer needs suits cdhs, a 3-1-1 board and an ace-low wheel")
    used = 2 if v.hand_selection == "two_hole_three_board" else None
    return CardCapabilities(ruleset_id, v.ranks, v.hole_cards, used, order, v.board_cards)


# --------------------------------------------------------------------- evaluator


def _pack(strength: int, tb: Sequence[int]) -> int:
    v = strength
    n = len(tb)
    for i in range(5):
        v = (v << 4) | (tb[i] if i < n else 0)
    return v


def _rank_value(counts: Sequence[int], mask: int, caps: CardCapabilities) -> int:
    """Best non-flush value from rank counts (any number of cards)."""
    st = caps.strength
    quads: list[int] = []
    trips: list[int] = []
    pairs: list[int] = []
    singles: list[int] = []
    for r in _R_DESC:
        k = counts[r]
        if k == 1:
            singles.append(r)
        elif k == 2:
            pairs.append(r)
        elif k == 3:
            trips.append(r)
        elif k:
            quads.append(r)
    if not caps.standard_priority:
        return _best_rank_value(counts, mask, caps)
    if quads:
        q = quads[0]
        kick = next((r for r in _R_DESC if r != q and counts[r]), 0)
        return _pack(st[QUADS], (q, kick))
    if trips and (len(trips) > 1 or pairs):
        p = max(trips[1] if len(trips) > 1 else -1, pairs[0] if pairs else -1)
        return _pack(st[FULL_HOUSE], (trips[0], p))
    high = caps.straight_high[mask]
    if high >= 0:
        return _pack(st[STRAIGHT], (high,))
    if trips:
        return _pack(st[TRIPS], (trips[0], *singles[:2]))
    if len(pairs) >= 2:
        kick = max(pairs[2] if len(pairs) > 2 else -1, singles[0] if singles else -1)
        return _pack(st[TWO_PAIR], (pairs[0], pairs[1], max(kick, 0)))
    if pairs:
        return _pack(st[PAIR], (pairs[0], *singles[:3]))
    return _pack(st[HIGH_CARD], singles[:5])


def _best_rank_value(counts: Sequence[int], mask: int, caps: CardCapabilities) -> int:
    """The strongest non-flush five-card hand under any category order (every candidate, then max)."""
    st = caps.strength
    ranks = [r for r in _R_DESC if counts[r]]
    multi = [r for r in ranks if counts[r] >= 2]
    out: list[int] = []
    if len(ranks) >= 5:
        out.append(_pack(st[HIGH_CARD], ranks[:5]))
    high = caps.straight_high[mask]
    if high >= 0:
        out.append(_pack(st[STRAIGHT], (high,)))
    if multi:
        p = multi[0]
        out.append(_pack(st[PAIR], (p, *[r for r in ranks if r != p][:3])))
    if len(multi) >= 2:
        a, b = multi[0], multi[1]
        out.append(_pack(st[TWO_PAIR], (a, b, next((r for r in ranks if r != a and r != b), 0))))
    for t in (r for r in ranks if counts[r] >= 3):
        out.append(_pack(st[TRIPS], (t, *[r for r in ranks if r != t][:2])))
        p = next((r for r in multi if r != t), -1)
        if p >= 0:
            out.append(_pack(st[FULL_HOUSE], (t, p)))
        break
    q = next((r for r in ranks if counts[r] >= 4), -1)
    if q >= 0:
        out.append(_pack(st[QUADS], (q, next((r for r in ranks if r != q), 0))))
    return max(out)


def _flush_value(suit_mask: int, caps: CardCapabilities) -> int:
    high = caps.straight_high[suit_mask]
    if high >= 0:
        return _pack(caps.strength[STRAIGHT_FLUSH], (high,))
    return _pack(caps.strength[FLUSH], [r for r in _R_DESC if suit_mask >> r & 1][:5])


def evaluate_any(cards: Sequence[int], caps: CardCapabilities) -> int:
    """Best five of ``cards`` (hold'em family)."""
    counts = [0] * 13
    sm = [0, 0, 0, 0]
    sc = [0, 0, 0, 0]
    mask = 0
    for c in cards:
        r = c >> 2
        s = c & 3
        counts[r] += 1
        sm[s] |= 1 << r
        sc[s] += 1
        mask |= 1 << r
    best = _rank_value(counts, mask, caps)
    for s in range(4):
        if sc[s] >= 5:
            fv = _flush_value(sm[s], caps)
            if fv > best:
                best = fv
    return best


_RANK5: dict[str, dict[int, int]] = {}


def _rank5(caps: CardCapabilities) -> dict[int, int]:
    """Prime-product key of a five-rank multiset -> non-flush value (6,175 entries for 13 ranks)."""
    table = _RANK5.get(caps.ruleset_id)
    if table is None:
        table = {}
        for combo in combinations_with_replacement(caps.rank_indices, 5):
            counts = [0] * 13
            mask = key = 0
            for r in combo:
                counts[r] += 1
                mask |= 1 << r
            if max(counts) > 4:
                continue
            key = 1
            for r in combo:
                key *= PRIMES[r]
            table[key] = _rank_value(counts, mask, caps)
        _RANK5[caps.ruleset_id] = table
    return table


def evaluate_omaha(hole: Sequence[int], board: Sequence[int], caps: CardCapabilities) -> int:
    """Exactly two hole cards and three board cards."""
    r5 = _rank5(caps)
    hp = {PRIMES[a >> 2] * PRIMES[b >> 2] for a, b in combinations(hole, 2)}
    bp = {PRIMES[a >> 2] * PRIMES[b >> 2] * PRIMES[c >> 2] for a, b, c in combinations(board, 3)}
    best = max(r5[x * y] for x in hp for y in bp)
    for s in range(4):
        hm = bm = 0
        hn = bn = 0
        for c in hole:
            if c & 3 == s:
                hm |= 1 << (c >> 2)
                hn += 1
        if hn < 2:
            continue
        for c in board:
            if c & 3 == s:
                bm |= 1 << (c >> 2)
                bn += 1
        if bn < 3:
            continue
        for w, high in caps.windows:
            if (w & hm).bit_count() == 2 and (w & bm).bit_count() == 3:
                v = _pack(caps.strength[STRAIGHT_FLUSH], (high,))
                break
        else:
            top_h = [r for r in _R_DESC if hm >> r & 1][:2]
            top_b = [r for r in _R_DESC if bm >> r & 1][:3]
            v = _pack(caps.strength[FLUSH], sorted(top_h + top_b, reverse=True))
        if v > best:
            best = v
    return best


def evaluate(hole: Sequence[int], board: Sequence[int], caps: CardCapabilities) -> int:
    if caps.hole_cards_used == 2:
        return evaluate_omaha(hole, board, caps)
    return evaluate_any((*hole, *board), caps)


# ----------------------------------------------------------- canonical hole class


def canonical_hole_key(hole: Sequence[int]) -> int:
    """Suit-isomorphism class of the hole cards, packed 6 bits per card.

    Suits are relabelled in order of (cards in suit, ranks in suit) descending, which
    is invariant under suit permutations and keeps non-isomorphic hands distinct.
    """
    groups: dict[int, list[int]] = {}
    for c in hole:
        groups.setdefault(c & 3, []).append(c >> 2)
    ordered = sorted((sorted(rs, reverse=True) for rs in groups.values()), key=lambda rs: (len(rs), rs),
                     reverse=True)
    out = sorted((r * 4 + s for s, rs in enumerate(ordered) for r in rs), reverse=True)
    key = 0
    for c in out:
        key = (key << 6) | c
    return key


def hole_label(hole: Sequence[int]) -> str:
    """``AKs``/``T9o``/``77`` for two cards; canonical cards such as ``AaKaQbJb`` otherwise."""
    if len(hole) == 2:
        a, b = sorted(hole, reverse=True)
        if a >> 2 == b >> 2:
            return RANKS[a >> 2] * 2
        return RANKS[a >> 2] + RANKS[b >> 2] + ("s" if a & 3 == b & 3 else "o")
    key = canonical_hole_key(hole)
    cards = [(key >> (6 * i)) & 63 for i in range(len(hole) - 1, -1, -1)]
    return "".join(RANKS[c >> 2] + "abcd"[c & 3] for c in cards)


# ------------------------------------------------------------------- made hands


@dataclass(frozen=True, slots=True)
class MadeHand:
    value: int
    category: int
    level: int
    kicker: int
    """Highest hole rank not used as the pair rank (pair levels), else -1."""


def _pair_level(p: int, bc: Sequence[int], distinct_desc: Sequence[int]) -> int:
    if bc[p] >= 2:
        return 0
    top = distinct_desc[0]
    second = distinct_desc[1] if len(distinct_desc) > 1 else -1
    if bc[p] == 1:
        return 3 if p == top else 2 if p == second else 1
    if p > top:
        return 4
    return 2 if p > second else 1


def made_hand(hole: Sequence[int], board: Sequence[int], caps: CardCapabilities) -> MadeHand:
    value = evaluate(hole, board, caps)
    cat = caps.category(value)
    bc = [0] * 13
    for c in board:
        bc[c >> 2] += 1
    distinct = [r for r in _R_DESC if bc[r]]
    board_max = max(bc)
    t0 = (value >> 16) & 15
    t1 = (value >> 12) & 15
    hole_ranks = sorted((c >> 2 for c in hole), reverse=True)
    kicker = -1
    if cat >= STRAIGHT:
        plays_board = False
        if caps.hole_cards_used is None and len(board) >= 5:
            plays_board = evaluate_any(board, caps) == value
        elif cat == QUADS and board_max == 4:
            plays_board = True
        level = 0 if plays_board else min(10, 7 + caps.strength[cat] - caps.strength[STRAIGHT])
    elif cat == TRIPS:
        level = 0 if board_max >= 3 else 6
    elif cat == TWO_PAIR:
        board_pairs = sum(1 for r in distinct if bc[r] >= 2)
        if board_pairs >= 2:
            level = 0
        elif board_pairs == 1:
            own = t1 if bc[t0] >= 2 else t0
            level = _pair_level(own, bc, distinct)
            kicker = next((r for r in hole_ranks if r != own), -1)
        else:
            level = 5
    elif cat == PAIR:
        level = _pair_level(t0, bc, distinct)
        kicker = next((r for r in hole_ranks if r != t0), -1)
    else:
        level = 0
    return MadeHand(value, cat, level, kicker)


# ------------------------------------------------------------------------ draws


@dataclass(frozen=True, slots=True)
class Draws:
    flush_draw: bool = False
    nut_flush_draw: bool = False
    backdoor_flush: bool = False
    straight_out_ranks: int = 0
    oesd: bool = False
    gutshot: bool = False
    outs: int = 0
    clean_outs: int = 0
    draw_class: int = 0


NO_DRAWS = Draws()


def _top_unseen_rank(suit: int, board: Sequence[int], caps: CardCapabilities) -> int:
    on_board = {c >> 2 for c in board if c & 3 == suit}
    return next(r for r in reversed(caps.rank_indices) if r not in on_board)


def draw_class(outs: int) -> int:
    return 0 if outs == 0 else 1 if outs <= 4 else 2 if outs <= 7 else 3 if outs <= 11 else 4


def draws(hole: Sequence[int], board: Sequence[int], caps: CardCapabilities, made: MadeHand | None = None) -> Draws:
    """Flush and straight draws with one card to come or more; nothing on the river."""
    n = len(board)
    if n < 3 or n >= caps.board_count:
        return NO_DRAWS
    made = made or made_hand(hole, board, caps)
    have_flush = made.category in (FLUSH, STRAIGHT_FLUSH)
    have_straight = made.category in (STRAIGHT, STRAIGHT_FLUSH)
    omaha = caps.hole_cards_used == 2
    seen = 0
    for c in hole:
        seen |= 1 << c
    for c in board:
        seen |= 1 << c
    hs = [0, 0, 0, 0]
    bs = [0, 0, 0, 0]
    for c in hole:
        hs[c & 3] += 1
    for c in board:
        bs[c & 3] += 1
    flush_draw = nut = backdoor = False
    flush_cards = 0
    flush_suits = 0
    if not have_flush:
        for s in range(4):
            if omaha:
                fd = hs[s] >= 2 and bs[s] == 2
                bd = hs[s] >= 2 and bs[s] == 1 and n == 3
            else:
                fd = hs[s] >= 1 and hs[s] + bs[s] == 4
                bd = hs[s] >= 1 and hs[s] + bs[s] == 3 and n == 3
            if bd:
                backdoor = True
            if fd:
                flush_draw = True
                flush_suits |= 1 << s
                top = _top_unseen_rank(s, board, caps)
                if any(c == top * 4 + s for c in hole):
                    nut = True
                for r in caps.rank_indices:
                    if not seen >> (r * 4 + s) & 1:
                        flush_cards |= 1 << (r * 4 + s)
    out_ranks = 0
    if not have_straight:
        bm = 0
        for c in board:
            bm |= 1 << (c >> 2)
        if omaha:
            hr = {c >> 2 for c in hole}
            for a, b in combinations(sorted(hr), 2):
                pm = (1 << a) | (1 << b)
                for w, _ in caps.windows:
                    if w & pm == pm:
                        miss = (w & ~pm) & ~bm
                        if miss and miss & (miss - 1) == 0:
                            out_ranks |= miss
        else:
            hm = 0
            for c in hole:
                hm |= 1 << (c >> 2)
            allm = hm | bm
            for w, _ in caps.windows:
                miss = w & ~allm
                if miss and miss & (miss - 1) == 0 and w & hm & ~bm:
                    out_ranks |= miss
    straight_cards = 0
    if out_ranks:
        for r in caps.rank_indices:
            if out_ranks >> r & 1:
                for s in range(4):
                    if not seen >> (r * 4 + s) & 1:
                        straight_cards |= 1 << (r * 4 + s)
    out_cards = flush_cards | straight_cards
    outs = out_cards.bit_count()
    dirty = 0
    if outs:
        board_ranks = 0
        for c in board:
            board_ranks |= 1 << (c >> 2)
        x = out_cards
        while x:
            low = x & -x
            c = low.bit_length() - 1
            x ^= low
            s = c & 3
            if flush_cards >> c & 1:
                if board_ranks >> (c >> 2) & 1:
                    dirty += 1
            elif bs[s] >= 2 and not flush_suits >> s & 1:
                dirty += 1
    nr = out_ranks.bit_count()
    return Draws(flush_draw, nut, backdoor, out_ranks, nr >= 2, nr == 1, outs, outs - dirty, draw_class(outs))


# ---------------------------------------------------------------------- blockers


@dataclass(frozen=True, slots=True)
class Blockers:
    nut_flush: bool = False
    """Hero holds the highest rank of a suit with three or more board cards."""
    nut_flush_draw: bool = False
    """Hero holds the highest rank of a two-card board suit (draws still live)."""
    top_rank: int = 0
    board_ranks: int = 0
    straight: int = 0
    """Hero cards whose ranks complete a board straight for an opponent."""


NO_BLOCKERS = Blockers()


def blockers(hole: Sequence[int], board: Sequence[int], caps: CardCapabilities) -> Blockers:
    if not board:
        return NO_BLOCKERS
    bs = [0, 0, 0, 0]
    bm = 0
    for c in board:
        bs[c & 3] += 1
        bm |= 1 << (c >> 2)
    nut_flush = nut_fd = False
    for s in range(4):
        if bs[s] >= 2:
            top = _top_unseen_rank(s, board, caps) * 4 + s
            if top in hole:
                if bs[s] >= 3:
                    nut_flush = True
                elif len(board) < caps.board_count:
                    nut_fd = True
    top_rank = max(c >> 2 for c in board)
    need = 0
    for w, _ in caps.windows:
        if (w & bm).bit_count() >= 3:
            need |= w & ~bm
    return Blockers(nut_flush, nut_fd, sum(1 for c in hole if c >> 2 == top_rank),
                    sum(1 for c in hole if bm >> (c >> 2) & 1), sum(1 for c in hole if need >> (c >> 2) & 1))


# ----------------------------------------------------------------------- texture


@dataclass(frozen=True, slots=True)
class Texture:
    cards: int = 0
    max_suit: int = 0
    flush_possible: bool = False
    flush_draw_possible: bool = False
    paired: bool = False
    pairs: int = 0
    trips: bool = False
    high_rank: int = -1
    broadway: int = 0
    straight_windows: int = 0
    """Straight windows holding three or more board ranks: a made straight is possible."""
    draw_windows: int = 0
    """Windows holding exactly two board ranks with cards to come: straight-draw density."""
    nut_category: int = HIGH_CARD
    dynamic: float = 0.0
    saturation: float = 0.0
    key: int = 0
    """Coarse texture key 0..17 used by the equity tables: flush state x paired x straight state."""


NO_TEXTURE = Texture()


def texture(board: Sequence[int], caps: CardCapabilities) -> Texture:
    n = len(board)
    if n == 0:
        return NO_TEXTURE
    bs = [0, 0, 0, 0]
    bc = [0] * 13
    bm = 0
    for c in board:
        bs[c & 3] += 1
        bc[c >> 2] += 1
        bm |= 1 << (c >> 2)
    max_suit = max(bs)
    to_come = n < caps.board_count
    pairs = sum(1 for k in bc if k >= 2)
    trips = max(bc) >= 3
    w3 = w2 = 0
    for w, _ in caps.windows:
        k = (w & bm).bit_count()
        if k >= 3:
            w3 += 1
        elif k == 2 and to_come:
            w2 += 1
    flush_possible = max_suit >= 3
    fdp = max_suit == 2 and to_come
    sf_possible = False
    if flush_possible:
        for s in range(4):
            if bs[s] >= 3:
                sm = 0
                for c in board:
                    if c & 3 == s:
                        sm |= 1 << (c >> 2)
                if any((w & sm).bit_count() >= 3 for w, _ in caps.windows):
                    sf_possible = True
    if sf_possible:
        nut = STRAIGHT_FLUSH
    elif pairs or trips:
        nut = QUADS
    else:
        st = caps.strength
        candidates = [TRIPS]
        if w3:
            candidates.append(STRAIGHT)
        if flush_possible:
            candidates.append(FLUSH)
        nut = max(candidates, key=lambda c: st[c])
    n_windows = len(caps.windows)
    dynamic = min(1.0, 0.5 * w2 / max(1, n_windows // 2) + (0.5 if fdp else 0.0)) if to_come else 0.0
    saturation = (min(1.0, w3 / 2) + (1.0 if flush_possible else 0.0) + (1.0 if pairs else 0.0)) / 3.0
    flush_state = 2 if flush_possible else 1 if max_suit == 2 else 0
    straight_state = 2 if w3 else 1 if w2 >= 2 else 0
    key = flush_state * 6 + (3 if pairs else 0) + straight_state
    return Texture(n, max_suit, flush_possible, fdp, pairs > 0, pairs, trips, max(c >> 2 for c in board),
                   sum(1 for c in board if c >> 2 >= TEN), w3, w2, nut, round(dynamic, 4), round(saturation, 4), key)


@dataclass(frozen=True, slots=True)
class StreetDelta:
    turn_class: int = 0
    """Index into :data:`TURN_CLASSES` for the newest board card (0 on the flop)."""
    pairs_board: bool = False
    overcard: bool = False
    completes_flush: bool = False
    completes_straight: bool = False
    nut_up: bool = False


NO_DELTA = StreetDelta()


def street_delta(prev: Texture, now: Texture, new_card: int | None, prev_board: Sequence[int]) -> StreetDelta:
    """How the newest board card changed the board (turn and river)."""
    if new_card is None or not prev_board:
        return NO_DELTA
    r = new_card >> 2
    pairs_board = any(c >> 2 == r for c in prev_board)
    overcard = r > prev.high_rank
    completes_flush = now.flush_possible and not prev.flush_possible
    completes_straight = now.straight_windows > prev.straight_windows
    nut_up = now.nut_category != prev.nut_category
    changes = int(completes_flush) + int(completes_straight) + int(pairs_board)
    if changes >= 2:
        cls = 6
    elif completes_flush:
        cls = 3
    elif completes_straight:
        cls = 4
    elif pairs_board:
        cls = 5
    elif overcard or r >= TEN:
        cls = 2
    else:
        cls = 1
    return StreetDelta(cls, pairs_board, overcard, completes_flush, completes_straight, nut_up)


def postflop_key(street: int, made_level: int, draw_cls: int, texture_key: int) -> int:
    """Cell index of the postflop equity table (``street`` 1 flop .. 3 river)."""
    return (((street - 1) * MADE_LEVELS + made_level) * DRAW_CLASSES + draw_cls) * TEXTURE_KEYS + texture_key


POSTFLOP_CELLS = 3 * MADE_LEVELS * DRAW_CLASSES * TEXTURE_KEYS


# ------------------------------------------------------------- per-street bundle


@dataclass(frozen=True, slots=True)
class CardFeatures:
    street: int
    hole_key: int
    label: str
    equity: float
    """Equity against ``opponents`` random hands, from the offline tables (ties as shares)."""
    equity_sd: float
    """Spread of hand equity within the bucket (preflop: across flops)."""
    equity_source: str
    """``table`` (exact cell), ``parent`` (coarser cell), ``prior`` (no table)."""
    percentile: float
    """Preflop: combo-weighted strength percentile of the class (1.0 = best)."""
    bucket: int
    """Preflop strength decile 0..9 (9 = best); the latent bucket for range beliefs."""
    made: MadeHand | None
    draws: Draws
    blockers: Blockers
    texture: Texture
    delta: StreetDelta
    level_delta: int
    """Change in made level since the previous street (0 preflop and on the flop)."""


def postflop_cards(hole: Sequence[int], board: Sequence[int], caps: CardCapabilities) -> tuple[MadeHand, Draws,
                                                                                               Blockers, Texture]:
    made = made_hand(hole, board, caps)
    return made, draws(hole, board, caps, made), blockers(hole, board, caps), texture(board, caps)
