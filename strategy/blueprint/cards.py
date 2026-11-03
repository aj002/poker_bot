"""Card buckets of the blueprint, computed by the same code offline and at decision time.

Preflop: 15 buckets of the 169 canonical classes by combo-weighted strength percentile
(:class:`strategy.features.equity.EquityTables`), finer among the strongest hands.

Postflop: the equity (against one random hand) of the hand's feature cell
(made level x draw class x board texture, :func:`strategy.features.cards.postflop_key`)
in 8 bins, times an auxiliary dimension:

* flop and turn (4): no draw with a weak hole hand, no draw with a strong hole hand
  (preflop percentile >= 0.6: overcards, kickers), weak draw (<= 7 outs), strong draw;
* river (2): weak or strong hole hand.

So 32 flop and turn buckets and 16 river buckets. The cell is a coarse, equity-only
summary (no blockers, no exact kicker on made hands): the abstraction's main loss.
"""

from __future__ import annotations

from collections.abc import Sequence

from strategy.blueprint.abstraction import BUCKETS
from strategy.features.cards import (CardCapabilities, canonical_hole_key, draws, made_hand, postflop_key,
                                     texture)
from strategy.features.equity import EquityTables

PREFLOP_PCT_EDGES = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.68, 0.75, 0.81, 0.86, 0.90, 0.93, 0.96, 0.98)
FLOP_TURN_EQ_EDGES = (0.30, 0.37, 0.45, 0.55, 0.65, 0.75, 0.87)
RIVER_EQ_EDGES = (0.20, 0.28, 0.40, 0.55, 0.65, 0.75, 0.87)
STRONG_HOLE_PCT = 0.6
EQ_BINS = 8


def _bin(x: float, edges: Sequence[float]) -> int:
    b = 0
    for e in edges:
        if x > e:
            b += 1
    return b


class CardAbstraction:
    """Bucket lookups for one hold'em variant; build once per process."""

    def __init__(self, caps: CardCapabilities, directory: str | None = None) -> None:
        if caps.hole_count != 2 or caps.hole_cards_used is not None:
            raise ValueError(f"{caps.ruleset_id}: the blueprint card abstraction is for two-card hold'em")
        self.caps = caps
        self.tables = EquityTables(caps, directory)
        if not (self.tables.has_preflop and self.tables.has_postflop):
            raise LookupError(f"{caps.ruleset_id}: preflop and postflop equity tables are required")
        self._pre: dict[int, tuple[int, float]] = {}
        self._tex: dict[tuple[int, ...], int] = {}

    def texture_key(self, board: Sequence[int]) -> int:
        """Texture key of a board, memoised for the few boards of the current deal or hand."""
        k = tuple(board)
        hit = self._tex.get(k)
        if hit is None:
            if len(self._tex) >= 16:
                self._tex.clear()
            hit = self._tex[k] = texture(board, self.caps).key
        return hit

    def preflop(self, hole: Sequence[int]) -> tuple[int, float]:
        """``(bucket, percentile)`` of a hole hand."""
        key = canonical_hole_key(hole)
        hit = self._pre.get(key)
        if hit is None:
            pct = self.tables.preflop(hole, 1).percentile
            hit = self._pre[key] = (_bin(pct, PREFLOP_PCT_EDGES), pct)
        return hit

    def postflop(self, hole: Sequence[int], board: Sequence[int], street: int, pct: float) -> int:
        caps = self.caps
        made = made_hand(hole, board, caps)
        strong = 1 if pct >= STRONG_HOLE_PCT else 0
        if street == 3:
            cell = postflop_key(3, made.level, 0, self.texture_key(board))
            eq = self.tables.postflop(cell, made.level, 0, 1).equity
            return _bin(eq, RIVER_EQ_EDGES) * 2 + strong
        dc = draws(hole, board, caps, made).draw_class
        cell = postflop_key(street, made.level, dc, self.texture_key(board))
        eq = self.tables.postflop(cell, made.level, dc, 1).equity
        aux = strong if dc == 0 else 2 if dc <= 2 else 3
        return _bin(eq, FLOP_TURN_EQ_EDGES) * 4 + aux

    def bucket(self, hole: Sequence[int], board: Sequence[int]) -> int:
        b, pct = self.preflop(hole)
        n = len(board)
        if n == 0:
            return b
        return self.postflop(hole, board, 1 if n == 3 else 2 if n == 4 else 3, pct)

    def street_buckets(self, hole: Sequence[int], board: Sequence[int]) -> tuple[int, int, int, int]:
        """Buckets on all four streets for a hole hand and a full five-card board."""
        b, pct = self.preflop(hole)
        return (b, self.postflop(hole, board[:3], 1, pct), self.postflop(hole, board[:4], 2, pct),
                self.postflop(hole, board[:5], 3, pct))


assert len(PREFLOP_PCT_EDGES) + 1 == BUCKETS[0]
assert (len(FLOP_TURN_EQ_EDGES) + 1) * 4 == BUCKETS[1] == BUCKETS[2]
assert (len(RIVER_EQ_EDGES) + 1) * 2 == BUCKETS[3]
