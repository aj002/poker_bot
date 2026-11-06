"""Push/fold subgames, keys and harvests with the antes in the pot (decision-time safe up to the
solver: :class:`AnteSnapshot`, :func:`ante_seated_snapshot` and :func:`ante_keys_for` import no numpy).

The ante-free modules (:mod:`strategy.endgame.keys`, ``subgame``, ``harvest``, ``points``) are
extended here rather than edited, so the committed ante-free tables and their provenance stay
as they are. Antes follow the arena (docs/assumptions.md, "Antes"):

* ``classic``: every dealt seat antes ``min(ante, stack)`` first and posts its blind from what is
  left; antes are contribution levels (a seat all-in on its ante wins only what it matched);
* ``big_blind``: the big blind posts its blind, then antes what is left up to the ante; that
  ante is dead money in the main pot.

Keys are the ante-free ladder plus :func:`ante_bucket`, the ante money over the big blind, at
every level, offset so no ante key equals an ante-free one.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from strategy.endgame.keys import Snapshot, keys_for
from strategy.endgame.table_state import SeatedSnapshot, seated_snapshot

POSTINGS = ("classic", "big_blind")
ANTE_EDGES = (0.15, 0.3, 0.45, 0.6, 0.8, 1.05, 1.3, 1.7, 2.3)
"""Ante money in the pot over the big blind."""
ANTE_KEY_SHIFT = 52
"""Ante-free keys use at most 48 bits; an ante key adds ``(bucket + 1) << ANTE_KEY_SHIFT``."""


@dataclass(frozen=True)
class AnteSnapshot(Snapshot):
    antes: tuple[int, ...] = ()
    """Ante each player posted, preflop action order; ``posts`` are the blinds posted after them."""
    posting: str = "classic"

    @property
    def dead(self) -> int:
        return sum(self.antes) if self.posting == "big_blind" else 0

    @property
    def ante_total(self) -> int:
        return sum(self.antes)


def ante_bucket(snap: AnteSnapshot) -> int:
    return bisect_right(ANTE_EDGES, snap.ante_total / snap.bb)


def ante_keys_for(kind: str, snap: AnteSnapshot, hero: int, shover: int | None = None,
                  caller: int | None = None) -> tuple[int, ...]:
    off = (ante_bucket(snap) + 1) << ANTE_KEY_SHIFT
    return tuple(off + k for k in keys_for(kind, snap, hero, shover, caller))


def seat_antes(order: Sequence[int], start_stacks: Mapping[int, int], big_blind: int, bb: int, ante: int,
               posting: str) -> dict[int, int]:
    if posting == "classic":
        return {s: min(ante, start_stacks[s]) for s in order}
    if posting == "big_blind":
        return {big_blind: min(ante, max(0, start_stacks[big_blind] - bb))}
    raise ValueError(f"ante posting is one of {POSTINGS}, not {posting!r}")


def ante_seated_snapshot(*, ring: Sequence[int], start_stacks: Mapping[int, int], button: int,
                         small_blind_seat: int | None, big_blind: int, heads_up: bool, blinds: tuple[int, int],
                         payouts: Sequence[float], hand_number: int, hand_cap: int, ante: int,
                         posting: str) -> SeatedSnapshot:
    """:func:`~strategy.endgame.table_state.seated_snapshot` with the level's ante posted."""
    base = seated_snapshot(ring=ring, start_stacks=start_stacks, button=button, small_blind_seat=small_blind_seat,
                           big_blind=big_blind, heads_up=heads_up, blinds=blinds, payouts=payouts,
                           hand_number=hand_number, hand_cap=hand_cap)
    order = base.seats
    sb, bb = blinds
    antes = seat_antes(order, start_stacks, big_blind, bb, ante, posting)
    left = {s: start_stacks[s] - (antes.get(s, 0) if posting == "classic" else 0) for s in order}
    sb_seat = button if heads_up else small_blind_seat
    posts = tuple(min(sb, left[s]) if s == sb_seat and s != big_blind else
                  min(bb, left[s]) if s == big_blind else 0 for s in order)
    b = base.snapshot
    snap = AnteSnapshot(b.stacks, posts, b.bb, b.payouts, b.hand_number, b.hand_cap,
                        tuple(antes.get(s, 0) for s in order), posting)
    return SeatedSnapshot(snap, base.seats, base.odd_order)
