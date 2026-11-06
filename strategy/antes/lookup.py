"""Decision-time push/fold with the antes solved in (no numpy on the decision path).

:func:`ante_preflop_spot` is :func:`strategy.endgame.lookup.preflop_spot` over an
:class:`~strategy.antes.pushfold.AnteSnapshot`: the hand-start stacks, the blinds and
the antes as posted, and the ante keys. :class:`AntePushFoldBot` plays a hand with an
ante from the posting mode's ante table (:mod:`strategy.antes.tables`), and falls back
to the ante-free leaf (its M-equivalent approximation) where that table has no support.
Hands without an ante never reach this module's tables.

The ante tables are endgame tables cut to the rows a lookup can use
(:func:`strategy.antes.tables.compact`), loaded as :class:`~strategy.endgame.lookup.PushFoldTable`
from :func:`~strategy.tables.store.stream_arrays`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from arena.agents.sdk import call, check, fold, raise_to
from strategy.antes.pushfold import POSTINGS, AnteSnapshot, ante_keys_for, ante_seated_snapshot
from strategy.endgame.bot import BLIND_OPEN_MAX_BB, PushFoldBot, _uniform
from strategy.endgame.keys import MAX_DEPTH_BB
from strategy.endgame.lookup import PushFoldTable, TableMissing, _hand_events
from strategy.endgame.table_state import SeatedSnapshot
from strategy.tables import store


@dataclass(frozen=True)
class AnteSpot:
    kind: str
    hero: int
    shover: int | None
    caller: int | None
    seated: SeatedSnapshot
    risk: int
    behind_live: bool

    @property
    def snapshot(self) -> AnteSnapshot:
        return self.seated.snapshot

    @property
    def risk_bb(self) -> float:
        return self.risk / self.snapshot.bb

    def keys(self) -> tuple[int, ...]:
        return ante_keys_for(self.kind, self.snapshot, self.hero, self.shover, self.caller)


def ante_preflop_spot(obs) -> AnteSpot | None:
    """The push/fold spot of a preflop decision in a hand with an ante, or None."""
    if obs.street != "preflop":
        return None
    events = _hand_events(obs)
    if events is None:
        return None
    start = events[0].data
    ante = int(start.get("ante") or 0)
    if not ante:
        return None
    pos = start["positions"]
    stacks = {int(s): int(c) for s, c in start["start_stacks"].items()}
    small = start.get("small_blind_seat")
    seated = ante_seated_snapshot(ring=sorted(obs.seats), start_stacks=stacks, button=pos["button"],
                                  small_blind_seat=small if small in stacks else None, big_blind=pos["big_blind"],
                                  heads_up=bool(pos["heads_up"]), blinds=(int(start["small_blind"]),
                                                                          int(start["big_blind"])),
                                  payouts=obs.payouts, hand_number=obs.hand_number, hand_cap=obs.hand_cap,
                                  ante=ante, posting=start["ante_posting"])
    snap = seated.snapshot
    seats = seated.seats
    if obs.hero_seat not in seats:
        return None
    index = {s: i for i, s in enumerate(seats)}
    posted = [0] * snap.n
    for e in events:
        if e.type == "AntePosted":
            i = index.get(int(e.data["seat"]))
            if i is None:
                return None
            posted[i] += int(e.data["amount"])
    if tuple(posted) != snap.antes:
        return None
    committed = [p + a for p, a in zip(snap.posts, snap.antes)]
    shover = caller = None
    acted: list[int] = []
    for e in events:
        if e.type != "ActionApplied" or e.data.get("street") != "preflop":
            continue
        i = index.get(e.data["seat"])
        if i is None:
            return None
        kind = e.data["action"]["type"]
        committed[i] += int(e.data.get("added") or 0)
        all_in = committed[i] >= snap.stacks[i]
        acted.append(i)
        if kind == "fold":
            continue
        if shover is None:
            if kind == "raise_to" and all_in:
                shover = i
                continue
            return None
        if kind in ("call", "raise_to") and caller is None:
            caller = i
            continue
        return None
    hero = index[obs.hero_seat]
    free = [snap.posts[p] + snap.antes[p] < snap.stacks[p] for p in range(snap.n)]
    if acted != [p for p in range(hero) if free[p]]:
        return None
    if shover is None:
        if hero == snap.n - 1:
            return None
        kind = "open"
        big = max(range(hero + 1, snap.n), key=lambda p: (snap.stacks[p], -p))
        risk = min(snap.stacks[hero], snap.stacks[big])
    elif caller is None:
        kind = "call"
        risk = min(snap.stacks[hero], snap.stacks[shover])
    else:
        kind = "overcall"
        risk = min(snap.stacks[hero], max(snap.stacks[shover], snap.stacks[caller]))
    return AnteSpot(kind, hero, shover, caller, seated, risk, any(free[p] for p in range(hero + 1, snap.n)))


@lru_cache(maxsize=None)
def load_ante_table(posting: str, ruleset_id: str = "practice_nlhe", directory: str | None = None) -> PushFoldTable:
    name = f"antes-pushfold-{posting}-{ruleset_id}"
    streamed = store.stream_arrays(name, directory)
    if streamed is None:
        where = Path(directory) if directory else store.table_dir()
        raise TableMissing(f"{name}.npz not found in {where}; run `python -m strategy.antes.tables --posting {posting}`")
    return PushFoldTable(*streamed)


class AntePushFoldBot(PushFoldBot):
    """The point-aware push/fold leaf, with each hand that has an ante played from its posting's ante table."""

    def __init__(self, mode: str = "points", **kw) -> None:
        super().__init__(mode, **kw)
        if mode != "points":
            raise ValueError("the ante tables are point-aware: mode must be 'points'")
        self.name = f"endgame_{mode}_antes"
        self.ante_stats = {"ante": 0, "ante_miss": 0}
        # In the untimed load: a restored worker goes straight to a timed decision.
        for posting in POSTINGS:
            load_ante_table(posting)

    def push_fold(self, obs, base, seed: str):
        blinds = obs.blinds
        if blinds is None or not blinds.ante or obs.street != "preflop":
            return super().push_fold(obs, base, seed)
        spot = ante_preflop_spot(obs)
        if spot is None:
            return super().push_fold(obs, base, seed)
        if spot.kind == "open":
            limit = BLIND_OPEN_MAX_BB if spot.hero == spot.snapshot.n - 2 else self.open_max_bb
        else:
            limit = MAX_DEPTH_BB
        snap = spot.snapshot
        sb_bb = blinds.small_blind + blinds.big_blind
        if spot.risk * sb_bb > limit * snap.bb * (sb_bb + snap.ante_total):
            return None
        found = load_ante_table(spot.snapshot.posting, obs.ruleset_id).lookup(spot.kind, spot.keys(),
                                                                             base.hand_class(obs))
        if found is None:
            self.ante_stats["ante_miss"] += 1
            return super().push_fold(obs, base, seed)
        self.ante_stats["ante"] += 1
        p, _ = found
        legal = obs.legal
        if _uniform(seed, obs.decision_id) < p:
            if legal.can_raise and (spot.kind == "open" or spot.behind_live) and legal.all_in_raise_to is not None:
                return raise_to(legal.all_in_raise_to)
            return call() if legal.call_amount else check()
        return check() if legal.check else fold()
