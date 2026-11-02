"""Decision-time lookup of the push/fold tables (decision-time safe).

:func:`preflop_spot` reads the current hand from an observation's public history and,
when the action so far fits the push/fold abstraction, returns the decision kind, the
hero's index in the subgame's preflop order and the start-of-hand snapshot.
:class:`PushFoldTable` maps a spot and the hero's hand class to the probability of the
aggressive action, taking the finest key level with enough support.

The abstraction: before the hero everyone folded (``open``), or exactly one player went
all-in and the rest folded (``call``), or one all-in and one call (``overcall``). Any
all-in after the first is a call, as in the solved tree, where a call commits the
caller's whole stack. Limps, non-all-in raises and postflop streets are out of scope.

Antes (documented approximation; the tables are solved without antes): the snapshot
holds the stacks left after the antes, and its depth unit is the M-equivalent big blind
``bb * (sb + bb + antes) / (sb + bb)``, so a stack that pays the same per orbit maps to
the same depth key. The dead ante money itself is not in the solved pots, which makes
the tables a little too tight with antes. A seat all-in on its ante leaves the
abstraction.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from strategy.endgame.classes import hand_classes
from strategy.endgame.keys import MAX_DEPTH_BB, Snapshot, keys_for
from strategy.endgame.table_state import SeatedSnapshot, seated_snapshot
from strategy.tables import store

MIN_COUNT = 3
"""A key level is used only when at least this many solved states support it."""


class TableMissing(LookupError):
    pass


@dataclass(frozen=True)
class Spot:
    kind: str
    hero: int
    shover: int | None
    caller: int | None
    seated: SeatedSnapshot
    risk: int
    """Chips at risk for the hero (the key's depth before bucketing)."""
    behind_live: bool
    """Some player after the hero can still act."""

    @property
    def snapshot(self) -> Snapshot:
        return self.seated.snapshot

    @property
    def risk_bb(self) -> float:
        return self.risk / self.snapshot.bb

    def keys(self) -> tuple[int, ...]:
        return keys_for(self.kind, self.snapshot, self.hero, self.shover, self.caller)


def _hand_events(obs):
    events = []
    for e in reversed(obs.action_history):
        if e.hand_id != obs.hand_id:
            break
        events.append(e)
        if e.type == "HandStarted":
            break
    events.reverse()
    return events if events and events[0].type == "HandStarted" else None


def seated_from_observation(obs, start: dict, antes: dict | None = None) -> SeatedSnapshot:
    pos = start["positions"]
    stacks = {int(s): int(c) for s, c in start["start_stacks"].items()}
    heads_up = bool(pos["heads_up"])
    small = start.get("small_blind_seat")
    sb, bb = int(start["small_blind"]), int(start["big_blind"])
    if antes:
        stacks = {s: c - antes.get(s, 0) for s, c in stacks.items()}
    seated = seated_snapshot(ring=sorted(obs.seats), start_stacks=stacks, button=pos["button"],
                             small_blind_seat=small if small in stacks else None, big_blind=pos["big_blind"],
                             heads_up=heads_up, blinds=(sb, bb), payouts=obs.payouts, hand_number=obs.hand_number,
                             hand_cap=obs.hand_cap)
    if not antes:
        return seated
    depth_bb = max(bb, round(bb * (sb + bb + sum(antes.values())) / (sb + bb)))
    return dataclasses.replace(seated, snapshot=dataclasses.replace(seated.snapshot, bb=depth_bb))


def preflop_spot(obs) -> Spot | None:
    """The push/fold spot the hero faces, or None when the hand left the abstraction."""
    if obs.street != "preflop":
        return None
    events = _hand_events(obs)
    if events is None:
        return None
    antes = {int(e.data["seat"]): int(e.data["amount"]) for e in events if e.type == "AntePosted"}
    if antes:
        start_stacks = {int(s): int(c) for s, c in events[0].data["start_stacks"].items()}
        if any(a >= start_stacks[s] for s, a in antes.items()):
            return None
    seated = seated_from_observation(obs, events[0].data, antes)
    snap = seated.snapshot
    seats = seated.seats
    if obs.hero_seat not in seats:
        return None
    index = {s: i for i, s in enumerate(seats)}
    committed = list(snap.posts)
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
    if acted != [p for p in range(hero) if snap.posts[p] < snap.stacks[p]]:
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
    behind_live = any(snap.posts[p] < snap.stacks[p] for p in range(hero + 1, snap.n))
    return Spot(kind, hero, shover, caller, seated, risk, behind_live)


class PushFoldTable:
    """One generated ``endgame-<mode>-<ruleset>`` table, checked against the variant's classes."""

    def __init__(self, provenance: dict, arrays: dict) -> None:
        from strategy.endgame.keys import KINDS

        ruleset_id = provenance["ruleset_id"]
        hc = hand_classes(ruleset_id)
        if provenance.get("variant_hash") != hc.variant_hash or tuple(arrays["class_keys"].tolist()) != hc.keys:
            raise store.TableError(f"{provenance['name']}: built for different hand classes than {ruleset_id}")
        self.provenance = provenance
        self.ruleset_id = ruleset_id
        self.mode = provenance["mode"]
        self._rows: dict[str, dict[int, int]] = {}
        self._probs: dict[str, list[bytes]] = {}
        self._count: dict[str, list[int]] = {}
        for kind in KINDS:
            keys = arrays[f"{kind}_keys"].tolist()
            self._rows[kind] = {k: r for r, k in enumerate(keys)}
            self._probs[kind] = [bytes(row) for row in arrays[f"{kind}_probs"]]
            self._count[kind] = arrays[f"{kind}_count"].tolist()

    def lookup(self, kind: str, keys: tuple[int, ...], cls: int, *, min_count: int = MIN_COUNT,
               levels: tuple[int, ...] | None = None) -> tuple[float, int] | None:
        """``(probability of the aggressive action, key level used)`` or None without support."""
        rows = self._rows[kind]
        for level, key in enumerate(keys):
            if levels is not None and level not in levels:
                continue
            r = rows.get(key)
            if r is not None and self._count[kind][r] >= min_count:
                return self._probs[kind][r][cls] / 255.0, level
        return None

    def rows(self, kind: str) -> int:
        return len(self._rows[kind])


@lru_cache(maxsize=None)
def load_table(mode: str, ruleset_id: str, directory: str | None = None) -> PushFoldTable:
    name = f"endgame-{mode}-{ruleset_id}"
    loaded = store.load_arrays(name, directory)
    if loaded is None:
        where = Path(directory) if directory else store.table_dir()
        raise TableMissing(f"{name}.npz not found in {where}; run `make tables-endgame`")
    return PushFoldTable(*loaded)


def in_table_range(spot: Spot) -> bool:
    return spot.risk_bb <= MAX_DEPTH_BB
