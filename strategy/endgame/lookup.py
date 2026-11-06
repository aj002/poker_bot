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
from array import array
from bisect import bisect_left
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

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

    def __init__(self, provenance: dict, arrays) -> None:
        """``arrays``: the table's arrays as a mapping, or an iterable of ``(name, array)`` pairs
        (:func:`~strategy.tables.store.stream_arrays`), each read once and not kept."""
        from strategy.endgame.keys import KINDS

        ruleset_id = provenance["ruleset_id"]
        hc = hand_classes(ruleset_id)
        if provenance.get("variant_hash") != hc.variant_hash:
            raise store.TableError(f"{provenance['name']}: built for different hand classes than {ruleset_id}")
        self.provenance = provenance
        self.ruleset_id = ruleset_id
        self.mode = provenance["mode"]
        # Compact arrays instead of dicts and per-row bytes objects: sorted keys searched with
        # bisect, one shared copy of each distinct probability row, and a row index per key.
        self._keys: dict[str, array] = {}
        self._row: dict[str, array] = {}
        self._probs: dict[str, memoryview] = {}
        self._count: dict[str, array] = {}
        self._width = len(hc.keys)
        late: dict[str, np.ndarray] = {}
        class_keys = None
        for name, a in (arrays.items() if isinstance(arrays, Mapping) else arrays):
            if name == "class_keys":
                class_keys = a
                continue
            kind, _, field = name.rpartition("_")
            if kind not in KINDS:
                continue
            if field == "keys":
                keys = np.ascontiguousarray(a, dtype=np.int64)
                if keys.size > 1 and not bool((keys[1:] > keys[:-1]).all()):
                    raise store.TableError(f"{provenance['name']}: {kind} keys are not strictly increasing")
                self._keys[kind] = _typed("q", keys)
            elif field == "count":
                self._count[kind] = _typed("i", np.ascontiguousarray(a, dtype=np.int32))
            elif field == "probs":
                late[kind] = a
            else:
                continue
            if kind in late and kind in self._keys:
                self._pack(kind, late.pop(kind))
        if class_keys is None or tuple(class_keys.tolist()) != hc.keys:
            raise store.TableError(f"{provenance['name']}: built for different hand classes than {ruleset_id}")
        if late or any(k not in self._keys or k not in self._count or k not in self._probs for k in KINDS):
            raise store.TableError(f"{provenance['name']}: incomplete table")

    def _pack(self, kind: str, probs: np.ndarray) -> None:
        probs = np.ascontiguousarray(probs, dtype=np.uint8)
        if probs.shape != (len(self._keys[kind]), self._width):
            raise store.TableError(f"{self.provenance['name']}: {kind} probabilities have shape {probs.shape}")
        distinct, inverse = _distinct_rows(probs)
        self._row[kind] = _typed("i", inverse)
        self._probs[kind] = memoryview(np.ascontiguousarray(distinct).reshape(-1))

    def lookup(self, kind: str, keys: tuple[int, ...], cls: int, *, min_count: int = MIN_COUNT,
               levels: tuple[int, ...] | None = None) -> tuple[float, int] | None:
        """``(probability of the aggressive action, key level used)`` or None without support."""
        table = self._keys[kind]
        n = len(table)
        counts = self._count[kind]
        for level, key in enumerate(keys):
            if levels is not None and level not in levels:
                continue
            r = bisect_left(table, key)
            if r < n and table[r] == key and counts[r] >= min_count:
                return self._probs[kind][self._row[kind][r] * self._width + cls] / 255.0, level
        return None

    def rows(self, kind: str) -> int:
        return len(self._keys[kind])


_CHUNK = 64


def _distinct_rows(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(distinct rows, int32 index of each row into them)``, found by a position-weighted 64-bit
    hash in small chunks (``np.unique(axis=0)`` would transiently hold several copies of the table).
    The grouping is verified row by row; on a hash collision every row is kept as it is."""
    n, width = rows.shape
    x = np.arange(1, width + 1, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    weights = (x ^ (x >> np.uint64(31))) | np.uint64(1)
    hashes = np.empty(n, dtype=np.uint64)
    for s in range(0, n, _CHUNK):
        hashes[s:s + _CHUNK] = (rows[s:s + _CHUNK].astype(np.uint64) * weights).sum(axis=1, dtype=np.uint64)
    _, first, inverse = np.unique(hashes, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1).astype(np.int32)
    distinct = rows[first]
    if all(np.array_equal(distinct[inverse[s:s + _CHUNK]], rows[s:s + _CHUNK]) for s in range(0, n, _CHUNK)):
        return distinct, inverse
    return rows, np.arange(n, dtype=np.int32)


def _typed(code: str, values) -> array:
    out = array(code)
    out.frombytes(values.tobytes())
    return out


@lru_cache(maxsize=None)
def load_table(mode: str, ruleset_id: str, directory: str | None = None) -> PushFoldTable:
    name = f"endgame-{mode}-{ruleset_id}"
    streamed = store.stream_arrays(name, directory)
    if streamed is None:
        where = Path(directory) if directory else store.table_dir()
        raise TableMissing(f"{name}.npz not found in {where}; run `make tables-endgame`")
    return PushFoldTable(*streamed)


def in_table_range(spot: Spot) -> bool:
    return spot.risk_bb <= MAX_DEPTH_BB
