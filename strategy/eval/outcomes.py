"""Per-table records for evaluation, extracted from the runner's private histories.

The benchmark runner returns points per entrant; finish positions, cap/ICM method
and decision context live in the table's event log. :func:`table_record` reads a
history once, keeps a compact summary and lets the caller delete the file. Every
record's points are checked against the runner's points, so the objective the
report uses is provably the table's own scoring.

Decision opportunities are counted from public quantities only (stacks, blinds,
street, the price to call, hand number), in the same slices the report uses.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from strategy.eval.luck import LuckSettings, table_luck
from strategy.objective.points import Finish, TerminalOutcome, chip_diagnostics, outcome_from_events

DEPTH_BUCKETS = ((0, 8, "<8bb"), (8, 15, "8-15bb"), (15, 25, "15-25bb"), (25, 50, "25-50bb"), (50, 10 ** 12, "50bb+"))
HORIZON_BUCKETS = ((0, 5, "0-5"), (6, 20, "6-20"), (21, 50, "21-50"), (51, 100, "51-100"), (101, 10 ** 9, "101+"))
ACTION_CLASSES = ("fold", "check", "call", "raise", "all_in", "fallback")


def depth_bucket(bb: float) -> str:
    return next(label for lo, hi, label in DEPTH_BUCKETS if lo <= bb < hi)


def horizon_bucket(hands_to_cap: int) -> str:
    return next(label for lo, hi, label in HORIZON_BUCKETS if lo <= hands_to_cap <= hi)


@dataclass
class EntrantTable:
    entrant: int
    seat: int
    points: float
    finish: Finish
    bust_level: int | None
    cap_icm_points: float
    chips: dict[str, Any]
    opportunities: Counter = field(default_factory=Counter)
    """``"<slice>=<bucket>|<action class>"`` -> count; ``"<slice>=<bucket>|*"`` is the denominator."""
    luck: dict[str, float] | None = None
    """Zero-mean luck terms (:mod:`strategy.eval.luck`), when the run asked for them."""


@dataclass
class TableRecord:
    table_id: str
    block: int
    rotation: int
    seats: dict[int, int]
    """seat -> entrant index"""
    valid: bool
    error: str | None
    hands: int
    points: dict[int, float]
    """entrant -> points"""
    ended_at_cap: bool = False
    survivors_at_end: int = 0
    entrants: dict[int, EntrantTable] = field(default_factory=dict)
    """Focus entrants only."""
    timeouts: int = 0
    """Deadline timeouts at every seat in the final attempt."""


def _level(hand_number: int, hands_per_level: int, levels: int) -> int:
    return min((hand_number - 1) // hands_per_level, levels - 1)


def opportunities(events: Iterable[Mapping[str, Any]], seat: int, hand_cap: int) -> Counter:
    """Count ``seat``'s decisions by public slice and action class."""
    out: Counter = Counter()
    start: dict[int, int] = {}
    committed: dict[int, int] = {}
    street_bets: dict[int, int] = {}
    folded: set[int] = set()
    street = "preflop"
    hand_number = level = bb = 0
    pending: dict[str, Any] | None = None
    for e in events:
        kind, d = e["type"], e["data"]
        if kind == "HandStarted":
            start = {int(s): int(v) for s, v in d["start_stacks"].items()}
            committed = dict.fromkeys(start, 0)
            street_bets = dict.fromkeys(start, 0)
            folded, street, pending = set(), "preflop", None
            hand_number, level, bb = int(d["hand_number"]), int(d["level"]), int(d["big_blind"])
        elif kind == "BlindPosted":
            s = int(d["seat"])
            committed[s] += int(d["amount"])
            street_bets[s] += int(d["amount"])
        elif kind == "AntePosted":
            committed[int(d["seat"])] += int(d["amount"])
        elif kind == "BoardDealt":
            street = d["street"]
            street_bets = dict.fromkeys(street_bets, 0)
        elif kind == "ExcessReturned":
            committed[int(d["seat"])] -= int(d["amount"])
        elif kind == "DecisionRequested" and int(d["seat"]) == seat and seat in start:
            stack = start[seat] - committed[seat]
            to_call = min(max(street_bets.values()) - street_bets[seat], stack)
            rivals = [o for o in start if o != seat and o not in folded]
            eff = min(start[seat], max((start[o] for o in rivals), default=start[seat])) / bb
            pending = {"stack": stack, "keys": [
                f"street={street}|facing={'bet' if to_call > 0 else 'none'}",
                *([f"preflop_depth={depth_bucket(eff)}"] if street == "preflop" else []),
                f"blind_level=L{level + 1}",
                f"hands_to_cap={horizon_bucket(hand_cap - hand_number)}",
                f"players_in_hand={len(rivals) + 1}",
            ], "fallback": False}
        elif kind == "ActionFallback" and int(d["seat"]) == seat and pending is not None:
            pending["fallback"] = True
        elif kind == "ActionApplied":
            s = int(d["seat"])
            if not d.get("auto") and s == seat and pending is not None:
                act = d["action"]["type"]
                cls = ("fallback" if pending["fallback"] else
                       "all_in" if act == "raise_to" and int(d["added"]) >= pending["stack"] else
                       "raise" if act == "raise_to" else act)
                for key in pending["keys"]:
                    out[f"{key}|*"] += 1
                    out[f"{key}|{cls}"] += 1
                pending = None
            committed[s] += int(d["added"])
            street_bets[s] = int(d["street_committed"])
            if d["action"]["type"] == "fold":
                folded.add(s)
    return out


def table_record(result: Any, history_path: Path | None, focus: Sequence[int] = (0,), *,
                 delete: bool = True, luck: LuckSettings | None = None) -> TableRecord:
    """Summarize one final table attempt. ``result`` is a runner ``TableResult``; with ``luck``
    the focus entrants also get their luck terms."""
    rec = TableRecord(result.table_id, result.block, result.rotation, dict(result.seats), result.valid,
                      result.error, result.hands, dict(result.points),
                      timeouts=sum(f.get("timeout", 0) for f in getattr(result, "fallbacks", {}).values()))
    if not result.valid:
        return rec
    if history_path is None or not history_path.exists():
        raise FileNotFoundError(f"history for valid table {result.table_id} is missing")
    events = json.loads(history_path.read_text(encoding="ascii"))["events"]
    if delete:
        history_path.unlink()
    outcome = outcome_from_events(events)
    seat_of = {e: s for s, e in result.seats.items()}
    for entrant, pts in result.points.items():
        if abs(outcome.points(seat_of[entrant]) - pts) > 1e-9:
            raise AssertionError(f"{result.table_id}: objective {outcome.points(seat_of[entrant])} != runner {pts}")
    config = next(e["data"]["config"] for e in events if e["type"] == "TableStarted")
    rec.ended_at_cap = outcome.ended_at_cap
    rec.survivors_at_end = len(outcome.survivors)
    for entrant in focus:
        seat = seat_of[entrant]
        rec.entrants[entrant] = _entrant_table(entrant, seat, outcome, events, config)
        if luck is not None:
            rec.entrants[entrant].luck = table_luck(events, seat, luck)["terms"]
    return rec


def _entrant_table(entrant: int, seat: int, outcome: TerminalOutcome, events: list[dict[str, Any]],
                   config: Mapping[str, Any]) -> EntrantTable:
    f = outcome.finishes[seat]
    bust = _level(f.hand_number, config["hands_per_level"], len(config["blind_levels"])) \
        if f.method == "elimination" else None
    return EntrantTable(entrant, seat, f.points, f, bust, outcome.cap_icm_points(seat), chip_diagnostics(outcome, seat),
                        opportunities(events, seat, int(config["hand_cap"])))
