"""Authoritative in-process table: hands, eliminations, levels, the cap and scoring.

This is the only place game transitions happen; the headless runner, tests and
(later) the API all drive a :class:`Table` through :meth:`Table.submit`.
Deadlines, persistence and the command queue are layered on top later.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

import arena
from arena.domain.cards import full_deck
from arena.domain.events import EVENT_SCHEMA_VERSION, EventEnvelope
from arena.domain.models import Action, LegalActions, TournamentConfig, level_index
from arena.domain.rng import Rng, SeededRng, SystemRng, shuffled
from arena.rules.base import HandEvent, HandResult, IllegalAction
from arena.rules.pokerkit_adapter import POKERKIT_VERSION, HandSpec, PokerKitHand
from arena.rules.profiles import RuleProfile, get_profile
from arena.table.positions import Positions, initial_positions, next_positions, positions_from_button
from arena.table.state_hash import decision_hash, hand_start_hash, hand_start_state
from arena.tournament.scoring import Placement, Standings

DeckSource = Callable[[int, RuleProfile], tuple[str, ...]]


class ActionRejected(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass
class HandRecord:
    hand_number: int
    hand_id: str
    level: int
    blinds: tuple[int, int]
    positions: Positions
    start_stacks: dict[int, int]
    deck: tuple[str, ...]
    events: list[HandEvent] = field(default_factory=list)
    result: HandResult | None = None
    eliminations: list[Placement] = field(default_factory=list)
    ante: int = 0
    ante_posting: str | None = None


@dataclass(frozen=True)
class Receipt:
    hand_id: str
    decision_id: str
    seat: int
    action: Action
    fallback_reason: str | None
    state_version: int


def rng_info(rng: Rng | None) -> dict[str, str | None]:
    """Private audit description of an environment RNG (never projected)."""
    if rng is None:
        return {"implementation": "explicit-deck-source", "seed": None, "stream": None}
    return {
        "implementation": rng.implementation,
        "seed": getattr(rng, "seed", None),
        "stream": getattr(rng, "stream", None),
    }


def seeded_deck_source(rng: Rng) -> DeckSource:
    def source(hand_number: int, profile: RuleProfile) -> tuple[str, ...]:
        return tuple(shuffled(full_deck(profile.ranks), rng))

    return source


_UNRECORDED_HAND_EVENTS = {"HandStarted", "DecisionRequested", "PlayerEliminated"}


def _hand_records(log: list[EventEnvelope]) -> list[HandRecord]:
    """Rebuild finished hands' records from their stored events (results are not re-derived)."""
    records: list[HandRecord] = []
    for e in log:
        if e.hand_id is None:
            continue
        if e.type == "HandStarted":
            d = e.data
            records.append(HandRecord(
                hand_number=d["hand_number"],
                hand_id=e.hand_id,
                level=d["level"],
                blinds=(d["small_blind"], d["big_blind"]),
                positions=Positions(**d["positions"]),
                start_stacks={int(s): v for s, v in d["start_stacks"].items()},
                deck=tuple(d["deck"]),
                ante=d.get("ante", 0),
                ante_posting=d.get("ante_posting"),
            ))
        elif e.type == "PlayerEliminated":
            d = e.data
            records[-1].eliminations.append(Placement(d["seat"], tuple(d["positions"]), d["points"], "elimination",
                                                      d["hand_number"]))
        elif e.type not in _UNRECORDED_HAND_EVENTS:
            records[-1].events.append(HandEvent(e.type, dict(e.data)))
    return records


class Table:
    def __init__(
        self,
        config: TournamentConfig,
        *,
        table_id: str = "t_local",
        env_rng: Rng | None = None,
        deck_source: DeckSource | None = None,
        initial_button: int | None = None,
        start_stacks: Mapping[int, int] | None = None,
        rng_record: Mapping[str, str | None] | None = None,
        on_hand_start: Callable[[Table], None] | None = None,
    ) -> None:
        self.on_hand_start = on_hand_start
        self.config = config
        self.config_hash = config.config_hash()
        self.table_id = table_id
        self.profile = get_profile(config.ruleset_id)
        self.seat_ring: tuple[int, ...] = tuple(config.resolved_seats)
        self.env_rng: Rng = env_rng if env_rng is not None else SystemRng()
        self.deck_source = deck_source or seeded_deck_source(self.env_rng)
        if rng_record is None:
            rng_record = rng_info(self.env_rng if deck_source is None else None)
        if start_stacks is not None:
            if set(start_stacks) != set(self.seat_ring):
                raise ValueError("start_stacks must cover every seat")
            self.stacks = dict(start_stacks)
        else:
            self.stacks = {s: config.starting_stack for s in self.seat_ring}
        self.total_chips = sum(self.stacks.values())
        self.standings = Standings(config.resolved_payouts, self.seat_ring)
        self.hands: list[HandRecord] = []
        self.hand: PokerKitHand | None = None
        self.positions: Positions | None = None
        self.hand_number = 0
        self.state_version = 0
        self._decision_seq = 0
        self.decision_id: str | None = None
        self.receipts: list[Receipt] = []
        self.complete = False
        self._initial_button = initial_button
        self.log: list[EventEnvelope] = []
        self.hand_log_start = 0
        self._emit(
            None,
            "TableStarted",
            {
                "config": config.resolved(),
                "config_hash": self.config_hash,
                "seat_ring": list(self.seat_ring),
                "start_stacks": dict(self.stacks),
                "rng": dict(rng_record),
                "engine": {
                    "arena_version": arena.__version__,
                    "pokerkit_version": POKERKIT_VERSION,
                    "event_schema_version": EVENT_SCHEMA_VERSION,
                },
            },
        )
        self._start_next_hand()

    def _emit(self, hand_id: str | None, event_type: str, data: dict) -> None:
        self.log.append(EventEnvelope.make(self.table_id, hand_id, len(self.log), event_type, data))

    def _emit_hand_events(self, events: list[HandEvent]) -> None:
        hand_id = self.hands[-1].hand_id
        for event in events:
            self._emit(hand_id, event.type, event.data)

    # ------------------------------------------------------------------ views

    @property
    def live_seats(self) -> list[int]:
        return [s for s in self.seat_ring if self.stacks[s] > 0]

    @property
    def current(self) -> HandRecord | None:
        return self.hands[-1] if self.hands and self.hand is not None else None

    @property
    def actor_seat(self) -> int | None:
        if self.complete or self.hand is None:
            return None
        return self.hand.actor_seat

    def legal_actions(self) -> LegalActions:
        if self.hand is None or self.actor_seat is None:
            raise ActionRejected("no_decision", "no seat is to act")
        return self.hand.legal_actions()

    def points(self) -> dict[int, float]:
        if not self.complete:
            raise RuntimeError("table not complete")
        return self.standings.points()

    # ----------------------------------------------------------------- resume

    def resume_point(self) -> dict[str, Any]:
        """Private, JSON-compatible state at a hand boundary, before the next hand starts.

        Only meaningful from an ``on_hand_start`` callback. It holds the dealer RNG
        state (seed included), so it is as private as the event log.
        """
        getstate = getattr(self.env_rng, "getstate", None)
        return {
            "hand_number": self.hand_number + 1,
            "seq": len(self.log),
            "state_version": self.state_version,
            "decision_seq": self._decision_seq,
            "stacks": {str(s): v for s, v in self.stacks.items()},
            "positions": self.positions.as_dict() if self.positions is not None else None,
            "initial_button": self._initial_button,
            "placements": [
                {"seat": p.seat, "positions": list(p.positions), "points": float(p.points), "method": p.method,
                 "hand_number": p.hand_number}
                for p in self.standings.placements.values()
            ],
            "env_rng": getstate() if getstate is not None else None,
        }

    @classmethod
    def resume_at_hand(
        cls,
        config: TournamentConfig,
        point: Mapping[str, Any],
        log_prefix: list[EventEnvelope],
        *,
        env_rng: Rng,
        deck_source: DeckSource | None = None,
        on_hand_start: Callable[[Table], None] | None = None,
    ) -> Table:
        """Rebuild a table from a :meth:`resume_point` and the log before that hand, then deal it.

        The engine deals the hand itself, so the caller must compare the regenerated
        ``HandStarted`` (and everything after it) with the stored log; nothing here
        trusts the snapshot beyond what those hashes confirm.
        """
        if not log_prefix or log_prefix[0].type != "TableStarted":
            raise ValueError("log prefix must start with TableStarted")
        if point["seq"] != len(log_prefix):
            raise ValueError("resume point does not sit at the end of the log prefix")
        started = log_prefix[0].data
        if config.config_hash() != started["config_hash"]:
            raise ValueError("config does not match the stored config_hash")
        t = cls.__new__(cls)
        t.on_hand_start = on_hand_start
        t.config = config
        t.config_hash = config.config_hash()
        t.table_id = log_prefix[0].table_id
        t.profile = get_profile(config.ruleset_id)
        t.seat_ring = tuple(config.resolved_seats)
        t.env_rng = env_rng
        t.deck_source = deck_source or seeded_deck_source(env_rng)
        t.stacks = {int(s): int(v) for s, v in point["stacks"].items()}
        if set(t.stacks) != set(t.seat_ring):
            raise ValueError("resume point stacks must cover every seat")
        t.total_chips = sum(started["start_stacks"].values())
        if sum(t.stacks.values()) != t.total_chips:
            raise ValueError("resume point does not conserve table chips")
        t.standings = Standings(config.resolved_payouts, t.seat_ring)
        for p in point["placements"]:
            t.standings.placements[p["seat"]] = Placement(
                p["seat"], tuple(p["positions"]), float(p["points"]), p["method"], p["hand_number"]
            )
        t.hands = _hand_records(log_prefix)
        t.hand = None
        t.positions = Positions(**point["positions"]) if point["positions"] is not None else None
        t.hand_number = point["hand_number"] - 1
        t.state_version = point["state_version"]
        t._decision_seq = point["decision_seq"]
        t.decision_id = None
        t.receipts = []
        t.complete = False
        t._initial_button = point["initial_button"]
        t.log = list(log_prefix)
        t.hand_log_start = 0
        t._start_next_hand()
        return t

    # --------------------------------------------------------------- commands

    def submit(
        self,
        seat: int,
        action: Action | Mapping[str, Any],
        *,
        hand_id: str | None = None,
        decision_id: str | None = None,
        expected_state_version: int | None = None,
    ) -> Receipt:
        """Validate and apply one action atomically. Rejections never mutate state."""
        if self.complete or self.hand is None:
            raise ActionRejected("table_complete", "table has finished")
        record = self.hands[-1]
        if hand_id is not None and hand_id != record.hand_id:
            raise ActionRejected("wrong_hand", f"hand {hand_id} is not current")
        if decision_id is not None and decision_id != self.decision_id:
            raise ActionRejected("stale_decision", f"decision {decision_id} is not current")
        if expected_state_version is not None and expected_state_version != self.state_version:
            raise ActionRejected("stale_version", "state version mismatch")
        if seat != self.actor_seat:
            raise ActionRejected("out_of_turn", f"seat {seat} is not to act")
        if not isinstance(action, Action):
            try:
                action = Action.model_validate(action, strict=True)
            except ValidationError as exc:
                raise ActionRejected("invalid_schema", str(exc.errors()[0]["msg"])) from None
        try:
            return self._apply(seat, action, None)
        except IllegalAction as exc:
            raise ActionRejected("illegal_action", str(exc)) from None

    def fallback_action(self) -> Action:
        legal = self.legal_actions()
        return Action(type="check") if legal.check else Action(type="fold")

    def apply_fallback(self, seat: int, reason: str, rejected: Any = None) -> Receipt:
        """Consume the current decision with check if legal, otherwise fold.

        ``reason`` is a short code (``timeout``, ``invalid_action``, ...). ``rejected``
        is the JSON-compatible rejected output, recorded privately.
        """
        if self.complete or self.hand is None:
            raise ActionRejected("table_complete", "table has finished")
        if seat != self.actor_seat:
            raise ActionRejected("out_of_turn", f"seat {seat} is not to act")
        return self._apply(seat, self.fallback_action(), reason, rejected)

    def _apply(self, seat: int, action: Action, fallback_reason: str | None, rejected: Any = None) -> Receipt:
        assert self.hand is not None and self.decision_id is not None
        record = self.hands[-1]
        decision_id = self.decision_id
        events = self.hand.apply(action)
        if fallback_reason is not None:
            fallback = HandEvent("ActionFallback", {"seat": seat, "reason": fallback_reason, "rejected": rejected})
            record.events.append(fallback)
            self._emit_hand_events([fallback])
        record.events.extend(events)
        self._emit_hand_events(events)
        self.state_version += 1
        receipt = Receipt(record.hand_id, decision_id, seat, action, fallback_reason, self.state_version)
        self.receipts.append(receipt)
        if self.hand.is_complete:
            self._settle_hand()
        else:
            self._new_decision()
        return receipt

    # ------------------------------------------------------------- hand loop

    def _new_decision(self) -> None:
        self._decision_seq += 1
        self.decision_id = f"d_{self._decision_seq:05d}"
        self._emit(
            self.hands[-1].hand_id,
            "DecisionRequested",
            {
                "decision_id": self.decision_id,
                "seat": self.actor_seat,
                "state_version": self.state_version,
                "state_hash": decision_hash(self),
            },
        )

    def _start_next_hand(self) -> None:
        while True:
            live = self.live_seats
            if len(live) < 2 or self.hand_number >= self.config.hand_cap:
                self._finalize()
                return
            if self.on_hand_start is not None:
                self.on_hand_start(self)
            self.hand_number += 1
            if self.positions is None:
                if self._initial_button is not None:
                    self.positions = positions_from_button(self.seat_ring, live, self._initial_button)
                else:
                    self.positions = initial_positions(self.seat_ring, live, self.env_rng)
            else:
                self.positions = next_positions(self.positions, self.seat_ring, live)
            pos = self.positions
            sb, bb = self.config.blinds_for_hand(self.hand_number)
            ante = self.config.ante_for_hand(self.hand_number)
            posting = self.config.ante_posting if ante else None
            deck = self.deck_source(self.hand_number, self.profile)
            start = {s: self.stacks[s] for s in live}
            spec = HandSpec(
                profile=self.profile,
                seat_ring=self.seat_ring,
                stacks=start,
                button=pos.button,
                small_blind_seat=pos.small_blind_seat(live),
                big_blind_seat=pos.big_blind,
                small_blind=sb,
                big_blind=bb,
                deck=deck,
                local_rules=self.config.local_rules,
                ante=ante,
                ante_posting=posting,
            )
            record = HandRecord(
                hand_number=self.hand_number,
                hand_id=f"h_{self.hand_number:03d}",
                level=level_index(self.hand_number, self.config.hands_per_level),
                blinds=(sb, bb),
                positions=pos,
                start_stacks=start,
                deck=deck,
                ante=ante,
                ante_posting=posting,
            )
            self.hands.append(record)
            start_state = hand_start_state(
                self, hand_number=self.hand_number, positions=pos.as_dict(), blinds=(sb, bb),
                start_stacks=start, deck=deck, ante=(ante, posting),
            )
            self.hand_log_start = len(self.log)
            started = {
                "hand_number": self.hand_number,
                "level": record.level,
                "small_blind": sb,
                "big_blind": bb,
                "positions": pos.as_dict(),
                "small_blind_seat": spec.small_blind_seat,
                "start_stacks": start,
                "deck": list(deck),
                "start_hash": hand_start_hash(start_state),
            }
            if ante:
                started.update(ante=ante, ante_posting=posting)
            self._emit(record.hand_id, "HandStarted", started)
            self.hand = PokerKitHand(spec)
            record.events.extend(self.hand.events)
            self._emit_hand_events(self.hand.events)
            self.state_version += 1
            if not self.hand.is_complete:
                self._new_decision()
                return
            self._record_result()

    def _record_result(self) -> None:
        assert self.hand is not None
        record = self.hands[-1]
        result = self.hand.result()
        record.result = result
        for seat, stack in result.final_stacks.items():
            self.stacks[seat] = stack
        if sum(self.stacks.values()) != self.total_chips:
            raise AssertionError("table chips not conserved")
        record.eliminations = self.standings.record_eliminations(
            self.hand_number, record.start_stacks, self.stacks
        )
        for p in record.eliminations:
            self._emit(
                record.hand_id,
                "PlayerEliminated",
                {"seat": p.seat, "hand_number": p.hand_number, "positions": list(p.positions), "points": float(p.points)},
            )
        self.hand = None
        self.decision_id = None

    def _settle_hand(self) -> None:
        self._record_result()
        self._start_next_hand()

    def _finalize(self) -> None:
        self.hand = None
        self.decision_id = None
        survivors = {s: self.stacks[s] for s in self.standings.remaining}
        self.final_placements = self.standings.finalize(self.hand_number, survivors)
        self.complete = True
        self.state_version += 1
        placements = sorted(self.standings.placements.values(), key=lambda p: (p.positions, p.seat))
        self._emit(
            None,
            "TableScored",
            {
                "placements": [
                    {
                        "seat": p.seat,
                        "positions": list(p.positions),
                        "points": float(p.points),
                        "method": p.method,
                        "hand_number": p.hand_number,
                    }
                    for p in placements
                ],
                "points": {s: float(v) for s, v in self.standings.points().items()},
            },
        )


def seeded_table(config: TournamentConfig, seed: int | str, **kwargs: Any) -> Table:
    """Benchmark/test table whose decks and first button come from a versioned seeded stream."""
    return Table(config, env_rng=SeededRng(seed, "env"), **kwargs)
