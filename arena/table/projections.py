"""Seat, spectator and post-table projections, built from explicit allowlists.

The private event log and the engine state hold every card. Nothing here copies
private state and then deletes fields: each view field is chosen explicitly, and
each event type has an explicit list of the data fields a viewer may see. An
event type without an entry fails closed with :class:`UnknownEventType`.

Viewers:

* ``Viewer.spectator()`` sees public information only.
* ``Viewer.for_seat(s)`` additionally sees seat ``s``'s own hole cards.
* ``Viewer.reviewer()`` sees the full private log, only once the table is complete
  (an authorized post-table review). Asking earlier raises :class:`ProjectionDenied`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from arena.domain.events import EVENT_MODELS, EventEnvelope, UnknownEventType
from arena.domain.models import level_index
from arena.domain.observations import (
    BlindsView,
    DecisionBudget,
    HistoryView,
    NextLevelView,
    Observation,
    PlayerView,
    PotView,
    PublicEvent,
    TableView,
)

if TYPE_CHECKING:
    from arena.table.service import HandRecord, Table


class ProjectionDenied(PermissionError):
    pass


@dataclass(frozen=True)
class Viewer:
    kind: Literal["seat", "spectator", "reviewer"]
    seat: int | None = None

    @classmethod
    def spectator(cls) -> Viewer:
        return cls("spectator")

    @classmethod
    def for_seat(cls, seat: int) -> Viewer:
        return cls("seat", seat)

    @classmethod
    def reviewer(cls) -> Viewer:
        return cls("reviewer")


# Data fields any viewer may see, per event type. ``None`` hides the whole event.
PUBLIC_FIELDS: dict[str, tuple[str, ...] | None] = {
    "TableStarted": ("config", "config_hash", "seat_ring", "start_stacks"),
    "HandStarted": ("hand_number", "level", "small_blind", "big_blind", "positions", "small_blind_seat", "start_stacks",
                    "ante", "ante_posting"),
    "AntePosted": ("seat", "amount"),
    "BlindPosted": ("seat", "amount"),
    "HoleDealt": ("seat",),
    "CardBurned": None,
    "BoardDealt": ("street", "cards"),
    "DecisionRequested": ("decision_id", "seat", "state_version"),
    "ActionFallback": ("seat", "reason"),
    "ActionApplied": ("seat", "street", "action", "added", "street_committed", "nominal_big_blind_call", "auto"),
    "BetsCollected": ("bets",),
    "ExcessReturned": ("seat", "amount"),
    "CardsRevealed": ("seat", "cards"),
    "HandKilled": None,
    "EngineChipsPushed": None,
    "ChipsPulled": None,
    "OddChipCorrection": None,
    "PotAwarded": ("pot_index", "amount", "eligible", "awards"),
    "HandCompleted": ("final_stacks",),
    "PlayerEliminated": ("seat", "hand_number", "positions", "points"),
    "TableScored": ("placements", "points"),
}

# Extra fields visible only to the seat named in the event's ``seat`` field.
OWNER_FIELDS: dict[str, tuple[str, ...]] = {
    "HoleDealt": ("seat", "card"),
}


def project_event(event: EventEnvelope, viewer: Viewer) -> PublicEvent | None:
    if event.type not in PUBLIC_FIELDS or event.type not in EVENT_MODELS:
        raise UnknownEventType(f"no projection rule for event type {event.type!r}")
    if viewer.kind == "reviewer":
        fields: tuple[str, ...] | None = tuple(event.data)
    elif viewer.kind == "seat" and event.type in OWNER_FIELDS and event.data.get("seat") == viewer.seat:
        fields = OWNER_FIELDS[event.type]
    else:
        fields = PUBLIC_FIELDS[event.type]
    if fields is None:
        return None
    # JSON form (string map keys), so the in-memory view equals its wire form.
    wire = EVENT_MODELS[event.type].model_validate(event.data).model_dump(mode="json")
    # Fields a payload omits when unset (HandStarted's ante) are skipped, not invented.
    return PublicEvent(seq=event.seq, hand_id=event.hand_id, type=event.type,
                       data={k: wire[k] for k in fields if k in wire})


def project_events(events: Iterable[EventEnvelope], viewer: Viewer) -> list[PublicEvent]:
    out = []
    for event in events:
        projected = project_event(event, viewer)
        if projected is not None:
            out.append(projected)
    return out


def history_view(events: list[EventEnvelope], viewer: Viewer, *, table_complete: bool) -> HistoryView:
    """Downloadable history for ``viewer``. The reviewer view requires a completed table."""
    if viewer.kind == "reviewer" and not table_complete:
        raise ProjectionDenied("the full private history is available only after the table is complete")
    if not events:
        raise ValueError("empty event log")
    return HistoryView(
        perspective=viewer.kind,
        table_id=events[0].table_id,
        viewer_seat=viewer.seat,
        table_complete=table_complete,
        events=project_events(events, viewer),
    )


def table_history_view(table: Table, viewer: Viewer) -> HistoryView:
    return history_view(table.log, viewer, table_complete=table.complete)


def display_pots(collected: dict[int, int], folded: Iterable[int],
                 dead: dict[int, int] | None = None) -> list[PotView]:
    """Pots formed by chips already collected, merged where the eligible sets match.

    Current-street bets are not in a pot yet (see ``players[].street_committed``).
    Settlement still awards one pot per contribution level; only odd-chip
    placement can differ from this merged display, and ``PotAwarded`` events are
    authoritative. ``dead`` (big-blind antes, by seat) is not a contribution level:
    it sits in the main pot.
    """
    folded = set(folded)
    if dead:
        collected = {s: c - dead.get(s, 0) for s, c in collected.items()}
    pots: list[PotView] = []
    previous = 0
    for level in sorted({c for c in collected.values() if c > 0}):
        contributors = sorted(s for s, c in collected.items() if c >= level)
        amount = (level - previous) * len(contributors)
        previous = level
        eligible = [s for s in contributors if s not in folded]
        if pots and (pots[-1].eligible == eligible or not eligible):
            pots[-1] = PotView(amount=pots[-1].amount + amount, eligible=pots[-1].eligible)
        else:
            pots.append(PotView(amount=amount, eligible=eligible))
    if dead:
        extra = sum(dead.values())
        if pots:
            pots[0] = PotView(amount=pots[0].amount + extra, eligible=pots[0].eligible)
        else:
            pots.append(PotView(amount=extra, eligible=sorted(s for s in collected if s not in folded)))
    return pots


def _next_level(table: Table) -> NextLevelView | None:
    config = table.config
    if table.complete or table.hand_number < 1:
        return None
    current = level_index(table.hand_number, config.hands_per_level)
    if current + 1 >= len(config.blind_levels):
        return None
    starts = (current + 1) * config.hands_per_level + 1
    if starts > config.hand_cap:
        return None
    sb, bb = config.blind_levels[current + 1]
    if config.antes is None:
        return NextLevelView(small_blind=sb, big_blind=bb, starts_at_hand=starts)
    return NextLevelView(small_blind=sb, big_blind=bb, starts_at_hand=starts, ante=config.antes[current + 1],
                         ante_posting=config.ante_posting)


def _blinds_view(record: HandRecord) -> BlindsView:
    if not record.ante:
        return BlindsView(small_blind=record.blinds[0], big_blind=record.blinds[1])
    return BlindsView(small_blind=record.blinds[0], big_blind=record.blinds[1], ante=record.ante,
                      ante_posting=record.ante_posting)


def table_view(table: Table, viewer: Viewer) -> TableView:
    """The current state as ``viewer`` may see it (seat or spectator)."""
    if viewer.kind == "reviewer":
        raise ProjectionDenied("reviewers use history_view on a completed table")
    if viewer.kind == "seat" and viewer.seat not in table.seat_ring:
        raise ProjectionDenied(f"seat {viewer.seat} is not at this table")
    hand = table.hand
    record = table.hands[-1] if table.hands else None
    in_hand = hand is not None and record is not None
    hero = viewer.seat if viewer.kind == "seat" else None

    players: list[PlayerView] = []
    if in_hand:
        assert hand is not None
        engine_players = {p.seat: p for p in hand.players()}
        for seat in table.seat_ring:
            p = engine_players.get(seat)
            if p is None:
                players.append(PlayerView(seat=seat, stack=table.stacks[seat], street_committed=0, hand_committed=0,
                                          status="eliminated", shown_cards=None))
                continue
            shown = hand.revealed.get(seat)
            players.append(PlayerView(seat=seat, stack=p.stack, street_committed=p.street_committed,
                                      hand_committed=p.hand_committed, status=p.status,
                                      shown_cards=list(shown) if shown is not None else None))
        collected = hand.collected()
        pots = display_pots(collected, hand.folded, hand.dead_money())
        board = list(hand.board)
        pot_total = hand.pot_total()
        settled = sum(collected.values())
        level = hand.current_level()
        street = hand.street
        hole = list(hand.hole_cards.get(hero, [])) if hero is not None else None
        if hero is not None and hero not in hand.hole_cards:
            hole = None
    else:
        for seat in table.seat_ring:
            stack = table.stacks[seat]
            players.append(PlayerView(seat=seat, stack=stack, street_committed=0, hand_committed=0,
                                      status="active" if stack > 0 else "eliminated", shown_cards=None))
        pots, board, pot_total, settled, level, street, hole = [], [], 0, 0, 0, None, None

    positions = table.positions if in_hand else None
    small_blind_seat = None
    if positions is not None and record is not None:
        small_blind_seat = positions.small_blind_seat(sorted(record.start_stacks))
    actor = table.actor_seat
    legal = hand.legal_actions() if in_hand and hero is not None and hero == actor else None
    current_events = []
    if in_hand:
        assert record is not None
        current_events = project_events(table.log[table.hand_log_start :], viewer)
    hands_played = len(table.hands) - (1 if in_hand else 0)

    return TableView(
        perspective=viewer.kind,
        table_id=table.table_id,
        ruleset_id=table.config.ruleset_id,
        config_hash=table.config_hash,
        state_version=table.state_version,
        hero_seat=hero,
        hole_cards=hole,
        seats=list(table.seat_ring),
        initial_player_count=table.config.initial_player_count,
        surviving_player_count=sum(1 for s in table.seat_ring if table.stacks[s] > 0),
        hand_id=record.hand_id if in_hand and record is not None else None,
        hand_number=table.hand_number,
        decision_id=table.decision_id if in_hand else None,
        button_seat=positions.button if positions is not None else None,
        small_blind_seat=small_blind_seat,
        big_blind_seat=positions.big_blind if positions is not None else None,
        street=street,
        actor_seat=actor,
        board=board,
        pot_total=pot_total,
        settled_pot=settled,
        current_bet_to=level,
        pots=pots,
        players=players,
        blinds=_blinds_view(record) if in_hand and record else None,
        next_level=_next_level(table),
        payouts=list(table.config.resolved_payouts),
        hands_played=hands_played,
        hands_remaining=table.config.hand_cap - hands_played,
        hand_cap=table.config.hand_cap,
        decision_budget=DecisionBudget(bot_budget_ns=table.config.bot_budget_ns,
                                       human_turn_ms=table.config.human_turn_ms),
        legal=legal,
        action_history=current_events,
        table_complete=table.complete,
        points={s: float(v) for s, v in table.points().items()} if table.complete else None,
    )


def observation(table: Table, seat: int) -> Observation:
    """The decision request for the acting ``seat``; denied for any other seat."""
    if table.complete or table.actor_seat is None:
        raise ProjectionDenied("no decision is pending")
    if seat != table.actor_seat:
        raise ProjectionDenied(f"seat {seat} is not to act")
    view = table_view(table, Viewer.for_seat(seat))
    return Observation(**dict(view))
