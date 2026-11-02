"""Observation and view types shared by bots, the API and the browser (schema version 1).

These are the stable contracts documented in ``docs/observations-and-projections.md``
and pinned by the JSON Schemas in ``docs/schemas/``. Build them only through
:mod:`arena.table.projections`; never construct them from the private event log or
engine objects by hand.

Chips are integers. Seats are the table's physical seat numbers (0..7), which are
anonymous: they carry no account or artifact identity.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import (BaseModel, ConfigDict, SerializerFunctionWrapHandler, StrictBool, StrictInt, StrictStr,
                      model_serializer)

from arena.domain.models import LegalActions

OBSERVATION_SCHEMA_VERSION = 1

PlayerStatus = Literal["active", "folded", "all_in", "eliminated"]
Perspective = Literal["seat", "spectator"]


class _View(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class PlayerView(_View):
    seat: StrictInt
    stack: StrictInt
    street_committed: StrictInt
    hand_committed: StrictInt
    status: PlayerStatus
    shown_cards: list[StrictStr] | None
    """Cards this player revealed at showdown in the current hand; otherwise null."""


class PotView(_View):
    amount: StrictInt
    eligible: list[StrictInt]


def _without_zero_ante(data: dict[str, Any]) -> dict[str, Any]:
    """Ante fields are on the wire only when an ante is set, so ante-free views are unchanged."""
    if not data.get("ante"):
        data.pop("ante", None)
        data.pop("ante_posting", None)
    return data


class BlindsView(_View):
    small_blind: StrictInt
    big_blind: StrictInt
    ante: StrictInt = 0
    """Nominal ante this hand: per seat (``classic``) or the big blind's (``big_blind``)."""
    ante_posting: Literal["classic", "big_blind"] | None = None

    @model_serializer(mode="wrap")
    def _omit_absent_ante(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        return _without_zero_ante(handler(self))


class NextLevelView(_View):
    small_blind: StrictInt
    big_blind: StrictInt
    starts_at_hand: StrictInt
    ante: StrictInt = 0
    ante_posting: Literal["classic", "big_blind"] | None = None

    @model_serializer(mode="wrap")
    def _omit_absent_ante(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        return _without_zero_ante(handler(self))


class DecisionBudget(_View):
    bot_budget_ns: StrictInt
    human_turn_ms: StrictInt


class PublicEvent(_View):
    """One event as a given viewer may see it: only allowlisted fields of ``data``."""

    seq: StrictInt
    hand_id: StrictStr | None
    type: StrictStr
    data: dict[str, Any]


class TableView(_View):
    """Current table state from one seat's or a spectator's perspective."""

    schema_version: Literal[1] = OBSERVATION_SCHEMA_VERSION
    perspective: Perspective
    compatibility_status: Literal["practice_assumptions"] = "practice_assumptions"
    table_id: StrictStr
    ruleset_id: StrictStr
    config_hash: StrictStr
    state_version: StrictInt
    hero_seat: StrictInt | None
    hole_cards: list[StrictStr] | None
    seats: list[StrictInt]
    initial_player_count: StrictInt
    surviving_player_count: StrictInt
    hand_id: StrictStr | None
    hand_number: StrictInt
    decision_id: StrictStr | None
    button_seat: StrictInt | None
    small_blind_seat: StrictInt | None
    big_blind_seat: StrictInt | None
    street: Literal["preflop", "flop", "turn", "river"] | None
    actor_seat: StrictInt | None
    board: list[StrictStr]
    pot_total: StrictInt
    settled_pot: StrictInt
    current_bet_to: StrictInt
    pots: list[PotView]
    players: list[PlayerView]
    blinds: BlindsView | None
    next_level: NextLevelView | None
    payouts: list[StrictInt]
    hands_played: StrictInt
    hands_remaining: StrictInt
    hand_cap: StrictInt
    decision_budget: DecisionBudget
    legal: LegalActions | None
    action_history: list[PublicEvent]
    table_complete: StrictBool
    points: dict[int, float] | None


class Observation(TableView):
    """What the acting seat receives at a decision. Identical for humans and bots."""

    perspective: Literal["seat"] = "seat"
    hero_seat: StrictInt
    hole_cards: list[StrictStr]
    hand_id: StrictStr
    decision_id: StrictStr
    actor_seat: StrictInt
    street: Literal["preflop", "flop", "turn", "river"]
    legal: LegalActions


class HistoryView(_View):
    """A downloadable history for one viewer (seat, spectator or post-table reviewer)."""

    schema_version: Literal[1] = OBSERVATION_SCHEMA_VERSION
    perspective: Literal["seat", "spectator", "reviewer"]
    compatibility_status: Literal["practice_assumptions"] = "practice_assumptions"
    table_id: StrictStr
    viewer_seat: StrictInt | None
    table_complete: StrictBool
    events: list[PublicEvent]
