"""Versioned table event schema (event log format version 1).

Every committed transition appends :class:`EventEnvelope` records to the table's
private, ordered event log. Each event type has a strict data model below; an
unknown type or a malformed payload is rejected when it is appended and when a
stored history is loaded.

The log is private: it contains the deck, every hole and burn card, the RNG seed,
and decision state hashes. Never serialize it to players or spectators directly.
:mod:`arena.table.projections` is the only path to a seat or public view, and it
works from explicit per-type allowlists.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import (BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler, StrictBool, StrictFloat, StrictInt,
                      StrictStr, model_serializer)

EVENT_SCHEMA_VERSION = 1

Street = Literal["preflop", "flop", "turn", "river"]


class _Data(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class RngInfo(_Data):
    implementation: StrictStr
    seed: StrictStr | None = None
    stream: StrictStr | None = None


class EngineInfo(_Data):
    arena_version: StrictStr
    pokerkit_version: StrictStr
    event_schema_version: StrictInt


class PositionsData(_Data):
    button: StrictInt
    small_blind: StrictInt
    big_blind: StrictInt
    heads_up: StrictBool


class ActionData(_Data):
    type: Literal["fold", "check", "call", "raise_to"]
    amount: StrictInt | None = None


class PlacementData(_Data):
    seat: StrictInt
    positions: list[StrictInt]
    points: StrictFloat
    method: Literal["elimination", "winner", "icm"]
    hand_number: StrictInt


class TableStarted(_Data):
    config: dict[str, Any]
    config_hash: StrictStr
    seat_ring: list[StrictInt]
    start_stacks: dict[int, StrictInt]
    rng: RngInfo
    engine: EngineInfo


class HandStarted(_Data):
    hand_number: StrictInt
    level: StrictInt
    small_blind: StrictInt
    big_blind: StrictInt
    positions: PositionsData
    small_blind_seat: StrictInt | None
    start_stacks: dict[int, StrictInt]
    deck: list[StrictStr]
    start_hash: StrictStr
    ante: StrictInt = 0
    """The level's nominal ante; present in the payload only when nonzero."""
    ante_posting: Literal["classic", "big_blind"] | None = None

    @model_serializer(mode="wrap")
    def _omit_absent_ante(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        return _without_zero_ante(handler(self))


def _without_zero_ante(data: dict[str, Any]) -> dict[str, Any]:
    """Ante-free payloads keep exactly the keys they had before antes existed."""
    if not data.get("ante"):
        data.pop("ante", None)
        data.pop("ante_posting", None)
    return data


class AntePosted(_Data):
    seat: StrictInt
    amount: StrictInt


class BlindPosted(_Data):
    seat: StrictInt
    amount: StrictInt


class HoleDealt(_Data):
    seat: StrictInt
    card: StrictStr


class CardBurned(_Data):
    card: StrictStr


class BoardDealt(_Data):
    street: Street
    cards: list[StrictStr]


class DecisionRequested(_Data):
    decision_id: StrictStr
    seat: StrictInt
    state_version: StrictInt
    state_hash: StrictStr


class ActionFallback(_Data):
    seat: StrictInt
    reason: StrictStr
    rejected: Any = None


class ActionApplied(_Data):
    seat: StrictInt
    street: Street
    action: ActionData
    added: StrictInt
    street_committed: StrictInt
    nominal_big_blind_call: StrictBool = False
    auto: StrictStr | None = None


class BetsCollected(_Data):
    bets: dict[int, StrictInt]


class ExcessReturned(_Data):
    seat: StrictInt
    amount: StrictInt


class CardsRevealed(_Data):
    seat: StrictInt
    cards: list[StrictStr]


class HandKilled(_Data):
    seat: StrictInt


class EngineChipsPushed(_Data):
    pot_index: StrictInt
    awards: dict[int, StrictInt]


class ChipsPulled(_Data):
    seat: StrictInt
    amount: StrictInt


class OddChipCorrection(_Data):
    transfers: dict[int, StrictInt]
    engine_awards: dict[int, StrictInt]


class PotAwarded(_Data):
    pot_index: StrictInt
    amount: StrictInt
    eligible: list[StrictInt]
    awards: dict[int, StrictInt]


class HandCompleted(_Data):
    final_stacks: dict[int, StrictInt]


class PlayerEliminated(_Data):
    seat: StrictInt
    hand_number: StrictInt
    positions: list[StrictInt]
    points: StrictFloat


class TableScored(_Data):
    placements: list[PlacementData]
    points: dict[int, StrictFloat]


EVENT_MODELS: dict[str, type[_Data]] = {
    cls.__name__: cls
    for cls in (
        TableStarted,
        HandStarted,
        AntePosted,
        BlindPosted,
        HoleDealt,
        CardBurned,
        BoardDealt,
        DecisionRequested,
        ActionFallback,
        ActionApplied,
        BetsCollected,
        ExcessReturned,
        CardsRevealed,
        HandKilled,
        EngineChipsPushed,
        ChipsPulled,
        OddChipCorrection,
        PotAwarded,
        HandCompleted,
        PlayerEliminated,
        TableScored,
    )
}


class UnknownEventType(ValueError):
    pass


def normalize_event_data(event_type: str, data: dict[str, Any], *, from_json: bool = False) -> dict[str, Any]:
    """Validate a payload against its type's model and return the canonical dict.

    ``from_json`` accepts the JSON form (string map keys) of a stored payload.
    """
    model = EVENT_MODELS.get(event_type)
    if model is None:
        raise UnknownEventType(f"unknown event type {event_type!r}")
    if from_json:
        parsed = model.model_validate_json(json.dumps(data))
    else:
        parsed = model.model_validate(data)
    return parsed.model_dump(mode="python")


class EventEnvelope(BaseModel):
    """One private, ordered event. ``data`` is already normalized for ``type``."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal[1] = EVENT_SCHEMA_VERSION
    table_id: StrictStr
    hand_id: StrictStr | None
    seq: StrictInt = Field(ge=0)
    type: StrictStr
    data: dict[str, Any]

    @classmethod
    def make(cls, table_id: str, hand_id: str | None, seq: int, event_type: str, data: dict[str, Any]) -> EventEnvelope:
        return cls(
            table_id=table_id,
            hand_id=hand_id,
            seq=seq,
            type=event_type,
            data=normalize_event_data(event_type, data),
        )

    @classmethod
    def from_json_obj(cls, obj: dict[str, Any]) -> EventEnvelope:
        if not isinstance(obj, dict):
            raise ValueError("event must be an object")
        if obj.get("schema_version") != EVENT_SCHEMA_VERSION:
            raise ValueError(f"unsupported event schema_version {obj.get('schema_version')!r}")
        envelope = cls.model_validate_json(json.dumps(obj))
        return envelope.model_copy(
            update={"data": normalize_event_data(envelope.type, envelope.data, from_json=True)}
        )

    def to_json_obj(self) -> dict[str, Any]:
        model = EVENT_MODELS[self.type]
        return {
            "schema_version": self.schema_version,
            "table_id": self.table_id,
            "hand_id": self.hand_id,
            "seq": self.seq,
            "type": self.type,
            "data": model.model_validate(self.data).model_dump(mode="json"),
        }
