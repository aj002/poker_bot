"""Strict configuration, action and legal-action schemas.

Chips are integers everywhere. JSON booleans, floats, NaN/infinity, negative and
oversized values are rejected rather than coerced.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_validator

RulesetId = Literal["practice_nlhe", "uncertainty_shortdeck", "uncertainty_plo4", "reveal_variant"]
RULESET_IDS: tuple[str, ...] = ("practice_nlhe", "uncertainty_shortdeck", "uncertainty_plo4")
"""The practice rulesets. Registries, sweeps and fingerprints iterate these."""
REVEAL_RULESET_ID = "reveal_variant"
"""The build-day slot: a variant described at run time by a spec file (arena.rules.reveal)."""
ALL_RULESET_IDS: tuple[str, ...] = (*RULESET_IDS, REVEAL_RULESET_ID)
AntePosting = Literal["classic", "big_blind"]
REVEAL_ENV = "ARENA_REVEAL_VARIANT"
"""Directory of the active reveal variant (``variant.json``); sandboxed bots never see it."""

MAX_STARTING_STACK = 1_000_000_000
MAX_SEATS = 8
MAX_TOTAL_CHIPS = MAX_STARTING_STACK * MAX_SEATS

ChipInt = StrictInt


def default_payouts(player_count: int) -> list[int]:
    if player_count == 2:
        return [65, 35]
    return ([50, 30, 20] + [0] * player_count)[:player_count]


def default_blind_levels() -> list[list[int]]:
    return [[10, 20], [15, 30], [25, 50], [50, 100], [75, 150], [100, 200], [150, 300], [200, 400]]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LocalRules(_Strict):
    """Local rule switches. Each one is documented in docs/assumptions.md."""

    short_bet_reopens_checker: StrictBool = False


class TournamentConfig(_Strict):
    ruleset_id: RulesetId = "practice_nlhe"
    initial_player_count: StrictInt = Field(8, ge=2, le=MAX_SEATS)
    seats: list[StrictInt] | None = None
    starting_stack: StrictInt = Field(2000, ge=1, le=MAX_STARTING_STACK)
    hand_cap: StrictInt = Field(200, ge=1)
    hands_per_level: StrictInt = Field(25, ge=1)
    blind_levels: list[list[StrictInt]] = Field(default_factory=default_blind_levels)
    ante: Literal[0] = 0
    """Legacy flat-ante switch, always 0. Antes are set per level by ``antes``."""
    antes: list[StrictInt] | None = None
    """Ante per blind level (parallel to ``blind_levels``); ``None`` means no antes at any level."""
    ante_posting: AntePosting | None = None
    """``classic``: every dealt seat posts the ante. ``big_blind``: the big blind posts it once for
    the table, after its blind. Required with ``antes``, forbidden without."""
    payout_points: list[StrictInt] | None = None
    bot_budget_ns: StrictInt = Field(1_000_000, ge=1)
    human_turn_ms: StrictInt = Field(20_000, ge=1)
    button_policy: Literal["dead_button"] = "dead_button"
    runout_count: Literal[1] = 1
    rake: Literal[0] = 0
    compatibility_status: Literal["practice_assumptions"] = "practice_assumptions"
    local_rules: LocalRules = Field(default_factory=LocalRules)
    deal: Literal["seat_keyed"] | None = None
    """Evaluation option. ``seat_keyed`` gives each physical seat the same deck slots every hand
    (hole card ``j`` of the ``i``-th ring seat from slot ``j * seats + i``, burns and board from
    the slots after every seat's), so who holds which cards does not depend on who has busted.
    A permutation fixed before the shuffle leaves the deal uniform. ``None``: the standard
    deal, one card at a time from the first seat after the button."""

    @model_validator(mode="after")
    def _validate(self) -> TournamentConfig:
        n = self.initial_player_count
        if self.seats is not None:
            if len(self.seats) != n:
                raise ValueError("seats must list exactly initial_player_count seat ids")
            if len(set(self.seats)) != n or any(not 0 <= s < MAX_SEATS for s in self.seats):
                raise ValueError("seat ids must be distinct integers in 0..7")
        if not self.blind_levels:
            raise ValueError("blind_levels must not be empty")
        prev_sb, prev_bb = 0, 0
        for level in self.blind_levels:
            if len(level) != 2:
                raise ValueError("each blind level is [small_blind, big_blind]")
            sb, bb = level
            if not 0 < sb <= bb <= MAX_STARTING_STACK:
                raise ValueError(f"invalid blind level {level}")
            if bb <= prev_bb or sb < prev_sb:
                raise ValueError("blind levels must increase")
            prev_sb, prev_bb = sb, bb
        if len(self.blind_levels) * self.hands_per_level < self.hand_cap:
            raise ValueError("blind schedule does not cover hand_cap")
        if (self.antes is None) != (self.ante_posting is None):
            raise ValueError("antes and ante_posting are set together")
        if self.antes is not None:
            if len(self.antes) != len(self.blind_levels):
                raise ValueError("antes must list one ante per blind level")
            if any(not 0 <= a <= MAX_STARTING_STACK for a in self.antes):
                raise ValueError("antes must be integers in 0..MAX_STARTING_STACK")
            if any(a > b for a, b in zip(self.antes, self.antes[1:])):
                raise ValueError("antes must not decrease")
            if not any(self.antes):
                raise ValueError("antes are all zero; leave antes and ante_posting unset")
        payouts = self.resolved_payouts
        if len(payouts) != n:
            raise ValueError("payout_points must have length initial_player_count")
        if any(p < 0 for p in payouts):
            raise ValueError("payout_points must be nonnegative")
        if any(a < b for a, b in zip(payouts, payouts[1:])):
            raise ValueError("payout_points must be nonincreasing")
        if sum(payouts) <= 0:
            raise ValueError("payout_points must have a positive total")
        return self

    @property
    def resolved_payouts(self) -> list[int]:
        if self.payout_points is None:
            return default_payouts(self.initial_player_count)
        return list(self.payout_points)

    @property
    def resolved_seats(self) -> list[int]:
        if self.seats is None:
            return list(range(self.initial_player_count))
        return sorted(self.seats)

    def resolved(self) -> dict:
        """The canonical config dict. Ante and deal fields appear only when set, so a config
        without them resolves (and hashes) exactly as it did before they existed."""
        data = self.model_dump()
        if self.antes is None:
            del data["antes"], data["ante_posting"]
        if self.deal is None:
            del data["deal"]
        data["payout_points"] = self.resolved_payouts
        data["seats"] = self.resolved_seats
        return data

    def config_hash(self) -> str:
        canonical = json.dumps(self.resolved(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def blinds_for_hand(self, hand_number: int) -> tuple[int, int]:
        sb, bb = self.blind_levels[level_index(hand_number, self.hands_per_level)]
        return sb, bb

    def ante_for_hand(self, hand_number: int) -> int:
        if self.antes is None:
            return 0
        return self.antes[level_index(hand_number, self.hands_per_level)]


def level_index(hand_number: int, hands_per_level: int) -> int:
    if hand_number < 1:
        raise ValueError("hand numbers are one-based")
    return (hand_number - 1) // hands_per_level


ActionType = Literal["fold", "check", "call", "raise_to"]


class Action(_Strict):
    """A canonical action. ``amount`` is the total street contribution for ``raise_to``."""

    type: ActionType
    amount: StrictInt | None = None

    @model_validator(mode="after")
    def _amount_rules(self) -> Action:
        if self.type == "raise_to":
            if self.amount is None:
                raise ValueError("raise_to requires an amount")
            if not 1 <= self.amount <= MAX_TOTAL_CHIPS:
                raise ValueError("raise_to amount out of range")
        elif self.amount is not None:
            raise ValueError(f"{self.type} does not take an amount; the server prices calls")
        return self


class ActionEnvelope(_Strict):
    """A client's action submission (spec section 6.2).

    The acting seat comes from the authenticated session or worker lease, never
    from this body. ``action_id`` is the client's idempotency key.
    """

    action_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
    table_id: str = Field(min_length=1, max_length=128)
    hand_id: str = Field(min_length=1, max_length=64)
    decision_id: str = Field(min_length=1, max_length=64)
    expected_state_version: StrictInt = Field(ge=0)
    action: Action


class LegalActions(_Strict):
    """Legal-action descriptor for the acting seat.

    Raise targets are street totals. Legal raise targets are the integers in
    ``[min_full_raise_to, max_raise_to]`` plus ``short_all_in_raise_to`` when set;
    the two need not be contiguous.
    """

    fold: bool
    check: bool
    call_amount: int | None
    can_raise: bool
    min_full_raise_to: int | None
    max_raise_to: int | None
    short_all_in_raise_to: int | None
    all_in_raise_to: int | None
    pot_raise_to: int | None
    raise_blocked_reason: str | None = None

    def allows(self, action: Action) -> bool:
        match action.type:
            case "fold":
                return self.fold
            case "check":
                return self.check
            case "call":
                return self.call_amount is not None
            case "raise_to":
                assert action.amount is not None
                if not self.can_raise:
                    return False
                if action.amount == self.short_all_in_raise_to:
                    return True
                return (
                    self.min_full_raise_to is not None
                    and self.max_raise_to is not None
                    and self.min_full_raise_to <= action.amount <= self.max_raise_to
                )
        return False
