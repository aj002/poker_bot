"""Rules adapter: variant + tournament structure -> rules hash, capabilities, canonical actions.

Decision-time safe: it reads only :class:`~arena.agents.sdk.TableInfo`, the
observation and :class:`~arena.agents.sdk.LegalActions`. The same adapter is what a
thin official-kit shim must produce from the revealed game, so the hash and the
action menu stay stable across simulators.

Hashes
    ``variant_hash``  the game alone (deck, deal, showdown, ranking, betting).
    ``rules_hash``    the variant plus everything that changes terminal points: seats,
                      stacks, blind schedule, hand cap, payouts, button and local rules.
                      Decision budgets and UI timers are execution settings, not rules.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from arena.agents.sdk import Action, LegalActions, Observation, TableInfo, raise_to
from strategy.rules.canonical import canonical_hash
from strategy.rules.reveal import get_variant
from strategy.rules.variants import VariantSpec

RULES_HASH_DOMAIN = "strategy-rules-v1"

ActionKind = Literal["fold", "check", "call", "raise_min", "raise_fraction", "raise_pot", "raise_all_in"]
_RAISE_PRIORITY = {"raise_fraction": 0, "raise_min": 1, "raise_pot": 2, "raise_all_in": 3}


@dataclass(frozen=True)
class TournamentStructure:
    seats: tuple[int, ...]
    starting_stack: int
    hand_cap: int
    hands_per_level: int
    blind_levels: tuple[tuple[int, int], ...]
    ante: int
    payouts: tuple[int, ...]
    button_policy: str
    runout_count: int
    rake: int
    local_rules: Mapping[str, Any]

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> TournamentStructure:
        """From a resolved config dict (``TableInfo.config`` or ``TournamentConfig.resolved()``)."""
        payouts = config.get("payout_points")
        seats = config.get("seats")
        if payouts is None or seats is None:
            raise ValueError("config must be resolved (payout_points and seats filled in)")
        return cls(
            seats=tuple(sorted(int(s) for s in seats)),
            starting_stack=int(config["starting_stack"]),
            hand_cap=int(config["hand_cap"]),
            hands_per_level=int(config["hands_per_level"]),
            blind_levels=tuple((int(sb), int(bb)) for sb, bb in config["blind_levels"]),
            ante=int(config.get("ante", 0)),
            payouts=tuple(int(p) for p in payouts),
            button_policy=str(config.get("button_policy", "dead_button")),
            runout_count=int(config.get("runout_count", 1)),
            rake=int(config.get("rake", 0)),
            local_rules=dict(config.get("local_rules", {})),
        )

    @property
    def player_count(self) -> int:
        return len(self.seats)

    def level(self, hand_number: int) -> int:
        if hand_number < 1:
            raise ValueError("hand numbers are one-based")
        return min((hand_number - 1) // self.hands_per_level, len(self.blind_levels) - 1)

    def blinds(self, hand_number: int) -> tuple[int, int]:
        return self.blind_levels[self.level(hand_number)]

    def hands_to_next_level(self, hand_number: int) -> int | None:
        """Hands after this one before the blinds rise, or None if they never rise before the cap."""
        nxt = (self.level(hand_number) + 1) * self.hands_per_level + 1
        if self.level(hand_number) + 1 >= len(self.blind_levels) or nxt > self.hand_cap:
            return None
        return nxt - hand_number - 1

    def hands_to_cap(self, hand_number: int) -> int:
        """Hands still to be dealt after this one before the cap ends the tournament."""
        return max(0, self.hand_cap - hand_number)

    def as_dict(self) -> dict[str, Any]:
        return {
            "seats": list(self.seats),
            "starting_stack": self.starting_stack,
            "hand_cap": self.hand_cap,
            "hands_per_level": self.hands_per_level,
            "blind_levels": [list(x) for x in self.blind_levels],
            "ante": self.ante,
            "payouts": list(self.payouts),
            "button_policy": self.button_policy,
            "runout_count": self.runout_count,
            "rake": self.rake,
            "local_rules": dict(sorted(self.local_rules.items())),
        }


def rules_hash(variant: VariantSpec, structure: TournamentStructure) -> str:
    return canonical_hash({"variant": variant.as_dict(), "structure": structure.as_dict()}, domain=RULES_HASH_DOMAIN)


@dataclass(frozen=True)
class CanonicalAction:
    """One entry of the variant-agnostic action menu. ``amount`` is a street total."""

    kind: ActionKind
    amount: int | None = None
    fraction: float | None = None

    @property
    def key(self) -> str:
        return f"raise_fraction:{self.fraction:g}" if self.kind == "raise_fraction" else self.kind

    def action(self) -> Action:
        if self.kind in ("fold", "check", "call"):
            return Action(type=self.kind)
        assert self.amount is not None
        return raise_to(self.amount)


def canonical_actions(legal: LegalActions, *, pot_total: int | None = None, current_bet_to: int | None = None,
                      fractions: Sequence[float] = ()) -> tuple[CanonicalAction, ...]:
    """Fold/check/call, min raise, pot raise (pot-limit or no-limit), all-in, plus pot-fraction raises.

    A pot-fraction raise goes to ``current_bet_to + f * (pot_total + to_call)``, the pot
    after calling; it is kept only when it is a legal full raise. When two raises land
    on the same amount one entry survives: all-in over pot over min over a fraction.
    Every entry satisfies ``legal.allows``.
    """
    menu: list[CanonicalAction] = []
    if legal.fold:
        menu.append(CanonicalAction("fold"))
    if legal.check:
        menu.append(CanonicalAction("check"))
    if legal.call_amount is not None:
        menu.append(CanonicalAction("call"))
    if not legal.can_raise:
        return tuple(menu)
    raises: list[CanonicalAction] = []
    lo, hi = legal.min_full_raise_to, legal.max_raise_to
    if lo is not None:
        raises.append(CanonicalAction("raise_min", lo))
    if legal.pot_raise_to is not None:
        raises.append(CanonicalAction("raise_pot", legal.pot_raise_to))
    all_in = legal.all_in_raise_to if legal.all_in_raise_to is not None else legal.short_all_in_raise_to
    if all_in is not None:
        raises.append(CanonicalAction("raise_all_in", all_in))
    if fractions and lo is not None and hi is not None:
        if pot_total is None or current_bet_to is None:
            raise ValueError("pot-fraction raises need pot_total and current_bet_to")
        after_call = pot_total + (legal.call_amount or 0)
        for f in sorted(set(fractions)):
            if f <= 0:
                raise ValueError("pot fractions must be positive")
            target = current_bet_to + int(f * after_call)
            if lo <= target <= hi:
                raises.append(CanonicalAction("raise_fraction", target, float(f)))
    best: dict[int, CanonicalAction] = {}
    for r in raises:
        assert r.amount is not None
        kept = best.get(r.amount)
        if kept is None or _RAISE_PRIORITY[r.kind] > _RAISE_PRIORITY[kept.kind]:
            best[r.amount] = r
    menu.extend(best[a] for a in sorted(best))
    return tuple(menu)


def default_action(legal: LegalActions) -> Action:
    """The arena's own fallback: check when free, otherwise fold. Always legal."""
    return Action(type="check") if legal.check else Action(type="fold")


class RulesAdapter:
    """The rules a strategy acts under, built from what a seated bot is told."""

    def __init__(self, variant: VariantSpec, structure: TournamentStructure) -> None:
        self.variant = variant
        self.structure = structure
        self.rules_hash = rules_hash(variant, structure)
        self.variant_hash = variant.variant_hash()

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | Any, variant: VariantSpec | None = None) -> RulesAdapter:
        """``config`` is a resolved config dict or anything with ``resolved()`` (a TournamentConfig)."""
        data = config.resolved() if hasattr(config, "resolved") else dict(config)
        spec = variant or get_variant(str(data["ruleset_id"]))
        if spec.ruleset_id != data["ruleset_id"]:
            raise ValueError(f"variant {spec.ruleset_id!r} does not match config ruleset {data['ruleset_id']!r}")
        return cls(spec, TournamentStructure.from_config(data))

    @classmethod
    def from_table_info(cls, info: TableInfo, variant: VariantSpec | None = None) -> RulesAdapter:
        return cls.from_config(info.config, variant)

    @property
    def capabilities(self) -> frozenset[str]:
        return self.variant.capabilities

    def actions(self, observation: Observation, fractions: Sequence[float] = ()) -> tuple[CanonicalAction, ...]:
        return canonical_actions(observation.legal, pot_total=observation.pot_total,
                                 current_bet_to=observation.current_bet_to, fractions=fractions)

    def describe(self) -> dict[str, Any]:
        return {
            "ruleset_id": self.variant.ruleset_id,
            "rules_hash": self.rules_hash,
            "variant_hash": self.variant_hash,
            "capabilities": sorted(self.capabilities),
            "variant": self.variant.as_dict(),
            "structure": self.structure.as_dict(),
        }
