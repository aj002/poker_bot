"""The population bot: one sampled family member behind the local SDK contract.

A :class:`PopulationBot` holds a frozen member spec (family id and parameter vector). At
each decision it parses the observation with the style engine's :func:`parse`, swaps in
the ruleset's card adapter (:mod:`strategy.population.cards`), attaches the member's
style knobs to the spot, and asks its family to decide: family hooks first (squeeze,
trap, texture, ICM, ...), then the anchor style's rules from
:mod:`arena.agents.styles.engine`, then the family's post-processing of sizes. The
result is clamped to the legal-action descriptor, so a population bot never submits an
illegal action, on any ruleset.

Hidden labels: the bot's ``name`` is the opaque member id, its actions carry no note,
and the family and regime live only in the bot object and its explicit state, neither of
which any other seat or learner observation can read.

Randomness: the seat's ``policy_seed`` drives a :class:`~arena.agents.styles.engine.Chance`
stream counted in state, plus per-hand hashes (:func:`~strategy.population.params.unit_hash`)
for draws that must not depend on how many decisions came before (gear, card noise,
regime switch hand). A seeded table therefore replays exactly and restores after a crash.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from arena.agents.sdk import ExplicitStateBot, clamp_raise_to
from arena.agents.styles.engine import Chance, Decision, Spot, parse
from arena.domain.models import REVEAL_ENV, REVEAL_RULESET_ID, RULESET_IDS, Action
from strategy.population import cards
from strategy.population.params import unit_hash

STATE_VERSION = 1


@dataclass
class History:
    """Facts about the current hand from the public action history that the spot lacks."""

    callers: int = 0
    """Preflop calls after the latest raise (excluding the hero)."""
    hero_limped: bool = False
    hero_squeezed: bool = False
    street_callers: int = 0
    """Calls of the current bet on this street (excluding the hero)."""
    flop_raises: int = 0
    hero_bet_flop: bool = False
    hero_bet_turn: bool = False
    opener: int | None = None
    """The seat that made the first preflop raise."""


def scan(obs: Any, hero: int) -> History:
    h = History()
    pre_raises = 0
    street_raised = False
    for e in obs.action_history:
        if e.type != "ActionApplied" or e.data.get("auto"):
            continue
        d = e.data
        seat, street, kind = d["seat"], d["street"], d["action"]["type"]
        if street == "preflop":
            if kind == "raise_to":
                if seat == hero and pre_raises == 1 and h.callers >= 1:
                    h.hero_squeezed = True
                if pre_raises == 0:
                    h.opener = seat
                pre_raises += 1
                h.callers = 0
            elif kind == "call":
                if seat == hero and pre_raises == 0:
                    h.hero_limped = True
                elif seat != hero and pre_raises > 0:
                    h.callers += 1
        if street == "flop" and kind == "raise_to":
            h.flop_raises += 1
            h.hero_bet_flop |= seat == hero
        if street == "turn" and kind == "raise_to" and seat == hero:
            h.hero_bet_turn = True
        if street == obs.street:
            if kind == "raise_to":
                street_raised = True
                h.street_callers = 0
            elif kind == "call" and street_raised and seat != hero:
                h.street_callers += 1
    return h


@dataclass
class Ctx:
    """Everything a family hook needs for one decision."""

    s: Spot
    ch: Chance
    obs: Any
    p: Mapping[str, float]
    state: dict[str, Any]
    ruleset: str
    top: float
    """Top fraction of the variant's starting hands (0 best), after any card noise."""
    x: History
    _strength: tuple[str, int, tuple[int, ...]] | None = field(default=None, repr=False)

    @property
    def bucket(self) -> str:
        return self.strength[0]

    @property
    def outs(self) -> int:
        return self.strength[1]

    @property
    def strength(self) -> tuple[str, int, tuple[int, ...]]:
        if self._strength is None:
            self._strength = self.s.strength()
        return self._strength

    @property
    def texture(self) -> str:
        return cards.texture(self.s.board)

    def uniform(self) -> float:
        """A uniform draw from the seat's counted stream."""
        self.ch.n += 1
        return unit_hash(self.ch.seed, "u", self.ch.n)

    def hand_uniform(self, tag: str) -> float:
        """A uniform draw fixed for this hand (and tag), whatever was decided before."""
        return unit_hash(self.state["seed"], tag, self.obs.hand_id)

    @property
    def live_opponents(self) -> list[Any]:
        return [p for p in self.obs.players if p.seat != self.s.hero and p.status in ("active", "all_in")]

    @property
    def allin_opponents(self) -> int:
        return sum(1 for p in self.live_opponents if p.status == "all_in")

    @property
    def active_opponents(self) -> int:
        return sum(1 for p in self.live_opponents if p.status == "active")


def equity_estimate(c: Ctx) -> float:
    """A rough chance to win a showdown: preflop from the Hold'em table, post-flop from the bucket."""
    if c.s.street == "preflop":
        return cards.nlhe_equity(cards.equivalent_class(c.top))
    base = {"monster": 0.85, "strong": 0.68, "medium": 0.5, "weak": 0.25}[c.bucket]
    draw = 0.0 if c.s.street == "river" else min(0.3, c.outs * (0.02 if c.s.street == "turn" else 0.035))
    return min(0.95, base + draw * (1 - base))


# --------------------------------------------------------------------- legality


def legal_or_safe(action: Action, legal: Any) -> Action:
    """``action`` when legal; otherwise the closest legal raise, a check, a call or a fold."""
    if legal.allows(action):
        return action
    if action.type == "raise_to" and action.amount is not None:
        a = clamp_raise_to(legal, action.amount)
        if a is not None:
            return a
    if legal.check:
        return Action(type="check")
    if action.type in ("call", "raise_to") and legal.call_amount is not None:
        return Action(type="call")
    return Action(type="fold")


# --------------------------------------------------------------------- the bot


class PopulationBot(ExplicitStateBot):
    """One frozen population member. Build with :meth:`from_spec` or ``Member.make_bot()``."""

    rulesets = tuple(RULESET_IDS)  # type: ignore[assignment]

    def __init__(self, spec: Mapping[str, Any]) -> None:
        from strategy.population.families import ALL_FAMILIES

        if os.environ.get(REVEAL_ENV):
            # The reveal variant is readable only through the environment, which a
            # sandboxed member never gets; in-process members play it via the nearest
            # practice card adapter (cards.card_ruleset).
            self.rulesets = (*RULESET_IDS, REVEAL_RULESET_ID)
        self.spec = dict(spec)
        self.family = ALL_FAMILIES[self.spec["family"]]
        self.params: dict[str, float] = dict(self.spec["params"])
        self.name = str(self.spec["member_id"])
        self._knobs: dict[tuple[bool, bool], tuple[dict[str, float], dict[str, Any]]] = {}

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> PopulationBot:
        return cls(spec)

    def active(self, switched: bool, hot: bool) -> tuple[dict[str, float], dict[str, Any]]:
        """(parameters, style knobs) for the regime and the family's hot mode."""
        key = (switched, hot)
        if key not in self._knobs:
            p = dict(self.params)
            if switched:
                p = self.family.prior.shifted(p, self.family.regime)
            if hot and self.family.hot_shift:
                p = self.family.prior.shifted(p, self.family.hot_shift)
            self._knobs[key] = (p, self.family.knobs(p))
        return self._knobs[key]

    def initial_state(self, table: Any) -> dict[str, Any]:
        seed = table.policy_seed
        switch = 0
        if self.family.regime and unit_hash(seed, "regime") < self.family.regime_p:
            switch = 30 + int(unit_hash(seed, "regime-hand") * 140)
        for key in ((False, False), (bool(switch), False)):
            self.active(*key)
        if self.family.hot is not None:
            self.active(False, True)
            self.active(bool(switch), True)
        state: dict[str, Any] = {"v": STATE_VERSION, "seed": seed, "n": 0, "sw": switch}
        state.update(self.family.initial_state())
        return state

    def decide(self, observation: Any, state: Any) -> tuple[Action, dict[str, Any]]:
        new = dict(state)
        ch = Chance(new["seed"], new["n"])
        obs = observation
        ruleset = obs.ruleset_id
        if self.family.observe_fn is not None:
            self.family.observe_fn(obs, new, self.params)
        switched = bool(new["sw"]) and obs.hand_number >= new["sw"]
        hot = self.family.hot is not None and self.family.hot(obs, new, self.params)
        p, knobs = self.active(switched, hot)
        s = parse(obs)
        s.knobs = knobs
        top = cards.preflop_top(obs.hole_cards, ruleset)
        if self.family.card_noise:
            top, s.evaluator = self.family.card_noise(p, new["seed"], obs, top)
        elif ruleset != "practice_nlhe":
            s.evaluator = cards.strength_for(ruleset)
        if ruleset != "practice_nlhe" or self.family.card_noise:
            s.hand = cards.equivalent_class(top)
        c = Ctx(s=s, ch=ch, obs=obs, p=p, state=new, ruleset=ruleset, top=top, x=scan(obs, s.hero))
        decision: Decision | None = None
        for hook in self.family.hooks:
            decision = hook(c)
            if decision is not None:
                break
        if decision is None:
            decision = self.family.base(c)
        action = decision[0]
        for post in self.family.post:
            action = post(c, action)
        new["n"] = ch.n
        return legal_or_safe(action, obs.legal), new


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))
