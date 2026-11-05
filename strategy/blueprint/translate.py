"""Off-tree bet translation for the blueprint, outside the shipped bot until it is adopted.

The blueprint's context reads the real price, so an unseen size lands in a price bucket.
The trained sizes sit at the top of their buckets, so a bet is read as the next trained
size up (a third-pot bet as a half-pot bet). ``translate="postflop"`` (or ``"all"``, which
includes preflop) instead reads a bet between two price edges as the lower bucket with the
pseudo-harmonic probability of the bet-to-pot fractions at those edges
(:func:`lower_price_probability`), drawn from a stream keyed by the table's policy seed
and the decision id; a bet at a trained size is read as before.

:class:`TranslatingBlueprintBot` is the blueprint bot with translation;
:class:`strategy.final.translated.TranslatingFinalBot` is the final bot with it, and
``strategy.final.package --override translate=...`` bundles that one. The default bundle
never imports this module, so its bytes and decisions do not depend on it.
"""

from __future__ import annotations

from strategy.blueprint import abstraction as A
from strategy.blueprint.policy import PolicyPack
from strategy.blueprint.runtime import BlueprintBot, BlueprintPolicy, HandReplay, _uniform

TRANSLATE = ("off", "postflop", "all")

# Bet-to-pot fraction ``x`` of a plain bet whose price ``x / (1 + 2x)`` is each price edge.
EDGE_FRACTIONS = tuple(num / (den - 2 * num) for num, den in A.PRICE_EDGES)
PRICE_STRIDE = A.R_DEPTH * A.R_ACTED * A.R_INIT * A.R_PRESSURE


def lower_price_probability(to_call: int, pot: int) -> float:
    """Probability of reading a bet as the price bucket below its own: the pseudo-harmonic
    weight ``(B - x)(1 + A) / ((B - A)(1 + x))`` of the edges ``A < x <= B`` around the
    bet's price-equivalent fraction ``x``. Zero at an edge (a trained size), in the
    nothing-to-call and smallest buckets, and above the top edge."""
    b = A.price_bucket(to_call, pot)
    if b < 2 or b > len(A.PRICE_EDGES):
        return 0.0
    x = to_call / (pot - to_call)
    lo, hi = EDGE_FRACTIONS[b - 2], EDGE_FRACTIONS[b - 1]
    if x >= hi:
        return 0.0
    return min(1.0, (hi - x) * (1.0 + lo) / ((hi - lo) * (1.0 + x)))


def check_mode(translate: str) -> str:
    if translate not in TRANSLATE:
        raise ValueError(f"translate is one of {TRANSLATE}")
    return translate


class TranslatingPolicy(BlueprintPolicy):
    """:class:`BlueprintPolicy` whose :meth:`features` translate the bet the hero faces.

    ``seed``: the policy seed of the table being played, set by the bot before each decision."""

    def __init__(self, ruleset_id: str, pack: PolicyPack | None = None, translate: str = "off") -> None:
        super().__init__(ruleset_id, pack)
        self.translate = check_mode(translate)
        self.seed = ""

    def plain_features(self, obs, rp: HandReplay | None = None):
        """The untranslated :meth:`BlueprintPolicy.features`."""
        return super().features(obs, rp)

    def features(self, obs, rp: HandReplay | None = None):
        f = super().features(obs, rp)
        if f is None or self.translate == "off" or (self.translate == "postflop" and obs.street == "preflop"):
            return f
        call_amount = obs.legal.call_amount or 0
        if not call_amount:
            return f
        stack = next(p.stack for p in obs.players if p.seat == obs.hero_seat)
        p_low = lower_price_probability(min(call_amount, stack), obs.pot_total)
        if p_low and _uniform("translate|" + self.seed, obs.decision_id) < p_low:
            return (f[0] - PRICE_STRIDE, *f[1:])
        return f


class Translating:
    """Mixin for a :class:`BlueprintBot` subclass: its policies are :class:`TranslatingPolicy`."""

    translate = "off"

    def policy(self, ruleset_id: str) -> TranslatingPolicy:
        p = self._policies.get(ruleset_id)
        if p is None:
            pack = PolicyPack(self._pack_path) if self._pack_path else None
            p = self._policies[ruleset_id] = TranslatingPolicy(ruleset_id, pack, self.translate)
        return p

    def decide(self, obs, state):
        if obs.ruleset_id in self.rulesets:
            self.policy(obs.ruleset_id).seed = (state or {}).get("seed", "")
        return super().decide(obs, state)


class TranslatingBlueprintBot(Translating, BlueprintBot):
    def __init__(self, leaf: str = "points", pack=None, translate: str = "off") -> None:
        self.translate = check_mode(translate)
        super().__init__(leaf=leaf, pack=pack)
