"""Off-tree bet translation for the blueprint, outside the shipped bot until it is adopted.

The blueprint's context reads the real price, so an unseen size lands in a price bucket.
The trained sizes sit at the top of their buckets, so a bet is read as the next trained
size up (a third-pot bet as a half-pot bet). ``translate="postflop"`` (or ``"all"``, which
includes preflop) instead reads a bet between two price edges as the lower bucket with the
pseudo-harmonic probability of the bet-to-pot fractions at those edges
(:func:`lower_price_probability`), drawn from a stream keyed by the table's policy seed
and the decision id; a bet at a trained size is read as before.

``translate="overbet"`` leaves bets up to a pot alone and reads a postflop overbet (above
the largest trained size, a pot, and below the all-in) as the pot-sized raise or the
all-in; ``"harmonic"`` also reads a bet between a half pot and a pot as one of those two
(:func:`size_map`); ``"shove"`` reads every overbet as the all-in and ``"pot"`` as the pot-sized raise. The key is then that of the mapped bet: price, depth and bet level
as if it had been made (:func:`read_as`), while the hero's actions stay the real ones.

:class:`TranslatingBlueprintBot` is the blueprint bot with translation;
:class:`strategy.final.translated.TranslatingFinalBot` is the final bot with it, and
``strategy.final.package --override translate=...`` bundles that one. The default bundle
never imports this module, so its bytes and decisions do not depend on it.
"""

from __future__ import annotations

from strategy.blueprint import abstraction as A
from strategy.blueprint.policy import PolicyPack
from strategy.blueprint.runtime import BlueprintBot, BlueprintPolicy, HandReplay, _uniform

TRANSLATE = ("off", "postflop", "all", "overbet", "harmonic", "shove", "pot")

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


def pseudo_harmonic(x: float, lo: float, hi: float) -> float:
    """Probability of reading a bet of ``x`` pots as the smaller of the sizes ``lo < hi``."""
    if x <= lo:
        return 1.0
    if x >= hi:
        return 0.0
    return (hi - x) * (1.0 + lo) / ((hi - lo) * (1.0 + x))


def last_raise(obs):
    """``(seat, level before, raise-to, pot after)`` of the current street's last raise, or None."""
    events = []
    for e in reversed(obs.action_history):
        if e.hand_id != obs.hand_id or e.type == "HandStarted":
            break
        if e.type == "ActionApplied":
            if e.data["street"] != obs.street:
                break
            events.append(e.data)
    after = 0
    for i, d in enumerate(events):
        if d["action"]["type"] == "raise_to":
            level = max((int(o["street_committed"]) for o in events[i + 1:]), default=0)
            return int(d["seat"]), level, int(d["street_committed"]), obs.pot_total - after
        after += int(d.get("added", 0))
    return None


def size_map(obs, u: float, between: bool = False, shove: bool = False, pot: bool = False):
    """``(raise-to, mapped raise-to)`` for the postflop bet the hero faces, or None to read it as it is.

    The bettor's fraction ``x`` of the pot (the tree's ``cur + x * (pot + to_call)``) is read
    as one of its two neighbouring trained sizes with the pseudo-harmonic weight, ``u`` picking
    the side: an overbet (above a pot, below the effective all-in) as the pot-sized raise or
    the all-in, and with ``between`` a bet between a half pot and a pot as one of those two.
    All-ins, bets that put the hero all-in and raises after the last sized raise are left as
    they are."""
    me = next((p for p in obs.players if p.seat == obs.hero_seat), None)
    call = obs.legal.call_amount or 0
    if me is None or obs.street == "preflop" or not call or call >= me.stack:
        return None
    lr = last_raise(obs)
    if lr is None:
        return None
    seat, level, t, pot_after = lr
    if t != obs.current_bet_to or seat == me.seat:
        return None
    bettor = next((p for p in obs.players if p.seat == seat), None)
    if bettor is None or bettor.status != "active" or bettor.stack <= 0:
        return None
    raises = sum(1 for e in obs.action_history if e.hand_id == obs.hand_id and e.type == "ActionApplied"
                 and e.data["street"] == obs.street and e.data["action"]["type"] == "raise_to")
    base = pot_after - t + level
    if raises > A.MAX_SIZED_RAISES or base <= 0:
        return None
    t_all = min(t + bettor.stack, me.street_committed + me.stack)
    x, x_all = (t - level) / base, (t_all - level) / base
    if 1.0 < x < x_all:
        if shove:
            return t, t_all
        if pot:
            return t, level + base
        lo, hi = 1.0, x_all
    elif between and 0.5 < x < min(1.0, x_all):
        lo, hi = 0.5, min(1.0, x_all)
    else:
        return None
    small = level + round(lo * base)
    big = t_all if hi == x_all else level + round(hi * base)
    return t, (small if u < pseudo_harmonic(x, lo, hi) else big)


def read_as(obs, t: int, mapped: int):
    """The observation as the hero would see it had every player at bet level ``t`` put in ``mapped``."""
    delta = mapped - t
    players, pot = [], obs.pot_total
    for p in obs.players:
        if p.seat != obs.hero_seat and p.status not in ("folded", "eliminated") and p.street_committed == t:
            d = min(delta, p.stack)
            pot += d
            p = p.model_copy(update={"stack": p.stack - d, "street_committed": p.street_committed + d,
                                     "hand_committed": p.hand_committed + d})
        players.append(p)
    me = next(p for p in obs.players if p.seat == obs.hero_seat)
    legal = obs.legal.model_copy(update={"call_amount": mapped - me.street_committed})
    return obs.model_copy(update={"players": players, "pot_total": pot, "current_bet_to": mapped, "legal": legal})


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
        if f is not None and self.translate in ("overbet", "harmonic", "shove", "pot"):
            m = size_map(obs, _uniform("size-map|" + self.seed, obs.decision_id), self.translate == "harmonic",
                         self.translate == "shove", self.translate == "pot")
            g = super().features(read_as(obs, *m), rp) if m is not None else None
            return f if g is None else (g[0], *f[1:])
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
