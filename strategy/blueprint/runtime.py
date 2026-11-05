"""The deployed blueprint bot: observation -> abstract key -> table lookup -> legal action.

Decision-time path only: no numba, no solver, no search. Per decision it replays the
current hand's public events (bounded by the hand's length), computes the card bucket
(:mod:`strategy.blueprint.cards`) and the context (:func:`abstraction.context_key`, the
function the trainer compiled), looks the key up in the :class:`~strategy.blueprint.policy.PolicyPack`
with back-off, masks the abstract actions with the real legal actions and samples one
with a stream keyed by the table's policy seed and the decision id.

Layers, in order:

1. Stage 3 push/fold tables for the short-stack preflop spots they cover
   (:class:`strategy.endgame.bot.PushFoldBot`): point-aware (``leaf="points"``), chip-EV
   (``"chips"``), confidence-capped population responses (``"capped"``) or none;
2. the blueprint;
3. on a miss at every back-off level, the endgame bots' base policy.

Off-tree bets need no translation table: the context reads the real price, pot, stacks
and raise count, so an unseen size lands in a price and SPR bucket. The trained sizes sit
at the top of their price buckets, so by default a bet is read as the next trained size
up (a third-pot bet as a half-pot bet). ``translate`` (off by default) instead reads a
bet between two price edges as the lower bucket with the pseudo-harmonic probability of
the bet-to-pot fractions at those edges (:func:`lower_price_probability`); a bet at a
trained size is read as before. The hero's own sized raises are clamped to the nearest
legal raise.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path

from arena.agents.sdk import ExplicitStateBot, Observation, call, check, clamp_raise_to, fold, raise_to, with_note
from strategy.blueprint import abstraction as A
from strategy.blueprint.cards import CardAbstraction
from strategy.blueprint.policy import PolicyPack
from strategy.endgame.bot import BasePolicy, PushFoldBot
from strategy.endgame.icm import pressure
from strategy.features.cards import capabilities, card_ints
from strategy.tables import store

RULESETS = ("practice_nlhe",)
LEAVES = ("points", "chips", "capped", "none")
TRANSLATE = ("off", "postflop", "all")
STREETS = ("preflop", "flop", "turn", "river")
WARM_DIR = Path(__file__).resolve().parent / "data"
WARM_PASSES = 3


def warm_observations(ruleset_id: str) -> list[str]:
    """Recorded observations (JSON) replayed through :meth:`BlueprintBot.decide` at load."""
    path = WARM_DIR / f"warm-{ruleset_id}.json.gz"
    return json.loads(gzip.decompress(path.read_bytes())) if path.exists() else []


def pack_path(ruleset_id: str) -> Path:
    override = os.environ.get("BLUEPRINT_PACK")
    if override:
        return Path(override)
    return store.table_dir() / f"blueprint-{ruleset_id}.npz"


def _uniform(seed: str, decision_id: str) -> float:
    h = hashlib.sha256(f"blueprint-bot-v1|{seed}|{decision_id}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2.0 ** 64


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


class HandReplay:
    """Per-street betting facts of the current hand from its public events.

    ``HandReplay(obs)`` replays the observation's events. A caller that already walks the
    events can build the same facts incrementally: :meth:`begin` on ``HandStarted``,
    :meth:`feed` for every later event of the hand, then :meth:`at` for the decision's
    street (which leaves the incremental replay untouched)."""

    __slots__ = ("street", "cur", "last_full", "raises", "aggr", "prev_aggr", "acted_round", "acted_street",
                 "start", "bb", "ok")

    def __init__(self, obs=None) -> None:
        self.ok = False
        if obs is None:
            return
        events = []
        for e in reversed(obs.action_history):
            if e.hand_id != obs.hand_id:
                break
            events.append(e)
            if e.type == "HandStarted":
                break
        events.reverse()
        if not events or events[0].type != "HandStarted":
            return
        self.begin(events[0].data)
        for e in events[1:]:
            self.feed(e.type, e.data)
        s = STREETS.index(obs.street)
        if s != self.street:
            self._new_street(s)
        self.ok = True

    def begin(self, d, start: dict[int, int] | None = None) -> None:
        """``start``: the hand's ``{seat: start stack}`` if the caller already built it (kept as is)."""
        self.start = {int(s): int(c) for s, c in d["start_stacks"].items()} if start is None else start
        self.bb = int(d["big_blind"])
        self.street = 0
        self.cur = 0
        self.last_full = self.bb
        self.raises = 0
        self.aggr = -1
        self.prev_aggr = -1
        self.acted_round: set[int] = set()
        self.acted_street: set[int] = set()

    def feed(self, t: str, d) -> None:
        if t == "BlindPosted":
            self.cur = max(self.cur, int(d["amount"]))
        elif t == "ActionApplied":
            s = STREETS.index(d["street"])
            if s != self.street:
                self._new_street(s)
            seat = int(d["seat"])
            self.acted_round.add(seat)
            self.acted_street.add(seat)
            if d["action"]["type"] == "raise_to":
                new = int(d["street_committed"])
                if new - self.cur >= self.last_full:
                    self.last_full = new - self.cur
                    self.acted_round = {seat}
                self.cur = max(self.cur, new)
                self.raises += 1
                self.aggr = seat

    def at(self, street: str) -> HandReplay:
        """The replay as :class:`HandReplay` (obs) would leave it for a decision on ``street``."""
        out = HandReplay()
        out.street, out.cur, out.last_full, out.raises = self.street, self.cur, self.last_full, self.raises
        out.aggr, out.prev_aggr, out.acted_round, out.acted_street = (self.aggr, self.prev_aggr, self.acted_round,
                                                                      self.acted_street)
        out.start, out.bb = self.start, self.bb
        s = STREETS.index(street)
        if s != out.street:
            out._new_street(s)
        out.ok = True
        return out

    def _new_street(self, s: int) -> None:
        self.prev_aggr = self.aggr if s == self.street + 1 else -1
        self.street = s
        self.cur = 0
        self.last_full = self.bb
        self.raises = 0
        self.aggr = -1
        self.acted_round = set()
        self.acted_street = set()


class BlueprintPolicy:
    """Everything the bot loads once per process for one ruleset."""

    def __init__(self, ruleset_id: str, pack: PolicyPack | None = None) -> None:
        if ruleset_id not in RULESETS:
            raise ValueError(f"no blueprint for {ruleset_id}")
        self.ruleset_id = ruleset_id
        self.caps = capabilities(ruleset_id)
        self.cards = CardAbstraction(self.caps)
        self.pack = pack or PolicyPack(pack_path(ruleset_id))
        self.offsets = A.OFFSETS
        self.translate = "off"
        self._pressure: dict[tuple[str, int], int] = {}
        for i, a in enumerate(self.caps.deck):
            for b in self.caps.deck[i + 1:]:
                self.cards.preflop((a, b))

    def pressure_bucket(self, obs, start: dict[int, int]) -> int:
        k = (obs.hand_id, obs.hero_seat)
        b = self._pressure.get(k)
        if b is None:
            seats = sorted(s for s, c in start.items() if c > 0)
            stacks = [start[s] for s in seats]
            n = len(seats)
            first = float(obs.payouts[0]) if obs.payouts and obs.payouts[0] else 1.0
            prizes = [float(obs.payouts[i]) / first if i < len(obs.payouts) else 0.0 for i in range(n)]
            hero = seats.index(obs.hero_seat)
            others = [q for q in range(n) if q != hero]
            b = 0
            if others:
                v = max(others, key=lambda q: (stacks[q], -q))
                b = A.pressure_bucket(pressure(stacks, hero, v, prizes))
            if len(self._pressure) > 4096:
                self._pressure.clear()
            self._pressure[k] = b
        return b

    def features(self, obs, rp: HandReplay | None = None):
        """``(key, mask, targets)`` of the hero's decision, or None outside the abstraction.

        ``rp``: the hand's replay if the caller already has it (see :meth:`HandReplay.at`)."""
        if rp is None:
            rp = HandReplay(obs)
        if not rp.ok:
            return None
        legal = obs.legal
        street = STREETS.index(obs.street)
        hero = obs.hero_seat
        me = None
        cur = obs.current_bet_to
        behind = active = other_live = 0
        other_max = other_max_start = 0
        for p in obs.players:
            if p.status in ("folded", "eliminated"):
                continue
            active += 1
            if p.seat == hero:
                me = p
                continue
            if p.status == "active":
                other_live += 1
                if p.seat not in rp.acted_round or p.street_committed < cur:
                    behind += 1
            other_max = max(other_max, p.stack)
            other_max_start = max(other_max_start, p.stack + p.hand_committed)
        if me is None:
            return None
        to_call = min(legal.call_amount or 0, me.stack)
        pot = obs.pot_total
        eff = min(me.stack, other_max)
        eff_start = min(me.stack + me.hand_committed, other_max_start)
        bb = obs.blinds.big_blind if obs.blinds else rp.bb
        ctx = A.context_key(street, behind, active, rp.raises, to_call, pot, eff, eff_start, bb,
                            1 if hero in rp.acted_street else 0, 1 if rp.prev_aggr == hero else 0,
                            self.pressure_bucket(obs, rp.start), cur, me.street_committed)
        bucket = self.cards.bucket(card_ints(obs.hole_cards), card_ints(obs.board))
        key = A.infoset_key(street, bucket, ctx, self.offsets)
        max_to = legal.all_in_raise_to or legal.max_raise_to or 0
        min_to = legal.min_full_raise_to if legal.min_full_raise_to is not None else max_to
        mask, t_min, t_a, t_b, t_all = A.action_targets(street, cur, to_call, pot, min_to, max_to, rp.raises,
                                                        1 if legal.can_raise and max_to > cur else 0)
        if not legal.fold:
            mask &= ~(1 << A.FOLD)
        return key, mask, (0, 0, t_min, t_a, t_b, t_all)

    def play_features(self, obs, seed: str, rp: HandReplay | None = None):
        """:meth:`features` of a decision being played: with :attr:`translate` on, the key
        of a facing bet may move one price bucket down (:func:`lower_price_probability`),
        drawn from a stream keyed by the policy seed and the decision id."""
        f = self.features(obs, rp)
        if f is None or self.translate == "off" or (self.translate == "postflop" and obs.street == "preflop"):
            return f
        call_amount = obs.legal.call_amount or 0
        if not call_amount:
            return f
        stack = next(p.stack for p in obs.players if p.seat == obs.hero_seat)
        p_low = lower_price_probability(min(call_amount, stack), obs.pot_total)
        if p_low and _uniform("translate|" + seed, obs.decision_id) < p_low:
            return (f[0] - PRICE_STRIDE, *f[1:])
        return f

    def act(self, obs, u: float, seed: str = ""):
        """``(action, note)`` from the blueprint, or None on a table miss."""
        return self.act_on(obs, u, self.play_features(obs, seed))

    def act_on(self, obs, u: float, f):
        """:meth:`act` given ``f = self.features(obs)``."""
        if f is None:
            return None
        return self.act_row(obs, u, f, self.pack.lookup(f[0]))

    def act_row(self, obs, u: float, f, hit):
        """:meth:`act_on` given ``hit = self.pack.lookup(f[0])`` (``f`` not None)."""
        if hit is None:
            return None
        _, mask, targets = f
        row, level = hit
        weights = [row[a] if (mask >> a) & 1 else 0 for a in range(A.NUM_ACTIONS)]
        total = sum(weights)
        if total <= 0:
            return None
        x = u * total
        a = max(i for i in range(A.NUM_ACTIONS) if weights[i] > 0)
        for i, w in enumerate(weights):
            if w and x < w:
                a = i
                break
            x -= w
        legal = obs.legal
        if a == A.FOLD:
            action = fold()
        elif a == A.CALL:
            action = call() if legal.call_amount else check()
        elif a == A.MIN_RAISE:
            action = raise_to(legal.min_full_raise_to) if legal.min_full_raise_to else None
        elif a == A.ALL_IN:
            action = raise_to(legal.all_in_raise_to or legal.max_raise_to)
        else:
            action = clamp_raise_to(legal, targets[a])
        if action is None:
            action = call() if legal.call_amount else check()
        return action, f"bp L{level} {A.ACTION_NAMES[a]} p={weights[a] / total:.2f}"


class BlueprintBot(ExplicitStateBot):
    rulesets = RULESETS

    def __init__(self, leaf: str = "points", pack: str | Path | None = None, translate: str = "off") -> None:
        if leaf not in LEAVES:
            raise ValueError(f"leaf is one of {LEAVES}")
        if translate not in TRANSLATE:
            raise ValueError(f"translate is one of {TRANSLATE}")
        self.leaf = leaf
        self.translate = translate
        self.name = "blueprint" if leaf == "points" else f"blueprint_{leaf}"
        self._pack_path = Path(pack) if pack else None
        self._policies: dict[str, BlueprintPolicy] = {}
        self._leaf = PushFoldBot(leaf)
        self.stats = self._fresh_stats()
        for ruleset_id in RULESETS:
            self.warm(ruleset_id)

    @staticmethod
    def _fresh_stats() -> dict:
        return {"leaf": 0, "blueprint": 0, "fallback": 0, "levels": [0, 0, 0, 0]}

    def policy(self, ruleset_id: str) -> BlueprintPolicy:
        p = self._policies.get(ruleset_id)
        if p is None:
            pack = PolicyPack(self._pack_path) if self._pack_path else None
            p = self._policies[ruleset_id] = BlueprintPolicy(ruleset_id, pack)
            p.translate = self.translate
        return p

    def initial_state(self, table):
        self.policy(table.ruleset_id)
        self._leaf.initial_state(table)
        return {"seed": table.policy_seed}

    def warm(self, ruleset_id: str) -> None:
        """Load the pack and tables and play the recorded observations, from the constructor.

        A sandboxed worker builds the bot in its untimed load but calls ``initial_state``
        only on a fresh table: a worker restored after a restart, or replaying a timeout,
        goes straight to a timed decision. Loading there costs about 80 ms, and cold first
        decisions (unspecialised bytecode, cold caches) about half the 1 ms budget. Only
        memo caches of pure functions are touched, so play is unchanged.
        """
        self.policy(ruleset_id)
        recorded = warm_observations(ruleset_id)
        for _ in range(WARM_PASSES if recorded else 0):
            for raw in recorded:
                self.decide(Observation.model_validate_json(raw), {"seed": "warm"})
        self.policy(ruleset_id)._pressure.clear()
        self.stats = self._fresh_stats()

    def decide(self, obs, state):
        seed = (state or {}).get("seed", "")
        policy = self.policy(obs.ruleset_id)
        base = self._leaf._policy(obs.ruleset_id)
        action = self._leaf.push_fold(obs, base, seed)
        if action is not None:
            self.stats["leaf"] += 1
            return with_note(action, "push/fold table"), state
        hit = policy.act(obs, _uniform(seed, obs.decision_id), seed)
        if hit is not None:
            action, note = hit
            self.stats["blueprint"] += 1
            self.stats["levels"][int(note[4])] += 1
            return with_note(action, note), state
        self.stats["fallback"] += 1
        return with_note(base.decide(obs), "blueprint miss: base policy"), state
