"""Variant-agnostic fallback bot: legal and sane on any ruleset the rules adapter can describe.

This is the build-day safe first submission. It needs no generated tables, no search
and no per-variant code: everything it knows about the game comes from the variant's
:class:`~strategy.rules.variants.VariantSpec` (deck ranks, hole-card count, showdown
selection, category order) through :mod:`strategy.features.cards`, and every action
goes through :func:`~strategy.policy.fallback.legalize`, so pot-limit and no-limit
caps, short all-ins and closed betting are the legal set's business, not the policy's.

Strategy, in the order it is tried each decision:

* **Preflop** a starting-hand score (high cards on the variant's own rank scale, pairs,
  suits, connectivity; for exactly-two-hole showdowns the best two-card subsets plus
  coordination) is turned into a percentile against random hands of the same variant
  (a seeded empirical CDF built once per process). Short stacks (``JAM_BB`` or less)
  shove or fold; deeper stacks open-raise or fold, tighter with more players behind;
  facing a raise they re-raise the top, call the next band when the price is small,
  and otherwise fold. The big blind checks its option.
* **Antes** (documented approximation, no ante-specific tables): preflop the antes are
  the chips already collected (``settled_pot``). Stack depth for the shove/fold and
  re-shove cut-offs is the M-equivalent ``eff * (sb + bb) / (sb + bb + antes)``, the
  limper count ignores the ante pot, and the open and shove thresholds drop by up to
  ``ANTE_WIDEN`` percentile as the antes grow to the size of the blinds. With no ante
  every number is exactly the ante-free one.
* **Postflop** the made-hand level of :func:`~strategy.features.cards.made_hand`
  (whether the hole cards improve the board) and a cheap flush/straight draw count
  pick value bets, pot-odds calls and folds. Big bets need a strong hand.
* **Card-blind mode** if the card layer cannot read the variant, preflop shoves only
  when very short and every other decision checks or folds.

Explicit state is ``{"v", "seed", "n", "fb"}``; randomness comes from the seat's policy
seed in counter mode, so a restored worker replays exactly. Any exception plays the
constant-time legal fallback and is counted in ``state["fb"]``.
"""

from __future__ import annotations

import bisect
import hashlib
import random
from collections.abc import Sequence
from itertools import combinations
from typing import Any

from arena.agents.sdk import ExplicitStateBot
from strategy.policy.fallback import fallback_action, is_legal, legalize

STATE_VERSION = 1
RULESETS = ("practice_nlhe", "uncertainty_shortdeck", "uncertainty_plo4", "reveal_variant")
JAM_BB = 12.0
ANTE_WIDEN = 0.06
CDF_SAMPLES = 3000
_SCALE = 1.0 / 18446744073709551616.0
_RANKS = "23456789TJQKA"
_CDF: dict[tuple[str, int, int | None], list[float]] = {}


class _Draws:
    __slots__ = ("prefix", "n")

    def __init__(self, seed: str, n: int) -> None:
        self.prefix = seed.encode() + b":"
        self.n = n

    def __call__(self) -> float:
        self.n += 1
        h = hashlib.blake2b(self.prefix + str(self.n).encode(), digest_size=8).digest()
        return int.from_bytes(h, "big") * _SCALE


class _Game:
    """What the policy reads from the variant, resolved once per table."""

    __slots__ = ("ruleset_id", "caps", "ranks", "hole", "used", "index", "cdf")

    def __init__(self, ruleset_id: str) -> None:
        from strategy.rules.reveal import get_variant

        spec = get_variant(ruleset_id)
        self.ruleset_id = ruleset_id
        self.ranks = spec.ranks
        self.hole = spec.hole_cards
        self.used: int | None = 2 if spec.hand_selection == "two_hole_three_board" else None
        self.index = {r: i / max(1, len(spec.ranks) - 1) for i, r in enumerate(spec.ranks)}
        try:
            from strategy.features.cards import capabilities, made_hand

            self.caps: Any = capabilities(ruleset_id)
            deck = self.caps.deck
            made_hand(deck[:self.hole], deck[self.hole:self.hole + 3], self.caps)  # builds the lazy rank tables
        except Exception:  # the card layer cannot read this variant: card-blind mode
            self.caps = None
        key = (spec.ranks, spec.hole_cards, self.used)
        cdf = _CDF.get(key)
        if cdf is None:
            cdf = _CDF[key] = self._build_cdf(spec.suits)
        self.cdf = cdf

    def _build_cdf(self, suits: str) -> list[float]:
        deck = [r + s for r in self.ranks for s in suits]
        rng = random.Random(f"generic-fallback-cdf-v1|{self.ranks}|{self.hole}|{self.used}")
        return sorted(self.score(rng.sample(deck, self.hole)) for _ in range(CDF_SAMPLES))

    def _pair_score(self, a: str, b: str) -> float:
        x, y = self.index[a[0]], self.index[b[0]]
        hi, lo = (x, y) if x >= y else (y, x)
        if a[0] == b[0]:
            return 0.55 + 0.45 * hi
        gap = round((hi - lo) * (len(self.ranks) - 1))
        if hi == 1.0 and lo * (len(self.ranks) - 1) <= 3:
            gap = min(gap, round(lo * (len(self.ranks) - 1)) + 1)
        s = 0.30 * hi + 0.18 * lo + (0.06 if a[1] == b[1] else 0.0)
        return s + (0.05 if gap <= 1 else 0.03 if gap == 2 else 0.0)

    def score(self, hole: Sequence[str]) -> float:
        if len(hole) == 2:
            return self._pair_score(hole[0], hole[1])
        pairs = sorted((self._pair_score(a, b) for a, b in combinations(hole, 2)), reverse=True)
        s = pairs[0] + 0.4 * pairs[1] + 0.2 * pairs[2]
        rank_counts = [sum(1 for c in hole if c[0] == r) for r in {c[0] for c in hole}]
        suit_counts = sorted((sum(1 for c in hole if c[1] == u) for u in {c[1] for c in hole}), reverse=True)
        if max(rank_counts) >= 3:
            s -= 0.08
        if suit_counts[0] >= 3:
            s -= 0.02 * (suit_counts[0] - 2)
        if len(suit_counts) > 1 and suit_counts[0] >= 2 and suit_counts[1] >= 2:
            s += 0.04
        return s

    def percentile(self, hole: Sequence[str]) -> float:
        return bisect.bisect_left(self.cdf, self.score(hole)) / len(self.cdf)


class GenericBot(ExplicitStateBot):
    name = "strategy_fallback"
    rulesets = RULESETS

    def __init__(self, *, name: str | None = None) -> None:
        if name:
            self.name = name
        self._game: _Game | None = None
        self._hand: tuple[str | None, float] = (None, 0.0)
        self._street: tuple[str | None, int, int, int] = (None, -1, 0, 0)

    def initial_state(self, table: Any) -> dict[str, Any]:
        self._game = _Game(table.ruleset_id)
        self._hand = (None, 0.0)
        self._street = (None, -1, 0, 0)
        return {"v": STATE_VERSION, "seed": table.policy_seed, "n": 0, "fb": 0}

    def decide(self, observation: Any, state: Any) -> tuple[Any, dict[str, Any]]:
        legal = observation.legal
        draws = _Draws(state["seed"], state["n"])
        fb = state["fb"]
        try:
            if self._game is None or self._game.ruleset_id != observation.ruleset_id:
                self._game = _Game(observation.ruleset_id)
            kind, amount, all_in_to = self._intent(observation, draws)
            action = legalize(kind, amount, legal, all_in_to)
            if not is_legal(action, legal):
                raise ValueError(f"illegal {action!r}")
        except Exception:  # consumed by the legal fallback and counted
            action = fallback_action(legal)
            fb += 1
        return action, {"v": STATE_VERSION, "seed": state["seed"], "n": draws.n, "fb": fb}

    # ----------------------------------------------------------------- policy

    def _intent(self, obs: Any, draws: _Draws) -> tuple[str, int | None, int]:
        g = self._game
        assert g is not None
        legal = obs.legal
        hero = obs.hero_seat
        me = None
        live_totals: list[int] = []
        active: list[int] = []
        for p in obs.players:
            if p.seat == hero:
                me = p
            elif p.status in ("active", "all_in"):
                live_totals.append(p.stack + p.hand_committed)
                if p.status == "active":
                    active.append(p.seat)
        assert me is not None
        all_in_to = me.stack + me.street_committed
        bb = max(1, obs.blinds.big_blind if obs.blinds is not None else 1)
        total = me.stack + me.hand_committed
        eff = min(total, max(live_totals, default=0)) / bb
        to_call = legal.call_amount or 0
        pot = obs.pot_total
        opponents = len(live_totals)
        if obs.street == "preflop":
            return self._preflop(obs, g, draws, bb, eff, to_call, pot, total, all_in_to, active)
        return self._postflop(obs, g, draws, to_call, pot, total, all_in_to, opponents)

    def _behind(self, obs: Any, active: Sequence[int]) -> int:
        ring = sorted(obs.seats)
        start = obs.big_blind_seat if obs.big_blind_seat in ring else ring[0]
        i = ring.index(start)
        order = ring[i + 1:] + ring[:i + 1]
        hero = obs.hero_seat
        if hero not in order:
            return len(active)
        after = order[order.index(hero) + 1:]
        live = set(active)
        return sum(1 for s in after if s in live)

    def _preflop(self, obs: Any, g: _Game, draws: _Draws, bb: int, eff: float, to_call: int, pot: int, total: int,
                 all_in_to: int, active: Sequence[int]) -> tuple[str, int | None, int]:
        legal = obs.legal
        if self._hand[0] != obs.hand_id:
            pct = g.percentile(obs.hole_cards) if g.caps is not None else 0.0
            self._hand = (obs.hand_id, pct)
        pct = self._hand[1]
        if g.caps is None:
            if eff <= 4 and legal.can_raise:
                return "allin", None, all_in_to
            return "check", None, all_in_to
        behind = self._behind(obs, active)
        raised = obs.current_bet_to > bb
        antes = obs.settled_pot
        widen = 0.0
        if antes:
            blinds = obs.blinds.small_blind + bb if obs.blinds is not None else int(1.5 * bb)
            eff = eff * blinds / (blinds + antes)
            widen = ANTE_WIDEN * min(1.0, antes / blinds)
        if not raised:
            if eff <= JAM_BB:
                thr = 0.62 + 0.03 * behind - 0.20 * (1.0 - eff / JAM_BB)
                if widen:
                    thr -= widen
                if pct >= thr:
                    return "allin", None, all_in_to
                return ("check" if legal.check else "fold"), None, all_in_to
            thr = 0.70 + 0.035 * behind
            if widen:
                thr -= widen
            if pct >= thr:
                limpers = max(0, (pot - antes - int(1.5 * bb)) // bb)
                return "raise", int((2.5 + limpers) * bb), all_in_to
            return ("check" if legal.check else "fold"), None, all_in_to
        price = to_call / max(1, total)
        if pct >= 0.95:
            if eff <= 25:
                return "allin", None, all_in_to
            return "raise", 3 * obs.current_bet_to, all_in_to
        if pct >= 0.80 + 0.4 * price and price <= 0.30:
            return "call", None, all_in_to
        if obs.hero_seat == obs.big_blind_seat and to_call <= bb and pct >= 0.45:
            return "call", None, all_in_to
        return "fold", None, all_in_to

    def _street_read(self, obs: Any, g: _Game) -> tuple[int, int]:
        """(made level, draw outs class) for this street, computed once."""
        key = (obs.hand_id, len(obs.board))
        if self._street[:2] == key:
            return self._street[2], self._street[3]
        from strategy.features.cards import card_ints, made_hand

        hole = card_ints(obs.hole_cards)
        board = card_ints(obs.board)
        level = made_hand(hole, board, g.caps).level
        draw = _draw_class(hole, board, g.used, g.caps) if len(board) < 5 else 0
        self._street = (obs.hand_id, len(obs.board), level, draw)
        return level, draw

    def _postflop(self, obs: Any, g: _Game, draws: _Draws, to_call: int, pot: int, total: int, all_in_to: int,
                  opponents: int) -> tuple[str, int | None, int]:
        if g.caps is None:
            return "check", None, all_in_to
        level, draw = self._street_read(obs, g)
        omaha = g.used is not None
        strong_at, medium_at = (7, 5) if omaha else (5, 3)
        if opponents >= 3:
            strong_at, medium_at = strong_at + 1, medium_at + 1
        strong, medium = level >= strong_at, level >= medium_at
        if to_call == 0:
            if strong:
                return "raise", obs.current_bet_to + int(0.66 * pot), all_in_to
            if (medium and opponents <= 2 and draws() < 0.55) or (draw >= 1 and opponents == 1 and draws() < 0.35):
                return "raise", obs.current_bet_to + int(0.5 * pot), all_in_to
            return "check", None, all_in_to
        odds = to_call / max(1, pot + to_call)
        commit = to_call / max(1, total)
        if strong:
            if level >= strong_at + 2 or (level >= strong_at + 1 and draws() < 0.5):
                return "raise", obs.current_bet_to + to_call + pot, all_in_to
            return "call", None, all_in_to
        if medium and odds <= 0.30 and commit <= 0.35:
            return "call", None, all_in_to
        if draw >= 1 and odds <= 0.22 and commit <= 0.20:
            return "call", None, all_in_to
        return "fold", None, all_in_to


def _draw_class(hole: Sequence[int], board: Sequence[int], used: int | None, caps: Any) -> int:
    """Count of strong draws (0-2): four to a flush, four to a straight, with the hole's required help."""
    need = 2 if used else 1
    cls = 0
    for s in range(4):
        h = sum(1 for c in hole if c & 3 == s)
        b = sum(1 for c in board if c & 3 == s)
        if (used and h >= 2 and b == 2) or (not used and h >= 1 and h + b == 4):
            cls += 1
            break
    hole_mask = board_mask = 0
    for c in hole:
        hole_mask |= 1 << (c >> 2)
    for c in board:
        board_mask |= 1 << (c >> 2)
    for w, _high in caps.windows:
        hn, bn = (hole_mask & w).bit_count(), (board_mask & w).bit_count()
        if hn >= need and bn <= 3 and min(hn, 2 if used else 5) + bn == 4:
            cls += 1
            break
    return cls


def make_bot(**kwargs: Any) -> GenericBot:
    return GenericBot(**kwargs)
