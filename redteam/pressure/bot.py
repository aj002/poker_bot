"""Pressure: a fast, adaptive, ICM-aware NLHE tournament bot.

The policy is deliberately polarized.  It opens and reraises aggressively, uses
large post-flop sizes, and applies extra pressure when it covers medium stacks.
It still protects its own tournament life and uses observed fold rates only
after a useful sample.
"""

from __future__ import annotations

import hashlib

from arena.agents.sdk import ExplicitStateBot, call, check, clamp_raise_to, fold, with_note
from arena.agents.styles.engine import parse
from arena.agents.styles.handeval import RV, strength
from arena.table.player_stats import StatsTracker


def _random(state: dict) -> float:
    state["draw"] += 1
    raw = hashlib.blake2b(
        f"{state['seed']}:{state['draw']}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(raw, "big") / 18446744073709551616.0


def _preflop_power(hole: list[str]) -> float:
    """Cheap continuous starting-hand score, roughly 0..1."""
    a, b = sorted((RV[c[0]] for c in hole), reverse=True)
    pair = a == b
    suited = hole[0][1] == hole[1][1]
    gap = a - b
    if pair:
        return min(1.0, 0.48 + 0.043 * a)
    score = 0.10 + 0.038 * a + 0.018 * b
    if a == 12:
        score += 0.08
    if suited:
        score += 0.075
    if gap == 1:
        score += 0.055
    elif gap == 2:
        score += 0.025
    elif gap >= 4:
        score -= 0.045 * min(3, gap - 3)
    if a >= 9 and b >= 8:
        score += 0.05
    return max(0.0, min(1.0, score))


def _act(action, why: str):
    return with_note(action, why)


def _pass(spot, why: str):
    return _act(check() if spot.obs.legal.check else fold(), why)


def _continue(spot, why: str):
    return _act(check() if spot.obs.legal.check else call(), why)


def _raise(spot, target: int, why: str):
    action = clamp_raise_to(spot.obs.legal, target)
    return _act(action, why) if action is not None else _continue(spot, why)


def _jam(spot, why: str):
    legal = spot.obs.legal
    target = legal.all_in_raise_to or legal.max_raise_to or legal.short_all_in_raise_to
    return _raise(spot, target, why) if target is not None else _continue(spot, why)


def _active_opponents(spot):
    return [
        p
        for p in spot.obs.players
        if p.seat != spot.hero and p.status in ("active", "all_in")
    ]


def _tournament_factors(spot) -> tuple[float, float]:
    """Return (pressure bonus, survival penalty) on a compact 0..~0.2 scale."""
    opponents = _active_opponents(spot)
    if not opponents:
        return 0.0, 0.0
    hero_total = spot.stack + spot.committed
    stacks = sorted(p.stack + p.street_committed for p in opponents)
    survivors = spot.obs.surviving_player_count
    late = survivors <= max(4, len(spot.obs.payouts) // 2 + 1)
    near_cap = spot.obs.hands_remaining <= 25

    covered_medium = sum(
        1
        for x in stacks
        if hero_total >= 1.25 * x and x >= 6 * spot.bb
    )
    pressure = min(0.10, 0.022 * covered_medium)
    if late or near_cap:
        pressure *= 1.45

    shorter = sum(1 for x in stacks if x < hero_total * 0.55)
    survival = 0.0
    if (late or near_cap) and shorter and hero_total < max(stacks) * 1.15:
        survival = min(0.18, 0.055 + 0.025 * shorter)
    return pressure, survival


def _fold_edge(spot, tracker: StatsTracker) -> float:
    """Confidence-weighted exploit bonus against players who surrender to reraises."""
    candidates = _active_opponents(spot)
    if spot.pre_aggressor is not None:
        candidates = [p for p in candidates if p.seat == spot.pre_aggressor] or candidates
    edge = 0.0
    for player in candidates:
        rates = tracker.rates(player.seat)
        hands = int(rates["hands"] or 0)
        fold_rate = rates["fold_to_reraise"]
        rfi = rates["rfi"]
        if hands >= 8 and fold_rate is not None:
            edge = max(edge, (fold_rate - 0.45) * min(1.0, hands / 24))
        elif hands >= 12 and rfi is not None and rfi < 0.12:
            edge = max(edge, 0.04)
    return max(-0.04, min(0.16, edge))


class PressureBot(ExplicitStateBot):
    name = "redteam_pressure"
    rulesets = ("practice_nlhe",)

    def initial_state(self, table):
        return {
            "seed": table.policy_seed,
            "draw": 0,
            "stats": StatsTracker(table.seats, max_showdowns=1).to_state(),
        }

    def decide(self, obs, state):
        tracker = StatsTracker.from_state(state["stats"])
        tracker.observe(obs)
        spot = parse(obs)
        pressure, survival = _tournament_factors(spot)
        exploit = _fold_edge(spot, tracker)

        if spot.street == "preflop":
            action = self._preflop(spot, state, pressure, survival, exploit)
        else:
            action = self._postflop(spot, state, pressure, survival, exploit)
        state["stats"] = tracker.to_state()
        return action, state

    def _preflop(self, s, state, pressure, survival, exploit):
        power = _preflop_power(s.hole)
        total = s.stack + s.committed
        bb_stack = total / max(1, s.bb)
        late = s.label in ("HJ", "CO", "BTN", "SB")
        unopened = s.pre_raises == 0

        # Short stacks use decisive jams; survival pressure tightens the threshold.
        if bb_stack <= 10:
            threshold = 0.72 + 0.08 * survival
            if unopened:
                threshold -= 0.11 if late else 0.02
            elif s.pre_raises == 1:
                threshold += 0.09
            if power >= threshold:
                return _jam(s, f"short-stack pressure jam ({bb_stack:.1f}bb)")
            return _pass(s, "short stack, outside jam range")

        if unopened:
            threshold = 0.68 if s.group == "early" else 0.61
            if late:
                threshold -= 0.09
            threshold += survival - 0.55 * pressure - 0.30 * exploit
            if power >= threshold or (
                late
                and s.folded_to
                and power >= 0.40
                and _random(state) < max(0.0, 0.10 + pressure + 0.5 * exploit)
            ):
                mult = 3.4 if s.limpers == 0 else 4.2
                target = int(mult * s.bb + s.limpers * s.bb)
                return _raise(s, target, "large open isolates and pressures covered stacks")
            return _pass(s, "declined marginal open")

        if s.pre_raises == 1 and not s.hero_raised_pre:
            value = 0.86 + 0.35 * survival
            light = 0.69 + 0.25 * survival - 0.45 * pressure - 0.40 * exploit
            light_roll = 0.10 + 0.7 * pressure + 0.6 * exploit
            if power >= value or (power >= light and _random(state) < light_roll):
                target = int(max(s.cur * (4.0 if not s.in_position else 3.5), s.cur + 4 * s.bb))
                if target >= total * 0.42:
                    return _jam(s, "polarized 3-bet commits enough to jam")
                return _raise(s, target, "high-pressure polarized 3-bet")
            # Calls are intentionally scarce and position/price sensitive.
            if power >= 0.74 + 0.25 * survival and s.to_call <= total * 0.09 and s.in_position:
                return _continue(s, "strong playable hand in position")
            return _pass(s, "folded rather than make a dominated flat")

        # Four-bet/jam mostly for value, with rare blocker-style pressure.
        jam_line = power >= 0.91 + 0.25 * survival
        bluff_line = (
            power >= 0.76
            and exploit > 0.08
            and _random(state) < 0.04 + 0.5 * exploit
        )
        if jam_line or bluff_line:
            return _jam(s, "polarized four-bet jam")
        if s.hero_raised_pre and power >= 0.82 + 0.30 * survival and s.to_call <= total * 0.22:
            return _continue(s, "continued strong range versus reraise")
        return _pass(s, "protected tournament life versus heavy action")

    def _postflop(self, s, state, pressure, survival, exploit):
        bucket, outs, _ = strength(s.hole, s.board)
        raiser = s.pre_aggressor == s.hero
        heads_up = s.opponents <= 1
        fold_bonus = pressure + exploit
        pot_before = max(1, s.pot - s.to_call)
        price = s.to_call / pot_before

        if s.to_call:
            if bucket == "monster":
                # Large raises deny equity and maximize value against capped ranges.
                return _raise(s, s.cur + int(1.35 * (s.pot + s.to_call)), "monster overraise")
            if bucket == "strong":
                if price <= 1.05 and (
                    price <= 0.55 or _random(state) < 0.30 + fold_bonus
                ):
                    if s.obs.legal.can_raise and _random(state) < 0.38 + pressure:
                        return _raise(s, s.cur * 3 + s.pot // 2, "strong polarized raise")
                    return _continue(s, "strong hand continues")
                return _pass(s, "strong-but-vulnerable hand avoids ICM disaster")
            if outs >= 8 and price <= 0.65:
                if s.obs.legal.can_raise and _random(state) < 0.46 + fold_bonus - survival:
                    return _raise(s, s.cur * 3 + s.pot // 2, "maximum-pressure semi-bluff")
                return _continue(s, "high-equity draw receives acceptable price")
            if bucket == "medium" and price <= 0.34 - 0.35 * survival:
                return _continue(s, "medium showdown value at a cheap price")
            if (
                heads_up
                and s.obs.legal.can_raise
                and price <= 0.75
                and _random(state) < max(0.0, 0.025 + 0.55 * fold_bonus - survival)
            ):
                return _raise(s, s.cur * 3 + s.pot, "targeted bluff raise into overfolder")
            return _pass(s, "range bottom releases to resistance")

        # Checked to us: value and bluffs share deliberately large sizes.
        if bucket == "monster":
            frac = 1.35 if heads_up else 0.90
            return _raise(s, s.cur + int(frac * s.pot), "polarized overbet for value")
        if bucket == "strong":
            frac = 1.05 if heads_up else 0.72
            return _raise(s, s.cur + int(frac * s.pot), "large value bet")
        if outs >= 8 and _random(state) < 0.72 + min(0.18, fold_bonus):
            return _raise(s, s.cur + int(1.10 * s.pot), "overbet semi-bluff")

        street_rate = {"flop": 0.72, "turn": 0.43, "river": 0.24}.get(s.street, 0.0)
        if raiser and heads_up and _random(state) < max(
            0.0, street_rate + fold_bonus - 0.75 * survival
        ):
            frac = 0.82 if s.street == "flop" else 1.18
            return _raise(s, s.cur + int(frac * s.pot), "relentless polarized barrel")
        if bucket == "medium" and _random(state) < 0.34 + 0.5 * pressure:
            return _raise(s, s.cur + int(0.62 * s.pot), "thin value under stack pressure")
        if heads_up and _random(state) < max(0.0, 0.04 + 0.55 * fold_bonus - survival):
            return _raise(s, s.cur + int(1.15 * s.pot), "delayed overbet attack")
        return _pass(s, "checked back range bottom")


def make_bot():
    return PressureBot()
