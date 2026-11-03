"""Family behaviours layered over the anchor style rules.

A *hook* sees the decision first and returns a decision or ``None`` (let the next hook,
then the anchor style, decide). A *post* step rewrites the chosen action, for example
its size. Hooks read the member's parameters from ``c.p`` and draw randomness only from
``c.ch`` (counted) or ``c.hand_uniform`` (fixed per hand), so play stays seeded.

Every action goes through the style engine's helpers, which clamp raises to the legal
descriptor; :func:`strategy.population.policy.legal_or_safe` is the final guard.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from arena.agents.sdk import clamp_raise_to
from arena.agents.styles.engine import (
    Decision,
    _all_in,
    _bet,
    _call,
    _check_or_fold,
    _open_size,
    _raise,
    _re,
)
from arena.domain.models import Action
from strategy.population import cards
from strategy.population.params import unit_hash
from strategy.population.policy import Ctx, equity_estimate, sigmoid


def _check(c: Ctx, why: str) -> Decision:
    return _call(c.s, why) if c.s.obs.legal.check else _check_or_fold(c.s, why)


# ------------------------------------------------------------------ 3 chaotic aggressor

def chaotic_observe(obs: Any, state: dict[str, Any], p: Mapping[str, float]) -> None:
    """Remember the hero's stack at the start of each hand; a big drop starts a tilt."""
    if state.get("h") == obs.hand_number:
        return
    start = next((int(e.data["start_stacks"].get(str(obs.hero_seat), 0)) for e in obs.action_history
                  if e.type == "HandStarted"), None)
    if start is None:
        return
    last = state.get("st", 0)
    if last and (last - start) / last >= p["tilt_loss"]:
        state["tilt"] = obs.hand_number + int(p["tilt_hands"])
    state["h"], state["st"] = obs.hand_number, start


def chaotic_hot(obs: Any, state: dict[str, Any], p: Mapping[str, float]) -> bool:
    mode = int(p["mode"])
    if mode == 1:
        return state.get("tilt", 0) >= obs.hand_number
    if mode == 2:
        alive = [pl.stack + pl.hand_committed for pl in obs.players if pl.status != "eliminated"]
        me = next(pl.stack + pl.hand_committed for pl in obs.players if pl.seat == obs.hero_seat)
        return bool(alive) and me >= 1.4 * sum(alive) / len(alive)
    return False


# ------------------------------------------------------------------ 5 tight surrenderer

def rare_bluff(c: Ctx) -> Decision | None:
    s = c.s
    if s.street == "river" and not s.to_call and c.bucket == "weak" and c.ch(c.p["river_bluff"]):
        return _bet(s, 0.5, "rare river bluff")
    return None


# ------------------------------------------------------------------ 6 smooth switcher

def smooth_gear(c: Ctx) -> str:
    """Soft gates on M, the projected M after the next blind rise, and hands to the cap."""
    s, p, obs = c.s, c.p, c.obs
    m = s.m
    if obs.next_level is not None:
        to_level = max(0, obs.next_level.starts_at_hand - obs.hand_number)
        proximity = max(0.0, 1.0 - to_level / 25.0)
        m_next = (s.stack + s.committed) / max(1, obs.next_level.small_blind + obs.next_level.big_blind)
        m = m + p["horizon"] * proximity * (m_next - m)
    urgency = p["cap_urgency"] * max(0.0, 1.0 - obs.hands_remaining / 40.0)
    p_push = sigmoid((p["m_push"] - m) / p["soft"])
    p_aggr = sigmoid((p["m_aggr"] - m) / p["soft"])
    p_aggr = p_aggr + (1 - p_aggr) * urgency
    u = c.hand_uniform("gear")
    if u < p_push:
        return "push_fold"
    if u < max(p_push, p_aggr):
        return "aggressive"
    return "conservative"


# ------------------------------------------------------------------ 7 limp / isolation

def limp_iso(c: Ctx) -> Decision | None:
    s, p, q = c.s, c.p, c.top
    if s.street != "preflop":
        return None
    if s.pre_raises == 0:
        if s.limpers == 0:
            if q <= p["open_top"]:
                if c.ch(p["limp_trap"]):
                    return _call(s, "limped a premium to reraise")
                return _raise(s, _open_size(s, p["open_mult"]), "opened")
            if q <= p["limp_width"]:
                return _call(s, "limped")
            return _check_or_fold(s, "folded")
        if q <= p["iso_top"]:
            return _raise(s, _open_size(s, p["iso_mult"]), "isolated the limpers")
        if q <= p["overlimp_width"]:
            return _call(s, "over-limped")
        return _check_or_fold(s, "folded behind limpers")
    if c.x.hero_limped and s.pre_raises == 1:
        if q <= p["limp_reraise_top"]:
            return _raise(s, _re(s, 3.5), "limp-reraised")
        if q <= p["limp_call"] and s.to_call <= p["limp_call_price"] * s.stack:
            return _call(s, "limp-called")
        return _check_or_fold(s, "limp-folded")
    return None


# ------------------------------------------------------------------ 8 squeezer

def squeeze(c: Ctx) -> Decision | None:
    s, p = c.s, c.p
    if s.street != "preflop":
        return None
    if s.pre_raises == 1 and not s.hero_raised_pre and c.x.callers >= 1:
        late = s.group in ("late", "blinds")
        width = p["squeeze_top"] * (1 + p["late_bonus"] * late)
        if c.top <= width or (c.top <= 0.45 and c.ch(p["light_squeeze"])):
            return _raise(s, int(s.cur * (p["squeeze_mult"] + c.x.callers)), "squeezed")
        return None
    if s.pre_raises >= 2 and c.x.hero_squeezed:
        if c.top <= p["keep_vs_4bet"]:
            return _all_in(s, "moved in over the four-bet")
        return _check_or_fold(s, "gave up the squeeze")
    return None


# ------------------------------------------------------------------ 9 trapper

def trap(c: Ctx) -> Decision | None:
    s, p = c.s, c.p
    if s.street == "preflop":
        if s.pre_raises == 1 and not s.hero_raised_pre and c.top <= p["flat_top"] and c.ch(p["flat_p"]):
            return _call(s, "flatted a premium")
        return None
    strong = c.bucket in ("monster", "strong")
    if not strong:
        return None
    wet = c.texture in ("wet", "monotone")
    if not s.to_call:
        if not s.hero_checked and c.ch(p["trap_p"] * (1 - p["abandon"] * wet)):
            return _check(c, "checked to trap")
        return None
    if s.hero_checked and not s.faced_reraise:
        if c.ch(p["cr_p"]):
            return _raise(s, _re(s, p["cr_mult"]), "check-raised")
        return _call(s, "slow-played")
    if c.bucket == "monster" and not s.faced_reraise and c.ch(p["slowplay_call"]):
        return _call(s, "slow-played")
    return None


# ------------------------------------------------------------------ 10 texture c-bettor

def texture_cbet(c: Ctx) -> Decision | None:
    s, p = c.s, c.p
    if s.street == "preflop" or s.to_call or s.pre_aggressor != s.hero or c.bucket in ("monster", "strong"):
        return None
    tex = c.texture
    mw = p["mw_mult"] ** max(0, s.opponents - 1)
    size = p["size_dry"] if tex in ("dry", "paired") else p["size_wet"]
    if s.street == "flop" and not s.hero_checked:
        if c.ch(p["cbet_" + tex] * mw):
            return _bet(s, size, f"continuation bet on a {tex} flop")
        return _check(c, "checked back the flop")
    if s.street == "turn" and not s.hero_checked:
        if c.x.hero_bet_flop:
            turn, flop_top = cards.cards_rank(s.board[3]), max(cards.cards_rank(x) for x in s.board[:3])
            prob = p["barrel"] + p["scare"] * (turn > flop_top)
        elif c.x.flop_raises == 0:
            prob = p["delayed"]
        else:
            return None
        if c.ch(prob * mw):
            return _bet(s, size, "second barrel")
        return _check(c, "gave up on the turn")
    return None


# ------------------------------------------------------------------ 11 draw chaser / attacker

def draws(c: Ctx) -> Decision | None:
    s, p = c.s, c.p
    if s.street == "preflop":
        return None
    if s.street == "river":
        if not s.to_call and c.bucket == "weak" and missed_draw(c) and c.ch(p["river_bluff"]):
            return _bet(s, p["bluff_frac"], "bluffed a missed draw")
        return None
    if c.outs < 4 or c.bucket in ("monster", "strong"):
        return None
    attack = p["attack"]
    if s.to_call:
        if c.ch(attack * p["semi"]):
            return _raise(s, _re(s, 3), "semi-bluff raise")
        if s.bet_frac <= p["price_tol"] * c.outs / 8:
            return _call(s, "chased the draw")
        return _check_or_fold(s, "folded the draw")
    if c.ch(attack):
        return _bet(s, p["draw_frac"], "semi-bluff bet")
    return _check(c, "checked the draw")


def missed_draw(c: Ctx) -> bool:
    if len(c.s.board) < 5:
        return False
    fn = c.s.evaluator or cards.strength_for("practice_nlhe")
    _bucket, outs, _ = fn(c.s.hole, c.s.board[:4])
    return outs >= 8


# ------------------------------------------------------------------ 12 sizing tells

def sizing_tell(c: Ctx, action: Action) -> Action:
    """Bet and raise sizes rise (positive), fall (negative) or stay flat (absent) with strength."""
    s, p, legal = c.s, c.p, c.obs.legal
    if action.type != "raise_to" or action.amount == legal.all_in_raise_to:
        return action
    sign = int(p["tell"]) - 1
    if s.street == "preflop":
        if s.pre_raises:
            return action
        score = 1.0 - min(1.0, c.top * 2.5)
        target = s.bb * p["open_mult"] * math.exp(sign * p["amp"] * (score - 0.5)) + s.limpers * s.bb
    else:
        score = cards.BUCKET_SCORE[c.bucket]
        frac = p["base_frac"] * math.exp(sign * p["amp"] * 2 * (score - 0.5))
        target = s.cur + s.to_call + frac * (s.pot + s.to_call)
    a = clamp_raise_to(legal, int(round(target)))
    return a if a is not None else action


# ------------------------------------------------------------------ 13 image-responsive

def image_observe(obs: Any, state: dict[str, Any], p: Mapping[str, float]) -> None:
    """Decayed per-seat counts of hands seen and preflop opens, from public events only."""
    img: dict[str, list[float]] = state["img"]
    last = state.get("seq", -1)
    opened = False
    for e in obs.action_history:
        if e.seq <= last:
            if e.type == "ActionApplied" and e.data["street"] == "preflop" and e.data["action"]["type"] == "raise_to":
                opened = True
            continue
        if e.type == "HandStarted":
            for k, v in img.items():
                img[k] = [round(v[0] * p["memory"], 4), round(v[1] * p["memory"], 4)]
            for seat in e.data["start_stacks"]:
                if int(seat) != obs.hero_seat:
                    v = img.setdefault(str(seat), [0.0, 0.0])
                    v[0] = round(v[0] + 1, 4)
            opened = False
        elif e.type == "ActionApplied" and not e.data.get("auto") and e.data["street"] == "preflop":
            if e.data["action"]["type"] == "raise_to":
                if not opened and e.data["seat"] != obs.hero_seat:
                    v = img.setdefault(str(e.data["seat"]), [0.0, 0.0])
                    v[1] = round(v[1] + 1, 4)
                opened = True
        state["seq"] = e.seq


def image_response(c: Ctx) -> Decision | None:
    s, p = c.s, c.p
    if s.street != "preflop" or s.pre_raises != 1 or s.hero_raised_pre or c.x.opener is None:
        return None
    hands, opens = c.state["img"].get(str(c.x.opener), [0.0, 0.0])
    if hands < 3:
        return None
    loose = (opens / hands - p["baseline"]) / p["baseline"]
    widen = max(0.25, min(3.0, 1 + p["amp"] * loose))
    if c.top <= p["rr_top"] * widen:
        return _raise(s, _re(s, 3), "reraised an opener it reads as loose" if loose > 0 else "reraised")
    if c.top <= p["call_top"] * widen and s.to_call <= 0.12 * s.stack:
        return _call(s, "called the open")
    return _check_or_fold(s, "folded to the open")


# ------------------------------------------------------------------ 14 ICM / cap specialist

CALIBRATION = (1.0, 2.2, 0.15, -0.8)
"""Multipliers on the ICM risk premium: correct, overweighted, ignored, inverted."""


def icm_equity(stacks: list[float], payouts: list[float], hero: int) -> float:
    """Malmuth-Harville equity for the top three places (enough for the practice payouts)."""
    total = sum(stacks)
    if total <= 0 or stacks[hero] <= 0:
        return 0.0
    pay = list(payouts) + [0.0] * 3
    eq = pay[0] * stacks[hero] / total
    if pay[1] or pay[2]:
        for j, sj in enumerate(stacks):
            if j == hero or sj <= 0:
                continue
            p_j = sj / total
            rest = total - sj
            eq += pay[1] * p_j * stacks[hero] / rest
            if pay[2]:
                for k, sk in enumerate(stacks):
                    if k in (hero, j) or sk <= 0:
                        continue
                    eq += pay[2] * p_j * (sk / rest) * stacks[hero] / (rest - sk)
    return eq


def icm_threshold(c: Ctx, risk: int) -> tuple[float, float]:
    """(chip-EV break-even, ICM break-even) win probability for calling ``risk`` chips."""
    obs, s = c.obs, c.s
    alive = [pl for pl in obs.players if pl.status != "eliminated"]
    hero = next(i for i, pl in enumerate(alive) if pl.seat == s.hero)
    bettor_seat = max((pl for pl in alive if pl.seat != s.hero), key=lambda pl: pl.street_committed).seat
    bettor = next(i for i, pl in enumerate(alive) if pl.seat == bettor_seat)
    stacks = [float(pl.stack) for pl in alive]
    pot = float(obs.pot_total)
    paid = [float(x) for x in obs.payouts]
    fold = list(stacks)
    fold[bettor] += pot
    win = list(stacks)
    win[hero] += pot
    lose = list(stacks)
    lose[hero] -= risk
    lose[bettor] += pot + risk
    e_fold, e_win, e_lose = (icm_equity(x, paid, hero) for x in (fold, win, lose))
    chip = risk / (pot + risk)
    if e_win <= e_lose:
        return chip, chip
    return chip, (e_fold - e_lose) / (e_win - e_lose)


def icm_cap(c: Ctx) -> Decision | None:
    s, p, obs = c.s, c.p, c.obs
    calib = CALIBRATION[int(p["calibration"])]
    near_cap = obs.hands_remaining <= p["cap_window"]
    aversion = p["aversion"] * (1 + p["cap_preserve"] * near_cap)
    if s.to_call and s.to_call >= p["commit_frac"] * (s.stack + s.committed):
        risk = min(s.to_call, s.stack)
        chip, icm = icm_threshold(c, risk)
        need = min(0.95, max(0.05, chip + calib * aversion * (icm - chip)))
        if equity_estimate(c) < need:
            return _check_or_fold(s, "folded: the tournament risk premium is too high")
        if s.to_call >= s.stack:
            return _call(s, "called all-in: enough equity after the risk premium")
        return None
    if s.street != "preflop" or s.pre_raises or s.limpers:
        return None
    alive = [pl.stack + pl.hand_committed for pl in obs.players if pl.status != "eliminated"]
    me = s.stack + s.committed
    if s.m < 8 and c.top <= p["short_jam"] * (1 + p["cap_urgency"] * near_cap):
        return _all_in(s, "short-stacked: moved in")
    if s.group in ("late", "blinds") and me >= 1.5 * sum(alive) / len(alive) and c.top <= 0.2 + 0.3 * p["bigstack"]:
        return _raise(s, _open_size(s, 2.5), "big stack: pressured the table")
    return None


# ------------------------------------------------------------------ 15 multiway specialist

def multiway(c: Ctx) -> Decision | None:
    s, p = c.s, c.p
    if s.street == "preflop":
        if (s.pre_raises == 1 and not s.hero_raised_pre and c.x.callers >= 1 and c.top <= p["overcall_pre"]
                and s.to_call <= 0.1 * s.stack):
            return _call(s, "overcalled a raise and a call")
        return None
    bucket, draw = c.bucket, c.outs >= 8
    if (not s.to_call and c.allin_opponents and c.active_opponents == 1 and len(c.obs.pots) <= 1
            and bucket != "monster" and c.ch(p["sidepot_aware"])):
        return _check(c, "checked: an all-in player cannot be bluffed and no side pot exists")
    if s.opponents < 2:
        return None
    if not s.to_call:
        if bucket == "weak" and not draw:
            if c.ch(p["mw_bluff"]):
                return _bet(s, 0.5, "multiway stab")
            return _check(c, "checked: too many players to bluff")
        if bucket == "medium" and c.ch(p["mw_value"]):
            return _check(c, "checked a medium hand multiway")
        return None
    if bucket == "medium" and c.x.street_callers >= 1 and s.bet_frac <= p["overcall"]:
        return _call(s, "overcalled")
    if bucket == "weak" and not draw:
        return _check_or_fold(s, "folded multiway")
    return None


# ------------------------------------------------------------------ 16 odd sizer / all-in heavy

def jam_heavy(c: Ctx) -> Decision | None:
    s, p = c.s, c.p
    if s.street == "preflop" and s.pre_raises >= 1 and c.top <= p["jam_width"] and c.ch(p["allin_rate"]):
        return _all_in(s, "moved in over a raise")
    return None


def odd_size(c: Ctx, action: Action) -> Action:
    s, p, legal = c.s, c.p, c.obs.legal
    if action.type != "raise_to" or action.amount is None:
        return action
    if c.ch(p["allin_rate"]):
        target = legal.all_in_raise_to or legal.max_raise_to or action.amount
    elif c.ch(p["minraise_rate"]):
        target = legal.min_full_raise_to or action.amount
    elif s.street != "preflop" and c.ch(p["overbet_rate"]):
        target = s.cur + s.to_call + (1.5 + 1.5 * c.uniform()) * (s.pot + s.to_call)
    else:
        target = action.amount * (1 + p["jitter"] * (2 * c.uniform() - 1)) + int(c.uniform() * s.bb)
    a = clamp_raise_to(legal, int(round(target)))
    return a if a is not None else action


# ------------------------------------------------------------------ 17 variant-noisy

def variant_noise(p: Mapping[str, float], seed: str, obs: Any, top: float):
    """Perturbed starting-hand rank and post-flop buckets, fixed per hand and street."""
    u1 = unit_hash(seed, "noise-pre", obs.hand_id)
    u2 = unit_hash(seed, "noise-pre2", obs.hand_id)
    gauss = math.sqrt(-2 * math.log(max(1e-12, u1))) * math.cos(2 * math.pi * u2)
    noisy = min(0.999, max(0.0, top + p["card_noise"] * gauss))
    base = cards.strength_for(obs.ruleset_id)
    flip = unit_hash(seed, "noise-post", obs.hand_id, obs.street)
    shift = 0 if flip >= p["bucket_flip"] else (1 if flip < p["bucket_flip"] / 2 else -1)
    out_shift = int(unit_hash(seed, "noise-outs", obs.hand_id, obs.street) * 7) - 3

    def evaluator(hole, board):
        bucket, outs, value = base(hole, board)
        i = max(0, min(3, cards.BUCKETS.index(bucket) + shift))
        return cards.BUCKETS[i], max(0, outs + out_shift if outs else 0), value

    return noisy, evaluator


# ------------------------------------------------------------------ HandHQ-calibrated families

def sticky(c: Ctx) -> Decision | None:
    """Calls a post-flop bet with a weak or medium hand at a price the seat tolerates, and stabs when checked to."""
    s, p = c.s, c.p
    if s.street == "preflop" or c.bucket in ("monster", "strong"):
        return None
    if s.to_call:
        if not s.faced_reraise and s.bet_frac <= p["sticky_price"] and c.ch(p["sticky"]):
            return _call(s, "called: does not believe the bet")
        return None
    if s.pre_aggressor != s.hero and not s.hero_checked and c.ch(p["stab"]):
        return _bet(s, p["stab_frac"], "stabbed at the pot")
    return None


def min_raiser(c: Ctx, action: Action) -> Action:
    """Turns a raise facing a bet or blind into the minimum legal raise with probability ``min_raise``."""
    legal = c.obs.legal
    if (action.type == "raise_to" and c.s.to_call and legal.min_full_raise_to is not None
            and legal.min_full_raise_to != legal.all_in_raise_to and c.ch(c.p["min_raise"])):
        return Action(type="raise_to", amount=legal.min_full_raise_to)
    return action
