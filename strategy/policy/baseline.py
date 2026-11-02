"""Stage 1 interpretable baseline: a stochastic rule policy with ~100 named parameters.

Every decision compares one hand-strength number with one threshold built from named
parameters, then mixes near the threshold with a logistic (``temp_pre`` / ``temp_post``)
or plays a named frequency. Hand strength is the preflop percentile (offline tables,
per variant) and, after the flop, equity against the live opponents' random hands
minus a named "range tax" for facing a bet. Tournament pressure enters through the
ICM risk premium (hand-start stacks, exact Malmuth-Harville), the cap horizon and
the stack share. Opponent counters move thresholds and bluff frequencies by at most
``exploit_cap``.

The defaults are the hand-written control policy. Search only changes numbers.

Antes (documented approximation, not a tuned parameter): preflop the Spot's ``eff_bb``
is the M-equivalent depth, so the ante-free jam and re-steal cut-offs apply unchanged,
and unopened open/jam thresholds drop by up to ``ANTE_WIDEN`` as the antes reach the
size of the blinds (more dead money to steal). Without antes nothing changes.

All parameters are variant-agnostic in form: the percentile, equity and texture come
from the variant's own tables, and sizes are clamped to the legal (pot- or no-limit)
interval at the runtime.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from strategy.policy.params import Param, ParamSpace
from strategy.policy.spot import AGGR, BB_G, FACED, FOLDED, HANDS, LATE_G, MIDDLE_G, PASSIVE, SB_G, VPIP, Spot

GROUPS = ("E", "M", "L", "SB", "BB")
STREET_NAMES = ("pre", "flop", "turn", "river")
PRIOR_VPIP = 0.30
PRIOR_FOLD = 0.45
PRIOR_AGGR = 0.35
ANTE_WIDEN = 0.06


def _p(name: str, default: float, lo: float, hi: float, group: str, doc: str) -> Param:
    return Param(name, default, lo, hi, group, doc)


PARAMS: tuple[Param, ...] = (
    # --- preflop opening
    *(_p(f"open_thr_{g}", d, 0.2, 0.99, "open", f"percentile needed to open from {g}")
      for g, d in zip(GROUPS, (0.85, 0.80, 0.62, 0.66, 0.80))),
    _p("open_deep_adj", 0.0, -0.15, 0.15, "open", "added to open thresholds at 75bb+ (linear from 25bb)"),
    _p("open_short_adj", -0.04, -0.2, 0.15, "open", "added to open thresholds at 10bb (linear from 20bb)"),
    _p("open_bb", 2.5, 2.0, 4.0, "size", "open raise size in big blinds"),
    _p("open_limper_bb", 1.0, 0.0, 2.5, "size", "extra big blinds per limper when isolating"),
    _p("sb_open_bb", 3.0, 2.0, 4.5, "size", "small-blind open size in big blinds"),
    _p("iso_pen", 0.04, -0.05, 0.15, "open", "extra percentile to raise over limpers, per limper"),
    _p("overlimp_band", 0.06, 0.0, 0.25, "open", "percentile band below the open threshold that over-limps"),
    _p("sb_complete_thr", 0.55, 0.2, 0.99, "open", "small blind completes an unopened pot above this"),
    _p("bb_iso_thr", 0.80, 0.4, 0.99, "open", "big blind raises limpers above this"),
    _p("steal_adj", 0.0, -0.15, 0.15, "open", "added to late/SB open thresholds when only blinds are behind"),
    # --- jam transition (push/fold)
    _p("jam_bb", 12.0, 4.0, 25.0, "jam", "at or below this effective depth, open-shove instead of raising"),
    *(_p(f"jam_thr_{g}", d, 0.15, 0.99, "jam", f"percentile to open-shove from {g} at jam_bb")
      for g, d in zip(GROUPS, (0.80, 0.75, 0.55, 0.50, 0.55))),
    _p("jam_slope", 0.15, 0.0, 0.5, "jam", "shove threshold drop from jam_bb down to 0bb"),
    _p("allin_commit", 0.45, 0.2, 0.9, "size", "a raise this large a share of the stack becomes all-in"),
    # --- facing one raise
    _p("tb_value_thr", 0.93, 0.7, 0.995, "3bet", "percentile to 3-bet for value"),
    _p("tb_oop_adj", 0.01, -0.05, 0.08, "3bet", "added to the 3-bet threshold out of position"),
    _p("tb_bluff_lo", 0.70, 0.3, 0.95, "3bet", "bottom of the 3-bet bluff band"),
    _p("tb_bluff_width", 0.06, 0.0, 0.25, "3bet", "width of the 3-bet bluff band"),
    _p("tb_bluff_freq", 0.25, 0.0, 1.0, "3bet", "3-bet frequency inside the bluff band"),
    _p("tb_size_ip", 3.0, 2.2, 4.5, "size", "3-bet size in position, times the open"),
    _p("tb_size_oop", 3.6, 2.5, 5.0, "size", "3-bet size out of position, times the open"),
    _p("gap_penalty", 0.06, -0.05, 0.25, "call", "Gap Concept: extra percentile to call over the raiser's open"),
    _p("call_price_coef", 0.8, 0.0, 3.0, "call", "threshold added per unit of (to call / stack)"),
    _p("bb_defend", 0.18, 0.0, 0.5, "call", "big blind's discount on the calling threshold"),
    _p("ip_call_bonus", 0.03, -0.05, 0.15, "call", "discount on calling thresholds in position"),
    _p("multi_call_bonus", 0.02, -0.05, 0.1, "call", "discount per caller already in"),
    _p("resteal_bb", 18.0, 5.0, 40.0, "jam", "at or below this depth, 3-bet means all-in"),
    _p("reshove_thr", 0.85, 0.5, 0.99, "jam", "percentile to re-shove over a raise"),
    _p("reshove_price", 0.3, 0.0, 1.5, "jam", "re-shove threshold added per unit of (to call / stack)"),
    # --- facing a 3-bet or more
    _p("fb_thr", 0.975, 0.85, 0.999, "4bet", "percentile to 4-bet after opening"),
    _p("fb_call_thr", 0.92, 0.7, 0.995, "4bet", "percentile to call a 3-bet after opening"),
    _p("fb_call_price", 0.5, 0.0, 2.0, "4bet", "3-bet calling threshold added per unit of (to call / stack)"),
    _p("cold4_thr", 0.98, 0.85, 0.999, "4bet", "percentile to cold 4-bet"),
    _p("cold_call_thr", 0.95, 0.75, 0.999, "4bet", "percentile to cold-call a 3-bet"),
    _p("fb_jam_bb", 40.0, 15.0, 120.0, "4bet", "4-bets are all-in at or below this depth"),
    # --- all-in and commitment decisions
    _p("shove_w_base", 0.25, 0.03, 0.8, "allin", "assumed shove range width (share of hands) of a deep shover"),
    _p("shove_w_slope", 0.35, 0.0, 0.8, "allin", "extra width as the shover's depth goes from 15bb to 0"),
    _p("pre_range_tax", 0.5, 0.0, 1.2, "allin", "equity lost to a narrow range, per (1 - width) and weak hand"),
    _p("allin_margin", 0.02, -0.08, 0.12, "allin", "equity above the price needed to call an all-in"),
    _p("icm_w", 1.0, 0.0, 2.5, "icm", "weight on the ICM risk premium when calling all-in"),
    _p("commit_frac", 0.4, 0.15, 0.9, "allin", "a call this large a share of the stack is a commitment decision"),
    _p("iso_allin_edge", 0.25, 0.05, 0.6, "allin", "equity edge over the price to re-raise all-in over a shove"),
    # --- postflop value and bluffs
    *(_p(f"vbet_thr_{s}", d, 0.4, 0.97, "post", f"equity to bet for value on the {s}")
      for s, d in zip(STREET_NAMES[1:], (0.70, 0.72, 0.74))),
    _p("multiway_pen", 0.02, -0.1, 0.15, "post", "value threshold added per extra opponent"),
    _p("cbet_dry_hu", 0.65, 0.0, 1.0, "bluff", "flop c-bet frequency, dry board, heads-up"),
    _p("cbet_wet_hu", 0.45, 0.0, 1.0, "bluff", "flop c-bet frequency, wet board, heads-up"),
    _p("cbet_dry_mw", 0.30, 0.0, 1.0, "bluff", "flop c-bet frequency, dry board, multiway"),
    _p("cbet_wet_mw", 0.20, 0.0, 1.0, "bluff", "flop c-bet frequency, wet board, multiway"),
    _p("wet_cut", 0.4, 0.05, 0.95, "bluff", "board dynamism at or above which a board counts as wet"),
    _p("barrel_turn", 0.40, 0.0, 1.0, "bluff", "turn barrel frequency after betting the flop"),
    _p("barrel_river", 0.30, 0.0, 1.0, "bluff", "river barrel frequency after betting the turn"),
    _p("river_bluff_max", 0.35, 0.05, 0.7, "bluff", "river bluffs only below this equity (value or bluff, not both)"),
    _p("semi_bonus", 0.25, 0.0, 0.8, "bluff", "bet frequency added per strong draw class (scaled to 0..1)"),
    _p("stab_freq", 0.25, 0.0, 1.0, "bluff", "bet frequency in position when checked to, without initiative"),
    _p("trap_thr", 0.92, 0.75, 0.995, "post", "equity at which an out-of-position hand may check to trap"),
    _p("trap_freq", 0.2, 0.0, 0.8, "post", "trap frequency"),
    *(_p(f"value_size_{s}", d, 0.25, 1.5, "size", f"value bet size on the {s}, pot fraction")
      for s, d in zip(STREET_NAMES[1:], (0.6, 0.66, 0.7))),
    _p("cbet_size_dry", 0.40, 0.2, 1.2, "size", "c-bet/bluff size on dry boards, pot fraction"),
    _p("cbet_size_wet", 0.60, 0.25, 1.5, "size", "c-bet/bluff size on wet boards, pot fraction"),
    _p("bluff_size", 0.60, 0.25, 1.5, "size", "turn and river bluff size, pot fraction"),
    _p("jam_spr", 1.2, 0.0, 4.0, "size", "value hands jam when chips behind are at most this many pots"),
    # --- postflop facing a bet
    *(_p(f"face_tax_{s}", d, 0.0, 0.5, "face", f"equity tax for facing a bet on the {s}")
      for s, d in zip(STREET_NAMES[1:], (0.18, 0.22, 0.28))),
    _p("raise_tax", 0.08, 0.0, 0.3, "face", "extra equity tax per raise beyond the first bet"),
    *(_p(f"raise_thr_{s}", d, 0.5, 0.99, "face", f"taxed equity to raise for value on the {s}")
      for s, d in zip(STREET_NAMES[1:], (0.75, 0.75, 0.78))),
    _p("raise_frac", 0.9, 0.4, 2.0, "size", "raise size: pot fraction added on top of the call"),
    _p("semi_raise_freq", 0.15, 0.0, 0.8, "face", "semi-bluff raise frequency with a strong draw"),
    _p("bluff_raise_freq", 0.04, 0.0, 0.4, "face", "bluff raise frequency against a single bet"),
    _p("call_margin", 0.02, -0.1, 0.15, "face", "taxed equity above pot odds needed to call"),
    _p("river_call_margin", 0.02, -0.1, 0.15, "face", "extra margin on the river"),
    _p("draw_call_bonus", 0.04, 0.0, 0.2, "face", "equity credit for a strong draw with cards to come"),
    # --- tournament pressure
    _p("icm_open", 0.6, 0.0, 2.5, "icm", "open/jam threshold added per unit of ICM risk premium"),
    _p("icm_call", 0.8, 0.0, 2.5, "icm", "calling threshold added per unit of risk premium vs the raiser"),
    _p("bully", 0.05, -0.1, 0.2, "icm", "open discount for a big stack near the money"),
    _p("cap_window", 30.0, 5.0, 80.0, "cap", "hands before the cap where cap urgency ramps in"),
    _p("cap_open_adj", 0.0, -0.15, 0.15, "cap", "open/jam threshold change at the cap"),
    _p("cap_call_adj", 0.0, -0.15, 0.15, "cap", "calling threshold change at the cap"),
    # --- bounded exploitation
    _p("exploit_prior_n", 15.0, 2.0, 80.0, "exploit", "pseudo-observations behind each opponent rate"),
    _p("exploit_bluff", 1.0, 0.0, 3.0, "exploit", "bluff frequency scale per unit of fold-to-bet over 0.45"),
    _p("exploit_call", 0.3, 0.0, 1.5, "exploit", "calling threshold drop per unit of bettor aggression over 0.35"),
    _p("exploit_steal", 0.3, 0.0, 1.5, "exploit", "late open threshold drop per unit of blinds' VPIP below 0.30"),
    _p("exploit_cap", 0.08, 0.0, 0.25, "exploit", "largest exploit adjustment (thresholds; x5 for frequency scales)"),
    # --- mixing
    _p("temp_pre", 0.02, 0.002, 0.1, "mix", "logistic temperature of preflop thresholds (percentile units)"),
    _p("temp_post", 0.03, 0.002, 0.15, "mix", "logistic temperature of postflop thresholds (equity units)"),
)

SPACE = ParamSpace(PARAMS)


@dataclass(frozen=True, slots=True)
class Intent:
    kind: str
    """fold | check | call | raise | allin."""
    amount: int | None = None
    """Raise-to target (street total) for ``raise``; the runtime clamps it to the legal interval."""
    tag: str = ""


FOLD = Intent("fold", tag="fold")
CHECK = Intent("check", tag="check")


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _sig(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


class BaselinePolicy:
    """``decide(spot, rand)``: ``rand()`` returns the next uniform draw of the seat's policy stream."""

    __slots__ = ("p", "open_thr", "jam_thr", "vbet", "vsize", "face", "rthr")

    def __init__(self, values: Mapping[str, float] | None = None) -> None:
        p = self.p = SPACE.complete(values)
        self.open_thr = tuple(p[f"open_thr_{g}"] for g in GROUPS)
        self.jam_thr = tuple(p[f"jam_thr_{g}"] for g in GROUPS)
        self.vbet = (0.0, *(p[f"vbet_thr_{s}"] for s in STREET_NAMES[1:]))
        self.vsize = (0.0, *(p[f"value_size_{s}"] for s in STREET_NAMES[1:]))
        self.face = (0.0, *(p[f"face_tax_{s}"] for s in STREET_NAMES[1:]))
        self.rthr = (0.0, *(p[f"raise_thr_{s}"] for s in STREET_NAMES[1:]))

    # ---------------------------------------------------------- helpers

    def _rate(self, s: Spot, seat: int, num: int, den: int, prior: float) -> float:
        n0 = self.p["exploit_prior_n"]
        return (s.stat(seat, num) + prior * n0) / (s.stat(seat, den) + n0)

    def _aggression(self, s: Spot, seat: int) -> float:
        n0 = self.p["exploit_prior_n"]
        a, q = s.stat(seat, AGGR), s.stat(seat, PASSIVE)
        return (a + PRIOR_AGGR * n0) / (a + q + n0)

    def _cap(self, x: float, scale: float = 1.0) -> float:
        c = self.p["exploit_cap"] * scale
        return _clip(x, -c, c)

    def _cap_urgency(self, s: Spot) -> float:
        w = self.p["cap_window"]
        return _clip((w - s.hands_left) / w, 0.0, 1.0)

    def _target_pot(self, s: Spot, frac: float) -> int:
        return int(s.current_bet_to + frac * (s.pot + s.to_call))

    def _raise(self, s: Spot, target: int, tag: str) -> Intent:
        if target >= self.p["allin_commit"] * (s.hero_stack + s.hero_street):
            return Intent("allin", tag=tag + ":allin")
        return Intent("raise", max(target, s.current_bet_to + s.bb), tag)

    # ----------------------------------------------------------- decide

    def decide(self, s: Spot, rand: Callable[[], float]) -> Intent:
        if s.to_call > 0 and (s.bettor_all_in or s.to_call >= self.p["commit_frac"] * s.hero_stack):
            return self._allin_call(s, rand)
        if s.street == 0:
            if s.pre_raises == 0:
                return self._unopened(s, rand)
            if s.pre_raises == 1 and not s.hero_raised_pre:
                return self._facing_open(s, rand)
            return self._facing_3bet(s, rand)
        if s.to_call == 0:
            return self._post_unbet(s, rand)
        return self._post_facing(s, rand)

    def _pass(self, x: float, thr: float, temp: float, rand: Callable[[], float]) -> bool:
        z = (x - thr) / temp
        if z > 12.0:
            return True
        if z < -12.0:
            return False
        return rand() < _sig(z)

    # ---------------------------------------------------------- preflop

    def _open_rp(self, s: Spot) -> float:
        """Risk premium against the biggest live stack (the open's worst confrontation)."""
        mem = s.memory
        if mem is None or not mem.live_opponents:
            return 0.0
        start = mem.ctx.start
        big = max(mem.live_opponents, key=lambda x: (start.get(x, 0), x))
        return s.clash(big)[1]

    def _unopened(self, s: Spot, rand: Callable[[], float]) -> Intent:
        p = self.p
        g, eff, pct = s.group, s.eff_bb, s.pct
        if g == BB_G:
            if s.to_call == 0:
                thr = p["bb_iso_thr"] + p["iso_pen"] * max(0, s.limpers - 1)
                if self._pass(pct, thr, p["temp_pre"], rand):
                    if eff <= p["jam_bb"]:
                        return Intent("allin", tag="bb_iso:jam")
                    return self._raise(s, int((p["open_bb"] + p["open_limper_bb"] * s.limpers + 1) * s.bb),
                                       "bb_iso")
                return CHECK
        rp = self._open_rp(s)
        adj = p["icm_open"] * rp + p["cap_open_adj"] * self._cap_urgency(s)
        if s.survivors - s.paid <= 2 and s.stack_share > 1.3:
            adj -= p["bully"]
        if s.ante_pot:
            adj -= ANTE_WIDEN * min(1.0, s.ante_pot / (1.5 * s.bb))
        if eff <= p["jam_bb"]:
            thr = self.jam_thr[g] - p["jam_slope"] * (1.0 - eff / p["jam_bb"]) + adj + p["iso_pen"] * s.limpers
            if self._pass(pct, thr, p["temp_pre"], rand):
                return Intent("allin", tag="open_jam")
            return FOLD
        thr = self.open_thr[g] + adj
        thr += p["open_deep_adj"] * _clip((eff - 25.0) / 50.0, 0.0, 1.0)
        thr += p["open_short_adj"] * _clip((20.0 - eff) / 10.0, 0.0, 1.0)
        thr += p["iso_pen"] * s.limpers
        if s.limpers == 0 and g in (LATE_G, SB_G) and s.behind <= 2:
            thr += p["steal_adj"] - self._cap(p["exploit_steal"] * (PRIOR_VPIP - self._behind_vpip(s)))
        if self._pass(pct, thr, p["temp_pre"], rand):
            bbs = p["sb_open_bb"] if g == SB_G else p["open_bb"] + p["open_limper_bb"] * s.limpers
            return self._raise(s, int(bbs * s.bb), "open")
        if s.limpers > 0 and pct >= thr - p["overlimp_band"]:
            return Intent("call", tag="overlimp")
        if g == SB_G and s.limpers == 0 and self._pass(pct, p["sb_complete_thr"], p["temp_pre"], rand):
            return Intent("call", tag="complete")
        return FOLD

    def _behind_vpip(self, s: Spot) -> float:
        mem = s.memory
        if mem is None:
            return PRIOR_VPIP
        order = mem.ctx.pre_order
        i = order.index(s.hero) if s.hero in order else len(order)
        rest = [x for x in order[i + 1:] if x in mem.live_opponents]
        if not rest:
            return PRIOR_VPIP
        return sum(self._rate(s, x, VPIP, HANDS, PRIOR_VPIP) for x in rest) / len(rest)

    def _facing_open(self, s: Spot, rand: Callable[[], float]) -> Intent:
        p = self.p
        pct, eff = s.pct, s.eff_bb
        price = s.to_call / max(1, s.hero_total)
        rp = s.clash(s.aggressor)[1]
        icm = p["icm_call"] * rp
        if eff <= p["resteal_bb"]:
            if self._pass(pct, p["reshove_thr"] + p["reshove_price"] * price + icm, p["temp_pre"], rand):
                return Intent("allin", tag="reshove")
        else:
            thr = p["tb_value_thr"] + (0.0 if s.in_position else p["tb_oop_adj"]) + 0.5 * icm
            mult = p["tb_size_ip"] if s.in_position else p["tb_size_oop"]
            target = int(s.current_bet_to * (mult + 0.5 * s.callers))
            if self._pass(pct, thr, p["temp_pre"], rand):
                return self._raise(s, target, "3bet")
            lo = p["tb_bluff_lo"]
            if s.callers == 0 and lo <= pct < lo + p["tb_bluff_width"]:
                fold = self._rate(s, s.aggressor, FOLDED, FACED, PRIOR_FOLD)
                freq = p["tb_bluff_freq"] * (1.0 + self._cap(p["exploit_bluff"] * (fold - PRIOR_FOLD), 5.0))
                if rand() < freq:
                    return self._raise(s, target, "3bet_bluff")
        ag = s.aggressor_group if s.aggressor_group >= 0 else MIDDLE_G
        thr = self.open_thr[min(ag, SB_G)] + p["gap_penalty"] + p["call_price_coef"] * price + icm
        thr += p["cap_call_adj"] * self._cap_urgency(s)
        if s.group == BB_G:
            thr -= p["bb_defend"]
        if s.in_position:
            thr -= p["ip_call_bonus"]
        thr -= p["multi_call_bonus"] * s.callers
        if self._pass(pct, thr, p["temp_pre"], rand):
            return Intent("call", tag="call_open")
        return FOLD

    def _facing_3bet(self, s: Spot, rand: Callable[[], float]) -> Intent:
        p = self.p
        pct = s.pct
        price = s.to_call / max(1, s.hero_total)
        icm = p["icm_call"] * s.clash(s.aggressor)[1]
        four, call = (p["fb_thr"], p["fb_call_thr"]) if s.hero_raised_pre else (p["cold4_thr"], p["cold_call_thr"])
        if self._pass(pct, four + icm, p["temp_pre"], rand):
            if s.eff_bb <= p["fb_jam_bb"]:
                return Intent("allin", tag="4bet_jam")
            return self._raise(s, int(s.current_bet_to * 2.3), "4bet")
        if self._pass(pct, call + p["fb_call_price"] * price + icm, p["temp_pre"], rand):
            return Intent("call", tag="call_3bet")
        return FOLD

    # ---------------------------------------------------------- all-in

    def _allin_call(self, s: Spot, rand: Callable[[], float]) -> Intent:
        p = self.p
        q, rp = s.clash(s.aggressor)
        if s.street == 0:
            mem = s.memory
            start = mem.ctx.start.get(s.aggressor, 0) if mem is not None else 0
            agg_bb = start / max(1, s.bb)
            w = p["shove_w_base"] + p["shove_w_slope"] * _clip((15.0 - agg_bb) / 15.0, 0.0, 1.0)
            if s.aggressor >= 0:
                vpip = self._rate(s, s.aggressor, VPIP, HANDS, PRIOR_VPIP)
                w += self._cap(vpip - PRIOR_VPIP, 3.0)
            w = _clip(w, 0.03, 1.0)
            eq = s.eq - p["pre_range_tax"] * (1.0 - w) * min(1.0 - s.pct, 0.3)
        else:
            eq = s.eq - self.face[s.street] - p["raise_tax"] * max(0, s.street_raises - 1)
            eq += self._cap(p["exploit_call"] * (self._aggression(s, s.aggressor) - PRIOR_AGGR))
        req = s.pot_odds + p["icm_w"] * rp + p["allin_margin"]
        temp = p["temp_pre"] if s.street == 0 else p["temp_post"]
        if s.active_opponents > 0 and not s.bettor_all_in and eq - req > p["iso_allin_edge"]:
            return Intent("allin", tag="iso_allin")
        if self._pass(eq, req, temp, rand):
            return Intent("call", tag="call_allin")
        return FOLD

    # --------------------------------------------------------- postflop

    def _fold_rate(self, s: Spot) -> float:
        mem = s.memory
        seats = mem.live_opponents if mem is not None else ()
        if not seats:
            return PRIOR_FOLD
        return sum(self._rate(s, x, FOLDED, FACED, PRIOR_FOLD) for x in seats) / len(seats)

    def _post_unbet(self, s: Spot, rand: Callable[[], float]) -> Intent:
        p = self.p
        st, eq, k = s.street, s.eq, max(1, s.opponents)
        if eq >= p["trap_thr"] and not s.in_position and rand() < p["trap_freq"]:
            return Intent("check", tag="trap")
        if self._pass(eq, self.vbet[st] + p["multiway_pen"] * (k - 1), p["temp_post"], rand):
            if s.spr <= p["jam_spr"]:
                return Intent("allin", tag="value_jam")
            return self._raise(s, self._target_pot(s, self.vsize[st]), "value")
        wet = s.wet >= p["wet_cut"]
        if s.initiative:
            if st == 1:
                key = ("cbet_wet_" if wet else "cbet_dry_") + ("hu" if k == 1 else "mw")
                freq = p[key]
            elif st == 2:
                freq = p["barrel_turn"]
            else:
                freq = p["barrel_river"] if eq < p["river_bluff_max"] else 0.0
        elif s.in_position and s.street_raises == 0:
            freq = p["stab_freq"] if st < 3 or eq < p["river_bluff_max"] else 0.0
        else:
            freq = 0.0
        if st < 3:
            freq += p["semi_bonus"] * s.draw_class / 4.0
        freq *= 1.0 + self._cap(p["exploit_bluff"] * (self._fold_rate(s) - PRIOR_FOLD), 5.0)
        if freq > 0.0 and rand() < min(0.95, freq):
            size = (p["cbet_size_wet"] if wet else p["cbet_size_dry"]) if st == 1 else p["bluff_size"]
            return self._raise(s, self._target_pot(s, size), "bluff" if st == 3 else "cbet" if s.initiative else "stab")
        return CHECK

    def _post_facing(self, s: Spot, rand: Callable[[], float]) -> Intent:
        p = self.p
        st = s.street
        tax = self.face[st] + p["raise_tax"] * max(0, s.street_raises - 1)
        tax -= self._cap(p["exploit_call"] * (self._aggression(s, s.aggressor) - PRIOR_AGGR))
        eq = s.eq - tax
        if st < 3 and s.draw_class >= 2:
            eq += p["draw_call_bonus"]
        if s.active_opponents > 0:
            if self._pass(eq, self.rthr[st], p["temp_post"], rand):
                if s.spr <= p["jam_spr"]:
                    return Intent("allin", tag="raise_jam")
                return self._raise(s, self._target_pot(s, p["raise_frac"]), "raise_value")
            if st < 3 and s.draw_class >= 2 and rand() < p["semi_raise_freq"]:
                return self._raise(s, self._target_pot(s, p["raise_frac"]), "raise_semi")
            if s.street_raises == 1 and rand() < p["bluff_raise_freq"]:
                return self._raise(s, self._target_pot(s, p["raise_frac"]), "raise_bluff")
        margin = p["call_margin"] + (p["river_call_margin"] if st == 3 else 0.0)
        margin += p["icm_call"] * s.clash(s.aggressor)[1] + p["cap_call_adj"] * self._cap_urgency(s)
        if self._pass(eq, s.pot_odds + margin, p["temp_post"], rand):
            return Intent("call", tag="call")
        return FOLD
