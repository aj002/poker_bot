"""The opponent families of the self-play plan, section 4, built on the six style bots.

Families 1-6 generalise the six anchors in :mod:`arena.agents.styles` (rock, aggressive,
super-aggressive, station, weak-tight, switcher): the anchor's rule table plays every
decision, with its numbers and ranges replaced by sampled knobs. Families 7-17 add one
behaviour each (a hook or a size rewrite from :mod:`strategy.population.behaviours`) over
an anchor. Family 18, frozen learner checkpoints and trained approximate responses, is a
registry slot type (:class:`strategy.population.registry.CheckpointSlot`), not a sampled
rule family.

``expected`` gives each family's declared range for statistics measured in the official
format (eight seats, 2,000 chips, 10/20 rising every 25 hands, 200-hand cap) over many
sampled members; ``tests/strategy/population`` checks them. ``split="holdout_only"``
families never appear in train or validation pools.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from arena.agents.styles.engine import Decision, decide_gear, decide_spot
from strategy.population import behaviours as b
from strategy.population.cards import top
from strategy.population.params import Param, Prior, choice
from strategy.population.policy import Ctx

Hook = Callable[[Ctx], Decision | None]
Post = Callable[[Ctx, Any], Any]
Range = tuple[float, float]


@dataclass(frozen=True)
class Family:
    id: str
    number: int
    """The family's number in the plan's section 4 list."""
    title: str
    anchor: str
    """The style from :mod:`arena.agents.styles` whose rules play every decision no hook takes."""
    summary: str
    prior: Prior
    knobs: Callable[[Mapping[str, float]], dict[str, Any]]
    hooks: tuple[Hook, ...] = ()
    post: tuple[Post, ...] = ()
    gear: Callable[[Ctx], str] | None = None
    regime: Mapping[str, float] = field(default_factory=dict)
    """Table-local regime: a shift in logit units applied from a hand drawn at table start."""
    regime_p: float = 0.25
    regime_doc: str = ""
    hot: Callable[[Any, dict[str, Any], Mapping[str, float]], bool] | None = None
    hot_shift: Mapping[str, float] = field(default_factory=dict)
    card_noise: Callable | None = None
    observe_fn: Callable[[Any, dict[str, Any], Mapping[str, float]], None] | None = None
    state0: Mapping[str, Any] = field(default_factory=dict)
    expected: Mapping[str, Range] = field(default_factory=dict)
    split: Literal["all", "holdout_only"] = "all"

    def base(self, c: Ctx) -> Decision:
        if self.gear is not None:
            return decide_gear(self.gear(c), c.s, c.ch)
        return decide_spot(self.anchor, c.s, c.ch)

    def initial_state(self) -> dict[str, Any]:
        return {k: (dict(v) if isinstance(v, dict) else v) for k, v in self.state0.items()}


def chart(early: float, middle: float, late: float) -> dict[str, frozenset[str]]:
    return {"early": top(early), "middle": top(middle), "late": top(late)}


def P(name: str, lo: float, hi: float, centre: float, spread: float = 0.6, doc: str = "") -> Param:
    return Param(name, lo, hi, centre, spread, doc)


# Shared parameter pieces ------------------------------------------------------------

def _open(centre: float, lo: float, hi: float) -> Param:
    return P("open_early", lo, hi, centre, 0.5, "top fraction of hands opened from early position")


def _agg_knobs(p: Mapping[str, float], prefix: str = "agg") -> dict[str, Any]:
    e = p["open_early"]
    late = p.get("late_mult", 1.45)
    return {f"{prefix}.chart": chart(e, e * (1 + (late - 1) * 0.35), e * late),
            "aggp.cbet": p.get("cbet", 0.75), "aggp.barrel": p.get("barrel", 0.4)}


# 1-6: the anchors as families ---------------------------------------------------------

def _selective(p: Mapping[str, float]) -> dict[str, Any]:
    e, rr = p["open_early"], p["reraise_top"]
    return {
        "rock.chart": chart(e, e * p["open_slope"], e * p["open_slope"] * p["late_mult"]),
        "rock.open_mult": p["open_mult"],
        "rock.reraise": top(rr), "rock.call": top(rr + p["call_band"]) - top(rr),
        "rock.call_price": p["call_price"],
        "rock.jam_3b": top(rr * 0.35), "rock.call_3b": top(rr) - top(rr * 0.35),
        "rockp.cbet_air": p["cbet_air"], "rockp.cbet_medium": p["cbet_medium"],
        "rockp.value_frac": p["value_frac"], "rockp.medium_call": p["medium_call"],
    }


def _wide(p: Mapping[str, float]) -> dict[str, Any]:
    k = _agg_knobs(p)
    k.update({"agg.reraise": top(p["reraise_top"]), "agg.light_reraise_p": p["light_reraise_p"],
              "agg.call": top(p["call_top"]), "agg.call_price": p["call_price"],
              "aggp.semi": p["semi"], "aggp.fight": p["fight"], "aggp.cbet_frac": p["cbet_frac"]})
    return k


_STEAL = {"BTN": 0.85, "CO": 0.6, "SB": 0.7}


def _chaotic(p: Mapping[str, float]) -> dict[str, Any]:
    sc = p["steal_scale"]
    return {"sup.steal": {k: min(1.0, v * sc) for k, v in _STEAL.items()}, "sup.loose_p": p["loose_p"],
            "sup.light_reraise_p": p["light_reraise_p"], "sup.light_jam_p": p["light_jam_p"],
            "sup.reraise_mult": p["raise_mult"], "supp.raise_mult": p["raise_mult"], "raise_mult": p["raise_mult"],
            "jam_frac": p["jam_frac"], "bet_scale": p["bet_scale"], "supp.cbet": p["cbet"],
            "supp.barrel": p["barrel"], "supp.bluff_raise": p["bluff_raise"]}


def _loose_passive(p: Mapping[str, float]) -> dict[str, Any]:
    e = p["chart_early"]
    return {"sta.chart": chart(e, e, e * p["late_mult"]), "sta.raise_p": p["raise_p"],
            "sta.call_price": p["call_price"], "sta.call_3b_price": p["call_3b_price"],
            "stap.weak_call": p["weak_call"], "stap.weak_price": p["weak_price"],
            "stap.bet_monster": p["bet_monster"], "stap.bet_strong": p["bet_monster"] * 0.4,
            "stap.raise_p": p["post_raise_p"], "stap.draw_outs": int(round(p["draw_outs"]))}


def _surrender(p: Mapping[str, float]) -> dict[str, Any]:
    e = p["chart_early"]
    return {"wt.chart": chart(e, e * p["mid_mult"], e * p["mid_mult"] * p["late_mult"]), "wt.raise_p": p["raise_p"],
            "wt.open_mult": p["open_mult"], "wt.call_price": p["call_price"], "wt.call": top(0.012 + p["call_band"]),
            "wtp.strong_call": p["strong_call"], "wtp.medium_call": p["medium_call"],
            "wtp.bet_monster": p["bet_monster"]}


def _switcher(p: Mapping[str, float]) -> dict[str, Any]:
    e = p["open_early"]
    sc = p["steal_scale"]
    return {"rock.chart": chart(e, e * 1.65, e * 4.1), "sup.steal": {k: min(1.0, v * sc) for k, v in _STEAL.items()},
            "pf.chart": chart(p["push_early"], p["push_early"] * 1.9, p["push_early"] * 3.4)}


# 7-17: anchors plus one behaviour ----------------------------------------------------

def _plain_agg(p: Mapping[str, float]) -> dict[str, Any]:
    return _agg_knobs(p)


def _odd(p: Mapping[str, float]) -> dict[str, Any]:
    sc = p["steal_scale"]
    return {"sup.steal": {k: min(1.0, v * sc) for k, v in _STEAL.items()}}


FAMILY_LIST: tuple[Family, ...] = (
    Family(
        id="selective_aggressor", number=1, title="Selective first-in aggressor", anchor="rock",
        summary="Tight, raise-or-fold entry with a narrow early range widening late; rarely calls a raise.",
        prior=Prior("selective_aggressor", (
            _open(0.06, 0.03, 0.14),
            P("open_slope", 1.2, 2.6, 1.65), P("late_mult", 1.5, 3.4, 2.5), P("open_mult", 2.2, 3.6, 3.0),
            P("reraise_top", 0.012, 0.05, 0.026), P("call_band", 0.015, 0.07, 0.04),
            P("call_price", 0.04, 0.14, 0.08), P("cbet_air", 0.1, 0.6, 0.3), P("cbet_medium", 0.35, 0.9, 0.6),
            P("value_frac", 0.4, 0.9, 0.6), P("medium_call", 0.3, 0.9, 0.6),
        ), (("open_early", "reraise_top", 0.4), ("open_early", "cbet_air", 0.35), ("open_mult", "value_frac", 0.3),
            ("call_band", "medium_call", 0.3))),
        knobs=_selective, regime={"open_early": 1.0, "cbet_air": 1.0}, regime_doc="opens up: wider and bluffier",
        expected={"vpip": (0.07, 0.22), "pfr": (0.06, 0.21), "limp": (0.0, 0.02), "three_bet": (0.005, 0.06),
                  "cold_call": (0.0, 0.05), "fold_to_reraise": (0.6, 1.0), "hu_bluff": (0.0, 0.25)},
    ),
    Family(
        id="wide_pressure", number=2, title="Wide pressure opener", anchor="aggressive",
        summary="High raise-first rate, frequent continuation bets and barrels, light reraises.",
        prior=Prior("wide_pressure", (
            _open(0.26, 0.15, 0.40), P("late_mult", 1.0, 1.8, 1.4),
            P("reraise_top", 0.03, 0.08, 0.053), P("light_reraise_p", 0.05, 0.4, 0.15),
            P("call_top", 0.08, 0.2, 0.13), P("call_price", 0.06, 0.18, 0.12),
            P("cbet", 0.55, 0.97, 0.78), P("barrel", 0.2, 0.75, 0.42), P("semi", 0.1, 0.55, 0.3),
            P("fight", 0.1, 0.6, 0.3), P("cbet_frac", 0.4, 0.9, 0.6),
        ), (("open_early", "cbet", 0.45), ("cbet", "barrel", 0.55), ("open_early", "light_reraise_p", 0.3),
            ("barrel", "fight", 0.3))),
        knobs=_wide, regime={"barrel": -1.2, "cbet": -1.0}, regime_doc="gets caught and stops barrelling",
        expected={"vpip": (0.15, 0.34), "pfr": (0.13, 0.3), "aggression": (3.0, 14.0), "cbet": (0.7, 1.0),
                  "hu_bluff": (0.15, 0.5)},
    ),
    Family(
        id="chaotic_aggressor", number=3, title="Chaotic over-aggressor", anchor="super_aggressive",
        summary="Very wide opens and reraises, big and varied sizes, frequent all-ins; persistent, tilting or "
                "stack-triggered.",
        prior=Prior("chaotic_aggressor", (
            P("steal_scale", 0.6, 1.15, 0.95), P("loose_p", 0.25, 0.85, 0.5),
            P("light_reraise_p", 0.12, 0.6, 0.3), P("light_jam_p", 0.1, 0.6, 0.3),
            P("raise_mult", 2.5, 5.5, 3.2), P("jam_frac", 0.15, 0.45, 0.35), P("bet_scale", 0.7, 1.8, 1.1),
            P("cbet", 0.65, 0.98, 0.85), P("barrel", 0.35, 0.85, 0.55), P("bluff_raise", 0.05, 0.35, 0.15),
            choice("mode", ("persistent", "tilt", "stack-triggered")),
            P("tilt_loss", 0.2, 0.6, 0.35), P("tilt_hands", 3, 20, 8),
        ), (("loose_p", "light_reraise_p", 0.4), ("raise_mult", "bet_scale", 0.4), ("cbet", "barrel", 0.5),
            ("jam_frac", "light_jam_p", -0.3))),
        knobs=_chaotic, regime={"loose_p": -1.5, "light_reraise_p": -1.5}, regime_doc="calms down",
        hot=b.chaotic_hot, hot_shift={"loose_p": 1.2, "light_reraise_p": 1.2, "light_jam_p": 1.0,
                                      "bluff_raise": 1.0, "jam_frac": -0.8},
        observe_fn=b.chaotic_observe,
        expected={"vpip": (0.3, 0.5), "pfr": (0.24, 0.44), "three_bet": (0.09, 0.25), "junk_entry": (0.2, 0.45),
                  "hu_bluff": (0.38, 0.7)},
    ),
    Family(
        id="loose_passive", number=4, title="Loose passive caller", anchor="station",
        summary="Limps and calls a wide range, sticky with pairs and draws, rarely raises.",
        prior=Prior("loose_passive", (
            P("chart_early", 0.25, 0.6, 0.42), P("late_mult", 1.0, 1.35, 1.13), P("raise_p", 0.05, 0.6, 0.3),
            P("call_price", 0.12, 0.4, 0.25), P("call_3b_price", 0.2, 0.5, 0.35), P("weak_call", 0.2, 0.75, 0.45),
            P("weak_price", 0.5, 1.3, 0.75), P("bet_monster", 0.15, 0.7, 0.4), P("post_raise_p", 0.03, 0.4, 0.2),
            P("draw_outs", 2, 8, 4),
        ), (("chart_early", "weak_call", 0.4), ("weak_call", "weak_price", 0.4), ("bet_monster", "post_raise_p", 0.3))),
        knobs=_loose_passive, regime={"chart_early": -1.2, "weak_call": -1.2}, regime_doc="tightens up",
        expected={"vpip": (0.28, 0.52), "pfr": (0.0, 0.02), "aggression": (0.0, 0.15), "limp": (0.3, 0.65),
                  "cold_call": (0.3, 0.7), "draw_continue": (0.8, 1.0), "strong_check": (0.55, 0.9)},
    ),
    Family(
        id="tight_surrender", number=5, title="Tight surrenderer", anchor="weak_tight",
        summary="Low entry, small raises, folds to almost any reraise; strong actions stay honest.",
        prior=Prior("tight_surrender", (
            P("chart_early", 0.025, 0.1, 0.057), P("mid_mult", 1.2, 2.0, 1.6), P("late_mult", 1.2, 2.2, 1.6),
            P("raise_p", 0.2, 0.85, 0.5), P("open_mult", 2.0, 3.0, 2.5), P("call_price", 0.03, 0.12, 0.08),
            P("call_band", 0.02, 0.08, 0.05), P("strong_call", 0.4, 1.0, 0.7), P("medium_call", 0.15, 0.6, 0.4),
            P("bet_monster", 0.4, 0.95, 0.7), P("river_bluff", 0.001, 0.15, 0.02),
        ), (("chart_early", "call_band", 0.4), ("strong_call", "medium_call", 0.5))),
        knobs=_surrender, hooks=(b.rare_bluff,), regime={"chart_early": 1.0, "medium_call": 1.0},
        regime_doc="gets braver",
        expected={"vpip": (0.04, 0.13), "pfr": (0.025, 0.08), "fold_to_reraise": (0.6, 1.0),
                  "strong_check": (0.35, 0.75), "hu_bluff": (0.0, 0.04), "big_call": (0.02, 0.12)},
    ),
    Family(
        id="smooth_switcher", number=6, title="Smooth stack/horizon switcher", anchor="switcher",
        summary="Moves between conservative, pressure and push/fold gears through soft gates on M, the next blind "
                "level and the hands left to the cap.",
        prior=Prior("smooth_switcher", (
            P("m_push", 4, 10, 7), P("m_aggr", 10, 24, 15), P("soft", 0.5, 5, 2),
            P("horizon", 0.01, 0.99, 0.5), P("cap_urgency", 0.01, 0.99, 0.4),
            _open(0.057, 0.035, 0.1), P("steal_scale", 0.6, 1.1, 0.9), P("push_early", 0.06, 0.16, 0.097),
        ), (("m_push", "m_aggr", 0.5), ("m_push", "push_early", 0.3))),
        knobs=_switcher, gear=b.smooth_gear, regime={"m_aggr": -1.0, "m_push": -1.0},
        regime_doc="becomes patient: lower gates",
        expected={"vpip": (0.13, 0.25), "pfr": (0.11, 0.23), "junk_entry": (0.08, 0.3), "short_allin": (0.45, 0.85)},
    ),
    Family(
        id="limp_iso", number=7, title="Limp / over-limp / isolation player", anchor="aggressive",
        summary="Limps a wide range, over-limps behind limpers, isolates them with strong hands and limp-reraises "
                "premiums: creates squeeze and sandwich spots.",
        prior=Prior("limp_iso", (
            P("limp_width", 0.15, 0.5, 0.3), P("open_top", 0.015, 0.08, 0.035), P("limp_trap", 0.02, 0.6, 0.25),
            P("open_mult", 2.5, 4.0, 3.0), P("iso_top", 0.05, 0.2, 0.1), P("iso_mult", 3.0, 5.5, 4.0),
            P("overlimp_width", 0.12, 0.5, 0.28), P("limp_reraise_top", 0.01, 0.05, 0.025),
            P("limp_call", 0.06, 0.3, 0.14), P("limp_call_price", 0.04, 0.2, 0.1),
            _open(0.2, 0.1, 0.3), P("cbet", 0.4, 0.85, 0.6),
        ), (("limp_width", "overlimp_width", 0.6), ("iso_top", "iso_mult", 0.3),
            ("limp_trap", "limp_reraise_top", 0.3))),
        knobs=_plain_agg, hooks=(b.limp_iso,), regime={"limp_width": -1.5, "open_top": 1.0},
        regime_doc="stops limping, raises first",
        expected={"vpip": (0.15, 0.3), "pfr": (0.015, 0.07), "limp": (0.17, 0.42), "iso": (0.04, 0.22),
                  "aggression": (0.25, 0.9)},
    ),
    Family(
        id="squeezer", number=8, title="Squeezer", anchor="aggressive",
        summary="An ordinary opener that reraises far more after a raise and a call, and gives up to a four-bet.",
        prior=Prior("squeezer", (
            _open(0.18, 0.1, 0.28), P("late_mult", 1.1, 1.8, 1.4), P("squeeze_top", 0.06, 0.2, 0.11),
            P("light_squeeze", 0.05, 0.4, 0.18), P("squeeze_mult", 3.0, 4.5, 3.5), P("late_bonus", 0.01, 1.0, 0.5),
            P("keep_vs_4bet", 0.008, 0.04, 0.02), P("cbet", 0.5, 0.9, 0.7),
        ), (("squeeze_top", "light_squeeze", 0.4), ("squeeze_top", "late_bonus", 0.2))),
        knobs=_plain_agg, hooks=(b.squeeze,), regime={"squeeze_top": -1.5, "light_squeeze": -1.5},
        regime_doc="stops squeezing",
        expected={"vpip": (0.12, 0.24), "pfr": (0.1, 0.22), "squeeze": (0.09, 0.3), "three_bet": (0.03, 0.1)},
    ),
    Family(
        id="trapper", number=9, title="Trapper / slow-player", anchor="aggressive",
        summary="Flats premiums, checks strong hands where a bet is expected, and check-raises; may abandon traps "
                "on wet boards.",
        prior=Prior("trapper", (
            _open(0.16, 0.1, 0.25), P("flat_top", 0.008, 0.04, 0.02), P("flat_p", 0.4, 0.95, 0.75),
            P("trap_p", 0.3, 0.9, 0.6), P("cr_p", 0.4, 0.95, 0.75), P("abandon", 0.01, 0.99, 0.5),
            P("slowplay_call", 0.2, 0.8, 0.5), P("cr_mult", 2.5, 4.0, 3.0),
        ), (("trap_p", "cr_p", 0.4), ("flat_p", "trap_p", 0.4))),
        knobs=_plain_agg, hooks=(b.trap,), regime={"trap_p": -1.5, "cr_p": -1.0}, regime_doc="abandons traps",
        expected={"vpip": (0.11, 0.23), "pfr": (0.08, 0.2), "strong_check": (0.25, 0.7), "xr_share": (0.25, 0.75)},
    ),
    Family(
        id="texture_cbettor", number=10, title="Texture-aware continuation bettor", anchor="aggressive",
        summary="Continuation-bets and barrels by board texture and field size instead of one global frequency.",
        prior=Prior("texture_cbettor", (
            _open(0.2, 0.12, 0.3), P("cbet_dry", 0.6, 0.98, 0.85), P("cbet_wet", 0.15, 0.6, 0.35),
            P("cbet_paired", 0.45, 0.95, 0.7), P("cbet_monotone", 0.1, 0.55, 0.3), P("mw_mult", 0.15, 0.7, 0.4),
            P("size_dry", 0.25, 0.5, 0.33), P("size_wet", 0.55, 1.0, 0.75), P("barrel", 0.2, 0.6, 0.4),
            P("scare", 0.01, 0.45, 0.2), P("delayed", 0.15, 0.7, 0.4),
        ), (("cbet_dry", "cbet_paired", 0.5), ("cbet_wet", "cbet_monotone", 0.5), ("barrel", "delayed", 0.3))),
        knobs=_plain_agg, hooks=(b.texture_cbet,), regime={"cbet_wet": 1.5, "mw_mult": 1.0},
        regime_doc="goes on autopilot: c-bets wet boards too",
        expected={"vpip": (0.14, 0.27), "pfr": (0.12, 0.25), "cbet_dry_hu": (0.7, 1.0), "cbet_wet": (0.3, 0.7)},
    ),
    Family(
        id="draw_player", number=11, title="Draw chaser / draw attacker", anchor="aggressive",
        summary="Chases draws at poor prices or attacks with them (semi-bluffs, river bluffs after a miss), by a "
                "sampled attack level.",
        prior=Prior("draw_player", (
            P("attack", 0.02, 0.98, 0.5, 1.4), P("price_tol", 0.4, 1.6, 0.9), P("semi", 0.2, 0.8, 0.5),
            P("draw_frac", 0.4, 1.0, 0.65), P("river_bluff", 0.05, 0.6, 0.3), P("bluff_frac", 0.5, 1.2, 0.75),
            _open(0.2, 0.12, 0.3),
        ), (("attack", "river_bluff", 0.6), ("attack", "price_tol", -0.4))),
        knobs=_plain_agg, hooks=(b.draws,), regime={"attack": -2.0}, regime_doc="stops attacking, only chases",
        expected={"vpip": (0.14, 0.27), "pfr": (0.12, 0.25), "draw_continue": (0.45, 0.85), "draw_aggr": (0.15, 0.7)},
    ),
    Family(
        id="sizing_tells", number=12, title="Sizing-tell bot", anchor="aggressive",
        summary="Bet size rises with strength (positive tell), falls with it (negative) or stays flat (absent).",
        prior=Prior("sizing_tells", (
            choice("tell", ("negative", "absent", "positive")), P("amp", 0.3, 1.0, 0.6),
            P("base_frac", 0.4, 0.9, 0.6), P("open_mult", 2.2, 3.5, 2.8), _open(0.2, 0.12, 0.3),
        )),
        knobs=_plain_agg, post=(b.sizing_tell,), regime={"amp": -1.5}, regime_doc="the tell fades",
        expected={"vpip": (0.13, 0.27), "pfr": (0.11, 0.24), "size_cv": (0.2, 0.6)},
    ),
    Family(
        id="image_responsive", number=13, title="Image-responsive opponent", anchor="aggressive",
        summary="Remembers how often each opponent opens and reraises loose openers wider, tight ones narrower.",
        prior=Prior("image_responsive", (
            P("memory", 0.8, 0.98, 0.92), P("amp", 0.3, 1.6, 0.8), P("baseline", 0.1, 0.3, 0.18),
            P("rr_top", 0.03, 0.08, 0.05), P("call_top", 0.08, 0.2, 0.12), _open(0.16, 0.1, 0.25),
        ), (("rr_top", "call_top", 0.4),)),
        knobs=_plain_agg, hooks=(b.image_response,), observe_fn=b.image_observe, state0={"img": {}, "seq": -1},
        regime={"amp": -1.5}, regime_doc="stops adjusting",
        expected={"vpip": (0.1, 0.22), "pfr": (0.09, 0.2), "three_bet": (0.015, 0.09), "junk_entry": (0.0, 0.02)},
        split="holdout_only",
    ),
    Family(
        id="icm_cap", number=14, title="ICM / cap specialist", anchor="aggressive",
        summary="Adds a tournament risk premium to big calls, pressures as the big stack, jams short near the cap; "
                "the premium is calibrated, overweighted, ignored or inverted.",
        prior=Prior("icm_cap", (
            P("aversion", 0.2, 1.0, 0.7), choice("calibration", ("correct", "overweighted", "ignored", "inverted")),
            P("commit_frac", 0.2, 0.6, 0.33), P("bigstack", 0.01, 0.99, 0.4), P("short_jam", 0.15, 0.5, 0.3),
            P("cap_window", 10, 60, 30), P("cap_preserve", 0.01, 1.5, 0.6), P("cap_urgency", 0.01, 0.99, 0.4),
            _open(0.13, 0.08, 0.22),
        ), (("aversion", "cap_preserve", 0.4), ("bigstack", "short_jam", 0.3))),
        knobs=_plain_agg, hooks=(b.icm_cap,), regime={"aversion": 1.0}, regime_doc="tightens near the money",
        expected={"vpip": (0.11, 0.23), "pfr": (0.08, 0.2), "short_allin": (0.9, 1.0)},
    ),
    Family(
        id="multiway_specialist", number=15, title="Multiway / side-pot specialist", anchor="aggressive",
        summary="Overcalls in multiway pots, suppresses multiway bluffs, demands more for multiway value, and "
                "checks when an all-in player cannot be bluffed out.",
        prior=Prior("multiway_specialist", (
            P("overcall_pre", 0.08, 0.35, 0.18), P("overcall", 0.3, 1.0, 0.6), P("mw_bluff", 0.001, 0.15, 0.04),
            P("mw_value", 0.01, 0.99, 0.6), P("sidepot_aware", 0.5, 0.999, 0.9), _open(0.17, 0.1, 0.26),
        ), (("overcall_pre", "overcall", 0.4),)),
        knobs=_plain_agg, hooks=(b.multiway,), regime={"mw_bluff": 2.0}, regime_doc="starts bluffing multiway",
        expected={"vpip": (0.12, 0.24), "pfr": (0.1, 0.22), "mw_bluff": (0.0, 0.1), "squeeze": (0.0, 0.04)},
    ),
    Family(
        id="odd_sizer", number=16, title="Odd legal sizer / all-in-heavy player", anchor="super_aggressive",
        summary="Min-raises, overbets, irregular amounts and frequent all-ins: tests off-tree translation.",
        prior=Prior("odd_sizer", (
            P("allin_rate", 0.06, 0.4, 0.18), P("minraise_rate", 0.05, 0.35, 0.15),
            P("overbet_rate", 0.05, 0.35, 0.15), P("jitter", 0.1, 0.6, 0.3), P("jam_width", 0.05, 0.25, 0.12),
            P("steal_scale", 0.5, 1.0, 0.8),
        ), (("allin_rate", "jam_width", 0.4),)),
        knobs=_odd, hooks=(b.jam_heavy,), post=(b.odd_size,), regime={"allin_rate": -1.5, "jitter": -1.5},
        regime_doc="sizes normally",
        expected={"vpip": (0.3, 0.5), "pfr": (0.24, 0.42), "min_raise_share": (0.04, 0.25), "size_cv": (0.5, 1.3),
                  "allin_share": (0.12, 0.32), "junk_entry": (0.2, 0.45)},
        split="holdout_only",
    ),
    Family(
        id="variant_noisy", number=17, title="Variant-noisy transfer bot", anchor="aggressive",
        summary="Keeps generic stack and price behaviour but misreads card strength: a noisy starting-hand rank, "
                "shifted post-flop buckets and miscounted outs.",
        prior=Prior("variant_noisy", (
            P("card_noise", 0.04, 0.3, 0.12), P("bucket_flip", 0.05, 0.4, 0.18), _open(0.15, 0.08, 0.25),
            P("cbet", 0.4, 0.85, 0.65),
        ), (("card_noise", "bucket_flip", 0.5),)),
        knobs=_plain_agg, card_noise=b.variant_noise, regime={"card_noise": 1.0}, regime_doc="gets noisier",
        expected={"vpip": (0.12, 0.24), "pfr": (0.1, 0.22), "junk_entry": (0.02, 0.14)},
    ),
)


# HandHQ-calibrated families ------------------------------------------------------------
# Fitted to real six-max players (PokerStars 2009, docs/handhq-population-report.md) and
# registered only by the train/validation extension in strategy.population.handhq, so the
# pop-v1 registry, its lock and its holdout are unchanged.

def _handhq_regular(p: Mapping[str, float]) -> dict[str, Any]:
    k = _wide(p)
    k.update({"agg.open_mult": p["open_mult"], "aggp.medium_call": p["medium_call"], "aggp.barrel": p["barrel"]})
    k.update(_handhq_common(p))
    return k


def _handhq_common(p: Mapping[str, float]) -> dict[str, Any]:
    return {"agg.call_3b": top(p["call_3b_top"]), "agg.jam_3b": top(min(p["call_3b_top"], 0.02)),
            "aggp.thin_value": p["thin_value"], "aggp.strong_raise": p["strong_raise"],
            "aggp.draw_bet": p["draw_bet"]}


def _handhq_recreational(p: Mapping[str, float]) -> dict[str, Any]:
    return {"agg.reraise": top(p["reraise_top"]), "agg.light_reraise_p": 0.02, "agg.call": top(p["call_top"]),
            "agg.call_price": p["call_price"], "aggp.cbet": p["cbet"], "aggp.barrel": p["barrel"],
            "aggp.medium_call": p["medium_call"], "aggp.semi": 0.15, "aggp.fight": p["fight"],
            **_handhq_common(p)}


HANDHQ_FAMILY_LIST: tuple[Family, ...] = (
    Family(
        id="handhq_regular", number=19, title="Real-player regular (HandHQ)", anchor="aggressive",
        summary="A 2009 online six-max regular: raise-first or fold, flats some raises, continuation-bets about "
                "two flops in three and calls down more than the style bots; knobs fitted to the HandHQ players "
                "with VPIP 14-30% whose PFR is at least 60% of their VPIP.",
        prior=Prior("handhq_regular", (
            _open(0.13, 0.07, 0.24), P("late_mult", 1.4, 2.6, 1.9), P("open_mult", 2.5, 4.0, 3.2),
            P("reraise_top", 0.02, 0.07, 0.04), P("light_reraise_p", 0.02, 0.3, 0.1),
            P("call_top", 0.08, 0.26, 0.16), P("call_price", 0.06, 0.16, 0.1),
            P("cbet", 0.3, 0.8, 0.5), P("barrel", 0.1, 0.6, 0.28), P("semi", 0.05, 0.4, 0.15),
            P("fight", 0.15, 0.6, 0.35), P("cbet_frac", 0.45, 0.9, 0.65), P("medium_call", 0.5, 1.4, 0.9),
            P("sticky", 0.1, 0.75, 0.55), P("sticky_price", 0.3, 0.9, 0.6), P("stab", 0.02, 0.25, 0.08),
            P("stab_frac", 0.3, 0.8, 0.5), P("min_raise", 0.002, 0.15, 0.03),
            P("call_3b_top", 0.03, 0.12, 0.065), P("thin_value", 0.03, 0.45, 0.12),
            P("strong_raise", 0.01, 0.25, 0.06), P("draw_bet", 0.1, 0.6, 0.3),
        ), (("open_early", "cbet", 0.3), ("open_early", "reraise_top", 0.4), ("cbet", "barrel", 0.5),
            ("medium_call", "sticky", 0.3))),
        knobs=_handhq_regular, hooks=(b.sticky,), post=(b.min_raiser,),
        regime={"open_early": 0.8, "light_reraise_p": 0.8}, regime_doc="opens up",
        expected={"vpip": (0.13, 0.27), "pfr": (0.1, 0.22), "three_bet": (0.02, 0.07), "cold_call": (0.05, 0.17),
                  "fold_to_reraise": (0.35, 0.7), "cbet": (0.5, 0.85), "aggression": (1.8, 5.5)},
    ),
    Family(
        id="handhq_recreational", number=20, title="Real-player recreational (HandHQ)", anchor="aggressive",
        summary="A 2009 online six-max recreational player: limps and over-limps, raises a narrow range, cold-calls "
                "raises, calls bets with weak hands at small prices and min-raises; knobs fitted to the HandHQ "
                "players whose PFR is under 60% of their VPIP.",
        prior=Prior("handhq_recreational", (
            P("limp_width", 0.15, 0.55, 0.31), P("open_top", 0.06, 0.24, 0.16), P("limp_trap", 0.01, 0.3, 0.08),
            P("open_mult", 2.5, 5.0, 3.5), P("iso_top", 0.03, 0.17, 0.09), P("iso_mult", 3.0, 6.0, 4.0),
            P("overlimp_width", 0.15, 0.55, 0.36), P("limp_reraise_top", 0.005, 0.04, 0.015),
            P("limp_call", 0.08, 0.4, 0.2), P("limp_call_price", 0.06, 0.25, 0.14),
            P("reraise_top", 0.01, 0.05, 0.025), P("call_top", 0.15, 0.5, 0.32), P("call_price", 0.08, 0.3, 0.16),
            P("cbet", 0.2, 0.7, 0.4), P("barrel", 0.1, 0.5, 0.25), P("medium_call", 0.6, 1.4, 0.95),
            P("fight", 0.3, 0.9, 0.6), P("sticky", 0.15, 0.7, 0.4), P("sticky_price", 0.3, 1.0, 0.65),
            P("stab", 0.03, 0.3, 0.12), P("stab_frac", 0.3, 0.8, 0.5), P("min_raise", 0.02, 0.4, 0.12),
            P("call_3b_top", 0.02, 0.1, 0.05), P("thin_value", 0.03, 0.4, 0.15),
            P("strong_raise", 0.01, 0.2, 0.05), P("draw_bet", 0.05, 0.5, 0.2),
        ), (("limp_width", "overlimp_width", 0.6), ("call_top", "sticky", 0.4), ("open_top", "iso_top", 0.4))),
        knobs=_handhq_recreational, hooks=(b.limp_iso, b.sticky), post=(b.min_raiser,),
        regime={"sticky": -1.0, "limp_width": -1.0}, regime_doc="tightens up",
        expected={"vpip": (0.19, 0.36), "pfr": (0.06, 0.15), "limp": (0.07, 0.25), "cold_call": (0.15, 0.36),
                  "fold_to_reraise": (0.2, 0.55), "aggression": (0.5, 1.5), "min_raise_share": (0.04, 0.25)},
    ),
)

FAMILIES: dict[str, Family] = {f.id: f for f in FAMILY_LIST}
ALL_FAMILIES: dict[str, Family] = {**FAMILIES, **{f.id: f for f in HANDHQ_FAMILY_LIST}}
"""Every family a member spec may name: the pop-v1 families and the HandHQ extension."""
HOLDOUT_ONLY: tuple[str, ...] = tuple(f.id for f in FAMILY_LIST if f.split == "holdout_only")
CHECKPOINT_FAMILY = "frozen_checkpoint"
"""Family 18: frozen learner checkpoints and trained approximate responses (registry slots)."""
