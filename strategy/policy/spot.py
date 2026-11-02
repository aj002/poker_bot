"""The runtime's slim per-decision view: public hand state, hand strength, points, opponent counts.

:class:`TableMemory` lives for one table. Each decision it consumes only the public
events it has not seen (by ``seq``) through :class:`~strategy.features.state.HandContext`,
updates a few integer counters per opponent, and builds a :class:`Spot` with exactly
the inputs :mod:`strategy.policy.baseline` reads. Hand strength comes from the offline
equity tables (one dictionary lookup preflop; made hand, draws and texture once per
street). ICM uses hand-start stacks and is computed once per hand, with all-in clash
thresholds computed lazily per opponent.

This is deliberately narrower than :class:`~strategy.features.engine.FeatureEngine`: no
Bayesian range or posterior updates on the decision path, so a decision costs tens of
microseconds and allocates little. The counters are the only cross-hand memory, and
they round-trip through the bot's explicit state (:meth:`TableMemory.counters_state`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from strategy.features.cards import (CardCapabilities, capabilities, card_ints, evaluate, postflop_cards,
                                     postflop_key)
from strategy.features.equity import EquityTables
from strategy.features.state import BLINDS, LATE, HandContext, PointModel

EARLY_G, MIDDLE_G, LATE_G, SB_G, BB_G = range(5)
STREETS = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}

# Per-opponent counters (cross-hand memory), all integers.
HANDS, VPIP, PFR, FACED, FOLDED, AGGR, PASSIVE = range(7)
N_COUNTERS = 7

_CAPS: dict[str, CardCapabilities] = {}
_TABLES: dict[str, EquityTables] = {}


def variant_assets(ruleset_id: str) -> tuple[CardCapabilities, EquityTables]:
    """Card capabilities and equity tables, loaded once per process per variant."""
    caps = _CAPS.get(ruleset_id)
    if caps is None:
        caps = capabilities(ruleset_id)
        n = caps.hole_count
        evaluate(caps.deck[:n], caps.deck[n:n + 3], caps)  # builds the lazy rank tables off the decision path
        _TABLES[ruleset_id] = EquityTables(caps, _tables_dir(ruleset_id))
        _CAPS[ruleset_id] = caps
    return caps, _TABLES[ruleset_id]


def _tables_dir(ruleset_id: str) -> str | None:
    """The reveal variant's tables travel with its spec (:func:`strategy.rules.reveal.reveal_tables_dir`)."""
    from strategy.rules.reveal import REVEAL_RULESET_ID, reveal_tables_dir

    if ruleset_id != REVEAL_RULESET_ID:
        return None
    d = reveal_tables_dir()
    return str(d) if d is not None else None


@dataclass(slots=True)
class Spot:
    street: int
    hero: int
    bb: int
    to_call: int
    pot: int
    """Everything in the middle, current street bets included."""
    current_bet_to: int
    hero_stack: int
    """Chips behind."""
    hero_street: int
    hero_total: int
    """Stack plus chips committed this hand."""
    eff_bb: float
    """min(hero total, largest live opponent total) in big blinds."""
    group: int
    opponents: int
    """Live (not folded) opponents."""
    active_opponents: int
    behind: int
    """Active players still to act after the hero on this street (preflop: in seat order)."""
    in_position: bool
    pre_raises: int
    limpers: int
    callers: int
    hero_raised_pre: bool
    aggressor: int
    aggressor_group: int
    street_raises: int
    initiative: bool
    """The hero was the last aggressor of the previous street."""
    hero_aggr_street: bool
    """The hero made the last bet or raise on this street."""
    bettor_all_in: bool
    pct: float
    eq: float
    """Equity against ``opponents`` random hands (offline tables)."""
    eq1: float
    """Preflop heads-up equity against one random hand."""
    made_level: int
    draw_class: int
    wet: float
    paired_board: bool
    survivors: int
    paid: int
    hands_left: int
    hand_cap: int
    stack_share: float
    """Hero hand-start chips over the average survivor's."""
    memory: Any = None

    @property
    def pot_odds(self) -> float:
        return self.to_call / max(1, self.pot + self.to_call)

    @property
    def bet_frac(self) -> float:
        return self.to_call / max(1, self.pot - self.to_call)

    @property
    def spr(self) -> float:
        return self.hero_stack / max(1, self.pot)

    def clash(self, seat: int) -> tuple[float, float]:
        """(ICM equity needed to call an all-in against ``seat``, risk premium over 0.5)."""
        if seat < 0 or self.memory is None:
            return 0.5, 0.0
        return self.memory.points.clash(seat)

    def max_risk_premium(self) -> float:
        """Largest risk premium against any live opponent (the cost of a confrontation now)."""
        mem = self.memory
        if mem is None:
            return 0.0
        return max((mem.points.clash(s)[1] for s in mem.live_opponents), default=0.0)

    def stat(self, seat: int, index: int) -> int:
        row = self.memory.counters.get(seat) if self.memory is not None else None
        return row[index] if row is not None else 0


def _event(e: Any) -> tuple[int, str | None, str, Mapping[str, Any]]:
    if isinstance(e, dict):
        return e["seq"], e.get("hand_id"), e["type"], e["data"]
    return e.seq, e.hand_id, e.type, e.data


class TableMemory:
    """Per-table runtime memory. Rebuildable from (counters state, current observation)."""

    __slots__ = ("ruleset_id", "hero", "seats", "caps", "tables", "ctx", "points", "counters", "counted_seq",
                 "_vpip", "_pfr", "_hand_id", "_hole", "_pre", "_board_n", "_street", "live_opponents")

    def __init__(self, ruleset_id: str, hero: int, seats: Sequence[int],
                 counters: Mapping[str, Sequence[int]] | None = None, counted_seq: int = -1) -> None:
        self.ruleset_id = ruleset_id
        self.hero = hero
        self.seats = list(seats)
        self.caps, self.tables = variant_assets(ruleset_id)
        self.ctx = HandContext(self.seats)
        self.points = PointModel()
        self.counters: dict[int, list[int]] = {s: [0] * N_COUNTERS for s in self.seats if s != hero}
        for k, row in (counters or {}).items():
            if int(k) in self.counters and len(row) == N_COUNTERS:
                self.counters[int(k)] = [int(x) for x in row]
        self.counted_seq = counted_seq
        self._vpip: set[int] = set()
        self._pfr: set[int] = set()
        self._hand_id: str | None = None
        self._hole: tuple[int, ...] = ()
        self._pre: dict[int, Any] = {}
        self._board_n = -1
        self._street: tuple[Any, ...] = ()
        self.live_opponents: list[int] = []

    def counters_state(self) -> dict[str, list[int]]:
        return {str(s): list(row) for s, row in self.counters.items()}

    # ------------------------------------------------------------- events

    def _count(self, etype: str, count: bool) -> None:
        ctx = self.ctx
        if etype == "HandStarted":
            self._vpip.clear()
            self._pfr.clear()
            if count:
                for s in ctx.dealt:
                    row = self.counters.get(s)
                    if row is not None:
                        row[HANDS] += 1
            return
        for rec in ctx.records:
            if rec.auto:
                continue
            kind = rec.kind
            row = self.counters.get(rec.seat)
            if rec.street == 0:
                if kind == "call" or kind == "raise_to":
                    if rec.seat not in self._vpip:
                        self._vpip.add(rec.seat)
                        if count and row is not None:
                            row[VPIP] += 1
                    if kind == "raise_to" and rec.seat not in self._pfr:
                        self._pfr.add(rec.seat)
                        if count and row is not None:
                            row[PFR] += 1
            elif count and row is not None:
                if rec.to_call > 0:
                    row[FACED] += 1
                    if kind == "fold":
                        row[FOLDED] += 1
                if kind == "raise_to":
                    row[AGGR] += 1
                elif kind != "fold":
                    row[PASSIVE] += 1
        ctx.records.clear()

    def feed(self, events: Sequence[Any]) -> None:
        ctx = self.ctx
        for e in events:
            seq, hand_id, etype, d = _event(e)
            if not ctx.apply(seq, hand_id, etype, d):
                continue
            if etype == "ActionApplied" or etype == "HandStarted":
                fresh = seq > self.counted_seq
                self._count(etype, fresh)
                if fresh:
                    self.counted_seq = seq
            elif seq > self.counted_seq:
                self.counted_seq = seq

    # ----------------------------------------------------------- decision

    def _cards(self, obs: Any) -> tuple[Any, ...]:
        board = obs.board
        n = len(board)
        if n == self._board_n:
            return self._street
        self._board_n = n
        made, dr, _bl, tx = postflop_cards(self._hole, card_ints(board), self.caps)
        street = STREETS[obs.street]
        self._street = (made, dr, tx, postflop_key(street, made.level, dr.draw_class, tx.key))
        return self._street

    def spot(self, obs: Any) -> Spot:
        if obs.hand_id != self._hand_id:
            self._hand_id = obs.hand_id
            self._hole = card_ints(obs.hole_cards)
            self._pre = {}
            self._board_n = -1
        self.feed(obs.action_history)
        ctx = self.ctx
        hero = self.hero
        bb = max(1, obs.blinds.big_blind if obs.blinds is not None else ctx.bb)
        street = STREETS[obs.street]
        me = None
        live: list[int] = []
        active: list[int] = []
        totals: dict[int, int] = {}
        for p in obs.players:
            if p.seat == hero:
                me = p
            elif p.status == "active" or p.status == "all_in":
                live.append(p.seat)
                totals[p.seat] = p.stack + p.hand_committed
                if p.status == "active":
                    active.append(p.seat)
        assert me is not None
        self.live_opponents = live
        hero_total = me.stack + me.hand_committed
        eff = min(hero_total, max(totals.values(), default=0))

        if hero == ctx.bb_seat:
            group = BB_G
        elif hero == ctx.sb_seat and not ctx.heads_up:
            group = SB_G
        else:
            g = ctx.pos_group(hero)
            group = LATE_G if g == LATE else (BB_G if g == BLINDS else g)
        aggr = ctx.aggressor
        if aggr < 0:
            agroup = -1
        elif aggr == ctx.bb_seat:
            agroup = BB_G
        elif aggr == ctx.sb_seat and not ctx.heads_up:
            agroup = SB_G
        else:
            g = ctx.pos_group(aggr)
            agroup = LATE_G if g == LATE else (BB_G if g == BLINDS else g)

        if street == 0:
            order = ctx.pre_order
            i = order.index(hero) if hero in order else 0
            act = set(active)
            behind = sum(1 for s in order[i + 1:] if s in act)
            post = ctx.post_order
            hi = post.index(hero) if hero in post else 0
            in_pos = all(post.index(s) < hi for s in active if s in post)
        else:
            post = ctx.post_order
            hi = post.index(hero) if hero in post else 0
            in_pos = all(post.index(s) < hi for s in active if s in post)
            behind = sum(1 for s in active if s in post and post.index(s) > hi)

        k = max(1, len(live))
        pre = self._pre.get(k)
        if pre is None:
            pre = self._pre[k] = self.tables.preflop(self._hole, k)
        pre1 = self._pre.get(1)
        if pre1 is None:
            pre1 = self._pre[1] = self.tables.preflop(self._hole, 1)
        if street == 0:
            eq, level, dcls, wet, paired = pre.equity, 0, 0, 0.0, False
        else:
            made, dr, tx, cell = self._cards(obs)
            eq = self.tables.postflop(cell, made.level, dr.draw_class, k).equity
            level, dcls, wet, paired = made.level, dr.draw_class, tx.dynamic, tx.paired

        start = ctx.start
        surv = {s: v for s, v in start.items() if v > 0}
        if hero in surv:
            self.points.prepare(obs.hand_id, surv, obs.payouts, hero)
        avg = sum(surv.values()) / max(1, len(surv))
        paid = sum(1 for x in obs.payouts if x > 0)
        bettor_all_in = False
        if aggr >= 0 and aggr != hero:
            for p in obs.players:
                if p.seat == aggr:
                    bettor_all_in = p.status == "all_in"
                    break
        prev_aggr = ctx.street_aggressors[street - 1] if street > 0 else -1
        return Spot(
            street=street, hero=hero, bb=bb, to_call=obs.legal.call_amount or 0, pot=obs.pot_total,
            current_bet_to=obs.current_bet_to, hero_stack=me.stack, hero_street=me.street_committed,
            hero_total=hero_total, eff_bb=eff / bb, group=group, opponents=len(live), active_opponents=len(active),
            behind=behind, in_position=in_pos, pre_raises=ctx.pre_raises, limpers=ctx.limpers, callers=ctx.callers,
            hero_raised_pre=hero in self._pfr,
            aggressor=aggr, aggressor_group=agroup, street_raises=ctx.raises, initiative=prev_aggr == hero,
            hero_aggr_street=ctx.street_aggressors[street] == hero, bettor_all_in=bettor_all_in,
            pct=pre.percentile, eq=eq, eq1=pre1.equity, made_level=level, draw_class=dcls, wet=wet,
            paired_board=paired, survivors=obs.surviving_player_count, paid=paid, hands_left=obs.hands_remaining,
            hand_cap=obs.hand_cap, stack_share=start.get(hero, hero_total) / max(1.0, avg), memory=self)
