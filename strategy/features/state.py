"""Public table-state features: stacks, pots, position, action closure, clocks, points.

Inputs are the acting seat's :class:`~arena.domain.observations.Observation` and the
current hand's public events. :class:`HandContext` consumes events incrementally (by
``seq``) and also emits one :class:`ActionRecord` per public action, with the
opportunity context the actor faced, for the belief layer.

Seat order assumes the observation's ``seats`` ring is clockwise, as in
:mod:`arena.table.positions` (dead button, possibly dead small blind).

Point pressure uses Malmuth-Harville ICM over hand-start stacks as a prior on terminal
points (exact at the hand cap, an approximation before it), for the hero only and over
paid places only, so an eight-handed table with three paid places costs about thirty
terms once per hand. Hand-start stack features are cached per hand too, and clash risk
premiums are computed lazily on first use.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import Any

STREET_INDEX = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
EARLY, MIDDLE, LATE, BLINDS = range(4)


# ------------------------------------------------------------------------- ICM


def icm_values(stacks: Sequence[int], prizes: Sequence[float]) -> list[float]:
    """Malmuth-Harville expected prize per player, enumerating only paid places."""
    n = len(stacks)
    out = [0.0] * n
    if n == 0:
        return out
    if n == 1:
        out[0] = float(prizes[0]) if prizes else 0.0
        return out
    paid = 0
    for i in range(min(n, len(prizes))):
        if prizes[i] > 0:
            paid = i + 1
    total = float(sum(stacks))
    layer: dict[int, tuple[float, float]] = {0: (1.0, 0.0)}
    for place in range(paid):
        prize = prizes[place]
        last = place + 1 == paid
        nxt: dict[int, tuple[float, float]] = {}
        for mask, (pr, used) in layer.items():
            scale = pr / (total - used)
            for i in range(n):
                if mask >> i & 1:
                    continue
                q = scale * stacks[i]
                out[i] += q * prize
                if not last:
                    m = mask | (1 << i)
                    prev = nxt.get(m)
                    nxt[m] = (q + prev[0], used + stacks[i]) if prev else (q, used + stacks[i])
        layer = nxt
    return out


def icm_value(stacks: Sequence[int], prizes: Sequence[float], i: int) -> float:
    """Malmuth-Harville expected prize of player ``i`` alone.

    Walks the sets of *other* players that can fill the places above ``i``, so with
    ``k`` paid places it touches ``sum_{d<k} C(n-1, d)`` sets instead of every player
    in every set, about 8x fewer terms than :func:`icm_values` for 8 players, 3 paid.
    """
    n = len(stacks)
    if n == 0:
        return 0.0
    if n == 1:
        return float(prizes[0]) if prizes else 0.0
    paid = 0
    for p in range(min(n, len(prizes))):
        if prizes[p] > 0:
            paid = p + 1
    total = float(sum(stacks))
    si = stacks[i]
    others = [j for j in range(n) if j != i]
    value = 0.0
    layer: dict[int, tuple[float, float]] = {0: (1.0, 0.0)}
    for place in range(paid):
        prize = prizes[place]
        nxt: dict[int, tuple[float, float]] = {}
        last = place + 1 == paid
        for mask, (pr, used) in layer.items():
            rest = total - used
            value += pr * si / rest * prize
            if last:
                continue
            scale = pr / rest
            for j in others:
                if mask >> j & 1:
                    continue
                m = mask | (1 << j)
                q = scale * stacks[j]
                prev = nxt.get(m)
                nxt[m] = (q + prev[0], used + stacks[j]) if prev else (q, used + stacks[j])
        layer = nxt
    return value


def survivor_icm(stacks: Mapping[int, int], payouts: Sequence[float], seat: int) -> float:
    """``seat``'s ICM points among ``stacks``; a zero stack takes the last place still open."""
    alive = {s: v for s, v in stacks.items() if v > 0}
    n = len(stacks)
    if stacks.get(seat, 0) <= 0:
        busted = n - len(alive)
        place = n - busted  # shares of tied busts are ignored for a single hero bust
        return float(payouts[place]) if place < len(payouts) else 0.0
    seats = sorted(alive)
    return icm_value([alive[s] for s in seats], list(payouts[:len(alive)]), seats.index(seat))


# --------------------------------------------------------------- hand context


@dataclass(slots=True)
class ActionRecord:
    """One public action with the opportunity the actor faced (before acting)."""

    seat: int
    hand_number: int
    street: int
    kind: str
    to_amount: int
    added: int
    to_call: int
    pot_before: int
    level_before: int
    raises_before: int
    live_before: int
    stack_before: int
    eff_bb: float
    pos_group: int
    initiative: bool
    """The actor was the last aggressor of the previous street (c-bet opportunity)."""
    all_in: bool
    auto: bool
    bb: int

    @property
    def size_frac(self) -> float:
        """Raise increment over the current level, as a fraction of the pot after calling."""
        if self.kind != "raise_to":
            return 0.0
        return (self.to_amount - self.level_before) / max(1, self.pot_before + self.to_call)


class HandContext:
    """Public state of the current hand, rebuilt incrementally from projected events."""

    __slots__ = ("last_seq", "hand_id", "hand_number", "level_index", "sb", "bb", "button", "sb_seat", "bb_seat",
                 "heads_up", "start", "dealt", "pre_order", "post_order", "committed", "street_committed",
                 "folded", "street", "level", "raises", "acted", "aggressor", "street_aggressors", "pre_raises",
                 "limpers", "callers", "pot", "records", "shown", "ring", "_pos_group", "ante", "ante_posting",
                 "antes_posted")

    def __init__(self, ring: Sequence[int] = ()) -> None:
        self.ring = list(ring)
        self.last_seq = -1
        self.records: list[ActionRecord] = []
        self._reset(None, 0)

    def _reset(self, hand_id: str | None, hand_number: int) -> None:
        self.hand_id = hand_id
        self.hand_number = hand_number
        self.level_index = 0
        self.sb = self.bb = 0
        self.button = self.sb_seat = self.bb_seat = -1
        self.heads_up = False
        self.start: dict[int, int] = {}
        self.dealt: list[int] = []
        self.pre_order: list[int] = []
        self.post_order: list[int] = []
        self.committed: dict[int, int] = {}
        self.street_committed: dict[int, int] = {}
        self.folded: set[int] = set()
        self.street = 0
        self.level = 0
        self.raises = 0
        self.acted: set[int] = set()
        self.aggressor = -1
        self.street_aggressors = [-1, -1, -1, -1]
        self.pre_raises = 0
        self.limpers = 0
        self.callers = 0
        self.pot = 0
        self.shown: dict[int, list[str]] = {}
        self._pos_group: dict[int, int] = {}
        self.ante = 0
        self.ante_posting: str | None = None
        self.antes_posted: dict[int, int] = {}

    # ------------------------------------------------------------ ordering

    def _clockwise_after(self, seat: int, members: Iterable[int]) -> list[int]:
        ring = self.ring or sorted(set(members) | {seat})
        if seat not in ring:
            ring = sorted(set(ring) | {seat})
        i = ring.index(seat)
        live = set(members)
        return [s for s in ring[i + 1:] + ring[:i + 1] if s in live]

    def _positions(self) -> None:
        dealt = self.dealt
        self.pre_order = self._clockwise_after(self.bb_seat, dealt)
        self.post_order = self._clockwise_after(self.button, dealt)
        groups: dict[int, int] = {}
        order = self.pre_order
        non_blind = [s for s in order if s not in (self.sb_seat, self.bb_seat)]
        if self.heads_up:
            groups = {self.button: LATE, self.bb_seat: BLINDS}
        else:
            for back, s in enumerate(reversed(non_blind)):
                groups[s] = LATE if back <= 1 else MIDDLE if back <= 3 else EARLY
            for s in (self.sb_seat, self.bb_seat):
                if s in self.start:
                    groups[s] = BLINDS
        self._pos_group = groups

    def pos_group(self, seat: int) -> int:
        return self._pos_group.get(seat, BLINDS)

    # --------------------------------------------------------------- input

    def feed(self, events: Iterable[Any]) -> None:
        for e in events:
            if isinstance(e, dict):
                self.apply(e["seq"], e.get("hand_id"), e["type"], e["data"])
            else:
                self.apply(e.seq, e.hand_id, e.type, e.data)

    def apply(self, seq: int, hand_id: str | None, etype: str, d: Mapping[str, Any]) -> bool:
        """Apply one event; False when ``seq`` was already seen."""
        if seq <= self.last_seq:
            return False
        self.last_seq = seq
        if etype == "ActionApplied":
            self._action(d)
        elif etype == "BoardDealt":
            self._board(d)
        elif etype == "BlindPosted":
            self._blind(d)
        elif etype == "AntePosted":
            self._ante(d)
        elif etype == "HandStarted":
            self._start(hand_id, d)
        elif etype == "CardsRevealed":
            self.shown[d["seat"]] = list(d["cards"])
        return True

    def _board(self, d: Mapping[str, Any]) -> None:
        self.street = STREET_INDEX[d["street"]]
        self.level = 0
        self.raises = 0
        self.acted = set()
        self.street_committed = dict.fromkeys(self.dealt, 0)

    def _blind(self, d: Mapping[str, Any]) -> None:
        s, amt = d["seat"], d["amount"]
        self.committed[s] += amt
        self.street_committed[s] += amt
        self.pot += amt
        if amt > self.level:
            self.level = amt

    def _ante(self, d: Mapping[str, Any]) -> None:
        """An ante is in the pot but not in the street: it never sets the price of a call."""
        s, amt = d["seat"], d["amount"]
        self.committed[s] += amt
        self.pot += amt
        self.antes_posted[s] = amt

    @property
    def dead_ante(self) -> int:
        """Big-blind antes: dead money in the main pot, contested by every live hand."""
        return sum(self.antes_posted.values()) if self.ante_posting == "big_blind" else 0

    def _start(self, hand_id: str | None, d: Mapping[str, Any]) -> None:
        self._reset(hand_id, d["hand_number"])
        self.level_index = d["level"]
        self.sb, self.bb = d["small_blind"], d["big_blind"]
        self.ante = d.get("ante", 0)
        self.ante_posting = d.get("ante_posting")
        pos = d["positions"]
        self.button, self.bb_seat, self.heads_up = pos["button"], pos["big_blind"], pos["heads_up"]
        sbs = d.get("small_blind_seat")
        self.sb_seat = -1 if sbs is None else sbs
        self.start = {int(s): int(v) for s, v in d["start_stacks"].items()}
        self.dealt = sorted(self.start)
        self.committed = dict.fromkeys(self.dealt, 0)
        self.street_committed = dict.fromkeys(self.dealt, 0)
        self._positions()

    def _action(self, d: Mapping[str, Any]) -> None:
        seat = d["seat"]
        kind = d["action"]["type"]
        added = d["added"]
        to_amount = d["street_committed"]
        to_call = max(0, self.level - self.street_committed.get(seat, 0))
        stack_before = self.start.get(seat, 0) - self.committed.get(seat, 0)
        live_before = len(self.dealt) - len(self.folded)
        actor_total = self.start.get(seat, 0)
        others = [self.start[s] for s in self.dealt if s != seat and s not in self.folded]
        eff = min(actor_total, max(others)) if others else actor_total
        prev_aggr = self.street_aggressors[self.street - 1] if self.street > 0 else -1
        rec = ActionRecord(seat, self.hand_number, self.street, kind, to_amount, added, to_call, self.pot, self.level,
                           self.raises, live_before, stack_before, eff / max(1, self.bb), self.pos_group(seat),
                           prev_aggr == seat, added >= stack_before > 0, bool(d.get("auto")), self.bb)
        self.records.append(rec)
        self.committed[seat] = self.committed.get(seat, 0) + added
        self.street_committed[seat] = to_amount
        self.pot += added
        if kind == "fold":
            self.folded.add(seat)
        elif kind == "raise_to":
            self.level = max(self.level, to_amount)
            self.raises += 1
            self.aggressor = seat
            self.street_aggressors[self.street] = seat
            self.acted = {seat}
            self.callers = 0
            if self.street == 0:
                self.pre_raises += 1
        else:
            self.acted.add(seat)
            if kind == "call":
                if self.street == 0 and self.raises == 0:
                    self.limpers += 1
                else:
                    self.callers += 1

    def drain(self) -> list[ActionRecord]:
        out, self.records = self.records, []
        return out


# ------------------------------------------------------------------- features


@dataclass(slots=True)
class StateFeatures:
    # identity and clock
    street: int
    hand_number: int
    hand_cap: int
    hands_left: int
    hands_to_cap: int
    hands_to_level: int
    level_index: int
    next_bb_ratio: float
    orbits_to_cap: float
    orbits_to_level: float
    cap_progress: float
    # blinds and depth
    sb: int
    bb: int
    stack: int
    committed: int
    start_stack: int
    stack_bb: float
    m: float
    m_next: float
    effective: int
    effective_bb: float
    effective_behind: int
    # players
    initial_players: int
    survivors: int
    dealt: int
    live: int
    can_act: int
    opponents: int
    all_in_opponents: int
    # position and closure
    pre_index: int
    post_index: int
    position_frac: float
    pos_group: int
    is_button: bool
    is_sb: bool
    is_bb: bool
    in_position: bool
    behind: int
    yet_to_act: int
    closes_action: bool
    # action so far
    raises: int
    pre_raises: int
    limpers: int
    callers: int
    aggressor: int
    hero_is_aggressor: bool
    hero_has_initiative: bool
    aggressor_before_hero: bool
    # price and geometry
    pot: int
    pot_bb: float
    to_call: int
    call_amount: int
    pot_odds: float
    pot_odds_effective: float
    bet_frac: float
    spr: float
    spr_after_call: float
    min_raise_frac: float
    pot_raise_frac: float
    all_in_frac: float
    can_raise: bool
    # coverage and side pots
    covered_by: int
    covers: int
    eff_vs_aggressor_bb: float
    side_pots: int
    winnable_if_call: int
    uncontestable: int
    dead_money: int
    # table stacks
    stack_rank: int
    covered_by_table: int
    shortest_bb: float
    largest_opp_bb: float
    average_bb: float
    chip_share: float
    stack_vector: tuple[float, ...]
    """Hand-start chip shares of all dealt seats, clockwise from the hero (hero first)."""
    # points
    icm_points: float
    icm_share: float
    icm_chip_ratio: float
    bust_points: float
    ladder_jump: float
    in_money: bool
    on_bubble: bool
    _clash: Any = field(default=None, repr=False)

    def clash(self, opponent: int | None = None) -> tuple[float, float]:
        """``(point-equity threshold, risk premium)`` for an even all-in against ``opponent``
        (default: the aggressor, else the biggest live opponent). Lazy and cached per hand."""
        return self._clash(opponent) if self._clash is not None else (0.5, 0.0)

    def vector(self) -> tuple[float, ...]:
        return tuple(float(getattr(self, f)) for f in NUMERIC_FIELDS)


NUMERIC_FIELDS: tuple[str, ...] = tuple(
    f.name for f in fields(StateFeatures) if f.name not in ("stack_vector", "_clash"))


class PointModel:
    """Per-hand ICM cache: base points and lazily computed clash thresholds."""

    __slots__ = ("hand_key", "base", "stacks", "payouts", "hero", "_clashes", "statics_key", "statics")

    def __init__(self) -> None:
        self.hand_key: Any = None
        self.statics_key: Any = None
        self.statics: tuple[Any, ...] = ()
        self.base = 0.0
        self._clashes: dict[int, tuple[float, float]] = {}

    def prepare(self, key: Any, stacks: dict[int, int], payouts: Sequence[float], hero: int) -> float:
        if key != self.hand_key:
            self.hand_key = key
            self.stacks, self.payouts, self.hero = stacks, payouts, hero
            self.base = survivor_icm(stacks, payouts, hero) if hero in stacks else 0.0
            self._clashes = {}
        return self.base

    def clash(self, opponent: int) -> tuple[float, float]:
        hit = self._clashes.get(opponent)
        if hit is not None:
            return hit
        st, hero = self.stacks, self.hero
        e = min(st.get(hero, 0), st.get(opponent, 0))
        if e <= 0:
            res = (0.5, 0.0)
        else:
            win = dict(st)
            win[hero] += e
            win[opponent] -= e
            lose = dict(st)
            lose[hero] -= e
            lose[opponent] += e
            uw = survivor_icm(win, self.payouts, hero)
            ul = survivor_icm(lose, self.payouts, hero)
            q = (self.base - ul) / (uw - ul) if uw > ul else 0.5
            res = (q, q - 0.5)
        self._clashes[opponent] = res
        return res


def orbit_cost(sb: int, bb: int, ante: int = 0, ante_posting: str | None = None, dealt: int = 0) -> int:
    """Chips one orbit costs the table (Harrington's M denominator): blinds plus every ante.

    Classic antes are paid by each of ``dealt`` seats every hand; a big-blind ante once.
    """
    if not ante:
        return sb + bb
    return sb + bb + (ante * dealt if ante_posting == "classic" else ante)


def ante_of(obs: Any) -> tuple[int, str | None]:
    """The current hand's ante and posting from the observation (``(0, None)`` without antes)."""
    blinds = getattr(obs, "blinds", None)
    ante = getattr(blinds, "ante", 0) if blinds is not None else 0
    return (ante, blinds.ante_posting) if ante else (0, None)


def state_features(obs: Any, ctx: HandContext, points: PointModel | None = None) -> StateFeatures:
    hero = obs.hero_seat
    players = {p.seat: p for p in obs.players}
    me = players[hero]
    blinds = obs.blinds
    sb, bb = (blinds.small_blind, blinds.big_blind) if blinds is not None else (ctx.sb, ctx.bb)
    bb = max(1, bb)
    street = STREET_INDEX[obs.street] if obs.street else ctx.street
    start = ctx.start or {p.seat: p.stack + p.hand_committed for p in obs.players if p.status != "eliminated"}
    start_stack = start.get(hero, me.stack + me.hand_committed)
    hero_total = me.stack + me.hand_committed

    live = [s for s, p in players.items() if p.status in ("active", "all_in")]
    can_act = [s for s in live if players[s].status == "active"]
    opp_live = [s for s in live if s != hero]
    totals = {s: players[s].stack + players[s].hand_committed for s in live}
    max_opp_total = max((totals[s] for s in opp_live), default=0)
    effective = min(hero_total, max_opp_total)
    effective_behind = max(0, effective - me.hand_committed)

    # position
    order = ctx.pre_order if street == 0 else ctx.post_order
    n_dealt = len(ctx.dealt) or len(live)
    pre_index = ctx.pre_order.index(hero) if hero in ctx.pre_order else 0
    post_index = ctx.post_order.index(hero) if hero in ctx.post_order else 0
    idx = order.index(hero) if hero in order else 0
    behind = sum(1 for s in order[idx + 1:] if s in can_act)
    level = obs.current_bet_to
    sc = ctx.street_committed
    yet = sum(1 for s in can_act if s != hero and (s not in ctx.acted or sc.get(s, 0) < level))
    in_pos = all(ctx.post_order.index(s) < post_index for s in can_act if s != hero and s in ctx.post_order)
    aggressor = ctx.aggressor
    prev_aggr = ctx.street_aggressors[street - 1] if street > 0 else -1

    # price
    legal = obs.legal
    to_call = (legal.call_amount or 0) if legal is not None else 0
    call_amount = min(to_call, me.stack)
    pot = obs.pot_total
    pot_odds = call_amount / (pot + call_amount) if call_amount else 0.0
    cap = me.hand_committed + call_amount
    winnable = 0
    uncontestable = 0
    dead = 0
    for s, p in players.items():
        c = p.hand_committed
        if s == hero:
            winnable += cap
            continue
        winnable += c if c < cap else cap
        if c > cap:
            uncontestable += c - cap
        if p.status == "folded":
            dead += c
    dead_ante = ctx.dead_ante
    if dead_ante:
        # The BB's dead ante is not part of its contribution level; it sits whole in the main pot.
        winnable, uncontestable, dead = _without_dead_ante(players, hero, cap, ctx.antes_posted, dead_ante)
    pot_odds_eff = call_amount / winnable if call_amount and winnable else 0.0
    caps_levels = {players[s].hand_committed for s in live if players[s].status == "all_in"}
    top = max((players[s].hand_committed for s in live), default=0)
    side_pots = sum(1 for c in caps_levels if c < top)
    bet_frac = to_call / max(1, pot - to_call) if to_call else 0.0
    spr = effective_behind / pot if pot else 0.0
    spr_after = (effective_behind - call_amount) / (pot + call_amount) if pot + call_amount else 0.0
    after_call = max(1, pot + call_amount)
    can_raise = bool(legal is not None and legal.can_raise)
    min_to = legal.min_full_raise_to if can_raise else None
    if min_to is None and can_raise:
        min_to = legal.short_all_in_raise_to
    min_raise_frac = (min_to - level) / after_call if min_to else 0.0
    pot_raise_frac = (legal.pot_raise_to - level) / after_call if can_raise and legal.pot_raise_to else 0.0
    all_in_frac = (me.stack - call_amount) / after_call

    covered_by = sum(1 for s in opp_live if totals[s] > hero_total)
    covers = len(opp_live) - covered_by
    eff_aggr = min(hero_total, totals[aggressor]) / bb if aggressor in totals and aggressor != hero else 0.0

    # table stacks and points depend only on hand-start stacks: once per hand
    pm = points or PointModel()
    key = (ctx.hand_id, hero, bb) if ctx.hand_id is not None else (None, hero, bb, tuple(sorted(start.items())))
    if pm.statics_key != key:
        pm.statics_key = key
        pm.statics = _hand_statics(key, start, start_stack, hero, obs.payouts, bb, ctx, pm)
    (surv, stack_rank, shortest_bb, largest_bb, average_bb, chip_share, stack_vector, icm_pts, icm_share, icm_ratio,
     bust_points, ladder) = pm.statics

    # clocks
    hands_left = obs.hands_remaining
    hands_to_cap = max(0, hands_left - 1)
    nl = obs.next_level
    survivors = obs.surviving_player_count
    ante, posting = ante_of(obs)
    next_ante = getattr(nl, "ante", 0) if nl is not None else 0
    orbit = orbit_cost(sb, bb, ante, posting, n_dealt) if ante else sb + bb
    if nl is not None:
        hands_to_level = nl.starts_at_hand - obs.hand_number
        next_bb_ratio = nl.big_blind / bb
        if next_ante:
            m_next = start_stack / max(1, orbit_cost(nl.small_blind, nl.big_blind, next_ante, nl.ante_posting,
                                                     n_dealt))
        else:
            m_next = start_stack / max(1, nl.small_blind + nl.big_blind)
    else:
        hands_to_level = hands_left
        next_bb_ratio = 1.0
        m_next = start_stack / max(1, orbit)

    biggest = max(opp_live, key=lambda s: (start.get(s, 0), -s), default=-1)
    default_opp = aggressor if aggressor in start and aggressor != hero else biggest

    def clash(opponent: int | None) -> tuple[float, float]:
        opp = default_opp if opponent is None else opponent
        if opp < 0 or opp not in surv or hero not in surv:
            return 0.5, 0.0
        return pm.clash(opp)

    return StateFeatures(
        street, obs.hand_number, obs.hand_cap, hands_left, hands_to_cap, hands_to_level, ctx.level_index,
        next_bb_ratio, hands_to_cap / max(1, survivors), hands_to_level / max(1, survivors),
        (obs.hand_number - 1) / max(1, obs.hand_cap),
        sb, bb, me.stack, me.hand_committed, start_stack, start_stack / bb, start_stack / max(1, orbit), m_next,
        effective, effective / bb, effective_behind,
        obs.initial_player_count, survivors, n_dealt, len(live), len(can_act), len(opp_live),
        sum(1 for s in opp_live if players[s].status == "all_in"),
        pre_index, post_index, post_index / max(1, n_dealt - 1), ctx.pos_group(hero),
        hero == ctx.button, hero == ctx.sb_seat, hero == ctx.bb_seat, in_pos, behind, yet, yet == 0,
        ctx.raises, ctx.pre_raises, ctx.limpers, ctx.callers, aggressor, aggressor == hero, prev_aggr == hero,
        aggressor in order and order.index(aggressor) < idx if aggressor >= 0 else False,
        pot, pot / bb, to_call, call_amount, pot_odds, pot_odds_eff, bet_frac, spr, spr_after, min_raise_frac,
        pot_raise_frac, all_in_frac, can_raise,
        covered_by, covers, eff_aggr, side_pots, winnable, uncontestable, dead,
        stack_rank, stack_rank - 1, shortest_bb, largest_bb, average_bb, chip_share, stack_vector,
        icm_pts, icm_share, icm_ratio, bust_points, ladder, bust_points > 0, bust_points == 0 and ladder > 0, clash)


def _without_dead_ante(players: Mapping[int, Any], hero: int, cap: int, antes: Mapping[int, int],
                       dead_ante: int) -> tuple[int, int, int]:
    """``(winnable, uncontestable, dead)`` with big-blind antes counted as main-pot dead money."""
    cap -= antes.get(hero, 0)
    winnable = dead_ante
    uncontestable = 0
    dead = dead_ante
    for s, p in players.items():
        c = p.hand_committed - antes.get(s, 0)
        if s == hero:
            winnable += cap
            continue
        winnable += c if c < cap else cap
        if c > cap:
            uncontestable += c - cap
        if p.status == "folded":
            dead += c
    return winnable, uncontestable, dead


def _hand_statics(key: Any, start: Mapping[int, int], start_stack: int, hero: int, payouts: Sequence[float], bb: int,
                  ctx: HandContext, pm: PointModel) -> tuple[Any, ...]:
    surv = {s: v for s, v in start.items() if v > 0}
    chips = sum(surv.values()) or 1
    opp_starts = [v for s, v in surv.items() if s != hero]
    stack_rank = 1 + sum(1 for v in opp_starts if v > start_stack)
    ring_from_hero = ctx._clockwise_after(hero, surv) if ctx.dealt else sorted(surv)
    if ring_from_hero and ring_from_hero[-1] == hero:
        ring_from_hero = [hero, *ring_from_hero[:-1]]
    stack_vector = tuple(round(surv[s] / chips, 4) for s in ring_from_hero)
    s_count = len(surv)
    icm_pts = pm.prepare(key, surv, payouts, hero)
    prize_pool = float(sum(payouts[:s_count])) or 1.0
    icm_share = icm_pts / prize_pool
    chip_share = start_stack / chips
    bust_points = float(payouts[s_count - 1]) if 0 < s_count <= len(payouts) else 0.0
    ladder = float(payouts[s_count - 2] - payouts[s_count - 1]) if 2 <= s_count <= len(payouts) else 0.0
    return (surv, stack_rank, min(surv.values()) / bb if surv else 0.0, max(opp_starts) / bb if opp_starts else 0.0,
            chips / max(1, s_count) / bb, chip_share, stack_vector, icm_pts, icm_share,
            icm_share / chip_share if chip_share else 0.0, bust_points, ladder)


def extract_state(obs: Any) -> StateFeatures:
    """One-shot features from a single observation (rebuilds the hand context)."""
    ctx = HandContext(obs.seats)
    ctx.feed(obs.action_history)
    return state_features(obs, ctx)
