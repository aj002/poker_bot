"""PokerKit adapter for one hand.

PokerKit owns chip movement, action order, bet collection and refunds, hand
evaluation, pot construction and awards. Every automation is off: the adapter
deals each card from the stored deck, posts blinds, collects bets, reveals at
showdown under the local visibility policy and pushes/pulls chips itself, and
records an event for each step.

A thin betting-rights overlay sits on top of PokerKit's legal-action
predicates. It only ever narrows what PokerKit allows, except for the
documented short-big-blind price correction (G20), and it raises
:class:`EngineDiscrepancy` if PokerKit refuses an action the overlay offers.
See docs/assumptions.md for each correction.
"""

from __future__ import annotations

import warnings
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from pokerkit import Card, Mode, State

from arena.domain.cards import DeckError, validate_deck
from arena.domain.models import Action, AntePosting, LegalActions, LocalRules
from arena.rules.base import (
    STREETS,
    EngineDiscrepancy,
    HandEvent,
    HandResult,
    IllegalAction,
    PlayerPublic,
    PotResult,
)
from arena.rules.profiles import RuleProfile

POKERKIT_VERSION = "0.7.7"


class _ArenaState(State):
    """PokerKit state whose pot-limit cap prices the call at the nominal big blind.

    ``arena_level_floor`` is the nominal big blind during preflop and 0 later.
    When the big blind is posted in full this is identical to PokerKit.
    """

    arena_level_floor: int = 0

    @property
    def min_completion_betting_or_raising_to_amount(self) -> int | None:
        minimum = super().min_completion_betting_or_raising_to_amount
        if minimum is None or max(self.bets) >= self.arena_level_floor:
            return minimum
        # Below the nominal big blind a completion is how the adapter prices a
        # call; the betting-rights overlay supplies the real raise minimum.
        i = self.actor_index
        assert i is not None
        return min(self.stacks[i] + self.bets[i], max(self.bets) + 1)

    @property
    def pot_completion_betting_or_raising_to_amount(self) -> int | None:
        try:
            self._verify_completion_betting_or_raising()
        except (ValueError, UserWarning):
            return None
        i = self.actor_index
        assert i is not None
        minimum = self.min_completion_betting_or_raising_to_amount
        assert minimum is not None
        level = max(max(self.bets), self.arena_level_floor)
        return min(
            self.stacks[i] + self.bets[i],
            max(minimum, 2 * level - self.bets[i] + self.total_pot_amount),
        )


def _strict(fn, *args):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return fn(*args)


@dataclass(frozen=True)
class HandSpec:
    """Everything needed to start one hand.

    ``seat_ring`` lists every physical seat at the table clockwise; markers may
    sit on seats that have busted. ``stacks`` holds only live seats.
    """

    profile: RuleProfile
    seat_ring: tuple[int, ...]
    stacks: dict[int, int]
    button: int
    small_blind_seat: int | None
    big_blind_seat: int
    small_blind: int
    big_blind: int
    deck: tuple[str, ...]
    local_rules: LocalRules = LocalRules()
    ante: int = 0
    ante_posting: AntePosting | None = None

    @property
    def heads_up(self) -> bool:
        return len(self.stacks) == 2


def seat_antes(spec: HandSpec) -> dict[int, int]:
    """The ante each live seat owes this hand (docs/assumptions.md, "Antes").

    ``classic``: every live seat owes the ante, before any blind; a stack that cannot cover
    it posts what it has. ``big_blind``: only the big blind owes it, once for the table and
    after its blind, so a short big blind antes only what is left over.
    """
    if not spec.ante:
        return {}
    if spec.ante_posting == "classic":
        return {s: min(spec.ante, v) for s, v in spec.stacks.items()}
    if spec.ante_posting == "big_blind":
        return {spec.big_blind_seat: min(spec.ante, max(0, spec.stacks[spec.big_blind_seat] - spec.big_blind))}
    raise ValueError(f"an ante needs ante_posting 'classic' or 'big_blind', not {spec.ante_posting!r}")


def engine_order(spec: HandSpec) -> tuple[int, ...]:
    """Physical seats in PokerKit player-index order.

    Three or more players: clockwise from the first live seat after the button
    marker, so index 0 is the SB (or the BB when the SB is dead) and the
    button acts last. Heads-up follows PokerKit's convention: index 0 is the
    BB and index 1 is the button/SB.
    """
    live = set(spec.stacks)
    if spec.heads_up:
        if spec.small_blind_seat != spec.button or spec.big_blind_seat == spec.button:
            raise ValueError("heads-up hands need button == small blind != big blind")
        if {spec.button, spec.big_blind_seat} != live:
            raise ValueError("heads-up markers must be the two live seats")
        return (spec.big_blind_seat, spec.button)
    ring = list(spec.seat_ring)
    start = ring.index(spec.button)
    rotated = ring[start + 1 :] + ring[: start + 1]
    order = tuple(s for s in rotated if s in live)
    expected_first = spec.small_blind_seat if spec.small_blind_seat is not None else spec.big_blind_seat
    if order[0] != expected_first:
        raise ValueError("a live seat lies between the button marker and the blinds")
    if spec.small_blind_seat is not None and order[1] != spec.big_blind_seat:
        raise ValueError("a live seat lies between the small and big blind")
    return order


class PokerKitHand:
    def __init__(self, spec: HandSpec) -> None:
        profile = spec.profile
        if len(spec.stacks) < 2:
            raise ValueError("a hand needs at least two live seats")
        if any(type(v) is not int or v <= 0 for v in spec.stacks.values()):
            raise ValueError("live stacks must be positive integers")
        validate_deck(spec.deck, profile.ranks)
        needed = len(spec.stacks) * profile.hole_card_count + 8
        if len(spec.deck) < needed:
            raise DeckError("deck too small for this hand")

        self.spec = spec
        self.profile = profile
        self.local_rules = spec.local_rules
        self.seats: tuple[int, ...] = engine_order(spec)
        self.index_of = {seat: i for i, seat in enumerate(self.seats)}
        n = len(self.seats)
        self._bb = spec.big_blind
        owed = seat_antes(spec)
        after_ante = {s: v - owed.get(s, 0) for s, v in spec.stacks.items()}

        if spec.heads_up:
            raw_blinds: tuple[int, ...] = (spec.small_blind, spec.big_blind)
            posted = [min(spec.big_blind, after_ante[self.seats[0]]), min(spec.small_blind, after_ante[self.seats[1]])]
        else:
            blinds = [0] * n
            if spec.small_blind_seat is not None:
                blinds[self.index_of[spec.small_blind_seat]] = spec.small_blind
            blinds[self.index_of[spec.big_blind_seat]] = spec.big_blind
            raw_blinds = tuple(blinds)
            posted = [min(b, after_ante[s]) for b, s in zip(blinds, self.seats)]

        # PokerKit reads antes in its blind convention: heads-up, index 0 posts entry 1.
        by_index = [owed.get(s, 0) for s in self.seats]
        raw_antes: tuple[int, ...] | int = (by_index[1], by_index[0]) if spec.heads_up else tuple(by_index)
        if not any(by_index):
            raw_antes = 0
        # Classic antes are ordinary contributions (a short ante caps a side pot). A big-blind
        # ante is dead money in the main pot: PokerKit's ante trimming off (assumption G25).
        self.dead_ante = spec.ante_posting == "big_blind" and any(by_index)
        self.ante_owed = {s: a for s, a in owed.items() if a}
        self.raw_blinds = raw_blinds
        game = profile.game_class((), True, 0, raw_blinds, spec.big_blind, mode=Mode.TOURNAMENT)
        # Only classic antes can leave no blind posted (every blind seat all-in for its ante).
        # Then nothing prices a call: preflop opens unbet, with the big blind as the minimum bet.
        self._nominal = max(posted) > 0
        self.short_blind_correction = self._nominal and max(posted) < spec.big_blind
        self.pk = _ArenaState(
            (),
            game.deck,
            game.hand_types,
            game.streets,
            game.betting_structure,
            not self.dead_ante,
            raw_antes,
            raw_blinds,
            0,
            tuple(spec.stacks[s] for s in self.seats),
            n,
            mode=Mode.TOURNAMENT,
        )
        self.pk.deck_cards = deque(Card.clean("".join(spec.deck)))
        self.pk.arena_level_floor = spec.big_blind if self._nominal else 0

        self.starting_stacks = {s: spec.stacks[s] for s in self.seats}
        self.total_chips = sum(self.starting_stacks.values())
        self.deck = tuple(spec.deck)
        self._next_card = 0
        self.hole_cards: dict[int, list[str]] = {s: [] for s in self.seats}
        self.burns: list[str] = []
        self.board: list[str] = []
        self.revealed: dict[int, tuple[str, ...]] = {}
        self.folded: list[int] = []
        self.pots: list[PotResult] = []
        self._showdown_hands: dict[int, object] | None = None
        self._engine_awards: dict[int, int] = {}
        self._correction: dict[int, int] = {}
        self.events: list[HandEvent] = []

        self._street_index: int | None = None
        self._reset_street(0)
        self._advance()

    # ------------------------------------------------------------------ state

    @property
    def is_complete(self) -> bool:
        return not self.pk.status

    @property
    def actor_seat(self) -> int | None:
        i = self.pk.actor_index
        return None if i is None else self.seats[i]

    @property
    def street(self) -> str | None:
        idx = self.pk.street_index
        return None if idx is None else STREETS[idx]

    def _level(self) -> int:
        pk_max = max(self.pk.bets)
        if self.pk.street_index == 0 and pk_max < self._bb and self._nominal and self.pk.can_complete_bet_or_raise_to():
            return self._bb
        return pk_max

    def current_level(self) -> int:
        """The street total a call matches (the nominal big blind while it is short)."""
        return self._level() if self.pk.status else 0

    def betting_rights(self) -> dict:
        """Overlay state that decides who may still raise; part of the state hash."""
        return {
            "full_increment": self._full_increment,
            "acted_level": {self.seats[i]: level for i, level in sorted(self._acted_level.items())},
        }

    def collected(self) -> dict[int, int]:
        """Chips already gathered into the pot, by seat (current street bets excluded)."""
        return self._contributions()

    def players(self) -> list[PlayerPublic]:
        out = []
        for i, seat in enumerate(self.seats):
            if not self.pk.statuses[i] and seat in self.folded:
                status = "folded"
            elif self.pk.stacks[i] == 0 and self.pk.status:
                status = "all_in"
            else:
                status = "active"
            committed = self.starting_stacks[seat] - self.pk.stacks[i] if self.pk.status else 0
            out.append(PlayerPublic(seat, self.pk.stacks[i], self.pk.bets[i] if self.pk.status else 0, committed, status))
        return out

    def pot_total(self) -> int:
        return self.pk.total_pot_amount if self.pk.status else 0

    def chips_in_play(self) -> int:
        return sum(self.pk.stacks) + self.pot_total()

    # ---------------------------------------------------------- betting rights

    def _reset_street(self, street_index: int) -> None:
        self._street_index = street_index
        self._full_increment = self._bb
        self._acted_level: dict[int, int] = {}
        self.pk.arena_level_floor = self._bb if street_index == 0 and self._nominal else 0

    def _has_raise_rights(self, i: int, level: int) -> bool:
        acted = self._acted_level.get(i)
        return acted is None or level - acted >= self._full_increment

    def legal_actions(self) -> LegalActions:
        i = self.pk.actor_index
        if i is None:
            raise IllegalAction("no seat is to act")
        level = self._level()
        bet = self.pk.bets[i]
        stack = self.pk.stacks[i]
        to_call = min(level - bet, stack)
        all_in_to = bet + stack

        engine_can_raise = self.pk.can_complete_bet_or_raise_to()
        rights = self._has_raise_rights(i, level)
        can_raise = engine_can_raise and rights and all_in_to > level
        min_full = max_to = short_all_in = all_in_raise = pot_to = None
        reason = None
        if can_raise:
            full = level + self._full_increment
            pot_formula = level + self.pk.total_pot_amount + (level - bet)
            if self.profile.betting == "pot_limit":
                cap = min(all_in_to, max(full, pot_formula))
            else:
                cap = all_in_to
            engine_max = self.pk.max_completion_betting_or_raising_to_amount
            if engine_max is None or engine_max < cap:
                raise EngineDiscrepancy(f"PokerKit max raise {engine_max} below adapter cap {cap}")
            if all_in_to < full:
                short_all_in = all_in_to
                all_in_raise = all_in_to
            else:
                min_full, max_to = full, cap
                if cap == all_in_to:
                    all_in_raise = all_in_to
                if pot_formula >= full:
                    pot_to = min(pot_formula, cap)
        elif to_call > 0 or level > bet:
            if stack <= level - bet:
                reason = "stack covers only a call"
            elif not engine_can_raise and not self._others_can_respond(i):
                reason = "no opponent can respond to a raise"
            elif not rights:
                reason = "betting was not reopened: raises since your last action total less than a full raise"
        return LegalActions(
            fold=to_call > 0,
            check=to_call == 0,
            call_amount=to_call if to_call > 0 else None,
            can_raise=can_raise,
            min_full_raise_to=min_full,
            max_raise_to=max_to,
            short_all_in_raise_to=short_all_in,
            all_in_raise_to=all_in_raise,
            pot_raise_to=pot_to,
            raise_blocked_reason=reason,
        )

    def _others_can_respond(self, i: int) -> bool:
        return any(
            j != i and self.pk.statuses[j] and self.pk.stacks[j] > 0 for j in range(len(self.seats))
        )

    # ----------------------------------------------------------------- actions

    def apply(self, action: Action, *, auto_reason: str | None = None) -> list[HandEvent]:
        start = len(self.events)
        legal = self.legal_actions()
        if not legal.allows(action):
            raise IllegalAction(f"{action.type} {action.amount or ''} is not legal here".strip())
        i = self.pk.actor_index
        assert i is not None
        seat = self.seats[i]
        level = self._level()
        bet = self.pk.bets[i]
        street = self.street
        corrected = False
        engine_level = max(self.pk.bets)

        match action.type:
            case "fold":
                _strict(self.pk.fold)
                self.folded.append(seat)
            case "check":
                op = _strict(self.pk.check_or_call)
                if op.amount != 0:
                    raise EngineDiscrepancy("check moved chips")
                self._acted_level[i] = level
            case "call":
                assert legal.call_amount is not None
                target = bet + legal.call_amount
                if target > max(self.pk.bets):
                    corrected = True
                    if not self.pk.can_complete_bet_or_raise_to(target):
                        raise EngineDiscrepancy(f"PokerKit refused nominal-BB call to {target}")
                    _strict(self.pk.complete_bet_or_raise_to, target)
                else:
                    op = _strict(self.pk.check_or_call)
                    if op.amount != legal.call_amount:
                        raise EngineDiscrepancy(f"call priced {op.amount}, adapter {legal.call_amount}")
                self._acted_level[i] = level
            case "raise_to":
                target = action.amount
                assert target is not None
                if not self.pk.can_complete_bet_or_raise_to(target):
                    raise EngineDiscrepancy(f"PokerKit refused raise_to {target} offered by adapter")
                increment = target - level
                _strict(self.pk.complete_bet_or_raise_to, target)
                if increment >= self._full_increment:
                    self._full_increment = increment
                elif level == 0 and self.local_rules.short_bet_reopens_checker:
                    for j, acted in list(self._acted_level.items()):
                        if acted == 0:
                            del self._acted_level[j]
                self._acted_level[i] = target

        if level > engine_level:
            # PokerKit measured this action against the short posted blind, so its
            # raise tracking is wrong; reset it to the most permissive values.
            self.pk.completion_betting_or_raising_amount = 0
            self.pk.consecutive_short_all_in_amounts.clear()

        data = {
            "seat": seat,
            "street": street,
            "action": action.model_dump(exclude_none=True),
            "added": self.pk.bets[i] - bet if action.type != "fold" else 0,
            "street_committed": self.pk.bets[i] if action.type != "fold" else bet,
        }
        if corrected:
            data["nominal_big_blind_call"] = True
        if auto_reason:
            data["auto"] = auto_reason
        self._emit("ActionApplied", data)
        self._advance()
        return self.events[start:]

    # -------------------------------------------------------------- automation

    def _emit(self, type_: str, data: dict) -> None:
        self.events.append(HandEvent(type_, data))

    def _draw(self, count: int = 1) -> list[str]:
        if self._next_card + count > len(self.deck):
            raise DeckError("deck exhausted")
        cards = list(self.deck[self._next_card : self._next_card + count])
        engine_next = [repr(c) for c in list(self.pk.deck_cards)[:count]]
        if engine_next != cards:
            raise EngineDiscrepancy("PokerKit deck diverged from the stored deck")
        self._next_card += count
        return cards

    def _advance(self) -> None:
        pk = self.pk
        while pk.status:
            if pk.street_index is not None and pk.street_index != self._street_index:
                self._reset_street(pk.street_index)
            if pk.actor_index is not None:
                i = pk.actor_index
                if not self._others_can_respond(i) and self._level() == pk.bets[i]:
                    self.apply(Action(type="check"), auto_reason="no_contestable_action")
                    return
                return
            if pk.can_post_ante():
                op = _strict(pk.post_ante)
                seat = self.seats[op.player_index]
                if op.amount != self.ante_owed.get(seat):
                    raise EngineDiscrepancy(f"PokerKit posted ante {op.amount} for seat {seat}, "
                                            f"adapter owes {self.ante_owed.get(seat, 0)}")
                self._emit("AntePosted", {"seat": seat, "amount": op.amount})
            elif pk.can_post_blind_or_straddle():
                op = _strict(pk.post_blind_or_straddle)
                self._emit("BlindPosted", {"seat": self.seats[op.player_index], "amount": op.amount})
            elif pk.can_burn_card():
                (card,) = self._draw()
                _strict(pk.burn_card, card)
                self.burns.append(card)
                self._emit("CardBurned", {"card": card})
            elif pk.can_deal_hole():
                i = pk.hole_dealee_index
                (card,) = self._draw()
                _strict(pk.deal_hole, card)
                self.hole_cards[self.seats[i]].append(card)
                self._emit("HoleDealt", {"seat": self.seats[i], "card": card})
            elif pk.can_deal_board():
                count = pk.board_dealing_count
                cards = self._draw(count)
                _strict(pk.deal_board, "".join(cards))
                self.board.extend(cards)
                street = {3: "flop", 4: "turn", 5: "river"}[len(self.board)]
                self._emit("BoardDealt", {"street": street, "cards": cards})
            elif pk.can_collect_bets():
                before = list(pk.stacks)
                op = _strict(pk.collect_bets)
                self._emit(
                    "BetsCollected",
                    {"bets": {self.seats[j]: b for j, b in enumerate(op.bets) if b}},
                )
                for j, (a, b) in enumerate(zip(before, pk.stacks)):
                    if b > a:
                        self._emit("ExcessReturned", {"seat": self.seats[j], "amount": b - a})
            elif pk.can_show_or_muck_hole_cards():
                op = _strict(pk.show_or_muck_hole_cards, True)
                seat = self.seats[op.player_index]
                cards = tuple(repr(c) for c in op.hole_cards)
                self.revealed[seat] = cards
                self._emit("CardsRevealed", {"seat": seat, "cards": list(cards)})
            elif pk.can_kill_hand():
                self._capture_showdown()
                op = _strict(pk.kill_hand)
                self._emit("HandKilled", {"seat": self.seats[op.player_index]})
            elif pk.can_push_chips():
                self._capture_showdown()
                op = _strict(pk.push_chips)
                awards = {self.seats[j]: a for j, a in enumerate(op.amounts) if a}
                for seat, amount in awards.items():
                    self._engine_awards[seat] = self._engine_awards.get(seat, 0) + amount
                self._emit("EngineChipsPushed", {"pot_index": op.pot_index, "awards": awards})
            elif pk.can_pull_chips():
                op = _strict(pk.pull_chips)
                self._emit("ChipsPulled", {"seat": self.seats[op.player_index], "amount": op.amount})
            else:
                raise EngineDiscrepancy("PokerKit state has no pending operation but is not terminal")
        self._finish()

    def _capture_showdown(self) -> None:
        """Snapshot live hands before PokerKit kills losing hands or pushes chips."""
        if self._showdown_hands is not None:
            return
        pk = self.pk
        self._showdown_hands = {}
        if sum(pk.statuses) > 1:
            for i, seat in enumerate(self.seats):
                if pk.statuses[i]:
                    self._showdown_hands[seat] = pk.get_hand(i, 0, 0)

    def _settle_by_level(self) -> list[PotResult]:
        """Spec settlement: one pot per contribution level, odd chips one each clockwise.

        Engine index order is clockwise from the first seat after the button
        (heads-up: BB then button), so it is also the odd-chip order.
        """
        contributions = self._contributions()
        live = [s for s in self.seats if s not in self.folded]
        if len(live) == 1:
            total = sum(contributions.values())
            return [PotResult(total, (live[0],), {live[0]: total})] if total else []
        dead = self.dead_money()
        for seat, amount in dead.items():
            contributions[seat] -= amount
        hands = self._showdown_hands or {}
        pots: list[PotResult] = []
        previous = 0
        for level in sorted({c for c in contributions.values() if c > 0}):
            contributors = [s for s in self.seats if contributions[s] >= level]
            if len(contributors) < 2:
                raise EngineDiscrepancy("unmatched excess left in the pot after collection")
            amount = (level - previous) * len(contributors)
            if previous == 0:
                amount += sum(dead.values())
            previous = level
            eligible = tuple(s for s in contributors if s in hands)
            if not eligible:
                raise EngineDiscrepancy("pot layer with no eligible player")
            best = max(hands[s] for s in eligible)
            winners = [s for s in eligible if hands[s] == best]
            share, remainder = divmod(amount, len(winners))
            awards = {w: share + (1 if k < remainder else 0) for k, w in enumerate(winners)}
            pots.append(PotResult(amount, eligible, awards))
        return pots

    def _engine_rule_awards(self) -> dict[int, int]:
        """What PokerKit's own split rule pays for the spec pots (assumptions D5, D6).

        PokerKit forms its pots after killing every hand that wins nothing, so adjacent
        pot layers merge when their eligible sets agree on the surviving hands. Each
        merged pot is split evenly among its tied winners, and the whole remainder goes
        to the first of them in engine order. The two settlements may then differ only
        in where the odd chips land, by up to ``winners - 1`` chips per pot for one seat.
        """
        survivors = {s for pot in self.pots for s in pot.awards}
        merged: list[tuple[int, tuple[int, ...], list[int]]] = []
        for pot in self.pots:
            key = tuple(s for s in pot.eligible if s in survivors)
            if merged and merged[-1][1] == key:
                merged[-1] = (merged[-1][0] + pot.amount, key, merged[-1][2])
            else:
                merged.append((pot.amount, key, list(pot.awards)))
        out: dict[int, int] = {}
        for amount, _, winners in merged:
            share, remainder = divmod(amount, len(winners))
            for k, seat in enumerate(winners):
                out[seat] = out.get(seat, 0) + share + (remainder if k == 0 else 0)
        return out

    def _finish(self) -> None:
        self.pots = self._settle_by_level()
        spec_awards: dict[int, int] = {}
        for pot in self.pots:
            for seat, amount in pot.awards.items():
                spec_awards[seat] = spec_awards.get(seat, 0) + amount
        if len(self.folded) < len(self.seats) - 1:
            engine_rule = {s: v for s, v in self._engine_rule_awards().items() if v}
            engine = {s: v for s, v in self._engine_awards.items() if v}
            if engine != engine_rule:
                raise EngineDiscrepancy(
                    f"settlement differs beyond odd chips: spec {spec_awards}, PokerKit {self._engine_awards}, "
                    f"PokerKit's rule applied to the spec pots {engine_rule}"
                )
            winners = {s for pot in self.pots for s in pot.awards}
            correction = {}
            for seat in self.seats:
                diff = spec_awards.get(seat, 0) - self._engine_awards.get(seat, 0)
                if diff:
                    if seat not in winners:
                        raise EngineDiscrepancy(f"odd-chip correction pays seat {seat}, which won no pot")
                    correction[seat] = diff
            if sum(correction.values()) != 0:
                raise EngineDiscrepancy("odd-chip correction is not chip-neutral")
            self._correction = correction
            if correction:
                self._emit("OddChipCorrection", {"transfers": correction, "engine_awards": dict(self._engine_awards)})
        for index, pot in enumerate(self.pots):
            self._emit(
                "PotAwarded",
                {"pot_index": index, "amount": pot.amount, "eligible": list(pot.eligible), "awards": pot.awards},
            )
        final = self.final_stacks()
        if sum(final.values()) != self.total_chips:
            raise EngineDiscrepancy("chips not conserved")
        self._emit("HandCompleted", {"final_stacks": final})

    # ----------------------------------------------------------------- results

    def final_stacks(self) -> dict[int, int]:
        return {s: self.pk.stacks[i] + self._correction.get(s, 0) for i, s in enumerate(self.seats)}

    def result(self) -> HandResult:
        if not self.is_complete:
            raise RuntimeError("hand is not complete")
        return HandResult(
            starting_stacks=dict(self.starting_stacks),
            final_stacks=self.final_stacks(),
            contributions=self._contributions(),
            pots=tuple(self.pots),
            board=tuple(self.board),
            revealed=dict(self.revealed),
            folded=tuple(self.folded),
            ended_by_fold=sum(1 for s in self.seats if s not in self.folded) == 1,
        )

    def dead_money(self) -> dict[int, int]:
        """Big-blind antes posted, by seat: in the main pot, which every live hand contests."""
        if not self.dead_ante:
            return {}
        return {ev.data["seat"]: ev.data["amount"] for ev in self.events if ev.type == "AntePosted"}

    def _contributions(self) -> dict[int, int]:
        """Chips each seat left in the pot (after unmatched excess was returned)."""
        out = {s: 0 for s in self.seats}
        for ev in self.events:
            if ev.type == "BetsCollected":
                for seat, amount in ev.data["bets"].items():
                    out[seat] += amount
        return out

    def card_ledger(self) -> dict[str, Sequence[str]]:
        dealt = [c for s in self.seats for c in self.hole_cards[s]]
        return {
            "hole": dealt,
            "burn": list(self.burns),
            "board": list(self.board),
            "undealt": list(self.deck[self._next_card :]),
        }
