"""An in-process stand-in for Slumbot's API, built straight on the arena engine with a fully
known deck. It speaks the same responses (action strings with street separators, ``client_pos``,
``winnings`` and ``bot_hole_cards`` at the end) and its opponent plays random legal actions,
with frequent all-ins, so the adapter meets every shape of hand."""

from __future__ import annotations

import random

from arena.domain.cards import RANKS, full_deck
from arena.domain.models import Action
from arena.table.service import Table
from strategy.eval import slumbot as S


class FakeSlumbot:
    min_interval = 0.0

    def __init__(self, seed: int = 0, allin_rate: float = 0.12, style: str = "random") -> None:
        self.style = style
        self.rng = random.Random(seed)
        self.allin_rate = allin_rate
        self.hands = 0
        self.token = "fake-token"
        self.requests = 0
        self.retries = 0
        self.table: Table | None = None
        self.parts: list[str] = []
        self.last_street = 0
        self.client_pos = 0

    def new_hand(self) -> dict:
        self.requests += 1
        self.hands += 1
        self.client_pos = self.hands % 2
        deck = list(full_deck(RANKS))
        self.rng.shuffle(deck)
        d = tuple(deck)
        self.table = S.MirrorTable(S.config(), first_hand=1, deck_source=lambda n, p: d,
                           initial_button=S.HERO if self.client_pos == 1 else S.VILLAIN)
        self.parts, self.last_street = [], 0
        return self._advance()

    def act(self, incr: str) -> dict:
        self.requests += 1
        t = self.table
        assert t is not None and t.actor_seat == S.HERO, "out of turn"
        legal = t.legal_actions()
        a = self._parse(incr)
        if a.type == "check" and not legal.check:
            return {"error_msg": "Illegal check", "old_action": "".join(self.parts)}
        if a.type == "call" and not legal.call_amount:
            return {"error_msg": "Illegal call", "old_action": "".join(self.parts)}
        self._apply(S.HERO, a)
        return self._advance()

    @staticmethod
    def _parse(incr: str) -> Action:
        if incr == "k":
            return Action(type="check")
        if incr == "c":
            return Action(type="call")
        if incr == "f":
            return Action(type="fold")
        assert incr[0] == "b", incr
        return Action(type="raise_to", amount=int(incr[1:]))

    def _street(self) -> int:
        t = self.table
        assert t is not None
        if t.hand is None:
            return 3
        return ("preflop", "flop", "turn", "river").index(t.hand.street)

    def _apply(self, seat: int, a: Action) -> None:
        t = self.table
        assert t is not None
        street_before = self._street()
        while self.last_street < street_before:
            self.parts.append("/")
            self.last_street += 1
        t.submit(seat, a)
        self.parts.append(S.format_action(a))

    def _villain(self) -> Action:
        t = self.table
        assert t is not None
        legal = t.legal_actions()
        if self.style == "station":
            return Action(type="call") if legal.call_amount else Action(type="check")
        r = self.rng.random()
        if legal.can_raise and r < self.allin_rate:
            return Action(type="raise_to", amount=legal.max_raise_to)
        if legal.can_raise and r < 0.4:
            hi = min(legal.max_raise_to, legal.min_full_raise_to * 6)
            return Action(type="raise_to", amount=self.rng.randint(legal.min_full_raise_to, hi))
        if legal.call_amount:
            return Action(type="fold") if r > 0.85 else Action(type="call")
        return Action(type="check")

    def _advance(self) -> dict:
        t = self.table
        assert t is not None
        while not t.complete and t.hands[-1].result is None and t.actor_seat == S.VILLAIN:
            self._apply(S.VILLAIN, self._villain())
        res = t.hands[-1].result
        hole = list(t.hands[-1].deck[i] for i in (S.SB_HOLE if self.client_pos == 1 else S.BB_HOLE))
        out = {"action": "".join(self.parts), "client_pos": self.client_pos, "hole_cards": hole,
               "board": self._board(), "token": self.token}
        if res is not None:
            action = "".join(self.parts)
            if action.endswith("c") and res.revealed and not res.ended_by_fold:
                action += "/" * (3 - action.count("/"))
            out["action"] = action
            out["winnings"] = int(res.final_stacks[S.HERO]) - S.STACK
            out["board"] = list(res.board)
            if res.revealed and S.VILLAIN in res.revealed:
                out["bot_hole_cards"] = list(res.revealed[S.VILLAIN])
        elif self._street() > self.last_street:
            out["action"] += "/" * (self._street() - self.last_street)
        return out

    def _board(self) -> list[str]:
        t = self.table
        assert t is not None
        if t.hand is None:
            return []
        return list(t.hand.board)
