"""A real game played with physical cards and chips, replayed through the arena engine for the final bot.

A session is a plain JSON document (setup plus, per hand, the chips, the button and an ordered log of what the
user entered: hole cards, actions, board cards, who won). Everything else is recomputed by replay, so undo is just
deleting the last log entry and a restarted server loses nothing.

Each hand is dealt by the engine from a deck built around what the user told us: the hero's two cards and the
board cards entered so far sit at the positions the engine will deal them from, every other card is a stand-in
the bot never sees. The bot gets exactly the observation it gets at an arena table, with the hand number and
decision ids unique across the session, so its explicit state (its opponent beliefs, keyed by seat) carries from
hand to hand the way it does in the arena. No clock is read and there is no decision deadline.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from arena.agents.sdk import decode_state, encode_state, load_bot, split_note
from arena.domain.cards import RANKS, SUITS, full_deck
from arena.domain.models import Action, TournamentConfig
from arena.rules.base import HandResult, PotResult
from arena.rules.pokerkit_adapter import HandSpec, PokerKitHand
from arena.rules.profiles import get_profile
from arena.table.positions import positions_from_button
from arena.table.projections import observation
from arena.table.service import ActionRejected, Table

RULESET = "practice_nlhe"
HAND_CAP = 100_000
SAMPLES = 120
MIX_STEP = 5
PAYOUT_PRESETS = {"winner": [100], "top2": [65, 35], "top3": [50, 30, 20]}
FINAL_BOT_ENTRY = Path(__file__).resolve().parents[2] / "strategy" / "final" / "bots" / "final.py"
TABLE_ID = "t_advisor"


class AdvisorError(Exception):
    def __init__(self, code: str, message: str, status: int = 422) -> None:
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


class _HandFinished(Exception):
    """Raised from the engine's next-hand hook: the engine would start another hand, we only ever want one."""


# ----------------------------------------------------------------------------- validation


def clean_card(card: Any) -> str:
    if not isinstance(card, str) or len(card) != 2 or card[0].upper() not in RANKS or card[1].lower() not in SUITS:
        raise AdvisorError("invalid_card", f"{card!r} is not a card; use a rank 2-9 T J Q K A and a suit c d h s")
    return card[0].upper() + card[1].lower()


def _int(value: Any, name: str, *, low: int = 0, high: int = 1_000_000_000) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise AdvisorError("invalid_number", f"{name} must be a whole number from {low} to {high}")
    return value


def clean_setup(raw: dict[str, Any]) -> dict[str, Any]:
    players = _int(raw.get("players"), "players", low=2, high=8)
    hero = _int(raw.get("hero"), "your seat", low=0, high=players - 1)
    sb = _int(raw.get("sb"), "small blind", low=1)
    bb = _int(raw.get("bb"), "big blind", low=1)
    if sb > bb:
        raise AdvisorError("invalid_blinds", "the small blind cannot be bigger than the big blind")
    ante = _int(raw.get("ante", 0), "ante", low=0)
    posting = raw.get("ante_posting") if ante else None
    if ante and posting not in ("classic", "big_blind"):
        raise AdvisorError("invalid_ante", "ante_posting must be 'classic' (everyone) or 'big_blind' (the big blind pays it)")
    mode = raw.get("mode", "cash")
    if mode not in ("cash", "tournament"):
        raise AdvisorError("invalid_mode", "mode must be cash or tournament")
    stacks = raw.get("stacks")
    if not isinstance(stacks, list) or len(stacks) != players:
        raise AdvisorError("invalid_stacks", "give one starting stack per player")
    stacks = [_int(s, "a stack", low=0) for s in stacks]
    if sum(1 for s in stacks if s > 0) < 2:
        raise AdvisorError("invalid_stacks", "at least two players need chips")
    if stacks[hero] <= 0:
        raise AdvisorError("invalid_stacks", "you need chips to play")
    payouts = raw.get("payouts")
    if mode == "tournament":
        payouts = PAYOUT_PRESETS.get(payouts, None) if isinstance(payouts, str) else payouts
        payouts = list(payouts or PAYOUT_PRESETS["winner"])
        payouts = [_int(p, "a payout", low=0) for p in payouts]
        if sum(payouts) <= 0 or any(a < b for a, b in zip(payouts, payouts[1:])):
            raise AdvisorError("invalid_payouts", "payouts must be largest first and add up to more than zero")
    else:
        payouts = PAYOUT_PRESETS["winner"]
    return {"players": players, "hero": hero, "sb": sb, "bb": bb, "ante": ante, "ante_posting": posting,
            "mode": mode, "payouts": payouts, "stacks": stacks}


# ----------------------------------------------------------------------------- engine helpers


def _ring(hand: dict[str, Any]) -> list[int]:
    return sorted(int(s) for s, v in hand["stacks"].items() if v > 0)


def config_for(setup: dict[str, Any], hand: dict[str, Any]) -> TournamentConfig:
    ring = _ring(hand)
    n = len(ring)
    payouts = (list(setup["payouts"]) + [0] * 8)[:n]
    if sum(payouts) <= 0:
        payouts = [100] + [0] * (n - 1)
    ante = hand.get("ante", 0)
    data: dict[str, Any] = {
        "ruleset_id": RULESET, "initial_player_count": n, "seats": ring, "hand_cap": HAND_CAP,
        "hands_per_level": HAND_CAP, "starting_stack": max(hand["stacks"].values()),
        "blind_levels": [[hand["sb"], hand["bb"]]], "payout_points": payouts,
    }
    if ante:
        data.update(antes=[ante], ante_posting=hand["ante_posting"])
    return TournamentConfig.model_validate(data)


_LAYOUTS: dict[tuple[tuple[int, ...], int], dict[str, Any]] = {}


def _layout(ring: tuple[int, ...], button: int, hero: int) -> dict[str, Any]:
    """Where the engine deals the hero's cards and each board card from: probed once per seating."""
    key = (ring, button)
    if key in _LAYOUTS:
        return _LAYOUTS[key]
    profile = get_profile(RULESET)
    deck = tuple(full_deck(profile.ranks))
    live = set(ring)
    pos = positions_from_button(ring, live, button)
    hand = PokerKitHand(HandSpec(
        profile=profile, seat_ring=ring, stacks={s: 1000 for s in ring}, button=pos.button,
        small_blind_seat=pos.small_blind_seat(sorted(live)), big_blind_seat=pos.big_blind, small_blind=1,
        big_blind=2, deck=deck))
    while not hand.is_complete:
        legal = hand.legal_actions()
        hand.apply(Action(type="check") if legal.check else Action(type="call"))
    index = {c: i for i, c in enumerate(deck)}
    layout = {
        "hole": {s: [index[c] for c in hand.hole_cards[s]] for s in ring},
        "board": [[index[c] for c in e.data["cards"]] for e in hand.events if e.type == "BoardDealt"],
    }
    _LAYOUTS[key] = layout
    return layout


def build_deck(ring: tuple[int, ...], button: int, hero: int, hole: list[str] | None, board: list[str]) -> tuple[str, ...]:
    layout = _layout(ring, button, hero)
    slots: dict[int, str] = {}
    if hole:
        for i, c in zip(layout["hole"][hero], hole, strict=True):
            slots[i] = c
    flat = [i for street in layout["board"] for i in street]
    for i, c in zip(flat, board, strict=False):
        slots[i] = c
    spare = iter(c for c in full_deck(get_profile(RULESET).ranks) if c not in set(slots.values()))
    return tuple(slots[i] if i in slots else next(spare) for i in range(52))


def make_table(config: TournamentConfig, hand: dict[str, Any], hand_number: int, deck: tuple[str, ...]) -> Table:
    """A table positioned at ``hand_number`` with the session's chips, button and deck; it stops after this hand."""
    holder: list[Table] = []
    calls = {"n": 0}

    def hook(t: Table) -> None:
        holder.append(t)
        calls["n"] += 1
        if calls["n"] > 1:
            raise _HandFinished
        t.hand_number = hand_number - 1
        t._decision_seq = hand_number * 100

    stacks = {int(s): v for s, v in hand["stacks"].items() if v > 0}
    try:
        table = Table(config, table_id=TABLE_ID, deck_source=lambda n, profile: deck, initial_button=hand["button"],
                      start_stacks=stacks, on_hand_start=hook)
    except _HandFinished:
        table = holder[0]
    return table


def dealt_board(table: Table) -> int:
    record = table.hands[-1]
    return sum(len(e.data["cards"]) for e in record.events if e.type == "BoardDealt")


def result_of(table: Table) -> HandResult | None:
    res = table.hands[-1].result
    if res is None or res.ended_by_fold:
        return res
    merged: list[PotResult] = []
    for pot in res.pots:
        if merged and merged[-1].eligible == pot.eligible:
            merged[-1] = replace(merged[-1], amount=merged[-1].amount + pot.amount)
        else:
            merged.append(pot)
    return replace(res, pots=tuple(merged))


# ----------------------------------------------------------------------------- the bot


_BOT_LOCK = threading.RLock()
_BOT: tuple[Any, Any] | None = None


def final_bot() -> Any:
    global _BOT
    with _BOT_LOCK:
        if _BOT is None:
            _BOT = load_bot(FINAL_BOT_ENTRY, module_name="arena_advisor_final_bot")
        return _BOT[0]


@dataclass
class Sim:
    table: Table
    hole: list[str] | None
    board: list[str]
    state_after: Any
    ring: list[int]
    pending: tuple[Any, Any] | None = None
    need_board: int | None = None
    complete: bool = False
    result: HandResult | None = None
    awards: list[list[int]] | None = None
    lines: list[str] = field(default_factory=list)


def new_session(raw: dict[str, Any]) -> dict[str, Any]:
    setup = clean_setup(raw)
    session = {"id": secrets.token_hex(8), "seed": secrets.token_hex(16), "setup": setup, "hands": []}
    first = _fresh_hand(setup, 1, {str(i): s for i, s in enumerate(setup["stacks"])}, None)
    if raw.get("button") is not None:
        button = _int(raw["button"], "the dealer button", low=0, high=setup["players"] - 1)
        if button not in _ring(first):
            raise AdvisorError("invalid_button", "the dealer button must be on a player with chips")
        first["button"] = button
    session["hands"].append(first)
    return session


def _fresh_hand(setup: dict[str, Any], number: int, stacks: dict[str, int], previous_button: int | None) -> dict[str, Any]:
    ring = sorted(int(s) for s, v in stacks.items() if v > 0)
    if previous_button is None:
        button = ring[0]
    else:
        later = [s for s in ring if s > previous_button]
        button = later[0] if later else ring[0]
    return {"n": number, "button": button, "stacks": dict(stacks), "sb": setup["sb"], "bb": setup["bb"],
            "ante": setup["ante"], "ante_posting": setup["ante_posting"], "log": []}


def _digest(parent: str, hand: dict[str, Any]) -> str:
    return hashlib.sha256((parent + json.dumps(hand, sort_keys=True)).encode()).hexdigest()


class Advisor:
    """Replays sessions through the engine and asks the final bot at the hero's turns."""

    def __init__(self, samples: int = SAMPLES) -> None:
        self.samples = samples
        self._cache: dict[str, dict[int, tuple[str, Any]]] = {}

    # -- bot plumbing
    def _bot_state0(self, session: dict[str, Any]) -> Any:
        from arena.api.seats import live_table_info

        hand = session["hands"][0]
        info = live_table_info(config_for(session["setup"], hand), TABLE_ID, session["setup"]["hero"], session["seed"])
        return decode_state(encode_state(final_bot().initial_state(info)))

    @staticmethod
    def _decide(obs: Any, state: Any) -> tuple[dict[str, Any], str | None, Any]:
        reply, new_state = final_bot().decide(obs, state)
        action, note = split_note(reply)
        return action, note, decode_state(encode_state(new_state))

    def state_in(self, session: dict[str, Any], index: int) -> Any:
        """The bot's state when hand ``index`` starts: its decisions in every earlier hand replayed in order."""
        cache = self._cache.setdefault(session["id"], {})
        chain = "root"
        state = None
        for i in range(index + 1):
            hand = session["hands"][i]
            chain = _digest(chain, hand)
            if i == index:
                break
            hit = cache.get(i + 1)
            if hit is not None and hit[0] == chain:
                state = hit[1]
                continue
            if state is None and i == 0:
                state = self._bot_state0(session)
            state = self.simulate(session, i, state).state_after
            cache[i + 1] = (chain, state)
        if index == 0:
            return self._bot_state0(session)
        return state

    # -- one hand
    def simulate(self, session: dict[str, Any], index: int, state_in: Any, log: list[dict[str, Any]] | None = None,
                 *, advice: bool = False) -> Sim:
        setup, hand = session["setup"], session["hands"][index]
        hero = setup["hero"]
        entries = hand["log"] if log is None else log
        ring = _ring(hand)
        hole = next((e["cards"] for e in entries if e["k"] == "hole"), None)
        board = [c for e in entries if e["k"] == "board" for c in e["cards"]]
        if hero not in ring:
            raise AdvisorError("hero_out", "you have no chips in this hand")
        if hand["button"] not in ring:
            raise AdvisorError("invalid_button", "the dealer button must be on a player with chips")
        deck = build_deck(tuple(ring), hand["button"], hero, hole, board)
        config = config_for(setup, hand)
        table = make_table(config, hand, hand["n"], deck)
        sim = Sim(table, hole, board, state_in, ring)
        entered = 0
        for pos, e in enumerate(entries):
            if e["k"] == "hole":
                if pos != 0:
                    raise AdvisorError("hole_first", "your cards come first")
                continue
            if hole is None:
                raise AdvisorError("hole_first", "enter your two cards first")
            if e["k"] == "act":
                seat = e["seat"]
                if table.actor_seat != seat:
                    raise AdvisorError("not_their_turn", f"it is not seat {seat + 1}'s turn")
                if seat == hero:
                    sim.state_after = self._decide(observation(table, hero), sim.state_after)[2]
                action = Action(type=e["type"], **({"amount": e["amount"]} if e["type"] == "raise_to" else {}))
                try:
                    table.submit(seat, action)
                except ActionRejected as exc:
                    raise AdvisorError("illegal_action", exc.args[0] if exc.args else str(exc)) from None
                except _HandFinished:
                    pass
            elif e["k"] == "board":
                if table.hands[-1].result is not None and table.actor_seat is None:
                    raise AdvisorError("hand_over", "the hand is over; no more board cards")
                if dealt_board(table) - entered != len(e["cards"]):
                    need = dealt_board(table) - entered
                    raise AdvisorError("wrong_board", f"enter {need} board card{'s' if need != 1 else ''} now")
                entered += len(e["cards"])
            elif e["k"] == "award":
                sim.awards = e["pots"]
        sim.complete = table.hands[-1].result is not None
        sim.result = result_of(table)
        if not sim.complete:
            if hole is not None and dealt_board(table) > entered:
                sim.need_board = dealt_board(table) - entered
            elif hole is not None and table.actor_seat == hero:
                obs = observation(table, hero)
                sim.pending = (obs, sim.state_after)
        return sim

    # -- views
    def view(self, session: dict[str, Any]) -> dict[str, Any]:
        index = len(session["hands"]) - 1
        state = self.state_in(session, index)
        sim = self.simulate(session, index, state)
        return self._render(session, index, sim)

    def _render(self, session: dict[str, Any], index: int, sim: Sim) -> dict[str, Any]:
        from arena.advisor.render import render

        return render(self, session, index, sim)
