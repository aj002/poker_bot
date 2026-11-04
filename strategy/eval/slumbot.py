"""Adapter between Slumbot's public API and the arena's rules engine (heads-up NLHE, 200 BB).

Slumbot (https://slumbot.com) plays 50/100 blinds with 20,000-chip stacks that reset every
hand. Its API speaks an *action string* such as ``b200c/kk/kb400``: ``k`` check, ``c`` call,
``f`` fold, ``b<N>`` bet or raise to a *street total* of N (the arena's ``raise_to``), ``/``
between streets. ``client_pos`` is 0 for the big blind and 1 for the small blind (the button).

This module is the pure part: parsing, formatting, and a *mirror table* that replays a
Slumbot hand through the arena engine so that every action is validated by the same rules
code the arena uses and the bot is shown an ordinary :class:`~arena.domain.observations.Observation`.
It does no networking; see :mod:`strategy.eval.slumbot_run` for the client and the runner.

Mirror design. Slumbot reveals cards progressively, so a hand cannot be dealt up front. The
mirror is therefore rebuilt at each decision: a fresh two-seat table whose deck holds the
cards known so far (hero's hole cards, the board seen so far) and unused placeholders for
everything unknown (the opponent's hole cards, future board cards), then the action prefix
is replayed. Placeholders are never visible to the bot: it sees only the hero's cards and the
dealt board. Hero is always seat 0 and Slumbot seat 1. Hand numbers are offset so each
Slumbot hand has its own ``hand_id``, which the bot's per-hand memo and belief updates need.

Off-distribution settings the mirror chooses on purpose (documented in docs/outside-bots.md):
two seats, one level of 50/100 blinds, a 200 BB stack that resets each hand, an effectively
unbounded hand cap, and winner-take-all points ``[1, 0]`` so that heads-up ICM equity is
linear in chips (chip-EV, which is what Slumbot's cash game pays).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from arena.domain.cards import RANKS, SUITS, full_deck
from arena.domain.models import Action, TournamentConfig
from arena.table.service import ActionRejected, Table

SMALL_BLIND = 50
BIG_BLIND = 100
STACK = 20_000
HERO, VILLAIN = 0, 1
HAND_CAP = 1_000_000
BUDGET_NS = 1_000_000
TABLE_ID = "t_slumbot"
STREET_BOARD = (0, 3, 4, 5)

_TOKEN = re.compile(r"[kcf]|b\d+|/")
_DECK = full_deck(RANKS)

# Engine deal order for a two-seat hand (verified by tests): the big blind's seat is dealt
# first, one card at a time; then burn + flop, burn + turn, burn + river.
BB_HOLE, SB_HOLE = (0, 2), (1, 3)
BOARD_SLOTS = (5, 6, 7, 9, 11)


class ProtocolMismatch(RuntimeError):
    """Slumbot's action string and the arena engine disagree (or the string is malformed)."""


def config() -> TournamentConfig:
    return TournamentConfig(
        initial_player_count=2, starting_stack=STACK, hand_cap=HAND_CAP, hands_per_level=HAND_CAP,
        blind_levels=[[SMALL_BLIND, BIG_BLIND]], payout_points=[1, 0], bot_budget_ns=BUDGET_NS,
    )


# ------------------------------------------------------------------- action strings


def tokenize(action: str) -> list[str]:
    """Split an action string into actions and slashes; anything else is an error."""
    tokens = _TOKEN.findall(action)
    if "".join(tokens) != action:
        raise ProtocolMismatch(f"unparseable action string {action!r}")
    return tokens


def parse_actions(action: str) -> list[Action]:
    """The engine actions in an action string, in order, with street separators dropped."""
    out: list[Action] = []
    for t in tokenize(action):
        if t == "/":
            continue
        if t == "k":
            out.append(Action(type="check"))
        elif t == "c":
            out.append(Action(type="call"))
        elif t == "f":
            out.append(Action(type="fold"))
        else:
            out.append(Action(type="raise_to", amount=int(t[1:])))
    return out


def format_action(action: Action) -> str:
    """The ``incr`` string Slumbot's ``/act`` expects for one engine action."""
    if action.type == "fold":
        return "f"
    if action.type == "check":
        return "k"
    if action.type == "call":
        return "c"
    if action.type == "raise_to":
        return f"b{action.amount}"
    raise ProtocolMismatch(f"cannot send action {action!r}")


def format_actions(actions: list[Action], street_ends: list[int] | None = None) -> str:
    """Join actions into an action string; ``street_ends`` are action counts at which a
    ``/`` follows (used by the in-process fake server and tests)."""
    ends = set(street_ends or [])
    parts = []
    for i, a in enumerate(actions, start=1):
        parts.append(format_action(a))
        if i in ends:
            parts.append("/")
    return "".join(parts)


# ----------------------------------------------------------------------- mirror table


class MirrorTable(Table):
    """A table that plays exactly one hand, numbered ``first_hand``, then completes."""

    def __init__(self, cfg: TournamentConfig, *, first_hand: int, **kw) -> None:
        self._first_hand = first_hand
        super().__init__(cfg, **kw)

    def _start_next_hand(self) -> None:
        if self.hands:
            self._finalize()
            return
        self.hand_number = self._first_hand - 1
        super()._start_next_hand()


def build_deck(client_pos: int, hero_hole: tuple[str, str], board: tuple[str, ...] = (),
               villain_hole: tuple[str, str] | None = None) -> tuple[str, ...]:
    """A full 52-card deck in engine deal order holding the known cards.

    Unknown positions get unused cards in canonical order, so the deck is always valid. The
    known cards are checked for duplicates.
    """
    known = [*hero_hole, *board, *(villain_hole or ())]
    if len(set(known)) != len(known):
        raise ProtocolMismatch(f"duplicate cards in {known}")
    for c in known:
        if len(c) != 2 or c[0] not in RANKS or c[1] not in SUITS:
            raise ProtocolMismatch(f"malformed card {c!r}")
    slots: list[str | None] = [None] * 52
    hero_slots, villain_slots = (SB_HOLE, BB_HOLE) if client_pos == 1 else (BB_HOLE, SB_HOLE)
    for slot, card in zip(hero_slots, hero_hole):
        slots[slot] = card
    if villain_hole is not None:
        for slot, card in zip(villain_slots, villain_hole):
            slots[slot] = card
    for slot, card in zip(BOARD_SLOTS, board):
        slots[slot] = card
    spare = iter(c for c in _DECK if c not in set(known))
    return tuple(c if c is not None else next(spare) for c in slots)


def replay(client_pos: int, hero_hole: tuple[str, str], board: tuple[str, ...], action: str, *,
           first_hand: int, villain_hole: tuple[str, str] | None = None,
           cfg: TournamentConfig | None = None) -> MirrorTable:
    """Build the mirror table and apply every action in ``action`` through the engine.

    Raises :class:`ProtocolMismatch` when the engine rejects an action Slumbot reports, when
    the slashes disagree with the engine's street, or when more actions follow a finished hand.
    """
    if client_pos not in (0, 1):
        raise ProtocolMismatch(f"client_pos {client_pos!r}")
    deck = build_deck(client_pos, hero_hole, board, villain_hole)
    table = MirrorTable(cfg or config(), first_hand=first_hand, deck_source=lambda n, p: deck,
                        initial_button=HERO if client_pos == 1 else VILLAIN, table_id=TABLE_ID)
    for tok in tokenize(action):
        if tok == "/":
            continue
        a = parse_actions(tok)[0]
        seat = table.actor_seat
        if seat is None:
            raise ProtocolMismatch(f"action {tok!r} after the hand ended in {action!r}")
        try:
            table.submit(seat, a)
        except ActionRejected as exc:
            raise ProtocolMismatch(f"engine rejects {tok!r} by seat {seat} in {action!r}: {exc}") from None
    slashes = action.count("/")
    if not table.complete and table.hand is not None:
        street = _street_index(table)
        if slashes != street:
            raise ProtocolMismatch(f"{slashes} street separators but the engine is on street {street} in {action!r}")
    return table


def _street_index(table: Table) -> int:
    return ("preflop", "flop", "turn", "river").index(table.hand.street)


def expected_board(street: int) -> int:
    return STREET_BOARD[street]


def hero_net(table: MirrorTable) -> int | None:
    """Hero's chip result for the mirrored hand, or None while it is unfinished."""
    if not table.hands or table.hands[0].result is None:
        return None
    return int(table.hands[0].result.final_stacks[HERO]) - STACK


@dataclass(frozen=True)
class HandState:
    """What a Slumbot response tells us about the hand, normalised."""

    action: str
    client_pos: int
    hole: tuple[str, str]
    board: tuple[str, ...]
    winnings: int | None
    villain_hole: tuple[str, str] | None

    @classmethod
    def from_response(cls, r: dict) -> HandState:
        hole = tuple(r.get("hole_cards") or ())
        if len(hole) != 2:
            raise ProtocolMismatch(f"response without hole cards: {r!r}")
        vh = tuple(r.get("bot_hole_cards") or ())
        return cls(action=r.get("action", ""), client_pos=r.get("client_pos"), hole=hole,  # type: ignore[arg-type]
                   board=tuple(r.get("board") or ()), winnings=r.get("winnings"),
                   villain_hole=vh if len(vh) == 2 else None)  # type: ignore[arg-type]
