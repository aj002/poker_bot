"""Golden-fixture builder and runner.

The deck builder encodes the spec's dealing rule directly (one card per player
clockwise starting left of the button, heads-up starting with the big blind,
one burn before each board street) rather than asking the adapter.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from arena.domain.cards import full_deck, parse_cards
from arena.domain.models import Action, LocalRules
from arena.rules.base import IllegalAction
from arena.rules.pokerkit_adapter import HandSpec, PokerKitHand
from arena.rules.profiles import get_profile
from tests.support.oracle_bridge import assert_matches_oracles

FIXTURE_ROOT = Path(__file__).resolve().parent.parent / "golden" / "fixtures"


def load_fixtures(group: str) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted((FIXTURE_ROOT / group).glob("*.json"))]


def deal_order(ring: list[int], live: list[int], button: int, big_blind: int) -> list[int]:
    if len(live) == 2:
        return [big_blind, button]
    i = ring.index(button)
    rotated = ring[i + 1 :] + ring[: i + 1]
    return [s for s in rotated if s in live]


def build_deck(
    ranks: str,
    order: list[int],
    hole_count: int,
    hole: dict[int, tuple[str, ...]],
    board: tuple[str, ...],
) -> tuple[str, ...]:
    reserved = [c for cards in hole.values() for c in cards] + list(board)
    if len(set(reserved)) != len(reserved):
        raise ValueError("fixture repeats a card")
    filler = [c for c in full_deck(ranks) if c not in reserved]
    filler.reverse()
    deck: list[str] = []
    for k in range(hole_count):
        for seat in order:
            given = hole.get(seat)
            deck.append(given[k] if given else filler.pop())
    board_left = list(board)
    for count in (3, 1, 1):
        deck.append(filler.pop())
        for _ in range(count):
            deck.append(board_left.pop(0) if board_left else filler.pop())
    deck.extend(reversed(filler))
    return tuple(deck)


def make_hand(fx: dict[str, Any]) -> PokerKitHand:
    setup = fx["setup"]
    profile = get_profile(fx.get("ruleset", "practice_nlhe"))
    ring = sorted(setup.get("seats", [int(s) for s in setup["stacks"]]))
    stacks = {int(s): v for s, v in setup["stacks"].items()}
    live = [s for s in ring if s in stacks]
    hole = {int(s): parse_cards(c) for s, c in setup.get("hole", {}).items()}
    board = parse_cards(setup.get("board", ""))
    order = deal_order(ring, live, setup["button"], setup["big_blind_seat"])
    deck = build_deck(profile.ranks, order, profile.hole_card_count, hole, board)
    sb, bb = setup.get("blinds", [10, 20])
    spec = HandSpec(
        profile=profile,
        seat_ring=tuple(ring),
        stacks=stacks,
        button=setup["button"],
        small_blind_seat=setup.get("small_blind_seat"),
        big_blind_seat=setup["big_blind_seat"],
        small_blind=sb,
        big_blind=bb,
        deck=deck,
        local_rules=LocalRules(**fx.get("local_rules", {})),
        ante=setup.get("ante", 0),
        ante_posting=setup.get("ante_posting"),
    )
    return PokerKitHand(spec)


def _seat_map(d: dict[str, Any]) -> dict[int, Any]:
    return {int(k): v for k, v in d.items()}


def check_state(hand: PokerKitHand, expected: dict[str, Any], where: str) -> None:
    players = {p.seat: p for p in hand.players()}
    for key, want in expected.items():
        if key == "street":
            got: Any = hand.street
        elif key == "actor":
            got = hand.actor_seat
        elif key == "pot_total":
            got = hand.pot_total()
        elif key == "board":
            got = " ".join(hand.board)
        elif key in ("street_committed", "hand_committed", "stack", "status"):
            want = _seat_map(want)
            got = {s: getattr(players[s], key) for s in want}
        elif key == "complete":
            got = hand.is_complete
        else:
            raise KeyError(f"unknown state key {key}")
        assert got == want, f"{where}: {key} expected {want!r}, got {got!r}"


def check_legal(hand: PokerKitHand, expected: dict[str, Any], where: str) -> None:
    legal = hand.legal_actions().model_dump()
    for key, want in expected.items():
        assert legal[key] == want, f"{where}: legal.{key} expected {want!r}, got {legal[key]!r} ({legal})"


def check_down(hand: PokerKitHand) -> None:
    while not hand.is_complete:
        legal = hand.legal_actions()
        hand.apply(Action(type="check") if legal.check else Action(type="call"))


def run_fixture(fx: dict[str, Any]) -> PokerKitHand:
    hand = make_hand(fx)
    for seat, cards in fx["setup"].get("hole", {}).items():
        assert tuple(hand.hole_cards[int(seat)]) == parse_cards(cards), f"{fx['id']}: seat {seat} dealt {hand.hole_cards[int(seat)]}"
    for n, step in enumerate(fx.get("steps", [])):
        where = f"{fx['id']} step {n}"
        if "state" in step:
            check_state(hand, step["state"], where)
        if "actor" in step:
            assert hand.actor_seat == step["actor"], f"{where}: actor {hand.actor_seat} != {step['actor']}"
        if "legal" in step:
            check_legal(hand, step["legal"], where)
        if "reject" in step:
            before = (hand.actor_seat, hand.chips_in_play(), [p for p in hand.players()], len(hand.events))
            try:
                hand.apply(Action.model_validate(step["reject"]))
            except IllegalAction:
                pass
            else:
                raise AssertionError(f"{where}: {step['reject']} was accepted")
            after = (hand.actor_seat, hand.chips_in_play(), [p for p in hand.players()], len(hand.events))
            assert before == after, f"{where}: rejected action mutated state"
        if "do" in step:
            hand.apply(Action.model_validate(step["do"]))
        if step.get("check_down"):
            check_down(hand)
    expect = fx.get("expect", {})
    if expect:
        assert hand.is_complete, f"{fx['id']}: hand not complete"
        check_result(hand, expect, fx["id"])
    if hand.is_complete:
        assert_matches_oracles(hand)
    return hand


def check_result(hand: PokerKitHand, expect: dict[str, Any], fid: str) -> None:
    result = hand.result()
    if "final_stacks" in expect:
        assert result.final_stacks == _seat_map(expect["final_stacks"]), f"{fid}: final {result.final_stacks}"
    if "contributions" in expect:
        assert result.contributions == _seat_map(expect["contributions"]), f"{fid}: contributions {result.contributions}"
    if "pots" in expect:
        got = [
            {"amount": p.amount, "eligible": sorted(p.eligible), "awards": dict(sorted(p.awards.items()))}
            for p in result.pots
        ]
        want = [
            {
                "amount": p["amount"],
                "eligible": sorted(p["eligible"]),
                **({"awards": dict(sorted(_seat_map(p["awards"]).items()))} if "awards" in p else {}),
            }
            for p in expect["pots"]
        ]
        for g, w in zip(got, want):
            if "awards" not in w:
                g.pop("awards")
        assert got == want, f"{fid}: pots {got}"
    if "revealed" in expect:
        assert sorted(result.revealed) == sorted(expect["revealed"]), f"{fid}: revealed {sorted(result.revealed)}"
    if "board_cards" in expect:
        assert len(result.board) == expect["board_cards"], f"{fid}: board {result.board}"
    if "ended_by_fold" in expect:
        assert result.ended_by_fold == expect["ended_by_fold"]
    if "first_hole_seat" in expect:
        first = next(e for e in hand.events if e.type == "HoleDealt")
        assert first.data["seat"] == expect["first_hole_seat"], f"{fid}: first hole card to {first.data['seat']}"
    if "event_counts" in expect:
        for t, count in expect["event_counts"].items():
            got = sum(1 for e in hand.events if e.type == t)
            assert got == count, f"{fid}: {t} count {got} != {count}"
    if "auto_actions" in expect:
        got = sum(1 for e in hand.events if e.type == "ActionApplied" and "auto" in e.data)
        assert got == expect["auto_actions"], f"{fid}: auto actions {got}"
    if "excess_returned" in expect:
        got = {e.data["seat"]: e.data["amount"] for e in hand.events if e.type == "ExcessReturned"}
        assert got == _seat_map(expect["excess_returned"]), f"{fid}: excess {got}"
    if "events_include" in expect:
        types = [e.type for e in hand.events]
        for t in expect["events_include"]:
            assert t in types, f"{fid}: missing event {t}"
    if "events_exclude" in expect:
        types = [e.type for e in hand.events]
        for t in expect["events_exclude"]:
            assert t not in types, f"{fid}: unexpected event {t}"
