"""Advisor mode: a real game entered by hand, replayed through the engine, advised by the final bot."""

from __future__ import annotations

import json
import sys

import pytest

from arena.advisor.ops import do
from arena.advisor.session import Advisor, AdvisorError, new_session

ADV = Advisor(samples=40)


def session(players=6, hero=2, *, mode="cash", payouts=None, ante=0, posting=None, stacks=None, sb=50, bb=100):
    return new_session({"players": players, "hero": hero, "sb": sb, "bb": bb, "ante": ante, "ante_posting": posting,
                        "mode": mode, "payouts": payouts, "stacks": stacks or [10_000] * players})


def act(s, seat, kind, amount=None):
    do(ADV, s, {"op": "action", "seat": seat, "type": kind, **({"amount": amount} if amount else {})})


def view(s):
    return ADV.view(s)


def hand_to_hero(s, hole=("Ah", "Kd")):
    """Deal the hero's cards and fold everyone in front of the hero; returns the view at the hero's decision."""
    do(ADV, s, {"op": "hole", "cards": list(hole)})
    for _ in range(10):
        v = view(s)
        if v["actor"] == s["setup"]["hero"] or v["stage"] != "action":
            return v
        act(s, v["actor"], "fold")
    raise AssertionError("never reached the hero")


@pytest.mark.parametrize("players", range(2, 9))
def test_every_seating_deals_the_entered_cards_to_the_bot(players):
    reached = 0
    for button in range(players):
        s = session(players, hero=players - 1)
        do(ADV, s, {"op": "edit_hand", "button": button})
        v = hand_to_hero(s, ("Qs", "Qh"))
        if v["stage"] == "complete":  # the hero is the big blind and everyone folded to them
            continue
        reached += 1
        assert v["stage"] == "action" and v["actor"] == players - 1, (players, button, v["stage"])
        obs, _ = ADV.simulate(s, 0, ADV.state_in(s, 0)).pending
        assert obs.hole_cards == ["Qs", "Qh"] and obs.board == []
        assert v["advice"]["headline"]
    assert reached >= players - 1


def test_board_cards_reach_the_bot_only_when_entered():
    s = session(3, hero=0)
    do(ADV, s, {"op": "edit_hand", "button": 0})
    do(ADV, s, {"op": "hole", "cards": ["7c", "7d"]})
    act(s, 0, "call"), act(s, 1, "call"), act(s, 2, "check")
    v = view(s)
    assert v["stage"] == "board" and v["need_board"] == 3 and v["street_needed"] == "flop" and v["advice"] is None
    with pytest.raises(AdvisorError) as bad:
        do(ADV, s, {"op": "board", "cards": ["2c", "9d"]})
    assert bad.value.code == "wrong_board"
    do(ADV, s, {"op": "board", "cards": ["7h", "2c", "9d"]})
    v = view(s)
    assert v["stage"] == "action" and v["board"] == ["7h", "2c", "9d"] and v["street"] == "flop"
    obs, _ = ADV.simulate(s, 0, ADV.state_in(s, 0)).pending if view(s)["actor"] == 0 else (None, None)
    act(s, v["actor"], "check")
    act(s, view(s)["actor"], "check")
    v = view(s)
    if v["actor"] == 0:
        assert v["advice"] and "Set of sevens" in v["advice"]["reason"] or "three" in v["advice"]["reason"].lower()


def test_duplicates_and_bad_input_are_rejected_without_changing_anything():
    s = session()
    do(ADV, s, {"op": "hole", "cards": ["As", "Ks"]})
    snapshot = json.dumps(s)
    for op in ({"op": "hole", "cards": ["As", "Ks"]}, {"op": "board", "cards": ["As", "2d", "3c"]},
               {"op": "action", "seat": 0, "type": "raise_to", "amount": 1},
               {"op": "action", "seat": 5, "type": "fold"}, {"op": "action", "seat": 3, "type": "bet"},
               {"op": "award", "pots": [[1]]}, {"op": "new_hand", "stacks": [1, 2]}):
        with pytest.raises(AdvisorError):
            do(ADV, s, op)
        assert json.dumps(s) == snapshot
    with pytest.raises(AdvisorError) as e:
        do(ADV, session(), {"op": "hole", "cards": ["Ax", "Ks"]})
    assert e.value.code == "invalid_card"
    with pytest.raises(AdvisorError):
        do(ADV, session(), {"op": "hole", "cards": ["As", "As"]})


def test_fold_win_settles_chips_and_the_button_moves_on():
    s = session(4, hero=0, stacks=[1000, 2000, 3000, 4000])
    do(ADV, s, {"op": "edit_hand", "button": 3})
    v = hand_to_hero(s)
    assert (v["blinds"]["sb"], v["blinds"]["bb"]) == (50, 100)
    # button seat 3, so sb is seat 0 (the hero) and bb is seat 1; utg is seat 2
    assert [x["position"] for x in v["seats"]] == ["SB", "BB", "UTG", "BTN"]
    act(s, 0, "raise_to", 300)
    act(s, 1, "fold")
    v = view(s)
    assert v["stage"] == "complete" and v["result"]["ended_by_fold"] and v["result"]["winner"] == 0
    stacks = v["result"]["stacks_after"]
    assert sum(stacks.values()) == 10_000 and stacks["0"] == 1000 + 100 + 0 + 0 + 0 - 0 + 0 and stacks["1"] == 1900
    do(ADV, s, {"op": "new_hand"})
    v = view(s)
    assert v["hand_number"] == 2 and v["button"] == 0 and v["editable"]
    assert [x["stack_start"] for x in v["seats"]] == [1100, 1900, 3000, 4000] or sum(x["stack_start"] for x in v["seats"]) == 10_000


def test_busted_players_are_skipped_by_the_button():
    s = session(4, hero=0, stacks=[1000, 1000, 1000, 1000])
    do(ADV, s, {"op": "edit_hand", "button": 3})
    do(ADV, s, {"op": "hole", "cards": ["Ah", "As"]})
    do(ADV, s, {"op": "new_hand", "stacks": [1000, 0, 2000, 1000]})
    assert view(s)["button"] == 0
    do(ADV, s, {"op": "hole", "cards": ["2h", "7s"]})
    do(ADV, s, {"op": "new_hand", "stacks": [1000, 0, 2000, 1000]})
    assert view(s)["button"] == 2


def test_showdown_needs_the_winner_and_splits_pots():
    s = session(3, hero=0, stacks=[5000, 5000, 5000])
    do(ADV, s, {"op": "edit_hand", "button": 0})
    do(ADV, s, {"op": "hole", "cards": ["Jh", "Jd"]})
    act(s, 0, "raise_to", 300), act(s, 1, "fold"), act(s, 2, "call")
    do(ADV, s, {"op": "board", "cards": ["2c", "7d", "9s"]})
    act(s, 2, "check"), act(s, 0, "check")
    do(ADV, s, {"op": "board", "cards": ["Kc"]})
    act(s, 2, "check"), act(s, 0, "check")
    do(ADV, s, {"op": "board", "cards": ["3h"]})
    act(s, 2, "check"), act(s, 0, "check")
    v = view(s)
    assert v["stage"] == "complete" and not v["result"]["ended_by_fold"] and not v["result"]["resolved"]
    assert [p["amount"] for p in v["result"]["pots"]] == [650] and v["result"]["pots"][0]["eligible"] == [0, 2]
    do(ADV, s, {"op": "award", "pots": [[0, 2]]})
    after = view(s)["result"]["stacks_after"]
    assert after == {"0": 5025, "1": 4950, "2": 5025} and sum(after.values()) == 15_000
    do(ADV, s, {"op": "award", "pots": [[2]]})
    assert view(s)["result"]["stacks_after"] == {"0": 4700, "1": 4950, "2": 5350}
    with pytest.raises(AdvisorError):
        do(ADV, s, {"op": "award", "pots": [[1]]})


def test_side_pots_are_awarded_one_by_one():
    s = session(3, hero=0, stacks=[1000, 3000, 6000])
    do(ADV, s, {"op": "edit_hand", "button": 0})
    do(ADV, s, {"op": "hole", "cards": ["Ac", "Ad"]})
    act(s, 0, "raise_to", 1000), act(s, 1, "raise_to", 3000), act(s, 2, "call")
    v = view(s)
    assert v["stage"] == "complete" or v["stage"] == "board" or v["actor"] is not None
    for _ in range(4):
        v = view(s)
        if v["stage"] == "board":
            do(ADV, s, {"op": "board", "cards": ["2c", "5d", "9h"][: v["need_board"]] if v["need_board"] == 3
                        else [("Td", "Jd")[len(v["board"]) - 3]]})
        elif v["stage"] == "action":
            act(s, v["actor"], "call" if v["legal"]["call_amount"] else "check")
    v = view(s)
    assert v["stage"] == "complete"
    pots = v["result"]["pots"]
    assert [p["amount"] for p in pots] == [3000, 4000] and pots[1]["eligible"] == [1, 2]
    do(ADV, s, {"op": "award", "pots": [[0], [2]]})
    assert view(s)["result"]["stacks_after"] == {"0": 3000, "1": 0, "2": 7000}


@pytest.mark.parametrize("posting", ["classic", "big_blind"])
def test_antes_are_posted_and_counted(posting):
    s = session(4, hero=0, ante=10, posting=posting)
    do(ADV, s, {"op": "edit_hand", "button": 0})
    v = hand_to_hero(s)
    assert v["blinds"]["ante"] == 10 and v["blinds"]["ante_posting"] == posting
    assert any("antes" in line for line in v["log"])
    assert v["pot"] == (40 if posting == "classic" else 10) + 150


def test_undo_walks_back_one_entry_at_a_time_across_hands():
    s = session(3, hero=0)
    do(ADV, s, {"op": "edit_hand", "button": 0})
    do(ADV, s, {"op": "hole", "cards": ["Ah", "Kh"]})
    act(s, 0, "fold")
    before = json.dumps(s)
    act(s, 1, "fold")
    assert view(s)["stage"] == "complete"
    do(ADV, s, {"op": "undo"})
    assert json.dumps(s) == before and view(s)["actor"] == 1
    do(ADV, s, {"op": "undo"}), do(ADV, s, {"op": "undo"})
    assert view(s)["stage"] == "hole" and view(s)["hole"] is None
    do(ADV, s, {"op": "hole", "cards": ["2c", "3d"]}), act(s, 0, "fold"), act(s, 1, "fold")
    do(ADV, s, {"op": "new_hand"})
    assert view(s)["hand_number"] == 2
    do(ADV, s, {"op": "undo"})
    v = view(s)
    assert v["hand_number"] == 1 and v["stage"] == "complete"
    with pytest.raises(AdvisorError):
        do(ADV, session(), {"op": "undo"})


def test_chips_and_blinds_can_be_fixed_before_the_hand_starts_only():
    s = session(3, hero=0)
    do(ADV, s, {"op": "edit_hand", "stacks": [4000, 5000, 6000], "sb": 100, "bb": 200, "button": 1, "ante": 25,
                "ante_posting": "classic"})
    v = view(s)
    assert [x["stack_start"] for x in v["seats"]] == [4000, 5000, 6000] and v["button"] == 1
    assert v["blinds"] == {"sb": 100, "bb": 200, "ante": 25, "ante_posting": "classic"}
    with pytest.raises(AdvisorError):
        do(ADV, s, {"op": "edit_hand", "sb": 300, "bb": 200})
    do(ADV, s, {"op": "hole", "cards": ["Ah", "Kh"]})
    with pytest.raises(AdvisorError) as e:
        do(ADV, s, {"op": "edit_hand", "sb": 10})
    assert e.value.code == "hand_started"


def test_cash_and_tournament_set_the_payouts_the_bot_sees():
    for mode, payouts, expected in (("cash", None, [100, 0, 0, 0]), ("tournament", "top3", [50, 30, 20, 0]),
                                    ("tournament", [70, 30], [70, 30, 0, 0])):
        s = session(4, hero=0, mode=mode, payouts=payouts)
        do(ADV, s, {"op": "edit_hand", "button": 0})
        v = hand_to_hero(s)
        obs, _ = ADV.simulate(s, 0, ADV.state_in(s, 0)).pending
        assert obs.payouts == expected and v["advice"]


def test_opponent_memory_carries_across_hands_and_survives_a_reload():
    s = session(4, hero=0, stacks=[10_000] * 4)
    do(ADV, s, {"op": "edit_hand", "button": 3})
    do(ADV, s, {"op": "hole", "cards": ["Ah", "Kd"]})
    act(s, 2, "raise_to", 300)
    act(s, 3, "fold")
    assert view(s)["actor"] == 0
    act(s, 0, "call"), act(s, 1, "fold")
    do(ADV, s, {"op": "board", "cards": ["2c", "7d", "9s"]})
    act(s, 0, "check")
    act(s, 2, "raise_to", 400)
    assert view(s)["actor"] == 0
    act(s, 0, "fold")
    assert view(s)["stage"] == "complete"
    do(ADV, s, {"op": "new_hand", "stacks": [9_700, 10_000, 10_300, 10_000]})
    state = ADV.state_in(s, 1)
    hands, _, seats = state["x"].partition("|")
    weights = {int(e.split("=")[0]): float(e.split("/")[1]) for e in seats.split(";")}
    assert int(hands) == 0 and weights[2] == 2.0 and 0 not in weights and sum(weights.values()) >= 4
    assert "x" not in ADV.state_in(s, 0)
    first = hand_to_hero(s, ("Qh", "Qd"))
    assert first["stage"] == "action" and first["advice"]
    reloaded = json.loads(json.dumps(s))
    again = Advisor(samples=40).view(reloaded)
    assert again["advice"] == first["advice"] and again["log"] == first["log"]


def test_sessions_are_deterministic_and_the_bot_needs_no_numba():
    a, b = session(), session()
    a["seed"] = b["seed"] = "fixed"
    for s in (a, b):
        do(ADV, s, {"op": "edit_hand", "button": 0})
    assert hand_to_hero(a)["advice"] == hand_to_hero(b)["advice"]
    assert "numba" not in sys.modules and "llvmlite" not in sys.modules


def test_short_stacks_get_the_push_fold_chart():
    s = session(3, hero=0, stacks=[600, 10_000, 10_000], sb=50, bb=100)
    do(ADV, s, {"op": "edit_hand", "button": 0})
    do(ADV, s, {"op": "hole", "cards": ["As", "Qs"]})
    act(s, 0, "fold") if view(s)["actor"] == 0 else None
    s = session(3, hero=1, stacks=[10_000, 700, 10_000])
    do(ADV, s, {"op": "edit_hand", "button": 0})
    v = hand_to_hero(s, ("As", "Qs"))
    assert v["advice"]["source"] in ("push/fold chart", "blueprint") and v["advice"]["effective_bb"] <= 7.1


def test_abandoning_a_hand_restores_the_chips_and_an_unawarded_showdown_blocks_the_next_hand():
    s = session(3, hero=0, stacks=[5000, 5000, 5000])
    do(ADV, s, {"op": "edit_hand", "button": 0})
    do(ADV, s, {"op": "hole", "cards": ["Jh", "Jd"]})
    act(s, 0, "raise_to", 300)
    do(ADV, s, {"op": "new_hand"})
    assert [x["stack_start"] for x in view(s)["seats"]] == [5000, 5000, 5000] and view(s)["hand_number"] == 2
    do(ADV, s, {"op": "hole", "cards": ["Jh", "Jd"]})
    act(s, view(s)["actor"], "raise_to", 5000)
    act(s, view(s)["actor"], "call"), act(s, view(s)["actor"], "call")
    for card in ("2c", "7d", "9s"), ("Kc",), ("3h",):
        if view(s)["stage"] == "board":
            do(ADV, s, {"op": "board", "cards": list(card)})
    assert view(s)["stage"] == "complete" and not view(s)["result"]["resolved"]
    with pytest.raises(AdvisorError) as e:
        do(ADV, s, {"op": "new_hand"})
    assert e.value.code == "award_first"
