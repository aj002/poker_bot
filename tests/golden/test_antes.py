"""Antes (docs/assumptions.md, "Antes"): handcrafted hands, config rules, wire shape and full tables."""

import json

import pytest
from pydantic import ValidationError

from arena.domain.models import TournamentConfig
from arena.history.canonical import TableHistory, export_history
from arena.history.phh import PhhHandMeta, export_hand, replay_phh
from arena.history.replay import replay
from arena.table.projections import Viewer, observation, table_view
from arena.domain.rng import SeededRng
from arena.table.driver import legal_random
from arena.table.service import seeded_table
from tests.support.fixtures import load_fixtures, run_fixture
from tests.support.tables import play_table

ANTE_FIXTURES = load_fixtures("antes")
LEVELS = [[10, 20], [15, 30], [25, 50]]


@pytest.mark.parametrize("fixture", ANTE_FIXTURES, ids=[f["id"] for f in ANTE_FIXTURES])
def test_ante_fixture(fixture):
    run_fixture(fixture)


@pytest.mark.parametrize("fixture", ANTE_FIXTURES, ids=[f["id"] for f in ANTE_FIXTURES])
def test_ante_fixture_phh_roundtrip(fixture):
    hand = run_fixture(fixture)
    spec = hand.spec
    meta = PhhHandMeta(
        table_id="t_fixture", hand_id="h_fixture", hand_number=1, level=0, ruleset_id=spec.profile.ruleset_id,
        seat_ring=tuple(spec.seat_ring), button=spec.button, small_blind_seat=spec.small_blind_seat,
        big_blind_seat=spec.big_blind_seat, small_blind=spec.small_blind, big_blind=spec.big_blind,
        start_stacks=dict(spec.stacks), local_rules=spec.local_rules.model_dump(),
        ante=spec.ante, ante_posting=spec.ante_posting,
    )
    doc = export_hand(meta, hand.events)
    parsed, replayed = replay_phh(doc)
    assert replayed.final_stacks() == hand.final_stacks() == parsed.finishing_stacks


def config(**kw) -> TournamentConfig:
    return TournamentConfig(ruleset_id="practice_nlhe", initial_player_count=6, starting_stack=1000,
                            blind_levels=LEVELS, hands_per_level=40, hand_cap=120, **kw)


@pytest.mark.parametrize("kw, message", [
    ({"antes": [0, 5, 10]}, "set together"),
    ({"ante_posting": "classic"}, "set together"),
    ({"antes": [5, 10], "ante_posting": "classic"}, "one ante per blind level"),
    ({"antes": [10, 5, 10], "ante_posting": "classic"}, "must not decrease"),
    ({"antes": [0, 0, 0], "ante_posting": "big_blind"}, "all zero"),
    ({"antes": [-1, 5, 10], "ante_posting": "classic"}, "0..MAX_STARTING_STACK"),
    ({"antes": [5, 10, 20], "ante_posting": "button"}, "ante_posting"),
])
def test_config_rejects_bad_antes(kw, message):
    with pytest.raises(ValidationError, match=message):
        config(**kw)


def test_ante_free_config_has_no_ante_keys():
    plain = config()
    assert "antes" not in plain.resolved() and "ante_posting" not in plain.resolved()
    with_antes = config(antes=[0, 5, 10], ante_posting="classic")
    assert with_antes.resolved()["antes"] == [0, 5, 10]
    assert with_antes.ante_for_hand(1) == 0 and with_antes.ante_for_hand(41) == 5 and with_antes.ante_for_hand(81) == 10


def play(posting: str, seed: str, antes=(5, 10, 20)):
    return play_table("practice_nlhe", 6, seed, stack=1000, blind_levels=LEVELS, hands_per_level=40, hand_cap=120,
                      antes=list(antes), ante_posting=posting)


@pytest.mark.parametrize("posting", ["classic", "big_blind"])
def test_table_with_antes_replays_and_conserves_chips(posting):
    table = play(posting, f"ante-table-{posting}")
    assert sum(table.stacks.values()) == 6 * 1000
    history = TableHistory.loads(export_history(table).dumps())
    report = replay(history)
    assert report.complete and report.final_stacks == table.stacks and report.notes == []
    posted = [e for e in table.log if e.type == "AntePosted"]
    assert posted
    per_hand = {}
    for e in posted:
        per_hand[e.hand_id] = per_hand.get(e.hand_id, 0) + 1
    if posting == "big_blind":
        assert set(per_hand.values()) == {1}
    else:
        assert max(per_hand.values()) > 1


def test_ante_keys_reach_the_wire_only_when_an_ante_is_live():
    cfg = TournamentConfig(ruleset_id="practice_nlhe", initial_player_count=6, starting_stack=1000,
                           blind_levels=LEVELS, hands_per_level=2, hand_cap=6, antes=[0, 5, 10],
                           ante_posting="classic")
    table = seeded_table(cfg, "ante-wire", table_id="t_ante_wire")
    rng = SeededRng("ante-wire", "policy")
    seen_zero = seen_ante = False
    while not table.complete:
        if table.hand is not None:
            view = json.loads(table_view(table, Viewer.spectator()).model_dump_json())
            started = next(e for e in view["action_history"] if e["type"] == "HandStarted")
            if table.hands[-1].ante:
                seen_ante = True
                assert view["blinds"]["ante"] == table.hands[-1].ante and view["blinds"]["ante_posting"] == "classic"
                assert started["data"]["ante"] == table.hands[-1].ante
                assert any(e["type"] == "AntePosted" for e in view["action_history"])
                assert sum(p["amount"] for p in view["pots"]) == view["settled_pot"]
            else:
                seen_zero = True
                assert "ante" not in view["blinds"] and "ante" not in started["data"]
                assert view["next_level"] is None or view["next_level"]["ante"] == 5
            obs = observation(table, table.actor_seat)
            assert obs.blinds.ante == table.hands[-1].ante
        seat = table.actor_seat
        table.submit(seat, legal_random(seat, table.legal_actions(), rng), hand_id=table.hands[-1].hand_id,
                     decision_id=table.decision_id, expected_state_version=table.state_version)
    assert seen_zero and seen_ante
