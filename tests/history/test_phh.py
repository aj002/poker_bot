"""PHH export round trip (spec 9.2): cards, actions and final stacks survive export and re-import.

Reading back uses ``tomllib`` and the arena engine, not PokerKit's notation
module. PokerKit's own PHH reader is a secondary cross-check, and the hands it
cannot judge (practice-rule corrections D1 and D5) are counted as explicit
exclusions rather than passes.
"""

from collections import Counter

import pytest
from pokerkit import HandHistory

from arena.history.canonical import export_history
from arena.history.phh import PhhError, PhhHandMeta, export_hand, export_history_phh, export_history_phhs, \
    reexport, replay_phh
from tests.support.fixtures import load_fixtures, run_fixture
from tests.support.tables import RULESETS, play_table

CORE = {fx["id"]: fx for fx in load_fixtures("core")}


def meta_for(hand, hand_id="h_fixture") -> PhhHandMeta:
    spec = hand.spec
    return PhhHandMeta(
        table_id="t_fixture",
        hand_id=hand_id,
        hand_number=1,
        level=0,
        ruleset_id=spec.profile.ruleset_id,
        seat_ring=tuple(spec.seat_ring),
        button=spec.button,
        small_blind_seat=spec.small_blind_seat,
        big_blind_seat=spec.big_blind_seat,
        small_blind=spec.small_blind,
        big_blind=spec.big_blind,
        start_stacks=dict(spec.stacks),
        local_rules=spec.local_rules.model_dump(),
    )


def check_roundtrip(doc: str, original_events) -> None:
    parsed, hand = replay_phh(doc)
    assert reexport(parsed, hand) == doc
    original_holes = {}
    for e in original_events:
        if e.type == "HoleDealt":
            original_holes.setdefault(e.data["seat"], []).append(e.data["card"])
    assert parsed.holes == original_holes
    board = [c for e in original_events if e.type == "BoardDealt" for c in e.data["cards"]]
    assert [c for b in parsed.boards for c in b] == board
    final = next(e.data["final_stacks"] for e in original_events if e.type == "HandCompleted")
    assert hand.final_stacks() == parsed.finishing_stacks == final
    manual = [(e.data["seat"], e.data["action"]["type"]) for e in original_events
              if e.type == "ActionApplied" and not e.data.get("auto")]
    replayed = [(e.data["seat"], e.data["action"]["type"]) for e in hand.events
                if e.type == "ActionApplied" and not e.data.get("auto")]
    assert replayed == manual


@pytest.mark.parametrize("n", range(2, 9))
@pytest.mark.parametrize("ruleset", RULESETS)
def test_table_hands_roundtrip(ruleset, n):
    history = export_history(play_table(ruleset, n, f"phh-{n}"))
    docs = export_history_phh(history)
    assert len(docs) == history.summary.hands_played
    for number, doc in enumerate(docs, start=1):
        check_roundtrip(doc, history.hand_events(number))


@pytest.mark.parametrize("fid", ["G15", "G20", "G20b", "G20c", "G20d", "G20-plo", "G21a", "G21c", "G22", "G22b"])
def test_golden_fixture_hands_roundtrip(fid):
    hand = run_fixture(CORE[fid])
    doc = export_hand(meta_for(hand), hand.events)
    check_roundtrip(doc, hand.events)
    if fid.startswith("G20"):
        assert "_arena_short_big_blind_correction = true" in doc
    if fid == "G22b":
        assert "_arena_odd_chip_correction = true" in doc
    if fid == "G20":
        assert "cbr 20 # arena:nominal_bb_call" in doc
    if fid == "G20c":
        assert "cc # arena:auto" in doc


def test_pokerkit_reader_agrees_except_declared_exclusions():
    tally = Counter()
    for ruleset in RULESETS:
        for n in (2, 3, 4, 6, 8):
            history = export_history(play_table(ruleset, n, f"pkx-{n}"))
            for doc in export_history_phh(history):
                excluded = ("_arena_short_big_blind_correction = true" in doc
                            or "_arena_odd_chip_correction = true" in doc)
                try:
                    states = list(HandHistory.loads(doc))
                    agrees = list(states[-1].stacks) == HandHistory.loads(doc).finishing_stacks
                except ValueError:
                    agrees = False
                if excluded:
                    tally["excluded_practice_rule"] += 1
                else:
                    assert agrees, doc
                    tally["agree"] += 1
    assert tally["agree"] >= 200
    print(dict(tally))


ANTES = {"classic": [3, 4, 5, 10, 15, 25, 40, 50], "big_blind": [20, 30, 50, 100, 150, 200, 300, 400]}


@pytest.mark.parametrize("posting", ["classic", "big_blind"])
@pytest.mark.parametrize("n", (2, 3, 6, 8))
def test_ante_hands_roundtrip_and_pokerkit_reader(posting, n):
    tally = Counter()
    for seed in range(4):
        history = export_history(play_table("practice_nlhe", n, f"phh-ante-{posting}-{n}-{seed}", stack=1000,
                                            antes=ANTES[posting], ante_posting=posting))
        for number, doc in enumerate(export_history_phh(history), start=1):
            check_roundtrip(doc, history.hand_events(number))
            assert "_arena_ante = " in doc
            assert f"ante_trimming_status = {'true' if posting == 'classic' else 'false'}" in doc
            if "_arena_short_big_blind_correction = true" in doc or "_arena_odd_chip_correction = true" in doc:
                continue
            hh = HandHistory.loads(doc)
            assert list(list(hh)[-1].stacks) == hh.finishing_stacks, doc
            tally["agree"] += 1
    assert tally["agree"] >= 10


def test_phhs_bundle_and_rejections():
    history = export_history(play_table("practice_nlhe", 3, "bundle"))
    bundle = export_history_phhs(history)
    assert bundle.count("variant = ") == history.summary.hands_played
    doc = export_history_phh(history)[0]
    with pytest.raises(PhhError):
        replay_phh(doc.replace('_arena_ruleset_id = "practice_nlhe"', '_arena_ruleset_id = "uncertainty_plo4"'))
    tampered = doc.replace("finishing_stacks = [", "finishing_stacks = [1", 1)
    with pytest.raises(PhhError, match="finishing stacks"):
        replay_phh(tampered)
