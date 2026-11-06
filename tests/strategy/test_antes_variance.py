"""The luck-adjusted estimator (strategy.eval.luck) and seat-keyed dealing with antes in the pot."""

import copy
import itertools
import math

import numpy as np
import pytest

from arena.domain.models import TournamentConfig
from arena.table.driver import POLICIES, play_out, policy_rngs
from arena.table.service import seeded_table
from strategy.eval.luck import TERMS, LuckSettings, _hands, _scores, _Table, points_equity, table_luck
from strategy.objective.points import outcome_from_events

FAST = LuckSettings(runout_samples=100, equity_samples=100)
MIX = ("check_or_call", "legal_random", "shove", "check_or_call")
ANTES = {"classic": [3, 3, 5, 5, 10, 10, 15, 20], "big_blind": [20, 20, 30, 50, 75, 100, 150, 200]}
POSTINGS = tuple(ANTES)
CALM = ("check_or_call", "legal_random", "check_or_fold", "legal_random")


def play(seed, posting, players=4, stack=800, cap=20, policies=MIX, **config):
    # Antes from the first hand, and levels short enough for stacks to get all-in for the ante.
    cfg = TournamentConfig(initial_player_count=players, starting_stack=stack, hand_cap=cap, hands_per_level=5,
                           antes=ANTES[posting], ante_posting=posting, **config)
    t = seeded_table(cfg, seed)
    seats = cfg.resolved_seats
    play_out(t, {s: POLICIES[policies[i % len(policies)]] for i, s in enumerate(seats)}, policy_rngs(seed, seats))
    return [e.model_dump(mode="json") for e in t.log]


def _ante_hands(events):
    antes = {}
    hand = None
    for e in events:
        if e["type"] == "HandStarted":
            hand = int(e["data"]["hand_number"])
        elif e["type"] == "AntePosted":
            antes[hand] = antes.get(hand, 0) + int(e["data"]["amount"])
    return antes


@pytest.mark.parametrize("posting", POSTINGS)
def test_points_equity_telescopes_to_the_official_points_with_antes(posting):
    events = play(f"vr-ante-equity-{posting}", posting, cap=30, policies=CALM)
    assert len(_ante_hands(events)) > 10
    outcome = outcome_from_events(events)
    hands = list(_hands(events))
    config = next(e["data"]["config"] for e in events if e["type"] == "TableStarted")
    for seat in range(4):
        last = [h for h in hands if seat in h.start][-1]
        order = sorted(last.start)
        v = points_equity(seat, last.start, [[last.final[s] for s in order]], config["payout_points"])[0]
        assert v == pytest.approx(outcome.points(seat), abs=1e-9)


def _runout_hands(seeds, posting, cards_to_come, players=4):
    for seed in seeds:
        events = play(seed, posting, players=players)
        antes = _ante_hands(events)
        config = next(e["data"]["config"] for e in events if e["type"] == "TableStarted")
        for h in _hands(events):
            runout = [d for d in h.deals if d["index"] > h.last_decision]
            if (h.number in antes and runout and 5 - len(runout[0]["board"]) == cards_to_come
                    and len(runout[0]["live"]) >= 2):
                yield config, h, runout[0]


@pytest.mark.parametrize("posting", POSTINGS)
@pytest.mark.parametrize("cards_to_come", [1, 2])
def test_allin_term_is_an_exact_martingale_difference_with_antes(posting, cards_to_come):
    config, h, first = next(_runout_hands((f"vr-ante-runout-{posting}-{i}" for i in range(300)), posting,
                                          cards_to_come))
    board = first["board"]
    known = set(board) | set(first["burned"]) | {c for cs in h.holes.values() for c in cs}
    unseen = [c for c in range(52) if c not in known]
    seat = sorted(first["live"])[0]
    xs = []
    for extra in itertools.combinations(unseen, cards_to_come):
        alt = copy.copy(h)
        alt.board = list(board) + list(extra)
        t = _Table(seat, config["payout_points"], FAST)
        t.allin(alt, first)
        xs.append(t.result()["terms"]["allin"])
    assert abs(math.fsum(xs) / len(xs)) < 1e-9
    assert max(abs(x) for x in xs) > 0.1


@pytest.mark.parametrize("posting", POSTINGS)
def test_allin_realized_stacks_match_the_table_with_antes(posting):
    """The re-split of the realized runout reproduces the table's final stacks: the antes (a
    big-blind ante as dead money in the main pot) are in the awarded pots the terms re-split."""
    checked = side = 0
    for config, h, first in _runout_hands((f"vr-ante-real-{posting}-{i}" for i in range(80)), posting, 5,
                                          players=6):
        if any(len(awards) > 1 for _, _, awards in h.pots):
            continue
        live = sorted(first["live"])
        t = _Table(live[0], config["payout_points"], FAST)
        row = t.finals(h, live, _scores([h.holes[s] for s in live], first["board"],
                                        np.asarray([h.board[len(first["board"]):]], dtype=np.int64)))[0]
        assert row.tolist() == [float(h.final[s]) for s in sorted(h.start)]
        checked += 1
        side += len(h.pots) > 1
    assert checked >= 10 and side >= 1


@pytest.mark.parametrize("posting", POSTINGS)
def test_luck_terms_are_zero_mean_with_antes(posting):
    rows = [table_luck(play(f"vr-ante-zero-{posting}-{i}", posting, players=3, stack=300, cap=12), 0, FAST)
            for i in range(80)]
    for term in TERMS:
        xs = [r["terms"][term] for r in rows]
        sd = float(np.std(xs, ddof=1))
        if sd == 0:
            continue
        assert abs(float(np.mean(xs))) < 4 * sd / math.sqrt(len(xs)), term
    assert sum(r["counts"]["allin_hands"] for r in rows) > 20


@pytest.mark.parametrize("posting", POSTINGS)
def test_seat_keyed_deal_keeps_each_seats_cards_with_antes(posting):
    styles = (("check_or_call",), ("check_or_fold", "legal_random"), ("check_or_call", "check_or_fold"))
    tables = [play(f"vr-ante-keyed-{posting}", posting, policies=p, cap=40, deal="seat_keyed") for p in styles]
    cards = [{(h.number, s): tuple(c) for h in _hands(e) for s, c in h.holes.items()} for e in tables]
    shared = set(cards[0]) & set(cards[1]) & set(cards[2])
    assert len(shared) > 50 and all(cards[0][k] == cards[1][k] == cards[2][k] for k in shared)
