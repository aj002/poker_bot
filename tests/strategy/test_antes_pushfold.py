"""Ante push/fold subgames against the arena engine, and the ante keys."""

import random

import pytest

from arena.domain.models import Action, TournamentConfig
from arena.domain.rng import SeededRng
from arena.table.service import Table
from strategy.antes.pushfold import ANTE_KEY_SHIFT, AnteSnapshot, ante_keys_for, ante_seated_snapshot
from strategy.antes.subgame import AnteSubgameTree, ante_subgame_for, settle_dead
from strategy.endgame.keys import Snapshot, keys_for
from strategy.endgame.points import settle
from strategy.endgame.subgame import FOLD, SHOVE

PAYOUTS = (50.0, 30.0, 20.0)


def deal(posting: str, n: int, stacks: list[int], ante: int, seed: str) -> Table:
    cfg = TournamentConfig(ruleset_id="practice_nlhe", initial_player_count=n, starting_stack=1000,
                           blind_levels=[[10, 20], [15, 30]], hands_per_level=5, hand_cap=10,
                           antes=[ante, ante], ante_posting=posting)
    seats = list(cfg.resolved_seats)
    return Table(cfg, env_rng=SeededRng(seed, "dealer"), initial_button=seats[0],
                 start_stacks=dict(zip(seats, stacks)))


def snapshot_of(table: Table) -> tuple:
    rec = table.hands[0]
    pos = rec.positions
    live = [s for s in table.seat_ring if rec.start_stacks[s] > 0]
    return ante_seated_snapshot(ring=table.seat_ring, start_stacks=rec.start_stacks, button=pos.button,
                                small_blind_seat=pos.small_blind_seat(live), big_blind=pos.big_blind,
                                heads_up=pos.heads_up, blinds=rec.blinds, payouts=table.config.resolved_payouts, hand_number=1,
                                hand_cap=10, ante=rec.ante, posting=rec.ante_posting)


def act(table: Table, label: str) -> None:
    seat = table.actor_seat
    legal = table.legal_actions()
    if label == FOLD:
        a = Action(type="fold")
    elif label == SHOVE and legal.all_in_raise_to is not None:
        a = Action(type="raise_to", amount=legal.all_in_raise_to)
    elif legal.call_amount is not None:
        a = Action(type="call")
    else:
        a = Action(type="check")
    table.submit(seat, a)


def leaves(tree, idx: int) -> list[int]:
    nd = tree.nodes[idx]
    return [idx] if nd.player == -1 else [x for c in nd.children.values() for x in leaves(tree, c)]


@pytest.mark.parametrize("posting", ["classic", "big_blind"])
def test_ante_subgame_chips_match_the_engine(posting):
    rng = random.Random(f"ante-subgame-{posting}")
    compared = 0
    for t in range(400):
        n = rng.choice([2, 3, 4, 5, 6])
        ante = rng.choice([3, 5, 10, 20]) if posting == "classic" else rng.choice([10, 20, 30])
        stacks = [rng.choice([rng.randint(1, 40), rng.randint(20, 400), rng.randint(200, 1500)]) for _ in range(n)]
        table = deal(posting, n, stacks, ante, f"{posting}-{t}")
        seated = snapshot_of(table)
        snap = seated.snapshot
        posted = {e.data["seat"]: e.data["amount"] for e in table.hands[0].events if e.type == "AntePosted"}
        assert tuple(posted.get(s, 0) for s in seated.seats) == snap.antes
        try:
            tree = AnteSubgameTree(ante_subgame_for(snap, odd_order=seated.odd_order))
        except ValueError:
            continue
        idx = 0
        rec = table.hands[0]
        while tree.nodes[idx].player != -1 and rec.result is None:
            nd = tree.nodes[idx]
            if table.actor_seat != seated.seats[nd.player]:
                assert seated.seats.index(table.actor_seat) in nd.folded
                act(table, FOLD)
                continue
            label = rng.choice(sorted(nd.children))
            if label == FOLD and table.legal_actions().check:
                break
            act(table, label)
            idx = nd.children[label]
        if tree.nodes[idx].player != -1 and rec.result is None:
            continue
        postflop = False
        while rec.result is None:
            if table.legal_actions().call_amount is not None:
                assert seated.seats.index(table.actor_seat) in tree.nodes[idx].folded
                act(table, FOLD)
            else:
                postflop = True
                act(table, "check")
        result = rec.result
        if postflop:
            continue
        # The tree folds everyone after max_in_pot players are in; a player whose blind
        # already covers every short all-in stays in for free at the engine.
        capped = {seated.seats[p] for p in tree.nodes[idx].folded} - set(result.folded)
        if capped:
            continue
        final = [result.final_stacks[s] for s in seated.seats]
        ends = [end for leaf in leaves(tree, idx) for _, _, end in tree.chip_outcomes(leaf)[3]]
        assert final in ends, (t, stacks, ante, tree.path(idx), [(e.type, e.data) for e in rec.events if e.type != "HoleDealt"])
        compared += 1
    assert compared > 120


def test_settle_dead_puts_dead_chips_in_the_main_pot():
    levels, folded, strength = [20, 100, 300], [False, False, False], [0, -1, -2]
    assert settle_dead(levels, folded, strength, [0, 1, 2], 0) == settle(levels, folded, strength, [0, 1, 2])
    base = settle(levels, folded, [-2, -1, 0], [0, 1, 2])
    awards = settle_dead(levels, folded, [-2, -1, 0], [0, 1, 2], 30)
    assert [a - b for a, b in zip(awards, base)] == [0, 0, 30]
    awards = settle_dead([20, 100, 0], [False, False, True], [0, -1, 0], [0, 1, 2], 30)
    assert awards == [40 + 30, 80, 0]
    # Walk to a big blind with no small blind: its bet is returned, the dead ante is its win.
    assert settle_dead([0, 0, 0], [True, True, False], [0, 0, 0], [2, 0, 1], 300) == [0, 0, 300]


@pytest.mark.parametrize("posting", ["classic", "big_blind"])
def test_ante_spot_from_an_observation_matches_the_dealt_hand(posting):
    from arena.table.projections import observation
    from strategy.antes.lookup import ante_preflop_spot

    stacks = [300, 150, 90, 400, 60]
    table = deal(posting, 5, stacks, 5 if posting == "classic" else 20, f"spot-{posting}")
    seated = snapshot_of(table)
    spot = ante_preflop_spot(observation(table, table.actor_seat))
    assert spot is not None and spot.kind == "open" and spot.seated == seated
    act(table, SHOVE)
    spot = ante_preflop_spot(observation(table, table.actor_seat))
    assert spot.kind == "call" and spot.shover == 0 and spot.hero == 1
    assert spot.keys() == ante_keys_for("call", seated.snapshot, 1, 0)


def test_compact_table_reads_like_the_dict_table():
    from strategy.antes.lookup import CompactTable
    from strategy.antes.tables import compact
    from strategy.endgame.keys import KINDS
    from strategy.endgame.lookup import PushFoldTable
    from strategy.tables import store

    prov, arrays = store.load_arrays("endgame-points-practice_nlhe")
    full, small = PushFoldTable(prov, arrays), CompactTable(prov, compact(arrays))
    rng = random.Random(7)
    for kind in KINDS:
        keys = arrays[f"{kind}_keys"].tolist()
        for _ in range(300):
            probe = tuple(rng.choice(keys) + rng.choice([0, 0, 1]) for _ in range(4))
            cls = rng.randrange(small.h)
            got, want = small.lookup(kind, probe, cls), full.lookup(kind, probe, cls)
            assert (got is None) == (want is None)
            if got is not None:
                assert got[1] == want[1] and abs(got[0] - want[0]) <= 1 / 30 + 1e-9


def test_ante_keys_never_collide_with_ante_free_keys():
    plain = Snapshot((400, 400, 400), (0, 10, 20), 20, PAYOUTS, 30, 200)
    for antes, posting in (((5, 5, 5), "classic"), ((0, 0, 20), "big_blind"), ((0, 0, 0), "classic")):
        snap = AnteSnapshot(plain.stacks, plain.posts, plain.bb, plain.payouts, 30, 200, antes, posting)
        keys = ante_keys_for("open", snap, 0)
        assert all(k >> ANTE_KEY_SHIFT >= 1 for k in keys)
        assert [k & ((1 << ANTE_KEY_SHIFT) - 1) for k in keys] == list(keys_for("open", plain, 0))
