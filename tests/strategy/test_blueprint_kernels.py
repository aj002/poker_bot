"""Blueprint kernels (numba): payouts, utilities, trainer/runtime parity, exact HU games."""

from __future__ import annotations

import random

import numpy as np
import pytest

pytest.importorskip("numba")

from strategy.blueprint import abstraction as A  # noqa: E402
from strategy.blueprint import kernels as K  # noqa: E402

# --------------------------------------------------------------- settle, ICM


def _reference_settle(stacks_left, contrib, folded, ranks, post_order):
    """Side pots by contribution level; odd chips to winners in ``post_order`` order."""
    n = len(contrib)
    fin = [float(s) for s in stacks_left]
    live = [q for q in range(n) if not folded[q]]
    if len(live) == 1:
        fin[live[0]] += sum(contrib)
        return fin
    levels = sorted({contrib[q] for q in live})
    prev = 0
    for lev in levels:
        amount = sum(min(c, lev) - min(c, prev) for c in contrib)
        elig = [q for q in live if contrib[q] >= lev]
        best = max(ranks[q] for q in elig)
        win = [q for q in post_order if q in elig and ranks[q] == best]
        share, rem = divmod(amount, len(win))
        for q in win:
            fin[q] += share + (1 if rem > 0 else 0)
            rem -= 1
        prev = lev
    top = max(live, key=lambda q: contrib[q])
    fin[top] += sum(max(0, c - prev) for c in contrib)
    return fin


def test_settle_matches_a_reference_side_pot_split():
    rng = random.Random(9)
    I = np.zeros((2, K.CH, K.MAX_PLAYERS), np.int64)
    fin = np.zeros(K.MAX_PLAYERS)
    for _ in range(3000):
        n = rng.randrange(2, 9)
        contrib = [rng.choice([0, 20, 50, rng.randrange(1, 2000)]) for _ in range(n)]
        folded = [rng.random() < 0.3 for _ in range(n)]
        if all(folded):
            folded[0] = False
        left = [rng.randrange(0, 3000) for _ in range(n)]
        ranks = [rng.randrange(5) for _ in range(n)]
        order = list(range(n))
        rng.shuffle(order)
        I[:] = 0
        for q in range(n):
            I[0, 0, q], I[0, 2, q], I[0, 3, q] = left[q], contrib[q], 1 if folded[q] else 0
        K.settle(I, 0, n, np.array(ranks + [0] * (8 - n), np.int64), np.array(order + [-1] * (8 - n), np.int64), 0,
                 np.zeros((1, 1)), np.zeros(8, np.int64), fin)
        want = _reference_settle(left, contrib, folded, ranks, order)
        assert fin[:n].tolist() == pytest.approx(want)
        assert fin.sum() == pytest.approx(sum(left) + sum(contrib))


def test_icm_matches_the_endgame_icm():
    from strategy.endgame.points import icm

    rng = np.random.default_rng(4)
    f = np.zeros(1 << K.MAX_PLAYERS)
    for _ in range(300):
        m = int(rng.integers(2, 9))
        stacks = rng.integers(1, 5000, m).astype(np.float64)
        prizes = np.array([1.0, 0.6, 0.4, 0, 0, 0, 0, 0])
        alive = np.arange(8, dtype=np.int64)
        want = icm(stacks.astype(int).tolist(), prizes[:m].tolist())
        got = [K.icm_one(stacks, alive, m, prizes, w, f) for w in range(m)]
        assert got == pytest.approx(want, abs=1e-9)


def test_point_utilities_distribute_exactly_the_prizes():
    rng = np.random.default_rng(6)
    f = np.zeros(1 << K.MAX_PLAYERS)
    alive = np.zeros(K.MAX_PLAYERS, np.int64)
    prizes = np.array([1.0, 0.6, 0.4, 0, 0, 0, 0, 0])
    for _ in range(500):
        n = int(rng.integers(2, 9))
        start = np.zeros(8, np.int64)
        start[:n] = rng.integers(1, 4000, n)
        fin = np.zeros(8)
        fin[:n] = rng.integers(0, 4000, n) * (rng.random(n) < 0.7)
        if fin[:n].max() <= 0:
            fin[0] = 100.0
        total = sum(K.utility(q, n, start, fin, prizes, 1, 20, f, alive) for q in range(n))
        assert total == pytest.approx(prizes[:n].sum())


# ---------------------------------------------- trainer / runtime parity


class _NoPack:
    def __bool__(self):
        return True


def _replay_traced_hand(seed: int) -> int:
    from arena.domain.models import Action, TournamentConfig
    from arena.table.projections import observation
    from arena.domain.rng import SeededRng
    from arena.table.service import Table
    from strategy.blueprint.runtime import BlueprintPolicy
    from strategy.blueprint.states import state_row
    from strategy.endgame.table_state import seated_snapshot

    rng = random.Random(seed)
    n = rng.randrange(2, 9)
    config = TournamentConfig(initial_player_count=n, hand_cap=200)
    ring = list(config.resolved_seats)
    stacks = {s: rng.choice([rng.randrange(1, 15), rng.randrange(15, 120)]) * 20 + rng.randrange(0, 20) for s in ring}
    table = Table(config, env_rng=SeededRng(f"bp-parity-{seed}", "env"), start_stacks=stacks,
                  initial_button=rng.choice(ring))
    rec0 = table.hands[-1]
    pos = rec0.positions
    live = [s for s in ring if rec0.start_stacks.get(s, 0) > 0]
    args = dict(ring=ring, start_stacks=dict(rec0.start_stacks), button=pos.button,
                small_blind_seat=pos.small_blind if pos.small_blind in live else None, big_blind=pos.big_blind,
                heads_up=pos.heads_up, blinds=tuple(rec0.blinds), payouts=tuple(config.resolved_payouts),
                hand_number=rec0.hand_number, hand_cap=config.hand_cap)
    seats = seated_snapshot(**args).seats
    row = state_row(**args)
    assert row is not None
    I = np.zeros((K.MAX_DEPTH, K.CH, K.MAX_PLAYERS), np.int64)
    trace = np.zeros((K.MAX_DEPTH, len(K.TRACE_FIELDS)), np.int64)
    weights = np.array([rng.random() for _ in range(A.NUM_ACTIONS)]) + 0.05
    k, code = K.trace_hand(seed, row["n"], I, np.array(row["stacks"], np.int64), np.array(row["posts"], np.int64),
                           row["bb"], np.array(row["post_order"], np.int64), np.array(row["pressure"], np.int64), 3,
                           weights, trace)
    policy = BlueprintPolicy("practice_nlhe", _NoPack())
    for i in range(k):
        street, p, ctx, mask, a, target, t_min, t_a, t_b, t_all = (int(x) for x in trace[i])
        seat = table.actor_seat
        assert seat == seats[p], (i, seat, seats[p])
        obs = observation(table, seat)
        assert obs.street == ("preflop", "flop", "turn", "river")[street]
        key, rmask, targets = policy.features(obs)
        assert (key - A.OFFSETS[street]) % A.CONTEXTS == ctx, (i, A.context_fields(ctx),
                                                             A.context_fields((key - A.OFFSETS[street]) % A.CONTEXTS))
        assert rmask == mask, (i, bin(rmask), bin(mask))
        for slot, want in ((A.MIN_RAISE, t_min), (A.RAISE_A, t_a), (A.RAISE_B, t_b), (A.ALL_IN, t_all)):
            if (mask >> slot) & 1:
                assert targets[slot] == want, (i, slot, targets[slot], want)
        legal = obs.legal
        if a == A.FOLD:
            act = Action(type="fold")
        elif a == A.CALL:
            act = Action(type="call") if legal.call_amount else Action(type="check")
        else:
            act = Action(type="raise_to", amount=target)
        assert legal.allows(act), (i, act, legal)
        table.submit(seat, act)
    assert table.complete or table.hands[-1].hand_number > rec0.hand_number
    return k


def test_runtime_features_match_the_kernel_on_engine_replays():
    decisions = sum(_replay_traced_hand(seed) for seed in range(250))
    assert decisions > 1500


# ------------------------------------------------------- heads-up exact games


@pytest.fixture(scope="module")
def hu10():
    from strategy.blueprint.exploit import Game

    return Game("practice_nlhe", 10)


def test_cfr_plus_drives_unabstracted_exploitability_to_zero(hu10):
    from strategy.blueprint.exploit import vector_cfr

    sigma, curve = vector_cfr(hu10, "perfect", 300, report=(10, 300))
    assert curve[0]["mbb_per_hand"] > 20 * curve[-1]["mbb_per_hand"]
    assert curve[-1]["mbb_per_hand"] < 1.5
    v = hu10.value(sigma)
    assert curve[-1]["br0_bb"] >= v - 1e-9 and curve[-1]["br1_bb"] >= -v - 1e-9


def test_kernel_payoffs_match_the_exact_game(hu10):
    """The sampled kernel's value of a fixed random profile equals the exact vector value."""
    from strategy.blueprint.exploit import _deal_inputs

    g = hu10
    rng = np.random.default_rng(2)
    table = rng.random(g.B * A.CONTEXTS * A.NUM_ACTIONS).astype(np.float32) + 0.05
    sigma = g.profile(table.reshape(-1, A.NUM_ACTIONS), g.table_keys(g.bucket))
    exact = g.value(sigma)
    deck, pair_class, class_bucket = _deal_inputs(g, g.bucket)
    B = 60000
    out_bucket = np.zeros((B, 8, 4), np.uint8)
    out_class = np.zeros((B, 8), np.int16)
    K.deal_classes(5, B, 2, deck, pair_class, class_bucket, out_bucket, out_class)
    pool = g.tree.pool
    ip = np.zeros(K.NUM_IPARAMS, np.int64)
    ip[K.P_MAX_STREET], ip[K.P_EQ_SHOWDOWN], ip[K.P_OPP_FIXED], ip[K.P_TRAV_FIXED] = 0, 1, 1, 1
    ws = K.workspace(1)
    out = np.zeros((B, 8))
    K.seat_values(np.zeros(B, np.int64), out_bucket, np.zeros((B, 8), np.int64), out_class, pool.n,
                  pool.stacks.astype(np.int64), pool.posts.astype(np.int64), pool.bb.astype(np.int64), pool.prizes,
                  pool.post_order.astype(np.int64), pool.pressure.astype(np.int64), g.eq, np.zeros(4, np.int64),
                  np.zeros(1, np.float32), np.zeros(1, np.float32), table, ip, np.zeros(2), ws["I"][0],
                  ws["SIG"][0], ws["VAL"][0], ws["f"][0], ws["alive"][0], ws["fin"][0], np.zeros(1, np.int64), out)
    est, se = out[:, 0].mean(), out[:, 0].std() / np.sqrt(B)
    assert abs(est - exact) < 4 * se + 1e-3, (est, exact, se)
    other, se1 = out[:, 1].mean(), out[:, 1].std() / np.sqrt(B)
    assert abs(other + exact) < 4 * se1 + 1e-3


def test_sampled_training_improves_on_the_uniform_profile(hu10):
    from strategy.blueprint.exploit import es_run

    g = hu10
    keys = g.table_keys(g.bucket)
    uniform = g.exploitability(g.profile(np.ones((g.B * A.CONTEXTS, A.NUM_ACTIONS)), keys))["mbb_per_hand"]
    rows = es_run(g, variant="linear", deals=200_000, checkpoints=(200_000,))
    assert rows[-1]["mbb_per_hand"] < uniform / 5


# ----------------------------------------------------------- boundary audit


def test_audit_sequences_replay_in_the_kernel():
    from strategy.blueprint.audit import node_context, node_sequence
    from strategy.endgame.harvest import decision_nodes, subgame_for
    from strategy.endgame.subgame import SubgameTree
    from strategy.tables.endgame import default_config, synthetic_state

    rng = np.random.default_rng(3)
    config = default_config("practice_nlhe")
    seen = set()
    states = 0
    while states < 40:
        s = synthetic_state(rng, config)
        if s is None:
            continue
        states += 1
        tree = SubgameTree(subgame_for(s.snapshot, odd_order=s.odd_order))
        for kind, node, hero, shover, caller in decision_nodes(tree):
            found = node_context(s.snapshot, node_sequence(s.snapshot, kind, hero, shover, caller), hero)
            if found is None:
                continue
            ctx, mask = found
            assert 0 <= ctx < A.CONTEXTS and (mask >> A.CALL) & 1
            if kind != "open" and A.context_fields(ctx)["raises"] >= 1:
                assert (mask >> A.FOLD) & 1
            seen.add(kind)
    assert {"open", "call"} <= seen
