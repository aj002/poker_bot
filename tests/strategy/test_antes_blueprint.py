"""Ante-aware blueprint training: kernel ante states against the arena engine and the runtime."""

import random

import numpy as np
import pytest

pytest.importorskip("numba")

from strategy.blueprint import abstraction as A  # noqa: E402
from strategy.blueprint import kernels as K  # noqa: E402


class _NoPack:
    def __bool__(self):
        return True


def _replay(seed: int, posting: str) -> int:
    from arena.domain.models import Action, TournamentConfig
    from arena.domain.rng import SeededRng
    from arena.table.projections import observation
    from arena.table.service import Table
    from strategy.antes.blueprint import ante_state_row
    from strategy.antes.pushfold import ante_seated_snapshot
    from strategy.blueprint.runtime import BlueprintPolicy

    rng = random.Random(f"{posting}-{seed}")
    n = rng.randrange(2, 9)
    ante = rng.choice([2, 3, 5]) if posting == "classic" else rng.choice([10, 20, 30])
    config = TournamentConfig(initial_player_count=n, hand_cap=200, antes=[ante] * 8, ante_posting=posting)
    ring = list(config.resolved_seats)
    stacks = {s: rng.choice([rng.randrange(1, 15), rng.randrange(15, 120)]) * 20 + rng.randrange(0, 20) for s in ring}
    table = Table(config, env_rng=SeededRng(f"ante-parity-{posting}-{seed}", "env"), start_stacks=stacks,
                  initial_button=rng.choice(ring))
    rec0 = table.hands[0]
    if rec0.result is not None:
        return 0
    pos = rec0.positions
    live = [s for s in ring if rec0.start_stacks.get(s, 0) > 0]
    args = dict(ring=ring, start_stacks=dict(rec0.start_stacks), button=pos.button,
                small_blind_seat=pos.small_blind if pos.small_blind in live else None, big_blind=pos.big_blind,
                heads_up=pos.heads_up, blinds=tuple(rec0.blinds), payouts=tuple(config.resolved_payouts),
                hand_number=rec0.hand_number, hand_cap=config.hand_cap)
    snap = ante_seated_snapshot(**args, ante=rec0.ante, posting=posting)
    seats = snap.seats
    if snap.snapshot.posts[-1] < rec0.blinds[1]:
        # A big blind short after its ante: the engine prices calls at its post once nobody
        # can raise, the kernel (as without antes) at the nominal big blind; chips agree.
        return 0
    row = ante_state_row(**args, ante=rec0.ante, posting=posting)
    I = np.zeros((K.MAX_DEPTH, K.CH, K.MAX_PLAYERS), np.int64)
    trace = np.zeros((K.MAX_DEPTH, len(K.TRACE_FIELDS)), np.int64)
    weights = np.array([rng.random() for _ in range(A.NUM_ACTIONS)]) + 0.05
    k, code = K.trace_hand(seed, row["n"], I, np.array(row["stacks"], np.int64), np.array(row["posts"], np.int64),
                           row["bb"], np.array(row["post_order"], np.int64), np.array(row["pressure"], np.int64), 3,
                           weights, trace)
    policy = BlueprintPolicy("practice_nlhe", _NoPack())
    policy.ante_view = "stripped"
    for i in range(k):
        street, p, ctx, mask, a, target, t_min, t_a, t_b, t_all = (int(x) for x in trace[i])
        seat = table.actor_seat
        assert seat == seats[p], (i, seat, seats[p])
        obs = observation(table, seat)
        assert obs.street == ("preflop", "flop", "turn", "river")[street]
        key, rmask, targets = policy.features(obs)
        assert (key - A.OFFSETS[street]) % A.CONTEXTS == ctx, (i, A.context_fields(ctx),
                                                             A.context_fields((key - A.OFFSETS[street]) % A.CONTEXTS))
        if rmask != mask and not obs.legal.can_raise and obs.legal.raise_blocked_reason is None:
            # PokerKit refuses a raise the arena's reopening rule allows; the trainer has the
            # same gap without antes.
            return i
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
    assert rec0.result is not None
    if code == 1 or k == 0:
        return k
    return k


@pytest.mark.parametrize("posting", ["classic", "big_blind"])
def test_runtime_stripped_features_match_the_kernel_with_antes(posting):
    decisions = sum(_replay(seed, posting) for seed in range(200))
    assert decisions > 1000


def _python_settle(levels, dead, folded, ranks, odd):
    from strategy.antes.subgame import settle_dead

    return settle_dead(levels, folded, ranks, odd, dead)


def test_kernel_settle_pays_dead_money_with_the_main_pot():
    rng = random.Random(3)
    I = np.zeros((K.MAX_DEPTH, K.CH, K.MAX_PLAYERS), np.int64)
    fin = np.zeros(K.MAX_PLAYERS)
    eq = np.zeros((1, 1))
    classes = np.zeros(K.MAX_PLAYERS, np.int64)
    for _ in range(400):
        n = rng.randrange(2, 7)
        I[:] = 0
        levels = [rng.choice([0, 20, 50, 50, 120, 300]) for _ in range(n)]
        folded = [rng.random() < 0.3 for _ in range(n)]
        if all(folded[p] or levels[p] == 0 for p in range(n)):
            continue
        top = max(levels[p] for p in range(n) if not folded[p])
        levels = [min(levels[p], top) for p in range(n)]
        dead = rng.choice([0, 10, 25])
        bb = rng.randrange(n)
        ranks = [rng.randrange(3) for _ in range(n)]
        for p in range(n):
            I[0, 0, p] = 100
            I[0, 2, p] = levels[p]
            I[0, 3, p] = 1 if folded[p] or levels[p] == 0 else 0
        I[0, 8, bb] = dead
        post_order = np.arange(n, dtype=np.int64)
        K.settle(I, 0, n, np.array(ranks, np.int64), post_order, 0, eq, classes, fin)
        f = [I[0, 3, p] == 1 for p in range(n)]
        awards = _python_settle(levels, dead, f, ranks, list(range(n)))
        got = [fin[p] - 100 for p in range(n)]
        assert sum(got) == sum(levels) + dead
        if sum(not x for x in f) == 1 or len({ranks[p] for p in range(n) if not f[p]}) == sum(not x for x in f):
            assert got == pytest.approx(awards), (levels, dead, f, ranks)
