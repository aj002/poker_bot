"""Unabstracted boundary audit: the blueprint's point regret in exact push/fold states (offline).

Stage 3's table audit, applied to the blueprint. On held-out short-stack tournament
states (the generator seed :mod:`strategy.endgame.gates` audits with, never used for
tables), each state is solved exactly in tournament points (no card or context
abstraction; push/fold actions). At every covered decision the player plays the
blueprint instead, the others keep the exact solution, and the audit reports the
player's loss in normalised points (points / first prize).

The blueprint's abstract row for a decision comes from the kernel's own context
(:func:`strategy.blueprint.kernels.decision_context`) after replaying the node's action
sequence, and its card bucket for each hand class, so the audit sees exactly the keys
the bot looks up. A blueprint row maps to push/fold by commitment: every non-fold
action counts as a shove when opening and as a call when facing one.

The tolerance is declared here, before any measurement:
mean loss <= :data:`TOLERANCE_MEAN` and p90 <= :data:`TOLERANCE_P90`.

    python -m strategy.blueprint.audit --pack strategy/tables/data/blueprint-practice_nlhe.npz
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from strategy.blueprint import abstraction as A

TOLERANCE_MEAN = 0.005
TOLERANCE_P90 = 0.01
AUDIT_SEED = 97


def kernel_state(snap):
    """Kernel start arrays for a :class:`strategy.endgame.keys.Snapshot` (index order = preflop order)."""
    from strategy.endgame.keys import pressure

    n = snap.n
    stacks = np.zeros(8, np.int64)
    posts = np.zeros(8, np.int64)
    stacks[:n] = snap.stacks
    posts[:n] = snap.posts
    order = np.full(8, -1, np.int64)
    order[:n] = [1, 0] if n == 2 else [n - 2, n - 1, *range(n - 2)]
    first = float(snap.payouts[0])
    prizes = [float(snap.payouts[k]) / first if k < len(snap.payouts) else 0.0 for k in range(n)]
    press = np.zeros(8, np.int64)
    for p in range(n):
        v = max((q for q in range(n) if q != p), key=lambda q: (snap.stacks[q], -q))
        press[p] = A.pressure_bucket(pressure(list(snap.stacks), p, v, prizes))
    return stacks, posts, order, press


def node_context(snap, sequence, hero):
    """``(context, mask)`` of ``hero`` after ``sequence`` of ``(player, abstract action)``.

    None when the hero has no real decision there (e.g. a big blind facing only an
    all-in for less than the blind: nothing to call and nobody left to raise).
    """
    from strategy.blueprint import kernels as K

    n = snap.n
    stacks, posts, order, press = kernel_state(snap)
    I = np.zeros((K.MAX_DEPTH, K.CH, K.MAX_PLAYERS), np.int64)
    K.init_hand(I, n, stacks, posts, snap.bb)
    d = 0
    for player, action in [*sequence, (hero, None)]:
        code = K.resolve(I, d, n, order, 0, snap.bb)
        p = K.order_player(I[d, 7, K.ST_STREET], I[d, 7, K.ST_POS], order)
        if action is None and (code != 0 or p != player):
            return None
        if code != 0 or p != player:
            raise AssertionError(f"kernel expected player {p} (code {code}), sequence has {player}")
        ctx, mask, t_min, t_a, t_b, t_all = K.decision_context(I, d, n, p, snap.bb, press[p])
        if action is None:
            return int(ctx), int(mask)
        if action == A.ALL_IN and not (mask >> action) & 1:
            action = A.CALL
        if not (mask >> action) & 1:
            raise AssertionError(f"action {action} illegal for player {p}")
        K.copy_level(I, d)
        tgt = {A.MIN_RAISE: t_min, A.RAISE_A: t_a, A.RAISE_B: t_b, A.ALL_IN: t_all}.get(action, 0)
        K.apply_action(I, d + 1, n, p, action, tgt)
        d += 1
    raise AssertionError("unreachable")


def node_sequence(snap, kind, hero, shover, caller):
    """The abstract actions before a push/fold decision node (players all-in from posting never act)."""
    if kind == "open":
        seq = [(p, A.FOLD) for p in range(hero)]
    else:
        seq = [(p, A.FOLD) for p in range(shover)] + [(shover, A.ALL_IN)]
        if kind == "call":
            seq += [(p, A.FOLD) for p in range(shover + 1, hero)]
        else:
            seq += ([(p, A.FOLD) for p in range(shover + 1, caller)] + [(caller, A.CALL)]
                    + [(p, A.FOLD) for p in range(caller + 1, hero)])
    return [(p, a) for p, a in seq if snap.stacks[p] > snap.posts[p]]


def commit_probability(row, mask) -> float | None:
    w = [float(row[a]) if (mask >> a) & 1 else 0.0 for a in range(A.NUM_ACTIONS)]
    tot = sum(w)
    if tot <= 0:
        return None
    return 1.0 - w[A.FOLD] / tot


def _audit_state(args):
    from strategy.blueprint.cards import CardAbstraction
    from strategy.blueprint.policy import PolicyPack
    from strategy.endgame.harvest import decision_nodes, solve_snapshot
    from strategy.endgame.showdown import load_showdown
    from strategy.features.cards import capabilities

    ruleset, snap, odd, pack_path = args
    model = load_showdown(ruleset)
    pack = PolicyPack(pack_path)
    cards = CardAbstraction(capabilities(ruleset))
    bucket = [cards.preflop(list(c[0]))[0] for c in model.classes.combos]
    solved = solve_snapshot(snap, model, utility="points", odd_order=odd, with_gains=False)
    solver, sigma = solved.solver, solved.sigma
    base = solver.values(sigma)
    swapped: dict[int, dict[int, np.ndarray]] = {}
    covered = total = 0
    levels = [0] * len(pack.keys)
    for kind, node, hero, shover, caller in decision_nodes(solved.tree):
        found = node_context(snap, node_sequence(snap, kind, hero, shover, caller), hero)
        if found is None:
            continue
        total += 1
        ctx, mask = found
        p = np.zeros(model.h)
        ok = True
        for b in sorted(set(bucket)):
            hit = pack.lookup(A.infoset_key(0, b, ctx, A.OFFSETS))
            q = None if hit is None else commit_probability(hit[0], mask)
            if q is None:
                ok = False
                break
            levels[hit[1]] += 1
            p[[i for i, x in enumerate(bucket) if x == b]] = q
        if not ok:
            continue
        covered += 1
        swapped.setdefault(hero, {})[node] = np.stack([1 - p, p])
    loss = []
    for q in range(snap.n):
        if q in swapped:
            loss.append(float(base[q] - solver.values({**sigma, **swapped[q]})[q]) / snap.payouts[0])
    return snap.n, covered, total, loss, levels


def audit(pack_path: str, *, ruleset_id: str = "practice_nlhe", states: int = 160, jobs: int = 2) -> dict:
    from strategy.tables.endgame import default_config, synthetic_state

    config = default_config(ruleset_id)
    rng = np.random.default_rng(AUDIT_SEED)
    tasks = []
    while len(tasks) < states:
        s = synthetic_state(rng, config)
        if s is not None:
            tasks.append((ruleset_id, s.snapshot, s.odd_order, str(pack_path)))
    if jobs > 1:
        with ProcessPoolExecutor(jobs) as ex:
            rows = list(ex.map(_audit_state, tasks))
    else:
        rows = [_audit_state(t) for t in tasks]
    losses = [x for r in rows for x in r[3]]
    by_n = {}
    for n in sorted({r[0] for r in rows}):
        ls = [x for r in rows if r[0] == n for x in r[3]]
        by_n[str(n)] = {"players": len(ls), "mean_loss": float(np.mean(ls)) if ls else None,
                        "max_loss": float(max(ls)) if ls else None}
    mean = float(np.mean(losses))
    p90 = float(np.quantile(losses, 0.9))
    return {"states": len(rows), "unit": "normalised point", "pack": str(pack_path),
            "coverage": sum(r[1] for r in rows) / max(1, sum(r[2] for r in rows)),
            "lookup_levels": np.sum([r[4] for r in rows], axis=0).tolist(),
            "mean_loss": mean, "p90_loss": p90, "max_loss": float(max(losses)), "by_players": by_n,
            "tolerance": {"mean": TOLERANCE_MEAN, "p90": TOLERANCE_P90},
            "pass": mean <= TOLERANCE_MEAN and p90 <= TOLERANCE_P90}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Blueprint point regret in exact push/fold states")
    ap.add_argument("--pack", required=True)
    ap.add_argument("--ruleset", default="practice_nlhe")
    ap.add_argument("--states", type=int, default=160)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    res = audit(a.pack, ruleset_id=a.ruleset, states=a.states, jobs=a.jobs)
    print(json.dumps({k: v for k, v in res.items() if k != "by_players"}, indent=1))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
