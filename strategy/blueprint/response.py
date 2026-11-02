"""Approximate best responses to a blueprint pack in the full abstract game (offline).

One deviator learns against seven players fixed to the pack, exactly as the deployed
bot would play it (the per-key row :meth:`PolicyPack.resolved` gives, back-off
included). The deviator is a greedy Monte Carlo best response in the same abstraction,
trained by external sampling on the training state pool and fresh deals: at each of
its information sets it accumulates the sampled regrets of every action against the
strategy it played and their squares, and plays the action with the largest cumulative
regret once that regret exceeds ``z`` times the square root of its summed squares
(roughly, its mean advantage over what it played is ``z`` standard errors above zero);
until then it plays the pack's row. ``streets`` limits deviations to the first streets
(1: preflop only). Regret matching on this many information sets is mostly sampling
noise at feasible budgets and loses to the pack, which is why the response is greedy
and gated. The untrained responder is the pack itself (the first curve row checks
this: zero gain). It sees only what the abstraction distinguishes, so its gain is a
lower bound on the pack's exploitability by an abstraction-limited opponent.

The gain is measured on held-out deals: for every (deal, seat) the deviator's frozen
strategy and the pack itself play that seat against the pack, sharing opponent
and chance samples until their play diverges, and the paired difference is averaged
per player-hand. Units are tournament points (out of 100 for the 50/30/20 payout) per
player-hand; the trainer's utilities are points over the first prize.

    python -m strategy.blueprint.response --pack P.npz [--pack Q.npz ...] --pairs 3000000 --out R.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from strategy.blueprint import abstraction as A
from strategy.blueprint.deals import deal_stream, make_deals
from strategy.blueprint.policy import PolicyPack
from strategy.blueprint.states import StatePool

FIRST_PRIZE_POINTS = 50.0
EVAL_SEED = 7_000_001
TRAIN_SEED = 900


class Response:
    def __init__(self, pool: StatePool, pack: PolicyPack, *, threads: int, z: float, streets: int) -> None:
        from strategy.blueprint import kernels as K

        self.K = K
        self.pool = pool
        self.threads = threads
        self.fixed = np.ascontiguousarray(pack.resolved().ravel())
        self.regret = np.zeros(A.NUM_KEYS * A.NUM_ACTIONS, np.float32)
        self.sq = np.zeros(A.NUM_KEYS * A.NUM_ACTIONS, np.float32)
        self.streets = streets
        self.ws = K.workspace(threads)
        self.offsets = np.array(A.OFFSETS, np.int64)
        self.order = pool.post_order.astype(np.int64)
        self.press = pool.pressure.astype(np.int64)
        self.eq = np.zeros((1, 1))
        self.fp = np.array([-1e30, -1e30, z])
        self.pairs = 0
        self.nodes = 0

    def _ip(self, *, trav_fixed: bool, freeze: bool) -> np.ndarray:
        K = self.K
        ip = np.zeros(K.NUM_IPARAMS, np.int64)
        ip[K.P_MODE] = 1
        ip[K.P_MAX_STREET] = 3
        ip[K.P_OPP_FIXED] = 1
        ip[K.P_TRAV_FIXED] = 1 if trav_fixed else 0
        ip[K.P_FREEZE] = 1 if freeze else 0
        ip[K.P_RESPOND] = self.streets
        return ip

    def _args(self, deals: dict[str, np.ndarray], states: np.ndarray) -> tuple:
        p = self.pool
        return (states, deals["buckets"], deals["ranks"], deals["classes"], p.n, p.stacks, p.posts, p.bb, p.prizes,
                self.order, self.press, self.eq, self.offsets, self.regret, self.sq, self.fixed)

    def train(self, chunk_index: int, deals: dict[str, np.ndarray]) -> None:
        count = deals["buckets"].shape[0]
        rng = np.random.default_rng([TRAIN_SEED, chunk_index])
        states = rng.integers(0, len(self.pool), count).astype(np.int64)
        counters = np.zeros((self.threads, 1), np.int64)
        values = np.zeros((self.threads, 2))
        ws = self.ws
        self.K.train_batch(int(rng.integers(1 << 30)), self.threads, *self._args(deals, states),
                           self._ip(trav_fixed=False, freeze=False), np.zeros(count, np.bool_), self.fp, ws["I"],
                           ws["SIG"], ws["VAL"], ws["f"], ws["alive"], ws["fin"], counters, values)
        self.pairs += count
        self.nodes += int(counters.sum())

    def values(self, deals: dict[str, np.ndarray], states: np.ndarray, *, responder: bool) -> np.ndarray:
        out = np.full((states.size, 8), np.nan)
        counters = np.zeros((self.threads, 1), np.int64)
        ws = self.ws
        self.K.seat_values_par(EVAL_SEED, self.threads, *self._args(deals, states),
                               self._ip(trav_fixed=not responder, freeze=True), self.fp, ws["I"], ws["SIG"],
                               ws["VAL"], ws["f"], ws["alive"], ws["fin"], counters, out)
        return out


def paired_gain(base: np.ndarray, resp: np.ndarray) -> dict[str, float]:
    """Mean gain per player-hand in points, with a deal-clustered standard error."""
    d = (resp - base) * FIRST_PRIZE_POINTS
    live = ~np.isnan(d)
    per_deal = np.where(live, d, 0.0).sum(axis=1)
    n = live.sum(axis=1)
    gain = per_deal.sum() / n.sum()
    resid = per_deal - gain * n
    se = resid.std(ddof=1) / (n.mean() * math.sqrt(len(n)))
    return {"gain_points": float(gain), "se": float(se), "player_hands": int(n.sum()),
            "base_points": float(np.nanmean(base) * FIRST_PRIZE_POINTS)}


def respond(pack_path: Path, pool: StatePool, *, pairs: int, checkpoints: tuple[int, ...], eval_deals: int,
            threads: int, producers: int, z: float, streets: int, ruleset_id: str = "practice_nlhe",
            log=print) -> dict:
    pack = PolicyPack(pack_path)
    r = Response(pool, pack, threads=threads, z=z, streets=streets)
    rng = np.random.default_rng(EVAL_SEED)
    ev_states = rng.integers(0, len(pool), eval_deals).astype(np.int64)
    ev_deals = make_deals(ruleset_id, EVAL_SEED, eval_deals)
    t0 = time.time()
    base = r.values(ev_deals, ev_states, responder=False)
    log(f"{pack_path.name}: base evaluated in {time.time() - t0:.0f}s")
    rows = []
    g = paired_gain(base, r.values(ev_deals, ev_states, responder=True))
    g.update(pairs=0, nodes=0, seconds=0.0)
    rows.append(g)
    todo = sorted(c for c in checkpoints if c <= pairs)
    if not todo or todo[-1] != pairs:
        todo.append(pairs)
    stream = deal_stream(ruleset_id, 0, 20000, jobs=producers, base_seed=TRAIN_SEED)
    t0 = time.time()
    try:
        for ci, deals in stream:
            if r.pairs >= pairs:
                break
            r.train(ci, deals)
            if todo and r.pairs >= todo[0]:
                todo.pop(0)
                g = paired_gain(base, r.values(ev_deals, ev_states, responder=True))
                g.update(pairs=r.pairs, nodes=r.nodes, seconds=round(time.time() - t0, 1))
                rows.append(g)
                log(f"{pack_path.name} z {z} streets {streets}: {r.pairs:>10,} pairs  gain {g['gain_points']:+.4f} ± {g['se']:.4f}"
                    f" points/player-hand  ({g['seconds']:.0f}s)")
    finally:
        stream.close()
    return {"pack": str(pack_path), "provenance": {k: pack.provenance.get(k) for k in ("run", "milestone", "progress")},
            "z": z, "streets": streets, "eval_deals": eval_deals, "curve": rows}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pack", type=Path, action="append", required=True)
    ap.add_argument("--states", type=Path, default=Path("artifacts/blueprint/states-practice_nlhe-s1.npz"))
    ap.add_argument("--pairs", type=int, default=3_000_000)
    ap.add_argument("--checkpoint", type=int, action="append", default=[])
    ap.add_argument("--z", type=float, default=2.0, help="deviation threshold in standard errors")
    ap.add_argument("--streets", type=int, action="append", default=[],
                    help="deviate on the first N streets (1: preflop only, 4: all; repeatable; default 4)")
    ap.add_argument("--eval-deals", type=int, default=100_000)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--producers", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    pool, _ = StatePool.load(a.states)
    results = []
    for p in a.pack:
        for streets in a.streets or [4]:
            results.append(respond(p, pool, pairs=a.pairs, checkpoints=tuple(a.checkpoint), eval_deals=a.eval_deals,
                                   threads=a.threads, producers=a.producers, z=a.z, streets=streets))
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps({"states": str(a.states), "results": results}, indent=1))


if __name__ == "__main__":
    main()
