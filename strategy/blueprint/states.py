"""Start-of-hand states for blueprint training, harvested from complete tournament rollouts.

Each state is one hand of a seeded arena tournament (8 players, the default blind
schedule and payouts) at its start: the full stack vector in preflop action order (big
blind last), the blind posts, the postflop order, the remaining prizes normalised by the
first prize, and each player's ICM risk-premium bucket. Rollouts use training-partition
lineups of the opponent population (optionally with fixed policies such as an earlier
blueprint in some seats); validation and holdout members never shape the pool.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from strategy.blueprint.abstraction import pressure_bucket
from strategy.endgame.keys import pressure
from strategy.endgame.table_state import seated_snapshot

MAX_PLAYERS = 8


@dataclass
class StatePool:
    n: np.ndarray            # int32 [M]
    stacks: np.ndarray       # int64 [M, 8] start-of-hand chips, preflop order
    posts: np.ndarray        # int64 [M, 8]
    bb: np.ndarray           # int64 [M]
    prizes: np.ndarray       # float64 [M, 8] place k+1 among the n live players, / first prize
    post_order: np.ndarray   # int8 [M, 8] player indices in postflop order
    pressure: np.ndarray     # int8 [M, 8] pressure bucket
    hand_number: np.ndarray  # int32 [M]

    def __len__(self) -> int:
        return int(self.n.shape[0])

    def arrays(self) -> dict[str, np.ndarray]:
        return {k: getattr(self, k) for k in ("n", "stacks", "posts", "bb", "prizes", "post_order", "pressure",
                                              "hand_number")}

    def digest(self) -> str:
        h = hashlib.sha256()
        for k, v in self.arrays().items():
            h.update(k.encode())
            h.update(np.ascontiguousarray(v).tobytes())
        return h.hexdigest()

    def save(self, path: Path, provenance: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        prov = dict(provenance, digest=self.digest(), states=len(self))
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, __provenance__=np.array(json.dumps(prov, sort_keys=True)), **self.arrays())
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> tuple[StatePool, dict]:
        with np.load(path) as z:
            prov = json.loads(str(z["__provenance__"]))
            pool = cls(**{k: z[k] for k in ("n", "stacks", "posts", "bb", "prizes", "post_order", "pressure",
                                             "hand_number")})
        if pool.digest() != prov["digest"]:
            raise ValueError(f"{path}: state pool digest mismatch")
        return pool, prov

    def subset(self, idx: np.ndarray) -> StatePool:
        return StatePool(**{k: v[idx] for k, v in self.arrays().items()})


def state_row(ring: Sequence[int], start_stacks: dict[int, int], button: int, small_blind_seat: int | None,
              big_blind: int, heads_up: bool, blinds: tuple[int, int], payouts: Sequence[float], hand_number: int,
              hand_cap: int) -> dict | None:
    seated = seated_snapshot(ring=ring, start_stacks=start_stacks, button=button, small_blind_seat=small_blind_seat,
                             big_blind=big_blind, heads_up=heads_up, blinds=blinds, payouts=payouts,
                             hand_number=hand_number, hand_cap=hand_cap)
    snap = seated.snapshot
    n = snap.n
    if n < 2 or n > MAX_PLAYERS:
        return None
    live = [s for s in ring if start_stacks.get(s, 0) > 0]
    i = ring.index(button) if button in ring else 0
    after = [ring[(i + 1 + k) % len(ring)] for k in range(len(ring))]
    post_seats = [s for s in after if s in live]
    post = [seated.seats.index(s) for s in post_seats]
    first = float(payouts[0])
    prizes = [float(payouts[k]) / first if k < len(payouts) else 0.0 for k in range(n)]
    stacks = list(snap.stacks)
    press = []
    for p in range(n):
        others = [q for q in range(n) if q != p]
        v = max(others, key=lambda q: (stacks[q], -q))
        press.append(pressure_bucket(pressure(stacks, p, v, prizes)))
    pad = MAX_PLAYERS - n
    return {"n": n, "stacks": stacks + [0] * pad, "posts": list(snap.posts) + [0] * pad, "bb": snap.bb,
            "prizes": prizes + [0.0] * pad, "post_order": post + [-1] * pad, "pressure": press + [0] * pad,
            "hand_number": hand_number}


def pool_from_rows(rows: Sequence[dict]) -> StatePool:
    return StatePool(
        n=np.array([r["n"] for r in rows], dtype=np.int32),
        stacks=np.array([r["stacks"] for r in rows], dtype=np.int64),
        posts=np.array([r["posts"] for r in rows], dtype=np.int64),
        bb=np.array([r["bb"] for r in rows], dtype=np.int64),
        prizes=np.array([r["prizes"] for r in rows], dtype=np.float64),
        post_order=np.array([r["post_order"] for r in rows], dtype=np.int8),
        pressure=np.array([r["pressure"] for r in rows], dtype=np.int8),
        hand_number=np.array([r["hand_number"] for r in rows], dtype=np.int32),
    )


def rollout_pool(ruleset_id: str, *, lineups: int, seed: int, jobs: int, fixed: Sequence[str] = ()) -> StatePool:
    """Every hand of ``lineups`` training lineups, each played in all seat rotations."""
    from arena.domain.models import TournamentConfig
    from strategy.value.rollouts import run_lineups, training_lineups

    config = TournamentConfig(ruleset_id=ruleset_id)
    teams = training_lineups(ruleset_id, config.initial_player_count, lineups, seed, fixed=fixed)
    rollouts = run_lineups(config, teams, blocks=1, base_seed=f"blueprint-states-{seed}", jobs=jobs)
    rows = []
    for ro in rollouts:
        for h in ro.hands:
            r = state_row(ro.ring, h.start_stacks, h.button, h.small_blind_seat, h.big_blind, h.heads_up, h.blinds,
                          ro.payouts, h.hand_number, ro.hand_cap)
            if r is not None:
                rows.append(r)
    return pool_from_rows(rows)


def heads_up_pool(stack_bb: float, *, bb: int = 100, payouts: Sequence[float] = (50, 30)) -> StatePool:
    """One heads-up state (button posts the small blind) with both players on ``stack_bb``."""
    chips = int(round(stack_bb * bb))
    r = state_row((0, 1), {0: chips, 1: chips}, 0, 0, 1, True, (bb // 2, bb), payouts, 1, 200)
    assert r is not None
    return pool_from_rows([r])


def main(argv: list[str] | None = None) -> None:
    import argparse
    import time

    ap = argparse.ArgumentParser(description="Harvest blueprint start states from training-lineup rollouts")
    ap.add_argument("--ruleset", default="practice_nlhe")
    ap.add_argument("--lineups", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--fixed", action="append", default=[], help="policy spec seated in every lineup")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    t = time.time()
    pool = rollout_pool(a.ruleset, lineups=a.lineups, seed=a.seed, jobs=a.jobs, fixed=a.fixed)
    pool.save(a.out, {"ruleset_id": a.ruleset, "lineups": a.lineups, "seed": a.seed, "fixed": a.fixed,
                      "partition": "train", "wall_seconds": round(time.time() - t, 1)})
    hist = np.bincount(pool.n, minlength=9)[2:]
    print(f"{len(pool)} states -> {a.out}; live-player counts 2..8: {hist.tolist()}; "
          f"pressure buckets: {np.bincount(pool.pressure[pool.stacks > 0].ravel(), minlength=4).tolist()}")


if __name__ == "__main__":
    main()
