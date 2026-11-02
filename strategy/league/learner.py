"""The league's abstract game: response training against an opponent mixture and paired evaluation.

Offline; the kernels need numba (``--group train``). Every policy is a ``uint8`` table
(:mod:`strategy.league.tables`) stacked in :class:`AbstractLeague`. A *seat mixture* is a
categorical distribution over those tables; each deal draws the table of every seat
independently from it. Hands start from the league's state pool (start-of-hand stacks,
blinds and prizes harvested from training-partition rollouts) and pay tournament points
over the first prize (ICM of the survivors plus finishing prizes), as in the blueprint
trainer.

Evaluation is paired and frozen: a fixed set of held-out deals, start states and seat
draws, with chance and opponent samples keyed by ``(seed, deal, seat)``. The gain of
table ``a`` over table ``b`` against a mixture is the per-player-hand difference of their
seat values, in points (out of 100), with a deal-clustered standard error
(:func:`strategy.blueprint.response.paired_gain`).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from strategy.blueprint import abstraction as A
from strategy.blueprint.deals import deal_stream
from strategy.blueprint.response import FIRST_PRIZE_POINTS, paired_gain
from strategy.blueprint.states import StatePool

EVAL_SEED = 7_100_001


@dataclass(frozen=True)
class SeatMixture:
    """A categorical over table indices; every non-learner seat draws from it independently."""

    rows: tuple[int, ...]
    probs: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.rows) != len(self.probs) or not self.rows:
            raise ValueError("a mixture needs one probability per table")
        if any(p < 0 for p in self.probs) or abs(sum(self.probs) - 1.0) > 1e-6:
            raise ValueError("mixture probabilities must be non-negative and sum to 1")

    @classmethod
    def pure(cls, row: int) -> SeatMixture:
        return cls((row,), (1.0,))

    @classmethod
    def combine(cls, parts: Sequence[tuple[float, Mapping[int, float]]]) -> SeatMixture:
        """``[(weight, {row: prob})]`` flattened into one categorical (weights renormalised)."""
        acc: dict[int, float] = {}
        total = sum(w for w, m in parts if m)
        for w, m in parts:
            s = sum(m.values())
            if not m or s <= 0:
                continue
            for r, p in m.items():
                acc[r] = acc.get(r, 0.0) + w / total * p / s
        rows = tuple(sorted(r for r, p in acc.items() if p > 0))
        return cls(rows, tuple(acc[r] for r in rows))

    def draw(self, rng: np.random.Generator, deals: int) -> np.ndarray:
        idx = rng.choice(len(self.rows), size=(deals, 8), p=np.asarray(self.probs))
        return np.asarray(self.rows, np.int64)[idx]


class AbstractLeague:
    def __init__(self, pool: StatePool, *, threads: int, ruleset_id: str = "practice_nlhe") -> None:
        from strategy.blueprint import kernels as BK
        from strategy.league import kernels as K

        self.K = K
        self.pool = pool
        self.ruleset_id = ruleset_id
        self.threads = threads
        self.ws = BK.workspace(threads)
        self.offsets = np.array(A.OFFSETS, np.int64)
        self.eq = np.zeros((1, 1))
        self.names: list[str] = []
        self.tabs = np.zeros((0, A.NUM_KEYS * A.NUM_ACTIONS), np.uint8)
        self._eval: dict | None = None

    # ------------------------------------------------------------------ tables

    def add(self, name: str, table: np.ndarray) -> int:
        if name in self.names:
            return self.names.index(name)
        if table.dtype != np.uint8 or table.shape != (A.NUM_KEYS * A.NUM_ACTIONS,):
            raise ValueError("a table is a flat uint8 array over every key and action")
        self.tabs = np.concatenate([self.tabs, table[None, :]])
        self.names.append(name)
        return len(self.names) - 1

    def row(self, name: str) -> int:
        return self.names.index(name)

    def _states(self, pool: StatePool | None = None) -> tuple:
        p = pool or self.pool
        return (p.n, p.stacks, p.posts, p.bb, p.prizes, p.post_order.astype(np.int64), p.pressure.astype(np.int64),
                self.eq, self.offsets)

    # ------------------------------------------------------------------ training

    def train_response(self, base: str, mixture: SeatMixture, *, pairs: int, z: float, seed: int,
                       producers: int = 2, chunk: int = 20000,
                       log: Callable[[str], None] | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
        """Regret and squared-increment sums of a gated response to ``mixture`` over table ``base``."""
        K, ws = self.K, self.ws
        slots = A.NUM_KEYS * A.NUM_ACTIONS
        regret = np.zeros(slots, np.float32)
        sq = np.zeros(slots, np.float32)
        base_row = self.row(base)
        done = nodes = 0
        t0 = time.time()
        stream = deal_stream(self.ruleset_id, 0, chunk, jobs=producers, base_seed=seed)
        try:
            for ci, deals in stream:
                if done >= pairs:
                    break
                rng = np.random.default_rng([seed, ci])
                count = deals["buckets"].shape[0]
                states = rng.integers(0, len(self.pool), count).astype(np.int64)
                seats = mixture.draw(rng, count)
                counters = np.zeros((self.threads, 1), np.int64)
                K.train_batch(int(rng.integers(1 << 30)), self.threads, states, seats, deals["buckets"], deals["ranks"],
                              deals["classes"], *self._states(), regret, sq, self.tabs, base_row, z, ws["I"],
                              ws["SIG"], ws["VAL"], ws["f"], ws["alive"], ws["fin"], counters)
                done += count
                nodes += int(counters.sum())
                if log and (ci + 1) % 25 == 0:
                    log(f"  {done:>10,} pairs  {nodes / 1e9:.2f}B nodes  {time.time() - t0:.0f}s")
        finally:
            stream.close()
        return regret, sq, {"pairs": done, "nodes": nodes, "seconds": round(time.time() - t0, 1), "z": z,
                            "seed": seed}

    # ------------------------------------------------------------------ evaluation

    def set_eval(self, deals: dict[str, np.ndarray], pool: StatePool | None = None, seed: int = EVAL_SEED) -> None:
        """Frozen evaluation deals and start states (``pool`` defaults to the training pool)."""
        pool = pool or self.pool
        rng = np.random.default_rng([seed, 1])
        count = deals["buckets"].shape[0]
        self._eval = {"deals": deals, "states": rng.integers(0, len(pool), count).astype(np.int64),
                      "seed": seed, "count": count, "pool": pool, "args": self._states(pool)}

    def seat_draws(self, mixture: SeatMixture, tag: int = 0) -> np.ndarray:
        """Frozen seat draws for ``mixture`` on the evaluation deals (the same for every learner)."""
        assert self._eval is not None, "call set_eval first"
        rng = np.random.default_rng([self._eval["seed"], 2, tag])
        return mixture.draw(rng, self._eval["count"])

    def values(self, learner: str, seats: np.ndarray) -> np.ndarray:
        """Points / first prize of every live seat played by ``learner`` against ``seats`` (NaN: no seat)."""
        assert self._eval is not None, "call set_eval first"
        ev = self._eval
        out = np.full((ev["count"], 8), np.nan)
        ws = self.ws
        counters = np.zeros((self.threads, 1), np.int64)
        d = ev["deals"]
        self.K.seat_values(ev["seed"], self.threads, ev["states"], seats, d["buckets"], d["ranks"], d["classes"],
                           *ev["args"], self.tabs, self.row(learner), ws["I"], ws["SIG"], ws["VAL"], ws["f"],
                           ws["alive"], ws["fin"], counters, out)
        return out

    def gain(self, a: str, b: str, seats: np.ndarray, *, base_values: np.ndarray | None = None) -> dict:
        """Points per player-hand of ``a`` over ``b`` in the same seat against ``seats``."""
        vb = base_values if base_values is not None else self.values(b, seats)
        return paired_gain(vb, self.values(a, seats))


def homogeneous(row: int, count: int) -> np.ndarray:
    return np.full((count, 8), row, np.int64)


def payoff_matrix(league: AbstractLeague, names: Sequence[str], cache: dict[str, dict] | None = None,
                  log: Callable[[str], None] | None = None) -> dict:
    """``G[a][b]``: points per player-hand of ``a`` over ``b`` when the other seats all play ``b``.

    ``G[b][b] = 0``. ``cache`` maps ``"a|b"`` to earlier entries computed on the same
    evaluation set and is updated in place.
    """
    cache = {} if cache is None else cache
    assert league._eval is not None
    count = league._eval["count"]
    for b in names:
        todo = [a for a in names if a != b and f"{a}|{b}" not in cache]
        if not todo:
            continue
        seats = homogeneous(league.row(b), count)
        vb = league.values(b, seats)
        for a in todo:
            cache[f"{a}|{b}"] = league.gain(a, b, seats, base_values=vb)
            if log:
                g = cache[f"{a}|{b}"]
                log(f"  G[{a}][{b}] = {g['gain_points']:+.4f} ± {g['se']:.4f}")
    g = [[0.0 if a == b else cache[f"{a}|{b}"]["gain_points"] for b in names] for a in names]
    se = [[0.0 if a == b else cache[f"{a}|{b}"]["se"] for b in names] for a in names]
    return {"names": list(names), "gain": g, "se": se, "units": "points per player-hand",
            "first_prize_points": FIRST_PRIZE_POINTS}
