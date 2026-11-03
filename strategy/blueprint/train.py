"""Blueprint training driver: external-sampling MCCFR with checkpoints and resume (offline).

A run directory holds ``config.json`` (fixed at creation), ``progress.jsonl`` (one line
per chunk) and checkpoints ``ckpt/`` (regrets, strategy sums, counters), replaced
atomically. ``python -m strategy.blueprint.train --run DIR ...`` creates or resumes; a
resumed run continues the same deal and state stream from the last checkpoint (chunk
seeds depend only on the chunk index). Lock-free multithreaded updates make runs with
``--threads > 1`` not bit-reproducible; ``--threads 1`` is.

Variants (``--variant``), all with external sampling and simple averaging at opponent
nodes:

``vanilla``  no discounting;
``linear``   linear CFR by interval discounting (Pluribus): after interval ``k`` regrets
             and strategy sums are multiplied by ``k / (k + 1)``, for ``linear_intervals``;
``dcfr``     discounted CFR (alpha 1.5, beta 0, gamma 2) on the same intervals;
``cfr+``     regrets floored at zero on every update, linear averaging.

``--prune-after`` enables regret-based pruning on 95% of traversals after that many
(deal, state) pairs: actions with zero probability and regret below the threshold are
not explored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from strategy.blueprint import abstraction as A
from strategy.blueprint.states import StatePool

VARIANTS = ("vanilla", "linear", "dcfr", "cfr+")
DCFR = (1.5, 0.0, 2.0)
STOP = "STOP"
"""A file of this name in the run directory ends training at the next chunk, with a checkpoint."""


def discount(regret: np.ndarray, strat: np.ndarray, variant: str, k: int) -> None:
    """In place, the discount applied after interval ``k`` (1-based)."""
    if variant in ("linear", "cfr+"):
        f = np.float32(k / (k + 1))
        if variant == "linear":
            regret *= f
        strat *= f
    elif variant == "dcfr":
        a, b, g = DCFR
        pos = np.float32(k ** a / (k ** a + 1))
        neg = np.float32(k ** b / (k ** b + 1))
        regret *= np.where(regret > 0, pos, neg).astype(np.float32)
        strat *= np.float32((k / (k + 1)) ** g)
    elif variant != "vanilla":
        raise ValueError(f"unknown variant {variant!r}")


@dataclass
class TrainConfig:
    ruleset_id: str = "practice_nlhe"
    states: str = ""
    states_digest: str = ""
    variant: str = "linear"
    mode: int = 1
    """0 chips (big blinds), 1 tournament points / first prize."""
    max_street: int = 3
    chunk: int = 20000
    reuse: int = 2
    """Start states paired with each deal (cards do not depend on stacks)."""
    seed: int = 1
    interval_pairs: int = 1_000_000
    linear_intervals: int = 40
    prune_after: int = 20_000_000
    prune_share: float = 0.95
    prune_threshold: float = -5.0
    regret_floor: float = -6.0
    abstraction: dict = field(default_factory=A.describe)

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


@dataclass
class Progress:
    chunks: int = 0
    pairs: int = 0
    traversals: int = 0
    nodes: int = 0
    intervals: int = 0
    wall_seconds: float = 0.0
    value_sum: float = 0.0


class Trainer:
    def __init__(self, cfg: TrainConfig, pool: StatePool, *, threads: int = 1) -> None:
        from strategy.blueprint import kernels as K

        self.K = K
        self.cfg = cfg
        self.pool = pool
        self.threads = threads
        self.slots = A.NUM_KEYS * A.NUM_ACTIONS
        self.regret = np.zeros(self.slots, np.float32)
        self.strat = np.zeros(self.slots, np.float32)
        self.progress = Progress()
        self.ws = K.workspace(threads)
        self.offsets = np.array(A.OFFSETS, np.int64)
        self.order = pool.post_order.astype(np.int64)
        self.press = pool.pressure.astype(np.int64)
        self.eq = np.zeros((1, 1))
        self.fixed = np.zeros(1, np.float32)

    # ------------------------------------------------------------------ steps

    def params(self, prune: bool) -> tuple[np.ndarray, np.ndarray]:
        K = self.K
        ip = np.zeros(K.NUM_IPARAMS, np.int64)
        ip[K.P_MODE] = self.cfg.mode
        ip[K.P_MAX_STREET] = self.cfg.max_street
        ip[K.P_UPDATE_AVG] = 1
        ip[K.P_PRUNE] = 1 if prune else 0
        ip[K.P_FLOOR_ZERO] = 1 if self.cfg.variant == "cfr+" else 0
        fp = np.array([self.cfg.prune_threshold, self.cfg.regret_floor if self.cfg.variant != "cfr+" else 0.0, 0.0])
        return ip, fp

    def batch(self, chunk_index: int, deals: dict[str, np.ndarray]) -> None:
        cfg = self.cfg
        count = deals["buckets"].shape[0]
        for r in range(cfg.reuse):
            rng = np.random.default_rng([cfg.seed, chunk_index, r])
            states = rng.integers(0, len(self.pool), count).astype(np.int64)
            prune = self.progress.pairs >= cfg.prune_after
            prune_mask = rng.random(count) < cfg.prune_share
            ip, fp = self.params(prune)
            counters = np.zeros((self.threads, 1), np.int64)
            values = np.zeros((self.threads, 2))
            ws = self.ws
            self.K.train_batch(int(rng.integers(1 << 30)), self.threads, states, deals["buckets"], deals["ranks"],
                               deals["classes"], self.pool.n, self.pool.stacks, self.pool.posts, self.pool.bb,
                               self.pool.prizes, self.order, self.press, self.eq, self.offsets, self.regret,
                               self.strat, self.fixed, ip, prune_mask, fp, ws["I"], ws["SIG"], ws["VAL"], ws["f"],
                               ws["alive"], ws["fin"], counters, values)
            p = self.progress
            p.pairs += count
            p.nodes += int(counters.sum())
            p.traversals += int(values[:, 1].sum())
            p.value_sum += float(values[:, 0].sum())
            self.maybe_discount()

    def maybe_discount(self) -> None:
        cfg = self.cfg
        p = self.progress
        while p.pairs >= (p.intervals + 1) * cfg.interval_pairs:
            p.intervals += 1
            if p.intervals <= cfg.linear_intervals:
                discount(self.regret, self.strat, cfg.variant, p.intervals)

    # ------------------------------------------------------------ checkpoints

    def save(self, run: Path) -> None:
        tmp = run / "ckpt.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        np.save(tmp / "regret.npy", self.regret)
        np.save(tmp / "strat.npy", self.strat)
        meta = {"config_digest": self.cfg.digest(), "progress": asdict(self.progress),
                "regret_sha256": hashlib.sha256(self.regret.tobytes()).hexdigest(),
                "strat_sha256": hashlib.sha256(self.strat.tobytes()).hexdigest()}
        (tmp / "meta.json").write_text(json.dumps(meta, indent=1))
        final = run / "ckpt"
        old = run / "ckpt.old"
        if final.exists():
            if old.exists():
                shutil.rmtree(old)
            final.rename(old)
        tmp.rename(final)
        if old.exists():
            shutil.rmtree(old)

    def load(self, run: Path) -> None:
        ck = run / "ckpt"
        meta = json.loads((ck / "meta.json").read_text())
        if meta["config_digest"] != self.cfg.digest():
            raise ValueError(f"{run}: checkpoint was written by a different config")
        self.regret = np.load(ck / "regret.npy")
        self.strat = np.load(ck / "strat.npy")
        if hashlib.sha256(self.regret.tobytes()).hexdigest() != meta["regret_sha256"]:
            raise ValueError(f"{run}: regret checkpoint is corrupt")
        if hashlib.sha256(self.strat.tobytes()).hexdigest() != meta["strat_sha256"]:
            raise ValueError(f"{run}: strategy checkpoint is corrupt")
        self.progress = Progress(**meta["progress"])

    def snapshot(self, run: Path, label: str) -> Path:
        """A milestone copy of the strategy sums (compressed), kept for later evaluation."""
        out = run / "milestones" / f"{label}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, strat=self.strat, progress=np.array(json.dumps(asdict(self.progress))))
        return out


def open_run(run: Path, cfg: TrainConfig | None) -> TrainConfig:
    path = run / "config.json"
    if path.exists():
        saved = TrainConfig(**json.loads(path.read_text()))
        if cfg is not None and cfg.digest() != saved.digest():
            raise SystemExit(f"{run} exists with a different config; resume without changing it")
        return saved
    if cfg is None:
        raise SystemExit(f"{run} has no config.json")
    run.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(cfg), indent=1, sort_keys=True))
    return cfg


def train(run: Path, cfg: TrainConfig, *, threads: int, producers: int, max_pairs: int, hours: float,
          checkpoint_minutes: float, milestones: tuple[int, ...] = (), log=print) -> Trainer:
    from strategy.blueprint.deals import deal_stream

    pool, _ = StatePool.load(Path(cfg.states))
    if pool.digest() != cfg.states_digest:
        raise SystemExit("state pool changed since the run was created")
    tr = Trainer(cfg, pool, threads=threads)
    if (run / "ckpt" / "meta.json").exists():
        tr.load(run)
        log(f"resumed at {tr.progress.pairs:,} pairs, chunk {tr.progress.chunks}")
    t0 = time.time()
    base_wall = tr.progress.wall_seconds
    last_ckpt = time.time()
    done = {int(p.stem.split("-")[1]) for p in (run / "milestones").glob("pairs-*.npz")} if (run / "milestones").exists() else set()
    stream = deal_stream(cfg.ruleset_id, tr.progress.chunks, cfg.chunk, jobs=producers, base_seed=cfg.seed)
    with (run / "progress.jsonl").open("a") as plog:
        try:
            for ci, deals in stream:
                if tr.progress.pairs >= max_pairs or time.time() - t0 >= hours * 3600 or (run / STOP).exists():
                    break
                tb = time.time()
                tr.batch(ci, deals)
                tr.progress.chunks = ci + 1
                tr.progress.wall_seconds = base_wall + time.time() - t0
                p = tr.progress
                rec = {"chunk": ci, "pairs": p.pairs, "nodes": p.nodes, "intervals": p.intervals,
                       "wall": round(p.wall_seconds, 1), "batch_s": round(time.time() - tb, 2),
                       "mean_value": p.value_sum / max(1, p.traversals)}
                plog.write(json.dumps(rec) + "\n")
                plog.flush()
                for m in milestones:
                    if p.pairs >= m and m not in done:
                        tr.snapshot(run, f"pairs-{m}")
                        done.add(m)
                if time.time() - last_ckpt >= checkpoint_minutes * 60:
                    tr.save(run)
                    last_ckpt = time.time()
                    log(f"checkpoint {p.pairs:,} pairs, {p.nodes:,} nodes, {p.wall_seconds / 60:.1f} min")
        finally:
            stream.close()
    tr.progress.wall_seconds = base_wall + time.time() - t0
    tr.save(run)
    return tr


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--states", type=Path)
    ap.add_argument("--ruleset", default="practice_nlhe")
    ap.add_argument("--variant", choices=VARIANTS, default="linear")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--reuse", type=int, default=2)
    ap.add_argument("--interval-pairs", type=int, default=1_000_000)
    ap.add_argument("--linear-intervals", type=int, default=40)
    ap.add_argument("--prune-after", type=int, default=20_000_000)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--producers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--max-pairs", type=int, default=10 ** 12)
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--checkpoint-minutes", type=float, default=15.0)
    ap.add_argument("--milestone", type=int, action="append", default=[])
    a = ap.parse_args(argv)
    cfg = None
    if not (a.run / "config.json").exists():
        if a.states is None:
            raise SystemExit("--states is required for a new run")
        pool, _ = StatePool.load(a.states)
        cfg = TrainConfig(ruleset_id=a.ruleset, states=str(a.states), states_digest=pool.digest(), variant=a.variant,
                          seed=a.seed, chunk=a.chunk, reuse=a.reuse, interval_pairs=a.interval_pairs,
                          linear_intervals=a.linear_intervals, prune_after=a.prune_after)
    cfg = open_run(a.run, cfg)
    tr = train(a.run, cfg, threads=a.threads, producers=a.producers, max_pairs=a.max_pairs, hours=a.hours,
               checkpoint_minutes=a.checkpoint_minutes, milestones=tuple(a.milestone))
    p = tr.progress
    print(f"done: {p.pairs:,} pairs, {p.traversals:,} traversals, {p.nodes:,} nodes, {p.wall_seconds / 60:.1f} min")


if __name__ == "__main__":
    main()
