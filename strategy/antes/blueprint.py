"""Ante-aware blueprint training (offline): ante start states, a continuation run, a preflop overlay pack.

* :func:`ante_state_row` is :func:`strategy.blueprint.states.state_row` with the 24-wide
  ``posts`` row of :mod:`strategy.blueprint.kernels` (blinds posted after a classic ante,
  live antes, dead big-blind antes).
* :func:`ante_pool` harvests every hand with an ante from rollouts of a named structure
  (:mod:`strategy.antes.structures`) on training-partition lineups; a share
  ``jitter`` of them gets an ante drawn from the posting mode's ratio range
  (:data:`strategy.antes.tables.RATIOS`) instead of the schedule's, so training covers
  a range of ante-to-big-blind ratios. Pools of both posting modes are concatenated.
* :func:`fork` writes a :mod:`strategy.blueprint.train` run on an ante pool, under a
  fresh seed, warm-started at the regrets of a merged sharded checkpoint
  (:func:`strategy.blueprint.shards.get_arrays`). Progress and strategy sums start at
  zero: the run's linear-CFR discount intervals fade the warm-start regrets, and its
  strategy sums are its average strategy in ante hands alone (the checkpoint's float32
  sums, up to ~4e7 a slot, would round such increments away).
* :func:`overlay` packs those sums for the preflop keys (:mod:`strategy.blueprint.policy`
  levels and thresholds). :class:`~strategy.antes.runtime.AntePolicy` reads an overlay
  before its pack in hands with an ante; the trainer's preflop contexts leave the antes
  out of the pot, as that policy does, so both packs share one key space.

    python -m strategy.antes.blueprint pool --lineups 40 --out artifacts/antes/states-antes.npz
    python -m strategy.antes.blueprint fork --base DIR/final --states POOL --run RUN
    python -m strategy.antes.blueprint overlay --run RUN --out PACK
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np

from strategy.antes.pushfold import POSTINGS, ante_seated_snapshot
from strategy.antes.structures import tournament_config
from strategy.blueprint import abstraction as A
from strategy.blueprint.states import MAX_PLAYERS, StatePool, pool_from_rows, state_row


def ante_state_row(ring: Sequence[int], start_stacks: dict[int, int], button: int, small_blind_seat: int | None,
                   big_blind: int, heads_up: bool, blinds: tuple[int, int], payouts: Sequence[float],
                   hand_number: int, hand_cap: int, ante: int, posting: str) -> dict | None:
    row = state_row(ring, start_stacks, button, small_blind_seat, big_blind, heads_up, blinds, payouts, hand_number,
                    hand_cap)
    if row is None:
        return None
    snap = ante_seated_snapshot(ring=ring, start_stacks=start_stacks, button=button, small_blind_seat=small_blind_seat,
                                big_blind=big_blind, heads_up=heads_up, blinds=blinds, payouts=payouts,
                                hand_number=hand_number, hand_cap=hand_cap, ante=ante, posting=posting).snapshot
    pad = [0] * (MAX_PLAYERS - snap.n)
    live = list(snap.antes) if posting == "classic" else [0] * snap.n
    dead = list(snap.antes) if posting == "big_blind" else [0] * snap.n
    row["posts"] = list(snap.posts) + pad + live + pad + dead + pad
    return row


def ante_pool(posting: str, *, lineups: int, seed: int, jobs: int, jitter: float = 0.5,
              fixed: Sequence[str] = ()) -> StatePool:
    from strategy.antes.tables import RATIOS
    from strategy.value.rollouts import run_lineups, training_lineups

    config = tournament_config(posting)
    teams = training_lineups(config.ruleset_id, config.initial_player_count, lineups, seed, fixed=fixed)
    rollouts = run_lineups(config, teams, blocks=1, base_seed=f"blueprint-ante-states-{posting}-{seed}", jobs=jobs)
    rng = np.random.default_rng([seed, POSTINGS.index(posting)])
    lo, hi = RATIOS[posting]
    rows = []
    for ro in rollouts:
        for h in ro.hands:
            ante = config.ante_for_hand(h.hand_number)
            if not ante:
                continue
            if rng.random() < jitter:
                ante = max(1, int(round(float(rng.uniform(lo, hi)) * h.blinds[1])))
            r = ante_state_row(ro.ring, h.start_stacks, h.button, h.small_blind_seat, h.big_blind, h.heads_up,
                               h.blinds, ro.payouts, h.hand_number, ro.hand_cap, ante, posting)
            if r is not None:
                rows.append(r)
    return pool_from_rows(rows)


def concat(pools: Sequence[StatePool]) -> StatePool:
    return StatePool(**{k: np.concatenate([getattr(p, k) for p in pools]) for k in pools[0].arrays()})


# ----------------------------------------------------------------------------- runs


def fork(base: Path, states: Path, run: Path, *, seed: int) -> dict:
    """A training run on the ante pool ``states``, warm-started at ``base``'s merged regrets."""
    from strategy.blueprint.shards import get_arrays
    from strategy.blueprint.train import TrainConfig

    from strategy.blueprint.train import Progress

    if (run / "config.json").exists():
        raise SystemExit(f"{run} exists")
    arrays, meta = get_arrays(base)
    pool, _ = StatePool.load(states)
    cfg = dict(meta["config"], states=str(states), states_digest=pool.digest(), seed=seed)
    tc = TrainConfig(**cfg)
    progress = asdict(Progress())
    arrays = {"regret": arrays["regret"], "strat": np.zeros_like(arrays["strat"])}
    (run / "ckpt").mkdir(parents=True)
    for n in ("regret", "strat"):
        np.save(run / "ckpt" / f"{n}.npy", arrays[n])
    (run / "ckpt" / "meta.json").write_text(json.dumps({
        "config_digest": tc.digest(), "progress": progress,
        "regret_sha256": hashlib.sha256(arrays["regret"].tobytes()).hexdigest(),
        "strat_sha256": hashlib.sha256(arrays["strat"].tobytes()).hexdigest()}, indent=1))
    (run / "config.json").write_text(json.dumps(asdict(tc), indent=1, sort_keys=True))
    info = {"base": str(base), "base_progress": meta["progress"], "base_config_digest": meta.get("config_digest"),
            "states": str(states), "states_digest": pool.digest(), "seed": seed}
    (run / "fork.json").write_text(json.dumps(info, indent=1))
    return info


def gains(run: Path) -> tuple[np.ndarray, dict]:
    """The run's strategy sums (float64; they start at zero at the fork) and its progress."""
    strat = np.load(run / "ckpt" / "strat.npy")
    meta = json.loads((run / "ckpt" / "meta.json").read_text())
    return strat.astype(np.float64), meta["progress"]


def overlay_levels(delta: np.ndarray, streets: int = 1, min_mass=None) -> dict[str, np.ndarray]:
    """Back-off levels of ``delta`` restricted to the first ``streets`` streets' keys."""
    from strategy.blueprint.policy import DEFAULT_MIN_MASS, build_levels

    d = np.zeros_like(delta)
    end = A.OFFSETS[streets] * A.NUM_ACTIONS
    d[:end] = delta[:end]
    return build_levels(d, min_mass or DEFAULT_MIN_MASS)


def overlay(run: Path, out: Path, *, streets: int = 1) -> dict:
    from strategy.blueprint.policy import DEFAULT_MIN_MASS, write_levels

    delta, progress = gains(run)
    fork_info = json.loads((run / "fork.json").read_text())
    levels = overlay_levels(delta, streets)
    cfg = json.loads((run / "config.json").read_text())
    return write_levels(out, levels, {
        "ruleset_id": cfg["ruleset_id"], "run": run.name, "config": cfg, "progress": progress,
        "overlay": {"base": fork_info["base"], "base_pairs": fork_info["base_progress"]["pairs"],
                    "pairs": progress["pairs"], "streets": streets,
                    "states_digest": fork_info["states_digest"]}}, DEFAULT_MIN_MASS)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pool")
    p.add_argument("--postings", nargs="+", default=list(POSTINGS))
    p.add_argument("--lineups", type=int, default=40)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--jitter", type=float, default=0.5)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--out", type=Path, required=True)
    f = sub.add_parser("fork")
    f.add_argument("--base", type=Path, required=True)
    f.add_argument("--states", type=Path, required=True)
    f.add_argument("--run", type=Path, required=True)
    f.add_argument("--seed", type=int, default=7001)
    o = sub.add_parser("overlay")
    o.add_argument("--run", type=Path, required=True)
    o.add_argument("--out", type=Path, required=True)
    o.add_argument("--streets", type=int, default=1)
    a = ap.parse_args(argv)
    if a.cmd == "pool":
        t = time.time()
        pool = concat([ante_pool(s, lineups=a.lineups, seed=a.seed, jobs=a.jobs, jitter=a.jitter) for s in a.postings])
        pool.save(a.out, {"ruleset_id": "practice_nlhe", "postings": a.postings, "lineups": a.lineups, "seed": a.seed,
                          "jitter": a.jitter, "partition": "train", "wall_seconds": round(time.time() - t, 1)})
        print(f"{len(pool)} ante states -> {a.out}; live-player counts 2..8: "
              f"{np.bincount(pool.n, minlength=9)[2:].tolist()}")
    elif a.cmd == "fork":
        print(json.dumps(fork(a.base, a.states, a.run, seed=a.seed), indent=1))
    else:
        prov = overlay(a.run, a.out, streets=a.streets)
        print(f"{a.out}: rows per level {prov['rows']}, {a.out.stat().st_size / 1e6:.2f} MB on disk")


if __name__ == "__main__":
    main()
