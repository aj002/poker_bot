"""Point-aware push/fold tables solved with the antes in the pot, one per ante posting mode.

The ante-free generator (:mod:`strategy.tables.endgame`) with three changes:

* states come from rollouts of the named ante structure (:mod:`strategy.antes.structures`)
  on training-partition lineups, plus synthetic states whose ante is drawn as a ratio of
  the big blind (``RATIOS``), so the table is parameterised by the ante-to-big-blind
  ratio and not by one schedule; hands of a level without an ante are left to the
  ante-free table;
* every state is solved as an :class:`~strategy.antes.subgame.AnteSubgame` (classic antes
  as contribution levels, a big-blind ante as dead money in the main pot);
* keys are :func:`~strategy.antes.pushfold.ante_keys_for`: the ante-free ladder plus the
  bucket of the ante money over the big blind at every level.

Arrays and provenance are those of the ante-free tables, so
:class:`~strategy.endgame.lookup.PushFoldTable` and the store read them unchanged.

    python -m strategy.antes.tables --posting classic --jobs 8
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from strategy.antes.pushfold import POSTINGS, ante_seated_snapshot
from strategy.antes.structures import BLIND_LEVELS, tournament_config
from strategy.endgame.keys import LEVELS, MAX_DEPTH_BB
from strategy.endgame.table_state import SeatedSnapshot
from strategy.tables import endgame as E
from strategy.tables import store

GENERATOR = "strategy.antes.tables"
INPUTS = [*E.INPUTS, "antes/pushfold.py", "antes/subgame.py", "antes/structures.py", "antes/tables.py"]
DEFAULTS = {"rollout_lineups": 8, "synthetic": 3000, "max_states": 6000, "seed": 1}
RATIOS = {"classic": (0.05, 0.2), "big_blind": (0.5, 1.25)}
"""Range of the synthetic states' ante over the big blind: per seat (classic) or the big blind's."""


def table_name(posting: str, ruleset_id: str = "practice_nlhe") -> str:
    return f"endgame-points-{posting}-{ruleset_id}"


def compact_name(posting: str, ruleset_id: str = "practice_nlhe") -> str:
    return f"antes-pushfold-{posting}-{ruleset_id}"


def compact(arrays: dict) -> dict:
    """The rows a lookup can use (count at least ``MIN_COUNT``), probabilities in 4 bits.

    ``{kind}_nib`` holds two hand classes a byte, the even class in the low nibble; a
    nibble ``q`` is the probability ``q / 15`` of the aggressive action."""
    from strategy.endgame.lookup import MIN_COUNT

    out = {"class_keys": arrays["class_keys"]}
    for k in ("open", "call", "overcall"):
        keep = arrays[f"{k}_count"] >= MIN_COUNT
        q = np.rint(arrays[f"{k}_probs"][keep].astype(np.float64) * (15 / 255)).astype(np.uint8)
        if q.shape[1] % 2:
            q = np.concatenate([q, np.zeros((q.shape[0], 1), np.uint8)], axis=1)
        out[f"{k}_keys"] = arrays[f"{k}_keys"][keep]
        out[f"{k}_nib"] = q[:, 0::2] | (q[:, 1::2] << 4)
        out[f"{k}_count"] = arrays[f"{k}_count"][keep]
    return out


FULL_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "antes" / "tables"
"""Where the full solves (every row, 8-bit probabilities, diagnostics) are kept for analysis."""


def rollout_states(posting: str, *, lineups: int, seed: int, jobs: int) -> list[SeatedSnapshot]:
    """In-scope ante hands of ``lineups`` training lineups under the named structure, every rotation."""
    from strategy.value.rollouts import run_lineups, training_lineups

    config = tournament_config(posting)
    teams = training_lineups(config.ruleset_id, config.initial_player_count, lineups, seed)
    rollouts = run_lineups(config, teams, blocks=1, base_seed=f"ante-endgame-states-{posting}-{seed}", jobs=jobs)
    out = []
    for ro in rollouts:
        for h in ro.hands:
            ante = config.ante_for_hand(h.hand_number)
            if not ante:
                continue
            s = ante_seated_snapshot(ring=ro.ring, start_stacks=h.start_stacks, button=h.button,
                                     small_blind_seat=h.small_blind_seat, big_blind=h.big_blind,
                                     heads_up=h.heads_up, blinds=h.blinds, payouts=ro.payouts,
                                     hand_number=h.hand_number, hand_cap=ro.hand_cap, ante=ante, posting=posting)
            if E.in_scope(s.snapshot):
                out.append(s)
    return out


def synthetic_state(rng: np.random.Generator, config, posting: str) -> SeatedSnapshot | None:
    """:func:`strategy.tables.endgame.synthetic_state` with an ante drawn from ``RATIOS``."""
    seats = config.initial_player_count
    n = int(rng.choice(np.arange(2, seats + 1), p=E._n_weights(seats)))
    u = rng.random()
    if u < 0.1:
        hand = config.hand_cap
    elif u < 0.2:
        hand = int(config.hand_cap - rng.integers(1, n + 1))
    else:
        hand = int(rng.integers(config.hands_per_level + 1, config.hand_cap + 1))
    sb, bb = E._blinds_at(config, hand)
    lo, hi = RATIOS[posting]
    ante = max(1, int(round(float(rng.uniform(lo, hi)) * bb)))
    total = seats * config.starting_stack
    shares = rng.dirichlet(np.full(n, float(rng.choice([0.7, 1.5, 4.0]))))
    stacks = np.maximum(1, np.floor(shares * total).astype(np.int64))
    stacks[int(np.argmax(stacks))] += total - int(stacks.sum())
    if stacks.min() <= 0:
        return None
    live = sorted(rng.choice(seats, n, replace=False).tolist())
    start = {s: int(c) for s, c in zip(live, stacks)}
    i = int(rng.integers(n))
    big = live[i]
    if n == 2:
        button = small = live[1 - i]
    else:
        small, button = live[i - 1], live[i - 2]
    s = ante_seated_snapshot(ring=tuple(range(seats)), start_stacks=start, button=button, small_blind_seat=small,
                             big_blind=big, heads_up=n == 2, blinds=(sb, bb), payouts=config.resolved_payouts,
                             hand_number=hand, hand_cap=config.hand_cap, ante=ante, posting=posting)
    return s if E.in_scope(s.snapshot) else None


def sample_states(posting: str, *, rollout_lineups: int, synthetic: int, max_states: int, seed: int,
                  jobs: int) -> list[SeatedSnapshot]:
    config = tournament_config(posting)
    states = rollout_states(posting, lineups=rollout_lineups, seed=seed, jobs=jobs) if rollout_lineups else []
    rng = np.random.default_rng([seed, POSTINGS.index(posting)])
    if len(states) > max_states - synthetic:
        keep = rng.choice(len(states), max_states - synthetic, replace=False)
        states = [states[i] for i in sorted(keep)]
    made = 0
    while made < synthetic:
        s = synthetic_state(rng, config, posting)
        if s is not None:
            states.append(s)
            made += 1
    return states


def _solve_chunk(args):
    from strategy.antes.subgame import AnteSubgameTree, ante_harvest, ante_subgame_for, solve_ante_snapshot

    items, levels = args
    acc: dict[tuple[str, int], list] = {}
    diags = []
    for snap, odd in items:
        try:
            AnteSubgameTree(ante_subgame_for(snap, odd_order=odd))
        except ValueError:
            continue
        solved = solve_ante_snapshot(snap, E._MODEL, utility="points", odd_order=odd)
        diags.append((snap.n, float(np.max(solved.gains)) / snap.payouts[0], solved.solver.iterations))
        for rec in ante_harvest(solved, max_depth_bb=MAX_DEPTH_BB):
            for level, key in enumerate(rec.keys):
                if level not in levels:
                    continue
                slot = acc.get((rec.kind, key))
                if slot is None:
                    slot = acc[(rec.kind, key)] = [np.zeros(len(rec.row)), 0.0, 0]
                slot[0] += rec.weight * rec.row
                slot[1] += rec.weight
                slot[2] += int(rec.weight >= E.COUNT_FLOOR)
    return acc, diags


def solve_states(ruleset_id: str, states, *, jobs: int):
    from concurrent.futures import ProcessPoolExecutor

    levels = frozenset(range(LEVELS))
    items = [(s.snapshot, s.odd_order) for s in states]
    items.sort(key=lambda it: -it[0].n)
    tasks = [(list(c), levels) for c in E._chunks(items, 8)]
    acc: dict = {}
    diags: list = []
    if jobs > 1:
        with ProcessPoolExecutor(jobs, initializer=E._init, initargs=(ruleset_id, None, 1)) as ex:
            for part, d in ex.map(_solve_chunk, tasks):
                E._merge(acc, part)
                diags.extend(d)
    else:
        E._init(ruleset_id, None, 1)
        for t in tasks:
            part, d = _solve_chunk(t)
            E._merge(acc, part)
            diags.extend(d)
    return acc, diags


def build(posting: str, *, rollout_lineups: int, synthetic: int, max_states: int, seed: int, jobs: int):
    from strategy.endgame.classes import hand_classes
    from strategy.endgame.showdown import load_showdown

    if posting not in POSTINGS:
        raise ValueError(f"posting is one of {POSTINGS}")
    ruleset_id = "practice_nlhe"
    hc = hand_classes(ruleset_id)
    model = load_showdown(ruleset_id)
    started = time.perf_counter()
    states = sample_states(posting, rollout_lineups=rollout_lineups, synthetic=synthetic, max_states=max_states,
                           seed=seed, jobs=jobs)
    sampled = time.perf_counter() - started
    acc, diags = solve_states(ruleset_id, states, jobs=jobs)
    arrays = E.to_arrays(acc, np.array(hc.keys, np.int64), diags)
    showdown_prov = store.load_arrays(f"showdown-{ruleset_id}")[0]
    extra = {"ruleset_id": ruleset_id, "variant_hash": hc.variant_hash, "mode": "points", "utility": "points",
             "value": "icm", "ante_posting": posting, "ante_ratios": list(RATIOS[posting]),
             "blind_levels": [list(b) for b in BLIND_LEVELS], "showdown_sha256": showdown_prov["data_sha256"],
             "states": len(states), "solved": len(diags), "sample_seconds": round(sampled, 1),
             "payouts": list(states[0].snapshot.payouts), "max_depth_bb": MAX_DEPTH_BB, "levels": LEVELS,
             "model_classes": model.h}
    return arrays, extra


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--posting", choices=POSTINGS, required=True)
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), type=int, default=v)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--full-dir", type=Path, default=FULL_DIR, help="directory of the full solve")
    ap.add_argument("--out", help="directory of the compact table (default: strategy/tables/data)")
    ap.add_argument("--compact-only", action="store_true", help="recompact an existing full solve")
    a = ap.parse_args(argv)
    params = {k: getattr(a, k) for k in DEFAULTS}
    name = table_name(a.posting)
    if not a.compact_only:
        started = time.perf_counter()
        arrays, extra = build(a.posting, jobs=a.jobs, **params)
        seconds = time.perf_counter() - started
        path = store.write_arrays(name, arrays, generator=GENERATOR, inputs=INPUTS,
                                  params={"ruleset_id": "practice_nlhe", "posting": a.posting, **params},
                                  seconds=seconds, extra=extra, directory=a.full_dir)
        g = arrays["diag_gain"]
        rows = sum(len(arrays[f"{k}_keys"]) for k in ("open", "call", "overcall"))
        print(f"wrote {path} ({path.stat().st_size} bytes, {extra['states']} states, {extra['solved']} solved, "
              f"{rows} rows, {seconds:.0f}s); deviation gain median {np.median(g):.2e} "
              f"p99 {np.quantile(g, 0.99):.2e} max {g.max():.2e}")
    store.load_arrays.cache_clear()
    full_prov, full = store.load_arrays(name, str(a.full_dir))
    small = compact(full)
    keep = ("ruleset_id", "variant_hash", "mode", "utility", "value", "ante_posting", "ante_ratios", "blind_levels",
            "showdown_sha256", "states", "solved", "payouts", "max_depth_bb", "levels", "model_classes")
    path = store.write_arrays(compact_name(a.posting), small, generator=GENERATOR, inputs=INPUTS,
                              params=full_prov["params"], seconds=full_prov["wall_seconds"],
                              extra={**{k: full_prov[k] for k in keep}, "full_table": name,
                                     "full_sha256": full_prov["data_sha256"], "probability_bits": 4},
                              directory=Path(a.out) if a.out else None)
    rows = sum(len(small[f"{k}_keys"]) for k in ("open", "call", "overcall"))
    print(f"wrote {path} ({path.stat().st_size} bytes, {rows} rows)")


if __name__ == "__main__":
    main()
