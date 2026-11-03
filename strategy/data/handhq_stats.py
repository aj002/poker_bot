"""Per-stake and per-player action frequencies of real six-max players (HandHQ, PokerStars 2009).

Every eligible hand (six players at a six-max table, plain blinds; :func:`strategy.data.handhq.convert`)
is replayed in the arena engine, and each seat whose starting stack is within
``DEPTH_RANGE`` big blinds (the arena's depths: 100 BB at the start, shallower as the blinds
rise) is counted three ways:

* the statistics of :class:`strategy.population.measure.Tally` that read no hole cards
  (:data:`STATS`), from the seat's own observation, with ``vpip``, ``pfr``, aggression and
  fold-to-reraise from :class:`arena.table.player_stats.StatsTracker` on the spectator
  events, exactly as a census of the arena families counts them;
* the belief layer's public observations (:func:`strategy.beliefs.posterior.from_record`):
  per stake, counts by opportunity, context cell and category; per player, by opportunity,
  depth bucket and category;
* the starting depth of the seat in big blinds.

Hole cards are hidden in HandHQ unless shown, so the card-based statistics (junk entry,
strong checks, draws, bluffs, size/strength correlation) are not computed.

    python -m strategy.data.handhq_stats --root ~/.cache/arena/phh-dataset/handhq --jobs 7 \
        --out artifacts/handhq/stats
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from strategy.data.handhq import (CITATION, COMMIT, DATASET, LICENCE, REPO_URL, SKIP_CODES, ImportRejected, convert,
                                  load_phhs)
from strategy.data.phh_import import Provenance, _replay

DEPTH_RANGE = (20.0, 150.0)
"""Starting stacks (in big blinds) counted; deeper or shorter seats are left out."""
DEPTH_EDGES = (0, 12, 20, 30, 50, 80, 100, 120, 150, 200, 300, 10**9)
COUNT_KEYS = ("hands", "vpip", "pfr", "bets_raises", "calls", "faced", "folded")
STATS = ("limp", "iso", "three_bet", "squeeze", "cold_call", "cbet", "cbet_dry_hu", "cbet_wet", "xr_share",
         "allin_share", "min_raise_share", "big_call", "short_allin")
"""Card-free :class:`~strategy.population.measure.Tally` statistics, as ``(hit, sample)`` pairs."""
COLUMNS = (*COUNT_KEYS, *(f"{s}.{x}" for s in STATS for x in ("hit", "n")))
N_OPP, N_DEPTH, N_CAT, CELLS = 5, 3, 3, 192


def stake_of(path: Path) -> str:
    """``"200NL"`` from ``PS-2009-07-01_2009-07-23_200NLH_OBFU/2/...`` (``"other"`` outside that layout)."""
    parts = path.parent.parent.name.split("_")
    return parts[-2].removesuffix("H") if len(parts) >= 2 and parts[-2].endswith("NLH") else "other"


class _Player:
    __slots__ = ("counts", "belief")

    def __init__(self) -> None:
        self.counts = np.zeros(len(COLUMNS), dtype=np.int64)
        self.belief = np.zeros((N_OPP, N_DEPTH, N_CAT), dtype=np.int64)


def _file(path: Path) -> dict[str, Any]:
    from arena.table.player_stats import StatsTracker
    from arena.table.projections import Viewer, project_events
    from strategy.exploit.collect import public_rows
    from strategy.population.measure import Tally

    stake = stake_of(path)
    report: Counter = Counter()
    rejected: Counter = Counter()
    depth = Counter()
    tally = Tally()
    cells = np.zeros((N_OPP, CELLS, N_CAT), dtype=np.int64)
    players: dict[str, _Player] = {}
    col = {k: i for i, k in enumerate(COLUMNS)}
    try:
        docs = load_phhs(path)
    except ImportRejected as exc:
        return {"path": str(path), "stake": stake, "error": exc.detail}
    for key, doc in docs.items():
        report["hands"] += 1
        rel = f"{path.parent.parent.name}/{path.parent.name}/{path.stem}/{key}"
        try:
            conv = convert(doc, rel)
        except ImportRejected as exc:
            report["skipped" if exc.code in SKIP_CODES else "rejected"] += 1
            if exc.code not in SKIP_CODES:
                rejected[exc.code] += 1
            continue
        prov = Provenance(dataset=DATASET, path=f"data/handhq/{rel}", sha256="", url=REPO_URL, commit=COMMIT,
                          licence=LICENCE, citation=CITATION, dataset_doi="10.5281/zenodo.10796885", paper_doi="")
        try:
            hand = _replay(conv.phh, prov, None, mixed_stacks=True)
        except ImportRejected as exc:
            report["rejected"] += 1
            rejected[exc.code] += 1
            continue
        report["replayed"] += 1
        bb = conv.phh.blinds[1]
        in_range: dict[int, str] = {}
        for seat, name in hand.seat_names.items():
            d = conv.phh.starting_stacks[seat] / bb
            depth[next(i for i, e in enumerate(DEPTH_EDGES[1:]) if d < e)] += 1
            if DEPTH_RANGE[0] <= d <= DEPTH_RANGE[1]:
                in_range[seat] = name
        report["seats"] += len(hand.seat_names)
        report["seats_in_range"] += len(in_range)
        if not in_range:
            continue
        per_seat = {s: Tally() for s in in_range}
        for dec in hand.decisions:
            t = per_seat.get(dec.seat)
            if t is not None:
                t.add(dec.observation, dec.action.model_dump(exclude_none=True))
        table = hand.table
        events = project_events(table.log, Viewer.spectator())
        first = next(e.hand_id for e in events if e.type == "HandStarted")
        events = [e for e in events if e.hand_id == first]
        tracker = StatsTracker(table.seat_ring)
        tracker.feed(events)
        for seat, name in in_range.items():
            t = per_seat[seat]
            c = tracker.counts[seat]
            p = players.get(name)
            if p is None:
                p = players[name] = _Player()
            for k in COUNT_KEYS:
                v = getattr(c, k)
                t.counts[k] += v
                p.counts[col[k]] += v
            for s, (hit, n) in t.n.items():
                if s in STATS:
                    p.counts[col[f"{s}.hit"]] += hit
                    p.counts[col[f"{s}.n"]] += n
            tally.merge(t)
        for _, seat, _, opp, cell, cat in public_rows(events, table.seat_ring, {s: 0 for s in in_range}):
            cells[opp, cell, cat] += 1
            players[in_range[seat]].belief[opp, (cell // 4) % 3, cat] += 1
            report["public_rows"] += 1
        report["decisions"] += sum(t.decisions for t in per_seat.values())
    tally.scores = []
    return {"path": str(path), "stake": stake, "report": dict(report), "rejected": dict(rejected),
            "depth": dict(depth), "tally": tally, "cells": cells,
            "players": {n: (p.counts, p.belief) for n, p in players.items()}}


def files_under(root: Path) -> list[Path]:
    return sorted(root.glob("PS-*/*/*.phhs"))


def run(paths: list[Path], *, jobs: int = 1, log=print) -> dict[str, Any]:
    from strategy.population.measure import Tally

    by_stake: dict[str, dict[str, Any]] = {}
    t0 = time.time()
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(jobs) as pool:
        for i, part in enumerate(pool.imap_unordered(_file, paths, chunksize=4)):
            st = by_stake.setdefault(part["stake"], {"files": 0, "report": Counter(), "rejected": Counter(),
                                                     "depth": Counter(), "tally": Tally(),
                                                     "cells": np.zeros((N_OPP, CELLS, N_CAT), dtype=np.int64),
                                                     "players": {}, "errors": []})
            st["files"] += 1
            if "error" in part:
                st["errors"].append(f"{part['path']}: {part['error']}")
                continue
            st["report"].update(part["report"])
            st["rejected"].update(part["rejected"])
            st["depth"].update(part["depth"])
            st["tally"].merge(part["tally"])
            if len(st["tally"].sizes) > 200_000:
                st["tally"].sizes = st["tally"].sizes[::2]
            st["cells"] += part["cells"]
            pl = st["players"]
            for name, (c, b) in part["players"].items():
                cur = pl.get(name)
                if cur is None:
                    pl[name] = (c, b)
                else:
                    cur[0][:] += c
                    cur[1][:] += b
            if (i + 1) % 200 == 0 or i + 1 == len(paths):
                log(f"handhq-stats: {i + 1}/{len(paths)} files, {time.time() - t0:.0f}s")
    return by_stake


def write(by_stake: dict[str, dict[str, Any]], out: Path, *, root: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"dataset": DATASET, "commit": COMMIT, "licence": LICENCE, "root": str(root),
                               "depth_range_bb": DEPTH_RANGE, "depth_edges_bb": DEPTH_EDGES, "columns": COLUMNS,
                               "stakes": {}}
    names, stakes, counts, beliefs = [], [], [], []
    order = sorted(by_stake, key=lambda s: int(s.removesuffix("NL")))
    for si, stake in enumerate(order):
        st = by_stake[stake]
        stats = st["tally"].stats()
        summary["stakes"][stake] = {
            "files": st["files"], "errors": st["errors"], **dict(st["report"]), "rejected_codes": dict(st["rejected"]),
            "players": len(st["players"]),
            "depth_histogram": {f"{DEPTH_EDGES[i]}-{DEPTH_EDGES[i + 1]}": st["depth"].get(i, 0)
                                for i in range(len(DEPTH_EDGES) - 1)},
            "stats": {k: [v, n] for k, (v, n) in sorted(stats.items()) if k in (*STATS, "vpip", "pfr", "aggression",
                                                                                 "fold_to_reraise", "size_cv")},
        }
        for name, (c, b) in st["players"].items():
            names.append(name)
            stakes.append(si)
            counts.append(c)
            beliefs.append(b)
    np.savez_compressed(out / "players.npz", names=np.array(names), stake=np.array(stakes, dtype=np.int8),
                        stakes=np.array(order), columns=np.array(COLUMNS),
                        counts=np.array(counts, dtype=np.int64).reshape(-1, len(COLUMNS)),
                        belief=np.array(beliefs, dtype=np.int64).reshape(-1, N_OPP, N_DEPTH, N_CAT))
    np.savez_compressed(out / "cells.npz", stakes=np.array(order),
                        cells=np.stack([by_stake[s]["cells"] for s in order]))
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, required=True, help="the data/handhq directory of the HandHQ checkout")
    ap.add_argument("--stakes", nargs="*", default=None, help="only these stakes, e.g. 200NL 400NL")
    ap.add_argument("--files-per-stake", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=max(1, (multiprocessing.cpu_count() or 2) - 1))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    paths = files_under(a.root)
    if a.stakes:
        paths = [p for p in paths if stake_of(p) in a.stakes]
    if a.files_per_stake:
        kept: Counter = Counter()
        sel = []
        for p in paths:
            if kept[stake_of(p)] < a.files_per_stake:
                kept[stake_of(p)] += 1
                sel.append(p)
        paths = sel
    summary = write(run(paths, jobs=a.jobs), a.out, root=a.root)
    for stake, s in summary["stakes"].items():
        st = s["stats"]
        print(f"{stake}: {s.get('replayed', 0):,} hands, {s['players']:,} players, vpip {st['vpip'][0]:.3f} "
              f"pfr {st['pfr'][0]:.3f} 3bet {st['three_bet'][0]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
