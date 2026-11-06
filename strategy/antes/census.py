"""Paired census of one hero across ante structures: points, and chips by depth and street.

Every structure plays the same lineups on the same table seeds and seat rotations, so a
block (lineup x seed, every rotation) is paired across structures: the cards dealt to hand
``k`` match until the play diverges. Per hero hand it records the effective stack, the
live-player count, the ante paid, the chip result, where the hero left the hand and its
first preflop decision, so differences can be located by depth band and street.

Offline only (drives :class:`arena.table.service.Table` in-process)::

    python -m strategy.antes.census run --lineups 40 --blocks 1 --lineup-seed 1 --out DIR
    python -m strategy.antes.census report DIR [DIR ...]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from strategy.antes.structures import STRUCTURES, tournament_config

HERO = "strategy/final/bots/final.py"
STREETS = ("preflop", "flop", "turn", "river")
DEPTH_BANDS = (10, 20, 40, 100)
"""Effective stack at hand start in big blinds: <=10, 10-20, 20-40, 40-100, >100."""
BAND_NAMES = ("<=10", "10-20", "20-40", "40-100", ">100")


def band(eff_bb: float) -> int:
    for i, e in enumerate(DEPTH_BANDS):
        if eff_bb <= e:
            return i
    return len(DEPTH_BANDS)


def hero_hand(rec, hero: int, end_stack: int) -> dict | None:
    start = dict(rec.start_stacks)
    if start.get(hero, 0) <= 0:
        return None
    bb = rec.blinds[1]
    others = [c for s, c in start.items() if s != hero and c > 0]
    eff = min(start[hero], max(others)) if others else 0
    ante_paid = 0
    blind_paid = 0
    folded: set[int] = set()
    street = 0
    raises = 0
    voluntary_before = False
    first = None
    jam = False
    last_street = 0
    hero_folded = False
    showdown = False
    for e in rec.events:
        t, d = e.type, e.data
        if t == "AntePosted" and d["seat"] == hero:
            ante_paid += d["amount"]
        elif t == "BlindPosted" and d["seat"] == hero:
            blind_paid += d["amount"]
        elif t == "BoardDealt":
            street = STREETS.index(d["street"])
            if hero not in folded:
                last_street = street
        elif t == "CardsRevealed":
            showdown = True
        elif t == "ActionApplied":
            seat, kind = d["seat"], d["action"]["type"]
            if seat == hero:
                if street == 0 and first is None and not d.get("auto"):
                    if raises == 0:
                        first = {"fold": "fold_unopened", "check": "check", "call": "limp",
                                 "raise_to": "open"}[kind]
                        if voluntary_before and kind == "raise_to":
                            first = "iso"
                        elif voluntary_before and kind == "call":
                            first = "overlimp"
                    else:
                        first = {"fold": "fold_to_raise", "call": "call_raise", "raise_to": "reraise",
                                 "check": "check"}[kind]
                    jam = kind == "raise_to" and d["added"] + blind_paid + ante_paid >= start[hero]
                if kind == "fold":
                    hero_folded = True
                    last_street = street
            else:
                if kind == "fold":
                    folded.add(seat)
                elif street == 0 and kind in ("call", "raise_to") and first is None:
                    voluntary_before = True
            if kind == "raise_to" and street == 0:
                raises += 1
    outcome = "fold" if hero_folded else "showdown" if showdown and hero not in folded else "uncontested"
    return {"h": rec.hand_number, "lvl": rec.level, "n": sum(1 for c in start.values() if c > 0),
            "eff": round(eff / bb, 2), "bb": bb, "s": start[hero], "ante": ante_paid, "blind": blind_paid,
            "d": end_stack - start[hero], "st": last_street, "out": outcome, "pf": first or "none",
            "jam": bool(jam)}


_CFG = None
_HERO = HERO


def _init(config_json: str, hero: str) -> None:
    global _CFG, _HERO
    from arena.domain.models import TournamentConfig

    _CFG = TournamentConfig.model_validate_json(config_json)
    _HERO = hero


def _play(args):
    from arena.agents.worker import play_table
    from arena.benchmark.runner import Entrant, TableRunner, table_info
    from arena.table.service import seeded_table
    from strategy.eval.policies import resolve_policy

    a, lineup = args
    config = _CFG
    entrants = []
    for i, spec in enumerate(lineup):
        p = resolve_policy(spec)
        entrants.append(Entrant(f"e{i}:{p.name}", p.path / "bot.py" if p.path.is_dir() else p.path, p.builtin))
    runner = TableRunner(config, entrants, isolation="in_process")
    agents = {}
    for seat, entrant in a.seats.items():
        agent = runner._agent(seat, entrant, table_info(config, a, seat, runner.budget_ns))
        agent.start()
        agents[seat] = agent
    table = seeded_table(config, a.seed, table_id=a.table_id)
    try:
        play_table(table, agents)
    finally:
        for agent in agents.values():
            agent.close()
    hero = next(s for s, e in a.seats.items() if lineup[e] == _HERO)
    hands = []
    recs = list(table.hands)
    for i, rec in enumerate(recs):
        end = recs[i + 1].start_stacks.get(hero, 0) if i + 1 < len(recs) else table.stacks.get(hero, 0)
        row = hero_hand(rec, hero, end)
        if row is not None:
            hands.append(row)
    pts = table.points()
    return {"seed": a.seed, "rotation": a.rotation, "hero_seat": hero, "points": float(pts[hero]),
            "hands_played": len(recs), "hands": hands}


def lineups_for(n: int, lineups: int, lineup_seed: int, hero: str) -> list[list[str]]:
    """Train-partition lineups with ``hero`` in the fixed slot; the opponents do not depend on ``hero``."""
    from strategy.value.rollouts import training_lineups

    teams = training_lineups("practice_nlhe", n, lineups, lineup_seed, fixed=[HERO])
    return [[hero if p == HERO else p for p in t] for t in teams]


def run(structure: str, *, lineups: int, blocks: int, lineup_seed: int, base_seed: str, jobs: int, out: Path,
        hero: str = HERO) -> Path:
    from arena.benchmark.runner import schedule

    config = tournament_config(structure)
    teams = lineups_for(config.initial_player_count, lineups, lineup_seed, hero)
    tasks = []
    for li, lineup in enumerate(teams):
        for a in schedule(config, len(lineup), blocks, f"{base_seed}|lineup{li}"):
            tasks.append(((a, lineup), li))
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{structure}.jsonl"
    with ProcessPoolExecutor(jobs, initializer=_init, initargs=(config.model_dump_json(), hero)) as ex:
        results = ex.map(_play, [t for t, _ in tasks], chunksize=2)
        with path.open("w") as f:
            for (t, li), r in zip(tasks, results):
                a = t[0]
                r["lineup"] = li
                r["block"] = a.seed.rsplit("|", 1)[0] if "|" in a.seed else a.seed
                f.write(json.dumps(r, separators=(",", ":")) + "\n")
    meta = {"structure": structure, "config": config.model_dump(mode="json"), "lineups": lineups, "blocks": blocks,
            "lineup_seed": lineup_seed, "base_seed": base_seed, "hero": hero, "partition": "train"}
    (out / f"{structure}.meta.json").write_text(json.dumps(meta, indent=1))
    return path


# --------------------------------------------------------------------------- report

def _load(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x]


def _block_key(r: dict) -> tuple:
    return (r["lineup"], r["block"])


def _t_ci(xs: list[float]) -> tuple[float, float, float]:
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, float("nan"), float("nan")
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    h = 1.96 * sd / math.sqrt(n)
    return m, m - h, m + h


def per_block(rows: list[dict], fn) -> dict[tuple, float]:
    """Mean over a block's tables of ``fn(table)``."""
    acc: dict[tuple, list] = defaultdict(list)
    for r in rows:
        acc[_block_key(r)].append(fn(r))
    out = {}
    for k, v in acc.items():
        if isinstance(v[0], dict):
            m: dict = defaultdict(float)
            for x in v:
                for c, y in x.items():
                    m[c] += y / len(v)
            out[k] = dict(m)
        else:
            out[k] = sum(v) / len(v)
    return out


def chips_by(r: dict, key) -> dict:
    """Big blinds won per tournament grouped by ``key(hand)``."""
    out: dict = defaultdict(float)
    for h in r["hands"]:
        out[key(h)] += h["d"] / h["bb"]
    return out


def report(dirs: list[Path]) -> dict:
    data = {}
    for d in dirs:
        for s in STRUCTURES:
            p = d / f"{s}.jsonl"
            if p.exists():
                data.setdefault(s, []).extend(_load(p) if len(dirs) == 1 else
                                              [dict(r, lineup=f"{d.name}:{r['lineup']}") for r in _load(p)])
    out: dict = {"structures": {}, "paired": {}}
    for s, rows in data.items():
        pts = [r["points"] for r in rows]
        hands = [h for r in rows for h in r["hands"]]
        pf = defaultdict(int)
        first_in = opens = 0
        for h in hands:
            pf[h["pf"]] += 1
            if h["pf"] in ("open", "fold_unopened", "limp"):
                first_in += 1
                opens += h["pf"] == "open"
        out["structures"][s] = {
            "tables": len(rows), "points_mean": sum(pts) / len(pts),
            "hero_hands_per_table": len(hands) / len(rows),
            "hands_per_table": sum(r["hands_played"] for r in rows) / len(rows),
            "hero_alive_at_end": sum(1 for r in rows if r["hands"] and r["hands"][-1]["s"] + r["hands"][-1]["d"] > 0)
            / len(rows),
            "first_in_open_rate": opens / first_in if first_in else 0.0,
            "preflop_first_action": {k: v / len(hands) for k, v in sorted(pf.items())},
            "bb_per_100_by_band": _rate_by(hands, lambda h: BAND_NAMES[band(h["eff"])]),
            "bb_per_100_by_street_outcome": _rate_by(hands, lambda h: f"{STREETS[h['st']]}:{h['out']}"),
            "bb_per_100_by_preflop": _rate_by(hands, lambda h: h["pf"]),
        }
    if "none" in data:
        for s, rows in data.items():
            if s != "none":
                out["paired"][s] = paired(data["none"], rows)
    return out


def paired(base: list[dict], rows: list[dict], names: tuple[str, str] = ("none", "ante")) -> dict:
    """Per-block differences ``rows - base`` over the blocks both cover."""
    b_pts = per_block(base, lambda r: r["points"])
    s_pts = per_block(rows, lambda r: r["points"])
    keys = sorted(set(b_pts) & set(s_pts))
    diffs = [s_pts[k] - b_pts[k] for k in keys]
    m, lo, hi = _t_ci(diffs)
    res = {"blocks": len(keys), "points_diff": [m, lo, hi],
           f"points_{names[0]}": sum(b_pts[k] for k in keys) / len(keys),
           f"points_{names[1]}": sum(s_pts[k] for k in keys) / len(keys)}
    for name, key in (("band", lambda h: BAND_NAMES[band(h["eff"])]),
                      ("street", lambda h: f"{STREETS[h['st']]}:{h['out']}"),
                      ("preflop", lambda h: h["pf"])):
        cats = sorted({key(h) for rows_ in (rows, base) for r in rows_ for h in r["hands"]})
        bs = per_block(base, lambda r: chips_by(r, key))
        ss = per_block(rows, lambda r: chips_by(r, key))
        res[f"bb_per_tournament_by_{name}"] = {}
        for c in cats:
            dd = [ss.get(k, {}).get(c, 0.0) - bs.get(k, {}).get(c, 0.0) for k in keys]
            res[f"bb_per_tournament_by_{name}"][c] = {
                names[0]: sum(bs[k].get(c, 0.0) for k in keys) / len(keys),
                names[1]: sum(ss[k].get(c, 0.0) for k in keys) / len(keys),
                "diff": list(_t_ci(dd))}
    return res


def compare(base_dir: Path, cand_dir: Path) -> dict:
    """The candidate's run against the base run on the same tables, per structure."""
    out = {}
    for s in STRUCTURES:
        b, c = base_dir / f"{s}.jsonl", cand_dir / f"{s}.jsonl"
        if b.exists() and c.exists():
            res = paired(_load(b), _load(c), ("base", "candidate"))
            rows = _load(c)
            hands = [h for r in rows for h in r["hands"]]
            first_in = [h for h in hands if h["pf"] in ("open", "fold_unopened", "limp")]
            res["candidate_first_in"] = {k: sum(h["pf"] == k for h in first_in) / max(1, len(first_in))
                                         for k in ("open", "limp", "fold_unopened")}
            out[s] = res
    return out


def _rate_by(hands: list[dict], key) -> dict:
    acc: dict = defaultdict(lambda: [0.0, 0])
    for h in hands:
        a = acc[key(h)]
        a[0] += h["d"] / h["bb"]
        a[1] += 1
    return {k: {"hands": n, "bb_per_100": 100 * s / n} for k, (s, n) in sorted(acc.items())}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--structures", nargs="+", default=list(STRUCTURES))
    r.add_argument("--lineups", type=int, default=40)
    r.add_argument("--blocks", type=int, default=1)
    r.add_argument("--lineup-seed", type=int, default=1)
    r.add_argument("--base-seed", default="antes-census-v1")
    r.add_argument("--jobs", type=int, default=7)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--hero", default=HERO, help="the policy in the fixed seat (default: the final bot)")
    c = sub.add_parser("compare")
    c.add_argument("base", type=Path)
    c.add_argument("candidate", type=Path)
    c.add_argument("--out", type=Path)
    p = sub.add_parser("report")
    p.add_argument("dirs", nargs="+", type=Path)
    p.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        for s in a.structures:
            print(run(s, lineups=a.lineups, blocks=a.blocks, lineup_seed=a.lineup_seed, base_seed=a.base_seed,
                      jobs=a.jobs, out=a.out, hero=a.hero), flush=True)
    else:
        res = report(a.dirs) if a.cmd == "report" else compare(a.base, a.candidate)
        text = json.dumps(res, indent=1)
        if a.out:
            a.out.write_text(text)
        print(text)


if __name__ == "__main__":
    main()
