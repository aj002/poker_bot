"""Red-team tournament suite: one command reruns every pre-registered cell against a bundle.

``preregistration.json`` fixes the lineups, block counts and seeds. For every cell (one
opponent lineup) and arm (the bundle under test, plus the blueprint-only, nlhe-v2 and
previous-final references) the suite plays paired seed blocks: each block is one deal
seeded from the cell seed, played once per seat rotation (8 tables), in process, scored
by the arena's official terminal points. All arms see identical deals and opponent
policy seeds, so differences between arms are the arm.

    python -m strategy.redteam run --bundle artifacts/redteam/bundles/final-handhq --out artifacts/redteam/runs/x
    python -m strategy.redteam run --pack NEW_BLUEPRINT.npz --exploit-pack NEW_EXPLOIT.npz --out ...
    python -m strategy.redteam report --out artifacts/redteam/runs/x

Per (cell, arm) results are cached in ``<out>/cells`` under a key of bundle hash, lineup,
blocks and seed, so an interrupted run resumes and a finished one is not replayed.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest import mock

from arena.agents.worker import artifact_hash
from arena.benchmark.runner import policy_seed
from arena.domain.models import TournamentConfig
from strategy.eval.population import Lineup, OpponentRef
from strategy.redteam import registry
from strategy.redteam.diagnostics import alive_band, table_diagnostics
from strategy.rules.canonical import canonical_hash

REPO = Path(__file__).resolve().parents[2]
PREREG = Path(__file__).resolve().parent / "preregistration.json"
NLHE = ("practice_nlhe",)
HERO = "ours"


def prereg() -> dict[str, Any]:
    return json.loads(PREREG.read_text())


def prereg_hash() -> str:
    return hashlib.sha256(PREREG.read_bytes()).hexdigest()


# ------------------------------------------------------------------ lineups


@dataclass(frozen=True)
class CellDef:
    cell_id: str
    set: str
    lineup: Lineup
    blocks: int
    seed: str

    def key(self) -> str:
        return canonical_hash([self.cell_id, self.set, [o.as_dict() for o in self.lineup.opponents], self.blocks,
                               self.seed], domain="redteam-cell-v1")


def _expand(spec: str) -> tuple[str, int]:
    name, _, n = spec.partition(" x")
    return name, int(n or 1)


def _opponent(name: str, previous: Mapping[str, str]) -> OpponentRef:
    if name.startswith("rt-"):
        return next(r for r in registry.refs() if r.member_id == name)
    if name.startswith("prev:"):
        ref = name[5:]
        return OpponentRef(f"prev.{ref}", "previous-bot", previous[ref], NLHE, artifact_hash(REPO / previous[ref]))
    raise ValueError(f"unknown opponent {name!r}")


def population_lineups(p: Mapping[str, Any]) -> list[tuple[str, list[OpponentRef]]]:
    """The three red-team bots plus ``population_seats`` validation members of pop-v1+handhq-v1, per lineup."""
    from strategy.eval.population import load_partition_source

    source = load_partition_source(p["source"])
    part = source.partition(p["partition"])
    if p["partition"] != "validation":
        raise PermissionError("the red-team suite never reads any partition but validation")
    drawn = source.sample_lineups(part, ruleset_id="practice_nlhe", opponents=p["population_seats"],
                                  count=p["lineups"], seed=p["seed"])
    red = [_opponent(n, {}) for n in p["red_team"]]
    return [(f"{p['id_prefix']}-{j + 1:02d}", [*red, *lu.opponents]) for j, lu in enumerate(drawn)]


def build_cells(pr: Mapping[str, Any], *, seed: str | None = None, blocks: int | None = None) -> list[CellDef]:
    base = seed or pr["base_seed"]
    previous = {k: v["path"] for k, v in pr["previous_bots"].items()}
    out: list[CellDef] = []

    def add(cell_id: str, set_name: str, opps: Sequence[OpponentRef]) -> None:
        n = blocks or pr["blocks"][set_name]
        s = hashlib.sha256(f"redteam-cell-seed-v1|{base}|{cell_id}".encode()).hexdigest()[:20]
        lid = "L_" + canonical_hash([o.member_id for o in opps], domain="strategy-lineup-id-v1")[:12]
        out.append(CellDef(cell_id, set_name, Lineup(lid, tuple(opps)), n, s))

    for c in pr["cells"]:
        opps = [_opponent(name, previous) for spec in c["opponents"] for name, k in [_expand(spec)] for _ in range(k)]
        add(c["id"], c["set"], opps)
    for cell_id, opps in population_lineups(pr["population"]):
        add(cell_id, "population", opps)
    for c in out:
        if len(c.lineup.opponents) != pr["table"]["players"] - 1:
            raise ValueError(f"cell {c.cell_id}: {len(c.lineup.opponents)} opponents")
    return out


def tags_for(cell: CellDef) -> dict[int, str]:
    """Entrant index -> diagnostics tag (the hero is entrant 0)."""
    tags = {0: "hero"}
    for i, o in enumerate(cell.lineup.opponents, start=1):
        tags[i] = o.member_id if o.member_id.startswith("rt-") else o.member_id if o.member_id.startswith("prev.") \
            else "pop"
    return tags


# ------------------------------------------------------------------ arms


def arm_paths(pr: Mapping[str, Any], bundle: str, refs: Mapping[str, str] | None = None) -> dict[str, str]:
    paths = {HERO: bundle}
    for name in pr["references"]:
        paths[name] = (refs or {}).get(name) or pr["previous_bots"][name]["path"]
    return paths


def bundle_hash(path: str) -> str:
    """The bundle's ``bundle_sha256`` (MANIFEST.json) or its artifact hash; ``baseline:``/``style:`` specs too."""
    p = Path(path) if Path(path).is_absolute() else REPO / path
    if not p.exists():
        from strategy.eval.policies import resolve_policy

        return resolve_policy(path).artifact_sha256
    for name, field in (("MANIFEST.json", "bundle_sha256"), ("manifest.json", "bundle_hash")):
        if (p / name).exists():
            h = json.loads((p / name).read_text()).get(field)
            if h:
                return h
    return artifact_hash(p)


def check_previous(pr: Mapping[str, Any]) -> list[str]:
    """Problems with the pinned previous bots (missing or not the pre-registered bytes)."""
    bad = []
    for name, spec in pr["previous_bots"].items():
        path = REPO / spec["path"]
        if not path.exists():
            bad.append(f"{name}: {spec['path']} is missing (python -m strategy.redteam build-previous)")
        elif not bundle_hash(spec["path"]).startswith(spec["sha256"]):
            bad.append(f"{name}: {spec['path']} is {bundle_hash(spec['path'])[:12]}, pre-registered {spec['sha256']}")
    return bad


# ------------------------------------------------------------------ running


def _phase0(table_seed: str, seat: int) -> int:
    return int(hashlib.sha256(str(policy_seed(table_seed, seat)).encode()).hexdigest()[:4], 16) % 2


class Collector:
    """Wraps ``strategy.eval.paired.table_record`` to read each history once before it is deleted."""

    def __init__(self, tags: Mapping[int, str], block_seeds: Sequence[str]) -> None:
        self.tags = dict(tags)
        self.block_seeds = list(block_seeds)
        self.diag: dict[tuple[int, int], dict[str, Any]] = {}

    def wrap(self, original: Any) -> Any:
        def table_record(result: Any, history_path: Path | None, focus: Sequence[int] = (0,), **kw: Any) -> Any:
            if result.valid and history_path is not None and history_path.exists():
                doc = json.loads(history_path.read_text(encoding="ascii"))
                seat_of = {e: s for s, e in result.seats.items()}
                tags = {seat_of[e]: t for e, t in self.tags.items()}
                seed = self.block_seeds[int(result.block)]
                phase0 = {seat_of[e]: _phase0(seed, seat_of[e]) for e, t in self.tags.items() if t == "rt-adaptive"}
                self.diag[(int(result.block), int(result.rotation))] = table_diagnostics(
                    doc["events"], seat_of[0], tags, phase0=phase0)
            return original(result, history_path, focus, **kw)

        return table_record


def run_cell_arm(cell: CellDef, arm: str, policy: str, *, config: TournamentConfig, work: Path, jobs: int,
                 isolation: str) -> dict[str, Any]:
    import strategy.eval.paired as paired
    from strategy.eval.paired import Arm, run_arm
    from strategy.eval.policies import resolve_policy

    resolved = resolve_policy(policy)
    from arena.benchmark.manifest import derive_seeds

    collector = Collector(tags_for(cell), derive_seeds(cell.seed, cell.blocks))
    t0 = time.time()
    with mock.patch.object(paired, "table_record", collector.wrap(paired.table_record)):
        run = run_arm(cell_as_cell(cell, config), Arm(arm, policy), resolved, isolation=isolation,
                      out_dir=work / "bench", jobs=jobs, keep_histories=False)
    tables = []
    for r in run.records:
        row: dict[str, Any] = {"block": r.block, "rotation": r.rotation, "valid": r.valid, "error": r.error,
                               "hands": r.hands, "points": r.points.get(0)}
        if r.valid and 0 in r.entrants:
            e = r.entrants[0]
            row |= {"finish": e.finish.method, "bust_level": e.bust_level, "survivors": r.survivors_at_end,
                    "diag": collector.diag.get((r.block, r.rotation))}
        tables.append(row)
    ex = run.execution
    return {"arm": arm, "policy": policy, "artifact_sha256": resolved.artifact_sha256, "tables": tables,
            "execution": {k: ex.get(k) for k in ("decisions", "fallback_total", "fallbacks", "timeouts",
                                                 "illegal_actions", "errors", "restarts", "failed_tables",
                                                 "invalid_tables")},
            "seat_timeouts": run.seat_timeouts, "wall_s": round(time.time() - t0, 1)}


def cell_as_cell(cell: CellDef, config: TournamentConfig) -> Any:
    from strategy.eval.paired import Cell

    return Cell(cell.cell_id, config, cell.lineup, cell.blocks, cell.seed)


def table_config(pr: Mapping[str, Any]) -> TournamentConfig:
    t = pr["table"]
    return TournamentConfig(ruleset_id="practice_nlhe", initial_player_count=t["players"],
                            starting_stack=t["starting_stack"], hand_cap=t["hand_cap"],
                            hands_per_level=t["hands_per_level"])


def code_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_suite(out: Path, bundle: str, *, refs: Mapping[str, str] | None = None, jobs: int = 8,
              seed: str | None = None, blocks: int | None = None, only: Sequence[str] | None = None,
              arms: Sequence[str] | None = None, reuse: Sequence[Path] = (), log: Any = sys.stderr) -> dict[str, Any]:
    pr = prereg()
    problems = check_previous(pr)
    if problems:
        raise SystemExit("previous bots differ from the pre-registration:\n  " + "\n  ".join(problems))
    cells = [c for c in build_cells(pr, seed=seed, blocks=blocks) if not only or c.cell_id in only]
    paths = arm_paths(pr, bundle, refs)
    if arms:
        paths = {k: v for k, v in paths.items() if k in arms}
    config = table_config(pr)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"prereg_sha256": prereg_hash(), "version": pr["version"], "commit": code_commit(),
            "base_seed": seed or pr["base_seed"], "isolation": pr["isolation"],
            "arms": {a: {"path": p, "bundle_sha256": bundle_hash(p)} for a, p in paths.items()},
            "registry": {"redteam": registry.registry_hash()}, "jobs": jobs, "blocks_override": blocks,
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "run.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    total = len(cells) * len(paths)
    done = 0
    for cell in cells:
        for arm, policy in paths.items():
            f = out / "cells" / cell.cell_id / f"{arm}.json"
            key = canonical_hash({"cell": cell.key(), "arm": arm, "artifact": bundle_hash(policy),
                                  "table": pr["table"], "isolation": pr["isolation"]}, domain="redteam-arm-v1")
            done += 1
            if f.exists() and json.loads(f.read_text()).get("key") == key:
                print(f"[{done}/{total}] {cell.cell_id}/{arm}: cached", file=log, flush=True)
                continue
            donor = next((d / "cells" / cell.cell_id / f"{arm}.json" for d in reuse
                          if (d / "cells" / cell.cell_id / f"{arm}.json").exists()
                          and json.loads((d / "cells" / cell.cell_id / f"{arm}.json").read_text()).get("key") == key),
                         None)
            if donor is not None:
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(donor.read_bytes())
                print(f"[{done}/{total}] {cell.cell_id}/{arm}: reused from {donor.parents[2]}", file=log, flush=True)
                continue
            res = run_cell_arm(cell, arm, policy, config=config, work=out / "work" / cell.cell_id / arm, jobs=jobs,
                               isolation=pr["isolation"])
            res |= {"key": key, "cell": cell.cell_id, "set": cell.set, "blocks": cell.blocks,
                    "lineup": [o.member_id for o in cell.lineup.opponents]}
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(res, separators=(",", ":")) + "\n")
            pts = [t["points"] for t in res["tables"] if t["valid"]]
            print(f"[{done}/{total}] {cell.cell_id}/{arm}: {len(pts)} tables, mean {sum(pts) / max(1, len(pts)):.2f}, "
                  f"{res['wall_s']}s", file=log, flush=True)
            _drop(out / "work" / cell.cell_id / arm)
    return meta


def _drop(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


# ------------------------------------------------------------------ building bundles


def build_candidate(pack: Path | None, exploit_pack: Path | None, out: Path) -> tuple[str, str]:
    """Final bundle and blueprint-only bundle from (optionally new) packs; returns their paths."""
    final, bp = out / "final", out / "blueprint-only"
    cmd = [sys.executable, "-m", "strategy.final.package", "--out", str(final)]
    if pack:
        cmd += ["--pack", str(pack)]
    if exploit_pack:
        cmd += ["--exploit-pack", str(exploit_pack)]
    subprocess.run(cmd, cwd=REPO, check=True, stdout=subprocess.DEVNULL)
    cmd = [sys.executable, "-m", "strategy.blueprint.package", "--pack",
           str(pack or "strategy/tables/data/blueprint-practice_nlhe.npz"), "--out", str(bp)]
    subprocess.run(cmd, cwd=REPO, check=True, stdout=subprocess.DEVNULL)
    return str(final), str(bp)


def build_previous(rev: str | None = None) -> None:
    """Rebuild the pinned previous bots from the packs at ``rev`` (default: the pre-registration tag) in a worktree."""
    pr = prereg()
    rev = rev or pr.get("previous_rev") or "HEAD"
    tree = REPO / "artifacts" / "redteam" / "_previous-src"
    if not tree.exists():
        subprocess.run(["git", "worktree", "add", "--detach", str(tree), rev], cwd=REPO, check=True)
    cmds = {
        "blueprint": [sys.executable, "-m", "strategy.blueprint.package", "--pack",
                      "strategy/tables/data/blueprint-practice_nlhe.npz"],
        "baseline": [sys.executable, "-m", "strategy.policy.package", "--params", "strategy/policy/vectors/nlhe-v2.json",
                     "--name", "strategy_baseline"],
        "final": [sys.executable, "-m", "strategy.final.package"],
    }
    for name, spec in pr["previous_bots"].items():
        dest = REPO / spec["path"]
        subprocess.run([*cmds[spec["build"]], "--out", str(dest)], cwd=tree, check=True, stdout=subprocess.DEVNULL,
                       env={**__import__("os").environ, "PYTHONPATH": str(tree)})
    bad = check_previous(pr)
    if bad:
        raise SystemExit("rebuilt previous bots differ from the pre-registration:\n  " + "\n  ".join(bad))


# ------------------------------------------------------------------ results


def rng(seed: str) -> random.Random:
    return random.Random(seed)


def block_means(tables: Sequence[Mapping[str, Any]]) -> dict[int, float]:
    by: dict[int, list[float]] = {}
    for t in tables:
        if t["valid"]:
            by.setdefault(t["block"], []).append(t["points"])
    return {b: sum(v) / len(v) for b, v in by.items() if len(v) == 8}


def load_cells(out: Path) -> dict[str, dict[str, dict[str, Any]]]:
    res: dict[str, dict[str, dict[str, Any]]] = {}
    for f in sorted((out / "cells").glob("*/*.json")):
        res.setdefault(f.parent.name, {})[f.stem] = json.loads(f.read_text())
    return res


def mean_ci(xs: Sequence[float], level: float = 0.95, center: float = 0.0) -> dict[str, Any]:
    from strategy.eval.stats import paired_t

    t = paired_t([x - center for x in xs], level)
    return {"n": t.n, "mean": t.mean + center, "sd": t.sd, "lo": None if t.lo is None else t.lo + center,
            "hi": None if t.hi is None else t.hi + center, "p": t.p_value}


def stratified(groups: Sequence[Sequence[float]], level: float = 0.95, center: float = 0.0) -> dict[str, Any]:
    """Equal-weight mean of per-cell means; standard error from the within-cell variances."""
    from strategy.eval.stats import mean, sample_sd, t_cdf, t_ppf

    groups = [g for g in groups if len(g) >= 2]
    k = len(groups)
    if not k:
        return {"n": 0, "mean": None, "lo": None, "hi": None, "p": None}
    m = sum(mean(g) for g in groups) / k
    var = sum(sample_sd(g) ** 2 / len(g) for g in groups) / k ** 2
    df = sum(len(g) for g in groups) - k
    se = math.sqrt(var)
    tc = t_ppf(0.5 + level / 2, df)
    p = 2 * (1 - t_cdf(abs(m - center) / se, df)) if se else None
    return {"n": sum(len(g) for g in groups), "mean": m, "sd": None, "lo": m - tc * se, "hi": m + tc * se, "p": p,
            "cells": k}


def classify(ci: Mapping[str, Any], par: float) -> str:
    if ci.get("lo") is None:
        return "n/a"
    return "LOSES" if ci["hi"] < par else "WINS" if ci["lo"] > par else "BREAK-EVEN"


def analyze(out: Path) -> dict[str, Any]:
    from strategy.eval.stats import holm

    pr = prereg()
    par, level = pr["par"], pr["level"]
    cells = load_cells(out)
    meta = json.loads((out / "run.json").read_text())
    res: dict[str, Any] = {"meta": meta, "par": par, "cells": {}, "sets": {}, "integrity": {}}
    ordered = sorted(cells, key=lambda c: next((i for i, x in enumerate(pr["cells"]) if x["id"] == c), 99) * 100
                     + (int(c.split("-")[-1]) if c.startswith("pop-") else 0))
    for cid in ordered:
        arms = cells[cid]
        blocks = {a: block_means(r["tables"]) for a, r in arms.items()}
        row: dict[str, Any] = {"set": next(iter(arms.values()))["set"], "lineup": next(iter(arms.values()))["lineup"],
                               "arms": {}, "diff": {}}
        for a, bm in blocks.items():
            vals = [bm[b] for b in sorted(bm)]
            row["arms"][a] = mean_ci(vals, level, par) | {"class": classify(mean_ci(vals, level), par)}
            ex = arms[a]["execution"]
            res["integrity"].setdefault(a, {"tables": 0, "invalid": 0, "illegal": 0, "errors": 0, "fallbacks": 0,
                                            "timeouts": 0, "decisions": 0, "opp_timeouts": 0})
            i = res["integrity"][a]
            i["tables"] += len(arms[a]["tables"])
            i["invalid"] += sum(1 for t in arms[a]["tables"] if not t["valid"])
            i["illegal"] += ex.get("illegal_actions") or 0
            i["errors"] += ex.get("errors") or 0
            i["fallbacks"] += ex.get("fallback_total") or 0
            i["timeouts"] += ex.get("timeouts") or 0
            i["decisions"] += ex.get("decisions") or 0
            i["opp_timeouts"] += sum(v.get("timeouts", 0) for k, v in (arms[a].get("seat_timeouts") or {}).items()
                                     if k != "candidate")
        for a in blocks:
            if a == HERO or HERO not in blocks:
                continue
            common = sorted(set(blocks[HERO]) & set(blocks[a]))
            diffs = [blocks[HERO][b] - blocks[a][b] for b in common]
            row["diff"][a] = mean_ci(diffs, level)
        res["cells"][cid] = row
    hero_p = {c: r["arms"][HERO]["p"] for c, r in res["cells"].items() if HERO in r["arms"]
              and r["arms"][HERO]["p"] is not None}
    for c, adj in holm(hero_p).items():
        res["cells"][c]["arms"][HERO]["p_holm"] = adj
        res["cells"][c]["arms"][HERO]["holm_significant"] = adj < pr["alpha"]
    for set_name in pr["sets"]:
        members = [c for c, r in res["cells"].items() if r["set"] == set_name]
        if not members:
            continue
        s: dict[str, Any] = {"cells": members, "arms": {}, "diff": {}}
        for a in {a for c in members for a in res["cells"][c]["arms"]}:
            groups = [[bm[b] for b in sorted(bm)] for c in members if a in cells[c]
                      for bm in [block_means(cells[c][a]["tables"])]]
            ci = stratified(groups, level, par)
            s["arms"][a] = ci | {"class": classify(ci, par)}
        for a in s["arms"]:
            if a == HERO or HERO not in s["arms"]:
                continue
            groups = []
            for c in members:
                if a in cells[c] and HERO in cells[c]:
                    h, r = block_means(cells[c][HERO]["tables"]), block_means(cells[c][a]["tables"])
                    groups.append([h[b] - r[b] for b in sorted(set(h) & set(r))])
            s["diff"][a] = stratified(groups, level, 0.0)
        res["sets"][set_name] = s
    res["attribution"] = attribution(cells, res, pr)
    res["census"] = census(cells)
    return res


# ------------------------------------------------------------------ attribution


def _boot(values: Sequence[tuple[float, float]], seed: str, resamples: int = 1000) -> tuple[float, float, float]:
    """Ratio sum(y)/sum(x) over tables with a 95% bootstrap interval over tables."""
    n = len(values)
    sx, sy = sum(v[0] for v in values), sum(v[1] for v in values)
    est = sy / sx if sx else 0.0
    if n < 2:
        return est, est, est
    r = rng(seed)
    stats = []
    for _ in range(resamples):
        xs = ys = 0.0
        for _ in range(n):
            a = values[r.randrange(n)]
            xs += a[0]
            ys += a[1]
        stats.append(ys / xs if xs else 0.0)
    stats.sort()
    return est, stats[int(0.025 * resamples)], stats[int(0.975 * resamples) - 1]


def _image_diff(diags: Sequence[Mapping[str, Any]], seed: str, resamples: int = 1000) -> dict[str, float] | None:
    """Hero chips per hand (bb) against the adaptive bot in its tight minus its loose phase, bootstrapped over tables."""
    rows = [(d["image"]["tight"], d["image"]["loose"]) for d in diags]
    if sum(t[0] for t, _ in rows) < 50 or sum(lo[0] for _, lo in rows) < 50:
        return None

    def stat(sample: Sequence[Any]) -> float:
        tn = sum(t[0] for t, _ in sample)
        ln = sum(lo[0] for _, lo in sample)
        return (sum(t[1] for t, _ in sample) / tn if tn else 0.0) - (sum(lo[1] for _, lo in sample) / ln if ln else 0.0)

    r = rng(seed)
    n = len(rows)
    boots = sorted(stat([rows[r.randrange(n)] for _ in range(n)]) for _ in range(resamples))
    return {"est": stat(rows), "lo": boots[int(0.025 * resamples)], "hi": boots[int(0.975 * resamples) - 1],
            "tight_hands": sum(t[0] for t, _ in rows), "loose_hands": sum(lo[0] for _, lo in rows)}


def _tables(arm: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [t["diag"] for t in arm["tables"] if t.get("diag")]


def _sum_resp(diags: Sequence[Mapping[str, Any]], bot_prefix: str, classes: Sequence[str]) -> list[int]:
    tot = [0, 0, 0]
    for d in diags:
        for k, v in d["resp"].items():
            tag, cls = k.split("|")
            if tag.startswith(bot_prefix) and cls in classes:
                for i in range(3):
                    tot[i] += v[i]
    return tot


def attribution(cells: Mapping[str, Mapping[str, Mapping[str, Any]]], res: Mapping[str, Any], pr: Mapping[str, Any]
                ) -> dict[str, Any]:
    """Pre-registered behaviour flags per cell and arm, plus pooled per-set diagnostics."""
    out: dict[str, Any] = {"cells": {}}
    for cid, arms in cells.items():
        row: dict[str, Any] = {}
        for a, r in arms.items():
            diags = _tables(r)
            if not diags:
                continue
            fold_over = _sum_resp(diags, "rt-", ("post_over",))
            fold_ref = _sum_resp(diags, "rt-", ("post_mid", "post_small"))
            fr_over = fold_over[0] / sum(fold_over) if sum(fold_over) else None
            fr_ref = fold_ref[0] / sum(fold_ref) if sum(fold_ref) else None
            ob = [(d["pots"]["overbet"][0], d["pots"]["overbet"][1]) for d in diags]
            ob_est, ob_lo, ob_hi = _boot(ob, f"{cid}|{a}|overbet")
            bandrows: dict[str, Any] = {}
            for band in ("8-6", "5", "4", "3", "2"):
                vals = [(d["alive_hands"].get(band, 0), d["alive_eq"].get(band, 0.0)) for d in diags]
                est, lo, hi = _boot(vals, f"{cid}|{a}|band{band}")
                bandrows[band] = {"hands": sum(v[0] for v in vals), "eq_per_table": sum(v[1] for v in vals) / len(vals),
                                  "eq_per_hand": est, "lo": lo, "hi": hi}
            late = [(1.0, d["alive_eq"].get("4", 0.0) + d["alive_eq"].get("3", 0.0)) for d in diags]
            l_est, l_lo, l_hi = _boot(late, f"{cid}|{a}|late")
            phases = [sum(d["phase_eq"][i] for d in diags) / len(diags) for i in range(3)]
            rates = {}
            for band in ("8-6", "5", "4", "3", "2"):
                dealt = sum(d["bandagg"].get(f"{band}|dealt", 0) for d in diags)
                rates[band] = sum(d["bandagg"].get(f"{band}|agg", 0) for d in diags) / dealt if dealt else None
            img = _image_diff(diags, f"{cid}|{a}|image")
            bust: dict[str, int] = {}
            for d in diags:
                b = d["bust"]
                key = "survived" if not b else f"{b.get('by') or 'none'}@{alive_band(b['alive'] or 8)}"
                bust[key] = bust.get(key, 0) + 1
            flags = {
                "OVERBET": bool(fr_over is not None and fr_ref is not None and sum(fold_over) >= 30
                                and fr_over - fr_ref >= 0.10 and ob_hi < 0),
                "BUBBLE": bool(l_hi < 0 and rates["4"] is not None and rates["8-6"] is not None
                               and max(rates["4"], rates["3"] or 0) > rates["8-6"]),
                "IMAGE": bool(img is not None and (img["lo"] > 0 or img["hi"] < 0)),
            }
            row[a] = {"tables": len(diags), "fold_to_overbet": fr_over, "fold_to_smaller": fr_ref,
                      "overbet_faced": fold_over, "overbet_hands": sum(x[0] for x in ob),
                      "overbet_chips_bb_per_hand": [ob_est, ob_lo, ob_hi], "bands": bandrows,
                      "bubble_money_eq_per_table": [l_est, l_lo, l_hi], "phase_eq": phases,
                      "redteam_agg_rate_by_band": rates, "image_tight_minus_loose": img, "bust": bust, "flags": flags}
        out["cells"][cid] = row
    return out


def census(cells: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    """Opponent behaviour pooled over every table of the arm under test, per opponent tag; and how the hero busts."""
    out: dict[str, Any] = {}
    for arm in (HERO,):
        tags: dict[str, Counter] = {}
        bust: Counter = Counter()
        tables = 0
        for arms in cells.values():
            for d in _tables(arms.get(arm, {"tables": []})):
                tables += 1
                for tag, c in d["agg"].items():
                    tags.setdefault(tag, Counter()).update(c)
                b = d["bust"]
                bust["survived" if not b else f"{b.get('by') or 'none'}@{alive_band(b['alive'] or 8)}"] += 1
        rows = {}
        for tag, c in sorted(tags.items()):
            hands = c["hands"]
            post = c["post_small"] + c["post_mid"] + c["post_over"]
            rows[tag] = {"hands": hands, "open_rate": c["opens"] / hands, "open_bb": c["open_bb"] / c["opens"]
                         if c["opens"] else None, "three_bet_plus_per_100": 100 * (c["pre_3bet"] + c["pre_4bet"]) / hands,
                         "shoves_per_100": 100 * c["shove"] / hands, "postflop_aggressive_per_100": 100 * post / hands,
                         "overbet_share_of_postflop": c["post_over"] / post if post else None}
        out[arm] = {"tables": tables, "behaviour": rows, "hero_bust": dict(bust)}
    return out


def main_run(args: Any) -> int:
    out = Path(args.out)
    if args.pack or args.exploit_pack:
        final, bp = build_candidate(Path(args.pack) if args.pack else None,
                                    Path(args.exploit_pack) if args.exploit_pack else None, out / "bundles")
        bundle, refs = final, {"blueprint-only": bp}
    else:
        bundle, refs = args.bundle, {}
    for kv in args.ref or []:
        k, _, v = kv.partition("=")
        refs[k] = v
    meta = run_suite(out, bundle, refs=refs, jobs=args.jobs, seed=args.seed, blocks=args.blocks,
                     only=args.only or None, arms=args.arm or None,
                     reuse=[Path(x) for x in args.reuse or []])
    from strategy.redteam.report import write_report

    res = analyze(out)
    write_report(out, res)
    print(f"report: {out / 'report.md'}")
    return 0 if meta else 1
