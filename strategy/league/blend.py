"""Blend frontier between the blueprint and a league checkpoint (amendment 2 of the pre-registration).

``B_lam`` is the blueprint pack with level-0 rows at the child's override keys replaced by
``(1 - lam) * blueprint_row + lam * child_row`` (the blueprint row resolved with back-off).
``lam = 0`` is the blueprint and ``lam = 1`` the child.

``frontier`` (train only): for every lam, the abstract population gain over the blueprint,
an exploiter ``X_lam`` (the league's exploiter config) and the antisymmetric advantage
``H(B_lam, j)`` over the strategic pool and ``X_lam``. Real-engine runs on the train
partition measure the population gain and ``X_lam``'s gain. The declared rule then picks lam.

``validate``: the chosen blend through the league's original promotion gates, curve,
selection, confirmations and deployment export, on the validation partition.

    uv run --group train python -m strategy.league.blend frontier --run artifacts/league/main-v1
    uv run --group train python -m strategy.league.blend validate --run artifacts/league/main-v1
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from strategy.league.runner import BLUEPRINT_PACK, REPO, League, _fmt, _mixture

CHILD = "C1"
CHILD_EXPLOITER = "E2"
LAMBDAS = (0.25, 0.5, 0.7, 0.85, 0.95)
POOL = ("blueprint", "baseline", "C1", "E2")
EXPLOITER_SEED = 3000
H_TOLERANCE = 0.15


def tag(lam: float) -> str:
    return f"{int(round(lam * 100)):02d}"


def blend_rows(base: np.ndarray, child: np.ndarray, keys: np.ndarray, lam: float) -> np.ndarray:
    """``uint8`` rows (sum 255) of ``(1 - lam) * base + lam * child`` at ``keys`` (tables flat or 2-D)."""
    from strategy.blueprint.policy import quantize
    from strategy.league.tables import ROW

    def norm(t):
        r = t.reshape(-1, ROW)[keys].astype(np.float64)
        s = r.sum(axis=1, keepdims=True)
        return np.where(s > 0, r / np.where(s > 0, s, 1.0), 1.0 / ROW)

    return quantize((1.0 - lam) * norm(base) + lam * norm(child))


def blend_deviations(base: np.ndarray, child: np.ndarray, keys: np.ndarray, lam: float):
    from strategy.league.tables import ROW, Deviations

    rows = blend_rows(base, child, keys, lam)
    best = child.reshape(-1, ROW)[keys].argmax(axis=1).astype(np.int64)
    return Deviations(np.asarray(keys, np.int64), rows, best)


def _spec(eid: str, a: str, b: str, *, partition: str, lineups=None, explicit=None, blocks: int,
          stage: str = "discovery", description: str = "") -> dict:
    from strategy.league import gates as Gt

    s = Gt._spec(eid, eid, description or f"league blend frontier ({partition})", a, b, base_seed=eid, stage=stage,
                 lineups=lineups, explicit=explicit, blocks=blocks)
    s["population"]["partition"] = partition
    return s


def _fill(partition: str, count: int, seats: int, seed: str) -> list[list[str]]:
    from strategy.population import PopulationSource

    source = PopulationSource()
    lus = source.sample_lineups(source.partition(partition), ruleset_id="practice_nlhe", opponents=seats,
                                count=count, seed=seed)
    return [[o.policy for o in lu.opponents] for lu in lus]


class Frontier:
    def __init__(self, run: Path) -> None:
        self.lg = League(run)
        self.dir = run / "blends"
        self.dir.mkdir(exist_ok=True)
        self.path = self.dir / "frontier.json"
        self.rec = json.loads(self.path.read_text()) if self.path.exists() else {"points": {}}

    def save(self) -> None:
        self.path.write_text(json.dumps(self.rec, indent=1))

    def keys(self) -> np.ndarray:
        d = np.load(self.lg.dir / "iterations" / "it1" / "deviations-z2.0m1.0.npz")
        return d["keys"].astype(np.int64)

    # ------------------------------------------------------------------ abstract (train data)

    def abstract(self) -> None:
        from strategy.league import tables as T
        from strategy.league.learner import SeatMixture, homogeneous

        lg = self.lg
        ab = lg.abstract()
        ab.pool = lg.train_pool(2)
        keys = self.keys()
        bp, child = ab.tabs[ab.row("blueprint")], ab.tabs[ab.row(CHILD)]
        count = ab._eval["count"]
        pop = ab.seat_draws(_mixture(lg.population_mixture()), tag=7)
        bp_pop = ab.values("blueprint", pop)
        world = {}

        def own_world(name):
            if name not in world:
                world[name] = ab.values(name, homogeneous(ab.row(name), count))
            return world[name]

        points = [(0.0, "blueprint", "E1"), *[(lam, f"B{tag(lam)}", f"X{tag(lam)}") for lam in LAMBDAS],
                  (1.0, CHILD, CHILD_EXPLOITER)]
        for lam, name, ex in points:
            key = tag(lam)
            if key in self.rec["points"] and "h" in self.rec["points"][key]:
                continue
            pt = self.rec["points"].setdefault(key, {"lam": lam, "name": name, "exploiter": ex})
            if name not in ab.names:
                dev = blend_deviations(bp, child, keys, lam)
                pack = self.dir / f"{name}.npz"
                T.write_override_pack(REPO / BLUEPRINT_PACK, dev, pack,
                                      {"ruleset_id": lg.cfg.ruleset_id, "league": lg.cfg.run, "member": name,
                                       "kind": "blend", "child": CHILD, "lam": lam, "overrides_from": CHILD})
                ab.add(name, T.apply_deviations(bp, dev))
                pt["pack"] = str(pack.relative_to(lg.dir))
            if ex not in ab.names:
                lg.log(f"blend {name}: training exploiter {ex} (1,000,000 pairs)")
                base = ab.tabs[ab.row(name)]
                regret, sq, stats = ab.train_response(name, SeatMixture.pure(ab.row(name)),
                                                      pairs=lg.cfg.exploiter_pairs, z=lg.cfg.exploiter_z,
                                                      seed=EXPLOITER_SEED + int(round(lam * 100)),
                                                      producers=lg.cfg.producers, log=lg.log)
                dev = T.deviations(regret, sq, base, z=lg.cfg.exploiter_z, mix=1.0)
                del regret, sq
                ab.add(ex, T.apply_deviations(base, dev))
                pack = self.dir / f"{ex}.npz"
                T.write_override_pack(self.dir / f"{name}.npz", dev, pack,
                                      {"ruleset_id": lg.cfg.ruleset_id, "league": lg.cfg.run, "member": ex,
                                       "kind": "exploiter", "target": name, "train": stats})
                pt["exploiter_pack"] = str(pack.relative_to(lg.dir))
                pt["exploiter_deviations"] = len(dev)
                pt["exploiter_train"] = stats
            from strategy.blueprint.response import paired_gain

            pt["population_gain"] = paired_gain(bp_pop, ab.values(name, pop)) if name != "blueprint" else None
            h, g = {}, {}
            for j in (*[lg.table_name(p) for p in POOL], ex):
                if j == name:
                    continue
                a_in_j = paired_gain(own_world(j), ab.values(name, homogeneous(ab.row(j), count)))
                j_in_a = paired_gain(own_world(name), ab.values(j, homogeneous(ab.row(name), count)))
                g[j] = {"self_over_j": a_in_j, "j_over_self": j_in_a}
                h[j] = (a_in_j["gain_points"] - j_in_a["gain_points"]) / 2
            pt["h"] = h
            pt["g"] = g
            pt["worst_h"] = min(h.values())
            pt["worst_vs"] = min(h, key=h.get)
            pt["exploiter_gain"] = g[ex]["j_over_self"]
            pt["eligible"] = bool(pt["worst_h"] >= -H_TOLERANCE)
            self.save()
            pop_s = (f"{pt['population_gain']['gain_points']:+.4f}" if pt["population_gain"] else "0")
            lg.log(f"blend {name}: population {pop_s}; {ex} gain {pt['exploiter_gain']['gain_points']:+.4f} ± "
                   f"{pt['exploiter_gain']['se']:.4f}; worst H {pt['worst_h']:+.4f} vs {pt['worst_vs']}; "
                   f"eligible {pt['eligible']}")

    # ------------------------------------------------------------------ real engine, train partition

    def bundle(self, name: str) -> str:
        import os

        from strategy.blueprint.package import build, smoke

        if name in self.lg.state["members"]:
            return self.lg.bundle(name)
        out = self.dir / "bundles" / name
        if not (out / "MANIFEST.json").exists():
            build(self.dir / f"{name}.npz", out)
            smoke(out)
        return os.path.relpath(out, REPO)

    def train_runs(self) -> None:
        from strategy.league import gates as Gt

        lg = self.lg
        bp = self.bundle("blueprint")
        for key, pt in sorted(self.rec["points"].items()):
            name, ex = pt["name"], pt["exploiter"]
            b, e = self.bundle(name), self.bundle(ex)
            run = lg.cfg.run
            if name != "blueprint" and "train_pop" not in pt:
                s = _spec(f"league-{run}-blend{key}-train-pop", bp, b, partition="train", lineups=12, blocks=8)
                pt["train_pop"] = Gt.run(Gt.write_spec(s, lg.dir / "specs"), jobs=lg.cfg.jobs)
                lg.log(f"blend {name} train pop: {_fmt(pt['train_pop'])}")
                self.save()
            if "train_exploit" not in pt:
                fill = _fill("train", 8, 4, f"league-{run}-blend{key}-train-exploit")
                s = _spec(f"league-{run}-blend{key}-train-exploit", b, e, partition="train",
                          explicit=[[b] * 3 + f for f in fill], blocks=6)
                pt["train_exploit"] = Gt.run(Gt.write_spec(s, lg.dir / "specs"), jobs=lg.cfg.jobs)
                lg.log(f"blend {ex} over {name} train: {_fmt(pt['train_exploit'])}")
                self.save()

    def choose(self) -> dict:
        """Declared rule: among eligible lam in (0, 1), the largest real-engine train population gain;
        stop (null) unless that gain's lower bound is above 0."""
        cands = [pt for pt in self.rec["points"].values() if 0 < pt["lam"] < 1 and pt["eligible"]
                 and pt.get("train_pop", {}).get("diff") is not None]
        if not cands:
            out = {"chosen": None, "reason": "no eligible blend"}
        else:
            best = max(cands, key=lambda p: p["train_pop"]["diff"])
            ok = best["train_pop"]["lo"] > 0
            out = {"chosen": best["name"] if ok else None, "lam": best["lam"], "candidate": best["name"],
                   "reason": None if ok else f"train population gain lower bound {best['train_pop']['lo']:+.2f} <= 0"}
        self.rec["choice"] = out
        self.save()
        self.lg.log(f"blend choice: {out}")
        return out

    # ------------------------------------------------------------------ validation

    def validate(self) -> dict:
        from strategy.league import gates as Gt
        from strategy.league import runner as R

        lg = self.lg
        choice = self.rec["choice"]
        name = choice["chosen"]
        assert name, "no blend was chosen; nothing to validate"
        pt = next(p for p in self.rec["points"].values() if p["name"] == name)
        ex = pt["exploiter"]
        val_path = self.dir / "validation.json"
        val = json.loads(val_path.read_text()) if val_path.exists() else {}
        run = lg.cfg.run
        if name not in lg.state["members"]:
            lg.add_member(name, "learner", pack=self.dir / f"{name}.npz", iteration=0, parent="blueprint",
                          info={"blend": pt["lam"], "child": CHILD})
            eg = pt["exploiter_gain"]
            lg.add_member(ex, "exploiter", pack=self.dir / f"{ex}.npz", iteration=0, parent=name,
                          info={"target": name, "gain_vs_target": eg})
            lo = eg["gain_points"] - 1.96 * eg["se"]
            lg.set_status(ex, "promoted" if lo > 0 else "retired",
                          f"abstract held-out gain over {name}: {eg['gain_points']:+.4f} ± {eg['se']:.4f}")
        cand = (name, lg.bundle(name))
        if "gates" not in val:
            runs = {}
            s = Gt.improve_spec(run, 0, ("blueprint", lg.bundle("blueprint")), cand, lg.rule)
            s["experiment_id"] = s["base_seed"] = f"league-{run}-{name}-improve"
            runs["improve"] = Gt.run(Gt.write_spec(s, lg.dir / "specs"), jobs=lg.cfg.jobs)
            lg.log(f"{name}: improve {_fmt(runs['improve'])}")
            regress = {}
            for j in ("blueprint", "baseline", CHILD, CHILD_EXPLOITER, ex):
                s = Gt.regress_spec(run, 0, (j, lg.bundle(j)), cand, lg.rule, tag=f"-{name}")
                s["experiment_id"] = s["base_seed"] = f"league-{run}-{name}-regress-{j}"
                regress[j] = runs[f"regress-{j}"] = Gt.run(Gt.write_spec(s, lg.dir / "specs"), jobs=lg.cfg.jobs)
                lg.log(f"{name}: regress-{j} {_fmt(regress[j])}")
            decision = Gt.decide(lg.rule, runs["improve"], regress)
            for c in decision["checks"]:
                lg.log(f"{name}:   {'PASS' if c['passed'] else 'FAIL'} {c['check']}: {c['detail']}")
            val["gates"] = {"runs": runs, "decision": decision, "promote": decision["promote"]}
            val_path.write_text(json.dumps(val, indent=1))
            if lg.member(name)["status"] == "candidate":
                ids = ", ".join(r["manifest_id"] for r in runs.values() if r.get("manifest_id"))
                if decision["promote"]:
                    lg.set_status(name, "promoted", f"blend gates: {ids}")
                else:
                    failed = "; ".join(c["check"] for c in decision["checks"] if not c["passed"])
                    lg.set_status(name, "retired", f"blend gates failed: {failed}")
        if "exploit" not in val:
            s = Gt.regress_spec(run, 0, (name, lg.bundle(name)), (ex, lg.bundle(ex)), lg.rule, tag="-exploit")
            s["experiment_id"] = s["base_seed"] = f"league-{run}-exploit-{ex}-vs-{name}"
            s["description"] = "Amendment 2, descriptive: the blend's exploiter vs the blend in blend-rich lineups."
            val["exploit"] = Gt.run(Gt.write_spec(s, lg.dir / "specs"), jobs=lg.cfg.jobs)
            lg.log(f"{ex} over {name}: {_fmt(val['exploit'])}")
            val_path.write_text(json.dumps(val, indent=1))
        if lg.member(name)["status"] != "promoted":
            val["outcome"] = "not promoted: the blueprint stays the default"
            val_path.write_text(json.dumps(val, indent=1))
            return val
        if "selection" not in val:
            R.curve(lg)
            val["selection"] = R.select(lg)
            val_path.write_text(json.dumps(val, indent=1))
        if val["selection"]["selected"] != name:
            val["outcome"] = f"promoted but not selected ({val['selection']['selected']}): the blueprint stays"
            val_path.write_text(json.dumps(val, indent=1))
            return val
        if "confirm" not in val:
            val["confirm"] = R.confirm(lg, name, tag=f"-{name}")
            val_path.write_text(json.dumps(val, indent=1))
        confirmed = all(v["confirmed"] for v in val["confirm"]["holm"].values())
        val["outcome"] = ("selected and confirmed: export and gate the deployed bundle" if confirmed else
                          "selected but not every confirmation held: the blueprint stays")
        val_path.write_text(json.dumps(val, indent=1))
        return val


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("frontier", "choose", "validate", "show"))
    ap.add_argument("--run", type=Path, required=True)
    a = ap.parse_args(argv)
    f = Frontier(a.run)
    if a.command == "frontier":
        f.abstract()
        f.train_runs()
        f.choose()
    elif a.command == "choose":
        f.choose()
    elif a.command == "validate":
        print(json.dumps({k: v for k, v in f.validate().items() if k in ("outcome",)}, indent=1))
    else:
        for key, pt in sorted(f.rec["points"].items()):
            tp, te = pt.get("train_pop") or {}, pt.get("train_exploit") or {}
            pop = pt.get("population_gain") or {"gain_points": 0.0}
            print(f"lam {pt['lam']:.2f} {pt['name']:>9}: abstract pop {pop['gain_points']:+.4f}, "
                  f"{pt['exploiter']} {pt['exploiter_gain']['gain_points']:+.4f}, worst H {pt['worst_h']:+.4f} "
                  f"({pt['worst_vs']}), eligible {pt['eligible']}; train pop {tp.get('diff', math.nan):+.2f}, "
                  f"train exploit {te.get('diff', math.nan):+.2f}")
        print(f.rec.get("choice"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
