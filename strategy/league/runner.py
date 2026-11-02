"""League runner (Stage 5 / M6): frozen checkpoints, an empirical opponent mixture, gated promotion.

``python -m strategy.league.runner <command> --run artifacts/league/<name>`` with commands

``init``       freeze the config, build the anchor bundles (blueprint, baseline), register them in
               the run's checkpoint ledger, clone the training population and the baseline into
               abstract tables, harvest the evaluation state pool and the held-out deals;
``iterate``    run iterations until ``config.iterations`` (resumable: every step writes its result
               and is skipped when the result exists);
``curve``      the learning curve: every promoted checkpoint against the blueprint on validation;
``select``     choose the deployment checkpoint by the declared rule;
``confirm``    the pre-registered confirmation family for the selection, Holm-corrected;
``export``     copy the selected pack to ``strategy/tables/data/league-<ruleset>.npz``;
``status``     print the league state.

One iteration ``k``:

1. **States.** Start-of-hand states from training lineups with the incumbent seated, added to
   the evaluation pool for training.
2. **Exploiter.** A greedy gated response ``E_k`` to the incumbent alone (an approximate best
   response). It joins the pool, as an ``approximate_response`` slot, when its held-out gain
   over the incumbent is significantly positive; that gain is the incumbent's
   exploitability proxy.
3. **Meta-solve.** The payoff matrix ``G`` over the strategic pool (anchors, promoted
   checkpoints, promoted exploiters) on frozen held-out deals; the mixture ``sigma_k`` by
   regret matching (:mod:`strategy.league.meta`), with a floor so every member keeps some
   weight; its NashConv in the linearised meta-game and the exact mixed-seat check.
4. **Learner.** A gated response ``L_k`` to the *seat mixture*: each opponent seat is a
   training-population family model with probability ``population``, an odd legal policy
   (uniform over legal actions) with probability ``odd``, and a pool member drawn from
   ``sigma_k`` otherwise. The learner starts from the incumbent's table and deviates only
   where the regret evidence clears the gate; its deviations are blended with the
   incumbent's row. The gate and blend are chosen on held-out abstract deals by the declared
   rule (largest gain against the seat mixture, no loss beyond tolerance against any pool
   member). A learner that does not beat the incumbent on the held-out mixture is rejected
   without real-engine runs.
5. **Gates.** Real-engine paired runs on official points (:mod:`strategy.league.gates`): candidate
   vs incumbent on the validation population, and candidate vs every earlier deployable pool
   member in that member's rich lineups. The candidate is promoted only if the declared rule
   passes; otherwise it is retired and the incumbent stays.

The holdout partition is never read. Nothing here needs numba except the abstract steps.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
BLUEPRINT_PACK = "strategy/tables/data/blueprint-practice_nlhe.npz"
BASELINE_PARAMS = "strategy/policy/vectors/nlhe-v2.json"


@dataclass
class Config:
    run: str
    ruleset_id: str = "practice_nlhe"
    iterations: int = 4
    clone_lineups: int = 160
    clone_blocks: int = 1
    clone_seed: int = 1
    clone_alpha: float = 2.0
    eval_states_lineups: int = 40
    train_states_lineups: int = 24
    eval_deals: int = 30000
    population: float = 0.45
    odd: float = 0.02
    sigma_floor: float = 0.05
    meta_iters: int = 20000
    learner_pairs: int = 2_000_000
    learner_z: tuple[float, ...] = (2.0, 2.5, 3.0)
    learner_mix: tuple[float, ...] = (0.5, 1.0)
    abstract_tolerance: float = 0.15
    """Points per player-hand a learner may lose to the incumbent in any pool member's homogeneous
    world (a catastrophe guard: the train-partition pilot showed such abstract losses do not carry
    over to mixed real lineups; the real-engine regression gates decide)."""
    exploiter_pairs: int = 1_000_000
    exploiter_z: float = 3.0
    threads: int = 6
    producers: int = 2
    jobs: int = 7
    rule: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> Config:
        raw = json.loads(path.read_text())
        raw["learner_z"] = tuple(raw["learner_z"])
        raw["learner_mix"] = tuple(raw["learner_mix"])
        return cls(**raw)


class League:
    def __init__(self, run_dir: Path) -> None:
        from strategy.league.gates import Rule
        from strategy.population.registry import CheckpointLedger

        self.dir = run_dir
        self.cfg = Config.load(run_dir / "config.json")
        self.rule = Rule(**self.cfg.rule)
        self.ledger = CheckpointLedger(run_dir / "ledger.json")
        self.state_path = run_dir / "league.json"
        self.state = json.loads(self.state_path.read_text())
        self._abstract = None

    # ------------------------------------------------------------------ bookkeeping

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.dir / "log.txt", "a") as f:
            f.write(line + "\n")

    def save(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1) + "\n")
        tmp.replace(self.state_path)

    def it_dir(self, k: int) -> Path:
        d = self.dir / "iterations" / f"it{k}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def member(self, name: str) -> dict:
        return self.state["members"][name]

    def bundle(self, name: str) -> str:
        return os.path.relpath(self.dir / "bundles" / name, REPO)

    def strategic(self) -> list[str]:
        """The meta-game pool: anchors, promoted checkpoints and promoted exploiters, in order of entry."""
        return [n for n, m in self.state["members"].items() if m["status"] == "promoted"]

    def deployable(self) -> list[str]:
        return [n for n in self.strategic() if self.member(n)["kind"] in ("anchor", "learner_checkpoint")]

    # ------------------------------------------------------------------ abstract game

    def abstract(self):
        """The abstract league with every table loaded (numba)."""
        if self._abstract is not None:
            return self._abstract
        from strategy.blueprint.states import StatePool
        from strategy.league import tables as T
        from strategy.league.clone import Counts, model_table
        from strategy.league.learner import AbstractLeague

        pool, _ = StatePool.load(self.dir / "states" / "eval.npz")
        lg = AbstractLeague(pool, threads=self.cfg.threads, ruleset_id=self.cfg.ruleset_id)
        bp = T.pack_table(_pack(REPO / BLUEPRINT_PACK))
        lg.add("blueprint", bp)
        lg.add("uniform", T.uniform_table())
        for label in self.state["models"]:
            path = self.dir / "models" / f"{label}.npz"
            lg.add(f"model:{label}", model_table(Counts.load(path), bp, alpha=self.cfg.clone_alpha))
        for name, m in self.state["members"].items():
            if m.get("pack") and name != "blueprint":
                lg.add(name, T.pack_table(_pack(self.dir / m["pack"])))
        with np.load(self.dir / "eval-deals.npz") as z:
            deals = {k: z[k] for k in z.files}
        lg.set_eval(deals, pool)
        self._abstract = lg
        return lg

    def table_name(self, member: str) -> str:
        m = self.member(member)
        return f"model:{m['model']}" if m.get("model") else member

    def matrix(self, names: list[str]) -> dict:
        from strategy.league.learner import payoff_matrix

        cache_path = self.dir / "matrix-cache.json"
        cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        lg = self.abstract()
        tabs = [self.table_name(n) for n in names]
        out = payoff_matrix(lg, tabs, cache, log=self.log)
        cache_path.write_text(json.dumps(cache, indent=0))
        out["names"] = names
        return out

    def population_mixture(self) -> dict[int, float]:
        lg = self.abstract()
        return {lg.row(f"model:{m}"): 1.0 for m in self.state["models"] if m in self.state["population_models"]}

    def seat_mixture(self, sigma: dict[str, float]):
        from strategy.league.learner import SeatMixture

        lg = self.abstract()
        league = {lg.row(self.table_name(n)): p for n, p in sigma.items()}
        return SeatMixture.combine([(self.cfg.population, self.population_mixture()),
                                    (self.cfg.odd, {lg.row("uniform"): 1.0}),
                                    (1.0 - self.cfg.population - self.cfg.odd, league)])

    def train_pool(self, k: int):
        from strategy.blueprint.states import StatePool, pool_from_rows

        path = self.dir / "states" / f"it{k}.npz"
        if not path.exists():
            inc = self.state["incumbent"]
            self.log(f"it{k}: harvesting training states with {inc} seated")
            rows = harvest(self.cfg.ruleset_id, [self.bundle(inc)], self.cfg.train_states_lineups, seed=100 + k,
                           jobs=self.cfg.jobs)
            pool_from_rows(rows).save(path, {"incumbent": inc, "lineups": self.cfg.train_states_lineups,
                                             "partition": "train"})
        extra, _ = StatePool.load(path)
        base, _ = StatePool.load(self.dir / "states" / "eval.npz")
        return StatePool(**{a: np.concatenate([getattr(base, a), getattr(extra, a)]) for a in base.arrays()})

    # ------------------------------------------------------------------ members

    def add_member(self, name: str, kind: str, *, pack: Path | None, iteration: int, parent: str | None,
                   info: dict) -> None:
        from strategy.blueprint.package import build, smoke

        out = self.dir / "bundles" / name
        if pack is not None and not (out / "MANIFEST.json").exists():
            build(pack, out)
            smoke(out)
        slot = self.ledger.register(out, kind="approximate_response" if kind == "exploiter" else "learner_checkpoint",
                                    partition="train", rulesets=[self.cfg.ruleset_id], note=f"{self.cfg.run}:{name}")
        self.state["members"][name] = {
            "kind": "approximate_response" if kind == "exploiter" else "learner_checkpoint", "status": "candidate",
            "iteration": iteration, "parent": parent, "pack": os.path.relpath(pack, self.dir) if pack else None,
            "bundle": self.bundle(name), "slot": slot.slot_id, "info": info}
        self.save()

    def set_status(self, name: str, status: str, evidence: str) -> None:
        m = self.member(name)
        if status == "promoted":
            self.ledger.promote(m["slot"], evidence)
        else:
            self.ledger.retire(m["slot"], evidence)
        m["status"] = status
        m["evidence"] = evidence
        self.save()

    # ------------------------------------------------------------------ iteration

    def iterate(self) -> None:
        while len(self.state["iterations"]) < self.cfg.iterations:
            k = len(self.state["iterations"]) + 1
            self.iteration(k)

    def iteration(self, k: int) -> None:
        from strategy.league import meta
        from strategy.league import tables as T

        d = self.it_dir(k)
        inc = self.state["incumbent"]
        self.log(f"=== iteration {k}: incumbent {inc}")
        lg = self.abstract()
        lg.pool = self.train_pool(k)
        inc_tab = self.table_name(inc)

        # 2. exploiter
        ex_path = d / "exploiter.json"
        if not ex_path.exists():
            prior = [n for n, m in self.state["members"].items()
                     if m["kind"] == "approximate_response" and (m.get("info") or {}).get("target") == inc]
            if prior:
                rec = {"reused": prior[0], "target": inc}
                self.log(f"it{k}: incumbent unchanged; exploiter {prior[0]} reused")
            else:
                from strategy.league.learner import SeatMixture, homogeneous

                name = f"E{k}"
                self.log(f"it{k}: training exploiter {name} against {inc} ({self.cfg.exploiter_pairs:,} pairs)")
                regret, sq, stats = lg.train_response(inc_tab, SeatMixture.pure(lg.row(inc_tab)),
                                                      pairs=self.cfg.exploiter_pairs, z=self.cfg.exploiter_z,
                                                      seed=2000 + k, producers=self.cfg.producers, log=self.log)
                base = lg.tabs[lg.row(inc_tab)]
                dev = T.deviations(regret, sq, base, z=self.cfg.exploiter_z, mix=1.0)
                del regret, sq
                lg.add(name, T.apply_deviations(base, dev))
                seats = homogeneous(lg.row(inc_tab), lg._eval["count"])
                gain = lg.gain(name, inc_tab, seats)
                pack = self.dir / "packs" / f"{name}.npz"
                T.write_override_pack(self._pack_path(inc), dev, pack,
                                      {"ruleset_id": self.cfg.ruleset_id, "league": self.cfg.run, "member": name,
                                       "kind": "exploiter", "target": inc, "train": stats})
                rec = {"name": name, "target": inc, "train": stats, "deviations": len(dev),
                       "by_street": dev.by_street(), "by_action": dev.by_action(), "gain_vs_target": gain}
                self.add_member(name, "exploiter", pack=pack, iteration=k, parent=inc, info={"target": inc, **rec})
                lo = gain["gain_points"] - 1.96 * gain["se"]
                self.set_status(name, "promoted" if lo > 0 else "retired",
                                f"abstract held-out gain over {inc}: {gain['gain_points']:+.4f} ± {gain['se']:.4f}")
                rec["status"] = self.member(name)["status"]
                self.log(f"it{k}: {name} gain over {inc} {gain['gain_points']:+.4f} ± {gain['se']:.4f} "
                         f"({len(dev):,} deviations) -> {rec['status']}")
            ex_path.write_text(json.dumps(rec, indent=1))

        # 3. meta-solve
        meta_path = d / "meta.json"
        if not meta_path.exists():
            names = self.strategic()
            self.log(f"it{k}: payoff matrix over {names}")
            G = self.matrix(names)
            sigma = meta.solve(G["gain"], iters=self.cfg.meta_iters, floor=self.cfg.sigma_floor)
            raw = meta.solve(G["gain"], iters=self.cfg.meta_iters)
            sig = {n: float(p) for n, p in zip(names, sigma)}
            exact = self.exact_meta(names, sig)
            rec = {"names": names, "matrix": G, "sigma": sig, "sigma_raw": dict(zip(names, raw.tolist())),
                   "nash_conv": meta.nash_conv(G["gain"], sigma), "nash_conv_raw": meta.nash_conv(G["gain"], raw),
                   "values": dict(zip(names, meta.values(G["gain"], sigma).tolist())),
                   "cycles": meta.cycles(G["gain"], G["se"], names), "ranking": meta.ranking(G["gain"], names),
                   "exact": exact}
            meta_path.write_text(json.dumps(rec, indent=1))
            self.log(f"it{k}: sigma {json.dumps({n: round(p, 3) for n, p in sig.items()})}; "
                     f"NashConv {rec['nash_conv']:.4f} (raw {rec['nash_conv_raw']:.4f}); cycles {rec['cycles']}")
        meta_rec = json.loads(meta_path.read_text())

        # 4. learner
        learn_path = d / "learner.json"
        if not learn_path.exists():
            self.learner(k, meta_rec["sigma"], learn_path)
        learn = json.loads(learn_path.read_text())

        # 5. gates
        gate_path = d / "gates.json"
        if not gate_path.exists():
            if not learn["screen"]["passed"]:
                rec = {"skipped": True, "reason": learn["screen"]["reason"], "promote": False}
            else:
                rec = self.gates(k, learn["name"])
            gate_path.write_text(json.dumps(rec, indent=1))
        gates = json.loads(gate_path.read_text())

        # 6. promotion
        name = learn["name"]
        if self.member(name)["status"] == "candidate":
            if gates["promote"]:
                self.set_status(name, "promoted", f"league {self.cfg.run} it{k} gates: "
                                + ", ".join(g["manifest_id"] for g in gates["runs"].values() if g.get("manifest_id")))
                self.state["incumbent"] = name
            else:
                why = gates.get("reason") or "; ".join(c["check"] for c in gates["decision"]["checks"]
                                                       if not c["passed"])
                self.set_status(name, "retired", f"it{k} not promoted: {why}")
        self.state["iterations"].append({"k": k, "incumbent_before": inc, "candidate": name,
                                         "promoted": self.member(name)["status"] == "promoted",
                                         "incumbent_after": self.state["incumbent"]})
        self.save()
        self.log(f"=== iteration {k} done: {name} {self.member(name)['status']}; incumbent {self.state['incumbent']}")

    def _pack_path(self, name: str) -> Path:
        m = self.member(name)
        return self.dir / m["pack"] if m.get("pack") else REPO / BLUEPRINT_PACK

    def exact_meta(self, names: list[str], sigma: dict[str, float]) -> dict:
        """Each member's gain over the sigma-average against seats drawn from sigma (no linearisation)."""
        from strategy.league.learner import SeatMixture

        lg = self.abstract()
        rows = {lg.row(self.table_name(n)): p for n, p in sigma.items()}
        mix = SeatMixture.combine([(1.0, rows)])
        seats = lg.seat_draws(mix, tag=11)
        vals = {n: lg.values(self.table_name(n), seats) for n in names}
        from strategy.blueprint.response import FIRST_PRIZE_POINTS

        mean = {n: float(np.nanmean(v)) * FIRST_PRIZE_POINTS for n, v in vals.items()}
        avg = sum(sigma[n] * mean[n] for n in names)
        dev = {n: mean[n] - avg for n in names}
        return {"value_minus_sigma_average": dev, "nash_conv_exact": max(dev.values())}

    def learner(self, k: int, sigma: dict[str, float], out: Path) -> None:
        from strategy.league import tables as T
        from strategy.league.learner import homogeneous

        lg = self.abstract()
        inc = self.state["incumbent"]
        inc_tab = self.table_name(inc)
        mix = self.seat_mixture(sigma)
        name = f"C{k}"
        self.log(f"it{k}: training learner {name} from {inc} against the seat mixture "
                 f"({self.cfg.learner_pairs:,} pairs)")
        regret, sq, stats = lg.train_response(inc_tab, mix, pairs=self.cfg.learner_pairs,
                                              z=min(self.cfg.learner_z), seed=1000 + k,
                                              producers=self.cfg.producers, log=self.log)
        base = lg.tabs[lg.row(inc_tab)]
        count = lg._eval["count"]
        mix_seats = lg.seat_draws(mix, tag=100 + k)
        pop_seats = lg.seat_draws(_mixture(self.population_mixture()), tag=7)
        envs = {"mixture": mix_seats, "population": pop_seats}
        for n in self.strategic():
            envs[f"vs:{n}"] = homogeneous(lg.row(self.table_name(n)), count)
        base_vals = {e: lg.values(inc_tab, s) for e, s in envs.items()}
        grid = []
        best = None
        for z in self.cfg.learner_z:
            for m in self.cfg.learner_mix:
                dev = T.deviations(regret, sq, base, z=z, mix=m)
                tag = f"{name}@z{z}m{m}"
                np.savez_compressed(out.parent / f"deviations-z{z}m{m}.npz", keys=dev.keys, rows=dev.rows,
                                    best=dev.best)
                lg.add(tag, T.apply_deviations(base, dev))
                from strategy.blueprint.response import paired_gain

                gains = {e: paired_gain(base_vals[e], lg.values(tag, s)) for e, s in envs.items()}
                worst = min(g["gain_points"] for e, g in gains.items() if e != "mixture")
                row = {"z": z, "mix": m, "deviations": len(dev), "by_street": dev.by_street(),
                       "by_action": dev.by_action(), "gains": gains, "worst": worst,
                       "admissible": worst >= -self.cfg.abstract_tolerance}
                grid.append(row)
                self.log(f"it{k}: {tag}: {len(dev):,} deviations; mixture {gains['mixture']['gain_points']:+.4f} "
                         f"± {gains['mixture']['se']:.4f}; population {gains['population']['gain_points']:+.4f}; "
                         f"worst {worst:+.4f}")
                key = (row["admissible"], gains["mixture"]["gain_points"] if row["admissible"] else worst)
                if best is None or key > best[0]:
                    best = (key, row, dev)
        del regret, sq
        assert best is not None
        _, chosen, dev = best
        g = chosen["gains"]["mixture"]
        lo = g["gain_points"] - 1.96 * g["se"]
        screen = {"passed": bool(chosen["admissible"] and lo > 0),
                  "reason": None if chosen["admissible"] and lo > 0 else
                  f"abstract screen: held-out mixture gain {g['gain_points']:+.4f} ± {g['se']:.4f}, "
                  f"admissible {chosen['admissible']}"}
        pack = self.dir / "packs" / f"{name}.npz"
        T.write_override_pack(self._pack_path(inc), dev, pack,
                              {"ruleset_id": self.cfg.ruleset_id, "league": self.cfg.run, "member": name,
                               "kind": "learner", "parent": inc, "z": chosen["z"], "mix": chosen["mix"],
                               "sigma": sigma, "train": stats})
        lg.add(name, T.apply_deviations(base, dev))
        rec = {"name": name, "parent": inc, "sigma": sigma, "seat_mixture": {lg.names[r]: p for r, p in
                                                                            zip(mix.rows, mix.probs)},
               "train": stats, "grid": grid, "chosen": {"z": chosen["z"], "mix": chosen["mix"]}, "screen": screen}
        self.add_member(name, "learner", pack=pack, iteration=k, parent=inc, info={"z": chosen["z"],
                                                                                   "mix": chosen["mix"]})
        out.write_text(json.dumps(rec, indent=1))
        self.log(f"it{k}: chose z {chosen['z']} mix {chosen['mix']}; screen {screen}")

    def gates(self, k: int, name: str) -> dict:
        from strategy.league import gates as Gt

        inc = self.state["incumbent"]
        spec_dir = self.dir / "specs"
        cand = (name, self.bundle(name))
        runs: dict[str, dict] = {}
        path = Gt.write_spec(Gt.improve_spec(self.cfg.run, k, (inc, self.bundle(inc)), cand, self.rule), spec_dir)
        self.log(f"it{k}: gate improve ({path.name})")
        runs["improve"] = Gt.run(path, jobs=self.cfg.jobs)
        self.log(f"it{k}: improve {_fmt(runs['improve'])}")
        earlier = self.deployable()
        exploiters = [n for n in self.strategic() if self.member(n)["kind"] == "approximate_response"]
        if exploiters:
            earlier.append(exploiters[-1])
        regress = {}
        for j in earlier:
            path = Gt.write_spec(Gt.regress_spec(self.cfg.run, k, (j, self.bundle(j)), cand, self.rule), spec_dir)
            self.log(f"it{k}: gate regress-{j}")
            regress[j] = runs[f"regress-{j}"] = Gt.run(path, jobs=self.cfg.jobs)
            self.log(f"it{k}: regress-{j} {_fmt(regress[j])}")
        decision = Gt.decide(self.rule, runs["improve"], regress)
        for c in decision["checks"]:
            self.log(f"it{k}:   {'PASS' if c['passed'] else 'FAIL'} {c['check']}: {c['detail']}")
        return {"runs": runs, "decision": decision, "promote": decision["promote"]}


def _fmt(r: dict) -> str:
    if r.get("withheld"):
        return f"withheld ({r.get('reason')})"
    if r.get("diff") is None or r.get("lo") is None:
        return f"no estimate over {r.get('blocks')} blocks; verdict {r.get('verdict')}; gates_ok {r.get('gates_ok')}"
    wf = r.get("worst_family") or {}
    worst = f"{wf['family']} {wf['mean']:+.2f}" if wf.get("mean") is not None else "n/a"
    return (f"{r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] over {r['blocks']} blocks; worst family "
            f"{worst}; verdict {r['verdict']}")


def _mixture(rows: dict[int, float]):
    from strategy.league.learner import SeatMixture

    return SeatMixture.combine([(1.0, rows)])


def _pack(path: Path):
    from strategy.blueprint.policy import PolicyPack

    return PolicyPack(path)


def harvest(ruleset_id: str, fixed: list[str], lineups: int, *, seed: int, jobs: int) -> list[dict]:
    """Start-of-hand state rows from training lineups (``fixed`` seats, then distinct training families)."""
    from arena.domain.models import TournamentConfig
    from strategy.blueprint.states import state_row
    from strategy.league.clone import training_lineups
    from strategy.value.rollouts import run_lineups

    teams, _ = training_lineups(ruleset_id, lineups, seed, fixed=fixed)
    config = TournamentConfig(ruleset_id=ruleset_id)
    rows = []
    for ro in run_lineups(config, teams, blocks=1, base_seed=f"league-states-{seed}", jobs=jobs):
        for h in ro.hands:
            r = state_row(ro.ring, h.start_stacks, h.button, h.small_blind_seat, h.big_blind, h.heads_up, h.blinds,
                          ro.payouts, h.hand_number, ro.hand_cap)
            if r is not None:
                rows.append(r)
    return rows


# ------------------------------------------------------------------ init


def init(run_dir: Path, cfg: Config) -> None:
    from strategy.blueprint.deals import make_deals
    from strategy.blueprint.package import build, smoke
    from strategy.blueprint.states import pool_from_rows
    from strategy.league import clone
    from strategy.league.learner import EVAL_SEED
    from strategy.policy.package import build_bundle, load_params
    from strategy.population.registry import CheckpointLedger

    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "config.json"
    text = json.dumps(asdict(cfg), indent=1) + "\n"
    if cfg_path.exists() and cfg_path.read_text() != text:
        raise SystemExit(f"{cfg_path} exists with a different config; a league config is frozen at init")
    cfg_path.write_text(text)
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)  # noqa: E731
    bundles = run_dir / "bundles"
    if not (bundles / "blueprint" / "MANIFEST.json").exists():
        log("building the blueprint anchor bundle")
        build(REPO / BLUEPRINT_PACK, bundles / "blueprint")
        smoke(bundles / "blueprint")
    if not (bundles / "baseline" / "manifest.json").exists():
        log("building the baseline anchor bundle")
        build_bundle(bundles / "baseline", load_params(REPO / BASELINE_PARAMS), name="strategy_baseline")
    ledger = CheckpointLedger(run_dir / "ledger.json")
    members = {}
    for name, note in (("blueprint", "M5 blueprint (main-v3 final pack)"), ("baseline", "M3 CEM baseline nlhe-v2")):
        slot = ledger.register(bundles / name, kind="learner_checkpoint", partition="train",
                               rulesets=[cfg.ruleset_id], note=note)
        if ledger.slots[slot.slot_id].status != "promoted":
            ledger.promote(slot.slot_id, f"anchor: {note}, promoted by its own milestone")
        members[name] = {"kind": "anchor", "status": "promoted", "iteration": 0, "parent": None,
                         "pack": None, "bundle": os.path.relpath(bundles / name, REPO), "slot": slot.slot_id,
                         "model": "baseline" if name == "baseline" else None, "info": {"note": note}}
    models_dir = run_dir / "models"
    summary_path = models_dir / "summary.json"
    if not summary_path.exists():
        log(f"cloning the training population and the baseline ({cfg.clone_lineups} lineups)")
        bp = os.path.relpath(bundles / "blueprint", REPO)
        bl = os.path.relpath(bundles / "baseline", REPO)
        lus, lbs = clone.training_lineups(cfg.ruleset_id, cfg.clone_lineups, cfg.clone_seed, fixed=[bp, bl],
                                          fixed_labels=[None, "baseline"])
        t0 = time.time()
        counts, stats = clone.record(lus, lbs, blocks=cfg.clone_blocks, base_seed=f"league-clone-{cfg.clone_seed}",
                                     jobs=cfg.jobs, ruleset_id=cfg.ruleset_id)
        for label, c in counts.items():
            c.save(models_dir / f"{label}.npz")
        summary = {"stats": stats, "seconds": round(time.time() - t0, 1), "lineups": cfg.clone_lineups,
                   "models": {lbl: {"decisions": c.decisions, "keys": int(c.keys.size),
                                    "by_street": c.street_decisions(), "actions": c.action_shares()}
                              for lbl, c in counts.items()}}
        summary_path.write_text(json.dumps(summary, indent=1))
        log(f"cloned {len(counts)} models: {stats}")
    summary = json.loads(summary_path.read_text())
    states = run_dir / "states" / "eval.npz"
    if not states.exists():
        log(f"harvesting the evaluation state pool ({cfg.eval_states_lineups} training lineups, blueprint seated)")
        rows = harvest(cfg.ruleset_id, [os.path.relpath(bundles / "blueprint", REPO)], cfg.eval_states_lineups,
                       seed=1, jobs=cfg.jobs)
        pool_from_rows(rows).save(states, {"lineups": cfg.eval_states_lineups, "partition": "train",
                                           "fixed": ["blueprint"]})
    deals = run_dir / "eval-deals.npz"
    if not deals.exists():
        log(f"generating {cfg.eval_deals:,} held-out deals")
        np.savez_compressed(deals, **make_deals(cfg.ruleset_id, EVAL_SEED, cfg.eval_deals))
    state_path = run_dir / "league.json"
    if not state_path.exists():
        state = {"run": cfg.run, "incumbent": "blueprint", "members": members, "iterations": [],
                 "models": sorted(summary["models"]),
                 "population_models": sorted(m for m in summary["models"] if m != "baseline")}
        state_path.write_text(json.dumps(state, indent=1) + "\n")
    log(f"initialised {run_dir}")


# ------------------------------------------------------------------ curve, select


def curve(lg: League, *, lineups: int = 8, blocks: int = 12) -> dict:
    """Every promoted league checkpoint against the blueprint on the validation population."""
    from strategy.league import gates as Gt

    out = {}
    for name in lg.deployable():
        if lg.member(name)["kind"] != "learner_checkpoint":
            continue
        spec = Gt._spec(f"league-{lg.cfg.run}-curve-{name}", f"League {lg.cfg.run} learning curve: {name} vs blueprint",
                        "Learning curve (discovery): A = blueprint anchor, B = league checkpoint; validation.",
                        lg.bundle("blueprint"), lg.bundle(name), base_seed=f"league-{lg.cfg.run}-curve-{name}",
                        stage="discovery", lineups=lineups, blocks=blocks)
        path = Gt.write_spec(spec, lg.dir / "specs")
        lg.log(f"curve: {name}")
        out[name] = Gt.run(path, jobs=lg.cfg.jobs)
        lg.log(f"curve: {name} {_fmt(out[name])}")
    (lg.dir / "curve.json").write_text(json.dumps(out, indent=1))
    return out


def select(lg: League) -> dict:
    """Deployment choice (declared rule): among promoted league checkpoints whose antisymmetric advantage
    over every other strategic pool member is at least ``-abstract_tolerance``, the one with the highest
    learning-curve score against the blueprint on validation; the blueprint when none qualifies."""
    from strategy.league import meta

    names = lg.strategic()
    G = lg.matrix(names)
    H = meta.antisymmetric(G["gain"])
    cv = json.loads((lg.dir / "curve.json").read_text()) if (lg.dir / "curve.json").exists() else {}
    rows = []
    for i, n in enumerate(names):
        if lg.member(n)["kind"] != "learner_checkpoint":
            continue
        others = [j for j in range(len(names)) if j != i]
        jw = min(others, key=lambda j: H[i][j]) if others else i
        worst = float(H[i][jw]) if others else 0.0
        c = cv.get(n, {})
        rows.append({"name": n, "worst_h": worst, "worst_vs": names[jw],
                     "h": {names[j]: float(H[i][j]) for j in others},
                     "eligible": bool(worst >= -lg.cfg.abstract_tolerance),
                     "curve": c.get("diff"), "curve_lo": c.get("lo")})
    eligible = [r for r in rows if r["eligible"] and r["curve"] is not None]
    choice = max(eligible, key=lambda r: r["curve"])["name"] if eligible else "blueprint"
    out = {"matrix": G, "rows": rows, "selected": choice}
    (lg.dir / "selection.json").write_text(json.dumps(out, indent=1))
    lg.log(f"selected {choice}: {rows}")
    return out


CONFIRM_MARGIN = 1.0
"""Non-inferiority margin (points per tournament) of the pre-registered regression confirmations."""


def confirm_specs(lg: League, selected: str) -> dict[str, tuple[dict, float]]:
    """The pre-registered confirmation family for ``selected`` (docs/league-training-preregistration.md).

    ``vs-blueprint``: superiority over the blueprint on the validation population (400 blocks).
    ``ni-<j>``: non-inferiority (margin ``CONFIRM_MARGIN``) to every other deployable pool member
    ``j`` in ``j``-rich lineups (200 blocks each). Fresh seeds; validation only.
    """
    from strategy.league import gates as Gt

    run = lg.cfg.run
    cand = (selected, lg.bundle(selected))
    out = {"vs-blueprint": (Gt._spec(
        f"league-{run}-confirm-vs-blueprint", f"League {run} confirmation: {selected} vs blueprint (validation)",
        f"Pre-registered confirmation. A = blueprint, B = {selected}; validation population, fresh seeds.",
        lg.bundle("blueprint"), cand[1], base_seed=f"league-{run}-confirm-vs-blueprint", stage="confirmation",
        lineups=40, blocks=10), 0.0)}
    for j in lg.deployable():
        if j == selected:
            continue
        spec = Gt.regress_spec(run, 0, (j, lg.bundle(j)), cand, lg.rule, tag="-confirm", stage="confirmation",
                               lineups=25, blocks=8)
        spec["experiment_id"] = spec["base_seed"] = f"league-{run}-confirm-ni-{j}"
        spec["title"] = f"League {run} confirmation: {selected} non-inferior to {j} in {j}-rich lineups"
        out[f"ni-{j}"] = (spec, -CONFIRM_MARGIN)
    return out


def confirm(lg: League, name: str | None = None) -> dict:
    """The confirmation family for the selection, or for ``name`` (a declared amendment, own result file)."""
    from strategy.league import gates as Gt

    selected = name or json.loads((lg.dir / "selection.json").read_text())["selected"]
    result = lg.dir / (f"confirm-{name}.json" if name else "confirm.json")
    if lg.member(selected)["kind"] != "learner_checkpoint":
        out = {"selected": selected, "skipped": True,
               "reason": "no league checkpoint selected: the confirmations are not run (a null result)"}
        result.write_text(json.dumps(out, indent=1))
        return out
    runs, rows = {}, {}
    for key, (spec, margin) in confirm_specs(lg, selected).items():
        path = Gt.write_spec(spec, lg.dir / "specs")
        lg.log(f"confirm: {key} ({path.name})")
        runs[key] = r = Gt.run(path, jobs=lg.cfg.jobs)
        lg.log(f"confirm: {key} {_fmt(r)}")
        rows[key] = {"diff": r.get("diff"), "sd": r.get("sd"), "blocks": r.get("blocks"), "margin": margin,
                     "lo": r.get("lo"), "hi": r.get("hi"), "gates_ok": r.get("gates_ok")}
    family = Gt.holm_family({k: v for k, v in rows.items() if v["diff"] is not None})
    out = {"selected": selected, "amendment": bool(name), "runs": runs, "holm": family, "alpha": 0.05}
    result.write_text(json.dumps(out, indent=1))
    for k, v in family.items():
        lg.log(f"confirm: {k}: {v['diff']:+.3f} (margin {v['margin']:+g}) p {v['p_one_sided']:.4g} "
               f"Holm {v['p_holm']:.4g} -> {'confirmed' if v['confirmed'] else 'not confirmed'}")
    return out


def export(lg: League, name: str) -> dict:
    """Copy the selected checkpoint's pack to the committed table store for the deployed league bot."""
    m = lg.member(name)
    if m["kind"] != "learner_checkpoint" or not m.get("pack"):
        raise SystemExit(f"{name} is not a league checkpoint; nothing to export")
    src = lg.dir / m["pack"]
    dst = REPO / "strategy" / "tables" / "data" / f"league-{lg.cfg.ruleset_id}.npz"
    shutil.copyfile(src, dst)
    pk = _pack(dst)
    return {"name": name, "pack": str(dst.relative_to(REPO)), "data_sha256": pk.provenance["data_sha256"],
            "bytes": dst.stat().st_size, "rows": pk.provenance["rows"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("init", "iterate", "curve", "select", "confirm", "export", "status"))
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--iterations", type=int)
    ap.add_argument("--learner-pairs", type=int)
    ap.add_argument("--exploiter-pairs", type=int)
    ap.add_argument("--clone-lineups", type=int)
    ap.add_argument("--eval-deals", type=int)
    ap.add_argument("--name", help="export/confirm: the checkpoint (default: the selection)")
    ap.add_argument("--smoke", action="store_true", help="init: a tiny league for checking the plumbing")
    a = ap.parse_args(argv)
    if a.command == "init":
        from strategy.league.gates import Rule

        cfg = Config(run=a.run.name)
        cfg.rule = asdict(Rule())
        if a.smoke:
            cfg = Config(run=a.run.name, iterations=2, clone_lineups=8, eval_states_lineups=3, train_states_lineups=2,
                         eval_deals=1500, learner_pairs=40000, exploiter_pairs=20000,
                         rule=asdict(Rule(improve_lineups=2, improve_blocks=1, regress_lineups=1, regress_blocks=1)))
        for k in ("iterations", "learner_pairs", "exploiter_pairs", "clone_lineups", "eval_deals"):
            if getattr(a, k) is not None:
                setattr(cfg, k, getattr(a, k))
        init(a.run, cfg)
        return 0
    lg = League(a.run)
    if a.command == "iterate":
        lg.iterate()
    elif a.command == "curve":
        curve(lg)
    elif a.command == "select":
        select(lg)
    elif a.command == "confirm":
        confirm(lg, a.name)
    elif a.command == "export":
        name = a.name or json.loads((lg.dir / "selection.json").read_text())["selected"]
        print(json.dumps(export(lg, name), indent=1))
    else:
        json.dump({"incumbent": lg.state["incumbent"], "iterations": lg.state["iterations"],
                   "members": {n: {k: m.get(k) for k in ("kind", "status", "iteration", "parent", "evidence")}
                               for n, m in lg.state["members"].items()}}, sys.stdout, indent=1)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
