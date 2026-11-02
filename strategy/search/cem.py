"""Parameter search over the baseline policy: CEM (default) or CMA-ES, with a kept incumbent.

Each generation:

1. draws a fresh task set (seeds, rotations, train-partition lineups, mirror seats);
2. evaluates the incumbent, the distribution mean and ``population`` samples (mirrored
   pairs around the mean) on *the same* tasks (common random numbers);
3. updates the search distribution from the elite fraction (CEM: smoothed diagonal
   Gaussian with a floor; CMA-ES: the standard rank-one and rank-mu updates);
4. re-evaluates the best vector of the generation against the incumbent on a second,
   fresh task set and replaces the incumbent only when the paired gain there has
   ``t >= accept_t`` (selection and confirmation never share tables, so a lucky
   generation cannot promote itself).

All search happens in the unit cube of :data:`strategy.policy.baseline.SPACE`. Every
evaluated vector is persisted with its parameter values, the hash of its task set,
the generation's population hash, the registry hash and every per-table point result.
The run resumes from ``state.json`` at generation granularity.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from strategy.policy.baseline import SPACE
from strategy.policy.params import vector_hash
from strategy.rules.canonical import canonical_hash
from strategy.search.evaluate import Evaluator, Task, make_tasks, paired, seed_set_hash, summarize
from strategy.search.population import DEFAULT_SOURCE, SEARCH_PARTITIONS, load_population, population_digest

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = ROOT / "artifacts" / "search"


@dataclass
class SearchConfig:
    run_id: str
    algorithm: str = "cem"
    generations: int = 30
    population: int = 24
    elite_frac: float = 0.25
    tables_per_gen: int = 48
    confirm_tables: int = 96
    init_sigma: float = 0.12
    sigma_floor: float = 0.02
    smoothing: float = 0.6
    accept_t: float = 1.28
    challenger: str = "best"
    """Which vector the confirmation tests: ``best`` (the top-scoring vector of the
    generation) or ``mean`` (the distribution mean, whose selection score is unbiased
    because it was fixed before the generation's tables were drawn)."""
    mirror_rate: float = 0.15
    source: str = DEFAULT_SOURCE
    partition: str = "train"
    ruleset_id: str = "practice_nlhe"
    players: int = 8
    starting_stack: int = 2000
    hand_cap: int = 200
    seed: str = "baseline-search-v1"
    start: dict[str, float] | None = None
    """Initial incumbent and mean (defaults: the hand-written control)."""
    frozen: list[str] = field(default_factory=list)
    """Parameter names held at their start value."""
    resume_from: str | None = None
    """Another run directory: start from its incumbent and search distribution."""
    structure: dict[str, Any] | None = None
    """Extra ``TournamentConfig`` fields (blind levels, hands per level, payouts) for an
    announced structure; ``players``, ``starting_stack`` and ``hand_cap`` still set those."""

    def __post_init__(self) -> None:
        if self.challenger not in ("best", "mean"):
            raise ValueError("challenger must be best or mean")
        if self.algorithm not in ("cem", "cmaes"):
            raise ValueError("algorithm must be cem or cmaes")
        if self.population < 4 or self.population % 2:
            raise ValueError("population must be an even number >= 4 (mirrored sampling)")
        if self.partition not in SEARCH_PARTITIONS:
            raise ValueError(f"the search trains only on {SEARCH_PARTITIONS}; validation is for promotion")
        unknown = set(self.frozen) - set(SPACE.names)
        if unknown:
            raise ValueError(f"unknown frozen parameters {sorted(unknown)}")
        clash = {"ruleset_id", "initial_player_count", "starting_stack", "hand_cap"} & set(self.structure or {})
        if clash:
            raise ValueError(f"set {sorted(clash)} through the search config fields, not structure")

    def table_config(self) -> Any:
        from arena.domain.models import TournamentConfig

        return TournamentConfig(ruleset_id=self.ruleset_id, initial_player_count=self.players,
                                starting_stack=self.starting_stack, hand_cap=self.hand_cap, **(self.structure or {}))

    def header_config(self) -> dict[str, Any]:
        """The recorded config; an unset ``structure`` is left out so older runs still resume."""
        out = asdict(self)
        if out["structure"] is None:
            del out["structure"]
        return out

    @classmethod
    def load(cls, path: Path | str) -> SearchConfig:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


# ----------------------------------------------------------------- strategies


class CEM:
    def __init__(self, mean: np.ndarray, sigma: float, cfg: SearchConfig, free: np.ndarray) -> None:
        self.mean = mean.copy()
        self.sigma = np.full_like(mean, sigma)
        self.cfg = cfg
        self.free = free

    def ask(self, rng: np.random.Generator) -> np.ndarray:
        half = self.cfg.population // 2
        eps = rng.standard_normal((half, self.mean.size)) * self.free
        z = np.concatenate([self.mean + self.sigma * eps, self.mean - self.sigma * eps])
        return np.clip(z, 0.0, 1.0)

    def tell(self, z: np.ndarray, scores: Sequence[float]) -> None:
        k = max(2, int(round(self.cfg.elite_frac * len(scores))))
        elite = z[np.argsort(scores)[::-1][:k]]
        a = self.cfg.smoothing
        self.mean = (1 - a) * self.mean + a * elite.mean(axis=0)
        sd = elite.std(axis=0, ddof=1)
        self.sigma = np.maximum(self.cfg.sigma_floor, (1 - a) * self.sigma + a * sd) * self.free + \
            self.sigma * (1 - self.free)

    def state(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "sigma": self.sigma.tolist()}

    def load(self, s: Mapping[str, Any]) -> None:
        self.mean = np.array(s["mean"])
        self.sigma = np.array(s["sigma"])


class CMAES:
    """Standard (mu/mu_w, lambda) CMA-ES (Hansen's tutorial defaults) on the free coordinates."""

    def __init__(self, mean: np.ndarray, sigma: float, cfg: SearchConfig, free: np.ndarray) -> None:
        self.full = mean.copy()
        self.idx = np.flatnonzero(free)
        n = self.n = self.idx.size
        self.m = mean[self.idx].copy()
        self.sigma = sigma
        self.cfg = cfg
        lam = cfg.population
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / np.sum(self.w ** 2)
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0.0, np.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chin = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.C = np.eye(n)
        self.gen = 0
        self._y: np.ndarray | None = None

    def _decomp(self) -> tuple[np.ndarray, np.ndarray]:
        vals, vecs = np.linalg.eigh((self.C + self.C.T) / 2)
        return vecs, np.sqrt(np.maximum(vals, 1e-20))

    def ask(self, rng: np.random.Generator) -> np.ndarray:
        B, D = self._decomp()
        half = self.cfg.population // 2
        eps = rng.standard_normal((half, self.n))
        eps = np.concatenate([eps, -eps])
        y = eps @ (B * D).T
        self._y = y
        x = self.m + self.sigma * y
        z = np.tile(self.full, (len(x), 1))
        z[:, self.idx] = x
        return np.clip(z, 0.0, 1.0)

    def tell(self, z: np.ndarray, scores: Sequence[float]) -> None:
        mu = self.w.size
        order = np.argsort(scores)[::-1][:mu]
        x = z[:, self.idx]
        y = (x[order] - self.m) / self.sigma
        yw = self.w @ y
        self.m = self.m + self.sigma * yw
        B, D = self._decomp()
        inv_sqrt = (B / D) @ B.T
        self.gen += 1
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * (inv_sqrt @ yw)
        hs = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * self.gen)) / self.chin < 1.4 + 2 / (self.n + 1)
        self.pc = (1 - self.cc) * self.pc + hs * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        rank_mu = (y.T * self.w) @ y
        self.C = ((1 - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pc, self.pc) +
                  (1 - hs) * self.cc * (2 - self.cc) * self.C) + self.cmu * rank_mu)
        self.sigma *= float(np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chin - 1)))
        self.sigma = max(self.sigma, self.cfg.sigma_floor)

    @property
    def mean(self) -> np.ndarray:
        z = self.full.copy()
        z[self.idx] = np.clip(self.m, 0.0, 1.0)
        return z

    def state(self) -> dict[str, Any]:
        return {"m": self.m.tolist(), "sigma": self.sigma, "C": self.C.tolist(), "pc": self.pc.tolist(),
                "ps": self.ps.tolist(), "gen": self.gen}

    def load(self, s: Mapping[str, Any]) -> None:
        self.m, self.sigma = np.array(s["m"]), float(s["sigma"])
        self.C, self.pc, self.ps = np.array(s["C"]), np.array(s["pc"]), np.array(s["ps"])
        self.gen = int(s["gen"])


# --------------------------------------------------------------------- store


class RunStore:
    """Append-only records of one search run. Nothing is overwritten except ``state.json``."""

    def __init__(self, run_dir: Path) -> None:
        self.dir = run_dir
        (run_dir / "vectors").mkdir(parents=True, exist_ok=True)
        (run_dir / "tasks").mkdir(exist_ok=True)

    def vector(self, params: Mapping[str, float]) -> str:
        h = vector_hash(params)
        path = self.dir / "vectors" / f"{h}.json"
        if not path.exists():
            path.write_text(json.dumps({"vector_hash": h, "params": dict(params)}, sort_keys=True) + "\n",
                            encoding="ascii")
        return h

    def tasks(self, tasks: Sequence[Task]) -> str:
        h = seed_set_hash(tasks)
        path = self.dir / "tasks" / f"{h}.json"
        if not path.exists():
            path.write_text(json.dumps([{"seed": t.seed, "rotation": t.rotation, "opponents": list(t.opponents),
                                         "families": list(t.families)} for t in tasks]) + "\n", encoding="ascii")
        return h

    def append(self, name: str, row: Mapping[str, Any]) -> None:
        with open(self.dir / name, "a", encoding="ascii") as f:
            f.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")

    def save_state(self, state: Mapping[str, Any]) -> None:
        tmp = self.dir / "state.json.tmp"
        tmp.write_text(json.dumps(state, sort_keys=True) + "\n", encoding="ascii")
        os.replace(tmp, self.dir / "state.json")

    def load_state(self) -> dict[str, Any] | None:
        path = self.dir / "state.json"
        return json.loads(path.read_text(encoding="ascii")) if path.exists() else None


# ----------------------------------------------------------------------- run


def _round(params: Mapping[str, float]) -> dict[str, float]:
    return {k: round(float(v), 6) for k, v in params.items()}


def run_search(cfg: SearchConfig, *, root: Path = DEFAULT_RUN_ROOT, jobs: int | None = None,
               log: Callable[[str], None] = print) -> Path:
    run_dir = root / cfg.run_id
    store = RunStore(run_dir)
    source = load_population(cfg.source)
    digest = population_digest(source, cfg.partition)
    header = {"config": cfg.header_config(), "space_hash": SPACE.space_hash(), "dimensions": len(SPACE),
              "population": digest, "git_commit": _git_commit()}
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        prev = json.loads(cfg_path.read_text(encoding="ascii"))
        same = {k: v for k, v in prev["config"].items() if k != "generations"} == \
            {k: v for k, v in header["config"].items() if k != "generations"}
        if not same or prev["space_hash"] != header["space_hash"]:
            raise RuntimeError(f"{run_dir} holds a different search; choose a new run_id")
    else:
        cfg_path.write_text(json.dumps(header, sort_keys=True, indent=1) + "\n", encoding="ascii")

    start = SPACE.complete(cfg.start)
    z0 = np.array(SPACE.to_unit(start))
    free = np.array([0.0 if n in cfg.frozen else 1.0 for n in SPACE.names])
    strat: CEM | CMAES = (CEM if cfg.algorithm == "cem" else CMAES)(z0, cfg.init_sigma, cfg, free)
    state = store.load_state()
    if state is None and cfg.resume_from:
        prior = json.loads((Path(cfg.resume_from) / "state.json").read_text(encoding="ascii"))
        prev_cfg = json.loads((Path(cfg.resume_from) / "config.json").read_text(encoding="ascii"))
        if prev_cfg["space_hash"] != SPACE.space_hash() or prev_cfg["config"]["algorithm"] != cfg.algorithm:
            raise RuntimeError(f"{cfg.resume_from} searched a different space or algorithm")
        strat.load(prior["strategy"])
        gen0, incumbent = 0, prior["incumbent"]
    elif state is None:
        gen0, incumbent = 0, _round(start)
    else:
        gen0, incumbent = state["next_gen"], state["incumbent"]
        strat.load(state["strategy"])
    inc_hash = store.vector(incumbent)
    tconf = cfg.table_config()
    with Evaluator(tconf, jobs) as ev:
        for gen in range(gen0, cfg.generations):
            t0 = time.time()
            rng = np.random.default_rng(int.from_bytes(canonical_hash([cfg.seed, gen], domain="cem-rng").encode()[:8],
                                                       "big"))
            z = strat.ask(rng)
            mean_params = _round(SPACE.from_unit(strat.mean))
            samples = [_round(SPACE.from_unit(row)) for row in z]
            vectors = [incumbent, mean_params, *samples]
            roles = ["incumbent", "mean", *(["sample"] * len(samples))]
            hashes = [store.vector(v) for v in vectors]
            pop_hash = canonical_hash(hashes, domain="strategy-search-population-v1")
            tasks = make_tasks(source, cfg.partition, ruleset_id=cfg.ruleset_id, players=cfg.players,
                               count=cfg.tables_per_gen, namespace=f"{cfg.seed}|gen{gen}|select",
                               mirror_rate=cfg.mirror_rate)
            ts_hash = store.tasks(tasks)
            res = ev.evaluate(vectors, tasks, incumbent=incumbent)
            summaries = [summarize(r) for r in res]
            for role, h, r, s in zip(roles, hashes, res, summaries):
                store.append("evaluations.jsonl", {"gen": gen, "stage": "select", "role": role, "vector_hash": h,
                                                   "seed_set_hash": ts_hash, "population_hash": pop_hash,
                                                   "registry_hash": digest["registry_hash"],
                                                   "points": [x["points"] for x in r], "summary": s})
            scores = [s["mean"] for s in summaries[2:]]
            strat.tell(z, scores)
            best = 1 if cfg.challenger == "mean" else 1 + int(np.argmax([s["mean"] for s in summaries[1:]]))
            challenger, ch_hash = vectors[best], hashes[best]
            decision: dict[str, Any] = {"challenger_role": roles[best], "challenger_hash": ch_hash,
                                        "select_vs_incumbent": paired(res[0], res[best])}
            accepted = False
            if ch_hash != inc_hash:
                ctasks = make_tasks(source, cfg.partition, ruleset_id=cfg.ruleset_id, players=cfg.players,
                                    count=cfg.confirm_tables, namespace=f"{cfg.seed}|gen{gen}|confirm",
                                    mirror_rate=cfg.mirror_rate)
                cts_hash = store.tasks(ctasks)
                cres = ev.evaluate([incumbent, challenger], ctasks, incumbent=incumbent)
                for role, h, r in (("incumbent", inc_hash, cres[0]), ("challenger", ch_hash, cres[1])):
                    store.append("evaluations.jsonl", {"gen": gen, "stage": "confirm", "role": role,
                                                       "vector_hash": h, "seed_set_hash": cts_hash,
                                                       "population_hash": pop_hash,
                                                       "registry_hash": digest["registry_hash"],
                                                       "points": [x["points"] for x in r], "summary": summarize(r)})
                pc = paired(cres[0], cres[1])
                decision.update(confirm_seed_set_hash=cts_hash, confirm=pc)
                accepted = pc["mean"] > 0 and pc["t"] >= cfg.accept_t
                if accepted:
                    incumbent, inc_hash = challenger, ch_hash
            fb = sum(s["internal_fallbacks"] + s["arena_fallbacks"] for s in summaries)
            row = {"gen": gen, "seed_set_hash": ts_hash, "population_hash": pop_hash,
                   "registry_hash": digest["registry_hash"], "incumbent_hash": inc_hash, "accepted": accepted,
                   "incumbent_select_mean": summaries[0]["mean"], "mean_select_mean": summaries[1]["mean"],
                   "best_sample_mean": max(scores), "median_sample_mean": float(np.median(scores)),
                   "fallbacks": fb, "sigma_mean": float(np.mean(strat.sigma)) if isinstance(strat, CEM)
                   else float(strat.sigma), "wall_s": round(time.time() - t0, 1), **decision}
            store.append("generations.jsonl", row)
            store.save_state({"next_gen": gen + 1, "incumbent": incumbent, "incumbent_hash": inc_hash,
                              "strategy": strat.state(), "mean_unit": strat.mean.tolist()})
            (run_dir / "incumbent.json").write_text(json.dumps({"vector_hash": inc_hash, "params": incumbent},
                                                               sort_keys=True, indent=1) + "\n", encoding="ascii")
            conf = decision.get("confirm")
            log(f"gen {gen}: inc {summaries[0]['mean']:.2f} mean {summaries[1]['mean']:.2f} best {max(scores):.2f} "
                f"med {np.median(scores):.2f} | challenger {roles[best]} "
                + (f"confirm {conf['mean']:+.2f} (t {conf['t']:.2f})" if conf else "is incumbent")
                + f" -> {'ACCEPT' if accepted else 'keep'} | fb {fb} | {row['wall_s']}s")
    return run_dir
