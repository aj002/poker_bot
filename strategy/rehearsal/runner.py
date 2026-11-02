"""Build day, end to end: an announced variant spec to a gated submission, every step timed.

    python -m strategy.rehearsal run strategy/rehearsal/heldback/plo5-8max.json --budget fast

The spec (``heldback/*.json``) is what an organiser's announcement turns into: a
:class:`~strategy.rules.variants.VariantSpec` dictionary under ``variant`` and the
tournament structure (``TournamentConfig`` fields) under ``structure``. A variant equal
to a practice one runs under that ruleset id; anything else runs in the build-day reveal
slot (``reveal_variant``, :mod:`arena.rules.reveal` and :mod:`strategy.rules.reveal`).

Steps, in order (``report.json`` records each one's wall time, status and detail):

``load``        parse the spec, match it against the practice variants, build the
                engine's rule profile and the tournament config (anything the engine
                cannot express fails here, loudly), write ``reveal/variant.json``.
``conformance`` the strategy rules layer against the engine: metadata, category
                ranking, lockstep differential hands with canonical-action legality,
                the card evaluator against PokerKit's showdown order on random hands,
                and complete smoke tournaments with the fallback in every seat.
``fallback``    package :mod:`strategy.policy.generic` with the variant: the first
                artifact, built before any table or search.
``gate_fallback`` the deployed gate (sandboxed seat, classified timeouts, headroom) for
                the fallback at the announced structure.
``submit_early`` the gated fallback as the first submission window (``submission/early``).
``tables``      regenerate the postflop equity table, the preflop table where the
                class count allows it, and decide push/fold feasibility (reasons are
                recorded; push/fold is not on the baseline's decision path).
``search``      a short CEM re-tune of the baseline on the train partition, starting
                from the NLHE vector (:data:`START_VECTOR`).
``package``     the tuned and the transferred (untuned) baseline bundles.
``validation``  in-process A/B experiments on the validation partition: tuned vs
                fallback, tuned vs transferred, transferred vs fallback.
``select``      :data:`SELECTION_RULE`, declared here before any result exists.
``gate``        the deployed gate for the selected bundle (the fallback's is reused).
``limits``      artifact bytes, resident memory, and that no numpy/numba is loaded.
``submit``      the final window (``submission/final``): a deterministic tarball of the
                selected bundle, the fallback as the rollback, ``SHA256SUMS`` and
                ``submission.json``.

Budgets (:data:`BUDGETS`) trade resolution for wall time; ``smoke`` exists for tests.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
import math
import os
import random
import subprocess
import sys
import tarfile
import time
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
HELDBACK = Path(__file__).resolve().parent / "heldback"
SPEC_SCHEMA = 1
START_VECTOR = REPO / "strategy" / "policy" / "vectors" / "nlhe-v2.json"
PREFLOP_MAX_COMBOS = 300_000
"""The preflop table enumerates every hole combo; PLO4's 270,725 is the largest built so far."""
MAX_ARTIFACT_BYTES = 100_000_000
MAX_BOT_RSS_BYTES = 100_000_000
POPULATION_SOURCE = "strategy.population:PopulationSource"

BUDGETS: dict[str, dict[str, Any]] = {
    "smoke": {"differential_hands": 40, "evaluator_hands": 300, "smoke_tables": 2, "postflop_samples": 4000,
              "preflop": {"samples": 200, "flops": 4, "inner": 4}, "preflop_plo": {"samples": 20, "flops": 2, "inner": 2},
              "cem": {"generations": 1, "population": 4, "tables_per_gen": 4, "confirm_tables": 4},
              "validation": {"lineups": 1, "blocks": 1}, "gate_decisions": 300, "gate_max_tables": 30},
    "fast": {"differential_hands": 200, "evaluator_hands": 3000, "smoke_tables": 8, "postflop_samples": 150_000,
             "preflop": {"samples": 20000, "flops": 100, "inner": 100},
             "preflop_plo": {"samples": 600, "flops": 24, "inner": 24},
             "cem": {"generations": 6, "population": 12, "tables_per_gen": 48, "confirm_tables": 96},
             "validation": {"lineups": 20, "blocks": 4}, "gate_decisions": 50_000, "gate_max_tables": 2000},
    "full": {"differential_hands": 400, "evaluator_hands": 10000, "smoke_tables": 16, "postflop_samples": 300_000,
             "preflop": {"samples": 20000, "flops": 100, "inner": 100},
             "preflop_plo": {"samples": 600, "flops": 24, "inner": 24},
             "cem": {"generations": 14, "population": 16, "tables_per_gen": 64, "confirm_tables": 128},
             "validation": {"lineups": 40, "blocks": 6}, "gate_decisions": 50_000, "gate_max_tables": 2000},
}

SELECTION_RULE = (
    "Candidates in order: tuned (the CEM incumbent), transferred (the NLHE vector, untuned), fallback. "
    "Tuned is selected when its validation difference against the fallback has a paired 95% lower bound "
    "above 0 and its mean difference against the transferred bundle is at least 0. Otherwise transferred "
    "is selected when its difference against the fallback has a paired 95% lower bound above 0. Otherwise "
    "the fallback. A selected bundle that fails the deployed gate drops to the next eligible candidate; "
    "the fallback must pass the gate or there is no submission."
)


class RehearsalFailed(RuntimeError):
    """A fatal step failed; the report records why and nothing later runs."""


@dataclass
class Step:
    name: str
    status: str = "pending"
    """``ok``, ``fail``, ``na`` (does not apply), ``skipped`` or ``fatal``."""
    wall_s: float = 0.0
    active_s: float = 0.0
    """Monotonic seconds; less than ``wall_s`` only when the host was suspended during the step."""
    started_s: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "status": self.status, "wall_s": round(self.wall_s, 2),
                "active_s": round(self.active_s, 2), "started_s": round(self.started_s, 2), "detail": self.detail,
                "error": self.error}


def load_announcement(path: Path | str) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != SPEC_SCHEMA:
        raise ValueError(f"announcement schema_version {raw.get('schema_version')!r}, expected {SPEC_SCHEMA}")
    for key in ("name", "variant", "structure"):
        if key not in raw:
            raise ValueError(f"announcement is missing {key!r}")
    return raw


def _same_game(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    drop = ("ruleset_id", "title", "family", "extra")
    return {k: v for k, v in a.items() if k not in drop} == {k: v for k, v in b.items() if k not in drop}


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts)


def deterministic_tar(src: Path, dest: Path, arcname: str) -> str:
    """A gzipped tar of ``src`` with sorted entries, zero mtimes and owners; returns its sha256."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for p in sorted(src.rglob("*")):
            if "__pycache__" in p.parts or not p.is_file():
                continue
            info = tarfile.TarInfo(f"{arcname}/{p.relative_to(src).as_posix()}")
            data = p.read_bytes()
            info.size, info.mtime, info.mode, info.uid, info.gid = len(data), 0, 0o644, 0, 0
            tar.addfile(info, io.BytesIO(data))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f, gzip.GzipFile(fileobj=f, mode="wb", mtime=0, filename="") as gz:
        gz.write(buf.getvalue())
    return _sha_file(dest)


class Rehearsal:
    def __init__(self, spec_path: Path | str, out: Path | str, *, budget: str = "fast", jobs: int | None = None,
                 gate: bool = True, log: Callable[[str], None] | None = None) -> None:
        if budget not in BUDGETS:
            raise ValueError(f"budget is one of {sorted(BUDGETS)}")
        self.spec_path = Path(spec_path)
        self.out = Path(out).resolve()
        self.budget_name = budget
        self.budget = BUDGETS[budget]
        self.jobs = jobs or max(1, (os.cpu_count() or 2) - 1)
        self.gate_enabled = gate
        self._log = log or (lambda s: print(s, file=sys.stderr, flush=True))
        self.steps: list[Step] = []
        self.t0, self.m0 = time.time(), time.monotonic()
        self.ctx: dict[str, Any] = {}

    def log(self, msg: str) -> None:
        self._log(f"[{time.time() - self.t0:7.1f}s] {msg}")

    # ---------------------------------------------------------------- driver

    def _run(self, name: str, fn: Callable[[Step], None], *, fatal: bool = True) -> Step:
        step = Step(name, started_s=time.time() - self.t0)
        self.steps.append(step)
        self.log(f"step {name} ...")
        t, m = time.time(), time.monotonic()
        try:
            fn(step)
            if step.status == "pending":
                step.status = "ok"
        except Exception as exc:
            step.error = f"{type(exc).__name__}: {exc}"
            step.detail.setdefault("traceback", traceback.format_exc(limit=6))
            step.status = "fatal" if fatal else "fail"
        step.wall_s, step.active_s = time.time() - t, time.monotonic() - m
        self.log(f"step {name}: {step.status} in {step.wall_s:.1f}s" + (f" ({step.error})" if step.error else ""))
        self._write_report()
        if step.status == "fatal":
            raise RehearsalFailed(f"{name}: {step.error}")
        return step

    def run(self) -> dict[str, Any]:
        self.out.mkdir(parents=True, exist_ok=True)
        try:
            self._run("load", self.step_load)
            self._run("conformance", self.step_conformance)
            self._run("fallback", self.step_fallback)
            self._run("gate_fallback", self.step_gate_fallback, fatal=False)
            self._run("submit_early", self.step_submit_early, fatal=False)
            self._run("tables", self.step_tables, fatal=False)
            self._run("search", self.step_search, fatal=False)
            self._run("package", self.step_package)
            self._run("validation", self.step_validation, fatal=False)
            self._run("select", self.step_select)
            self._run("gate", self.step_gate)
            self._run("limits", self.step_limits)
            self._run("submit", self.step_submit)
        except RehearsalFailed as exc:
            self.ctx["failure"] = str(exc)
        return self._write_report()

    # ----------------------------------------------------------------- steps

    def step_load(self, step: Step) -> None:
        from arena.domain.models import TournamentConfig
        from strategy.rules.reveal import REVEAL_ENV, REVEAL_RULESET_ID
        from strategy.rules.variants import KNOWN_VARIANTS, VariantSpec

        ann = load_announcement(self.spec_path)
        self.ctx["announcement"] = ann
        name = self.ctx["name"] = str(ann["name"])
        variant = {"schema_version": 1, "ruleset_id": REVEAL_RULESET_ID, **ann["variant"]}
        spec = VariantSpec.from_dict(variant)
        known = next((rid for rid, v in KNOWN_VARIANTS.items() if _same_game(v.as_dict(), spec.as_dict())), None)
        step.detail.update(name=name, announcement=ann.get("announcement"), variant_hash=spec.variant_hash(),
                           known_variant=known)
        if known is not None:
            rid = known
        else:
            rid = REVEAL_RULESET_ID
            reveal = self.out / "reveal"
            (reveal / "tables").mkdir(parents=True, exist_ok=True)
            for stale in (reveal / "tables").iterdir():
                stale.unlink()
            (reveal / "variant.json").write_text(json.dumps(variant, indent=1, sort_keys=True) + "\n",
                                                 encoding="utf-8")
            os.environ[REVEAL_ENV] = str(reveal)
            self.ctx["reveal"] = reveal
            from arena.rules.profiles import get_profile

            profile = get_profile(rid)
            step.detail["engine_profile"] = {"title": profile.title, "betting": profile.betting,
                                             "hand_selection": profile.hand_selection, "ranks": profile.ranks,
                                             "hole_cards": profile.hole_card_count}
        structure = dict(ann["structure"])
        if structure.get("ante"):
            raise ValueError(f"ante {structure['ante']}: the arena deals no antes (TournamentConfig.ante is 0), "
                             "so the announced structure cannot be played or rehearsed")
        config = TournamentConfig(ruleset_id=rid, **structure)
        from strategy.rules.adapter import RulesAdapter

        adapter = RulesAdapter.from_config(config)
        self.ctx.update(name=name, ruleset_id=rid, spec=spec, config=config,
                        structure={k: v for k, v in config.model_dump().items()
                                   if k in ("initial_player_count", "starting_stack", "hand_cap", "hands_per_level",
                                            "blind_levels", "payout_points")})
        self.ctx["structure"]["payout_points"] = config.resolved_payouts
        step.detail.update(ruleset_id=rid, config_hash=config.config_hash(), rules_hash=adapter.rules_hash,
                           capabilities=sorted(adapter.capabilities), structure=self.ctx["structure"])

    def step_conformance(self, step: Step) -> None:
        from strategy.rules.conformance import run_conformance

        rid, spec = self.ctx["ruleset_id"], self.ctx["spec"] if self.ctx.get("reveal") else None
        rep = run_conformance(rid, hands=self.budget["differential_hands"], spec=spec)
        checks = {}
        for c in rep.checks:
            d = c.as_dict()
            if c.name == "fixtures" and c.passed + c.failed == 0:
                d["status"] = "na"
            else:
                d["status"] = "ok" if c.ok else "fail"
            checks[c.name] = d
        evaluator = evaluator_differential(rid, hands=self.budget["evaluator_hands"])
        checks["evaluator"] = evaluator
        smoke = smoke_tournaments(self.ctx["config"], tables=self.budget["smoke_tables"])
        checks["smoke"] = smoke
        step.detail["checks"] = {k: {kk: v for kk, v in d.items() if kk != "failures" or v} for k, d in checks.items()}
        failed = [k for k, d in checks.items() if d["status"] == "fail"]
        if not smoke["ok"]:
            raise RuntimeError(f"the engine cannot play the variant: {smoke['errors'][:3]}")
        if failed:
            step.status = "fail"
            step.error = "failed checks: " + ", ".join(failed)
        self.ctx["card_layer_ok"] = checks["evaluator"]["status"] == "ok"

    def step_fallback(self, step: Step) -> None:
        from strategy.policy.package import build_bundle
        from strategy.search.cem import _git_commit

        path = build_bundle(self.out / "bots" / "fallback", kind="fallback", reveal=self.ctx.get("reveal"),
                            commit=_git_commit())
        self.ctx["bundles"] = {"fallback": path}
        man = json.loads((path / "manifest.json").read_text())
        step.detail.update(bundle=str(path), bundle_hash=man["bundle_hash"], bytes=_tree_bytes(path),
                           first_artifact_s=round(time.time() - self.t0, 2))

    def step_tables(self, step: Step) -> None:
        reveal = self.ctx.get("reveal")
        rid, spec = self.ctx["ruleset_id"], self.ctx["spec"]
        if reveal is None:
            from strategy.tables import store

            step.status = "na"
            step.detail["note"] = f"{rid} is a practice variant: the committed tables apply"
            step.detail["committed"] = {n: store.stale(n) is False for n in (f"preflop-{rid}", f"postflop-{rid}")}
            return
        from strategy.features.cards import capabilities
        from strategy.features.equity import EquityTables
        from strategy.rules.reveal import HASH_MARKER
        from strategy.tables import postflop, preflop, store

        tdir = reveal / "tables"
        built: dict[str, Any] = {}
        try:
            caps = capabilities(rid)
        except ValueError as exc:
            step.status = "fail"
            step.error = f"the card layer cannot read the variant ({exc}); the bots play card-blind"
            return
        samples = self.budget["postflop_samples"]
        data = postflop.build(rid, samples=samples, jobs=self.jobs)
        secs = data.pop("_seconds")
        store.write(postflop.table_name(rid), data, generator=postflop.GENERATOR, inputs=postflop.INPUTS,
                    params={"ruleset_id": rid, "samples": samples, "seed": 1, "variant_hash": spec.variant_hash()},
                    seconds=secs, directory=tdir)
        built["postflop"] = {"samples_per_street": samples, "seconds": round(secs, 1)}
        combos = math.comb(spec.deck_size, spec.hole_cards)
        if combos <= PREFLOP_MAX_COMBOS:
            params = dict(self.budget["preflop_plo" if spec.hole_cards > 2 else "preflop"])
            data = preflop.build(rid, jobs=self.jobs, **params)
            secs = data.pop("_seconds")
            store.write(preflop.table_name(rid), data, generator=preflop.GENERATOR, inputs=preflop.INPUTS,
                        params={"ruleset_id": rid, **params, "seed": 1, "variant_hash": spec.variant_hash()},
                        seconds=secs, directory=tdir)
            built["preflop"] = {**params, "classes": len(data["keys"]), "hole_combos": combos,
                                "seconds": round(secs, 1)}
        else:
            built["preflop"] = {"skipped": f"{combos:,} hole combos exceed {PREFLOP_MAX_COMBOS:,}: the runtime "
                                           "uses its labelled heuristic preflop prior"}
        (tdir / HASH_MARKER).write_text(spec.variant_hash() + "\n", encoding="ascii")
        store.load.cache_clear()
        eq = EquityTables(caps, str(tdir))
        built["loaded"] = {"preflop": eq.has_preflop, "postflop": eq.has_postflop}
        built["push_fold"] = push_fold_feasibility(spec, caps)
        step.detail.update(built)

    def step_search(self, step: Step) -> None:
        from strategy.policy.package import load_params
        from strategy.search.cem import SearchConfig, run_search

        s = self.ctx["structure"]
        start = load_params(START_VECTOR)
        self.ctx["vectors"] = {"transferred": start, "tuned": start}
        cem = self.budget["cem"]
        cfg = SearchConfig(run_id=f"{self.ctx['name']}-{self.budget_name}", ruleset_id=self.ctx["ruleset_id"],
                           players=s["initial_player_count"], starting_stack=s["starting_stack"],
                           hand_cap=s["hand_cap"], seed=f"rehearsal|{self.ctx['name']}", partition="train",
                           start=start,
                           structure={"hands_per_level": s["hands_per_level"], "blind_levels": s["blind_levels"],
                                      "payout_points": s["payout_points"]}, **cem)
        root = self.out / "search"
        step.detail["resumed"] = (root / cfg.run_id / "state.json").exists()
        run_dir = run_search(cfg, root=root, jobs=self.jobs, log=self.log)
        inc = json.loads((run_dir / "incumbent.json").read_text())
        gens = [json.loads(line) for line in (run_dir / "generations.jsonl").read_text().splitlines() if line]
        self.ctx["vectors"]["tuned"] = inc["params"]
        from strategy.policy.params import vector_hash

        step.detail.update(run_dir=str(run_dir), start_vector=str(START_VECTOR.relative_to(REPO)),
                           start_hash=vector_hash(start), incumbent_hash=inc["vector_hash"],
                           changed=inc["vector_hash"] != vector_hash(start), budget=cem,
                           accepted=sum(1 for g in gens if g["accepted"]),
                           fallbacks=sum(g["fallbacks"] for g in gens),
                           generations=[{k: g.get(k) for k in ("gen", "accepted", "incumbent_select_mean",
                                                                "best_sample_mean", "wall_s")} | (
                               {"confirm": g["confirm"]} if "confirm" in g else {}) for g in gens])

    def step_package(self, step: Step) -> None:
        from strategy.policy.package import build_bundle
        from strategy.search.cem import _git_commit

        vectors = self.ctx.get("vectors") or {}
        if not vectors:
            from strategy.policy.package import load_params

            vectors = self.ctx["vectors"] = {"transferred": load_params(START_VECTOR),
                                             "tuned": load_params(START_VECTOR)}
        commit = _git_commit()
        for role in ("transferred", "tuned"):
            path = build_bundle(self.out / "bots" / role, vectors[role], name="strategy_baseline", commit=commit,
                                reveal=self.ctx.get("reveal"))
            self.ctx["bundles"][role] = path
            man = json.loads((path / "manifest.json").read_text())
            step.detail[role] = {"bundle": str(path), "bundle_hash": man["bundle_hash"],
                                 "vector_hash": man["vector_hash"], "bytes": _tree_bytes(path)}
        same = step.detail["tuned"]["vector_hash"] == step.detail["transferred"]["vector_hash"]
        step.detail["tuned_equals_transferred"] = self.ctx["tuned_equals_transferred"] = same

    def step_validation(self, step: Step) -> None:
        pairs = [("tuned_vs_fallback", "fallback", "tuned"), ("tuned_vs_transferred", "transferred", "tuned"),
                 ("transferred_vs_fallback", "fallback", "transferred")]
        same = bool(self.ctx.get("tuned_equals_transferred"))
        results: dict[str, Any] = {}
        for label, a, b in pairs:
            if same and label == "tuned_vs_transferred":
                results[label] = {"skipped": "identical vectors: the search kept the start vector",
                                  "mean_difference": 0.0, "ci95": [0.0, 0.0]}
                continue
            if same and label == "transferred_vs_fallback":
                results[label] = {"same_as": "tuned_vs_fallback"}
                continue
            results[label] = self._experiment(label, a, b)
            step.detail[label] = results[label]
        if same:
            results["transferred_vs_fallback"] = dict(results["tuned_vs_fallback"], B="transferred",
                                                      same_as="tuned_vs_fallback")
            step.detail.update(results)
        self.ctx["validation"] = results
        bad = [k for k, v in results.items() if v.get("withheld") or v.get("error")]
        if bad:
            step.status = "fail"
            step.error = "withheld or failed: " + ", ".join(bad)

    def _experiment(self, label: str, a: str, b: str) -> dict[str, Any]:
        from strategy.experiments.runner import run_experiment
        from strategy.experiments.schema import ExperimentSpec

        s = self.ctx["structure"]
        val = self.budget["validation"]
        exp_id = f"rh-{self.ctx['name']}-{label}"[:64]
        spec = ExperimentSpec.model_validate({
            "experiment_id": exp_id,
            "title": f"Rehearsal {self.ctx['name']}: {b} (B) vs {a} (A), validation partition",
            "description": "Build-day rehearsal validation (strategy.rehearsal). Discovery stage: no promotion claim.",
            "stage": "discovery", "min_effect": 1.0,
            "arms": {"A": {"policy": str(self.ctx["bundles"][a])}, "B": {"policy": str(self.ctx["bundles"][b])}},
            "variants": [self.ctx["ruleset_id"]],
            "table": {"players": s["initial_player_count"], "starting_stack": s["starting_stack"],
                      "hand_cap": s["hand_cap"], "hands_per_level": s["hands_per_level"],
                      "blind_levels": s["blind_levels"], "payout_points": s["payout_points"]},
            "population": {"source": POPULATION_SOURCE, "partition": "validation",
                           "lineups_per_variant": val["lineups"]},
            "blocks_per_lineup": val["blocks"],
            "base_seed": f"rehearsal-{self.ctx['name']}-validation-v1",
            "isolation": "in_process", "interval": "paired_t",
        })
        import shutil

        shutil.rmtree(self.out / "lab" / exp_id, ignore_errors=True)
        run_dir, report = run_experiment(spec, root=self.out / "lab", jobs=self.jobs, keep_histories=False)
        an = report["analysis"]
        if an.get("withheld"):
            return {"withheld": True, "decision": an["decision"], "run_dir": str(run_dir)}
        p = an["primary"]
        fair = sum(s["payout_points"]) / s["initial_player_count"]
        return {"A": a, "B": b, "blocks": p["blocks"], "tables_per_arm": p["tables_per_arm"],
                "mean_points_A": p.get("mean_points_A"), "mean_points_B": p.get("mean_points_B"),
                "fair_share": fair, "mean_difference": p.get("mean_difference"), "ci95": p.get("chosen_interval"),
                "decision": an["decision"].get("verdict", an["decision"].get("decision")),
                "worst_family": (an.get("robustness") or {}).get("worst_family"),
                "manifest_id": run_dir.name, "run_dir": str(run_dir)}

    def step_select(self, step: Step) -> None:
        v = self.ctx.get("validation") or {}

        def lo(label: str) -> float | None:
            r = v.get(label) or {}
            ci = r.get("ci95")
            return ci[0] if ci else None

        order: list[str] = []
        why: dict[str, str] = {}
        t_f, t_t, x_f = lo("tuned_vs_fallback"), (v.get("tuned_vs_transferred") or {}).get("mean_difference"), \
            lo("transferred_vs_fallback")
        if t_f is not None and t_f > 0 and t_t is not None and t_t >= 0:
            order.append("tuned")
            why["tuned"] = f"tuned - fallback lower bound {t_f:+.2f} > 0 and tuned - transferred mean {t_t:+.2f} >= 0"
        else:
            why["tuned"] = f"not eligible (tuned - fallback lower bound {t_f}, tuned - transferred mean {t_t})"
        if x_f is not None and x_f > 0:
            order.append("transferred")
            why["transferred"] = f"transferred - fallback lower bound {x_f:+.2f} > 0"
        else:
            why["transferred"] = f"not eligible (transferred - fallback lower bound {x_f})"
        order.append("fallback")
        self.ctx["selection_order"] = order
        step.detail.update(rule=SELECTION_RULE, order=order, reasons=why)

    def step_gate_fallback(self, step: Step) -> None:
        if not self.gate_enabled:
            step.status = "skipped"
            step.detail["note"] = "gate disabled (--no-gate)"
            return
        r = self.ctx.setdefault("gates", {})["fallback"] = self._gate("fallback")
        step.detail.update(r)
        if not r["passed"]:
            step.status = "fail"
            step.error = "the fallback failed the deployed gate: no safe early submission or rollback"

    def step_submit_early(self, step: Step) -> None:
        fb = (self.ctx.get("gates") or {}).get("fallback")
        if fb is None or not fb["passed"]:
            step.status = "skipped" if fb is None else "fail"
            step.error = None if fb is None else "the fallback did not pass the gate"
            step.detail["note"] = "no gated fallback to submit"
            return
        step.detail.update(self._write_submission("early", "fallback", None))
        step.detail["early_submission_s"] = round(time.time() - self.t0, 2)

    def step_gate(self, step: Step) -> None:
        if not self.gate_enabled:
            step.status = "skipped"
            step.detail["note"] = "gate disabled (--no-gate); the selection is not deployable as reported"
            self.ctx["chosen"] = self.ctx["selection_order"][0]
            return
        results = self.ctx.setdefault("gates", {})
        chosen = None
        for role in self.ctx["selection_order"]:
            if role not in results:
                results[role] = self._gate(role)
                step.detail[role] = results[role]
            if results[role]["passed"]:
                chosen = role
                break
        step.detail["reused"] = sorted(set(results) - set(step.detail))
        if chosen is None:
            raise RuntimeError("no candidate passed the deployed gate, including the fallback")
        self.ctx["chosen"] = chosen
        step.detail["chosen"] = chosen
        if not results["fallback"]["passed"]:
            step.status = "fail"
            step.error = "the fallback failed the deployed gate (no safe rollback)"

    def _gate(self, role: str) -> dict[str, Any]:
        from strategy.policy.gate import run_gate

        s = self.ctx["structure"]
        rep = run_gate(self.ctx["bundles"][role], ruleset_id=self.ctx["ruleset_id"],
                       min_decisions=self.budget["gate_decisions"], max_tables=self.budget["gate_max_tables"],
                       structure=s, cost_tables=0, log=lambda m: self.log(f"gate {role}: {m}"))
        rep.pop("manifest", None)
        (self.out / "gates").mkdir(parents=True, exist_ok=True)
        (self.out / "gates" / f"{role}.json").write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
        w = rep.get("worker_wall") or {}
        return {"passed": rep["passed"], "decisions": rep["decisions"], "tables": rep["tables"],
                "p50_us": w.get("p50_us"), "p99_us": w.get("p99_us"), "p999_us": w.get("p999_us"),
                "max_us": w.get("max_us"), "fallbacks": rep["fallbacks"], "internal_fallbacks": rep["internal_fallbacks"],
                "restarts": rep["restarts"], "max_state_bytes": rep["max_state_bytes"],
                "timeout_classes": rep["timeout_classes"], "failed_checks": [k for k, ok in rep["checks"].items() if not ok],
                "attempts": len(rep["attempts"]), "wall_s": rep["wall_s"], "report": str(self.out / "gates" / f"{role}.json")}

    def step_limits(self, step: Step) -> None:
        roles = sorted({"fallback", self.ctx["chosen"]})
        bad = []
        for role in roles:
            path = self.ctx["bundles"][role]
            probe = probe_bundle(path, self.ctx["config"])
            probe["bytes"] = _tree_bytes(path)
            probe["ok"] = (probe["bytes"] <= MAX_ARTIFACT_BYTES and probe["bot_rss_bytes"] <= MAX_BOT_RSS_BYTES
                           and not probe["heavy_modules"] and probe["loaded_from_bundle"] and probe["fallbacks"] == 0)
            step.detail[role] = probe
            if not probe["ok"]:
                bad.append(role)
        if bad:
            raise RuntimeError(f"limits failed for {bad}")

    def step_submit(self, step: Step) -> None:
        chosen = self.ctx["chosen"]
        step.detail.update(self._write_submission("final", chosen, "fallback"))

    def _write_submission(self, window: str, role: str, rollback: str | None) -> dict[str, Any]:
        sub = self.out / "submission" / window
        if sub.exists():
            for p in sub.iterdir():
                p.unlink()
        files: dict[str, Any] = {}
        for role, kind in ((role, "submission"), (rollback, "rollback")):
            if role is None:
                continue
            if kind == "rollback" and role == files["submission"]["role"]:
                files["rollback"] = files["submission"]
                continue
            path = self.ctx["bundles"][role]
            man = json.loads((path / "manifest.json").read_text())
            arc = f"{self.ctx['name']}-{role}-{man['bundle_hash'][:12]}"
            dest = sub / f"{arc}.tar.gz"
            files[kind] = {"role": role, "file": dest.name, "sha256": deterministic_tar(path, dest, arc),
                           "bytes": dest.stat().st_size, "bundle_hash": man["bundle_hash"],
                           "vector_hash": man.get("vector_hash"), "commit": man.get("commit")}
        sums = sorted({(f["sha256"], f["file"]) for f in files.values()}, key=lambda x: x[1])
        (sub / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for h, n in sums), encoding="ascii")
        meta = {"name": self.ctx["name"], "window": window, "ruleset_id": self.ctx["ruleset_id"],
                "variant_hash": self.ctx["spec"].variant_hash(), "config_hash": self.ctx["config"].config_hash(),
                "gate": {k: (self.ctx.get("gates") or {}).get(v["role"], {}).get("passed") for k, v in files.items()},
                **files}
        (sub / "submission.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n", encoding="ascii")
        return {**meta, "directory": str(sub)}

    # ---------------------------------------------------------------- report

    def _write_report(self) -> dict[str, Any]:
        steps = [s.as_dict() for s in self.steps]
        total = time.time() - self.t0
        first = next((s.detail.get("first_artifact_s") for s in self.steps if s.name == "fallback"), None)
        early = next((s.detail.get("early_submission_s") for s in self.steps if s.name == "submit_early"), None)
        rep = {"spec": str(self.spec_path), "name": self.ctx.get("name"), "budget": self.budget_name,
               "budget_detail": self.budget, "jobs": self.jobs, "ruleset_id": self.ctx.get("ruleset_id"),
               "selection_rule": SELECTION_RULE, "chosen": self.ctx.get("chosen"),
               "failure": self.ctx.get("failure"), "total_wall_s": round(total, 1),
               "total_active_s": round(time.monotonic() - self.m0, 1), "first_artifact_s": first,
               "early_submission_s": early,
               "steps": steps}
        (self.out / "report.json").write_text(json.dumps(rep, indent=1, sort_keys=True, default=str) + "\n")
        (self.out / "report.md").write_text(render_markdown(rep))
        return rep


# ------------------------------------------------------------------ checks


def evaluator_differential(ruleset_id: str, *, hands: int, seed: str = "rehearsal-evaluator-v1") -> dict[str, Any]:
    """The strategy card evaluator's showdown order against PokerKit's on random two-player hands."""
    from pokerkit import Card

    from arena.rules.profiles import get_profile
    from strategy.features.cards import capabilities, card_ints, evaluate
    from strategy.rules.reveal import get_variant

    spec = get_variant(ruleset_id)
    try:
        caps = capabilities(ruleset_id)
    except ValueError as exc:
        return {"status": "fail", "hands": 0, "disagreements": 0, "error": f"card layer: {exc}"}
    hand_type = get_profile(ruleset_id).game_class.hand_types[0]
    deck = [r + s for r in spec.ranks for s in spec.suits]
    rng = random.Random(f"{seed}|{spec.variant_hash()}")
    bad, examples = 0, []
    t = time.perf_counter()
    for _ in range(hands):
        cards = rng.sample(deck, 2 * spec.hole_cards + 5)
        h1, h2, board = cards[:spec.hole_cards], cards[spec.hole_cards:2 * spec.hole_cards], cards[-5:]
        e1, e2 = (evaluate(card_ints(h), card_ints(board), caps) for h in (h1, h2))
        p1, p2 = (hand_type.from_game(tuple(Card.parse("".join(h))), tuple(Card.parse("".join(board))))
                  for h in (h1, h2))
        mine = (e1 > e2) - (e1 < e2)
        theirs = (p1 > p2) - (p1 < p2)
        if mine != theirs:
            bad += 1
            if len(examples) < 5:
                examples.append({"hole": [h1, h2], "board": board, "strategy": mine, "engine": theirs})
    return {"status": "ok" if bad == 0 else "fail", "hands": hands, "disagreements": bad, "examples": examples,
            "seconds": round(time.perf_counter() - t, 1)}


def smoke_tournaments(config: Any, *, tables: int, seed: str = "rehearsal-smoke-v1") -> dict[str, Any]:
    """Complete tournaments with the fallback in every seat: the engine deals, settles and scores the variant."""
    from arena.agents.sdk import bot_info, decode_state
    from arena.agents.worker import InProcessSeat, play_table
    from arena.benchmark.runner import Assignment, table_info
    from arena.table.service import seeded_table
    from strategy.policy.generic import GenericBot

    errors, hands, decisions, fallbacks, internal = [], 0, 0, 0, 0
    points_total = []
    for i in range(tables):
        try:
            a = Assignment(0, i, 0, f"{seed}|{i}", {s: k for k, s in enumerate(config.resolved_seats)})
            agents = {}
            for s in a.seats:
                bot = GenericBot()
                agents[s] = InProcessSeat(s, bot, bot_info(bot), table_info(config, a, s, config.bot_budget_ns), "smoke")
                agents[s].start()
            table = seeded_table(config, f"{seed}|{i}", table_id=a.table_id)
            play_table(table, agents)
            hands += len(table.hands)
            pts = table.points()
            points_total.append(round(sum(float(x) for x in pts.values()), 6))
            for ag in agents.values():
                decisions += ag.stats.decisions
                fallbacks += sum(ag.stats.fallbacks.values())
                st = decode_state(ag.committed)
                internal += int(st.get("fb", 0)) if isinstance(st, dict) else 0
        except Exception as exc:
            errors.append(f"table {i}: {type(exc).__name__}: {exc}")
    want = float(sum(config.resolved_payouts))
    points_ok = all(abs(p - want) < 1e-6 for p in points_total)
    ok = not errors and fallbacks == 0 and internal == 0 and points_ok and decisions > 0
    return {"status": "ok" if ok else "fail", "ok": ok, "tables": tables, "hands": hands, "decisions": decisions,
            "arena_fallbacks": fallbacks, "internal_fallbacks": internal, "points_conserved": points_ok,
            "errors": errors}


def push_fold_feasibility(spec: Any, caps: Any) -> dict[str, Any]:
    """Whether the push/fold (endgame) tables could be regenerated for this variant, and why not."""
    from strategy.endgame.classes import PUSH_FOLD_CAPABILITIES

    reasons = []
    missing = sorted(set(PUSH_FOLD_CAPABILITIES) - spec.capabilities)
    if missing:
        reasons.append(f"needs {', '.join(missing)} (dense two-card class matrices and a legal one-action shove)")
    if not caps.standard_priority:
        reasons.append("the numba showdown kernel ranks non-flush categories in hold'em order only")
    if importlib.util.find_spec("numba") is None:
        reasons.append("numba is not installed (uv run --group train)")
    reasons.append("the endgame generator samples states at the default 8-seat structure and takes 10-20 "
                   "minutes per mode; the baseline and fallback do not read push/fold tables")
    return {"feasible": len(reasons) == 1, "regenerated": False, "reasons": reasons}


PROBE = r"""
import gc, json, resource, sys
bundle = sys.argv[1]


def rss():
    with open("/proc/self/status") as f:
        return next(int(line.split()[1]) * 1024 for line in f if line.startswith("VmRSS:"))

sys.path.insert(0, bundle)
from arena.agents.sdk import bot_info, decode_state, load_bot
from arena.agents.worker import InProcessSeat, play_table
from arena.benchmark.runner import Assignment, table_info
from arena.domain.models import TournamentConfig
from arena.table.service import seeded_table
config = TournamentConfig.model_validate_json(sys.argv[2])
before = set(sys.modules)
gc.collect()
rss0 = rss()
load_bot(bundle + "/bot.py", module_name="rehearsal_probe_bot")
mod = sys.modules["rehearsal_probe_bot"]
import strategy
fb = 0
for i in range(2):
    a = Assignment(0, i, 0, f"probe|{i}", {s: k for k, s in enumerate(config.resolved_seats)})
    agents = {}
    for s in a.seats:
        bot = mod.make_bot()
        agents[s] = InProcessSeat(s, bot, bot_info(bot), table_info(config, a, s, config.bot_budget_ns), "probe")
        agents[s].start()
    play_table(seeded_table(config, f"probe|{i}", table_id=a.table_id), agents)
    for ag in agents.values():
        st = decode_state(ag.committed)
        fb += sum(ag.stats.fallbacks.values()) + (int(st.get("fb", 0)) if isinstance(st, dict) else 0)
rss1 = rss()
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
new = sorted(m for m in set(sys.modules) - before)
heavy = [m for m in new if m.split(".")[0] in ("numpy", "numba", "scipy", "torch")]
print(json.dumps({"strategy_file": strategy.__file__, "rss_before_bot": rss0, "rss_after": rss1, "peak": peak,
                  "heavy_modules": heavy, "strategy_modules": [m for m in new if m.startswith("strategy")],
                  "fallbacks": fb}))
"""


def probe_bundle(bundle: Path, config: Any) -> dict[str, Any]:
    """Load the bundle from its own directory in a fresh process, seat it everywhere for two tables, and
    measure the resident memory it adds (every seat at the table plays it) and the modules it pulls in."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(PYTHONPATH=str(bundle), PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run([sys.executable, "-c", PROBE, str(bundle), config.model_dump_json()], capture_output=True,
                         text=True, cwd=bundle, env=env, timeout=600)
    if out.returncode != 0:
        raise RuntimeError(f"bundle probe failed: {out.stderr[-2000:]}")
    r = json.loads(out.stdout.strip().splitlines()[-1])
    return {"loaded_from_bundle": str(r["strategy_file"]).startswith(str(bundle)),
            "bot_rss_bytes": max(0, r["rss_after"] - r["rss_before_bot"]), "process_rss_bytes": r["rss_after"],
            "process_peak_rss_bytes": r["peak"],
            "heavy_modules": r["heavy_modules"], "strategy_modules": len(r["strategy_modules"]),
            "fallbacks": r["fallbacks"]}


# ---------------------------------------------------------------- markdown


def _fmt(x: Any, nd: int = 2) -> str:
    if x is None:
        return "null"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def render_markdown(rep: Mapping[str, Any]) -> str:
    lines = [f"# Rehearsal: {rep.get('name')}", "",
             f"Spec `{rep['spec']}`, budget `{rep['budget']}`, ruleset `{rep.get('ruleset_id')}`, {rep['jobs']} jobs.",
             f"Total wall time {rep['total_wall_s']} s; first artifact at {_fmt(rep.get('first_artifact_s'))} s; "
             f"gated early submission at {_fmt(rep.get('early_submission_s'))} s.",
             f"Chosen: **{rep.get('chosen')}**." + (f" Failure: {rep['failure']}" if rep.get("failure") else "")]
    active = rep.get("total_active_s")
    if active is not None and rep["total_wall_s"] - active >= 1.0:
        lines.append(f"The host was suspended for {rep['total_wall_s'] - active:.1f} s of that ({active} s active, "
                     "monotonic clock); the active column excludes it.")
    if rep.get("timing_note"):
        lines.append(rep["timing_note"])
    lines += ["", "| step | status | wall s | active s | note |", "|---|---|---|---|---|"]
    for s in rep["steps"]:
        lines.append(f"| {s['name']} | {s['status']} | {s['wall_s']:.1f} | {_fmt(s.get('active_s'), 1)} | "
                     f"{(s['error'] or '').replace('|', '/')[:160]} |")
    steps = {s["name"]: s for s in rep["steps"]}
    val = steps.get("validation", {}).get("detail", {})
    rows = [(k, v) for k, v in val.items() if isinstance(v, dict) and "mean_difference" in v and "A" in v]
    if rows:
        lines += ["", "Validation (in-process, validation partition; B minus A, official points per tournament):", "",
                  "| comparison | blocks | A mean | B mean | fair share | B - A | 95% CI |", "|---|---|---|---|---|---|---|"]
        for k, v in rows:
            ci = v.get("ci95") or [None, None]
            label = k + (f" (same run as {v['same_as']})" if v.get("same_as") else "")
            lines.append(f"| {label} | {v['blocks']} | {_fmt(v['mean_points_A'])} | {_fmt(v['mean_points_B'])} | "
                         f"{_fmt(v['fair_share'])} | {_fmt(v['mean_difference'])} | [{_fmt(ci[0])}, {_fmt(ci[1])}] |")
        for k, v in val.items():
            if isinstance(v, dict) and v.get("skipped"):
                lines.append(f"| {k} | - | - | - | - | {_fmt(v.get('mean_difference'))} | not run: {v['skipped']} |")
    search = steps.get("search", {}).get("detail", {})
    if "accepted" in search:
        lines += ["", f"Search: {len(search.get('generations') or [])} generations, {search['accepted']} accepted, "
                      f"incumbent {'changed' if search.get('changed') else 'unchanged (tuned = transferred)'}."]
    grows = []
    if steps.get("gate_fallback", {}).get("detail", {}).get("passed") is not None:
        grows.append(("fallback", steps["gate_fallback"]["detail"]))
    grows += [(k, v) for k, v in steps.get("gate", {}).get("detail", {}).items()
              if isinstance(v, dict) and "passed" in v]
    if grows:
        lines += ["", "Deployed gate:", "", "| bundle | passed | decisions | p99 us | p99.9 us | timeouts | internal fb | failed checks |",
                  "|---|---|---|---|---|---|---|---|"]
        for k, v in grows:
            lines.append(f"| {k} | {v['passed']} | {v['decisions']} | {_fmt(v['p99_us'], 0)} | {_fmt(v['p999_us'], 0)} | "
                         f"{v['fallbacks'].get('timeout', 0)} | {v['internal_fallbacks']} | {', '.join(v['failed_checks']) or '-'} |")
    lines += ["", "Selection rule (declared before the run): " + rep["selection_rule"], ""]
    return "\n".join(lines) + "\n"


def run(spec: Path | str, out: Path | str | None = None, **kwargs: Any) -> dict[str, Any]:
    spec = Path(spec)
    name = json.loads(spec.read_text()).get("name", spec.stem)
    budget = kwargs.get("budget", "fast")
    out = Path(out) if out else REPO / "artifacts" / "rehearsal" / f"{name}-{budget}"
    return Rehearsal(spec, out, **kwargs).run()
