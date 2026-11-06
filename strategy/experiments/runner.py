"""Materialize a spec into an immutable manifest, run its paired cells, write the report.

Run directory (default ``artifacts/lab/<experiment_id>/<manifest_id>/``)::

    manifest.json        immutable, read-only; verified on every load
    private/seeds.json   per-cell seeds (private, like the arena's run seeds)
    cells/<cell>/<arm>/  the arena benchmark manifest of each arm run
    report.json          manifest + analysis (+ null calibration)
    report.md            the concise human report

The runner never edits hypothesis records or their conclusions; it only links to
them by ``hypothesis_id``.
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import arena
from arena.benchmark.manifest import code_info
from arena.rules.pokerkit_adapter import POKERKIT_VERSION
from strategy.eval.analysis import Declared, analyze
from strategy.eval.null import calibrate, exact_coupling
from strategy.eval.paired import Arm, Cell, CellResult, run_cell
from strategy.eval.parallel import capacity, safe_jobs
from strategy.eval.policies import ResolvedPolicy, resolve_policy
from strategy.eval.population import Lineup, PartitionSource, explicit_lineup, load_partition_source
from strategy.eval.report import finite, write_report
from strategy.experiments.schema import (
    CellEntry,
    EngineEntry,
    ExperimentManifest,
    ExperimentSpec,
    ObjectiveEntry,
    OpponentEntry,
    PartitionEntry,
    PolicyEntry,
    PopulationEntry,
    RulesEntry,
    SeedsEntry,
    load_manifest,
    write_manifest,
)
from strategy.objective.points import OBJECTIVE_DESCRIPTION, OBJECTIVE_ID, scorer_fingerprint
from strategy.rules.adapter import RulesAdapter
from strategy.rules.canonical import canonical_json

DEFAULT_LAB_DIR = Path("artifacts/lab")
SEED_DERIVATION = "sha256('strategy-cell-seed-v1|' + base_seed + '|' + partition seed namespace + '|' + cell_id)[:20]"


class AlreadyRun(FileExistsError):
    pass


@dataclass
class Plan:
    spec: ExperimentSpec
    manifest: ExperimentManifest
    cells: list[Cell]
    coupling_cells: list[Cell]
    arms: tuple[Arm, Arm]
    seeds: dict[str, str]


def _cell_seed(spec: ExperimentSpec, namespace: str, cell_id: str) -> str:
    return hashlib.sha256(f"strategy-cell-seed-v1|{spec.base_seed}|{namespace}|{cell_id}".encode()).hexdigest()[:20]


def _policy_entry(label: str, arm: Arm, work_dir: Path) -> tuple[PolicyEntry, ResolvedPolicy]:
    from strategy.eval.paired import resolve_arm

    base = resolve_policy(arm.policy)
    resolved = resolve_arm(arm, work_dir)
    return PolicyEntry(label=label, spec=arm.policy, salt=arm.salt, base_artifact_sha256=base.artifact_sha256,
                       artifact_sha256=resolved.artifact_sha256, artifact_bytes=resolved.artifact_bytes,
                       rulesets=list(base.rulesets)), resolved


def _lineups(spec: ExperimentSpec, source: PartitionSource, ruleset_id: str) -> list[Lineup]:
    if spec.population.explicit_lineups is not None:
        return [explicit_lineup(x, ruleset_id) for x in spec.population.explicit_lineups]
    partition = source.partition(spec.population.partition)
    return source.sample_lineups(partition, ruleset_id=ruleset_id, opponents=spec.table.players - 1,
                                 count=spec.population.lineups_per_variant, seed=spec.base_seed)


def materialize(spec: ExperimentSpec, work_dir: Path, source: PartitionSource | None = None) -> Plan:
    source = source or load_partition_source(spec.population.source)
    partition = source.partition(spec.population.partition)
    arms = (Arm("A", spec.arms.A.policy, spec.arms.A.salt), Arm("B", spec.arms.B.policy, spec.arms.B.salt))
    policies: dict[str, PolicyEntry] = {}
    for arm in arms:
        policies[arm.label], _ = _policy_entry(arm.label, arm, work_dir)
    rules, cells, cell_entries, coupling = [], [], [], []
    seeds: dict[str, str] = {}
    for ruleset_id in spec.variants:
        config = spec.table.config(ruleset_id)
        adapter = RulesAdapter.from_config(config)
        rules.append(RulesEntry(ruleset_id=ruleset_id, rules_hash=adapter.rules_hash, variant_hash=adapter.variant_hash,
                                arena_config_hash=config.config_hash(), capabilities=sorted(adapter.capabilities)))
        for arm in arms:
            if ruleset_id not in policies[arm.label].rulesets:
                raise ValueError(f"arm {arm.label} ({arm.policy}) does not play {ruleset_id}")
        for j, lineup in enumerate(_lineups(spec, source, ruleset_id)):
            for o in lineup.opponents:
                if ruleset_id not in o.rulesets:
                    raise ValueError(f"lineup member {o.member_id} does not play {ruleset_id}")
            cell_id = f"{ruleset_id}.{j:02d}"
            seeds[cell_id] = _cell_seed(spec, partition.seed_namespace, cell_id)
            cell = Cell(cell_id, config, lineup, spec.blocks_per_lineup, seeds[cell_id])
            cells.append(cell)
            cell_entries.append(_cell_entry(cell, adapter.rules_hash, "primary"))
            if spec.kind == "null_calibration" and j == 0 and spec.null and spec.null.exact_coupling_blocks:
                cid = f"{cell_id}.coupling"
                seeds[cid] = _cell_seed(spec, partition.seed_namespace, cid)
                cc = Cell(cid, config, lineup, spec.null.exact_coupling_blocks, seeds[cid])
                coupling.append(cc)
                cell_entries.append(_cell_entry(cc, adapter.rules_hash, "exact_coupling"))
    from arena.benchmark.manifest import derive_seeds, seed_list_hash

    all_seeds = [s for c in cells + coupling for s in derive_seeds(c.seed, c.blocks)]
    content: dict[str, Any] = {
        "experiment_id": spec.experiment_id,
        "spec_hash": spec.spec_hash(),
        "spec": spec.redacted(),
        "objective": ObjectiveEntry(id=OBJECTIVE_ID, description=OBJECTIVE_DESCRIPTION,
                                    scorer_module="arena.tournament.scoring", scorer_sha256=scorer_fingerprint()),
        "engine": EngineEntry(arena_version=arena.__version__, pokerkit_version=POKERKIT_VERSION),
        "rules": rules,
        "policies": policies,
        "population": PopulationEntry(
            source=spec.population.source, registry_id=source.registry_id, registry_hash=source.registry_hash(),
            stub=bool(partition.metadata.get("stub")),
            partition=PartitionEntry(name=partition.name, hash=partition.partition_hash(),
                                     seed_namespace=partition.seed_namespace,
                                     member_ids=[m.member_id for m in partition.members])),
        "cells": cell_entries,
        "seeds": SeedsEntry(blocks_total=len(all_seeds), seed_list_sha256=seed_list_hash(all_seeds),
                            derivation=SEED_DERIVATION + "; blocks via arena derive_seeds"),
        "isolation": spec.isolation,
    }
    draft = ExperimentManifest(manifest_id="pending", code=code_info().model_dump(mode="json"),
                               created_at=datetime.now(UTC).isoformat(timespec="seconds"), **content)
    manifest = draft.model_copy(update={"manifest_id": draft.computed_id()})
    return Plan(spec, manifest, cells, coupling, arms, seeds)


def _cell_entry(cell: Cell, rules_hash: str, role: str) -> CellEntry:
    return CellEntry(
        cell_id=cell.cell_id, ruleset_id=cell.ruleset_id, rules_hash=rules_hash, lineup_id=cell.lineup.lineup_id,
        opponents=[OpponentEntry(member_id=o.member_id, family=o.family, policy=o.policy,
                                 artifact_sha256=resolve_policy(o.policy).artifact_sha256)
                   for o in cell.lineup.opponents],
        blocks=cell.blocks, seed_sha256=hashlib.sha256(cell.seed.encode()).hexdigest(), role=role)


def default_run_dir(manifest: ExperimentManifest, root: Path = DEFAULT_LAB_DIR) -> Path:
    return root / manifest.experiment_id / manifest.manifest_id


def run_experiment(spec: ExperimentSpec, *, out_dir: Path | None = None, root: Path = DEFAULT_LAB_DIR,
                   jobs: int | None = None, source: PartitionSource | None = None, keep_histories: bool = False,
                   progress: Callable[[str, int, int], None] | None = None) -> tuple[Path, dict[str, Any]]:
    """Run a spec end to end. Returns the run directory and the report.

    ``jobs`` defaults to :func:`strategy.eval.parallel.safe_jobs` for the spec's isolation.
    """
    jobs = jobs or safe_jobs(spec.isolation)
    staging = (out_dir or root / spec.experiment_id) / "_arms"
    plan = materialize(spec, staging, source)
    run_dir = out_dir or default_run_dir(plan.manifest, root)
    if (run_dir / "report.json").exists():
        raise AlreadyRun(f"{run_dir} already holds a finished run of this manifest")
    write_manifest(run_dir / "manifest.json", plan.manifest)
    manifest = load_manifest(run_dir / "manifest.json")
    private = run_dir / "private"
    private.mkdir(parents=True, exist_ok=True)
    (private / "seeds.json").write_text(canonical_json({"manifest_id": manifest.manifest_id, "base_seed": spec.base_seed,
                                                        "cells": plan.seeds}) + "\n", encoding="ascii")
    from strategy.eval.paired import resolve_arm

    policies = {arm.label: resolve_arm(arm, run_dir) for arm in plan.arms}
    for label, policy in policies.items():
        if policy.artifact_sha256 != manifest.policies[label].artifact_sha256:
            raise RuntimeError(f"arm {label} artifact changed after the manifest was written")
    vr = spec.variance_reduction
    kwargs = dict(isolation=spec.isolation, work_dir=run_dir, jobs=jobs, budget_ns=spec.budget_ns,
                  keep_histories=keep_histories, progress=progress, gates=spec.gates.model_dump())
    if vr is not None:
        from strategy.eval.kernels import HAVE_NUMBA

        if not HAVE_NUMBA:
            warnings.warn("variance_reduction without numba (`uv sync --group train`): luck terms are about "
                          "40 times slower to compute", RuntimeWarning, stacklevel=2)
        kwargs["luck"] = vr.settings()
    results: list[CellResult] = [run_cell(c, plan.arms, policies, **kwargs) for c in plan.cells]
    declared = Declared(min_effect=spec.min_effect, alpha=spec.alpha, power=spec.power, interval=spec.interval,
                        bootstrap_resamples=spec.bootstrap_resamples, seed=manifest.manifest_id, stage=spec.stage,
                        gates=spec.gates.model_dump(), variance_reduction=vr.model_dump() if vr else None)
    analysis = analyze(results, declared)
    report: dict[str, Any] = {
        "report_kind": "strategy_experiment_report",
        "manifest": manifest.model_dump(mode="json"),
        "run": {"code": code_info().model_dump(mode="json"), "jobs": jobs,
                "safe_jobs": safe_jobs(spec.isolation), "host": capacity(),
                "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "benchmark_manifests": {f"{r.cell.cell_id}/{k}": v.benchmark_manifest_hash
                                        for r in results for k, v in r.runs.items()}},
        "analysis": analysis,
    }
    if spec.kind == "null_calibration" and spec.null is not None and analysis["withheld"]:
        report["null_calibration"] = {"withheld": True, "note": analysis["decision"]["note"]}
    elif spec.kind == "null_calibration" and spec.null is not None:
        same = (Arm("A", spec.arms.A.policy, spec.arms.A.salt), Arm("B", spec.arms.A.policy, spec.arms.A.salt))
        same_policies = {"A": policies["A"], "B": policies["A"]}
        coupling = [run_cell(c, same, same_policies, **kwargs) for c in plan.coupling_cells]
        report["null_calibration"] = calibrate(
            results, replicates=spec.null.replicates, blocks_per_replicate=spec.null.blocks_per_replicate,
            alpha=spec.alpha, min_effect=spec.min_effect, resamples=spec.bootstrap_resamples,
            seed=manifest.manifest_id, variance_reduction=vr.model_dump() if vr else None)
        report["null_calibration"]["exact_coupling"] = exact_coupling(coupling) if coupling else None
    write_report(run_dir, report)
    _prune_staging(staging)
    return run_dir, finite(report)


def _prune_staging(staging: Path) -> None:
    import shutil

    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)


def describe_plan(plan: Plan) -> dict[str, Any]:
    m = plan.manifest
    return {"manifest_id": m.manifest_id, "experiment_id": m.experiment_id, "cells": len(m.cells),
            "blocks_total": m.seeds.blocks_total,
            "tables_total": sum(c.blocks * c.config.initial_player_count * 2 for c in plan.cells + plan.coupling_cells),
            "rules": [r.model_dump() for r in m.rules], "policies": {k: v.model_dump() for k, v in m.policies.items()},
            "population": m.population.model_dump()}
