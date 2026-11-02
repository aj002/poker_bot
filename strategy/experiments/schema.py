"""Experiment specs and immutable run manifests.

An :class:`ExperimentSpec` is declared before running: arms, variants, table
structure, population partition, blocks, seeds, the minimum effect, the interval
method and execution gates. The objective is fixed to terminal points; chips cannot
be declared as a primary metric.

:class:`ExperimentManifest` is what a spec materializes into: it pins the rules hash
of every variant, the content hash of every policy artifact, the population registry
and partition hashes, every cell's lineup and a hash of the seed list. Its
``manifest_id`` is a hash of all of that, so two runs with the same id ran the same
experiment. Manifests are written once, read-only, and re-verified on load.
Seeds are private: the manifest carries only their hashes.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr, model_validator

from arena.domain.models import RulesetId, TournamentConfig, default_blind_levels
from strategy.rules.canonical import canonical_hash, canonical_json

SPEC_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
SPEC_DIR = Path(__file__).resolve().parent / "specs"
Stage = Literal["discovery", "confirmation", "final_holdout"]


class _S(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ArmSpec(_S):
    policy: StrictStr
    salt: StrictStr | None = None
    """Re-keys the policy random stream (identical artifact otherwise); for A/A tests."""


class ArmsSpec(_S):
    A: ArmSpec
    """Control or incumbent."""
    B: ArmSpec
    """Candidate."""


class TableSpec(_S):
    players: StrictInt = Field(8, ge=2, le=8)
    starting_stack: StrictInt = Field(2000, ge=1)
    hand_cap: StrictInt = Field(200, ge=1)
    hands_per_level: StrictInt = Field(25, ge=1)
    blind_levels: list[list[StrictInt]] | None = None
    payout_points: list[StrictInt] | None = None

    def config(self, ruleset_id: str) -> TournamentConfig:
        return TournamentConfig(ruleset_id=ruleset_id, initial_player_count=self.players,
                                starting_stack=self.starting_stack, hand_cap=self.hand_cap,
                                hands_per_level=self.hands_per_level,
                                blind_levels=self.blind_levels or default_blind_levels(),
                                payout_points=self.payout_points)


DEFAULT_POPULATION_SOURCE = "strategy.population:PopulationSource"


class PopulationSpec(_S):
    source: StrictStr = DEFAULT_POPULATION_SOURCE
    """``module:attribute`` naming a :class:`~strategy.eval.population.PartitionSource`, or ``stub``
    (arena baselines and styles; reproduces the published null calibrations)."""
    partition: Literal["train", "validation", "holdout"]
    lineups_per_variant: StrictInt = Field(1, ge=1)
    explicit_lineups: list[list[StrictStr]] | None = None
    """Fixed opponent policy specs per lineup; members are still hashed into the manifest."""


class ExecutionGates(_S):
    max_timeout_rate: StrictFloat | None = Field(0.005, ge=0, le=1)
    """Integrity gate, checked in both arms over every seat (candidate and opponents): a
    timed-out decision is a forced fold, so above this rate the points are withheld."""
    max_entrant_timeout_rate: StrictFloat | None = Field(0.02, ge=0, le=1)
    """Integrity gate for any single entrant (with at least ``ENTRANT_TIMEOUT_MIN_COUNT`` timeouts)."""
    max_illegal_actions: StrictInt | None = 0
    max_errors: StrictInt | None = 0
    max_timeouts: StrictInt | None = None
    max_fallback_rate: StrictFloat | None = None
    max_p99_latency_s: StrictFloat | None = None
    max_artifact_bytes: StrictInt | None = 100_000_000
    max_failed_tables: StrictInt | None = 0
    require_competitive: StrictBool = False
    control_every: StrictInt = Field(4, ge=0)
    """Sandbox runs play a control table before the first table and after every
    ``control_every`` tables of each arm, and classify timeouts against it
    (:mod:`strategy.eval.stalls`); 0 turns the classified timeout gate off."""
    control_iterations: StrictInt | None = Field(None, ge=0)
    """Loop iterations per decision of the workload-matched control
    (:mod:`strategy.eval.burn`); None calibrates them per arm to the candidate's
    median decide time."""
    timeout_replays_per_seat: StrictInt = Field(3, ge=0)
    """Timed-out explicit-state decisions replayed per seat per table."""
    max_bot_timeout_rate: StrictFloat | None = Field(0.0002, ge=0, le=1)
    """Arm B fails outright (no rerun) above this bot-caused timeout rate."""
    bot_p_min: StrictFloat = Field(0.01, gt=0, lt=1)
    """C-test level for arm B's bot-caused timeouts in excess of the workload-matched
    control's (rerun like a stall excess; a fail once reruns are exhausted)."""
    max_bot_timeouts: StrictInt | None = None
    """Optional absolute limit on arm B's bot-caused timeouts (fail, no rerun)."""
    max_control_timeout_rate: StrictFloat | None = Field(0.0005, ge=0, le=1)
    """Host ceiling: an arm whose control runs significantly above this rate is host-invalid."""
    stall_p_min: StrictFloat = Field(0.01, gt=0, lt=1)
    """C-test level for stall-like timeouts in excess of the control."""
    max_reruns: StrictInt = Field(2, ge=0)
    """Reruns of a host-invalid, bot-excess or stall-excess arm (same seeds; every attempt reported)."""


class NullCalibrationSpec(_S):
    replicates: StrictInt = Field(ge=2)
    blocks_per_replicate: StrictInt = Field(ge=2)
    exact_coupling_blocks: StrictInt = Field(1, ge=0)


class ExperimentSpec(_S):
    schema_version: Literal[1] = SPEC_SCHEMA_VERSION
    experiment_id: StrictStr = Field(pattern=r"^[A-Za-z0-9_.-]{1,64}$")
    title: StrictStr
    hypothesis_id: StrictStr | None = None
    description: StrictStr = ""
    kind: Literal["ab", "null_calibration"] = "ab"
    stage: Stage
    objective: Literal["terminal_points"] = "terminal_points"
    primary_metric: Literal["mean_point_difference"] = "mean_point_difference"
    min_effect: StrictFloat = Field(ge=0)
    alpha: StrictFloat = Field(0.05, gt=0, lt=0.5)
    power: StrictFloat = Field(0.8, gt=0, lt=1)
    interval: Literal["paired_t", "cluster_bootstrap"] = "paired_t"
    bootstrap_resamples: StrictInt = Field(2000, ge=100)
    arms: ArmsSpec
    variants: list[RulesetId] = Field(min_length=1)
    table: TableSpec = Field(default_factory=TableSpec)
    population: PopulationSpec
    blocks_per_lineup: StrictInt = Field(ge=1)
    base_seed: StrictStr = Field(min_length=1)
    isolation: Literal["sandbox", "in_process"] = "sandbox"
    budget_ns: StrictInt | None = None
    gates: ExecutionGates = Field(default_factory=ExecutionGates)
    null: NullCalibrationSpec | None = None

    @model_validator(mode="after")
    def _check(self) -> ExperimentSpec:
        if len(set(self.variants)) != len(self.variants):
            raise ValueError("variants must be distinct")
        if (self.stage == "final_holdout") != (self.population.partition == "holdout"):
            raise ValueError("the holdout partition is used by, and only by, a final_holdout run")
        if self.kind == "null_calibration":
            if self.null is None:
                raise ValueError("null_calibration needs a null section")
            if self.arms.A.policy != self.arms.B.policy:
                raise ValueError("a null calibration compares identical policies")
            if self.blocks_per_lineup != self.null.replicates * self.null.blocks_per_replicate:
                raise ValueError("blocks_per_lineup must equal replicates * blocks_per_replicate")
        elif self.null is not None:
            raise ValueError("only null_calibration experiments take a null section")
        if self.population.explicit_lineups is not None:
            if any(len(x) != self.table.players - 1 for x in self.population.explicit_lineups):
                raise ValueError("each explicit lineup names one opponent per non-candidate seat")
        for v in self.variants:
            self.table.config(v)
        return self

    def spec_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"), domain="strategy-experiment-spec-v1")

    def redacted(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        data["base_seed"] = "sha256:" + hashlib.sha256(self.base_seed.encode()).hexdigest()
        return data


def load_spec(path_or_id: str | Path) -> ExperimentSpec:
    """A spec file, or the id of a spec in ``strategy/experiments/specs/``."""
    path = Path(path_or_id)
    if not path.exists():
        path = SPEC_DIR / f"{path_or_id}.json"
    if not path.exists():
        known = sorted(p.stem for p in SPEC_DIR.glob("*.json"))
        raise FileNotFoundError(f"no experiment spec {path_or_id!r}; known: {', '.join(known)}")
    return ExperimentSpec.model_validate_json(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------- manifest


class ObjectiveEntry(_S):
    id: StrictStr
    description: StrictStr
    scorer_module: StrictStr
    scorer_sha256: StrictStr
    chips: Literal["diagnostic_only"] = "diagnostic_only"


class RulesEntry(_S):
    ruleset_id: StrictStr
    rules_hash: StrictStr
    variant_hash: StrictStr
    arena_config_hash: StrictStr
    capabilities: list[StrictStr]


class PolicyEntry(_S):
    label: StrictStr
    spec: StrictStr
    salt: StrictStr | None
    base_artifact_sha256: StrictStr
    artifact_sha256: StrictStr
    artifact_bytes: StrictInt
    rulesets: list[StrictStr]


class PartitionEntry(_S):
    name: StrictStr
    hash: StrictStr
    seed_namespace: StrictStr
    member_ids: list[StrictStr]


class PopulationEntry(_S):
    source: StrictStr
    registry_id: StrictStr
    registry_hash: StrictStr
    stub: StrictBool
    partition: PartitionEntry


class OpponentEntry(_S):
    member_id: StrictStr
    family: StrictStr
    policy: StrictStr
    artifact_sha256: StrictStr


class CellEntry(_S):
    cell_id: StrictStr
    ruleset_id: StrictStr
    rules_hash: StrictStr
    lineup_id: StrictStr
    opponents: list[OpponentEntry]
    blocks: StrictInt
    seed_sha256: StrictStr
    role: Literal["primary", "exact_coupling"] = "primary"


class SeedsEntry(_S):
    blocks_total: StrictInt
    seed_list_sha256: StrictStr
    derivation: StrictStr


class EngineEntry(_S):
    arena_version: StrictStr
    pokerkit_version: StrictStr


class ExperimentManifest(_S):
    manifest_kind: Literal["strategy_experiment_run"] = "strategy_experiment_run"
    manifest_schema_version: Literal[1] = MANIFEST_SCHEMA_VERSION
    compatibility_status: Literal["practice_assumptions"] = "practice_assumptions"
    manifest_id: StrictStr
    experiment_id: StrictStr
    spec_hash: StrictStr
    spec: dict[str, Any]
    """The spec with ``base_seed`` replaced by its hash."""
    objective: ObjectiveEntry
    engine: EngineEntry
    rules: list[RulesEntry]
    policies: dict[str, PolicyEntry]
    population: PopulationEntry
    cells: list[CellEntry]
    seeds: SeedsEntry
    isolation: StrictStr
    code: dict[str, Any]
    """Recorded, not hashed: the code that materialized the manifest."""
    created_at: StrictStr

    @staticmethod
    def hashed_content(data: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in data.items() if k not in ("manifest_id", "code", "created_at")}

    def computed_id(self) -> str:
        return "m_" + canonical_hash(self.hashed_content(self.model_dump(mode="json")),
                                     domain="strategy-experiment-manifest-v1")[:32]

    def verify(self) -> None:
        if self.computed_id() != self.manifest_id:
            raise ManifestTampered(f"manifest_id {self.manifest_id} does not match its content ({self.computed_id()})")


class ManifestTampered(ValueError):
    pass


class ManifestConflict(FileExistsError):
    pass


def write_manifest(path: Path, manifest: ExperimentManifest) -> None:
    """Write once. An existing file must hold the same manifest id; it is never replaced."""
    manifest.verify()
    if path.exists():
        existing = load_manifest(path)
        if existing.manifest_id != manifest.manifest_id:
            raise ManifestConflict(f"{path} holds manifest {existing.manifest_id}, not {manifest.manifest_id}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(canonical_json(manifest.model_dump(mode="json")) + "\n", encoding="ascii")
    os.chmod(tmp, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    tmp.replace(path)


def load_manifest(path: Path) -> ExperimentManifest:
    manifest = ExperimentManifest.model_validate_json(path.read_text(encoding="ascii"))
    manifest.verify()
    return manifest
