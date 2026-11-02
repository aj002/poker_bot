"""The registry as a :class:`strategy.eval.population.PartitionSource`.

Experiment specs name it as ``"population": {"source": "strategy.population:PopulationSource", ...}``.

* :attr:`PopulationSource.registry_id` is ``strategy.population/<version>``;
  :meth:`~PopulationSource.registry_hash` is the frozen registry hash, checked against
  ``registry.lock.json`` when the source is built.
* :meth:`~PopulationSource.partition` lists the partition's members, plus checkpoint slots
  promoted into it, as :class:`~strategy.eval.population.OpponentRef` entries. Each
  member's policy is a generated, self-contained bot bundle
  (``artifacts/population/<version>/<id>/``, see :meth:`Member.write_bot`; written on
  first use, identical on every machine), so the harness resolves, hashes, stages and
  sandboxes it like any other bot. ``params_hash`` is the member's spec hash (a
  checkpoint's artifact hash).
* :meth:`~PopulationSource.sample_lineups` draws distinct families under the partition's
  rules: every holdout lineup has a holdout-only family, every validation lineup has a
  validation-only pair, and no train lineup has one.

Members also play the build-day reveal variant while one is active
(:func:`strategy.rules.reveal.reveal_active`), through the nearest practice card adapter
(:func:`strategy.population.cards.card_ruleset`); otherwise the partitions, and so their
hashes, are exactly the practice ones.

``OpponentRef.family`` is for evaluation reports only; at the table an opponent's entrant
name is its opaque member id.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

from arena.domain.models import REVEAL_RULESET_ID, RULESET_IDS
from strategy.eval.population import Lineup, OpponentRef, Partition
from strategy.population.families import CHECKPOINT_FAMILY, HOLDOUT_ONLY
from strategy.population.params import PARTITIONS
from strategy.population.registry import (
    LOCK_PATH,
    VALIDATION_ONLY_PAIRS,
    CheckpointLedger,
    Registry,
    default_registry,
    draw_entries,
)
from strategy.rules.canonical import canonical_hash
from strategy.rules.reveal import reveal_active

REPO = Path(__file__).resolve().parents[2]
BOT_ROOT = REPO / "artifacts" / "population"


def _policy_path(path: Path) -> str:
    """Repo-relative when the bot lies under the working directory (the harness runs from the repo root)."""
    try:
        return os.path.relpath(path, Path.cwd()) if path.is_relative_to(Path.cwd()) else str(path)
    except ValueError:
        return str(path)


class PopulationSource:
    """The frozen opponent registry behind the evaluation harness's partition protocol."""

    def __init__(self, registry: Registry | None = None, bot_dir: str | Path | None = None, *,
                 check_lock: bool = True) -> None:
        self.registry = registry or default_registry()
        self.registry_id = f"strategy.population/{self.registry.version}"
        self.bot_dir = Path(bot_dir) if bot_dir is not None else BOT_ROOT / self.registry.version
        if check_lock and LOCK_PATH.exists():
            frozen = json.loads(LOCK_PATH.read_text())
            if frozen["version"] == self.registry.version and frozen["registry_hash"] != self.registry.hash:
                raise RuntimeError(f"registry {self.registry.version} differs from {LOCK_PATH.name}: "
                                   "bump REGISTRY_VERSION and re-lock")
        self._partitions: dict[str, Partition] = {}

    @property
    def ledger(self) -> CheckpointLedger:
        return self.registry.ledger

    def registry_hash(self) -> str:
        return self.registry.hash

    def partition(self, name: str) -> Partition:
        if name not in PARTITIONS:
            raise KeyError(f"unknown partition {name!r}; choose from {PARTITIONS}")
        if name not in self._partitions:
            self._partitions[name] = self._build(name)
        return self._partitions[name]

    def _build(self, name: str) -> Partition:
        self.bot_dir.mkdir(parents=True, exist_ok=True)
        rulesets = tuple(RULESET_IDS)
        if reveal_active():
            rulesets += (REVEAL_RULESET_ID,)
        refs = [OpponentRef(m.member_id, m.family, _policy_path(m.write_bot(self.bot_dir)), rulesets, m.hash)
                for m in sorted(self.registry.partition(name), key=lambda m: m.member_id)]
        slots = sorted((s for s in self.ledger.slots.values() if s.partition == name and s.status == "promoted"),
                       key=lambda s: s.slot_id)
        for s in slots:
            s.verify()
            refs.append(OpponentRef(s.slot_id, CHECKPOINT_FAMILY, _policy_path(Path(s.path)), s.rulesets, s.sha256))
        metadata: dict[str, Any] = {
            "stub": False,
            "registry_version": self.registry.version,
            "families": self.registry.families_in(name),
            "holdout_only": list(HOLDOUT_ONLY),
            "validation_only_pairs": [list(p) for p in VALIDATION_ONLY_PAIRS],
            "checkpoints": [s.slot_id for s in slots],
        }
        return Partition(name, self.registry_id, self.registry_hash(), f"{self.registry.version}|{name}",
                         tuple(refs), metadata)

    def sample_lineups(self, partition: Partition, *, ruleset_id: str, opponents: int, count: int,
                       seed: str) -> list[Lineup]:
        """Deterministic in (partition hash, ruleset, opponents, seed); lineup ``j`` uses its own stream."""
        if partition.registry_id != self.registry_id:
            raise ValueError(f"partition {partition.name!r} belongs to {partition.registry_id}, not {self.registry_id}")
        if not 1 <= opponents <= 7:
            raise ValueError("opponents must be 1..7 (the arena seats at most 8)")
        eligible = partition.eligible(ruleset_id)
        by_id = {o.member_id: o for o in eligible}
        members: dict[str, list[str]] = {}
        for o in eligible:
            if o.family != CHECKPOINT_FAMILY:
                members.setdefault(o.family, []).append(o.member_id)
        if not members:
            raise ValueError(f"partition {partition.name!r} has no members that play {ruleset_id}")
        checkpoints = sorted(o.member_id for o in eligible if o.family == CHECKPOINT_FAMILY)
        out = []
        for j in range(count):
            rng = random.Random(f"pop-lineup-v1|{partition.partition_hash()}|{ruleset_id}|{opponents}|{seed}|{j}")
            opps = tuple(by_id[e] for e in draw_entries(rng, partition.name, members, checkpoints, opponents))
            lid = "L_" + canonical_hash([o.member_id for o in opps] + [ruleset_id, j],
                                        domain="strategy-lineup-id-v1")[:12]
            out.append(Lineup(lid, opps))
        return out
