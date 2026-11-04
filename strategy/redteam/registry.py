"""The ``redteam-v1`` opponent family: three blind red-team bots, registered in validation only.

The bots were written by independent workers who never saw ``strategy/``; they are fixed
code, not sampled parameters, so this registry is three members of one family
(:data:`FAMILY`), each pinned by the content hash of its bot artifact
(``redteam.lock.json``). Specs name it as
``"source": "strategy.redteam.registry:RedteamPopulationSource"``.

* ``validation`` holds the three members. ``train`` and ``holdout`` are **refused**: the
  family exists to test a finished bot, so nothing may be fitted to it and the holdout may
  never see it. :func:`verify_lock` and ``tests/strategy/test_redteam.py`` check both.
* pop-v1 and ``pop-v1+handhq-v1`` are untouched (members, locks and holdout are
  byte-identical); the red-team members are also never added to their registries.

    python -m strategy.redteam.registry --write-lock
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from arena.agents.worker import artifact_hash
from arena.domain.models import RULESET_IDS
from strategy.eval.population import Lineup, OpponentRef, Partition
from strategy.rules.canonical import canonical_hash

REPO = Path(__file__).resolve().parents[2]
VERSION = "redteam-v1"
FAMILY = "redteam-v1"
PARTITION = "validation"
"""The only partition this family exists in."""
LOCK_PATH = Path(__file__).resolve().parent / "redteam.lock.json"
NLHE = ("practice_nlhe",)

MEMBERS: Mapping[str, str] = MappingProxyType({
    "rt-pressure": "redteam/pressure/bot.py",
    "rt-equilibrium": "redteam/equilibrium",
    "rt-adaptive": "redteam/adaptive/bot.py",
})
"""Member id -> bot artifact path, relative to the repository root."""
SHORT = MappingProxyType({"rt-pressure": "pressure", "rt-equilibrium": "equilibrium", "rt-adaptive": "adaptive"})


def member_hash(member_id: str) -> str:
    return artifact_hash(REPO / MEMBERS[member_id])


def registry_hash() -> str:
    return canonical_hash({"version": VERSION, "family": FAMILY, "partitions": [PARTITION],
                           "members": {m: member_hash(m) for m in sorted(MEMBERS)}}, domain="redteam-registry-v1")


def refs() -> tuple[OpponentRef, ...]:
    return tuple(OpponentRef(m, FAMILY, MEMBERS[m], NLHE, member_hash(m)) for m in sorted(MEMBERS))


def lock() -> dict[str, Any]:
    return {"version": VERSION, "family": FAMILY, "partitions": [PARTITION], "registry_hash": registry_hash(),
            "members": {m: {"path": MEMBERS[m], "artifact_sha256": member_hash(m), "partition": PARTITION}
                        for m in sorted(MEMBERS)},
            "rulesets": list(NLHE), "holdout_members": [], "train_members": []}


def verify_lock(path: Path = LOCK_PATH) -> None:
    frozen = json.loads(path.read_text())
    current = lock()
    if frozen != current:
        diff = sorted(k for k in set(frozen) | set(current) if frozen.get(k) != current.get(k))
        raise AssertionError(f"red-team registry differs from {path.name} in {diff}: bump VERSION and re-lock")


class RedteamPopulationSource:
    """:class:`~strategy.eval.population.PartitionSource` over the red-team members (validation only)."""

    registry_id = f"strategy.population/{VERSION}"

    def __init__(self, check_lock: bool = True) -> None:
        if check_lock and LOCK_PATH.exists():
            verify_lock()

    def registry_hash(self) -> str:
        return registry_hash()

    def partition(self, name: str) -> Partition:
        if name != PARTITION:
            raise PermissionError(f"{VERSION} is registered in the {PARTITION} partition only; "
                                  f"{name!r} is refused (never fit to train, never touch holdout)")
        return Partition(name, self.registry_id, registry_hash(), f"{VERSION}|{name}", refs(),
                         {"stub": False, "registry_version": VERSION, "families": [FAMILY],
                          "holdout_only": [], "validation_only_pairs": [], "checkpoints": []})

    def sample_lineups(self, partition: Partition, *, ruleset_id: str, opponents: int, count: int,
                       seed: str) -> list[Lineup]:
        """Each seat drawn uniformly, with replacement, from the three members (so every bot appears in multiples)."""
        if partition.registry_id != self.registry_id:
            raise ValueError(f"partition {partition.name!r} belongs to {partition.registry_id}, not {self.registry_id}")
        if ruleset_id not in RULESET_IDS:
            raise ValueError(f"unknown ruleset {ruleset_id!r}")
        pool = sorted(partition.eligible(ruleset_id), key=lambda m: m.member_id)
        if not pool:
            raise ValueError(f"{VERSION} has no members that play {ruleset_id}")
        out = []
        for j in range(count):
            rng = random.Random(f"redteam-lineup-v1|{partition.partition_hash()}|{ruleset_id}|{opponents}|{seed}|{j}")
            opps = tuple(pool[rng.randrange(len(pool))] for _ in range(opponents))
            lid = "L_" + canonical_hash([o.member_id for o in opps] + [ruleset_id, j],
                                        domain="strategy-lineup-id-v1")[:12]
            out.append(Lineup(lid, opps))
        return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write-lock", action="store_true")
    a = ap.parse_args(argv)
    cur = lock()
    if a.write_lock:
        LOCK_PATH.write_text(json.dumps(cur, indent=1, sort_keys=True) + "\n")
        print(f"wrote {LOCK_PATH} ({len(cur['members'])} members, registry {cur['registry_hash'][:12]})")
    else:
        print(json.dumps(cur, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
