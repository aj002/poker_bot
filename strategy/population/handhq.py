"""The pop-v1 registry extended with the HandHQ-calibrated families, in train and validation only.

:class:`HandhqRegistry` holds every pop-v1 member unchanged (same ids, parameters and
spec hashes) plus :data:`COUNTS` members of each family in
:data:`~strategy.population.families.HANDHQ_FAMILY_LIST`, drawn with seeds
``"handhq-v1|<partition>|<family>|<i>"`` inside the family's partition region. No new
member is ever in holdout, so the holdout partition holds exactly pop-v1's holdout members
(``handhq.lock.json`` records every member and :func:`verify_lock` checks it).

The extension has its own version, ``pop-v1+handhq-v1``, so its lineups, table seeds and
registry id are its own and never collide with pop-v1 runs. Specs name it as
``"source": "strategy.population:HandhqPopulationSource"``.

    python -m strategy.population.handhq --write-lock
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from arena.domain.canonical import canonical_hash
from arena.domain.models import RULESET_IDS
from strategy.population.families import HANDHQ_FAMILY_LIST
from strategy.population.params import PARTITIONS
from strategy.population.registry import (FINGERPRINT_CONFIG, HERE, REGISTRY_VERSION, CheckpointLedger, Member,
                                          Registry, member_id, member_seed)

EXTENSION_VERSION = "handhq-v1"
VERSION = f"{REGISTRY_VERSION}+{EXTENSION_VERSION}"
COUNTS: Mapping[str, int] = MappingProxyType({"train": 14, "validation": 3})
"""Members per HandHQ family: pop-v1's train and validation counts, and none in holdout."""
LOCK_PATH = HERE / "handhq.lock.json"
FAMILY_IDS = tuple(f.id for f in HANDHQ_FAMILY_LIST)


class HandhqRegistry(Registry):
    def __init__(self, ledger: CheckpointLedger | None = None) -> None:
        super().__init__(REGISTRY_VERSION, ledger if ledger is not None else CheckpointLedger(None))
        self.base_hash = super().hash
        self.version = VERSION
        for fam in HANDHQ_FAMILY_LIST:
            for partition, n in COUNTS.items():
                for i in range(n):
                    mid = member_id(EXTENSION_VERSION, partition, fam.id, i)
                    params = fam.prior.sample(member_seed(EXTENSION_VERSION, partition, fam.id, i), partition)
                    self.members[mid] = Member(mid, fam.id, partition, i, MappingProxyType(params), EXTENSION_VERSION)

    @property
    def hash(self) -> str:
        return canonical_hash({"base": self.base_hash, "version": self.version,
                               "members": {m: x.hash for m, x in sorted(self.members.items())
                                           if x.version == EXTENSION_VERSION}}, domain="pop-registry-ext-v1")

    def lock(self) -> dict[str, Any]:
        base = super().lock()
        base["behaviour"] = {v: extension_fingerprint(self, v) for v in RULESET_IDS}
        base["base_registry_hash"] = self.base_hash
        base["extension_families"] = list(FAMILY_IDS)
        base["extension_counts"] = dict(COUNTS)
        return base


def extension_fingerprint(registry: HandhqRegistry, variant: str) -> str:
    """Every action at short seeded tables of the first member of each HandHQ family (pop-v1's own
    fingerprint covers the other families)."""
    from strategy.population.measure import play

    firsts = [min((m for m in registry.members.values() if m.family == f), key=lambda m: (PARTITIONS.index(m.partition),
                                                                                          m.index))
              for f in FAMILY_IDS]
    seats = [firsts[i % len(firsts)] for i in range(8)]
    table, _ = play([(m.member_id, m.make_bot()) for m in seats], f"{registry.version}|fingerprint|{variant}|0",
                    variant, FINGERPRINT_CONFIG)
    actions = [[e.data["seat"], e.data["action"]["type"], e.data["action"].get("amount")]
               for e in table.log if e.type == "ActionApplied"]
    return canonical_hash(actions, domain="pop-behaviour-v1")


_DEFAULT: HandhqRegistry | None = None


def handhq_registry() -> HandhqRegistry:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = HandhqRegistry()
    return _DEFAULT


def verify_lock(path: Path = LOCK_PATH, registry: HandhqRegistry | None = None) -> None:
    frozen = json.loads(path.read_text())
    current = (registry or handhq_registry()).lock()
    if frozen != current:
        diff = sorted(k for k in set(frozen) | set(current) if frozen.get(k) != current.get(k))
        raise AssertionError(f"registry differs from {path.name} in {diff}: bump EXTENSION_VERSION and re-lock")


def HandhqPopulationSource(bot_dir: str | Path | None = None):
    """:class:`~strategy.population.source.PopulationSource` over :class:`HandhqRegistry`."""
    from strategy.population.source import BOT_ROOT, PopulationSource

    src = PopulationSource(handhq_registry(), BOT_ROOT / VERSION if bot_dir is None else bot_dir, check_lock=False)
    frozen = json.loads(LOCK_PATH.read_text())
    if frozen["registry_hash"] != src.registry.hash:
        raise RuntimeError(f"registry {VERSION} differs from {LOCK_PATH.name}: bump EXTENSION_VERSION and re-lock")
    return src


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write-lock", action="store_true")
    a = ap.parse_args(argv)
    lock = HandhqRegistry().lock()
    if a.write_lock:
        LOCK_PATH.write_text(json.dumps(lock, indent=1, sort_keys=True) + "\n")
        print(f"wrote {LOCK_PATH} ({len(lock['members'])} members, registry {lock['registry_hash'][:12]})")
    else:
        print(json.dumps({k: v for k, v in lock.items() if k != "members"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
