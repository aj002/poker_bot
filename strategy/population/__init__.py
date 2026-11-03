"""Opponent population: sampled style families, a frozen partitioned registry and lineup pools.

See ``docs/opponent-population.md``. Experiment specs use the registry through
``"source": "strategy.population:PopulationSource"`` (the evaluation harness's partition
protocol); training code can also use :func:`get_lineup_pool`. Exports load lazily:
sandboxed workers import only the families and policy, not the registry.
"""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "CHECKPOINT_FAMILY": "families", "FAMILIES": "families", "FAMILY_LIST": "families", "HOLDOUT_ONLY": "families",
    "Family": "families", "PopulationBot": "policy", "REGISTRY_VERSION": "registry", "CheckpointLedger": "registry",
    "CheckpointSlot": "registry", "Lineup": "registry", "LineupPool": "registry", "Member": "registry",
    "Registry": "registry", "default_registry": "registry", "get_lineup_pool": "registry", "verify_lock": "registry",
    "PopulationSource": "source", "HandhqPopulationSource": "handhq", "HandhqRegistry": "handhq",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"strategy.population.{_EXPORTS[name]}"), name)
