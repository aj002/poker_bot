"""Blueprint bot: point-aware push/fold leaf, then the MCCFR blueprint. See :mod:`strategy.blueprint.runtime`."""

from strategy.blueprint.runtime import BlueprintBot


def make_bot():
    return BlueprintBot("points")
