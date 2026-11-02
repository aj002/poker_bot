"""Blueprint bot with the confidence-capped push/fold leaf. See :mod:`strategy.blueprint.runtime`."""

from strategy.blueprint.runtime import BlueprintBot


def make_bot():
    return BlueprintBot("capped")
