"""Blueprint bot with no push/fold leaf layer. See :mod:`strategy.blueprint.runtime`."""

from strategy.blueprint.runtime import BlueprintBot


def make_bot():
    return BlueprintBot("none")
