"""Blueprint bot with the chip-EV push/fold leaf. See :mod:`strategy.blueprint.runtime`."""

from strategy.blueprint.runtime import BlueprintBot


def make_bot():
    return BlueprintBot("chips")
