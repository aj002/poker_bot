"""The competition bot (exploiting blueprint on NLHE, generic fallback elsewhere). See :mod:`strategy.final.runtime`."""

from strategy.final.runtime import FinalBot


def make_bot():
    return FinalBot()
