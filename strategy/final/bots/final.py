"""The competition bot (exploiting blueprint on NLHE with first-class antes, generic fallback elsewhere).

See :mod:`strategy.final.runtime` and :mod:`strategy.antes.runtime`; the default bundle of
:mod:`strategy.final.package` builds the same bot.
"""

from strategy.antes.runtime import AnteFinalBot


def make_bot():
    return AnteFinalBot(overrides={"antes": "tables"})
