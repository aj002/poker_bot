"""The competition bot (exploiting blueprint on NLHE with first-class antes and the calling-station
response, generic fallback elsewhere).

See :mod:`strategy.final.runtime`, :mod:`strategy.antes.runtime` and :mod:`strategy.exploit.station`;
the default bundle of :mod:`strategy.final.package` builds the same bot.
"""

from strategy.exploit.station import StationFinalBot


def make_bot():
    return StationFinalBot(overrides={"antes": "tables", "station": "value"})
