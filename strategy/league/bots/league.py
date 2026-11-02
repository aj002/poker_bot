"""League bot: the blueprint runtime on the selected league checkpoint pack.

The pack (``strategy/tables/data/league-practice_nlhe.npz``, written by
``python -m strategy.league.runner export``) is the blueprint pack with the league learners'
level-0 overrides; lookup, leaf and fallbacks are the blueprint bot's. No numba.
"""

from strategy.blueprint.runtime import BlueprintBot
from strategy.tables import store


def make_bot():
    bot = BlueprintBot("points", pack=store.table_dir() / "league-practice_nlhe.npz")
    bot.name = "league"
    return bot
