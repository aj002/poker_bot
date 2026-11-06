"""Development candidate: the final bot with ante hands' push/fold played from the ante tables."""

from strategy.antes.lookup import AntePushFoldBot
from strategy.final.runtime import FinalBot


def make_bot():
    bot = FinalBot()
    bot.teacher._leaf = AntePushFoldBot("points")
    return bot
