"""Development candidate: the ante push/fold tables and the blueprint's preflop pot read without the antes."""

from strategy.antes.lookup import AntePushFoldBot
from strategy.final.runtime import FinalBot


def make_bot():
    bot = FinalBot()
    bot.teacher._leaf = AntePushFoldBot("points")
    bot.teacher.policy("practice_nlhe").ante_view = "stripped"
    return bot
