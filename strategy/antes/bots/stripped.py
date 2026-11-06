"""Development candidate: the final bot with the blueprint's preflop pot read without the antes."""

from strategy.final.runtime import FinalBot


def make_bot():
    bot = FinalBot()
    bot.teacher.policy("practice_nlhe").ante_view = "stripped"
    return bot
