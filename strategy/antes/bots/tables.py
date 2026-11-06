"""Candidate: the final bot with ``antes="tables"`` (:mod:`strategy.antes.runtime`)."""

from strategy.antes.runtime import AnteFinalBot


def make_bot():
    return AnteFinalBot(overrides={"antes": "tables"})
