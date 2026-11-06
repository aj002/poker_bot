"""Candidate: the final bot with ``antes="overlay"`` (:mod:`strategy.antes.runtime`)."""

from strategy.antes.runtime import AnteFinalBot


def make_bot():
    return AnteFinalBot(overrides={"antes": "overlay"})
