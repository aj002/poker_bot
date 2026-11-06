"""Development candidate: the stripped preflop read plus an ante-trained preflop overlay pack.

``ANTE_OVERLAY`` names the pack (default ``artifacts/antes/overlay.npz``); ``ANTE_TABLES=1``
also swaps in the ante push/fold tables.
"""

import os
from pathlib import Path

from strategy.blueprint.policy import PolicyPack
from strategy.final.runtime import FinalBot

DEFAULT = Path(__file__).resolve().parents[3] / "artifacts" / "antes" / "overlay.npz"


def make_bot():
    bot = FinalBot()
    policy = bot.teacher.policy("practice_nlhe")
    policy.ante_view = "stripped"
    policy.overlay = PolicyPack(Path(os.environ.get("ANTE_OVERLAY", DEFAULT)))
    if os.environ.get("ANTE_TABLES") == "1":
        from strategy.antes.lookup import AntePushFoldBot

        bot.teacher._leaf = AntePushFoldBot("points")
    return bot
