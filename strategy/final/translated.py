"""The final bot with the blueprint's off-tree bet translation (:mod:`strategy.blueprint.translate`).

Not part of the shipped bundle: ``strategy.final.package --override translate=...`` builds
a bundle around :class:`TranslatingFinalBot` for experiments."""

from __future__ import annotations

from pathlib import Path

from strategy.blueprint.translate import Translating, check_mode
from strategy.exploit.runtime import ExploitBot
from strategy.final.runtime import NAME, WARM_GENERIC, FinalBot
from strategy.policy.generic import GenericBot, _Game


class TranslatingExploitBot(Translating, ExploitBot):
    """:class:`ExploitBot` with translation; ``overrides["translate"]`` picks the mode."""

    def __init__(self, pack: str | Path | None = None, exploit_pack: str | Path | None = None,
                 overrides: dict | None = None) -> None:
        self.translate = check_mode(str((overrides or {}).get("translate", "off")))
        super().__init__(pack=pack, exploit_pack=exploit_pack, overrides=overrides)


class TranslatingFinalBot(FinalBot):
    def __init__(self, pack: str | Path | None = None, exploit_pack: str | Path | None = None,
                 overrides: dict | None = None) -> None:
        self.teacher = TranslatingExploitBot(pack=pack, exploit_pack=exploit_pack, overrides=overrides)
        self.generic = GenericBot(name=NAME)
        for ruleset_id in WARM_GENERIC:
            try:
                _Game(ruleset_id)
            except Exception:  # no bundled reveal variant, or one the card layer cannot read
                pass
