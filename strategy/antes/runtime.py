"""The final bot with first-class antes (decision-time path; no numpy, no solver).

Not part of the shipped bundle: ``strategy.final.package --override antes=MODE`` builds a
bundle around :class:`AnteFinalBot`. Hands without an ante play exactly as the final bot.
Modes (:data:`MODES`), each adding to the one before:

``stripped``  the blueprint reads a preflop pot without the antes (the trainer's
              context: :mod:`strategy.blueprint.kernels` keys preflop decisions on the
              blinds and bets only), so an unopened pot keeps its unopened price;
``tables``    preflop push/fold spots of a hand with an ante play from the posting
              mode's ante table (:class:`~strategy.antes.lookup.AntePushFoldBot`);
``overlay``   preflop blueprint lookups in a hand with an ante read the ante-trained
              overlay pack (:mod:`strategy.antes.blueprint`) first, when its row is at
              the same or a finer back-off level than the pack's.

The antes are the preflop ``settled_pot``: before the flop nothing else is settled.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from arena.agents.sdk import Observation
from strategy.blueprint.policy import PolicyPack
from strategy.blueprint.runtime import WARM_PASSES, BlueprintPolicy, HandReplay
from strategy.exploit.runtime import ExploitBot
from strategy.final.runtime import NAME, WARM_GENERIC, FinalBot
from strategy.policy.generic import GenericBot, _Game
from strategy.tables import store

MODES = ("off", "stripped", "tables", "overlay")
WARM_DIR = Path(__file__).resolve().parent / "data"


def check_mode(mode: str) -> str:
    if mode not in MODES:
        raise ValueError(f"antes is one of {MODES}")
    return mode


def ante_warm_observations(ruleset_id: str) -> list[str]:
    """Recorded observations of hands with an ante (JSON; :mod:`strategy.antes.warm`)."""
    path = WARM_DIR / f"warm-{ruleset_id}.json.gz"
    return json.loads(gzip.decompress(path.read_bytes())) if path.exists() else []


def overlay_path(ruleset_id: str) -> Path:
    return store.table_dir() / f"blueprint-antes-{ruleset_id}.npz"


class _Stripped:
    """An observation whose ``pot_total`` leaves out the antes (attribute access delegates)."""

    __slots__ = ("_obs", "pot_total")

    def __init__(self, obs) -> None:
        self._obs = obs
        self.pot_total = obs.pot_total - obs.settled_pot

    def __getattr__(self, name):
        return getattr(self._obs, name)


class OverlayPack:
    """A pack whose lookups prefer ``overlay`` while ``active`` (set per decision by :class:`AntePolicy`)."""

    __slots__ = ("base", "overlay", "active")

    def __init__(self, base: PolicyPack, overlay: PolicyPack) -> None:
        self.base = base
        self.overlay = overlay
        self.active = False

    def __getattr__(self, name):
        return getattr(self.base, name)

    def lookup(self, key: int):
        got = self.base.lookup(key)
        if self.active:
            o = self.overlay.lookup(key)
            if o is not None and (got is None or o[1] <= got[1]):
                return o
        return got


class AntePolicy(BlueprintPolicy):
    def __init__(self, ruleset_id: str, pack: PolicyPack | None = None, overlay: PolicyPack | None = None) -> None:
        super().__init__(ruleset_id, pack)
        if overlay is not None:
            self.pack = OverlayPack(self.pack, overlay)

    def features(self, obs, rp: HandReplay | None = None):
        ante = obs.street == "preflop" and obs.settled_pot > 0
        pack = self.pack
        if isinstance(pack, OverlayPack):
            pack.active = ante
        return super().features(_Stripped(obs) if ante else obs, rp)


class Antes:
    """Mixin for a :class:`~strategy.blueprint.runtime.BlueprintBot` subclass (set ``antes`` before its constructor)."""

    antes = "off"
    _ante_overlay: str | Path | None = None

    def policy(self, ruleset_id: str) -> BlueprintPolicy:
        p = self._policies.get(ruleset_id)
        if p is None:
            pack = PolicyPack(self._pack_path) if self._pack_path else None
            if self.antes == "off":
                p = BlueprintPolicy(ruleset_id, pack)
            else:
                overlay = (PolicyPack(self._ante_overlay or overlay_path(ruleset_id))
                           if self.antes == "overlay" else None)
                p = AntePolicy(ruleset_id, pack, overlay)
            self._policies[ruleset_id] = p
        return p

    def warm(self, ruleset_id: str) -> None:
        """The bot's warm-up, then the recorded ante hands (``strategy/antes/data``) the same way."""
        if self.antes in ("tables", "overlay") and not hasattr(self._leaf, "ante_stats"):
            from strategy.antes.lookup import AntePushFoldBot

            self._leaf = AntePushFoldBot(self.leaf)
        super().warm(ruleset_id)
        recorded = ante_warm_observations(ruleset_id) if self.antes != "off" else []
        for _ in range(WARM_PASSES if recorded else 0):
            for raw in recorded:
                self.decide(Observation.model_validate_json(raw), {"seed": "warm"})
        if recorded:
            self.policy(ruleset_id)._pressure.clear()
            ctx = getattr(self, "_ctx", None)
            if ctx is not None:
                ctx.clear()
            if hasattr(self._leaf, "ante_stats"):
                self._leaf.ante_stats = {k: 0 for k in self._leaf.ante_stats}
            self.stats = self._fresh_stats()


class AnteExploitBot(Antes, ExploitBot):
    """:class:`ExploitBot` with first-class antes; ``overrides["antes"]`` picks the mode."""

    def __init__(self, pack: str | Path | None = None, exploit_pack: str | Path | None = None,
                 overrides: dict | None = None, ante_overlay: str | Path | None = None) -> None:
        self.antes = check_mode(str((overrides or {}).get("antes", "off")))
        self._ante_overlay = ante_overlay
        super().__init__(pack=pack, exploit_pack=exploit_pack, overrides=overrides)


class AnteFinalBot(FinalBot):
    """``ante_overlay``: the overlay pack of ``antes=overlay`` (default: the table directory's)."""

    def __init__(self, pack: str | Path | None = None, exploit_pack: str | Path | None = None,
                 overrides: dict | None = None, ante_overlay: str | Path | None = None) -> None:
        self.teacher = AnteExploitBot(pack=pack, exploit_pack=exploit_pack, overrides=overrides,
                                      ante_overlay=ante_overlay)
        self.generic = GenericBot(name=NAME)
        for ruleset_id in WARM_GENERIC:
            try:
                _Game(ruleset_id)
            except Exception:  # no bundled reveal variant, or one the card layer cannot read
                pass


def make_bot(**kwargs) -> AnteFinalBot:
    return AnteFinalBot(**kwargs)
