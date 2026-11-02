"""The competition bot: one entry point, the teacher on NLHE and the generic fallback elsewhere.

* ``practice_nlhe`` is played by :class:`~strategy.exploit.runtime.ExploitBot` (the MCCFR
  blueprint, its point-aware push/fold leaf and the confidence-capped exploitation layer),
  unchanged: the same actions, notes and explicit state as the exploit bundle.
* Every other ruleset the rules adapter can describe (short deck, PLO, a build-day
  ``reveal_variant`` bundled under ``strategy/reveal/``) is played by
  :class:`~strategy.policy.generic.GenericBot`, which reads everything it needs from the
  variant spec. No Hold'em table is reused on another variant.

Both sides keep their own explicit state; the ruleset is fixed for a table, so the two
never mix. An exception on the teacher path plays the constant-time legal fallback and
counts it in ``state["fb"]`` (the deployed gate's internal-fallback check), instead of
surfacing as an arena fallback. Nothing here reads the clock: the same seeds give the
same actions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from arena.agents.sdk import ExplicitStateBot, with_note
from strategy.exploit.runtime import ExploitBot
from strategy.policy.fallback import fallback_action
from strategy.policy.generic import RULESETS as GENERIC_RULESETS
from strategy.policy.generic import GenericBot, _Game

TEACHER_RULESETS = ("practice_nlhe",)
RULESETS = tuple(dict.fromkeys((*TEACHER_RULESETS, *GENERIC_RULESETS)))
WARM_GENERIC = ("uncertainty_shortdeck", "uncertainty_plo4", "reveal_variant")
NAME = "final_bot"


class FinalBot(ExplicitStateBot):
    rulesets = RULESETS
    name = NAME

    def __init__(self, pack: str | Path | None = None, exploit_pack: str | Path | None = None,
                 overrides: dict | None = None) -> None:
        self.teacher = ExploitBot(pack=pack, exploit_pack=exploit_pack, overrides=overrides)
        self.generic = GenericBot(name=NAME)
        # Builds the generic path's per-variant caches (rank tables, starting-hand CDFs) in the
        # untimed load: a restored worker goes straight to a timed decision without initial_state.
        for ruleset_id in WARM_GENERIC:
            try:
                _Game(ruleset_id)
            except Exception:  # no bundled reveal variant, or one the card layer cannot read
                pass

    def initial_state(self, table: Any) -> Any:
        if table.ruleset_id in TEACHER_RULESETS:
            return self.teacher.initial_state(table)
        return self.generic.initial_state(table)

    def decide(self, observation: Any, state: Any) -> tuple[Any, Any]:
        if observation.ruleset_id not in TEACHER_RULESETS:
            return self.generic.decide(observation, state)
        try:
            return self.teacher.decide(observation, state)
        except Exception:  # consumed by the legal fallback and counted
            out = dict(state or {})
            out["fb"] = int(out.get("fb", 0)) + 1
            return with_note(fallback_action(observation.legal), "teacher error: legal fallback"), out


def make_bot(**kwargs: Any) -> FinalBot:
    return FinalBot(**kwargs)
