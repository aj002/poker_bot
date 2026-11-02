"""The single deployed decision path: observation -> spot -> policy -> legal action.

:class:`RuntimeBot` is an explicit-state SDK bot. Its committed state is small JSON
(the policy seed, the draw counter, per-opponent counters, the last counted event
``seq`` and the fallback count), so a worker restored after a crash replays exactly.
Between decisions it also keeps a live :class:`~strategy.policy.spot.TableMemory` and
reuses it when the incoming state is the one it last returned; otherwise (a restore,
or an uncommitted reply) it rebuilds the memory from the state and the current hand's
public events.

Every failure on the decision path - an exception anywhere, or an action the legal set
does not allow - plays :func:`~strategy.policy.fallback.fallback_action` and counts it
in ``state["fb"]``, so the deployed-path gate can require zero internal fallbacks as
well as zero arena fallbacks.

``gc_mode="freeze"`` (used by packaged bots) collects and freezes the heap once the
table assets are loaded, so later collections never rescan the tables; the decision
path itself creates no reference cycles.
"""

from __future__ import annotations

import gc
import hashlib
from collections.abc import Mapping
from typing import Any

from arena.agents.sdk import ExplicitStateBot, with_note
from strategy.policy.baseline import BaselinePolicy
from strategy.policy.fallback import fallback_action, is_legal, legalize
from strategy.policy.spot import TableMemory, variant_assets

STATE_VERSION = 1
RULESETS = ("practice_nlhe", "uncertainty_shortdeck", "uncertainty_plo4", "reveal_variant")
_SCALE = 1.0 / 18446744073709551616.0


class Draws:
    """Uniform draws from the seat's policy seed, counter mode (blake2b)."""

    __slots__ = ("prefix", "n")

    def __init__(self, seed: str, n: int) -> None:
        self.prefix = seed.encode() + b":"
        self.n = n

    def __call__(self) -> float:
        self.n += 1
        h = hashlib.blake2b(self.prefix + str(self.n).encode(), digest_size=8).digest()
        return int.from_bytes(h, "big") * _SCALE


class RuntimeBot(ExplicitStateBot):
    name = "strategy_baseline"
    rulesets = RULESETS

    def __init__(self, params: Mapping[str, float] | None = None, *, name: str | None = None,
                 gc_mode: str = "none", notes: bool = False) -> None:
        if gc_mode not in ("none", "freeze"):
            raise ValueError(f"unknown gc_mode {gc_mode!r}")
        self.policy = BaselinePolicy(params)
        if name:
            self.name = name
        self.gc_mode = gc_mode
        self.notes = notes
        self._mem: TableMemory | None = None
        self._key: tuple[int, int, int] | None = None

    def initial_state(self, table: Any) -> dict[str, Any]:
        variant_assets(table.ruleset_id)
        self._mem, self._key = None, None
        if self.gc_mode == "freeze":
            gc.collect()
            gc.freeze()
        return {"v": STATE_VERSION, "seed": table.policy_seed, "n": 0, "d": 0, "seq": -1, "fb": 0, "c": {}}

    def _memory(self, obs: Any, state: Mapping[str, Any]) -> TableMemory:
        key = (state["d"], state["n"], state["seq"])
        if self._mem is not None and key == self._key and self._mem.hero == obs.hero_seat:
            return self._mem
        mem = TableMemory(obs.ruleset_id, obs.hero_seat, obs.seats, state["c"], state["seq"])
        self._mem = mem
        return mem

    def decide(self, observation: Any, state: Any) -> tuple[Any, dict[str, Any]]:
        legal = observation.legal
        draws = Draws(state["seed"], state["n"])
        fb = state["fb"]
        tag = ""
        mem: TableMemory | None = None
        try:
            mem = self._memory(observation, state)
            spot = mem.spot(observation)
            intent = self.policy.decide(spot, draws)
            action = legalize(intent.kind, intent.amount, legal, spot.hero_stack + spot.hero_street)
            if not is_legal(action, legal):
                raise ValueError(f"illegal {action!r}")
            tag = intent.tag
        except Exception as exc:  # the decision is consumed by the legal fallback and counted
            action = fallback_action(legal)
            fb += 1
            tag = f"fallback:{type(exc).__name__}"
            self._mem = mem = None
        new = {"v": STATE_VERSION, "seed": state["seed"], "n": draws.n, "d": state["d"] + 1,
               "seq": mem.counted_seq if mem is not None else state["seq"], "fb": fb,
               "c": mem.counters_state() if mem is not None else state["c"]}
        self._key = (new["d"], new["n"], new["seq"]) if mem is not None else None
        return (with_note(action, tag) if self.notes else action), new


def make_bot(params: Mapping[str, float] | None = None, **kwargs: Any) -> RuntimeBot:
    return RuntimeBot(params, **kwargs)
