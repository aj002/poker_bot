"""Local Python bot SDK (spec section 6.1). This is our contract, not Sequoia's Agent API.

A bot file defines ``make_bot()`` returning one of two kinds of bot:

* :class:`ExplicitStateBot`: ``decide(observation, state) -> (action, new_state)``.
  ``state`` is JSON, at most :data:`MAX_STATE_BYTES` (64 KiB) in canonical form.
  The arena commits a returned state only together with a valid, on-time action,
  and always hands the bot the committed state decoded from its canonical JSON.
  A crashed or killed worker is restored from that state, so an explicit-state bot
  that keeps everything it needs in ``state`` replays exactly.
* :class:`InstanceStateBot`: ``decide(observation) -> action`` with ordinary Python
  attributes. The process is a cache, not a checkpoint: after a crash or kill the
  bot restarts from a fresh instance and the table records that seat as
  ``restorable=False``.

Every bot declares the rulesets it supports in ``rulesets``. A bot is never seated
at a table with another ruleset: the pairing is rejected before the table starts.

Policy randomness must come from ``TableInfo.policy_seed`` (a per-seat stream that is
independent of the dealer's deck seed) and, for explicit-state bots, live in
``state``. Do not read the wall clock to make decisions.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal, Union

from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr

from arena.domain.models import ALL_RULESET_IDS, Action, LegalActions, RulesetId
from arena.domain.observations import Observation

__all__ = [
    "MAX_NOTE_CHARS",
    "MAX_STATE_BYTES",
    "Action",
    "BotInfo",
    "BotLoadError",
    "ExplicitStateBot",
    "InstanceStateBot",
    "JsonValue",
    "LegalActions",
    "Observation",
    "StateError",
    "TableInfo",
    "UnsupportedRuleset",
    "call",
    "check",
    "check_or_call",
    "check_or_fold",
    "clamp_raise_to",
    "decode_state",
    "encode_state",
    "fold",
    "load_bot",
    "raise_to",
    "require_supported",
    "with_note",
]

JsonValue = Union[None, bool, int, float, str, list["JsonValue"], dict[str, "JsonValue"]]
StateMode = Literal["explicit", "instance"]

MAX_STATE_BYTES = 64 * 1024
TABLE_INFO_SCHEMA_VERSION = 1
NOTE_KEY = "note"
MAX_NOTE_CHARS = 200


class TableInfo(BaseModel):
    """What a bot learns when it is seated. Freshly issued for every table.

    ``table_id`` is an anonymous per-table identifier and seats are physical seat
    numbers, so nothing here identifies an opponent's account or artifact.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal[1] = TABLE_INFO_SCHEMA_VERSION
    table_id: StrictStr
    hero_seat: StrictInt
    ruleset_id: RulesetId
    config: dict[str, Any]
    config_hash: StrictStr
    seats: list[StrictInt]
    decision_budget_ns: StrictInt
    policy_seed: StrictStr


class StateError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def encode_state(value: Any) -> bytes:
    """Canonical JSON for a bot state; raises :class:`StateError` if invalid or over 64 KiB."""
    try:
        data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise StateError("state_invalid", f"state is not JSON: {exc}") from None
    raw = data.encode("ascii")
    if len(raw) > MAX_STATE_BYTES:
        raise StateError("state_too_large", f"state is {len(raw)} bytes; the limit is {MAX_STATE_BYTES}")
    return raw


def decode_state(raw: bytes) -> JsonValue:
    return json.loads(raw)


# ------------------------------------------------------------------------ bots


class _BotBase:
    name: ClassVar[str] = "unnamed-bot"
    rulesets: ClassVar[tuple[RulesetId, ...]] = ()
    state_mode: ClassVar[StateMode]


class ExplicitStateBot(_BotBase):
    """Subclass and implement :meth:`decide` (and usually :meth:`initial_state`)."""

    state_mode: ClassVar[StateMode] = "explicit"

    def initial_state(self, table: TableInfo) -> JsonValue:
        return None

    def decide(self, observation: Observation, state: JsonValue) -> tuple[Action | dict, JsonValue]:
        raise NotImplementedError


class InstanceStateBot(_BotBase):
    """Subclass and implement :meth:`decide`. Not restorable after a crash."""

    state_mode: ClassVar[StateMode] = "instance"

    def start_table(self, table: TableInfo) -> None:
        return None

    def decide(self, observation: Observation) -> Action | dict:
        raise NotImplementedError


Bot = ExplicitStateBot | InstanceStateBot


@dataclass(frozen=True)
class BotInfo:
    name: str
    rulesets: tuple[str, ...]
    state_mode: StateMode

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "rulesets": list(self.rulesets), "state_mode": self.state_mode}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BotInfo:
        return cls(name=str(data["name"]), rulesets=tuple(data["rulesets"]), state_mode=data["state_mode"])


class BotLoadError(Exception):
    pass


class UnsupportedRuleset(Exception):
    def __init__(self, bot: str, ruleset_id: str, supported: tuple[str, ...]) -> None:
        super().__init__(f"bot {bot!r} does not support {ruleset_id!r} (supports: {', '.join(supported) or 'none'})")
        self.bot = bot
        self.ruleset_id = ruleset_id


def bot_info(bot: object) -> BotInfo:
    mode = getattr(bot, "state_mode", None)
    if mode not in ("explicit", "instance") or not callable(getattr(bot, "decide", None)):
        raise BotLoadError("make_bot() must return an ExplicitStateBot or InstanceStateBot")
    rulesets = tuple(getattr(bot, "rulesets", ()) or ())
    unknown = [r for r in rulesets if r not in ALL_RULESET_IDS]
    if not rulesets or unknown:
        raise BotLoadError(f"bot must declare supported rulesets from {ALL_RULESET_IDS}; got {rulesets!r}")
    name = str(getattr(bot, "name", "unnamed-bot"))[:64]
    return BotInfo(name=name, rulesets=rulesets, state_mode=mode)


def require_supported(info: BotInfo, ruleset_id: str) -> None:
    if ruleset_id not in info.rulesets:
        raise UnsupportedRuleset(info.name, ruleset_id, info.rulesets)


def load_bot(path: str | Path, module_name: str | None = None) -> tuple[object, BotInfo]:
    """Import a bot file and call its ``make_bot()``.

    Run this only inside a sandboxed worker for untrusted bots. In-process loading
    is for built-in baselines and diagnostics.
    """
    path = Path(path)
    module_name = module_name or "arena_bot_" + hashlib.sha256(str(path).encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise BotLoadError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    factory: Callable[[], object] | None = getattr(module, "make_bot", None)
    if not callable(factory):
        raise BotLoadError(f"{path} must define make_bot()")
    bot = factory()
    return bot, bot_info(bot)


# --------------------------------------------------------------------- actions


def fold() -> Action:
    return Action(type="fold")


def check() -> Action:
    return Action(type="check")


def call() -> Action:
    return Action(type="call")


def raise_to(amount: int) -> Action:
    """Raise (or bet) to a street total. ``amount`` is not an increment."""
    return Action(type="raise_to", amount=amount)


def with_note(action: Action | dict, note: str) -> dict:
    """``action`` plus a private reasoning note (at most :data:`MAX_NOTE_CHARS` characters).

    The arena strips the note before validating the action, stores it in the table's
    private record, and shows it only in the post-table reviewer replay. A note that is
    not a string is dropped; the action is unaffected either way.
    """
    payload = action.model_dump(exclude_none=True) if isinstance(action, Action) else dict(action)
    payload[NOTE_KEY] = str(note)[:MAX_NOTE_CHARS]
    return payload


def split_note(payload: Any) -> tuple[Any, str | None]:
    """Separate a bot's reply into the action payload and its note (arena side)."""
    if not isinstance(payload, dict) or NOTE_KEY not in payload:
        return payload, None
    rest = {k: v for k, v in payload.items() if k != NOTE_KEY}
    note = payload[NOTE_KEY]
    if not isinstance(note, str):
        return rest, None
    text = "".join(ch if ch.isprintable() else " " for ch in note[:MAX_NOTE_CHARS]).strip()
    return rest, text or None


def check_or_fold(legal: LegalActions) -> Action:
    return check() if legal.check else fold()


def check_or_call(legal: LegalActions) -> Action:
    return check() if legal.check else call()


def clamp_raise_to(legal: LegalActions, target: int) -> Action | None:
    """The legal raise closest to ``target``, or None when no raise is legal."""
    if not legal.can_raise:
        return None
    options: list[int] = []
    if legal.min_full_raise_to is not None and legal.max_raise_to is not None:
        options.append(min(max(target, legal.min_full_raise_to), legal.max_raise_to))
    if legal.short_all_in_raise_to is not None:
        options.append(legal.short_all_in_raise_to)
    if not options:
        return None
    return raise_to(min(options, key=lambda a: (abs(a - target), a)))
