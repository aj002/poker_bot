"""Bot seats at live tables: milestone 4 baselines, opponent styles and operator-registered bot files.

Baselines and opponent styles (:mod:`arena.agents.styles`) are trusted,
repository-owned files and run in-process
(:class:`~arena.agents.worker.InProcessSeat`). A registered file is untrusted: it
is staged once, hashed, loaded only inside a sandboxed worker
(:class:`~arena.agents.worker.WorkerSeat`) and runs under the table's 1 ms bot
budget with the same fallbacks as the benchmark runner. Registering a file and
seating it are operator-only.

The 3b policy ids stay accepted as aliases so existing clients keep working.
"""

from __future__ import annotations

import json
import secrets
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from arena.agents.baselines import BASELINES, baseline_info, load_baseline
from arena.agents.sdk import TableInfo, UnsupportedRuleset, load_bot
from arena.agents.styles import PROFILES, STYLES, style_artifact_hash
from arena.agents.worker import (
    AgentLoadError,
    Artifact,
    Decision,
    InProcessSeat,
    SeatAgent,
    WorkerPool,
    WorkerSeat,
    WorkerStartError,
    artifact_hash,
)
from arena.benchmark.runner import policy_seed
from arena.domain.models import RULESET_IDS, TournamentConfig
from arena.table.projections import observation
from arena.table.service import ActionRejected, Table

Source = Literal["baseline", "registered", "style", "strategy"]
BUILTIN_SOURCES = ("baseline", "style", "strategy")
"""Repository-owned bots: trusted, run in-process at live tables and allowed in in-process runs."""

BASELINE_TEXT: dict[str, tuple[str, str]] = {
    "check_fold": ("Check / fold", "Checks when free, otherwise folds."),
    "calling_station": ("Calling station", "Checks when free, otherwise calls."),
    "legal_random": ("Legal random", "A random legal action kind, then a random legal raise size."),
    "nlhe_heuristic": ("NLHE heuristic", "Simple hand-strength heuristic. Hold'em only; not a strong player."),
}

FINAL_BOT_ID = "final_bot"
FINAL_BOT_RULESETS = (*RULESET_IDS, "reveal_variant")
"""Kept equal to ``strategy.final.runtime.RULESETS`` by a test; importing that module here would load the whole bot."""
FINAL_BOT_ENTRY = Path(__file__).resolve().parents[2] / "strategy" / "final" / "bots" / "final.py"
FINAL_SUBMISSION = FINAL_BOT_ENTRY.parents[1] / "submission.json"

LEGACY_ALIASES = {"check_or_fold": "check_fold", "check_or_call": "calling_station"}


class BotRegistrationError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class BotSpec:
    id: str
    label: str
    description: str
    source: Source
    artifact_hash: str
    rulesets: tuple[str, ...]
    state_mode: str
    bot_name: str
    run_path: Path | None = None
    """What the benchmark runner stages: the baseline file, or the registered copy."""
    artifact: Artifact | None = None
    original_path: str | None = None
    registered_at: str | None = None
    style: str | None = None

    def public(self, *, operator: bool = False, alias_of: str | None = None, alias_id: str | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": alias_id or self.id,
            "label": self.label,
            "description": self.description,
            "source": self.source,
            "artifact_hash": self.artifact_hash,
            "rulesets": list(self.rulesets),
            "state_mode": self.state_mode,
            "bot_name": self.bot_name,
            "alias_of": alias_of,
        }
        if self.style is not None:
            out["style"] = self.style
        if operator and self.source == "registered":
            out["original_path"] = self.original_path
            out["registered_at"] = self.registered_at
        return out


def _baseline_spec(name: str) -> BotSpec:
    info = baseline_info(name)
    label, description = BASELINE_TEXT.get(name, (name, ""))
    return BotSpec(
        id=name, label=label, description=description, source="baseline",
        artifact_hash=artifact_hash(BASELINES[name]), rulesets=tuple(info.rulesets),
        state_mode=info.state_mode, bot_name=info.name, run_path=BASELINES[name],
    )


def _style_spec(style: str) -> BotSpec:
    profile = PROFILES[style]
    return BotSpec(
        id=profile.policy_id, label=profile.name, description=profile.description, source="style",
        artifact_hash=style_artifact_hash(style), rulesets=profile.rulesets, state_mode="explicit",
        bot_name=profile.policy_id, run_path=STYLES[style], style=style,
    )


def _final_bot_spec() -> BotSpec | None:
    """The competition bot as a trusted, in-process seat: no 1 ms deadline, so a busy laptop never makes it fold.

    Absent when the repository has no ``strategy/final`` (an ``arena``-only install).
    """
    if not FINAL_BOT_ENTRY.is_file():
        return None
    try:
        bundle = json.loads(FINAL_SUBMISSION.read_text())["submission_bundle_sha256"]
    except (OSError, KeyError, ValueError):
        bundle = artifact_hash(FINAL_BOT_ENTRY)
    return BotSpec(
        id=FINAL_BOT_ID, label="Final bot", source="strategy", artifact_hash=bundle,
        description=f"The competition entry (bundle {bundle[:8]}): exploiting blueprint on Hold'em, generic play on "
                    "other variants. Runs in the server process with no decision deadline; needs no numba.",
        rulesets=FINAL_BOT_RULESETS, state_mode="explicit", bot_name=FINAL_BOT_ID, run_path=FINAL_BOT_ENTRY,
    )


@dataclass
class BotRegistry:
    """Bot choices for live seats; owns the sandbox worker pool for registered files."""

    pool_size: int = 4
    baselines: dict[str, BotSpec] = field(init=False)
    styles: dict[str, BotSpec] = field(init=False)
    strategy: dict[str, BotSpec] = field(init=False)
    registered: dict[str, BotSpec] = field(init=False, default_factory=dict)
    _pool: WorkerPool | None = field(init=False, default=None)
    _lock: threading.Lock = field(init=False, default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.baselines = {name: _baseline_spec(name) for name in BASELINES}
        self.styles = {spec.id: spec for spec in (_style_spec(sid) for sid in STYLES)}
        final = _final_bot_spec()
        self.strategy = {final.id: final} if final is not None else {}

    # -- lookup
    def resolve(self, policy_id: str | None) -> BotSpec | None:
        if policy_id is None:
            return None
        policy_id = LEGACY_ALIASES.get(policy_id, policy_id)
        return (self.baselines.get(policy_id) or self.styles.get(policy_id) or self.strategy.get(policy_id)
                or self.registered.get(policy_id))

    def catalog(self, *, operator: bool = False) -> list[dict[str, Any]]:
        out = [s.public() for s in self.baselines.values()]
        out += [self.baselines[target].public(alias_of=target, alias_id=alias) for alias, target in LEGACY_ALIASES.items()]
        out += [s.public() for s in self.styles.values()]
        out += [s.public() for s in self.strategy.values()]
        if operator:
            out += [s.public(operator=True) for s in self.registered.values()]
        return out

    # -- registration (operator only; enforced by the caller)
    def pool(self) -> WorkerPool:
        with self._lock:
            if self._pool is None:
                try:
                    self._pool = WorkerPool(size=self.pool_size)
                except WorkerStartError as exc:
                    raise BotRegistrationError("sandbox_unavailable", str(exc)) from None
            return self._pool

    def register(self, path: str) -> BotSpec:
        src = Path(path).expanduser()
        if not src.is_absolute():
            raise BotRegistrationError("invalid_path", "give an absolute path to a bot file or package directory")
        if not src.exists():
            raise BotRegistrationError("not_found", "no file or directory at that path")
        pool = self.pool()
        try:
            artifact = pool.stage(src)
        except (OSError, ValueError) as exc:
            raise BotRegistrationError("invalid_artifact", str(exc)[:300]) from None
        existing = next((s for s in self.registered.values() if s.artifact_hash == artifact.sha256), None)
        if existing is not None:
            return existing
        info = self._probe(artifact)
        spec = BotSpec(
            id="file_" + artifact.sha256[:12],
            label=info.name,
            description=f"Registered file {src.name}; runs in a sandboxed worker under the 1 ms budget.",
            source="registered", artifact_hash=artifact.sha256, rulesets=tuple(info.rulesets),
            state_mode=info.state_mode, bot_name=info.name, artifact=artifact,
            run_path=artifact.entry if src.is_file() else artifact.root, original_path=str(src),
            registered_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        self.registered[spec.id] = spec
        return spec

    def _probe(self, artifact: Artifact):
        """Load the bot once in a sandboxed worker to read its declared name and rulesets."""
        config = TournamentConfig(ruleset_id=RULESET_IDS[0], initial_player_count=2)
        info = TableInfo(table_id="t_probe", hero_seat=config.resolved_seats[0], ruleset_id=config.ruleset_id,
                         config=config.resolved(), config_hash=config.config_hash(), seats=list(config.resolved_seats),
                         decision_budget_ns=config.bot_budget_ns, policy_seed="probe")
        seat = WorkerSeat(config.resolved_seats[0], artifact, info, self.pool().acquire)
        try:
            seat.start()
        except UnsupportedRuleset:
            pass
        except (AgentLoadError, WorkerStartError) as exc:
            raise BotRegistrationError("bot_load_failed", str(exc)[:300]) from None
        finally:
            seat.close()
        assert seat.info is not None
        return seat.info

    # -- live seats
    def agent(self, spec: BotSpec, seat: int, info: TableInfo) -> SeatAgent:
        if spec.source == "baseline":
            bot, bot_info = load_baseline(spec.id)
            return InProcessSeat(seat, bot, bot_info, info, spec.artifact_hash)
        if spec.source in ("style", "strategy"):
            assert spec.run_path is not None
            bot, bot_info = load_bot(spec.run_path, module_name=f"arena_{spec.id}")
            return InProcessSeat(seat, bot, bot_info, info, spec.artifact_hash)
        assert spec.artifact is not None
        return WorkerSeat(seat, spec.artifact, info, self.pool().acquire, budget_ns=info.decision_budget_ns)

    def close(self) -> None:
        with self._lock:
            if self._pool is not None:
                self._pool.close()
                self._pool = None


def live_table_info(config: TournamentConfig, table_id: str, seat: int, seed: str | None) -> TableInfo:
    return TableInfo(
        table_id=table_id, hero_seat=seat, ruleset_id=config.ruleset_id, config=config.resolved(),
        config_hash=config.config_hash(), seats=list(config.resolved_seats),
        decision_budget_ns=config.bot_budget_ns,
        policy_seed=policy_seed(seed if seed is not None else secrets.token_hex(16), seat),
    )


def bot_turn(table: Table, agent: SeatAgent) -> tuple[str, str | None]:
    """One bot decision, consumed exactly once: by the accepted action or by a fallback.

    Returns the decision id and the bot's private note when its action was accepted.
    """
    seat = agent.seat
    agent.prepare()
    obs = observation(table, seat)
    try:
        decision = agent.decide(obs)
    except Exception as exc:  # noqa: BLE001 - a broken seat still consumes the decision once
        decision = Decision(None, "bot_exception", {"detail": type(exc).__name__})
    if decision.fallback is None:
        try:
            table.submit(seat, decision.action, hand_id=obs.hand_id, decision_id=obs.decision_id,
                         expected_state_version=obs.state_version)
        except ActionRejected as exc:
            decision.fallback = exc.code
            decision.rejected = {"code": exc.code}
        else:
            agent.accept()
            return obs.decision_id, decision.note
    agent.reject()
    agent.stats.fallbacks[decision.fallback] += 1
    table.apply_fallback(seat, decision.fallback, decision.rejected)
    return obs.decision_id, None
