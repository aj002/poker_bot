"""Digests of seeded play on ante-free structures, pinned in ``no_ante_identity.json``.

Each case plays seeded tables in-process with a fixed seat mix and digests what an
outside observer could compare across commits: every event log line (the private JSON
form), every observation a bot received (its wire JSON), the config and rules hashes,
the PHH export, and a conformance report. A change that alters play, observations or
records on a structure without antes changes a digest.

``python -m tests.support.identity`` prints the digests; ``--write`` refreshes the pin.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

PIN = Path(__file__).with_name("no_ante_identity.json")
HELDBACK = Path(__file__).resolve().parents[2] / "strategy" / "rehearsal" / "heldback"

CASES: tuple[tuple[str, str | None, int], ...] = (
    ("practice_nlhe", None, 8),
    ("uncertainty_shortdeck", None, 6),
    ("uncertainty_plo4", None, 6),
    ("reveal_variant", "plo5-8max", 8),
    ("reveal_variant", "shortdeck-trips-6max-turbo", 6),
)
TABLES = 2
HAND_CAP = 60


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def family_firsts(registry: Any) -> list[Any]:
    """The first member of every population family (the behaviour fingerprint's choice)."""
    from strategy.population.params import PARTITIONS

    firsts: dict[str, Any] = {}
    for m in sorted(registry.members.values(), key=lambda m: (PARTITIONS.index(m.partition), m.index)):
        firsts.setdefault(m.family, m)
    return [firsts[f] for f in sorted(firsts)]


def _bot_factories(ruleset_id: str, practice: bool) -> list[Callable[[], Any]]:
    """Seat mix, keeping only bots that claim the ruleset. The tuned baseline needs the practice
    equity tables, so it sits only there."""
    from arena.agents.baselines import calling_station, legal_random, nlhe_heuristic
    from arena.agents.styles import aggressive, rock, station
    from strategy.policy.generic import GenericBot
    from strategy.population.registry import default_registry

    makers: list[Callable[[], Any]] = [GenericBot, *(m.make_bot for m in family_firsts(default_registry())),
                                       aggressive.make_bot,
                                       rock.make_bot, station.make_bot, nlhe_heuristic.make_bot,
                                       calling_station.make_bot, legal_random.make_bot]
    makers = [m for m in makers if ruleset_id in m().rulesets]
    if practice:
        from strategy.policy.package import load_params
        from strategy.policy.runtime import RuntimeBot

        vector = load_params(Path(__file__).resolve().parents[2] / "strategy" / "policy" / "vectors" / "nlhe-v2.json")
        makers += [RuntimeBot, lambda: RuntimeBot(vector)]
    return makers


def _play_case(ruleset_id: str, spec: str | None, players: int) -> dict[str, str]:
    from arena.agents.sdk import bot_info
    from arena.agents.worker import InProcessSeat, play_table
    from arena.benchmark.runner import Assignment, table_info
    from arena.domain.models import TournamentConfig
    from arena.history.canonical import export_history
    from arena.history.phh import export_history_phh
    from arena.table.service import seeded_table
    from strategy.rules.adapter import RulesAdapter
    from strategy.rules.conformance import run_conformance

    structure: dict[str, Any] = {}
    if spec is not None:
        structure = dict(json.loads((HELDBACK / f"{spec}.json").read_text())["structure"])
    structure["hand_cap"] = min(HAND_CAP, structure.get("hand_cap", HAND_CAP))
    structure.setdefault("initial_player_count", players)
    config = TournamentConfig.model_validate({"ruleset_id": ruleset_id, **structure})
    out = {"config_hash": config.config_hash(), "rules_hash": RulesAdapter.from_config(config).rules_hash}
    logs, observations, phh = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    makers = _bot_factories(ruleset_id, spec is None)
    n = config.initial_player_count
    for t in range(max(TABLES, -(-len(makers) // n))):
        seed = f"identity|{ruleset_id}|{spec}|{t}"
        a = Assignment(0, t, 0, seed, {s: k for k, s in enumerate(config.resolved_seats)})
        agents = {}
        for k, s in enumerate(a.seats):
            bot = makers[(k + n * t) % len(makers)]()
            agent = InProcessSeat(s, bot, bot_info(bot), table_info(config, a, s, config.bot_budget_ns), "identity")
            agent.start()
            decide = agent.decide

            def recorded(obs: Any, _decide: Any = decide) -> Any:
                observations.update(obs.model_dump_json().encode() + b"\n")
                return _decide(obs)

            agent.decide = recorded  # type: ignore[method-assign]
            agents[s] = agent
        run = play_table(seeded_table(config, seed, table_id=a.table_id), agents)
        for e in run.table.log:
            logs.update(json.dumps(e.to_json_obj(), sort_keys=True, separators=(",", ":")).encode() + b"\n")
        if ruleset_id != "reveal_variant":
            for doc in export_history_phh(export_history(run.table)):
                phh.update(doc.encode())
    out["event_logs"] = logs.hexdigest()
    out["observations"] = observations.hexdigest()
    out["phh"] = phh.hexdigest()
    rep = run_conformance(ruleset_id, hands=60)
    out["conformance"] = _sha(json.dumps(rep.as_dict(), sort_keys=True, default=str).encode())
    return out


def digests() -> dict[str, dict[str, str]]:
    from arena.domain.models import REVEAL_ENV

    result: dict[str, dict[str, str]] = {}
    saved = os.environ.get(REVEAL_ENV)
    try:
        for ruleset_id, spec, players in CASES:
            with tempfile.TemporaryDirectory() as d:
                if spec is not None:
                    variant = {"schema_version": 1, "ruleset_id": ruleset_id,
                               **json.loads((HELDBACK / f"{spec}.json").read_text())["variant"]}
                    (Path(d) / "variant.json").write_text(json.dumps(variant))
                    os.environ[REVEAL_ENV] = d
                result[f"{ruleset_id}:{spec or 'default'}"] = _play_case(ruleset_id, spec, players)
                os.environ.pop(REVEAL_ENV, None)
    finally:
        if saved is not None:
            os.environ[REVEAL_ENV] = saved
    return result


if __name__ == "__main__":
    got = digests()
    print(json.dumps(got, indent=1, sort_keys=True))
    if "--write" in sys.argv:
        PIN.write_text(json.dumps(got, indent=1, sort_keys=True) + "\n")
