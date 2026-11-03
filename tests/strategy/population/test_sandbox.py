"""Every family, and a checkpoint slot, plays every variant through the SDK's sandbox path.

The bots are told the 1 ms tournament budget, but the seats enforce ``CAUSE_BUDGET_NS``
so that a busy machine delaying a reply is not a fallback. Speed is checked on its
cause, the bot's own decide CPU time, rather than on the end-to-end timeout rate.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from arena.agents.baselines import baseline_path
from arena.agents.sdk import TableInfo
from arena.agents.worker import WorkerPool, WorkerSeat, play_table
from arena.benchmark.runner import policy_seed
from arena.domain.models import RULESET_IDS, TournamentConfig
from arena.table.service import seeded_table
from strategy.population.families import FAMILY_LIST
from strategy.population.registry import CheckpointLedger, Registry
from strategy.population.source import PopulationSource
from tests.agents.conftest import SANDBOX_SKIP

pytestmark = pytest.mark.skipif(SANDBOX_SKIP is not None, reason=SANDBOX_SKIP or "")

HAND_CAP = 30
CAUSE_BUDGET_NS = 50_000_000


def test_population_in_sandboxed_workers(tmp_path: Path):
    art = tmp_path / "learner.py"
    art.write_bytes(baseline_path("calling_station").read_bytes())
    ledger = CheckpointLedger(tmp_path / "checkpoints.json")
    slot = ledger.register(art, kind="learner_checkpoint", partition="holdout", rulesets=list(RULESET_IDS))
    ledger.promote(slot.slot_id, "test")
    source = PopulationSource(Registry(ledger=ledger), tmp_path / "bots")
    part = source.partition("holdout")
    firsts: dict[str, str] = {}
    for ref in part.members:
        firsts.setdefault(ref.family, ref.policy)
    paths = [Path(firsts[f.id]) for f in FAMILY_LIST] + [Path(firsts["frozen_checkpoint"])]
    paths += paths[: -len(paths) % 6]

    reasons: Counter = Counter()
    decisions = 0
    cpu: list[int] = []
    with WorkerPool(size=8) as pool:
        artifacts = {p: pool.stage(p) for p in set(paths)}
        for ruleset in RULESET_IDS:
            cfg = TournamentConfig(ruleset_id=ruleset, initial_player_count=6, hand_cap=HAND_CAP,
                                   hands_per_level=-(-HAND_CAP // 8))
            for t in range(len(paths) // 6):
                seed = f"pop-sandbox|{ruleset}|{t}"
                agents = {}
                for s, path in zip(cfg.resolved_seats, paths[6 * t: 6 * t + 6]):
                    info = TableInfo(table_id="t_pop", hero_seat=s, ruleset_id=ruleset, config=cfg.resolved(),
                                     config_hash=cfg.config_hash(), seats=cfg.resolved_seats,
                                     decision_budget_ns=cfg.bot_budget_ns, policy_seed=policy_seed(seed, s))
                    agents[s] = WorkerSeat(s, artifacts[path], info, pool.acquire, budget_ns=CAUSE_BUDGET_NS)
                    agents[s].start()
                table = seeded_table(cfg, seed, table_id="t_pop")

                def record(_seat, d, _table) -> None:
                    nonlocal decisions
                    decisions += 1
                    if d.fallback is not None:
                        reasons[d.fallback] += 1
                    if d.worker_cpu_ns is not None:
                        cpu.append(d.worker_cpu_ns)

                try:
                    play_table(table, agents, on_decision=record)
                finally:
                    for a in agents.values():
                        a.close()
                assert table.complete, (ruleset, t)
    assert decisions > 1000
    assert not reasons, dict(reasons)
    assert len(cpu) == decisions
    # Host stalls are charged to thread CPU time on a VM, so a few decides can read
    # long; the p99 is the bot's own speed.
    budget = TournamentConfig(ruleset_id=RULESET_IDS[0]).bot_budget_ns
    assert sorted(cpu)[int(0.99 * len(cpu))] < budget
