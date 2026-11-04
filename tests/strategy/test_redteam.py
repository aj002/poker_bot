"""The redteam-v1 family (validation only), its diagnostics and the suite plumbing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arena.agents.sdk import load_bot
from strategy.eval.population import load_partition_source
from strategy.redteam import registry, suite
from strategy.redteam.diagnostics import table_diagnostics

REPO = Path(__file__).resolve().parents[2]


def test_lock_matches_the_bot_files():
    registry.verify_lock()


def test_validation_only_and_nothing_else():
    src = registry.RedteamPopulationSource()
    part = src.partition("validation")
    assert {m.member_id for m in part.members} == set(registry.MEMBERS)
    assert {m.family for m in part.members} == {registry.FAMILY}
    assert all(m.rulesets == ("practice_nlhe",) for m in part.members)
    for name in ("train", "holdout"):
        with pytest.raises(PermissionError):
            src.partition(name)
    lock = json.loads(registry.LOCK_PATH.read_text())
    assert lock["partitions"] == ["validation"] and lock["holdout_members"] == [] and lock["train_members"] == []


def test_source_loads_through_the_harness_protocol_and_samples_deterministically():
    src = load_partition_source("strategy.redteam.registry:RedteamPopulationSource")
    part = src.partition("validation")
    a = src.sample_lineups(part, ruleset_id="practice_nlhe", opponents=7, count=6, seed="s")
    b = src.sample_lineups(part, ruleset_id="practice_nlhe", opponents=7, count=6, seed="s")
    assert [x.lineup_id for x in a] == [x.lineup_id for x in b]
    assert all(len(x.opponents) == 7 for x in a)
    assert len({o.member_id for x in a for o in x.opponents}) > 1


def test_other_registries_do_not_contain_the_red_team():
    from strategy.population import HandhqPopulationSource, PopulationSource
    from strategy.population.registry import default_registry, verify_lock

    verify_lock()
    for source in (PopulationSource(), HandhqPopulationSource()):
        for partition in ("train", "validation", "holdout"):
            ids = {m.member_id for m in source.partition(partition).members}
            assert not ids & set(registry.MEMBERS)
            assert all(m.family != registry.FAMILY for m in source.partition(partition).members)
    assert not [m for m in default_registry().members if m in registry.MEMBERS]


def test_preregistered_cells():
    pr = suite.prereg()
    cells = suite.build_cells(pr)
    assert len(cells) == 21 and len({c.cell_id for c in cells}) == 21
    assert all(len(c.lineup.opponents) == 7 for c in cells)
    singles = [c for c in cells if c.set == "single"]
    assert all(len({o.member_id for o in c.lineup.opponents}) == 1 for c in singles)
    pop = [c for c in cells if c.set == "population"]
    assert len(pop) == 8 and all({o.member_id for o in c.lineup.opponents[:3]} == set(registry.MEMBERS) for c in pop)
    assert len({c.seed for c in cells}) == 21
    assert [c.seed for c in suite.build_cells(pr)] == [c.seed for c in cells]
    assert [c.seed for c in suite.build_cells(pr, seed="other")] != [c.seed for c in cells]
    holdout_ids = {m.member_id for m in load_partition_source(pr["population"]["source"]).partition("holdout").members}
    assert not {o.member_id for c in cells for o in c.lineup.opponents} & holdout_ids


def test_suite_refuses_the_holdout_for_population_opponents():
    pr = suite.prereg()
    bad = dict(pr["population"], partition="holdout")
    with pytest.raises(PermissionError):
        suite.population_lineups(bad)


@pytest.mark.parametrize("member", sorted(registry.MEMBERS))
def test_bots_load_and_play_a_legal_table(member: str):
    from strategy.population.measure import play

    path = REPO / registry.MEMBERS[member]
    path = path / "bot.py" if path.is_dir() else path
    bots = [(f"s{i}", load_bot(path, module_name=f"rt_test_{member}_{i}")[0]) for i in range(8)]
    table, seated = play(bots, f"redteam-test|{member}", config={"hand_cap": 40, "hands_per_level": 5})
    assert table.complete
    for _, (_, seat) in seated.items():
        assert not seat.stats.fallbacks, dict(seat.stats.fallbacks)


def test_diagnostics_add_up():
    from strategy.population.measure import play
    from strategy.redteam.diagnostics import ALIVE_BANDS

    bots = [("hero", load_bot(REPO / "redteam/adaptive/bot.py", module_name="rt_diag_h")[0])]
    bots += [("rt-pressure", load_bot(REPO / "redteam/pressure/bot.py", module_name=f"rt_diag_p{i}")[0])
             for i in range(7)]
    table, seated = play(bots, "redteam-diag", config={"hand_cap": 60, "hands_per_level": 8})
    events = [{"type": e.type, "data": e.data} for e in table.log]
    seat = next(s for s, (k, _) in seated.items() if k == "hero")
    tags = {s: ("hero" if k == "hero" else k) for s, (k, _) in seated.items()}
    d = table_diagnostics(events, seat, tags)
    from strategy.objective.points import outcome_from_events

    final = outcome_from_events(events).points(seat)
    start = 100 / 8
    assert abs(sum(d["phase_eq"]) - (final - start)) < 1e-6
    assert abs(sum(d["alive_eq"].values()) - sum(d["phase_eq"])) < 1e-6
    assert set(d["alive_eq"]) <= set(ALIVE_BANDS)
    assert d["hands"] == sum(d["alive_hands"].values())
    assert d["agg"]["rt-pressure"]["hands"] >= d["hands"]
    assert sum(sum(v) for v in d["resp"].values()) >= 0


def test_suite_smoke_runs_and_resumes(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(suite, "check_previous", lambda pr: [])
    out = tmp_path / "run"
    refs = {"blueprint-only": "baseline:calling_station", "nlhe-v2": "baseline:calling_station",
            "previous-final": "baseline:calling_station"}
    pr = suite.prereg()
    monkeypatch.setattr(suite, "prereg", lambda: json.loads(json.dumps(pr)))
    meta = suite.run_suite(out, "baseline:nlhe_heuristic", refs=refs, jobs=1, blocks=2, only=["single-adaptive"],
                           arms=["ours", "nlhe-v2"], log=open(tmp_path / "log.txt", "w"))
    assert meta["arms"]["ours"]["bundle_sha256"]
    first = (out / "cells" / "single-adaptive" / "ours.json").read_bytes()
    data = json.loads(first)
    assert len([t for t in data["tables"] if t["valid"]]) == 16
    assert all(t["diag"] for t in data["tables"] if t["valid"])
    assert data["execution"]["illegal_actions"] == 0
    suite.run_suite(out, "baseline:nlhe_heuristic", refs=refs, jobs=1, blocks=2, only=["single-adaptive"],
                    arms=["ours", "nlhe-v2"], log=open(tmp_path / "log2.txt", "w"))
    assert (out / "cells" / "single-adaptive" / "ours.json").read_bytes() == first
    assert "cached" in (tmp_path / "log2.txt").read_text()
    res = suite.analyze(out)
    assert res["cells"]["single-adaptive"]["arms"]["ours"]["n"] == 2
    assert "nlhe-v2" in res["cells"]["single-adaptive"]["diff"]
    from strategy.redteam.report import render

    assert "single-adaptive" in render(res)
