"""League (Stage 5): meta-game, override packs, gates and Holm confirmations (numpy only)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from strategy.blueprint import abstraction as A
from strategy.blueprint.policy import PolicyPack
from strategy.experiments.schema import ExperimentSpec
from strategy.league import gates as Gt
from strategy.league import meta
from strategy.league import tables as T
from strategy.league.clone import Counts, action_slot
from strategy.league.runner import REPO, BLUEPRINT_PACK, Config

ROW = A.NUM_ACTIONS

# ------------------------------------------------------------------ meta-game


def test_regret_matching_finds_the_uniform_equilibrium_of_a_cycle():
    rps = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]
    sigma = meta.solve(rps, iters=5000)
    assert np.allclose(sigma, 1 / 3, atol=0.02)
    assert meta.nash_conv(rps, sigma) < 0.02
    assert meta.nash_conv(rps, [1, 0, 0]) == pytest.approx(1.0)


def test_regret_matching_puts_the_mass_on_a_dominant_policy():
    G = [[0, 0.5, 0.4], [-0.5, 0, 0.1], [-0.4, -0.1, 0]]
    sigma = meta.solve(G, iters=5000)
    assert sigma[0] > 0.98
    floored = meta.solve(G, iters=5000, floor=0.3)
    assert floored.min() >= 0.1 - 1e-9 and floored.sum() == pytest.approx(1.0)


def test_cycles_need_every_edge_beyond_its_error():
    names = ["a", "b", "c"]
    G = [[0, 1, -1], [-1, 0, 1], [1, -1, 0]]
    assert meta.cycles(G, [[0.1] * 3] * 3, names) == [["a", "b", "c"]]
    assert meta.cycles(G, [[1.0] * 3] * 3, names) == []
    assert [n for n, _ in meta.ranking([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]], names)] == names


# ------------------------------------------------------------------ tables and override packs


def test_deviations_pass_only_keys_whose_regret_clears_the_gate():
    regret = np.zeros(4 * ROW, np.float32)
    sq = np.ones(4 * ROW, np.float32)
    regret[0 * ROW + 2] = 5.0
    regret[1 * ROW + 3] = 2.0
    regret[2 * ROW + 1] = -9.0
    base = np.zeros(4 * ROW, np.uint8)
    base[0:ROW] = [0, 255, 0, 0, 0, 0]
    dev = T.deviations(regret, sq, base, z=3.0, mix=0.5)
    assert dev.keys.tolist() == [0]
    assert dev.best.tolist() == [2]
    row = dev.rows[0].astype(int)
    assert row.sum() == 255 and abs(row[1] - row[2]) <= 1
    greedy = T.deviations(regret, sq, base, z=1.0, mix=1.0)
    assert greedy.keys.tolist() == [0, 1]
    assert greedy.rows[1].tolist() == [0, 0, 0, 255, 0, 0]
    out = T.apply_deviations(base, greedy).reshape(-1, ROW)
    assert out[1].tolist() == [0, 0, 0, 255, 0, 0] and out[3].sum() == 0
    with pytest.raises(ValueError):
        T.deviations(regret, sq, base, z=1.0, mix=0.0)


def test_an_override_pack_changes_only_the_overridden_keys(tmp_path):
    parent = REPO / BLUEPRINT_PACK
    pack = PolicyPack(parent)
    keys = [k for k in range(0, A.NUM_KEYS, 997) if pack.lookup(k) is not None][:5]
    assert len(keys) == 5
    rows = np.zeros((2, ROW), np.uint8)
    rows[0, A.CALL] = rows[1, A.ALL_IN] = 255
    missing = next(k for k in range(A.NUM_KEYS - 1, 0, -1) if pack.lookup(k) is None)
    dev = T.Deviations(np.array(sorted([keys[0], missing]), np.int64), rows, np.array([A.CALL, A.ALL_IN]))
    out = tmp_path / "child.npz"
    T.write_override_pack(parent, dev, out, {"ruleset_id": "practice_nlhe", "member": "test"})
    child = PolicyPack(out)
    for k, row in zip(dev.keys.tolist(), dev.rows):
        assert child.lookup(k)[0] == tuple(row.tolist())
    for k in keys[1:]:
        assert child.lookup(k) == pack.lookup(k)
    assert child.provenance["overrides"] == 2
    assert child.provenance["parent"]["data_sha256"] == pack.provenance["data_sha256"]


# ------------------------------------------------------------------ behaviour cloning


def test_real_actions_map_to_the_nearest_legal_abstract_slot():
    mask = sum(1 << a for a in (A.FOLD, A.CALL, A.MIN_RAISE, A.RAISE_A, A.ALL_IN))
    targets = [0] * ROW
    targets[A.MIN_RAISE], targets[A.RAISE_A], targets[A.RAISE_B], targets[A.ALL_IN] = 200, 450, 800, 2000
    assert action_slot({"type": "fold"}, mask, targets) == A.FOLD
    assert action_slot({"type": "fold"}, mask & ~(1 << A.FOLD), targets) is None
    assert action_slot({"type": "check"}, mask, targets) == A.CALL
    assert action_slot({"type": "raise_to", "amount": 230}, mask, targets) == A.MIN_RAISE
    assert action_slot({"type": "raise_to", "amount": 790}, mask, targets) == A.RAISE_A
    assert action_slot({"type": "raise_to", "amount": 2000}, mask, targets) == A.ALL_IN


def test_counts_merge_sums_by_key():
    a = Counts.from_dict({5: [1, 0, 0, 0, 0, 0], 2: [0, 1, 0, 0, 0, 0]})
    b = Counts.from_dict({5: [0, 0, 3, 0, 0, 0]})
    m = Counts.merge([a, b])
    assert m.keys.tolist() == [2, 5]
    assert m.counts.tolist() == [[0, 1, 0, 0, 0, 0], [1, 0, 3, 0, 0, 0]]
    assert m.decisions == 5


# ------------------------------------------------------------------ gates


def _result(diff, sd, blocks, *, lo=None, gates_ok=True, worst=0.0):
    se = sd / math.sqrt(blocks)
    return {"withheld": False, "diff": diff, "sd": sd, "blocks": blocks, "excluded": 0,
            "lo": diff - 2 * se if lo is None else lo, "hi": diff + 2 * se, "gates_ok": gates_ok,
            "worst_family": {"family": "f", "mean": worst}}


def test_promotion_needs_an_improvement_and_no_regression():
    rule = Gt.Rule()
    good = Gt.decide(rule, _result(2.0, 4.0, 192), {"blueprint": _result(0.5, 6.0, 64)})
    assert good["promote"], good
    assert not Gt.decide(rule, _result(0.1, 8.0, 192), {"blueprint": _result(0.5, 6.0, 64)})["promote"]
    cycle = Gt.decide(rule, _result(2.0, 4.0, 192), {"blueprint": _result(-4.0, 6.0, 64)})
    assert not cycle["promote"]
    assert [c["check"] for c in cycle["checks"] if not c["passed"]] == ["regress-blueprint: mean",
                                                                         "regress-blueprint: lower bound",
                                                                         "regress pooled: non-inferiority"]
    broken = Gt.decide(rule, _result(2.0, 4.0, 192, gates_ok=False), {})
    assert [c["check"] for c in broken["checks"] if not c["passed"]] == ["improve: execution"]
    family = Gt.decide(rule, _result(2.0, 4.0, 192, worst=-6.0), {})
    assert not family["promote"]
    withheld = Gt.decide(rule, {"withheld": True, "reason": "x"}, {})
    assert not withheld["promote"]


def test_pooled_regression_is_inverse_variance_weighted():
    p = Gt.pooled([_result(1.0, 2.0, 4), _result(-1.0, 4.0, 4)])
    assert p["mean"] == pytest.approx((1.0 / 1 - 1.0 / 4) / (1 + 1 / 4))
    assert Gt.pooled([{"withheld": True}]) is None


def test_holm_family_corrects_step_down():
    rows = {"strong": {"diff": 3.0, "sd": 4.0, "blocks": 100}, "weak": {"diff": 0.6, "sd": 4.0, "blocks": 100},
            "null": {"diff": -0.2, "sd": 4.0, "blocks": 100}}
    out = Gt.holm_family(rows)
    assert out["strong"]["confirmed"] and not out["null"]["confirmed"]
    assert out["weak"]["p_one_sided"] < 0.08
    assert out["weak"]["p_holm"] == pytest.approx(min(1.0, 2 * out["weak"]["p_one_sided"]))
    assert out["null"]["p_holm"] >= out["weak"]["p_holm"]
    margin = Gt.holm_family({"m": {"diff": 0.5, "sd": 4.0, "blocks": 100, "margin": -1.0}})
    assert margin["m"]["confirmed"]


def test_gate_specs_are_valid_and_use_only_validation():
    rule = Gt.Rule(improve_lineups=2, improve_blocks=1, regress_lineups=2, regress_blocks=1)
    a = ("blueprint", "artifacts/league/x/bundles/blueprint")
    b = ("C1", "artifacts/league/x/bundles/C1")
    for spec in (Gt.improve_spec("x", 1, a, b, rule), Gt.regress_spec("x", 1, a, b, rule)):
        parsed = ExperimentSpec.model_validate(spec)
        assert parsed.population.partition == "validation"
        assert "holdout" not in json.dumps(spec)
    reg = Gt.regress_spec("x", 1, a, b, rule)
    lineups = reg["population"]["explicit_lineups"]
    assert len(lineups) == 2 and all(lu[:Gt.J_SEATS] == [a[1]] * Gt.J_SEATS and len(lu) == 7 for lu in lineups)


def test_gate_specs_are_never_edited(tmp_path):
    spec = {"experiment_id": "s", "x": 1}
    path = Gt.write_spec(spec, tmp_path)
    assert Gt.write_spec(spec, tmp_path) == path
    with pytest.raises(FileExistsError):
        Gt.write_spec({"experiment_id": "s", "x": 2}, tmp_path)


LEAGUE_PACK = REPO / "strategy" / "tables" / "data" / "league-practice_nlhe.npz"


@pytest.mark.skipif(not LEAGUE_PACK.exists(), reason="no exported league pack")
def test_the_league_bot_plays_legal_tournaments_within_budget(monkeypatch):
    import time

    from arena.domain.models import Action, TournamentConfig
    from strategy.blueprint import runtime
    from strategy.value.rollouts import run_rollouts

    pack = PolicyPack(LEAGUE_PACK)
    assert pack.provenance["kind"] == "learner" and pack.provenance["overrides"] > 0
    assert pack.nbytes() < 100_000_000
    timings: list[int] = []
    orig = runtime.BlueprintBot.decide

    def timed(self, obs, state):
        assert self.name == "league"
        t = time.perf_counter_ns()
        out = orig(self, obs, state)
        timings.append(time.perf_counter_ns() - t)
        payload = {k: v for k, v in out[0].items() if k != "note"}
        assert obs.legal.allows(Action.model_validate(payload)), (payload, obs.legal)
        return out

    monkeypatch.setattr(runtime.BlueprintBot, "decide", timed)
    config = TournamentConfig(initial_player_count=4, hand_cap=120)
    rollouts = run_rollouts(config, ["strategy/league/bots/league.py", "style:aggressive", "baseline:calling_station",
                                     "style:rock"], blocks=1, base_seed="league-bot-test", jobs=1)
    assert all(sum(r.points.values()) == pytest.approx(100.0) for r in rollouts)
    assert len(timings) > 200
    assert float(np.quantile(timings, 0.99)) < TournamentConfig().bot_budget_ns


def test_the_league_config_round_trips(tmp_path):
    cfg = Config(run="t", learner_z=(3.0,), learner_mix=(1.0,))
    from dataclasses import asdict

    (tmp_path / "config.json").write_text(json.dumps(asdict(cfg)))
    assert Config.load(tmp_path / "config.json") == cfg
