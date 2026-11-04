"""Stage 8: the final bot's routing, its bundle, the distillation student and the holdout runner (no numba)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from strategy.blueprint import abstraction as A
from tests.strategy.test_blueprint import uniform_pack
from tests.strategy.test_exploit import exploit_pack

ROOT = Path(__file__).resolve().parents[2]


def _stations(n: int = 3):
    from arena.agents.baselines import BASELINES
    from arena.agents.sdk import load_bot

    return [(f"cs{i}", load_bot(BASELINES["calling_station"])[0]) for i in range(n)]


def _strip(action) -> dict:
    from arena.domain.models import Action

    if isinstance(action, Action):
        return action.model_dump(exclude_none=True)
    return {k: v for k, v in dict(action).items() if k != "note"}


def _hero_records(bots, seed: str, ruleset: str = "practice_nlhe", hand_cap: int = 120):
    from arena.benchmark.runner import policy_seed
    from strategy.population.measure import play

    _, seated = play(bots, seed, ruleset, config={"initial_player_count": len(bots), "hand_cap": hand_cap}, keep=True)
    seat, (_, agent) = next((s, v) for s, v in seated.items() if v[0] == "hero")
    return policy_seed(seed, seat), [(o, a) for o, a in agent.records if a is not None]


def test_nlhe_is_the_teacher_decision_for_decision(tmp_path, monkeypatch):
    from strategy.exploit.pack import ExploitPack
    from strategy.exploit.runtime import ExploitBot
    from strategy.final.runtime import FinalBot

    pack, xpack = uniform_pack(tmp_path / "u.npz"), exploit_pack(tmp_path / "x.npz")
    monkeypatch.setattr(ExploitPack, "deviation", lambda self, e, key: A.ALL_IN)
    final = FinalBot(pack=pack, exploit_pack=xpack)
    teacher = ExploitBot(pack=pack, exploit_pack=xpack)
    checked = blended = 0
    for t in range(8):
        seed, recs = _hero_records([("hero", final), *_stations()], f"final-identity-{t}", hand_cap=200)
        state = teacher.initial_state(type("T", (), {"ruleset_id": "practice_nlhe", "policy_seed": seed})())
        for obs, action in recs:
            mine, state = teacher.decide(obs, state)
            assert _strip(mine) == action
            checked += 1
    blended = final.teacher.stats["exploit"]
    assert checked > 30 and blended > 0


@pytest.mark.parametrize("ruleset", ["uncertainty_shortdeck", "uncertainty_plo4"])
def test_other_rulesets_play_the_generic_fallback(tmp_path, ruleset):
    from arena.domain.models import Action
    from strategy.final.runtime import FinalBot
    from strategy.policy.generic import GenericBot

    final = FinalBot(pack=uniform_pack(tmp_path / "u.npz"), exploit_pack=exploit_pack(tmp_path / "x.npz"))
    seed, recs = _hero_records([("hero", final), *_stations()], f"final-generic-{ruleset}", ruleset, hand_cap=60)
    generic = GenericBot()
    state = generic.initial_state(type("T", (), {"ruleset_id": ruleset, "policy_seed": seed})())
    for obs, action in recs:
        assert obs.legal.allows(Action.model_validate(action))
        mine, state = generic.decide(obs, state)
        assert _strip(mine) == action
    assert len(recs) > 20 and state["fb"] == 0
    assert final.teacher.stats["blueprint"] == 0 and final.teacher.stats["leaf"] == 0


def test_a_teacher_error_plays_the_legal_fallback_and_is_counted(tmp_path, monkeypatch):
    from arena.domain.models import Action
    from strategy.final.runtime import FinalBot

    final = FinalBot(pack=uniform_pack(tmp_path / "u.npz"), exploit_pack=exploit_pack(tmp_path / "x.npz"))

    def boom(obs, state):
        raise RuntimeError("teacher bug")

    monkeypatch.setattr(final.teacher, "decide", boom)
    seen = []
    orig = final.decide

    def spy(obs, state):
        action, out = orig(obs, state)
        assert obs.legal.allows(Action.model_validate(_strip(action)))
        seen.append(out["fb"])
        return action, out

    monkeypatch.setattr(final, "decide", spy)
    _hero_records([("hero", final), *_stations()], "final-error", hand_cap=10)
    assert seen and seen == list(range(1, len(seen) + 1))


def test_rulesets_and_no_numba_on_the_deployed_path():
    from strategy.final.runtime import RULESETS

    assert RULESETS[0] == "practice_nlhe"
    assert {"uncertainty_shortdeck", "uncertainty_plo4", "reveal_variant"} <= set(RULESETS)
    code = ("import sys\n"
            "import strategy.final.runtime, strategy.final.bots.final\n"
            "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('numba', 'llvmlite')"
            " or m.startswith(('strategy.solve', 'strategy.exploit.kernels', 'strategy.blueprint.kernels',"
            " 'strategy.final.distill', 'strategy.final.holdout')))\n"
            "assert not bad, bad\n")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


# ------------------------------------------------------------------ package


def test_bundle_is_deterministic_self_contained_and_verified(tmp_path):
    from strategy.final import package

    pack, xpack = uniform_pack(tmp_path / "u.npz"), exploit_pack(tmp_path / "x.npz")
    m1 = package.build(tmp_path / "one", pack=pack, xpack=xpack)
    m2 = package.build(tmp_path / "elsewhere" / "two", pack=pack, xpack=xpack)
    assert m1["bundle_sha256"] == m2["bundle_sha256"] and m1["files"] == m2["files"]
    package.smoke(tmp_path / "one")
    assert package.verify(tmp_path / "one")["bundle_sha256"] == m1["bundle_sha256"]
    shipped = set(m1["files"])
    assert {"bot.py", "strategy/final/runtime.py", "strategy/exploit/runtime.py", "strategy/policy/generic.py",
            "strategy/tables/data/blueprint-practice_nlhe.npz", "strategy/tables/data/exploit-practice_nlhe.npz"} <= shipped
    assert not any(f.endswith(("kernels.py", "experts.py", "train.py", "distill.py", "holdout.py")) for f in shipped)
    assert m1["bytes"] < package.SIZE_LIMIT
    t1, t2 = package.submit(tmp_path / "one", tmp_path / "s1"), package.submit(tmp_path / "elsewhere" / "two",
                                                                               tmp_path / "s2")
    assert t1["sha256"] == t2["sha256"] and t1["bundle_sha256"] == m1["bundle_sha256"]
    assert (tmp_path / "s1" / "SHA256SUMS").read_text() == f"{t1['sha256']}  final-bot-{m1['bundle_sha256'][:12]}.tar.gz\n"
    (tmp_path / "one" / "strategy" / "final" / "runtime.py").write_text("# tampered\n")
    with pytest.raises(ValueError, match="differ"):
        package.verify(tmp_path / "one")


# ------------------------------------------------------------------ distillation


def test_quantized_rows_sum_to_255_and_keep_every_tail_action():
    from strategy.final.distill import quantize_keep_tail

    p = np.array([[0.999, 0.0005, 0.0005, 0, 0, 0], [0.2, 0.2, 0.2, 0.2, 0.2, 0.0], [1, 0, 0, 0, 0, 0]])
    q = quantize_keep_tail(p)
    assert (q.sum(axis=1) == 255).all()
    assert ((q > 0) == (p > 0)).all()


def test_agreement_metrics_on_identical_and_disjoint_rows():
    from strategy.final.distill import agreement

    t = np.array([[0.5, 0.5, 0, 0, 0, 0], [0.9, 0.1, 0, 0, 0, 0]])
    same = agreement(t, t.copy())
    assert same["tv_mean"] == 0 and same["same_action_crn"] == pytest.approx(1.0) and same["kl_mean"] == 0
    assert same["tail_kept"] == 1.0
    other = agreement(t, np.array([[0, 0, 1.0, 0, 0, 0], [0, 0, 1.0, 0, 0, 0]]))
    assert other["tv_mean"] == pytest.approx(1.0) and other["same_action_crn"] == 0 and other["argmax_agreement"] == 0


def test_student_rows_average_the_teacher_over_every_visit(tmp_path):
    from strategy.blueprint.policy import PolicyPack
    from strategy.final.distill import build_student, student_rows

    pack = PolicyPack(uniform_pack(tmp_path / "u.npz"))
    k_blend, k_plain = 12345, 777
    bp = np.array(pack.lookup(k_blend)[0], np.float64) / 255
    allin = np.eye(A.NUM_ACTIONS)[A.ALL_IN]
    data = {"key": np.array([k_blend, k_blend, k_blend, k_plain]), "mask": np.full(4, 63, np.uint8),
            "probs": np.array([bp, 0.5 * bp + 0.5 * allin, bp, bp], np.float32),
            "blended": np.array([0, 1, 0, 0], np.int8)}
    keys, rows, stats = student_rows(data, pack, prior=1.0)
    assert keys.tolist() == [k_blend] and stats["keys_blended"] == 1
    expect = (3 * bp + 0.5 * (allin - bp) + bp) / 4
    assert np.abs(rows[0] / 255 - expect).max() < 2 / 255
    np.savez(tmp_path / "d.npz", partition=np.array("train"), seed=np.array("t"), **data)
    build_student(tmp_path / "d.npz", tmp_path / "s.npz", pack=tmp_path / "u.npz")
    student = PolicyPack(tmp_path / "s.npz")
    assert student.lookup(k_blend) == (tuple(rows[0].tolist()), 0)
    assert student.lookup(k_plain) == pack.lookup(k_plain)
    np.savez(tmp_path / "v.npz", partition=np.array("validation"), **data)
    with pytest.raises(ValueError, match="train"):
        build_student(tmp_path / "v.npz", tmp_path / "s2.npz", pack=tmp_path / "u.npz")


def test_recording_teacher_plays_like_the_teacher_and_records_its_blend(tmp_path, monkeypatch):
    from strategy.blueprint.policy import PolicyPack
    from strategy.exploit.pack import ExploitPack
    from strategy.exploit.runtime import ExploitBot
    from strategy.final.distill import recording_teacher

    pack, xpack = uniform_pack(tmp_path / "u.npz"), exploit_pack(tmp_path / "x.npz")
    monkeypatch.setattr(ExploitPack, "deviation", lambda self, e, key: A.ALL_IN)
    rec = recording_teacher(pack=pack, exploit_pack=xpack)
    teacher = ExploitBot(pack=pack, exploit_pack=xpack)
    for t in range(3):
        seed, recs = _hero_records([("hero", rec), *_stations()], f"final-recording-{t}", hand_cap=200)
        state = teacher.initial_state(type("T", (), {"ruleset_id": "practice_nlhe", "policy_seed": seed})())
        for obs, action in recs:
            mine, state = teacher.decide(obs, state)
            assert _strip(mine) == action
    probs = np.array([r[2] for r in rec.rec])
    assert len(rec.rec) > 0 and np.allclose(probs.sum(axis=1), 1.0)
    blended = [r for r in rec.rec if r[3]]
    bp = PolicyPack(pack)

    def plain(key, mask):
        w = [bp.lookup(key)[0][a] if (mask >> a) & 1 else 0 for a in range(A.NUM_ACTIONS)]
        return [x / sum(w) for x in w]

    assert blended and all(r[2][A.ALL_IN] > plain(r[0], r[1])[A.ALL_IN] for r in blended)
    assert all(np.allclose(r[2], plain(r[0], r[1])) for r in rec.rec if not r[3])
    assert all(((r[1] >> a) & 1) or r[2][a] == 0 for r in rec.rec for a in range(A.NUM_ACTIONS))


def test_collect_refuses_the_holdout_partition():
    from strategy.final.distill import collect

    with pytest.raises(ValueError, match="holdout"):
        collect("holdout", 1)


def test_hybrid_counterfactual_counts_only_timeouts_the_saving_covers(tmp_path):
    from strategy.final.distill import hybrid

    b = {"budget_ns": 1_000_000}
    recs = [{"class": "bot_caused", "late_by_ns": 50_000, "worker_wall_ns": 400_000, "worker_parse_ns": 700_000, **b},
            {"class": "bot_caused", "late_by_ns": 300_000, "worker_wall_ns": 400_000, "worker_parse_ns": 800_000, **b},
            {"class": "stall_like", "drained_after_ns": 9_000, "worker_wall_ns": 100_000, "worker_parse_ns": 0, **b},
            {"class": "stall_like", "drained_after_ns": 500_000, "worker_wall_ns": 400_000, "worker_parse_ns": 100_000,
             "e2e_ns": 1_008_000, **b}]
    rep = {"decisions": 1000, "worker_wall": {"p50_us": 200.0}, "selection": {"used": 0},
           "attempts": [{"candidate": {"timeouts": {"timeouts": 4, "records": recs}}}]}
    (tmp_path / "g.json").write_text(json.dumps(rep))
    out = hybrid(tmp_path / "g.json", cost_ratio=0.7)
    assert [r["verdict"] for r in out["rows"]] == ["rescued", "not_rescued", "rescued", "undetermined"]
    assert out["by_class"]["bot_caused"] == {"timeouts": 2, "rescued": 1, "not_rescued": 1, "undetermined": 0,
                                             "unknown": 0}


# ------------------------------------------------------------------ holdout


def test_preregistration_matches_the_holdout_specs():
    from strategy.experiments.schema import load_spec
    from strategy.final.holdout import preregistration

    pre = preregistration()
    assert [h["id"] for h in pre["hypotheses"]] == ["H1", "H2", "H3", "H4"]
    assert pre["alpha"] == 0.05 and pre["margin"] == 0.5
    for h in pre["hypotheses"]:
        spec = load_spec(h["spec"])
        assert spec.stage == "final_holdout" and spec.population.partition == "holdout"
        assert spec.arms.B.policy == pre["bundle"]
        assert h["kind"] in ("superiority", "non_inferiority")
    assert [h["kind"] for h in pre["hypotheses"]].count("non_inferiority") == 1


def test_submission_record_ties_the_shipped_bundle_to_the_evaluated_one():
    from strategy.final.holdout import preregistration

    sub = json.loads((Path(__file__).parents[2] / "strategy/final/submission.json").read_text())
    assert sub["evaluated_bundle_sha256"] == preregistration()["bundle_sha256"]
    assert sub["evaluated_at"]["tag"] == preregistration()["tag"]
    if "adopted_by" in sub:
        assert sub["adopted_by"]["against_bundle_sha256"] == sub["previous_submission"]["bundle_sha256"]
        assert all(sub["adopted_by"][h]["rejected"] for h in sub["adopted_by"] if h.startswith("H"))
    elif sub["submission_bundle_sha256"] != sub["evaluated_bundle_sha256"]:
        for ruleset in ("practice_nlhe", "uncertainty_plo4"):
            eq = sub["equivalence"][ruleset]
            assert eq["identical"] and eq["identical_tables"] == eq["tables"] and eq["fallbacks"] == 0
    shipped = [g for g in sub["gate_runs"] if g["bundle_sha256"] == sub["submission_bundle_sha256"][:8]]
    assert {g["ruleset"] for g in shipped if g["passed"]} == {"practice_nlhe", "uncertainty_plo4"}


def test_holdout_runner_never_runs_a_finished_comparison_twice(tmp_path, monkeypatch):
    from strategy import population
    from strategy.exploit import evaluate
    from strategy.final import holdout

    monkeypatch.setattr(holdout, "OUT", tmp_path)
    monkeypatch.setattr(holdout, "LEDGER", tmp_path / "ledger.json")
    monkeypatch.setattr(holdout, "REPO", tmp_path.resolve())
    monkeypatch.setattr(holdout, "check_ready", lambda prereg: None)
    monkeypatch.setattr(holdout, "_git", lambda *args: "0" * 40)
    monkeypatch.setattr(population, "PopulationSource", lambda: SimpleNamespace(partition=lambda name: None))
    monkeypatch.chdir(tmp_path)
    calls = []

    def fake_run(spec, *, jobs):  # the harness returns a path relative to the working directory
        calls.append(spec)
        d = Path("lab") / spec
        d.mkdir(parents=True)
        return d, {}

    monkeypatch.setattr(evaluate, "run", fake_run)
    holdout.run(1)
    holdout.run(1)
    assert calls == [h["spec"] for h in holdout.preregistration()["hypotheses"]]
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert all(v["finished"] and not v["resumed"] for v in ledger.values())
