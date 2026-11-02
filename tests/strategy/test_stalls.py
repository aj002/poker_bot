import json
import math
import statistics

import pytest

from strategy.eval import burn, stalls
from strategy.eval.stalls import (
    assess,
    binomial_upper_tail,
    classify,
    excess_p_value,
    poisson_upper_tail,
    select_attempt,
)
from tests.agents.conftest import needs_sandbox

BUDGET = 1_000_000


def rec(**kw):
    base = {"outcome": "late_reply", "worker_parse_ns": 100_000, "worker_wall_ns": 50_000}
    return {**base, **kw}


def replay(*walls, fallback=None):
    return {"replay": {"budget_ns": 100_000_000, "runs": [
        {"fallback": fallback, "worker_parse_ns": 100_000, "worker_wall_ns": w} for w in walls]}}


def test_classify_transport_replay_and_unclassified():
    assert classify(rec(), BUDGET, 220_000) == ("stall_like", "transport")
    slow_reply = rec(worker_wall_ns=900_000)
    assert classify(slow_reply, BUDGET, 220_000) == ("unclassified", "no_replay")
    assert classify({**slow_reply, **replay(20_000, 30_000, 2_000_000)}, BUDGET, 220_000) == ("stall_like", "replay_fast")
    assert classify({**slow_reply, **replay(450_000, 700_000, 800_000)}, BUDGET, 220_000) == ("bot_caused", "replay_slow")
    assert classify({**slow_reply, **replay(10, fallback="timeout")}, BUDGET, 220_000) == ("bot_caused",
                                                                                             "replay_timeout")
    assert classify({"outcome": "unresponsive", "replay": {"error": "x"}}, BUDGET, 220_000) == ("unclassified",
                                                                                                 "replay_error")
    assert classify({"outcome": "unresponsive", **replay(20_000)}, BUDGET, 220_000) == ("stall_like", "replay_fast")
    assert classify(rec(), BUDGET, None) == ("unclassified", "no_replay")


def test_classify_uses_decide_cpu_against_wall():
    # Allowance 780 us. A CPU-bound decide past it is the bot's, even if its replay is fast.
    computing = rec(worker_wall_ns=900_000, worker_cpu_ns=880_000)
    assert classify(computing, BUDGET, 220_000) == ("bot_caused", "cpu_bound")
    assert classify({**computing, **replay(20_000)}, BUDGET, 220_000) == ("bot_caused", "cpu_bound")
    # Off-CPU for most of the decide: descheduled; a fast replay does not change that.
    off_cpu = rec(worker_wall_ns=900_000, worker_cpu_ns=300_000)
    assert classify(off_cpu, BUDGET, 220_000) == ("stall_like", "descheduled")
    assert classify({**off_cpu, **replay(20_000)}, BUDGET, 220_000) == ("stall_like", "descheduled")
    # ...but a slow replay is still bot-caused.
    assert classify({**off_cpu, **replay(600_000, 650_000, 700_000)}, BUDGET, 220_000) == ("bot_caused",
                                                                                           "replay_slow")
    # The decide fitted; the harness's parse did not.
    slow_parse = rec(worker_parse_ns=850_000, worker_wall_ns=150_000, worker_cpu_ns=149_000)
    assert classify(slow_parse, BUDGET, 220_000) == ("stall_like", "parse")
    # A parse that was off-CPU is descheduled, not a slow parse.
    assert classify({**slow_parse, "worker_parse_cpu_ns": 200_000}, BUDGET, 220_000) == ("stall_like", "descheduled")
    assert classify({**slow_parse, "worker_parse_cpu_ns": 840_000}, BUDGET, 220_000) == ("stall_like", "parse")
    # CPU ~ wall inside the allowance is transport.
    assert classify(rec(worker_cpu_ns=49_000), BUDGET, 220_000) == ("stall_like", "transport")
    assert stalls.cpu_bound(rec()) is None and stalls.cpu_bound(computing) is True


def test_tails_match_closed_forms():
    assert binomial_upper_tail(2, 2, 0.5) == pytest.approx(0.25)
    assert binomial_upper_tail(0, 5, 0.3) == 1.0 and binomial_upper_tail(6, 5, 0.3) == 0.0
    assert poisson_upper_tail(1, 2.0) == pytest.approx(1 - math.exp(-2.0))
    assert poisson_upper_tail(0, 0.0) == 1.0 and poisson_upper_tail(1, 0.0) == 0.0
    # Equal exposure, equal counts: no evidence of excess.
    assert excess_p_value(3, 1.0, 3, 1.0) == pytest.approx(binomial_upper_tail(3, 6, 0.5))
    assert excess_p_value(1, 0.0, 0, 1.0) is None


def tally(decisions=50_000, timeouts=0, bot=0, stall=0, unc=0, exposure=10.0):
    return {"decisions": decisions, "timeouts": timeouts, "timeout_rate": timeouts / decisions if decisions else None,
            "exposure_s": exposure, "overhead_ns": 200_000,
            "classes": {"bot_caused": bot, "stall_like": stall, "unclassified": unc},
            "rules": {}, "unrecorded": 0, "records": []}


def test_assess_orders_bot_then_host_then_stall():
    control = tally(12_000, 4, stall=4, exposure=2.4)
    assert assess(tally(timeouts=1, bot=1), control, strict=True)["verdict"] == "fail"
    assert assess(tally(timeouts=1, bot=1), control, strict=False)["verdict"] == "pass"
    assert assess(tally(timeouts=10, stall=10), control, strict=True)["verdict"] == "pass"
    excess = assess(tally(timeouts=60, stall=60), control, strict=True)
    assert excess["verdict"] == "stall_excess" and excess["excess_p_value"] < 0.01
    noisy = tally(12_000, 20, stall=20, exposure=2.4)
    host = assess(tally(timeouts=1, stall=1), noisy, strict=True)
    assert host["verdict"] == "host_invalid" and host["ceiling_p_value"] < stalls.CEILING_ALPHA
    # Noise around the ceiling alone is not a host failure.
    assert assess(tally(), tally(12_000, 8, stall=8, exposure=2.4), strict=True)["verdict"] == "pass"
    assert assess(tally(), None, strict=True)["verdict"] == "pass"


def test_assess_compares_stalls_with_the_controls_stalls_and_reports_its_bot_floor():
    # Control timeouts the rules call bot-caused are host events: they count for the
    # ceiling, not for the stall C-test, and are reported as the attribution floor.
    control = tally(50_000, 10, bot=8, stall=2, exposure=15.0)
    a = assess(tally(timeouts=14, stall=14, exposure=15.0), control, strict=True)
    assert a["verdict"] == "stall_excess" and a["control_stall_like_or_unclassified"] == 2
    b = assess(tally(timeouts=8, bot=8, exposure=15.0), control, strict=True)
    assert b["verdict"] == "fail" and b["control_bot_caused"] == 8
    assert b["bot_vs_control_p_value"] == pytest.approx(binomial_upper_tail(8, 16, 0.5))
    assert "workload-matched control had 8 on 50000" in b["reason"]
    # The host and stall tests are still evaluated under a bot-caused fail.
    assert b["stall_verdict"] == "pass" and b["ceiling_p_value"] is not None
    assert assess(tally(timeouts=15, bot=1, stall=14, exposure=15.0), control,
                  strict=True)["stall_verdict"] == "stall_excess"
    assert assess(tally(timeouts=2, stall=2, exposure=15.0), control, strict=True)["verdict"] == "pass"


def test_rerun_policy_uses_the_first_pass_and_reports_every_attempt():
    ok, bad, fail = ({"verdict": v, "reason": v} for v in ("pass", "stall_excess", "fail"))
    assert select_attempt([ok])["action"] == "use"
    assert select_attempt([bad])["action"] == "rerun"
    assert select_attempt([bad, ok]) == {"action": "use", "used": 1, "reason": "pass"}
    assert select_attempt([bad, bad, bad], max_reruns=2)["action"] == "invalid"
    assert select_attempt([bad, fail])["action"] == "fail"
    assert select_attempt([bad], max_reruns=0)["action"] == "invalid"


SCRIPTS = {
    "bot": {"A": [(tally(timeouts=1, stall=1), tally(12_000, 3, stall=3))],
            "B": [(tally(timeouts=2, bot=2), tally(12_000, 3, stall=3))]},
    "rerun": {"A": [(tally(timeouts=80, stall=80), tally(12_000, 2, stall=2)), (tally(), tally(12_000, 2))],
              "B": [(tally(), tally(12_000, 2))]},
    "host": {"A": [(tally(), tally(12_000, 2))],
             "B": [(tally(), tally(12_000, 40, stall=40))]},
}


def _scripted(monkeypatch, script):
    from strategy.eval import paired

    original = paired.run_arm
    calls = {"A": 0, "B": 0}
    dirs = []

    def run_arm(cell, arm, policy, **kw):
        dirs.append(kw["out_dir"])
        run = original(cell, arm, policy, **kw)
        steps = script[arm.label]
        cand, ctrl = steps[min(calls[arm.label], len(steps) - 1)]
        calls[arm.label] += 1
        run.timeout_classes = {"candidate": cand, "opponents": {}, "control": ctrl, "control_every": 4}
        return run

    monkeypatch.setattr(paired, "run_arm", run_arm)
    return calls, dirs


def _run(tmp_path, capsys, **kw):
    from strategy.cli import main
    from tests.strategy.test_experiments import tiny

    spec = tiny(experiment_id="pytest-stalls", **kw)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec.model_dump(mode="json")))
    out = tmp_path / "run"
    code = main(["experiment", str(path), "--out", str(out), "--jobs", "1", "--quiet"])
    return code, json.loads((out / "report.json").read_text()), (out / "report.md").read_text(), capsys.readouterr()


def test_bot_caused_timeouts_in_arm_b_fail_with_points_reported(monkeypatch, tmp_path, capsys):
    calls, _ = _scripted(monkeypatch, SCRIPTS["bot"])
    code, report, md, _ = _run(tmp_path, capsys)
    a = report["analysis"]
    assert code == 0 and not a["withheld"] and a["decision"]["verdict"] == "fail"
    assert "mean_difference" in a["primary"]
    gate = next(g for g in a["execution"]["gates"] if g["gate"] == "bot_caused_timeouts")
    assert gate == {"gate": "bot_caused_timeouts", "limit": 0, "observed": 2, "passed": False}
    assert calls == {"A": 1, "B": 1}
    assert "bot_caused_timeouts FAILED" in md and "Classified timeouts" in md


def test_stall_excess_is_rerun_and_every_attempt_reported(monkeypatch, tmp_path, capsys):
    calls, dirs = _scripted(monkeypatch, SCRIPTS["rerun"])
    code, report, md, _ = _run(tmp_path, capsys)
    a = report["analysis"]
    assert code == 0 and not a["withheld"] and calls == {"A": 2, "B": 1}
    cls = a["execution"]["timeout_classes"]["A"]
    [cell] = cls["cells"].values()
    assert cell["selection"]["action"] == "use" and cell["selection"]["used"] == 1
    assert [r["verdict"] for r in cell["selection"]["attempts"]] == ["stall_excess", "pass"]
    assert cls["reruns"] == 1 and dirs[1].name == "rerun-1" and dirs[1].parent == dirs[0]
    assert "a1 stall_excess; " in md and "a2 pass" in md


def test_host_invalid_after_reruns_withholds_points(monkeypatch, tmp_path, capsys):
    from strategy.cli import WITHHELD_EXIT

    calls, _ = _scripted(monkeypatch, SCRIPTS["host"])
    gates = {"max_reruns": 1}
    code, report, md, out = _run(tmp_path, capsys, gates=gates)
    a = report["analysis"]
    assert code == WITHHELD_EXIT and a["withheld"] and a["decision"]["verdict"] == "invalid"
    assert calls == {"A": 1, "B": 2}
    assert "classified timeout gate invalid" in a["decision"]["reason"] and "host ceiling" in a["decision"]["reason"]
    assert "RUN INVALID, POINTS WITHHELD" in out.err and "RUN INVALID: POINTS WITHHELD." in md


@needs_sandbox
def test_sandboxed_arms_classify_against_an_interleaved_control(tmp_path):
    from strategy.experiments import run_experiment
    from tests.strategy.test_experiments import tiny

    spec = tiny(experiment_id="pytest-stalls-sandbox", isolation="sandbox", blocks_per_lineup=2,
                table={"players": 3, "starting_stack": 300, "hand_cap": 6})
    _, report = run_experiment(spec, out_dir=tmp_path / "run", jobs=1)
    classes = report["analysis"]["execution"]["timeout_classes"]
    for arm in ("A", "B"):
        c = classes[arm]
        assert c["control"]["decisions"] > 0 and c["candidate"]["decisions"] > 0
        [cell] = c["cells"].values()
        assert cell["selection"]["attempts"][0]["control"]["decisions"] > 0
    assert classes["B"]["strict"] and not classes["A"]["strict"]
    calibration = cell["control_calibration"]
    assert calibration["policy"] == "cpu_burner" and calibration["calibration_decisions"] > 0
    assert calibration["iterations"] >= 0 and calibration["ns_per_iteration"] > 0


@needs_sandbox
def test_burner_is_matched_to_a_target_decide_time(tmp_path):
    from arena.domain.models import TournamentConfig

    info = burn.matched_iterations([400_000, 100_000, 250_000], ruleset_id="practice_nlhe", seed="pytest-burner")
    assert info["target_median_ns"] == 250_000 and info["iterations"] > 0
    config = TournamentConfig(ruleset_id="practice_nlhe", initial_player_count=3, starting_stack=300, hand_cap=6)
    policy = burn.burn_bundle(tmp_path / "burner", info["iterations"])
    median = statistics.median(burn.calibrate(config, [policy.entrant("sandbox")] * 3, seed="pytest-burner-check"))
    assert 0.6 * 250_000 < median < 1.6 * 250_000
    assert burn.matched_iterations([], ruleset_id="practice_nlhe", seed="x") is None
