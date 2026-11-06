import copy
import itertools
import math

import numpy as np
import pytest
from pydantic import ValidationError

from arena.domain.models import TournamentConfig
from arena.table.driver import POLICIES, play_out, policy_rngs
from arena.table.service import seat_keyed_deck, seeded_table
from strategy.eval.luck import TERMS, LuckSettings, _hands, _Table, points_equity, table_luck
from strategy.eval.variance import DEFAULT_TERMS, adjust
from strategy.experiments import ExperimentSpec, load_spec, run_experiment
from strategy.experiments.schema import SPEC_DIR
from strategy.objective.points import outcome_from_events
from strategy.search.validate import equity_path

FAST = LuckSettings(runout_samples=100, equity_samples=100)
MIX = ("check_or_call", "legal_random", "shove", "check_or_call")


def play(seed, players=4, stack=400, cap=20, policies=MIX, **config):
    cfg = TournamentConfig(initial_player_count=players, starting_stack=stack, hand_cap=cap, **config)
    t = seeded_table(cfg, seed)
    seats = cfg.resolved_seats
    play_out(t, {s: POLICIES[policies[i % len(policies)]] for i, s in enumerate(seats)}, policy_rngs(seed, seats))
    return [e.model_dump(mode="json") for e in t.log]


def test_points_equity_telescopes_to_the_official_points():
    events = play("vr-equity", cap=30)
    outcome = outcome_from_events(events)
    hands = list(_hands(events))
    config = next(e["data"]["config"] for e in events if e["type"] == "TableStarted")
    for seat in range(4):
        live = [h for h in hands if seat in h.start]
        last = live[-1]
        order = sorted(last.start)
        v = points_equity(seat, last.start, [[last.final[s] for s in order]], config["payout_points"])[0]
        assert v == pytest.approx(outcome.points(seat), abs=1e-9)
        # after every hand it is the phase split's points-equity (strategy.search.validate)
        for h in live[:-1]:
            order = sorted(h.start)
            v = points_equity(seat, h.start, [[h.final[s] for s in order]], config["payout_points"])[0]
            assert v == pytest.approx(equity_path(events, seat, [h.number])[0], abs=1e-9)


def _runout_hands(seeds, cards_to_come):
    for seed in seeds:
        events = play(seed)
        config = next(e["data"]["config"] for e in events if e["type"] == "TableStarted")
        for h in _hands(events):
            runout = [d for d in h.deals if d["index"] > h.last_decision]
            if runout and 5 - len(runout[0]["board"]) == cards_to_come and len(runout[0]["live"]) >= 2:
                yield config, h, runout[0]


@pytest.mark.parametrize("cards_to_come", [1, 2])
def test_allin_term_is_an_exact_martingale_difference(cards_to_come):
    """Averaged over every possible runout, the all-in term of a real hand is exactly zero."""
    config, h, first = next(_runout_hands((f"vr-runout-{i}" for i in range(200)), cards_to_come))
    board = first["board"]
    known = set(board) | set(first["burned"]) | {c for cs in h.holes.values() for c in cs}
    unseen = [c for c in range(52) if c not in known]
    seat = sorted(first["live"])[0]
    xs = []
    for extra in itertools.combinations(unseen, cards_to_come):
        alt = copy.copy(h)
        alt.board = list(board) + list(extra)
        t = _Table(seat, config["payout_points"], FAST)
        t.allin(alt, first)
        xs.append(t.result()["terms"]["allin"])
    assert len(xs) == math.comb(len(unseen), cards_to_come)
    assert abs(math.fsum(xs) / len(xs)) < 1e-9
    assert max(abs(x) for x in xs) > 0.1


def test_allin_realized_stacks_match_the_table():
    checked = 0
    for config, h, first in _runout_hands((f"vr-real-{i}" for i in range(40)), 5):
        if any(len(awards) > 1 for _, _, awards in h.pots):
            continue
        live = sorted(first["live"])
        t = _Table(live[0], config["payout_points"], FAST)
        from strategy.eval.luck import _scores

        row = t.finals(h, live, _scores([h.holes[s] for s in live], first["board"],
                                        np.asarray([h.board[len(first["board"]):]], dtype=np.int64)))[0]
        assert row.tolist() == [float(h.final[s]) for s in sorted(h.start)]
        checked += 1
    assert checked >= 3


def test_luck_terms_are_deterministic_and_zero_mean():
    rows = [table_luck(play(f"vr-zero-{i}", players=3, stack=300, cap=12), 0, FAST) for i in range(80)]
    assert table_luck(play("vr-zero-0", players=3, stack=300, cap=12), 0, FAST) == rows[0]
    assert set(rows[0]["terms"]) == set(TERMS)
    for term in TERMS:
        xs = [r["terms"][term] for r in rows]
        sd = float(np.std(xs, ddof=1))
        if sd == 0:
            continue
        assert abs(float(np.mean(xs))) < 4 * sd / math.sqrt(len(xs)), term
    assert sum(r["counts"]["allin_hands"] for r in rows) > 20


def test_seat_keyed_deal_keeps_each_seats_cards():
    deck = [f"c{i}" for i in range(52)]
    out = seat_keyed_deck(deck, [0, 1, 2, 3], (2, 3, 0), 2)
    assert sorted(out) == sorted(deck)
    assert out[:6] == ("c2", "c3", "c0", "c6", "c7", "c4") and out[6:14] == tuple(f"c{i}" for i in range(8, 16))
    assert TournamentConfig(deal="seat_keyed").config_hash() != TournamentConfig().config_hash()
    assert "deal" not in TournamentConfig().resolved()
    # different play, same seed: a seat dealt into the same hand number holds the same cards
    styles = (("check_or_call",), ("check_or_fold", "legal_random"), ("check_or_call", "check_or_fold"))

    def holes(**config):
        tables = [play("vr-keyed", policies=p, cap=40, **config) for p in styles]
        cards = [{(h.number, s): tuple(c) for h in _hands(e) for s, c in h.holes.items()} for e in tables]
        shared = set(cards[0]) & set(cards[1]) & set(cards[2])
        return len(shared), sum(cards[0][k] == cards[1][k] == cards[2][k] for k in shared)

    shared, same = holes(deal="seat_keyed")
    assert shared > 100 and same == shared
    shared, same = holes()
    assert same < shared


def test_adjust_is_unbiased_and_covers():
    rng = np.random.default_rng(7)
    theta, n, reps = 0.7, 64, 400
    est, cover, factor = [], 0, []
    for _ in range(reps):
        z = rng.standard_t(5, (n, 2)) * [3.0, 1.0]
        d = theta + z @ [1.8, -0.5] + rng.normal(0, 2.0, n)
        rows = [dict(zip(("allin", "flop_pot"), r)) for r in z.tolist()]
        v = adjust(d.tolist(), rows, ["c"] * n, list(range(n)), terms=("allin", "flop_pot"))
        est.append(v["mean_difference"])
        cover += v["lo"] <= theta <= v["hi"]
        factor.append(v["variance_reduction_factor"])
    se = float(np.std(est, ddof=1)) / math.sqrt(reps)
    assert abs(float(np.mean(est)) - theta) < 4 * se
    assert 0.92 <= cover / reps <= 0.98
    # (Var z@b + 4) / 4 is about 13 before the cost of fitting
    assert 8 < float(np.median(factor)) < 15
    fixed = adjust(d.tolist(), rows, ["c"] * n, list(range(n)), terms=("allin", "flop_pot"),
                   coefficients={"allin": 1.8, "flop_pot": -0.5})
    assert fixed["mean_difference"] == pytest.approx(float(np.mean(d - z @ [1.8, -0.5])))
    assert fixed["df"] == n - 1


def test_stratified_interval_is_within_lineup():
    rng = np.random.default_rng(3)
    cells = [f"c{i % 4}" for i in range(80)]
    d = np.array([3.0 * (i % 4) for i in range(80)]) + rng.normal(0, 1, 80)
    rows = [{"allin": 0.0} for _ in range(80)]
    plain = adjust(d.tolist(), rows, cells, list(range(80)), terms=("allin",), coefficients={"allin": 0.0})
    strat = adjust(d.tolist(), rows, cells, list(range(80)), terms=("allin",), coefficients={"allin": 0.0},
                   stratify=True)
    assert strat["mean_difference"] == plain["mean_difference"]
    assert strat["se"] < plain["se"] / 3 and strat["df"] == 76


def test_confirm_uses_the_declared_estimator(tmp_path):
    import json

    from strategy.exploit.evaluate import confirm, one_sided

    primary = {"blocks": 100, "mean_difference": 0.5, "sd_difference": 5.0, "chosen_interval": [-0.49, 1.49]}
    vr = {"available": True, "primary": True, "blocks": 100, "mean_difference": 0.4, "lo": 0.1, "hi": 0.7,
          "se": 0.15, "df": 98}
    for name, analysis in (("plain", {"primary": primary}), ("vr", {"primary": primary, "variance_reduced": vr}),
                           ("report_only", {"primary": primary, "variance_reduced": vr | {"primary": False}})):
        (tmp_path / name).mkdir()
        (tmp_path / name / "report.json").write_text(json.dumps({"manifest": {"experiment_id": name},
                                                                 "analysis": analysis}))
    out = confirm({n: tmp_path / n for n in ("plain", "vr", "report_only")}, {})["hypotheses"]
    assert "estimator" not in out["report_only"]
    assert out["report_only"]["ci95"] == out["plain"]["ci95"] == [-0.49, 1.49]
    assert out["plain"]["p_one_sided"] == one_sided(0.5, 5.0, 100, 0.0)
    assert out["vr"]["estimator"] == "luck_cv" and out["vr"]["ci95"] == [0.1, 0.7]
    assert out["vr"]["p_one_sided"] < 0.01 < out["plain"]["p_one_sided"]


def _tiny(**kw):
    data = {"experiment_id": "pytest-vr", "title": "tiny", "stage": "discovery", "min_effect": 0.0,
            "arms": {"A": {"policy": "baseline:check_fold"}, "B": {"policy": "baseline:calling_station"}},
            "variants": ["practice_nlhe"], "table": {"players": 3, "starting_stack": 300, "hand_cap": 12},
            "population": {"source": "stub", "partition": "train", "lineups_per_variant": 1},
            "blocks_per_lineup": 4, "base_seed": "pytest-vr", "isolation": "in_process",
            "bootstrap_resamples": 200}
    return ExperimentSpec.model_validate(data | kw)


def test_old_specs_and_reports_are_unchanged():
    for path in SPEC_DIR.glob("*.json"):
        spec = load_spec(path.stem)
        assert "variance_reduction" not in spec.model_dump(mode="json")
        assert "deal" not in spec.model_dump(mode="json")["table"]
        assert spec.model_copy(update={"variance_reduction": None}).spec_hash() == spec.spec_hash()
    vr = _tiny(variance_reduction={})
    assert vr.variance_reduction.terms == list(DEFAULT_TERMS)
    assert vr.spec_hash() != _tiny().spec_hash()


def test_spec_validation():
    with pytest.raises(ValidationError, match="unknown luck terms"):
        _tiny(variance_reduction={"terms": ["allin", "luck"]})
    with pytest.raises(ValidationError, match="exactly the declared terms"):
        _tiny(variance_reduction={"terms": ["allin"], "coefficients": {"flop_pot": 1.0}})
    with pytest.raises(ValidationError, match="supports practice_nlhe only"):
        _tiny(variance_reduction={}, variants=["uncertainty_shortdeck"])
    assert _tiny(variance_reduction={"terms": ["allin"], "coefficients": {"allin": 1.0}}).variance_reduction


def test_runner_reports_the_adjusted_estimate(tmp_path):
    _, plain = run_experiment(_tiny(), out_dir=tmp_path / "plain", jobs=1)
    assert "variance_reduced" not in plain["analysis"]
    assert "estimator" not in plain["analysis"]["decision"]
    spec = _tiny(variance_reduction={"terms": ["allin", "deal_stack"], "folds": 2,
                                     "runout_samples": 100, "equity_samples": 100},
                 table={"players": 3, "starting_stack": 300, "hand_cap": 12, "deal": "seat_keyed"})
    run_dir, rep = run_experiment(spec, out_dir=tmp_path / "vr", jobs=1)
    v = rep["analysis"]["variance_reduced"]
    assert v["available"] and v["primary"] and v["blocks"] == 4 and len(v["pairs"]) == 4
    assert rep["analysis"]["decision"]["estimator"] == "luck_cv"
    assert v["raw"]["mean_difference"] == pytest.approx(rep["analysis"]["primary"]["mean_difference"])
    assert set(v["pairs"][0]["luck_a"]) == set(TERMS)
    md = (run_dir / "report.md").read_text()
    assert "## Luck-adjusted estimate" in md and "| decision interval | luck-adjusted (below) |" in md
    assert "| decision interval | paired_t |" in (tmp_path / "plain" / "report.md").read_text()
    assert rep["manifest"]["spec"]["table"]["deal"] == "seat_keyed"


def test_null_calibration_reports_the_adjusted_method(tmp_path):
    spec = _tiny(experiment_id="pytest-vr-null", kind="null_calibration",
                 arms={"A": {"policy": "baseline:legal_random", "salt": "a"},
                       "B": {"policy": "baseline:legal_random", "salt": "b"}},
                 blocks_per_lineup=8, null={"replicates": 2, "blocks_per_replicate": 4, "exact_coupling_blocks": 1},
                 variance_reduction={"terms": ["allin"], "folds": 2, "runout_samples": 100, "equity_samples": 100})
    _, rep = run_experiment(spec, out_dir=tmp_path / "run", jobs=1)
    assert "luck_cv" in rep["null_calibration"]["methods"]
    assert rep["null_calibration"]["exact_coupling"]["identical"]
