from arena.domain.models import TournamentConfig
from strategy.policy.timeout_cost import draw, points_cost, summarize
from strategy.search.evaluate import make_tasks
from strategy.search.population import load_population

CALLING_STATION = "arena/agents/baselines/calling_station.py"


def test_draws_are_fixed_and_uniform():
    assert draw("s", 3, 7) == draw("s", 3, 7) and draw("s", 3, 7) != draw("s", 3, 8)
    xs = [draw("s", 0, i) for i in range(4000)]
    assert 0 <= min(xs) and max(xs) < 1 and abs(sum(xs) / len(xs) - 0.5) < 0.02


def test_summarize_reports_cost_per_table_and_per_forced_timeout():
    base = [{"points": 10.0, "decisions": 100, "forced": 0}] * 3
    forced = [{"points": 4.0, "decisions": 100, "forced": 2}, {"points": 10.0, "decisions": 100, "forced": 0},
              {"points": 8.0, "decisions": 100, "forced": 1}]
    s = summarize(base, forced, 0.01, 1)
    assert s["forced"] == 3 and s["tables_with_forced"] == 2 and s["unforced_tables_changed"] == 0
    assert s["cost_per_table"]["mean"] == 8 / 3 and s["cost_per_forced_timeout"]["mean"] == 8 / 3
    assert s["forced_rate"] == 0.01


def test_forced_timeouts_replay_unforced_tables_exactly():
    config = TournamentConfig(ruleset_id="practice_nlhe", initial_player_count=4, hand_cap=20)
    tasks = make_tasks(load_population(), "train", ruleset_id="practice_nlhe", players=4, count=6,
                       namespace="pytest-timeout-cost")
    rep = points_cost(CALLING_STATION, tasks, config, 0.02)
    one, ten = rep["arms"]
    assert rep["tables"] == 6 and one["rate"] == 0.02 and ten["rate"] == 0.2
    assert one["unforced_tables_changed"] == 0 and ten["unforced_tables_changed"] == 0
    assert ten["forced"] > one["forced"] and ten["tables_with_forced"] > 0
