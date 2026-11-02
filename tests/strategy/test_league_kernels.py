"""League abstract-game kernels (numba): paired values and a gated response to a mixture."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numba")

from strategy.blueprint import abstraction as A  # noqa: E402
from strategy.blueprint.deals import make_deals  # noqa: E402
from strategy.blueprint.states import heads_up_pool  # noqa: E402
from strategy.league import kernels as K  # noqa: E402,F401
from strategy.league import tables as T  # noqa: E402
from strategy.league.learner import AbstractLeague, SeatMixture, homogeneous, payoff_matrix  # noqa: E402

ROW = A.NUM_ACTIONS


def _caller() -> np.ndarray:
    """Calls (or checks) everywhere: the uniform kernel row put entirely on CALL."""
    t = np.zeros((A.NUM_KEYS, ROW), np.uint8)
    t[:, A.CALL] = 255
    return t.reshape(-1)


@pytest.fixture(scope="module")
def league():
    lg = AbstractLeague(heads_up_pool(20), threads=2)
    lg.add("uniform", T.uniform_table())
    lg.add("caller", _caller())
    lg.set_eval(make_deals("practice_nlhe", 11, 3000))
    return lg


def test_a_table_gains_nothing_over_itself(league):
    seats = homogeneous(league.row("caller"), league._eval["count"])
    g = league.gain("uniform", "uniform", seats)
    assert g["gain_points"] == 0.0 and g["player_hands"] > 0


def test_mixture_draws_follow_the_probabilities():
    mix = SeatMixture.combine([(0.5, {0: 1.0}), (0.5, {1: 1.0, 2: 3.0})])
    assert mix.rows == (0, 1, 2) and np.allclose(mix.probs, (0.5, 0.125, 0.375))
    draws = mix.draw(np.random.default_rng(0), 20000)
    assert abs((draws == 2).mean() - 0.375) < 0.01
    with pytest.raises(ValueError):
        SeatMixture((0,), (0.5,))


def test_a_gated_response_to_a_calling_station_beats_its_base(league):
    caller = league.row("caller")
    regret, sq, stats = league.train_response("uniform", SeatMixture.pure(caller), pairs=40000, z=2.0, seed=5,
                                              producers=1, chunk=10000)
    assert stats["pairs"] >= 40000 and stats["nodes"] > 0
    dev = T.deviations(regret, sq, league.tabs[league.row("uniform")], z=2.0, mix=1.0)
    assert len(dev) > 0
    league.add("response", T.apply_deviations(league.tabs[league.row("uniform")], dev))
    seats = homogeneous(caller, league._eval["count"])
    g = league.gain("response", "uniform", seats)
    assert g["gain_points"] - 3 * g["se"] > 0, g
    G = payoff_matrix(league, ["caller", "response"])
    assert G["gain"][0][0] == G["gain"][1][1] == 0.0
    direct = league.gain("response", "caller", seats)
    assert G["gain"][1][0] == pytest.approx(direct["gain_points"])
