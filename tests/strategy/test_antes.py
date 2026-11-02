"""Antes on the strategy side: rules structure, conformance against PokerKit, features and bots."""

import dataclasses
import json

import pytest

from arena.domain.models import TournamentConfig
from arena.table.projections import observation
from arena.table.service import seeded_table
from strategy.endgame.lookup import preflop_spot
from strategy.features.state import extract_state, orbit_cost
from strategy.policy.spot import TableMemory
from strategy.rules.adapter import RulesAdapter, TournamentStructure
from strategy.rules.conformance import ArenaSimulator, run_conformance

LEVELS = [[10, 20], [15, 30], [25, 50]]


def config(posting: str | None, antes=(5, 10, 20), n: int = 6, **kw) -> TournamentConfig:
    extra = {} if posting is None else {"antes": list(antes), "ante_posting": posting}
    return TournamentConfig(ruleset_id="practice_nlhe", initial_player_count=n, starting_stack=1000,
                            blind_levels=LEVELS, hands_per_level=kw.pop("hands_per_level", 2),
                            hand_cap=kw.pop("hand_cap", 6), **extra, **kw)


def first_observation(cfg: TournamentConfig, seed: str = "antes-features"):
    table = seeded_table(cfg, seed, table_id="t_antes")
    return table, observation(table, table.actor_seat)


# ------------------------------------------------------------------- rules


def test_structure_keeps_ante_free_hash_and_reads_antes():
    plain = TournamentStructure.from_config(config(None).resolved())
    assert "antes" not in plain.as_dict() and "ante_posting" not in plain.as_dict()
    assert plain.ante_at(1) == 0
    with_antes = TournamentStructure.from_config(config("classic").resolved())
    assert with_antes.as_dict()["antes"] == [5, 10, 20] and with_antes.ante_posting == "classic"
    assert with_antes.ante_at(1) == 5 and with_antes.ante_at(3) == 10 and with_antes.ante_at(6) == 20
    hashes = {RulesAdapter.from_config(config(p)).rules_hash for p in (None, "classic", "big_blind")}
    assert len(hashes) == 3


@pytest.mark.parametrize("posting", ["classic", "big_blind"])
def test_conformance_with_antes_includes_ante_fixtures_and_pokerkit(posting):
    report = run_conformance("practice_nlhe", hands=60, ante_posting=posting)
    assert report.ok, report.as_dict()
    checks = {c.name: c for c in report.checks}
    plain = {c.name: c for c in run_conformance("practice_nlhe", hands=5).checks}
    assert checks["fixtures"].passed == plain["fixtures"].passed + 6
    assert checks["pokerkit"].passed >= 40
    assert "pokerkit" not in plain


class IgnoresAntes(ArenaSimulator):
    name = "ignores-antes"

    def new_hand(self, setup):
        return super().new_hand(dataclasses.replace(setup, ante=0, ante_posting=None))


class LiveBigBlindAnte(ArenaSimulator):
    """Posts a big-blind ante from every seat, as if it were a classic ante."""

    name = "bb-ante-as-classic"

    def new_hand(self, setup):
        if setup.ante_posting == "big_blind":
            setup = dataclasses.replace(setup, ante_posting="classic")
        return super().new_hand(setup)


@pytest.mark.parametrize("sim, posting", [(IgnoresAntes(), "classic"), (IgnoresAntes(), "big_blind"),
                                          (LiveBigBlindAnte(), "big_blind")])
def test_pokerkit_check_catches_broken_ante_rules(sim, posting):
    report = run_conformance("practice_nlhe", simulator=sim, hands=30, ante_posting=posting)
    checks = {c.name: c for c in report.checks}
    assert not checks["pokerkit"].ok and not checks["fixtures"].ok


# ---------------------------------------------------------------- features


def test_classic_ante_features():
    table, obs = first_observation(config("classic"))
    s = extract_state(obs)
    start = obs.players[0].stack + obs.players[0].hand_committed
    assert obs.blinds.ante == 5 and obs.settled_pot == 30
    assert s.pot == obs.pot_total == 30 + 30
    assert s.m == pytest.approx(s.start_stack / orbit_cost(10, 20, 5, "classic", 6)) == pytest.approx(
        s.start_stack / 60)
    assert s.m_next == pytest.approx(s.start_stack / (15 + 30 + 10 * 6))
    assert s.pot_odds == pytest.approx(20 / (60 + 20))
    assert start == 1000


def test_big_blind_ante_is_dead_money_in_features():
    table, obs = first_observation(config("big_blind", antes=(20, 30, 50)))
    s = extract_state(obs)
    assert obs.pot_total == 50 and obs.settled_pot == 20
    assert s.m == pytest.approx(s.start_stack / (10 + 20 + 20))
    assert s.dead_money == 20
    assert s.winnable_if_call == 70 and s.uncontestable == 0


def test_ante_free_features_unchanged_by_the_ante_code():
    _, obs = first_observation(config(None))
    s = extract_state(obs)
    assert s.m == pytest.approx(s.start_stack / 30) and s.m_next == pytest.approx(s.start_stack / 45)


def test_push_fold_snapshot_uses_post_ante_stacks_and_m_equivalent_depth():
    _, obs = first_observation(config("classic", antes=(10, 10, 20)))
    spot = preflop_spot(obs)
    assert spot is not None and spot.kind == "open"
    assert spot.snapshot.stacks == (990,) * 6
    assert spot.snapshot.bb == round(20 * (30 + 60) / 30)
    _, plain = first_observation(config(None))
    assert preflop_spot(plain).snapshot.bb == 20


def test_baseline_spot_eff_bb_is_m_equivalent_preflop():
    _, obs = first_observation(config("classic", antes=(10, 10, 20)))
    spot = TableMemory(obs.ruleset_id, obs.hero_seat, obs.seats).spot(obs)
    assert spot.ante_pot == 60
    assert spot.eff_bb == pytest.approx(1000 / 20 * 30 / 90)


# --------------------------------------------------------------------- bots


def _makers():
    from arena.agents.baselines import calling_station, legal_random, nlhe_heuristic
    from arena.agents.styles import aggressive, rock, station, switcher
    from strategy.endgame.bot import PushFoldBot
    from strategy.exploit.runtime import ExploitBot
    from strategy.league.bots.league import make_bot as league
    from strategy.policy.generic import GenericBot
    from strategy.policy.package import load_params
    from strategy.policy.runtime import RuntimeBot
    from strategy.population.registry import default_registry
    from tests.strategy.test_rehearsal import ROOT

    vector = load_params(ROOT / "strategy" / "policy" / "vectors" / "nlhe-v2.json")
    from tests.support.identity import family_firsts

    members = family_firsts(default_registry())
    return [GenericBot, RuntimeBot, lambda: RuntimeBot(vector), lambda: PushFoldBot("points"), ExploitBot, league,
            *(m.make_bot for m in members), aggressive.make_bot, rock.make_bot, station.make_bot, switcher.make_bot,
            nlhe_heuristic.make_bot, calling_station.make_bot, legal_random.make_bot]


@pytest.mark.parametrize("posting", ["classic", "big_blind"])
def test_every_nlhe_bot_plays_ante_tables_without_fallbacks(posting):
    from arena.agents.sdk import bot_info
    from arena.agents.worker import InProcessSeat, play_table
    from arena.benchmark.runner import Assignment, table_info

    antes = (5, 10, 20) if posting == "classic" else (20, 30, 50)
    cfg = config(posting, antes=antes, n=8, hands_per_level=15, hand_cap=45)
    makers = _makers()
    seen = set()
    for t in range(-(-len(makers) // 8)):
        seed = f"ante-bots|{posting}|{t}"
        a = Assignment(0, t, 0, seed, {s: k for k, s in enumerate(cfg.resolved_seats)})
        agents = {}
        for k, s in enumerate(a.seats):
            i = (8 * t + k) % len(makers)
            bot = makers[i]()
            seen.add(i)
            agent = InProcessSeat(s, bot, bot_info(bot), table_info(cfg, a, s, cfg.bot_budget_ns), "antes")
            agent.start()
            agents[s] = agent
        run = play_table(seeded_table(cfg, seed, table_id=a.table_id), agents)
        log = run.table.log
        assert any(e.type == "AntePosted" for e in log)
        assert not [e.data for e in log if e.type == "ActionFallback"]
        for s, agent in agents.items():
            state = json.loads(agent.committed)
            if isinstance(state, dict) and "fb" in state:
                assert state["fb"] == 0, (agent.bot.name, state["fb"])
    assert len(seen) == len(makers)
