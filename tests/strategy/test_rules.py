import ast
import sys
from pathlib import Path

import pytest

from arena.domain.models import RULESET_IDS, LegalActions, TournamentConfig
from strategy.rules import (
    KNOWN_VARIANTS,
    RulesAdapter,
    TournamentStructure,
    UnsupportedVariant,
    VariantSpec,
    canonical_actions,
    default_action,
    get_variant,
)
from strategy.rules.conformance import ArenaSimulator, run_conformance
from strategy.rules.variants import CAP_DECK_36, CAP_FLUSH_OVER_FULL_HOUSE, CAP_NLHE_ASSETS, CAP_POT_LIMIT

DECISION_TIME_MODULES = ("strategy/rules/canonical.py", "strategy/rules/variants.py", "strategy/rules/adapter.py",
                         "strategy/rules/reveal.py")
ALLOWED_ARENA = {"arena.agents.sdk", "arena.domain.observations"}
ROOT = Path(__file__).resolve().parents[2]


def adapter(ruleset="practice_nlhe", **kw):
    return RulesAdapter.from_config(TournamentConfig(ruleset_id=ruleset, initial_player_count=6, **kw))


def test_every_arena_ruleset_has_variant_metadata():
    assert set(KNOWN_VARIANTS) == set(RULESET_IDS)
    for ruleset_id, spec in KNOWN_VARIANTS.items():
        assert VariantSpec.from_dict(spec.as_dict()) == spec
        assert spec.ruleset_id == ruleset_id


def test_variant_metadata():
    nlhe, sd, plo = (get_variant(r) for r in ("practice_nlhe", "uncertainty_shortdeck", "uncertainty_plo4"))
    assert (nlhe.deck_size, nlhe.hole_cards, nlhe.board_cards) == (52, 2, 5)
    assert (sd.deck_size, sd.hole_cards) == (36, 2)
    assert (plo.deck_size, plo.hole_cards, plo.betting) == (52, 4, "pot_limit")
    assert nlhe.supports(CAP_NLHE_ASSETS) and not sd.supports(CAP_NLHE_ASSETS) and not plo.supports(CAP_NLHE_ASSETS)
    assert sd.supports(CAP_DECK_36, CAP_FLUSH_OVER_FULL_HOUSE)
    assert plo.supports(CAP_POT_LIMIT)
    with pytest.raises(UnsupportedVariant):
        plo.require(CAP_NLHE_ASSETS, asset="preflop charts")
    with pytest.raises(UnsupportedVariant):
        get_variant("mystery_variant")


def test_rules_hash_tracks_rules_not_budgets():
    base = adapter()
    assert adapter().rules_hash == base.rules_hash
    assert adapter(bot_budget_ns=5_000_000).rules_hash == base.rules_hash
    assert adapter(human_turn_ms=1000).rules_hash == base.rules_hash
    assert adapter(hand_cap=150).rules_hash != base.rules_hash
    assert adapter(starting_stack=1500).rules_hash != base.rules_hash
    assert adapter(payout_points=[60, 25, 15, 0, 0, 0]).rules_hash != base.rules_hash
    other = adapter("uncertainty_shortdeck")
    assert other.rules_hash != base.rules_hash and other.variant_hash != base.variant_hash
    assert adapter(hand_cap=150).variant_hash == base.variant_hash


def test_structure_schedule():
    s = TournamentStructure.from_config(TournamentConfig(ruleset_id="practice_nlhe", initial_player_count=4,
                                                         hand_cap=60, hands_per_level=25).resolved())
    assert s.level(1) == 0 and s.level(25) == 0 and s.level(26) == 1
    assert s.hands_to_next_level(1) == 24 and s.hands_to_next_level(25) == 0
    assert s.hands_to_next_level(51) is None
    assert s.hands_to_cap(60) == 0 and s.hands_to_cap(1) == 59


def _legal(**kw):
    base = dict(fold=True, check=False, call_amount=20, can_raise=True, min_full_raise_to=40, max_raise_to=1000,
                short_all_in_raise_to=None, all_in_raise_to=1000, pot_raise_to=None)
    return LegalActions(**(base | kw))


def test_canonical_menu_is_legal_and_deduplicated():
    legal = _legal()
    menu = canonical_actions(legal, pot_total=10, current_bet_to=20, fractions=(0.5, 1.0, 100.0))
    keys = [a.key for a in menu]
    assert keys == ["fold", "call", "raise_min", "raise_fraction:1", "raise_all_in"]
    assert next(a for a in menu if a.kind == "raise_fraction").amount == 50
    assert all(legal.allows(a.action()) for a in menu)
    # 0.5 pot lands below the minimum raise and 100x above the stack: both dropped.
    # 1.2 pot below equals the pot-limit raise, which wins the tie.
    pot = _legal(pot_raise_to=80, max_raise_to=80, all_in_raise_to=None)
    menu = canonical_actions(pot, pot_total=30, current_bet_to=20, fractions=(1.2,))
    assert [a.key for a in menu] == ["fold", "call", "raise_min", "raise_pot"]
    short = _legal(can_raise=True, min_full_raise_to=None, max_raise_to=None, all_in_raise_to=None,
                   short_all_in_raise_to=30)
    assert [a.key for a in canonical_actions(short)] == ["fold", "call", "raise_all_in"]
    with pytest.raises(ValueError):
        canonical_actions(legal, fractions=(0.5,))


def test_default_action_is_check_or_fold():
    assert default_action(_legal()).type == "fold"
    assert default_action(_legal(check=True, call_amount=None)).type == "check"


def _imports(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize("module", DECISION_TIME_MODULES)
def test_decision_time_modules_use_only_public_contracts(module):
    for name in _imports(ROOT / module):
        top = name.split(".")[0]
        if top in sys.stdlib_module_names or top == "__future__":
            continue
        if top == "strategy":
            assert name.replace(".", "/") + ".py" in DECISION_TIME_MODULES, name
            continue
        assert name in ALLOWED_ARENA, f"{module} imports {name}"


@pytest.mark.parametrize("ruleset", sorted(RULESET_IDS))
def test_arena_engine_conforms(ruleset):
    report = run_conformance(ruleset, hands=40)
    assert report.ok, report.as_dict()
    names = {c.name for c in report.checks}
    assert {"metadata", "ranking", "fixtures", "scoring", "differential"} <= names
    fixtures = next(c for c in report.checks if c.name == "fixtures")
    assert fixtures.passed > 0


class StandardRankingEverywhere(ArenaSimulator):
    """A simulator that forgot short deck's flush-over-full-house rule."""

    name = "broken-ranking"

    def rank_five(self, ruleset_id, cards):
        from pokerkit import StandardHighHand

        return StandardHighHand(cards)


class WrongHoleCount(ArenaSimulator):
    name = "broken-metadata"

    def metadata(self, ruleset_id):
        return super().metadata(ruleset_id) | {"hole_cards": 3}


def test_conformance_detects_broken_simulators():
    report = run_conformance("uncertainty_shortdeck", simulator=StandardRankingEverywhere(), hands=5)
    assert not next(c for c in report.checks if c.name == "ranking").ok
    report = run_conformance("practice_nlhe", simulator=WrongHoleCount(), hands=5)
    assert not next(c for c in report.checks if c.name == "metadata").ok
