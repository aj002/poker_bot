"""The ante-aware final bot's runtime: the stripped preflop read and the overlay pack."""

from arena.domain.models import TournamentConfig
from arena.domain.rng import SeededRng
from arena.table.projections import observation
from arena.table.service import Table


class _Pack:
    def __init__(self, rows):
        self.rows = rows
        self.provenance = {"name": "fake"}

    def lookup(self, key):
        return self.rows.get(key)


def test_overlay_wins_only_while_active_and_at_the_same_or_a_finer_level():
    from strategy.antes.runtime import OverlayPack

    base = _Pack({1: ("base", 1), 2: ("base", 0), 3: ("base", 2)})
    pack = OverlayPack(base, _Pack({1: ("over", 1), 2: ("over", 1), 4: ("over", 3)}))
    assert pack.provenance == {"name": "fake"}
    assert [pack.lookup(k) for k in (1, 2, 3, 4)] == [("base", 1), ("base", 0), ("base", 2), None]
    pack.active = True
    assert [pack.lookup(k) for k in (1, 2, 3, 4)] == [("over", 1), ("base", 0), ("base", 2), ("over", 3)]


def _first_observation(antes):
    kw = {"antes": [antes] * 8, "ante_posting": "big_blind"} if antes else {}
    config = TournamentConfig(initial_player_count=6, hand_cap=200, **kw)
    table = Table(config, env_rng=SeededRng("antes-runtime", "env"), start_stacks={s: 1500 for s in range(6)},
                  initial_button=0)
    return observation(table, table.actor_seat)


def test_the_stripped_read_drops_exactly_the_antes_and_only_in_ante_hands():
    from strategy.antes.runtime import AntePolicy
    from strategy.blueprint.runtime import BlueprintPolicy

    plain = BlueprintPolicy("practice_nlhe")
    ante = AntePolicy("practice_nlhe", plain.pack)
    obs = _first_observation(0)
    assert ante.features(obs) == plain.features(obs)
    obs = _first_observation(20)
    assert obs.settled_pot == 20 and obs.pot_total == 50
    assert ante.features(obs) == plain.features(obs.model_copy(update={"pot_total": 30}))
    assert ante.features(obs) != plain.features(obs)
