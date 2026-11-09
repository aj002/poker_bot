"""The station response (strategy.exploit.station): settings, the reshaped rows, identity off-station, bundling."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from strategy.blueprint import abstraction as A
from tests.strategy.test_blueprint import uniform_pack
from tests.strategy.test_exploit import exploit_pack

CLASSIC = {"antes": [0, 3, 5, 10, 15, 25, 40, 50], "ante_posting": "classic"}


def test_settings_parse_off_value_and_mappings():
    from strategy.exploit.station import StationConfig, station_config

    assert station_config("off") is None and station_config(None) is None
    assert station_config("value") == StationConfig()
    assert station_config({"n_min": 12, "thin": 0}) == StationConfig(n_min=12.0, thin=0.0)
    with pytest.raises(ValueError, match="unknown"):
        station_config({"bluff": 1})
    with pytest.raises(ValueError):
        station_config("max")


def _decision(street: int, bucket: int, *, to_call: int = 0, stack: int = 2000, pot: int = 300,
              opponents: int = 1, raiser: int | None = None):
    """A postflop or preflop decision of hero seat 0 with every slot legal and fixed raise targets."""
    players = [SimpleNamespace(seat=0, status="active", stack=stack, street_committed=0)]
    players += [SimpleNamespace(seat=s, status="active", stack=stack, street_committed=to_call)
                for s in range(1, opponents + 1)]
    name = ("preflop", "flop", "turn", "river")[street]
    history = []
    if raiser is not None:
        history.append(SimpleNamespace(hand_id="h", type="ActionApplied",
                                       data={"street": name, "seat": raiser, "action": {"type": "raise_to"}}))
    obs = SimpleNamespace(hero_seat=0, players=players, street=name, hand_id="h", action_history=history,
                          pot_total=pot, blinds=SimpleNamespace(big_blind=20),
                          legal=SimpleNamespace(call_amount=to_call, fold=to_call > 0))
    mask = sum(1 << a for a in range(A.NUM_ACTIONS)) if to_call else sum(1 << a for a in range(1, A.NUM_ACTIONS))
    key = A.infoset_key(street, bucket, 0, A.OFFSETS)
    return obs, (key, mask, (0, 0, to_call + 40, to_call + pot // 2, to_call + pot, stack))


def _response():
    from strategy.exploit.station import StationConfig, StationResponse

    return StationResponse(StationConfig(), ["station", "blueprint"])


def test_no_bluff_bets_and_no_light_stack_offs_against_a_station():
    r = _response()
    flat = [0.0, 0.2, 0.2, 0.2, 0.2, 0.2]
    # a weak flop hand (equity bin 1) never bets: all the bet mass checks
    obs, f = _decision(1, 1 * 4)
    assert r.reshape(obs, f, flat, {1}) == pytest.approx([0.0, 1.0, 0, 0, 0, 0])
    # a value hand (bin 5) moves the all-in down to the pot bet (15% of the stack) and bets half of its checks
    obs, f = _decision(1, 5 * 4)
    assert r.reshape(obs, f, flat, {1}) == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4, 0.0])
    # with a short stack the pot bet commits too much as well: both go down to the half-pot bet
    obs, f = _decision(1, 5 * 4, stack=600)
    assert r.reshape(obs, f, flat, {1}) == pytest.approx([0.0, 0.1, 0.2, 0.7, 0.0, 0.0])
    # a near-nut hand (bin 7) may stack off
    obs, f = _decision(1, 7 * 4)
    assert r.reshape(obs, f, flat, {1})[A.ALL_IN] == pytest.approx(0.2)
    # facing a big bet from the station, a medium hand folds instead of calling or raising
    obs, f = _decision(2, 4 * 4, to_call=1000, raiser=1)
    p = r.reshape(obs, f, [0.2, 0.4, 0.1, 0.1, 0.1, 0.1], {1})
    assert p == pytest.approx([1.0, 0, 0, 0, 0, 0])
    # the same bet from a seat not detected as a station is still called
    obs, f = _decision(2, 4 * 4, to_call=1000, raiser=2, opponents=2)
    assert r.reshape(obs, f, [0.2, 0.4, 0.1, 0.1, 0.1, 0.1], {1})[A.CALL] > 0.4


def test_preflop_jams_need_strong_hands_and_isolation_raises_stand():
    from strategy.exploit.station import StationConfig, StationResponse

    r = _response()
    flat = [0.2, 0.2, 0.15, 0.15, 0.15, 0.15]
    obs, f = _decision(0, 3, to_call=20, pot=30)
    # a weak hand's jam raises the smallest size instead; the blueprint's own raises stand
    assert r.reshape(obs, f, flat, {1}) == pytest.approx([0.2, 0.2, 0.15, 0.3, 0.15, 0.0])
    strict = StationResponse(StationConfig(pre_raise=8), ["station", "blueprint"])
    p = strict.reshape(obs, f, flat, {1})
    assert p[A.ALL_IN] == 0 and p[A.MIN_RAISE] == p[A.RAISE_A] == p[A.RAISE_B] == 0
    obs, f = _decision(0, 14, to_call=20, pot=30)
    assert r.reshape(obs, f, flat, {1}) == flat
    # 10 BB deep: a good hand may jam
    obs, f = _decision(0, 10, to_call=20, pot=30, stack=200)
    assert r.reshape(obs, f, flat, {1})[A.ALL_IN] == pytest.approx(0.15)


def test_detection_needs_posterior_evidence_and_no_cooldown():
    from strategy.exploit.station import detected

    b = SimpleNamespace(seats={1: [[0.0], 8.0, [0.0], 0], 2: [[0.0], 3.0, [0.0], 0], 3: [[0.0], 9.0, [0.0], 4]},
                        raw={}, posterior=lambda seat: [0.9, 0.1])
    assert detected(b, 1, 0, 0.8, 6.0)
    assert not detected(b, 2, 0, 0.8, 6.0)
    assert not detected(b, 3, 0, 0.8, 6.0)
    assert not detected(b, 4, 0, 0.8, 6.0)
    assert not detected(b, 1, 0, 0.95, 6.0)


def test_the_depth_gate_ignores_stations_while_the_stacks_are_deep():
    from strategy.exploit.station import StationConfig, StationResponse

    b = SimpleNamespace(seats={1: [[0.0], 8.0, [0.0], 0]}, raw={}, posterior=lambda seat: [0.9, 0.1])
    gated = StationResponse(StationConfig(max_bb=40), ["station", "blueprint"])
    deep, _ = _decision(1, 20, stack=2000)
    short, _ = _decision(1, 20, stack=600)
    assert _response().stations(deep, b) == {1}
    assert gated.stations(deep, b) == set()
    assert gated.stations(short, b) == {1}
    # the hero's own short stack caps the depth against a deep station
    short.players[1].stack = 5000
    assert gated.stations(short, b) == {1}


def test_the_station_bot_plays_the_ante_bot_wherever_no_station_is_detected():
    """Decision for decision on the same observation and state, at every decision without a detected station."""
    from arena.agents.baselines import BASELINES
    from arena.agents.sdk import load_bot
    from strategy.antes.runtime import AnteExploitBot
    from strategy.exploit.station import StationExploitBot
    from strategy.population.measure import play
    from tests.strategy.test_final import _strip

    a = AnteExploitBot(overrides={"antes": "tables"})
    b = StationExploitBot(overrides={"antes": "tables", "station": "value"})
    orig = b.decide
    seen = {"same": 0, "station": 0}

    def decide(obs, state):
        action, out = orig(obs, state)
        if "st[" in (action.get("note", "") if isinstance(action, dict) else getattr(action, "note", "") or ""):
            seen["station"] += 1
        else:
            ra, sa = a.decide(obs, state)
            assert _strip(ra) == _strip(action) and sa == out
            seen["same"] += 1
        return action, out

    b.decide = decide
    others = [("cs0", load_bot(BASELINES["calling_station"])[0]), ("cs1", load_bot(BASELINES["calling_station"])[0]),
              ("h0", load_bot(BASELINES["nlhe_heuristic"])[0]), ("h1", load_bot(BASELINES["nlhe_heuristic"])[0])]
    for t in range(2):
        bots = list(others)
        bots.insert(t, ("hero", b))
        play(bots, f"station-shadow-{t}", config={"hand_cap": 40, **CLASSIC})
    assert seen["station"] > 0 and seen["same"] > 20


def test_a_station_override_bundles_the_station_bot(tmp_path):
    from strategy.final import package

    pack, xpack = uniform_pack(tmp_path / "u.npz"), exploit_pack(tmp_path / "x.npz")
    m = package.build(tmp_path / "s", pack=pack, xpack=xpack, overrides={"antes": "tables", "station": "value"})
    assert {"strategy/exploit/station.py", "strategy/antes/runtime.py"} <= set(m["files"])
    bot = (tmp_path / "s" / "bot.py").read_text()
    assert "StationFinalBot(" in bot and "'station': 'value'" in bot
    package.smoke(tmp_path / "s")
    assert package.entry({"antes": "tables"}) == ("strategy.antes.runtime", "AnteFinalBot")
    with pytest.raises(ValueError):
        package.entry({"translate": "postflop", "station": "value"})
