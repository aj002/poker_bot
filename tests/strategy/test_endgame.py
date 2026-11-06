"""Stage 3 endgame: evaluator, ICM, settlement, subgames, solver gates, keys, tables, lookup and bots."""

from __future__ import annotations


import numpy as np
import pytest

from arena.tournament.scoring import Standings, icm
from strategy.endgame import points
from strategy.endgame.classes import hand_classes
from strategy.endgame.keys import (
    DEPTH_EDGES,
    KINDS,
    LEVELS,
    Snapshot,
    depth_bucket,
    icm_player,
    keys_for,
    pressure,
)
from strategy.endgame.showdown import ShowdownModel, load_showdown
from strategy.endgame.solver import PushFoldSolver
from strategy.endgame.subgame import Subgame, SubgameTree, ordering_codes
from strategy.tables import store
from tests.oracles.pots import clockwise_after
from tests.oracles.pots import settle as oracle_settle

PAYOUTS8 = (50.0, 30.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0)


# ------------------------------------------------------------------ isolation


def test_runtime_and_generators_do_not_import_numba():
    """Only the showdown generator needs numba; bots, lookup, tables, value model and gates do not."""
    import subprocess
    import sys

    code = ("import sys\n"
            "import strategy.endgame.bot, strategy.endgame.lookup, strategy.endgame.gates, strategy.endgame.limp\n"
            "import strategy.tables.endgame, strategy.tables.showdown, strategy.value.model\n"
            "import strategy.endgame.bots.points, strategy.endgame.bots.chart, strategy.endgame.bots.response\n"
            "import strategy.endgame.bots.capped, strategy.endgame.response, strategy.tables.exploit\n"
            "assert 'numba' not in sys.modules, sorted(m for m in sys.modules if 'numba' in m)\n")
    subprocess.run([sys.executable, "-c", code], check=True)


# ------------------------------------------------------------------------ ICM


def test_icm_batch_matches_the_arena_scorer_on_random_stacks():
    rng = np.random.default_rng(3)
    for _ in range(200):
        n = int(rng.integers(2, 9))
        stacks = rng.integers(1, 5000, n)
        prizes = sorted(rng.integers(0, 60, int(rng.integers(1, n + 1))).tolist(), reverse=True)
        want = icm(stacks.tolist(), [float(p) for p in prizes] + [0.0] * (n - len(prizes)))
        got = points.icm_batch(stacks[None, :], prizes)[0]
        assert np.allclose(got, want, atol=1e-9)
        assert np.allclose([icm_player(stacks.tolist(), i, prizes) for i in range(n)], want, atol=1e-9)


def test_icm_invariants():
    rng = np.random.default_rng(4)
    for _ in range(100):
        n = int(rng.integers(2, 9))
        s = rng.integers(1, 3000, n).astype(float)
        v = points.icm_batch(s[None, :], PAYOUTS8[:n])[0]
        assert v.sum() == pytest.approx(sum(PAYOUTS8[:n]))
        order = np.argsort(s, kind="stable")
        assert (np.diff(v[order]) >= -1e-9).all(), "more chips never means fewer points"
        assert (v <= PAYOUTS8[0] + 1e-9).all() and (v >= -1e-12).all()
        assert np.allclose(points.icm_batch(2 * s[None, :], PAYOUTS8[:n])[0], v), "scale invariant"
        perm = rng.permutation(n)
        assert np.allclose(points.icm_batch(s[perm][None, :], PAYOUTS8[:n])[0], v[perm]), "symmetric"
    # Heads-up ICM is affine in chips.
    v = points.icm_batch(np.array([[300.0, 700.0]]), (65.0, 35.0))[0]
    assert v[0] == pytest.approx(35 + 30 * 0.3)


def test_decision_time_icm_is_bit_identical_to_the_keys_and_leaves_no_garbage():
    import gc
    import random
    import struct

    from strategy.endgame import icm

    rng = random.Random(11)
    for _ in range(5000):
        n = rng.randint(1, 9)
        stacks = [rng.choice([0, rng.randint(1, 20000), rng.random() * 5000]) for _ in range(n)]
        prizes = sorted((rng.random() for _ in range(rng.randint(0, 6))), reverse=True) + [0.0] * rng.randint(0, 3)
        hero = rng.randrange(n)
        want, got = icm_player(stacks, hero, prizes), icm.icm_player(stacks, hero, prizes)
        assert struct.pack("<d", got) == struct.pack("<d", want), (stacks, hero, prizes)
    for _ in range(2000):
        n = rng.randint(2, 8)
        stacks = [rng.choice([0, rng.randint(1, 20000)]) for _ in range(n)]
        hero, villain = rng.sample(range(n), 2)
        want, got = pressure(stacks, hero, villain, PAYOUTS8), icm.pressure(stacks, hero, villain, PAYOUTS8)
        assert struct.pack("<d", got) == struct.pack("<d", want), (stacks, hero, villain)
    gc.collect()
    gc.disable()
    try:
        for _ in range(50):
            icm.pressure([1500, 900, 2400, 300, 1200, 800, 2000, 700], 0, 2, list(PAYOUTS8))
        assert gc.collect() == 0
    finally:
        gc.enable()


def test_eliminations_follow_the_arena_standings():
    rng = np.random.default_rng(5)
    for _ in range(300):
        n = int(rng.integers(2, 9))
        payouts = list(PAYOUTS8[:n]) if n < 8 else list(PAYOUTS8)
        seats = list(range(n))
        start = rng.choice([100, 200, 300, 400], n).tolist()
        end = [0 if rng.random() < 0.4 else 1 for _ in range(n)]
        if all(e == 0 for e in end):
            end[0] = 1
        st = Standings(payouts, seats)
        want = {p.seat: p.points for p in st.record_eliminations(1, dict(enumerate(start)), dict(enumerate(end)))}
        assert points.eliminations(start, end, payouts, n) == pytest.approx(want)


def test_settlement_matches_the_pot_oracle():
    rng = np.random.default_rng(6)
    for _ in range(500):
        n = int(rng.integers(2, 9))
        contrib = rng.integers(0, 400, n).tolist()
        folded = [bool(rng.random() < 0.3) for _ in range(n)]
        live = [p for p in range(n) if not folded[p] and contrib[p] > 0]
        if not live:
            continue
        top = max(range(n), key=lambda p: contrib[p])
        folded[top] = False
        strength = rng.integers(0, 4, n).tolist()
        button = int(rng.integers(n))
        odd = clockwise_after(button, range(n))
        got = points.settle(contrib, folded, strength, odd)
        want = oracle_settle({p: c for p, c in enumerate(contrib)}, [p for p in range(n) if folded[p]],
                             {p: strength[p] for p in range(n)}, button)
        assert got == [want.totals.get(p, 0) for p in range(n)]
        assert sum(got) == sum(contrib)


# ------------------------------------------------------------------- subgames


def _game(stacks, bb=100, payouts=PAYOUTS8, **kw):
    n = len(stacks)
    posts = [0] * n
    if n == 2:
        posts = [min(bb // 2, stacks[0]), min(bb, stacks[1])]
    else:
        posts[-2], posts[-1] = min(bb // 2, stacks[-2]), min(bb, stacks[-1])
    return Subgame(tuple(stacks), tuple(posts), tuple(payouts), bb, **kw)


@pytest.mark.parametrize("stacks", [(1000, 1000), (300, 2500), (900, 1200, 600), (500, 2000, 800, 1500, 100)])
def test_terminal_utilities_conserve_chips_and_points(stacks):
    for utility in ("points", "chips"):
        tree = SubgameTree(_game(stacks, utility=utility))
        total = sum(PAYOUTS8[:len(stacks)]) if utility == "points" else 0.0
        for t in tree.terminals.values():
            assert np.allclose(t.utility.sum(1), total), (utility, t)


def test_tree_shape_follows_push_fold_rules():
    tree = SubgameTree(_game((1000, 1000, 1000)))
    walk = tree.find(["f", "f"])
    assert tree.nodes[walk].player == -1 and tree.terminals[walk].showdown == (2,)
    three = tree.find(["s", "c", "c"])
    assert tree.terminals[three].showdown == (0, 1, 2)
    assert len(ordering_codes(3)) == 13
    # A player all-in from posting never acts and always reaches showdown.
    auto = SubgameTree(_game((1000, 1000, 60)))
    assert all(nd.player != 2 for nd in auto.nodes)
    assert all(2 in t.showdown for t in auto.terminals.values())


# ------------------------------------------------------------- synthetic model


def synthetic_model(h: int = 5, seed: int = 0) -> ShowdownModel:
    rng = np.random.default_rng(seed)
    w2 = rng.integers(1, 9, (h, h)).astype(float)
    w2 = w2 + w2.T
    prior = w2.sum(1) / w2.sum()
    strength = np.sort(rng.random(h))
    pt = np.full((h, h), 0.04)
    pw = np.clip(0.48 + 0.9 * (strength[:, None] - strength[None, :]), 0.02, 0.94)
    pw = (pw + (1 - pt - pw.T)) / 2
    np.fill_diagonal(pw, 0.48)
    codes = tuple(c for c, _ in ordering_codes(3))
    j3 = np.full((1, 1, 1, len(codes)), 1.0 / len(codes))
    return ShowdownModel("practice_nlhe", hand_classes("practice_nlhe"), {}, prior, w2 / w2.sum(1, keepdims=True),
                         w2 / w2.sum(), pw, pt, 1 - pw - pt, np.zeros(h, np.int64), 1, prior[None, :], j3, codes,
                         strength)


# -------------------------------------------------------------------- gates


@pytest.fixture(scope="module")
def nlhe():
    return load_showdown("practice_nlhe")


@pytest.mark.parametrize("bb_depth,ratio,final", [(2, 1.0, False), (8, 1.0, False), (15, 0.4, True), (25, 2.5, False)])
def test_heads_up_action_regret_gate(nlhe, bb_depth, ratio, final):
    bb = 100
    a = int(bb_depth * bb)
    stacks = (a, int(a * ratio))
    game = _game(stacks, payouts=(65.0, 35.0), final_hand=final)
    solver = PushFoldSolver(SubgameTree(game), nlhe).run(400)
    sigma = solver.average()
    for q in range(2):
        r = solver.report(sigma, q)
        assert r.max_action_regret / game.payouts[0] < 0.01
        assert r.gain / game.payouts[0] < 1e-3


def test_two_acting_players_with_a_posted_all_in_meet_the_gate(nlhe):
    """Three players left, the big blind all-in from posting: a non-affine two-player point game."""
    game = Subgame((900, 1400, 60), (0, 50, 60), PAYOUTS8, 100)
    solver = PushFoldSolver(SubgameTree(game), nlhe).run(400)
    sigma = solver.average()
    for q in (0, 1):
        assert solver.report(sigma, q).max_action_regret / PAYOUTS8[0] < 0.01


def test_points_equal_chips_when_points_are_affine_in_chips(nlhe):
    """Heads-up, ICM is affine in chips; CFR+ is invariant to a positive affine utility map."""
    game = _game((700, 1300), payouts=(65.0, 35.0))
    pts = PushFoldSolver(SubgameTree(game), nlhe).run(150).average()
    chips = PushFoldSolver(SubgameTree(_game((700, 1300), payouts=(65.0, 35.0), utility="chips")), nlhe)
    chips = chips.run(150).average()
    assert max(float(np.abs(pts[d] - chips[d]).max()) for d in pts) < 1e-6


def test_heads_up_ten_big_blind_nash_ranges(nlhe):
    """Matches published heads-up 10bb push/fold Nash: SB shoves about 58%, BB calls about 37%."""
    solver = PushFoldSolver(SubgameTree(_game((1000, 1000), utility="chips")), nlhe).run(400)
    sigma = solver.average()
    prior = solver.prior
    push = float(solver.strategy_at(sigma, []) @ prior)
    call = float(solver.strategy_at(sigma, ["s"]) @ prior)
    assert 0.55 < push < 0.62 and 0.34 < call < 0.41
    hc = hand_classes("practice_nlhe")
    assert solver.strategy_at(sigma, [])[hc.label_index("AA")] > 0.99
    assert solver.strategy_at(sigma, [])[hc.label_index("72o")] < 0.01


def test_icm_tightens_calls_on_the_bubble(nlhe):
    """Four left, three paid, a short stack in the big blind: equal big stacks must call each other
    far tighter than chip EV says, so the first of them shoves wider."""
    game = _game((3000, 3000, 3000, 600))
    solver = PushFoldSolver(SubgameTree(game), nlhe).run(250)
    sigma = solver.average()
    chips = PushFoldSolver(SubgameTree(_game((3000, 3000, 3000, 600), utility="chips")), nlhe).run(250)
    csig = chips.average()
    prior = solver.prior

    def share(s, sig, actions):
        return float(s.strategy_at(sig, actions) @ prior)

    assert share(solver, sigma, ["s"]) < 0.6 * share(chips, csig, ["s"])
    assert share(solver, sigma, []) > share(chips, csig, [])
    gains = solver.deviation_gains(sigma)
    assert gains.max() / PAYOUTS8[0] < 1e-3


# ----------------------------------------------------------------------- keys


def test_keys_are_distinct_per_kind_and_level_and_in_range():
    snap = Snapshot((1500, 400, 900, 2600, 300), (0, 0, 0, 50, 100), 100, PAYOUTS8, 120, 200)
    seen = set()
    for kind, args in (("open", (0,)), ("open", (2,)), ("call", (1, 0)), ("overcall", (3, 0, 1))):
        ks = keys_for(kind, snap, *args)
        assert len(ks) == LEVELS
        assert len(set(ks)) == LEVELS
        seen |= {(kind, k) for k in ks}
    assert depth_bucket(30 * 100, 100) == len(DEPTH_EDGES)
    assert depth_bucket(140, 100) == 0
    assert all(k in KINDS for k, _ in seen)


def test_pressure_is_zero_heads_up_and_positive_on_the_bubble():
    hu = (1000, 3000)
    assert abs(pressure(hu, 0, 1, (65.0, 35.0))) < 1e-12
    bubble = (3000, 3000, 3000, 600)
    assert pressure(bubble, 0, 1, PAYOUTS8[:4]) > 0.05


# ----------------------------------------------------------------- npz store


def test_npz_tables_round_trip_and_detect_tampering(tmp_path):
    arrays = {"a": np.arange(6, dtype=np.int64).reshape(2, 3), "b": np.array([0.5], np.float32)}
    path = store.write_arrays("unit", arrays, generator="t", inputs=["endgame/keys.py"], params={"x": 1},
                              seconds=0.0, extra={"ruleset_id": "practice_nlhe"}, directory=tmp_path)
    prov, got = store.load_arrays("unit", str(tmp_path))
    assert prov["ruleset_id"] == "practice_nlhe" and prov["data_sha256"] == store.arrays_sha256(arrays)
    assert np.array_equal(got["a"], arrays["a"]) and not got["a"].flags.writeable
    with np.load(path) as z:
        parts = {k: z[k] for k in z.files}
    parts["a"] = parts["a"] + 1
    np.savez(path, **parts)
    store.load_arrays.cache_clear()
    with pytest.raises(store.TableError):
        store.load_arrays("unit", str(tmp_path))
    store.load_arrays.cache_clear()


RULESETS = ("practice_nlhe", "uncertainty_shortdeck")
ENDGAME_TABLES = [f"endgame-{m}-{r}" for r in RULESETS for m in ("points", "chips", "chart", "response", "capped")] + \
    [f"showdown-{r}" for r in RULESETS] + [f"response-{r}" for r in RULESETS]


@pytest.mark.parametrize("name", ENDGAME_TABLES)
def test_committed_endgame_tables_are_fresh(name):
    assert store.array_path(name).exists(), f"{name} missing: make tables-endgame"
    assert not store.stale(name), f"{name} is stale: make tables-showdown tables-endgame"


def test_tables_are_per_variant_and_plo_is_rejected():
    from strategy.endgame.lookup import PushFoldTable, load_table
    from strategy.rules.variants import UnsupportedVariant
    from strategy.tables.endgame import build

    nlhe, short = load_table("points", "practice_nlhe"), load_table("points", "uncertainty_shortdeck")
    assert nlhe.provenance["variant_hash"] != short.provenance["variant_hash"]
    prov, arrays = store.load_arrays("endgame-points-practice_nlhe")
    with pytest.raises(store.TableError):
        PushFoldTable({**prov, "ruleset_id": "uncertainty_shortdeck"}, arrays)
    with pytest.raises(UnsupportedVariant):
        build("uncertainty_plo4", "points", rollout_lineups=0, synthetic=1, max_states=1, seed=1, jobs=1)


class _DictTable:
    """The original dict-and-bytes implementation, as the reference for the array layout."""

    def __init__(self, arrays):
        self.rows = {k: {key: r for r, key in enumerate(arrays[f"{k}_keys"].tolist())} for k in KINDS}
        self.probs = {k: [bytes(row) for row in arrays[f"{k}_probs"]] for k in KINDS}
        self.count = {k: arrays[f"{k}_count"].tolist() for k in KINDS}

    def lookup(self, kind, keys, cls, min_count=3, levels=None):
        for level, key in enumerate(keys):
            if levels is not None and level not in levels:
                continue
            r = self.rows[kind].get(key)
            if r is not None and self.count[kind][r] >= min_count:
                return self.probs[kind][r][cls] / 255.0, level
        return None


@pytest.mark.parametrize("name", ["endgame-points-practice_nlhe", "endgame-capped-uncertainty_shortdeck"])
def test_array_lookup_returns_exactly_what_the_dict_lookup_did(name):
    import random

    from strategy.endgame.lookup import PushFoldTable

    prov, arrays = store.load_arrays(name)
    table, ref = PushFoldTable(prov, arrays), _DictTable(arrays)
    streamed = PushFoldTable(*store.stream_arrays(name))
    rng = random.Random(11)
    n_classes = len(arrays["class_keys"])
    for kind in KINDS:
        keys = arrays[f"{kind}_keys"].tolist()
        assert table.rows(kind) == len(keys) == streamed.rows(kind)
        # every stored key at every class for the support levels in use, as a one-level chain
        for key in keys:
            for cls in (0, n_classes // 2, n_classes - 1, rng.randrange(n_classes)):
                for mc in (0, 3):
                    want = ref.lookup(kind, (key,), cls, mc)
                    assert table.lookup(kind, (key,), cls, min_count=mc) == want
        for key in rng.sample(keys, 300):
            for cls in range(n_classes):
                assert table.lookup(kind, (key,), cls) == ref.lookup(kind, (key,), cls)
                assert streamed.lookup(kind, (key,), cls) == ref.lookup(kind, (key,), cls)
        # back-off chains with absent keys, out-of-range keys and level filters
        for _ in range(4000):
            chain = tuple(rng.choice([rng.choice(keys), rng.choice(keys) + rng.choice([-1, 1]), rng.randrange(1 << 40),
                                      -3, 1 << 62, 1 << 70]) for _ in range(rng.randint(1, 5)))
            cls, mc = rng.randrange(n_classes), rng.choice([0, 1, 3, 10])
            levels = rng.choice([None, (0,), (1, 3), (0, 1, 2, 3)])
            want = ref.lookup(kind, chain, cls, mc, levels)
            assert table.lookup(kind, chain, cls, min_count=mc, levels=levels) == want


def test_array_lookup_keeps_every_row_when_probability_rows_collide(monkeypatch):
    from strategy.endgame import lookup

    prov, arrays = store.load_arrays("endgame-chart-practice_nlhe")
    monkeypatch.setattr(lookup, "_CHUNK", 7)
    rows = arrays["open_probs"]
    distinct, inverse = lookup._distinct_rows(rows)
    assert np.array_equal(distinct[inverse], rows) and len(distinct) <= len(rows)
    doubled = np.concatenate([rows, rows[::-1]])
    distinct, inverse = lookup._distinct_rows(doubled)
    assert len(distinct) == len(np.unique(rows, axis=0)) and np.array_equal(distinct[inverse], doubled)


def test_stream_arrays_checks_the_data_hash_and_leaves_no_raw_copy(tmp_path):
    arrays = {"b": np.arange(12, dtype=np.int32).reshape(3, 4), "a": np.array([1.5, 2.5])}
    store.write_arrays("t", arrays, generator="tests", inputs=["tables/store.py"], params={}, seconds=0.0,
                       directory=tmp_path)
    prov, items = store.stream_arrays("t", str(tmp_path))
    got = list(items)
    assert [k for k, _ in got] == ["a", "b"] and all(not a.flags.writeable for _, a in got)
    assert np.array_equal(got[1][1], arrays["b"]) and prov["data_sha256"] == store.arrays_sha256(arrays)
    assert store.stream_arrays("missing", str(tmp_path)) is None
    assert store.load_arrays("t", str(tmp_path), cache=False)[1]["b"].tolist() == arrays["b"].tolist()
    z = dict(np.load(tmp_path / "t.npz"))
    z["a"] = np.array([9.0, 2.5])
    np.savez(tmp_path / "t.npz", **z)
    _, items = store.stream_arrays("t", str(tmp_path))
    with pytest.raises(store.TableError, match="hash"):
        list(items)


FIELDS_PER_LEVEL = {"open": (6, 4, 3, 2), "call": (7, 6, 5, 3), "overcall": (5, 4, 3, 2)}


def _key_level(kind: str, key: int) -> int:
    code = KINDS.index(kind)
    hits = [lv for lv, nf in enumerate(FIELDS_PER_LEVEL[kind]) if key >> (6 * nf) == (code << 3 | lv)]
    if code == 0 and len(hits) > 1:
        # Open keys have kind code 0, so every shorter open key also has zero bits above level 0's fields.
        hits.remove(0)
    assert len(hits) == 1, (kind, key)
    return hits[0]


def test_key_layout_matches_the_level_decoder():
    snap = Snapshot((1500, 400, 900, 2600, 300), (0, 0, 0, 50, 100), 100, PAYOUTS8, 120, 200)
    for kind, args in (("open", (1,)), ("call", (2, 0)), ("overcall", (3, 0, 1))):
        assert [_key_level(kind, k) for k in keys_for(kind, snap, *args)] == list(range(LEVELS))


@pytest.mark.parametrize("ruleset", ["practice_nlhe", "uncertainty_shortdeck"])
def test_table_rows_have_the_variant_classes_and_mode_levels(ruleset):
    h = hand_classes(ruleset).count
    for mode in ("points", "chips", "chart"):
        prov, a = store.load_arrays(f"endgame-{mode}-{ruleset}")
        assert prov["mode"] == mode and prov["ruleset_id"] == ruleset
        for kind in KINDS:
            keys = a[f"{kind}_keys"]
            assert (np.diff(keys) > 0).all()
            assert a[f"{kind}_probs"].shape == (len(keys), h)
            levels = {_key_level(kind, int(k)) for k in keys.tolist()}
            assert levels == ({LEVELS - 1} if mode == "chart" else set(range(LEVELS))), (mode, kind)


# --------------------------------------------------------------- lookup, bots


def _play(policies, *, players=6, hand_cap=120, seed="endgame-test"):
    from arena.domain.models import TournamentConfig
    from strategy.value.rollouts import run_rollouts

    config = TournamentConfig(initial_player_count=players, hand_cap=hand_cap, starting_stack=600)
    return run_rollouts(config, policies, blocks=1, base_seed=seed, jobs=1)


def test_preflop_spots_match_the_engine(monkeypatch):
    """Every parsed spot agrees with the observation: stacks, posts, who acted, who is to act."""
    from strategy.endgame import bot as bot_mod
    from strategy.endgame import lookup

    checked = {"spots": 0, "decisions": 0, "kinds": set()}
    orig = bot_mod.PushFoldBot.push_fold

    def probe(self, obs, base, seed):
        checked["decisions"] += 1
        spot = lookup.preflop_spot(obs)
        if spot is not None:
            checked["spots"] += 1
            checked["kinds"].add(spot.kind)
            snap = spot.snapshot
            seats = spot.seated.seats
            assert seats[spot.hero] == obs.hero_seat
            by_seat = {p.seat: p for p in obs.players}
            for i, s in enumerate(seats):
                p = by_seat[s]
                assert p.stack + p.hand_committed == snap.stacks[i]
            assert obs.blinds.big_blind == snap.bb
            if spot.kind == "open":
                assert (obs.legal.call_amount or 0) <= snap.bb
        return orig(self, obs, base, seed)

    monkeypatch.setattr(bot_mod.PushFoldBot, "push_fold", probe)
    me = "strategy/endgame/bots/points.py"
    rollouts = _play([me, "style:aggressive", me, "baseline:calling_station", "style:rock", me])
    assert sum(rollouts[0].points.values()) == pytest.approx(100.0)
    assert checked["spots"] > 50 and {"open", "call"} <= checked["kinds"]


def test_bot_decisions_fit_the_budget(monkeypatch):
    import time

    from arena.domain.models import TournamentConfig
    from strategy.endgame import bot as bot_mod

    timings: list[int] = []
    orig = bot_mod.PushFoldBot.decide

    def timed(self, obs, state):
        t = time.perf_counter_ns()
        out = orig(self, obs, state)
        timings.append(time.perf_counter_ns() - t)
        return out

    monkeypatch.setattr(bot_mod.PushFoldBot, "decide", timed)
    _play(["strategy/endgame/bots/points.py", "strategy/endgame/bots/chips.py", "style:station",
           "strategy/endgame/bots/chart.py"], players=4, hand_cap=80, seed="budget")
    assert len(timings) > 100
    assert float(np.quantile(timings, 0.99)) < TournamentConfig().bot_budget_ns


def test_short_deck_bot_uses_its_own_tables(monkeypatch):
    from strategy.endgame import bot as bot_mod

    used: list[str] = []
    orig = bot_mod.load_table

    def spy(mode, ruleset_id, directory=None):
        used.append(ruleset_id)
        return orig(mode, ruleset_id, directory)

    monkeypatch.setattr(bot_mod, "load_table", spy)
    from arena.domain.models import TournamentConfig
    from strategy.value.rollouts import run_rollouts

    config = TournamentConfig(ruleset_id="uncertainty_shortdeck", initial_player_count=4, hand_cap=60,
                              starting_stack=400)
    me = "strategy/endgame/bots/points.py"
    run_rollouts(config, [me, "baseline:calling_station", me, "baseline:check_fold"], blocks=1, base_seed="sd")
    assert used and set(used) == {"uncertainty_shortdeck"}


@pytest.mark.parametrize("n", [2, 3, 4])
def test_harvest_weights_are_reach_probabilities(nlhe, n):
    from strategy.endgame.harvest import harvest, solve_snapshot

    stacks = tuple([1200] * n)
    posts = (50, 100) if n == 2 else tuple([0] * (n - 2) + [50, 100])
    snap = Snapshot(stacks, posts, 100, PAYOUTS8, 50, 200)
    solved = solve_snapshot(snap, nlhe, iterations=60, with_gains=False)
    recs = harvest(solved, max_depth_bb=30)
    first = [r for r in recs if r.kind == "open" and r.hero == 0]
    assert first and first[0].weight == pytest.approx(1.0)
    assert all(0.0 <= r.weight <= 1.0 + 1e-12 for r in recs)
    assert all(((r.row >= -1e-12) & (r.row <= 1 + 1e-12)).all() for r in recs)


# --------------------------------------------------------------- value model


def test_value_model_conserves_points_and_stays_within_prizes():
    from arena.domain.models import TournamentConfig
    from strategy.value.model import FEATURES, ValueModel

    base = ValueModel.icm(TournamentConfig())
    rng = np.random.default_rng(8)
    wild = ValueModel({k: rng.normal(0, 2.0, len(FEATURES)) for k in range(2, 9)}, base.blind_levels,
                      base.hands_per_level, base.hand_cap, base.payouts)
    for _ in range(50):
        k = int(rng.integers(2, 9))
        stacks = rng.integers(50, 6000, (20, k)).astype(float)
        prizes = base.payouts[:k]
        hand = int(rng.integers(1, 200))
        assert np.allclose(base.values(stacks, prizes, hand_number=hand), points.icm_batch(stacks, prizes))
        v = wild.values(stacks, prizes, hand_number=hand)
        assert np.allclose(v.sum(1), sum(prizes))
        assert (v >= min(prizes) - 1e-9).all() and (v <= max(prizes) + 1e-9).all()


def test_cap_continuation_reorders_survivors_consistently(nlhe):
    """Survivors arrive next-big-blind first; the cap hand is solved in its own preflop order."""
    from strategy.value.model import cap_value_fn, late_values

    snap = Snapshot((2000, 900, 1500), (0, 50, 100), 100, PAYOUTS8, 199, 200)
    fn = cap_value_fn(snap, nlhe, iterations=60)
    survivors = np.array([[1100, 2000, 1300]])
    got = fn(survivors, PAYOUTS8[:3])[0]
    cap = Snapshot((2000, 1300, 1100), (0, 50, 100), 100, PAYOUTS8, 200, 200)
    want = late_values(cap, nlhe, iterations=60)
    assert np.allclose(got, [want[2], want[0], want[1]])
    # Each player's value conditions the others on its own hand class, so the sum is
    # conserved only up to the class model's card-removal approximation.
    assert got.sum() == pytest.approx(100.0, abs=0.1)


# ----------------------------------------------------------- population response


def _ev(seat, kind, added=0, amount=None):
    from types import SimpleNamespace

    action = {"type": kind} if amount is None else {"type": kind, "amount": amount}
    return SimpleNamespace(type="ActionApplied", data={"seat": seat, "street": "preflop", "action": action,
                                                       "added": added})


def test_response_decisions_follow_the_abstraction():
    from strategy.endgame.response import hand_decisions

    # Preflop order seats 3, 4, 5 (small blind), 6 (big blind); 1000 chips each.
    snap = Snapshot((1000, 1000, 1000, 1000), (0, 0, 50, 100), 100, PAYOUTS8, 10, 200)
    seats = (3, 4, 5, 6)
    members = {3, 4, 5, 6}
    shove = [_ev(3, "fold"), _ev(4, "raise_to", 1000, 1000), _ev(5, "call", 950), _ev(6, "fold")]
    got = hand_decisions(snap, seats, shove, members)
    assert got == [("open", 0, None, None, 3, 0), ("open", 1, None, None, 4, 1), ("call", 2, 1, None, 5, 1),
                   ("overcall", 3, 1, 2, 6, 0)]
    # A non-all-in raise is recorded as "not a shove" and ends the abstraction.
    small = [_ev(3, "raise_to", 300, 300), _ev(4, "call", 300)]
    assert hand_decisions(snap, seats, small, members) == [("open", 0, None, None, 3, 0)]
    # Our own seat is never recorded; folding to the big blind is not a decision.
    folds = [_ev(3, "fold"), _ev(4, "fold"), _ev(5, "fold")]
    assert [d[4] for d in hand_decisions(snap, seats, folds, {3, 5, 6})] == [3, 5]


def test_logistic_fit_recovers_known_coefficients():
    from strategy.endgame.response import fit_logistic

    rng = np.random.default_rng(4)
    x = np.hstack([np.ones((20000, 1)), rng.normal(size=(20000, 2))])
    beta = np.array([-1.0, 2.0, -0.5])
    y = (rng.random(20000) < 1 / (1 + np.exp(-x @ beta))).astype(float)
    got = fit_logistic(lambda sl: x[sl], len(y), y, np.array([0.0, 1e-3, 1e-3]), chunk=7000)
    assert np.allclose(got, beta, atol=0.08)


def test_response_curves_come_from_training_members_only():
    from strategy.population import PopulationSource
    from strategy.value.rollouts import training_lineups

    source = PopulationSource()
    train = {m.policy for m in source.partition("train").eligible("practice_nlhe")}
    other = {m.policy for part in ("validation", "holdout") for m in source.partition(part).eligible("practice_nlhe")}
    teams = training_lineups("practice_nlhe", 8, 20, 11, fixed=("strategy/endgame/bots/points.py",))
    assert all(set(t[1:]) <= train and not set(t) & other for t in teams)
    for r in RULESETS:
        prov, _ = store.load_arrays(f"response-{r}")
        assert prov["partition"] == "train" and prov["ruleset_id"] == r


@pytest.mark.parametrize("ruleset", RULESETS)
def test_exploit_tables_match_the_committed_curves(ruleset):
    curves = store.load_arrays(f"response-{ruleset}")[0]
    for mode in ("response", "capped"):
        prov, _ = store.load_arrays(f"endgame-{mode}-{ruleset}")
        assert prov["response_sha256"] == curves["data_sha256"]
        assert prov["variant_hash"] == curves["variant_hash"]


def test_best_response_to_the_curves_beats_equilibrium_against_them(nlhe):
    from strategy.endgame.harvest import subgame_for
    from strategy.endgame.response import load_response, population_profile
    from strategy.tables.exploit import responses

    snap = Snapshot((900, 1400, 700, 1000), (0, 0, 50, 100), 100, PAYOUTS8, 120, 200)
    response = load_response("practice_nlhe")
    solver = PushFoldSolver(SubgameTree(subgame_for(snap)), nlhe)
    pop = population_profile(solver, snap, response)
    assert set(pop) == set(solver.decisions)
    eq = solver.run(250).average()
    for q in range(snap.n):
        mine = [d for d in solver.tree.player_nodes[q]]
        br_value, _ = solver.best_response(pop, q)
        eq_value = solver.values({**pop, **{d: eq[d] for d in mine}})[q]
        assert br_value >= eq_value - 1e-9
    records, gains, _ = responses(snap, None, "response", nlhe, response)
    assert records and all(set(np.unique(r.row)) <= {0.0, 1.0} for r in records)
    assert min(gains) >= -1e-9
