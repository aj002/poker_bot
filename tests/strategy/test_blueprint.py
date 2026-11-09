"""Blueprint abstraction, policy pack and deployed runtime (no numba needed)."""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from strategy.blueprint import abstraction as A
from strategy.blueprint.policy import LEVEL_DIVISORS, PolicyPack, build_levels, quantize, write_levels, write_pack

ROOT = Path(__file__).resolve().parents[2]


def uniform_pack(path: Path) -> Path:
    """A pack whose coarsest level covers every key with a uniform row."""
    n = A.NUM_KEYS // LEVEL_DIVISORS[-1]
    arrays = {}
    for level in range(len(LEVEL_DIVISORS) - 1):
        arrays[f"keys{level}"] = np.zeros(0, np.int32)
        arrays[f"probs{level}"] = np.zeros((0, A.NUM_ACTIONS), np.uint8)
        arrays[f"mass{level}"] = np.zeros(0, np.uint8)
    last = len(LEVEL_DIVISORS) - 1
    arrays[f"keys{last}"] = np.arange(n, dtype=np.int32)
    arrays[f"probs{last}"] = quantize(np.ones((n, A.NUM_ACTIONS)))
    arrays[f"mass{last}"] = np.full(n, 255, np.uint8)
    write_levels(path, arrays, {"ruleset_id": "practice_nlhe", "run": "uniform-test"})
    return path


# ------------------------------------------------------------------ abstraction


def test_context_key_is_in_range_and_decodes():
    rng = random.Random(3)
    for _ in range(5000):
        street = rng.randrange(4)
        behind, active = rng.randrange(0, 8), rng.randrange(2, 9)
        raises = rng.randrange(0, 6)
        bb = rng.choice([20, 100, 400])
        pot = rng.randrange(bb, 40 * bb)
        to_call = rng.choice([0, rng.randrange(1, 20 * bb)])
        eff_start = rng.randrange(bb, 120 * bb)
        eff = rng.randrange(0, eff_start + 1)
        cur = rng.randrange(0, eff_start + 1)
        committed = rng.randrange(0, cur + 1)
        acted, init, press = rng.randrange(2), rng.randrange(2), rng.randrange(4)
        ctx = A.context_key(street, behind, active, raises, to_call, pot, eff, eff_start, bb, acted, init, press,
                            cur, committed)
        assert 0 <= ctx < A.CONTEXTS
        f = A.context_fields(ctx)
        assert f["behind"] == min(behind, A.R_BEHIND - 1)
        assert f["active"] == min(active, A.R_ACTIVE + 1)
        assert f["raises"] == min(raises, A.R_RAISES - 1)
        assert f["price"] == A.price_bucket(to_call, pot)
        assert f["acted"] == acted and f["pressure"] == press
        if street == 0:
            want = A.depth_bucket(eff_start, bb) if raises == 0 else A.commit_bucket(cur, eff_start)
            assert f["depth"] == want
            assert f["initiative"] == (1 if committed * 10 > eff_start else 0)
        else:
            assert f["depth"] == A.spr_bucket(eff, pot) and f["initiative"] == init
        b = rng.randrange(A.BUCKETS[street])
        assert A.split_key(A.infoset_key(street, b, ctx, A.OFFSETS)) == (street, b, ctx)


def test_bucket_edges_are_monotone():
    bb = 100
    assert [A.depth_bucket(x * bb, bb) for x in (1, 8, 9, 13, 30, 80)] == [0, 0, 1, 2, 4, 7]
    assert [A.commit_bucket(c, 1000) for c in (24, 25, 45, 70, 120, 200, 350, 700, 1000)] == [0, 1, 2, 3, 4, 5, 6,
                                                                                             7, 7]
    assert A.price_bucket(0, 300) == 0 and A.price_bucket(1, 300) == 1 and A.price_bucket(300, 300) == 5


def test_action_targets_are_ordered_and_capped():
    rng = random.Random(5)
    for _ in range(5000):
        street = rng.randrange(4)
        bb = 100
        cur = rng.choice([0, bb, rng.randrange(bb, 3000)])
        committed = rng.randrange(0, cur + 1)
        to_call = cur - committed
        pot = cur + committed + rng.randrange(0, 4000)
        max_to = committed + rng.randrange(1, 8000)
        to_call = min(to_call, max_to - committed)
        min_to = cur + max(bb, rng.randrange(bb, 2000))
        raises = rng.randrange(0, 5)
        can = rng.randrange(2)
        mask, t_min, t_a, t_b, t_all = A.action_targets(street, cur, to_call, pot, min_to, max_to, raises, can)
        assert (mask >> A.CALL) & 1
        assert bool((mask >> A.FOLD) & 1) == (to_call > 0)
        if not can or max_to <= cur:
            assert mask & ~((1 << A.FOLD) | (1 << A.CALL)) == 0
            continue
        assert (mask >> A.ALL_IN) & 1 and t_all == max_to
        sized = [(a, t) for a, t in ((A.MIN_RAISE, t_min), (A.RAISE_A, t_a), (A.RAISE_B, t_b)) if (mask >> a) & 1]
        if raises >= A.MAX_SIZED_RAISES:
            assert sized == []
        for a, t in sized:
            assert cur < t < max_to or (a == A.MIN_RAISE and t == min_to < max_to)
        if (mask >> A.RAISE_A) & 1 and (mask >> A.RAISE_B) & 1:
            assert t_a != t_b


# ----------------------------------------------------------------------- pack


def test_quantize_rows_sum_to_255_and_keep_order():
    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.ones(A.NUM_ACTIONS), size=500)
    p[:10] = 0.0
    q = quantize(p)
    assert q.dtype == np.uint8 and (q.astype(int).sum(1) == 255).all()
    assert np.abs(q[10:] / 255.0 - p[10:]).max() <= 1.0 / 255.0 + 1e-12
    assert (q[:10].astype(int).max(1) - q[:10].astype(int).min(1) <= 1).all()


def test_pack_round_trip_and_back_off(tmp_path):
    n = LEVEL_DIVISORS[-1] * 4
    strat = np.zeros((n, A.NUM_ACTIONS))
    strat[5] = [0, 10, 0, 0, 0, 30]
    strat[LEVEL_DIVISORS[1] * 7 + 1] = [8, 0, 0, 0, 0, 0]
    strat[LEVEL_DIVISORS[1] * 7 + 2] = [0, 8, 0, 0, 0, 0]
    path = tmp_path / "pack.npz"
    prov = write_pack(path, strat.ravel(), {"ruleset_id": "practice_nlhe"})
    pack = PolicyPack(path)
    assert pack.provenance["data_sha256"] == prov["data_sha256"]
    row, level = pack.lookup(5)
    assert level == 0 and row == (0, 64, 0, 0, 0, 191)
    row, level = pack.lookup(LEVEL_DIVISORS[1] * 7 + 3)
    assert level == 1 and row[A.FOLD] in (127, 128) and sum(row) == 255
    assert pack.lookup(n - 1) is None
    dense = pack.dense(0)
    assert dense.shape == (A.NUM_KEYS, A.NUM_ACTIONS) and dense[5, 5] == pytest.approx(191 / 255)


def test_lookup_is_a_binary_search_on_a_full_size_pack(tmp_path):
    import time

    n = 500_000
    arrays = {}
    for level in range(len(LEVEL_DIVISORS)):
        keys = np.arange(0, A.NUM_KEYS // LEVEL_DIVISORS[level], 7, dtype=np.int32)[:n]
        arrays[f"keys{level}"] = keys
        arrays[f"probs{level}"] = quantize(np.ones((keys.size, A.NUM_ACTIONS)))
        arrays[f"mass{level}"] = np.full(keys.size, 255, np.uint8)
    path = tmp_path / "big.npz"
    write_levels(path, arrays, {"ruleset_id": "practice_nlhe"})
    pack = PolicyPack(path)
    probe = [int(k) for k in np.random.default_rng(0).integers(0, A.NUM_KEYS, 2000)]
    t = time.perf_counter()
    for k in probe:
        pack.lookup(k)
    assert (time.perf_counter() - t) / len(probe) < 50e-6


def test_resolved_table_matches_lookup(tmp_path):
    rng = np.random.default_rng(0)
    strat = np.zeros((A.NUM_KEYS, A.NUM_ACTIONS), np.float32)
    hot = rng.choice(A.NUM_KEYS, 4000, replace=False)
    strat[hot] = rng.random((hot.size, A.NUM_ACTIONS)) * rng.choice([1.0, 5.0, 40.0], (hot.size, 1))
    path = tmp_path / "pack.npz"
    write_pack(path, strat.ravel(), {"ruleset_id": "practice_nlhe"})
    pack = PolicyPack(path)
    table = pack.resolved()
    probe = np.concatenate([hot[:300], hot[:300] ^ 1, rng.integers(0, A.NUM_KEYS, 300)])
    levels = set()
    for key in probe.tolist():
        hit = pack.lookup(key)
        if hit is None:
            assert not table[key].any()
        else:
            levels.add(hit[1])
            assert np.allclose(table[key], np.array(hit[0]) / 255.0)
    assert len(levels) >= 3


def test_pack_rejects_tampering_and_other_abstractions(tmp_path):
    path = uniform_pack(tmp_path / "u.npz")
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    prov = json.loads(str(arrays.pop("__provenance__")))
    bad = dict(arrays)
    bad["probs3"] = bad["probs3"].copy()
    bad["probs3"][0, 0] ^= 1
    np.savez(tmp_path / "tampered.npz", __provenance__=np.array(json.dumps(prov)), **bad)
    with pytest.raises(ValueError, match="hash"):
        PolicyPack(tmp_path / "tampered.npz")
    other = dict(prov, abstraction=dict(prov["abstraction"], contexts=1))
    np.savez(tmp_path / "other.npz", __provenance__=np.array(json.dumps(other)), **arrays)
    with pytest.raises(ValueError, match="abstraction"):
        PolicyPack(tmp_path / "other.npz")


def test_build_levels_rejects_partial_key_spaces():
    with pytest.raises(ValueError):
        build_levels(np.zeros((LEVEL_DIVISORS[-1] + 1) * A.NUM_ACTIONS))


# -------------------------------------------------------------------- runtime


def test_deployed_path_does_not_import_numba():
    code = ("import sys\n"
            "import strategy.blueprint.runtime, strategy.blueprint.bots.blueprint\n"
            "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('numba', 'llvmlite')"
            " or m in ('strategy.blueprint.kernels', 'strategy.blueprint.train', 'strategy.solve'))\n"
            "assert not bad, bad\n")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


def test_bundle_is_self_contained(tmp_path):
    from strategy.blueprint import package

    pack = uniform_pack(tmp_path / "u.npz")
    manifest = package.build(pack, tmp_path / "bundle")
    package.smoke(tmp_path / "bundle")
    assert manifest["bytes"] < package.SIZE_LIMIT
    shipped = {str(p.relative_to(tmp_path / "bundle")) for p in (tmp_path / "bundle").rglob("*.py")}
    assert "strategy/blueprint/runtime.py" in shipped and "bot.py" in shipped
    assert not any(f in shipped for f in ("strategy/blueprint/kernels.py", "strategy/blueprint/train.py"))
    assert not any(f.startswith("strategy/solve/") for f in shipped)
    assert "strategy/tables/data/blueprint-practice_nlhe.npz" in {
        str(p.relative_to(tmp_path / "bundle")) for p in (tmp_path / "bundle").rglob("*.npz")}


def test_bot_is_loaded_and_warm_after_construction(tmp_path, monkeypatch):
    """A restored sandbox worker decides without calling initial_state first."""
    from strategy.blueprint import runtime

    monkeypatch.setenv("BLUEPRINT_PACK", str(uniform_pack(tmp_path / "u.npz")))
    played = []
    monkeypatch.setattr(runtime.BlueprintBot, "decide", lambda self, obs, state: played.append(obs) or (None, state))
    bot = runtime.BlueprintBot("points")
    assert set(bot._policies) == set(runtime.RULESETS)
    assert len(played) == runtime.WARM_PASSES * len(runtime.warm_observations("practice_nlhe")) > 0
    assert bot.stats == runtime.BlueprintBot._fresh_stats()


def _play(policies, *, players, hand_cap=120, seed="blueprint-test"):
    from arena.domain.models import TournamentConfig
    from strategy.value.rollouts import run_rollouts

    config = TournamentConfig(initial_player_count=players, hand_cap=hand_cap)
    return run_rollouts(config, policies, blocks=1, base_seed=seed, jobs=1)


@pytest.mark.parametrize("leaf", ["blueprint", "blueprint_capped", "blueprint_noleaf"])
def test_bot_plays_legal_tournaments_within_budget(tmp_path, monkeypatch, leaf):
    import time

    from arena.domain.models import Action, TournamentConfig
    from strategy.blueprint import runtime

    monkeypatch.setenv("BLUEPRINT_PACK", str(uniform_pack(tmp_path / "u.npz")))
    timings: list[int] = []
    notes: list[str] = []
    orig = runtime.BlueprintBot.decide

    def timed(self, obs, state):
        t = time.perf_counter_ns()
        out = orig(self, obs, state)
        timings.append(time.perf_counter_ns() - t)
        payload = dict(out[0])
        notes.append(payload.pop("note", ""))
        assert obs.legal.allows(Action.model_validate(payload)), (payload, obs.legal)
        return out

    monkeypatch.setattr(runtime.BlueprintBot, "decide", timed)
    me = f"strategy/blueprint/bots/{leaf}.py"
    rollouts = _play([me, "style:aggressive", "baseline:calling_station", "style:rock"], players=4)
    assert all(sum(r.points.values()) == pytest.approx(100.0) for r in rollouts)
    assert len(timings) > 200
    assert not any("miss" in n for n in notes)
    assert sum(n.startswith("bp L3") for n in notes) > 100
    if leaf == "blueprint_noleaf":
        assert not any("push/fold" in n for n in notes)
    assert float(np.quantile(timings, 0.99)) < TournamentConfig().bot_budget_ns


def test_price_translation_probability():
    from strategy.blueprint.translate import lower_price_probability

    pot_before = 1000
    def p(frac):
        bet = round(frac * pot_before)
        return lower_price_probability(bet, pot_before + bet)

    # Trained sizes sit on price edges and keep their bucket; so do the smallest and largest buckets.
    for frac in (0.5, 1.0, 2.0, 0.1, 3.0):
        assert p(frac) == 0.0
    assert lower_price_probability(0, 1000) == 0.0
    # Just above an edge reads as the lower bucket almost surely; the weight falls with the size.
    assert p(0.51) > 0.95 and p(1.02) > 0.95 and p(0.22) > 0.95
    assert p(0.3) > p(0.4) > p(0.49) > 0.0
    assert 0.0 < p(0.75) < 1.0


def test_translation_only_moves_the_price_digit():
    from arena.domain.observations import Observation
    from strategy.blueprint.runtime import warm_observations
    from strategy.blueprint.translate import PRICE_STRIDE, TranslatingBlueprintBot

    off, on = TranslatingBlueprintBot(), TranslatingBlueprintBot(translate="all")
    pol_off, pol_on = off.policy("practice_nlhe"), on.policy("practice_nlhe")
    pol_off.seed = pol_on.seed = "s"
    moved = 0
    for raw in warm_observations("practice_nlhe"):
        obs = Observation.model_validate_json(raw)
        f = pol_off.plain_features(obs)
        assert pol_off.features(obs) == f and pol_on.plain_features(obs) == f
        g = pol_on.features(obs)
        if f is None:
            assert g is None
            continue
        assert g[1:] == f[1:] and g[0] in (f[0], f[0] - PRICE_STRIDE)
        if g[0] != f[0]:
            moved += 1
            assert A.split_key(g[0])[:2] == A.split_key(f[0])[:2]
    assert moved > 0
    with pytest.raises(ValueError):
        TranslatingBlueprintBot(translate="sometimes")


def test_pseudo_harmonic_weights():
    from strategy.blueprint.translate import pseudo_harmonic

    assert pseudo_harmonic(1.0, 1.0, 4.0) == 1.0 and pseudo_harmonic(4.0, 1.0, 4.0) == 0.0
    assert 1.0 > pseudo_harmonic(1.5, 1.0, 4.0) > pseudo_harmonic(2.0, 1.0, 4.0) > pseudo_harmonic(3.0, 1.0, 4.0) > 0.0
    assert pseudo_harmonic(2.0, 1.0, 5.0) == pytest.approx(0.5)


RECORDER = '''from strategy.blueprint.runtime import BlueprintBot


class Recorder(BlueprintBot):
    def decide(self, obs, state):
        with open({out!r}, "a") as f:
            f.write(obs.model_dump_json() + "\\n")
        return super().decide(obs, state)


def make_bot():
    return Recorder()
'''
OVERBETTOR = '''from strategy.exploiters.bots import OverbetExploitBot


def make_bot():
    return OverbetExploitBot(pack={pack!r}, mode="polar", size={size})
'''


def overbet_observations(tmp_path):
    """The hero's observations at a three-handed table with two overbettors."""
    from arena.benchmark.runner import Entrant, TableRunner, schedule
    from arena.domain.models import TournamentConfig
    from arena.domain.observations import Observation
    from strategy.blueprint.runtime import pack_path

    out = tmp_path / "obs.jsonl"
    (tmp_path / "hero.py").write_text(RECORDER.format(out=str(out)))
    ents = [Entrant("hero", tmp_path / "hero.py")]
    for i, size in enumerate((1.5, 3.0)):
        (tmp_path / f"ob{i}.py").write_text(OVERBETTOR.format(pack=str(pack_path("practice_nlhe")), size=size))
        ents.append(Entrant(f"ob{i}", tmp_path / f"ob{i}.py"))
    cfg = TournamentConfig(ruleset_id="practice_nlhe", initial_player_count=3, starting_stack=2000, hand_cap=60)
    runner = TableRunner(cfg, ents, isolation="in_process")
    for a in schedule(cfg, 3, 1, "overbet-translation-test")[:3]:
        runner.run(a)
    return [Observation.model_validate_json(line) for line in out.read_text().splitlines()]


@pytest.mark.parametrize("mode", ["overbet", "harmonic"])
def test_size_translation_rekeys_only_mapped_bets(tmp_path, mode):
    from strategy.blueprint.translate import TranslatingBlueprintBot, read_as, size_map

    pol = TranslatingBlueprintBot(translate=mode).policy("practice_nlhe")
    pol.seed = "s"
    faced = mapped = 0
    for obs in overbet_observations(tmp_path):
        f = pol.plain_features(obs)
        lo, hi = size_map(obs, 0.0, mode == "harmonic"), size_map(obs, 1.0 - 1e-12, mode == "harmonic")
        if obs.street != "preflop" and obs.legal.call_amount:
            faced += 1
        if lo is None:
            assert hi is None and pol.features(obs) == f
            continue
        mapped += 1
        t = lo[0]
        assert hi[0] == t and lo[1] < t < hi[1]
        assert pol.plain_features(read_as(obs, t, t)) == f
        keys = [pol.plain_features(read_as(obs, *m)) for m in (lo, hi)]
        for g in keys:
            assert A.split_key(g[0])[:2] == A.split_key(f[0])[:2]
        assert keys[0][0] != f[0] or keys[1][0] != f[0]
        assert pol.features(obs) in [(g[0], *f[1:]) for g in keys]
    assert faced > mapped > 0


@pytest.mark.parametrize("mode", ["pot", "shove"])
def test_one_sided_overbet_reads(tmp_path, mode):
    from strategy.blueprint.translate import TranslatingBlueprintBot, read_as, size_map

    pol = TranslatingBlueprintBot(translate=mode).policy("practice_nlhe")
    pol.seed = "s"
    mapped = 0
    for obs in overbet_observations(tmp_path):
        f = pol.plain_features(obs)
        m = size_map(obs, 0.5, shove=mode == "shove", pot=mode == "pot")
        if m is None:
            assert size_map(obs, 0.0) is None and pol.features(obs) == f
            continue
        mapped += 1
        assert m == size_map(obs, 0.0, shove=mode == "shove", pot=mode == "pot")
        assert (m[1] < m[0]) if mode == "pot" else (m[1] > m[0])
        assert pol.features(obs) == (pol.plain_features(read_as(obs, *m))[0], *f[1:])
    assert mapped > 0


def test_reused_bot_does_not_read_another_tables_pressure(tmp_path, monkeypatch):
    """Hand ids repeat across tables; a bot instance reused for a new table starts with no cached pressure."""
    from types import SimpleNamespace

    from strategy.blueprint import runtime

    monkeypatch.setenv("BLUEPRINT_PACK", str(uniform_pack(tmp_path / "u.npz")))
    bot = runtime.BlueprintBot("points")
    policy = bot.policy("practice_nlhe")

    def obs(table_id):
        return SimpleNamespace(table_id=table_id, hand_id="h_001", hero_seat=0, payouts=[50, 30, 20])

    short = {0: 500, 1: 3000, 2: 100}
    deep = {0: 1000, 1: 1000, 2: 1000}
    fresh_deep = runtime.BlueprintBot("points").policy("practice_nlhe").pressure_bucket(obs("t1"), deep)
    first = policy.pressure_bucket(obs("t1"), short)
    assert first != fresh_deep, "stacks must give different buckets for the test to bite"

    # same table id and hand id, as the advisor and repeated tournaments produce
    bot.initial_state(SimpleNamespace(ruleset_id="practice_nlhe", policy_seed="s", table_id="t1"))
    assert policy.pressure_bucket(obs("t1"), deep) == fresh_deep

    # different table ids never collide even without initial_state
    assert policy.pressure_bucket(obs("t2"), short) == first
