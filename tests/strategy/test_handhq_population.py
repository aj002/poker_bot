"""HandHQ population work: per-player extraction, Dirichlet fits, the train/validation-only
registry extension and the HandHQ-calibrated families."""

from __future__ import annotations

import json

import numpy as np
import pytest

from strategy.data import handhq_priors as hp
from strategy.data import handhq_stats as hs
from strategy.data.phh_import import FIXTURE_DIR
from strategy.population.families import ALL_FAMILIES, FAMILIES, HANDHQ_FAMILY_LIST
from strategy.population.handhq import COUNTS, FAMILY_IDS, LOCK_PATH, HandhqRegistry, verify_lock
from strategy.population.registry import default_registry

SAMPLE = FIXTURE_DIR.parent / "handhq" / "ps-25nl-sample.phhs"


def test_extraction_counts_the_sample():
    part = hs._file(SAMPLE)
    rep = part["report"]
    assert (rep["hands"], rep["replayed"]) == (3, 3)
    assert rep["seats"] == 18 and 0 < rep["seats_in_range"] <= 18
    counts = sum(c for c, _ in part["players"].values())
    col = {k: i for i, k in enumerate(hs.COLUMNS)}
    assert counts[col["hands"]] == rep["seats_in_range"]
    assert counts[col["vpip"]] >= counts[col["pfr"]] >= 1
    beliefs = sum(b for _, b in part["players"].values())
    assert beliefs.sum() == part["cells"].sum() == rep["public_rows"] > 0


def test_dirichlet_fits_recover_a_known_prior():
    rng = np.random.default_rng(3)
    alpha = np.array([12.0, 3.0, 5.0])
    p = rng.dirichlet(alpha, size=3000)
    counts = np.array([rng.multinomial(int(n), q) for n, q in zip(rng.integers(20, 120, size=3000), p)])
    got = hp.fit_dirichlet(counts)
    assert np.allclose(got / got.sum(), alpha / alpha.sum(), atol=0.02)
    assert 0.8 * alpha.sum() < got.sum() < 1.25 * alpha.sum()
    k = hp.fit_concentration(counts, alpha / alpha.sum())
    assert 0.8 * alpha.sum() < k < 1.25 * alpha.sum()


def test_extension_keeps_pop_v1_and_never_touches_holdout():
    base, ext = default_registry(), HandhqRegistry()
    assert ext.base_hash == base.hash
    assert all(ext.members[m].spec == x.spec for m, x in base.members.items())
    new = [m for m in ext.members.values() if m.member_id not in base.members]
    assert {m.family for m in new} == set(FAMILY_IDS)
    assert {m.partition for m in new} == {"train", "validation"}
    for fam in FAMILY_IDS:
        assert {p: sum(1 for m in new if m.family == fam and m.partition == p) for p in COUNTS} == dict(COUNTS)
    assert {m.member_id for m in ext.partition("holdout")} == {m.member_id for m in base.partition("holdout")}
    assert not set(FAMILY_IDS) & set(FAMILIES) and set(FAMILY_IDS) <= set(ALL_FAMILIES)


def test_extension_lock_is_current():
    verify_lock(LOCK_PATH)


def test_extension_lock_freezes_members():
    lock = json.loads(LOCK_PATH.read_text())
    ext = HandhqRegistry()
    assert lock["registry_hash"] == ext.hash and lock["base_registry_hash"] == default_registry().hash
    assert lock["extension_families"] == list(FAMILY_IDS)
    assert all(v["partition"] != "holdout" for k, v in lock["members"].items() if v["family"] in FAMILY_IDS)


def test_handhq_families_play_legally_and_stay_in_their_declared_ranges():
    from strategy.population.measure import census

    ext = HandhqRegistry()
    entries = [(m.member_id, m.spec) for m in ext.partition("train") if m.family in FAMILY_IDS][::3]
    tallies = census(entries, tables=6, seed="handhq-family-smoke", seats=6, config=hp.SIX_MAX_100BB)
    by_family: dict[str, list] = {}
    for key, t in tallies.items():
        assert not t.fallbacks
        by_family.setdefault(ext.members[key].family, []).append(t)
    assert sorted(by_family) == sorted(FAMILY_IDS)
    from strategy.population.measure import family_stats

    for fam in HANDHQ_FAMILY_LIST:
        stats = family_stats(by_family[fam.id])
        for stat, (lo, hi) in fam.expected.items():
            if stat in ("vpip", "pfr"):
                value, n = stats[stat]
                assert value is not None and lo - 0.05 <= value <= hi + 0.05, (fam.id, stat, value, n)
