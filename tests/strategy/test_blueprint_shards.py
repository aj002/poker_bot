import numpy as np
import pytest

from strategy.blueprint import shards as S


def test_arrays_round_trip_in_parts(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "PART_BYTES", 1000)
    rng = np.random.default_rng(0)
    arrays = {"regret": rng.normal(size=5000).astype(np.float32), "strat": np.zeros(5000, np.float32)}
    S.put_arrays(tmp_path, arrays, {"x": 1})
    first = sorted(p.name for p in tmp_path.iterdir())
    assert sum(n.startswith("regret.") for n in first) > 1
    got, meta = S.get_arrays(tmp_path)
    assert meta == {"x": 1}
    for k in arrays:
        assert np.array_equal(got[k], arrays[k])
    arrays["strat"][3] = 2.0
    S.put_arrays(tmp_path, arrays, {"x": 2})
    got, meta = S.get_arrays(tmp_path)
    assert meta["x"] == 2 and got["strat"][3] == 2.0
    assert not set(first) & {p.name for p in tmp_path.iterdir() if p.name != "manifest.json"}


def test_corrupt_part_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "time", type("T", (), {"sleep": staticmethod(lambda s: None), "time": staticmethod(lambda: 1.0)}))
    S.put_arrays(tmp_path, {"regret": np.ones(10, np.float32)}, {})
    part = next(p for p in tmp_path.iterdir() if p.name.startswith("regret."))
    raw = bytearray(part.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    part.write_bytes(bytes(raw))
    with pytest.raises(Exception):
        S.get_arrays(tmp_path)


def test_merge_is_base_plus_shard_deltas_with_the_regret_floor():
    base = {"regret": np.array([1.0, -6.0, 0.0], np.float32), "strat": np.array([2.0, 0.0, 1.0], np.float32)}
    a = {"regret": np.array([3.0, -6.0, -6.0], np.float32), "strat": np.array([5.0, 1.0, 1.0], np.float32)}
    b = {"regret": np.array([0.0, -2.0, -6.0], np.float32), "strat": np.array([2.0, 4.0, 3.0], np.float32)}
    m = S.merge(base, [a, b], -6.0)
    assert m["strat"].tolist() == [5.0, 5.0, 3.0]
    assert m["regret"].tolist() == [2.0, -2.0, -6.0]
    assert S.merge(base, [a], -6.0)["regret"].tolist() == a["regret"].tolist()


def test_merged_progress_sums_shard_increments():
    base = {"chunks": 10, "pairs": 100, "traversals": 50, "nodes": 1000, "intervals": 100, "wall_seconds": 5.0,
            "value_sum": 1.0}
    s1 = dict(base, chunks=12, pairs=180, traversals=90, nodes=1800, wall_seconds=9.0, value_sum=2.0)
    s2 = dict(base, chunks=11, pairs=130, traversals=60, nodes=1300, wall_seconds=7.0, value_sum=1.5)
    p = S.merge_progress(base, [s1, s2], interval_pairs=1)
    assert (p["pairs"], p["nodes"], p["chunks"], p["intervals"]) == (210, 2100, 13, 210)


def test_shard_seeds_are_distinct():
    seeds = {S.shard_seed(e, s) for e in range(50) for s in range(20)}
    assert len(seeds) == 1000 and 1 not in seeds


def test_ledger_accumulates_and_survives_reboots(tmp_path, monkeypatch):
    busy = iter([100.0, 160.0, 5.0, 25.0])
    boot = iter(["a", "a", "b", "b"])
    monkeypatch.setattr(S.Ledger, "_busy", staticmethod(lambda: next(busy)))
    real = S.Path.read_text

    def fake(self, *a, **k):
        return next(boot) if str(self).endswith("boot_id") else real(self, *a, **k)

    monkeypatch.setattr(S.Path, "read_text", fake)
    led = S.Ledger(tmp_path / "l.json", start_cpu_s=7.0)
    assert led.tick() == 7.0 and led.tick() == 67.0
    led = S.Ledger(tmp_path / "l.json")
    assert led.tick() == 67.0 and led.tick() == 87.0


def test_status_flags_stale_and_missing_shards(tmp_path):
    now = S._now()
    S._write_json(tmp_path / "merger.json", {"time": now, "state": "running", "epoch": 0, "cpu_hours_total": 50.0,
                                             "stop_at": 292, "signals": None})
    S._write_json(tmp_path / "workers" / "shard-00.json", {"shard": 0, "host": "h", "state": "training", "epoch": 0,
                                                            "pairs": 1, "cpu_s": 3600, "restarts": 0, "time": now,
                                                            "last_push": now})
    S._write_json(tmp_path / "workers" / "shard-01.json", {"shard": 1, "host": "h", "state": "training", "epoch": 0,
                                                            "pairs": 1, "cpu_s": 3600, "restarts": 0,
                                                            "time": now - 3600, "last_push": now - 3600})
    verdict, lines, problems = S.status(tmp_path, expect=2)
    assert verdict == "ESCALATE"
    assert any("shard-01 heartbeat" in p for p in problems) and any("1 live shards" in p for p in problems)
    assert S.cpu_total(tmp_path, 30.0) == 32.0
