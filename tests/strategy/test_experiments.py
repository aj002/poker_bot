import json
import os
import stat
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from strategy.cli import WITHHELD_EXIT, main
from strategy.eval.execution import seat_timeouts
from strategy.eval.null import exact_coupling
from strategy.eval.outcomes import TableRecord
from strategy.eval.policies import resolve_policy, salted_bundle
from strategy.eval.population import Partition, StubPartitionSource, load_partition_source, sample_lineups
from strategy.experiments import (
    AlreadyRun,
    ExperimentSpec,
    ManifestConflict,
    ManifestTampered,
    load_manifest,
    load_spec,
    materialize,
    run_experiment,
    write_manifest,
)
from strategy.experiments.schema import DEFAULT_POPULATION_SOURCE, SPEC_DIR


def tiny(**kw):
    data = {
        "experiment_id": "pytest-tiny",
        "title": "tiny",
        "stage": "discovery",
        "min_effect": 1.0,
        "arms": {"A": {"policy": "baseline:check_fold"}, "B": {"policy": "baseline:calling_station"}},
        "variants": ["practice_nlhe"],
        "table": {"players": 3, "starting_stack": 300, "hand_cap": 12},
        "population": {"source": "stub", "partition": "train", "lineups_per_variant": 1},
        "blocks_per_lineup": 3,
        "base_seed": "pytest",
        "isolation": "in_process",
        "bootstrap_resamples": 200,
    }
    for key, value in kw.items():
        data[key] = value
    return ExperimentSpec.model_validate(data)


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("lab") / "run"
    run_dir, report = run_experiment(tiny(), out_dir=out, jobs=1)
    return run_dir, report


def test_checked_in_specs_validate():
    ids = sorted(p.stem for p in SPEC_DIR.glob("*.json"))
    assert {"lab-smoke", "null-aa-nlhe", "null-aa-variants", "null-aa-sandbox", "population-smoke"} <= set(ids)
    for spec_id in ids:
        spec = load_spec(spec_id)
        assert spec.experiment_id == spec_id
        assert spec.objective == "terminal_points"
    # the published null numbers were measured on the stub; the sandboxed null uses the registry
    assert load_spec("null-aa-nlhe").population.source == load_spec("null-aa-variants").population.source == "stub"
    assert load_spec("null-aa-sandbox").population.source == DEFAULT_POPULATION_SOURCE
    assert load_spec("null-aa-sandbox").isolation == "sandbox"
    assert load_spec("lab-smoke").population.source == DEFAULT_POPULATION_SOURCE


def test_default_population_is_the_registry(tmp_path):
    assert DEFAULT_POPULATION_SOURCE == "strategy.population:PopulationSource"
    spec = tiny(population={"partition": "train", "lineups_per_variant": 1})
    assert spec.population.source == DEFAULT_POPULATION_SOURCE
    m = materialize(spec, tmp_path / "w").manifest
    assert m.population.source == DEFAULT_POPULATION_SOURCE and not m.population.stub
    assert m.population.registry_id != StubPartitionSource.registry_id
    assert m.manifest_id != materialize(tiny(), tmp_path / "s").manifest.manifest_id


def test_spec_rules():
    with pytest.raises(ValidationError):
        tiny(population={"partition": "holdout"})
    with pytest.raises(ValidationError):
        tiny(stage="final_holdout")
    assert tiny(stage="final_holdout", population={"partition": "holdout"}).stage == "final_holdout"
    with pytest.raises(ValidationError):
        tiny(kind="null_calibration", null={"replicates": 2, "blocks_per_replicate": 2})
    with pytest.raises(ValidationError):
        tiny(primary_metric="chips")
    with pytest.raises(ValidationError):
        tiny(population={"partition": "train", "explicit_lineups": [["baseline:check_fold"]]})


def test_report_end_to_end(tiny_run):
    run_dir, report = tiny_run
    a = report["analysis"]
    assert a["primary"]["blocks"] == 3
    assert a["decision"]["verdict"] in {"pass", "fail", "inconclusive"}
    assert a["slices"]["seat"] and a["slices"]["variant"]
    assert a["execution"]["A"]["decisions"] > 0
    assert a["execution"]["B"]["illegal_actions"] == 0
    assert a["slices"]["opportunity_rates"]
    for pair in a["pairs"]:
        assert pair["d"] == pytest.approx(pair["u_b"] - pair["u_a"])
    assert a["withheld"] is False and a["execution"]["timeout_integrity"]["passed"]
    assert a["execution"]["timeout_integrity"]["arms"]["A"]["by_entrant"].keys() >= {"candidate"}
    assert report["run"]["jobs"] == 1 and report["run"]["safe_jobs"] >= 1
    md = (run_dir / "report.md").read_text()
    assert "Primary result" in md and "chips are diagnostics only" in md
    assert "Deadline integrity (every seat, both arms): ok" in md and "WITHHELD" not in md
    assert json.loads((run_dir / "report.json").read_text())["manifest"]["manifest_id"] == report["manifest"]["manifest_id"]
    assert not list(run_dir.rglob("tables/*.json")), "private histories must be deleted after summarizing"


def test_manifest_is_immutable_and_pins_everything(tiny_run):
    run_dir, report = tiny_run
    path = run_dir / "manifest.json"
    assert not os.stat(path).st_mode & stat.S_IWUSR
    m = load_manifest(path)
    assert m.manifest_id == report["manifest"]["manifest_id"]
    assert m.population.stub and m.population.partition.name == "train"
    assert m.rules[0].ruleset_id == "practice_nlhe" and len(m.rules[0].rules_hash) == 64
    assert m.policies["A"].artifact_sha256 == resolve_policy("baseline:check_fold").artifact_sha256
    assert m.spec["base_seed"].startswith("sha256:"), "the base seed must be redacted"
    assert '"base_seed":"pytest"' not in path.read_text()
    seeds = json.loads((run_dir / "private" / "seeds.json").read_text())
    assert seeds["base_seed"] == "pytest"
    with pytest.raises(AlreadyRun):
        run_experiment(tiny(), out_dir=run_dir)


def test_manifest_tamper_and_conflict(tiny_run, tmp_path):
    run_dir, _ = tiny_run
    data = json.loads((run_dir / "manifest.json").read_text())
    data["cells"][0]["blocks"] = 99
    bad = tmp_path / "manifest.json"
    bad.write_text(json.dumps(data))
    with pytest.raises(ManifestTampered):
        load_manifest(bad)
    other = materialize(tiny(min_effect=2.0), tmp_path / "w").manifest
    with pytest.raises(ManifestConflict):
        write_manifest(run_dir / "manifest.json", other)


def test_manifest_id_tracks_content(tmp_path):
    a = materialize(tiny(), tmp_path / "a").manifest
    b = materialize(tiny(), tmp_path / "b").manifest
    assert a.manifest_id == b.manifest_id
    assert materialize(tiny(base_seed="other"), tmp_path / "c").manifest.manifest_id != a.manifest_id
    assert materialize(tiny(table={"players": 3, "starting_stack": 300, "hand_cap": 13}),
                       tmp_path / "d").manifest.manifest_id != a.manifest_id


def test_salted_bundles(tmp_path):
    base = resolve_policy("style:aggressive")
    a = salted_bundle(base, "a", tmp_path / "a")
    a2 = salted_bundle(base, "a", tmp_path / "a2")
    b = salted_bundle(base, "b", tmp_path / "b")
    assert a.artifact_sha256 == a2.artifact_sha256 != b.artifact_sha256
    assert a.rulesets == base.rulesets


def test_salted_bundle_directory(tmp_path):
    bundle = tmp_path / "bundle"
    (bundle / "data").mkdir(parents=True)
    (bundle / "data" / "name.txt").write_text("dir_random\n")
    (bundle / "bot.py").write_text(
        "from pathlib import Path\n"
        "from arena.agents.baselines.legal_random import LegalRandom\n"
        "NAME = (Path(__file__).resolve().parent / 'data' / 'name.txt').read_text().strip()\n"
        "class DirRandom(LegalRandom):\n    name = NAME\n"
        "def make_bot():\n    return DirRandom()\n")
    a = salted_bundle(resolve_policy(str(bundle)), "a", tmp_path / "a")
    assert (a.path / "inner_bot.py").read_text() == (bundle / "bot.py").read_text()
    assert (a.path / "data" / "name.txt").exists() and "SALT = 'a'" in (a.path / "bot.py").read_text()
    spec = tiny(experiment_id="pytest-null-dir", kind="null_calibration",
                arms={"A": {"policy": str(bundle), "salt": "a"}, "B": {"policy": str(bundle), "salt": "b"}},
                blocks_per_lineup=4, null={"replicates": 2, "blocks_per_replicate": 2, "exact_coupling_blocks": 1})
    _, report = run_experiment(spec, out_dir=tmp_path / "run", jobs=1)
    assert report["null_calibration"]["exact_coupling"]["identical"]
    assert any(p["d"] != 0 for p in report["analysis"]["pairs"])


def test_real_null_calibration_and_exact_coupling(tmp_path):
    spec = tiny(experiment_id="pytest-null", kind="null_calibration",
                arms={"A": {"policy": "baseline:legal_random", "salt": "a"},
                      "B": {"policy": "baseline:legal_random", "salt": "b"}},
                blocks_per_lineup=6, null={"replicates": 3, "blocks_per_replicate": 2, "exact_coupling_blocks": 2})
    _, report = run_experiment(spec, out_dir=tmp_path / "run", jobs=1)
    null = report["null_calibration"]
    assert null["replicates"] == 3 and len(null["replicate_mean_differences"]) == 3
    assert set(null["methods"]) == {"paired_t", "cluster_bootstrap", "unpaired_welch_tables"}
    assert null["exact_coupling"] == {"blocks": 2, "tables": 6, "tables_with_timeouts": 0,
                                      "diverged_after_timeout": 0, "max_abs_difference": 0.0, "identical": True}
    diffs = [p["d"] for p in report["analysis"]["pairs"]]
    assert any(d != 0 for d in diffs), "salted arms must have independent policy noise"


def test_timed_out_run_withholds_points_loudly(monkeypatch, capsys, tmp_path):
    from strategy.eval import paired

    def late(attempts, labels):
        rows = seat_timeouts(attempts, labels)
        rows["candidate"]["timeouts"] = rows["candidate"]["decisions"] // 10
        return rows

    monkeypatch.setattr(paired, "seat_timeouts", late)
    spec = tiny(experiment_id="pytest-late", kind="null_calibration",
                arms={"A": {"policy": "baseline:legal_random", "salt": "a"},
                      "B": {"policy": "baseline:legal_random", "salt": "b"}},
                blocks_per_lineup=4, null={"replicates": 2, "blocks_per_replicate": 2})
    path = tmp_path / "late.json"
    path.write_text(json.dumps(spec.model_dump(mode="json")))
    out = tmp_path / "run"
    assert main(["experiment", str(path), "--out", str(out), "--jobs", "1", "--quiet"]) == WITHHELD_EXIT
    assert "RUN INVALID, POINTS WITHHELD: timeout integrity gate failed" in capsys.readouterr().err
    report = json.loads((out / "report.json").read_text())
    a = report["analysis"]
    assert a["withheld"] and a["decision"]["verdict"] == "invalid"
    assert {"pairs", "slices", "robustness"}.isdisjoint(a) and "mean_difference" not in a["primary"]
    assert not a["execution"]["timeout_integrity"]["passed"]
    assert report["null_calibration"]["withheld"]
    md = (out / "report.md").read_text()
    assert "RUN INVALID: POINTS WITHHELD." in md and "Primary result" not in md
    assert "Deadline integrity (every seat, both arms): FAILED" in md


def test_exact_coupling_sets_aside_tables_with_timeouts():
    def rec(rotation, points, timeouts=0):
        return TableRecord(f"t{rotation}", 0, rotation, {0: 0, 1: 1}, True, None, 10, points, timeouts=timeouts)

    config = SimpleNamespace(initial_player_count=2, resolved_seats=[0, 1])
    cell = SimpleNamespace(cell_id="c", ruleset_id="practice_nlhe", config=config, blocks=1,
                           lineup=SimpleNamespace(lineup_id="l", families=("f",)))

    def coupling(b_rows):
        runs = {"A": SimpleNamespace(records=[rec(0, {0: 30, 1: 0}), rec(1, {0: 0, 1: 30})]),
                "B": SimpleNamespace(records=b_rows)}
        return exact_coupling([SimpleNamespace(cell=cell, runs=runs)])

    same = coupling([rec(0, {0: 30, 1: 0}), rec(1, {0: 0, 1: 30})])
    assert same["identical"] and same["tables"] == 2 and same["tables_with_timeouts"] == 0
    late = coupling([rec(0, {0: 30, 1: 0}), rec(1, {0: 30, 1: 0}, timeouts=1)])
    assert late["identical"] and late["tables_with_timeouts"] == 1 and late["diverged_after_timeout"] == 1
    broken = coupling([rec(0, {0: 0, 1: 30}), rec(1, {0: 0, 1: 30})])
    assert not broken["identical"] and broken["max_abs_difference"] == 30


def test_stub_population_and_custom_sources():
    src = StubPartitionSource()
    train = src.partition("train")
    assert train.partition_hash() == src.partition("train").partition_hash()
    assert {m.member_id for m in train.members}.isdisjoint(m.member_id for m in src.partition("holdout").members)
    one = sample_lineups(train, ruleset_id="uncertainty_plo4", opponents=5, count=3, seed="x")
    assert one == sample_lineups(train, ruleset_id="uncertainty_plo4", opponents=5, count=3, seed="x")
    assert all("uncertainty_plo4" in o.rulesets for lu in one for o in lu.opponents)
    custom = load_partition_source("tests.strategy.test_experiments:OnePartition")
    assert custom.partition("train").members == ()
    with pytest.raises(ValueError):
        custom.sample_lineups(custom.partition("train"), ruleset_id="practice_nlhe", opponents=2, count=1, seed="s")


class OnePartition:
    registry_id = "empty"

    def registry_hash(self):
        return "0" * 64

    def partition(self, name):
        return Partition(name, self.registry_id, self.registry_hash(), "empty", ())

    def sample_lineups(self, partition, **kw):
        return sample_lineups(partition, **kw)


def test_cli(capsys, tmp_path):
    assert main(["sample-size", "--sigma", "10", "--delta", "2", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["blocks"] >= 197
    assert main(["specs"]) == 0
    assert "null-aa-nlhe" in capsys.readouterr().out
    assert main(["rules", "--ruleset", "uncertainty_plo4", "--json"]) == 0
    assert "betting.pot_limit" in capsys.readouterr().out
    assert main(["experiment", "lab-smoke", "--dry-run", "--out", str(tmp_path / "dry")]) == 0
    assert json.loads(capsys.readouterr().out)["experiment_id"] == "lab-smoke"
    assert main(["capacity", "--json"]) == 0
    cap = json.loads(capsys.readouterr().out)
    assert cap["cores_per_sandbox_job"] == 4 and cap["safe_jobs"]["sandbox"] >= 1
