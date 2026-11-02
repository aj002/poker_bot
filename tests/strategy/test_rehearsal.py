"""Build-day reveal slot, the variant-agnostic fallback and the rehearsal pipeline.

Card-layer caches are per process and keyed by ruleset id, and a process serves one
reveal variant, so every check that activates a reveal variant runs in a subprocess.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from strategy.policy.package import FALLBACK_FILES, build_bundle

ROOT = Path(__file__).resolve().parents[2]
HELDBACK = ROOT / "strategy" / "rehearsal" / "heldback"
PLAYABLE = ("plo5-8max", "shortdeck-trips-6max-turbo")
ALLOWED_ARENA = {"arena.agents.sdk", "arena.domain.observations"}


def reveal_dir(tmp_path: Path, name: str, **override) -> Path:
    ann = json.loads((HELDBACK / f"{name}.json").read_text())
    d = tmp_path / f"reveal-{name}"
    d.mkdir()
    (d / "variant.json").write_text(json.dumps({"ruleset_id": "reveal_variant", **ann["variant"], **override}))
    return d


def run_py(code: str, reveal: Path | None = None, timeout: int = 600) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "ARENA_REVEAL_VARIANT"}
    if reveal is not None:
        env["ARENA_REVEAL_VARIANT"] = str(reveal)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, env=env,
                         timeout=timeout)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("name", PLAYABLE)
def test_reveal_variant_conforms_and_the_card_layer_matches_the_engine(tmp_path, name):
    rep = run_py(
        "import json\n"
        "from arena.domain.models import TournamentConfig\n"
        "from strategy.rules.conformance import run_conformance\n"
        "from strategy.rehearsal.runner import evaluator_differential, smoke_tournaments\n"
        "from strategy.rules.reveal import reveal_variant\n"
        "r = run_conformance('reveal_variant', hands=30)\n"
        "checks = {c.name: (c.ok, c.passed, c.failed) for c in r.checks}\n"
        "ev = evaluator_differential('reveal_variant', hands=400)\n"
        "sm = smoke_tournaments(TournamentConfig(ruleset_id='reveal_variant', initial_player_count=5, hand_cap=40),"
        " tables=2)\n"
        "print(json.dumps({'checks': checks, 'ev': ev, 'smoke': sm, 'hash': reveal_variant().variant_hash()}))\n",
        reveal_dir(tmp_path, name))
    checks = rep["checks"]
    for name_ in ("metadata", "ranking", "differential", "scoring"):
        assert checks[name_][0], (name_, checks[name_])
    assert checks["fixtures"][1:] == [0, 0]  # no fixtures exist for a reveal variant: not applicable, not failed
    assert rep["ev"]["status"] == "ok" and rep["ev"]["disagreements"] == 0 and rep["ev"]["hands"] == 400
    assert rep["smoke"]["ok"] and rep["smoke"]["points_conserved"] and rep["smoke"]["internal_fallbacks"] == 0


@pytest.mark.parametrize("override, message", [
    ({"hole_cards": 6}, "hole cards"),
    ({"betting": "fixed_limit"}, "betting"),
    ({"ranks": "3456789TJQKA"}, "deck"),
    ({"lowest_straight": "6789T"}, "lowest straight"),
    ({"board_streets": [3, 2]}, "streets"),
])
def test_unsupported_reveal_variants_fail_before_a_hand_is_dealt(tmp_path, override, message):
    rep = run_py(
        "import json\n"
        "from arena.rules.profiles import get_profile\n"
        "try:\n"
        "    get_profile('reveal_variant'); print(json.dumps({'error': None}))\n"
        "except Exception as exc:\n"
        "    print(json.dumps({'error': f'{type(exc).__name__}: {exc}'}))\n",
        reveal_dir(tmp_path, "plo5-8max", **override))
    assert rep["error"] and rep["error"].startswith("UnsupportedRules") and message in rep["error"], rep


def test_reveal_slot_is_inactive_without_a_spec():
    from strategy.rules.reveal import reveal_active
    from strategy.rules.variants import UnsupportedVariant

    if os.environ.get("ARENA_REVEAL_VARIANT"):
        pytest.skip("a reveal variant is active in this environment")
    assert not reveal_active()
    from strategy.rules.reveal import get_variant

    with pytest.raises(UnsupportedVariant):
        get_variant("reveal_variant")


@pytest.mark.parametrize("ruleset", ["practice_nlhe", "uncertainty_shortdeck", "uncertainty_plo4"])
def test_fallback_is_legal_and_fast_on_practice_variants(ruleset):
    from arena.domain.models import TournamentConfig
    from strategy.rehearsal.runner import smoke_tournaments

    rep = smoke_tournaments(TournamentConfig(ruleset_id=ruleset, initial_player_count=6, hand_cap=60), tables=2)
    assert rep["ok"], rep
    assert rep["arena_fallbacks"] == 0 and rep["internal_fallbacks"] == 0 and rep["decisions"] > 100


def test_fallback_card_blind_mode_stays_legal(tmp_path):
    """A variant the card layer cannot read (here: straights start at the five lowest ranks) plays card-blind."""
    rep = run_py(
        "import json\n"
        "from arena.domain.models import TournamentConfig\n"
        "from strategy.policy.generic import _Game\n"
        "from strategy.rehearsal.runner import smoke_tournaments\n"
        "g = _Game('reveal_variant')\n"
        "sm = smoke_tournaments(TournamentConfig(ruleset_id='reveal_variant', initial_player_count=4, hand_cap=40),"
        " tables=2)\n"
        "print(json.dumps({'blind': g.caps is None, 'smoke': sm}))\n",
        reveal_dir(tmp_path, "shortdeck-trips-6max-turbo", lowest_straight="6789T"))
    assert rep["blind"] and rep["smoke"]["ok"], rep


def _imports(path: Path) -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.add(node.module)
    return out


def test_fallback_files_import_only_public_contracts_and_each_other():
    staged = {f[:-3].replace("/", ".").removesuffix(".__init__") for f in FALLBACK_FILES}
    for rel in FALLBACK_FILES:
        for name in _imports(ROOT / rel):
            top = name.split(".")[0]
            if top in sys.stdlib_module_names or top == "__future__":
                continue
            if top == "strategy":
                assert name in staged, (rel, name)
            else:
                assert name in ALLOWED_ARENA, (rel, name)


@pytest.mark.parametrize("name", PLAYABLE)
def test_fallback_import_closure_is_staged_and_has_no_numpy(tmp_path, name):
    loaded = run_py(
        "import json, sys\n"
        "from arena.domain.models import TournamentConfig\n"
        "from strategy.rehearsal.runner import smoke_tournaments\n"
        "before = set(sys.modules)\n"
        "for r in ('practice_nlhe', 'uncertainty_shortdeck', 'uncertainty_plo4', 'reveal_variant'):\n"
        "    assert smoke_tournaments(TournamentConfig(ruleset_id=r, initial_player_count=4, hand_cap=15),"
        " tables=1)['ok'], r\n"
        "from strategy.policy import generic\n"
        "mods = sorted(m for m in sys.modules if m.split('.')[0] in ('strategy', 'numpy', 'numba'))\n"
        "print(json.dumps(mods))\n",
        reveal_dir(tmp_path, name))
    staged = {f[:-3].replace("/", ".").removesuffix(".__init__") for f in FALLBACK_FILES}
    bot_side = {m for m in loaded if m.startswith(("strategy.policy", "strategy.features", "strategy.rules.reveal",
                                                   "strategy.rules.variants", "numpy", "numba"))}
    bot_side -= {"strategy.policy.runtime", "strategy.policy.baseline", "strategy.policy.params",
                 "strategy.policy.spot", "strategy.features.equity", "strategy.features.state"}
    assert bot_side <= staged, bot_side - staged
    assert not any(m.split(".")[0] in ("numpy", "numba") for m in bot_side)


def test_fallback_bundle_carries_the_variant_but_no_tables(tmp_path):
    reveal = reveal_dir(tmp_path, "plo5-8max")
    (reveal / "tables").mkdir()
    (reveal / "tables" / "variant_hash").write_text("x\n")
    fb = build_bundle(tmp_path / "fb", kind="fallback", reveal=reveal, commit="c")
    base = build_bundle(tmp_path / "base", None, reveal=reveal, commit="c")
    mf, mb = (json.loads((p / "manifest.json").read_text()) for p in (fb, base))
    assert mf["kind"] == "fallback" and mf["vector_hash"] is None and not (fb / "params.json").exists()
    assert set(mf["files"]) == {*FALLBACK_FILES, "bot.py", "strategy/reveal/variant.json"}
    assert "strategy/reveal/tables/variant_hash" in mb["files"] and "strategy/reveal/variant.json" in mb["files"]
    assert mf["reveal_variant_hash"] == mb["reveal_variant_hash"]
    again = build_bundle(tmp_path / "fb2", kind="fallback", reveal=reveal, commit="c")
    assert json.loads((again / "manifest.json").read_text())["bundle_hash"] == mf["bundle_hash"]
    assert sum(p.stat().st_size for p in fb.rglob("*") if p.is_file()) < 1_000_000


def test_packaged_bot_reads_its_own_variant_and_rejects_foreign_tables(tmp_path):
    reveal = reveal_dir(tmp_path, "shortdeck-trips-6max-turbo")
    (reveal / "tables").mkdir()
    (reveal / "tables" / "variant_hash").write_text("0" * 64 + "\n")
    bundle = build_bundle(tmp_path / "bot", None, reveal=reveal, commit="c")
    code = ("import json, sys\n"
            f"sys.path.insert(0, {str(bundle)!r})\n"
            "from strategy.rules.reveal import reveal_dir, reveal_variant\n"
            "from strategy.policy.spot import variant_assets\n"
            "d = str(reveal_dir())\n"
            "try:\n"
            "    variant_assets('reveal_variant'); err = None\n"
            "except Exception as exc:\n"
            "    err = type(exc).__name__\n"
            "print(json.dumps({'dir': d, 'ranks': reveal_variant().ranks, 'err': err}))\n")
    env = {k: v for k, v in os.environ.items() if k != "ARENA_REVEAL_VARIANT"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=bundle, env=env,
                         timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    rep = json.loads(out.stdout.strip().splitlines()[-1])
    assert rep["dir"].startswith(str(bundle)) and rep["ranks"] == "6789TJQKA"
    assert rep["err"] == "UnsupportedVariant"


def test_rehearsal_smoke_end_to_end(tmp_path):
    """The whole pipeline at the smoke budget; the deployed gate runs too when the sandbox is available."""
    gate = shutil.which("unshare") is not None
    out = tmp_path / "rh"
    cmd = [sys.executable, "-m", "strategy.rehearsal", "run", str(HELDBACK / "shortdeck-trips-6max-turbo.json"),
           "--budget", "smoke", "--out", str(out), "--jobs", "2"] + ([] if gate else ["--no-gate"])
    env = {k: v for k, v in os.environ.items() if k != "ARENA_REVEAL_VARIANT"}
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=env, timeout=900)
    assert proc.returncode == 0, proc.stderr[-3000:]
    rep = json.loads((out / "report.json").read_text())
    steps = {s["name"]: s for s in rep["steps"]}
    assert list(steps) == ["load", "conformance", "fallback", "gate_fallback", "submit_early", "tables", "search",
                           "package", "validation", "select", "gate", "limits", "submit"]
    assert all(s["status"] in ("ok", "skipped") for s in steps.values()), {k: (s["status"], s["error"])
                                                                         for k, s in steps.items()}
    assert all(s["wall_s"] >= 0 for s in steps.values()) and rep["first_artifact_s"] is not None
    assert steps["tables"]["detail"]["loaded"] == {"preflop": True, "postflop": True}
    assert steps["tables"]["detail"]["push_fold"]["feasible"] is False
    assert steps["limits"]["detail"]["fallback"]["heavy_modules"] == []
    sub = json.loads((out / "submission" / "final" / "submission.json").read_text())
    assert sub["rollback"]["role"] == "fallback" and (out / "submission" / "final" / sub["submission"]["file"]).exists()
    if gate:
        early = json.loads((out / "submission" / "early" / "submission.json").read_text())
        assert early["submission"]["role"] == "fallback" and early["gate"]["submission"] is True
        assert rep["early_submission_s"] is not None
    assert rep["chosen"] in ("tuned", "transferred", "fallback")
    assert (out / "report.md").read_text().startswith("# Rehearsal: shortdeck-trips-6max-turbo")


def test_rehearsal_rejects_antes_loudly(tmp_path):
    from strategy.rehearsal.runner import Rehearsal

    rep = Rehearsal(HELDBACK / "nlhe-ante-6max.json", tmp_path / "rh", budget="smoke", log=lambda s: None).run()
    assert rep["chosen"] is None and rep["failure"].startswith("load: ValueError: ante 50")
    assert [s["status"] for s in rep["steps"]] == ["fatal"]


def test_announcement_equal_to_a_practice_variant_uses_its_ruleset(tmp_path):
    from strategy.rehearsal.runner import Rehearsal, Step

    ann = json.loads((HELDBACK / "nlhe-ante-6max.json").read_text())
    del ann["structure"]["ante"]
    ann["name"] = "nlhe-6max"
    spec = tmp_path / "nlhe.json"
    spec.write_text(json.dumps(ann))
    r = Rehearsal(spec, tmp_path / "rh", budget="smoke", log=lambda s: None)
    step = Step("load")
    r.step_load(step)
    assert step.detail["known_variant"] == "practice_nlhe" and r.ctx["ruleset_id"] == "practice_nlhe"
    assert "reveal" not in r.ctx and r.ctx["structure"]["initial_player_count"] == 6
