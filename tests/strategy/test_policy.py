"""Baseline policy, deployed runtime, fallback and packaging (strategy.policy)."""

from __future__ import annotations

import ast
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from arena.agents.sdk import bot_info, decode_state, encode_state
from arena.agents.worker import InProcessSeat, play_table
from arena.domain.models import RULESET_IDS, TournamentConfig
from strategy.policy.baseline import SPACE
from strategy.policy.fallback import fallback_action, is_legal, legalize
from strategy.policy.package import RUNTIME_FILES, build_bundle
from strategy.policy.params import vector_hash
from strategy.policy.runtime import RuntimeBot
from strategy.search.evaluate import Task, _init, seat_agents

ROOT = Path(__file__).resolve().parents[2]
ALLOWED_ARENA = {"arena.agents.sdk", "arena.domain.observations"}
OPPONENTS = ("baseline:legal_random", "baseline:calling_station", "baseline:check_fold", "baseline:legal_random",
             "baseline:calling_station")


def test_parameter_space():
    assert 50 <= len(SPACE) <= 200
    d = SPACE.defaults()
    for p in SPACE.params:
        assert p.lo <= p.default <= p.hi, p.name
        assert p.doc and p.group
    assert SPACE.from_unit(SPACE.to_unit(d)) == pytest.approx(d)
    assert SPACE.complete({"open_thr_E": 9.0})["open_thr_E"] == SPACE["open_thr_E"].hi
    with pytest.raises(KeyError):
        SPACE.complete({"no_such_param": 1.0})
    assert vector_hash(d) == vector_hash(dict(reversed(list(d.items()))))
    assert vector_hash(d) != vector_hash(d | {"jam_bb": d["jam_bb"] + 0.5})


def test_promoted_vectors_are_complete_and_in_bounds():
    for path in sorted((ROOT / "strategy/policy/vectors").glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert set(raw["params"]) == {p.name for p in SPACE.params}, path.name
        assert SPACE.complete(raw["params"]) == raw["params"], path.name
        assert vector_hash(raw["params"]) == raw["vector_hash"], path.name


class CheckingBot(RuntimeBot):
    """The runtime plus, at every real decision, a legalization fuzz of random intents."""

    def __init__(self, params=None, seed=0):
        super().__init__(params)
        self.rng = random.Random(seed)
        self.checked = 0

    def decide(self, observation, state):
        legal = observation.legal
        assert is_legal(fallback_action(legal), legal)
        hi = (legal.all_in_raise_to or 0) + 50
        for kind in ("fold", "check", "call", "raise", "allin"):
            act = legalize(kind, self.rng.randint(-10, hi), legal, legal.all_in_raise_to or 0)
            assert is_legal(act, legal) and legal.allows(act), (kind, act, legal)
            self.checked += 1
        return super().decide(observation, state)


def _play(ruleset_id, params, seed, *, players=6, hand_cap=40, bot_cls=CheckingBot):
    config = TournamentConfig(ruleset_id=ruleset_id, initial_player_count=players, hand_cap=hand_cap)
    _init(config.model_dump_json())
    task = Task(seed, 1, OPPONENTS[:players - 1], ("x",) * (players - 1))
    made = {}

    def candidate(seat, info):
        made["bot"] = bot = bot_cls(params) if bot_cls is not CheckingBot else CheckingBot(params, hash(seed))
        return InProcessSeat(seat, bot, bot_info(bot), info, "test")

    agents, cs, table = seat_agents(config, task, candidate)
    play_table(table, agents)
    return agents[cs], made["bot"], table


VECTORS = {
    "defaults": None,
    "all_low": SPACE.from_unit([0.0] * len(SPACE)),
    "all_high": SPACE.from_unit([1.0] * len(SPACE)),
    "random": SPACE.from_unit([random.Random(7).random() for _ in range(len(SPACE))]),
}


@pytest.mark.parametrize("ruleset_id", sorted(RULESET_IDS))
@pytest.mark.parametrize("vector", sorted(VECTORS))
def test_runtime_is_always_legal(ruleset_id, vector):
    decisions = checked = 0
    for k in range(12):
        agent, bot, _ = _play(ruleset_id, VECTORS[vector], f"legal-{ruleset_id}-{vector}-{k}")
        assert sum(agent.stats.fallbacks.values()) == 0, agent.stats.fallbacks
        assert decode_state(agent.committed)["fb"] == 0
        assert agent.stats.max_state_bytes < 2048
        decisions += agent.stats.decisions
        checked += bot.checked
        if decisions >= 40:
            break
    assert decisions >= 40
    assert checked >= 5 * decisions


class RecordingBot(RuntimeBot):
    def __init__(self, params=None):
        super().__init__(params)
        self.log = []

    def decide(self, observation, state):
        action, new = super().decide(observation, state)
        self.log.append((observation, encode_state(state), action, encode_state(new)))
        return action, new


@pytest.mark.parametrize("ruleset_id", ["practice_nlhe", "uncertainty_plo4"])
def test_restore_from_committed_state_replays_exactly(ruleset_id):
    """A fresh bot (a restarted worker) given the committed state decides exactly as the live one."""
    _, bot, _ = _play(ruleset_id, None, f"restore-{ruleset_id}", bot_cls=RecordingBot)
    assert len(bot.log) > 20
    for obs, state, action, new in bot.log[::3]:
        fresh = RuntimeBot()
        a2, n2 = fresh.decide(obs, decode_state(state))
        assert (a2, encode_state(n2)) == (action, new)


def test_same_task_same_result():
    a = _play("practice_nlhe", None, "crn")[2].points()
    b = _play("practice_nlhe", None, "crn")[2].points()
    assert a == b


def _module_level(tree: ast.AST):
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            yield node
            yield from _module_level(node)


def _imports(path: Path) -> set[str]:
    """Imports that run on module load; lazy ones are covered by the closure test."""
    out = set()
    for node in _module_level(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.add(node.module)
    return out


def test_runtime_files_import_only_public_contracts_and_each_other():
    staged = {f[:-3].replace("/", ".").removesuffix(".__init__") for f in RUNTIME_FILES}
    for rel in RUNTIME_FILES:
        for name in _imports(ROOT / rel):
            top = name.split(".")[0]
            if top in sys.stdlib_module_names or top == "__future__":
                continue
            if top == "strategy":
                assert name in staged, (rel, name)
            else:
                assert name in ALLOWED_ARENA, (rel, name)


def test_runtime_import_closure_is_staged():
    """Every strategy module the deployed path loads while playing is in the bundle."""
    probe = (
        "import sys, json\n"
        "from arena.agents.sdk import bot_info\n"
        "from arena.agents.worker import InProcessSeat, play_table\n"
        "from arena.benchmark.runner import Assignment, table_info\n"
        "from arena.domain.models import TournamentConfig\n"
        "from arena.table.service import seeded_table\n"
        "from strategy.policy.runtime import RuntimeBot\n"
        "for r in ('practice_nlhe', 'uncertainty_shortdeck', 'uncertainty_plo4'):\n"
        "    c = TournamentConfig(ruleset_id=r, initial_player_count=4, hand_cap=15)\n"
        "    a = Assignment(0, 0, 0, 'closure', {s: i for i, s in enumerate(c.resolved_seats)})\n"
        "    agents = {}\n"
        "    for s in a.seats:\n"
        "        b = RuntimeBot()\n"
        "        agents[s] = InProcessSeat(s, b, bot_info(b), table_info(c, a, s, c.bot_budget_ns), 't')\n"
        "        agents[s].start()\n"
        "    play_table(seeded_table(c, 'closure', table_id=a.table_id), agents)\n"
        "    assert not any(ag.stats.fallbacks for ag in agents.values())\n"
        "print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in ('strategy', 'numpy', 'numba'))))\n"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, cwd=ROOT, check=True)
    loaded = set(json.loads(out.stdout))
    staged = {f[:-3].replace("/", ".").removesuffix(".__init__") for f in RUNTIME_FILES}
    assert loaded <= staged, loaded - staged


def test_bundle_is_deterministic(tmp_path):
    a = build_bundle(tmp_path / "a", None, commit="x")
    b = build_bundle(tmp_path / "b", None, commit="x")
    ma, mb = (json.loads((p / "manifest.json").read_text()) for p in (a, b))
    assert ma["bundle_hash"] == mb["bundle_hash"]
    assert ma["vector_hash"] == vector_hash(SPACE.defaults())
    for rel in RUNTIME_FILES:
        assert (a / rel).read_bytes() == (ROOT / rel).read_bytes()
    assert any(k.startswith("strategy/tables/data/preflop-") for k in ma["files"])


@pytest.mark.skipif(shutil.which("unshare") is None, reason="sandboxed workers need util-linux unshare")
def test_bundle_plays_in_the_real_sandbox(tmp_path):
    from strategy.policy.gate import run_gate

    bundle = build_bundle(tmp_path / "bot", SPACE.from_unit([0.5] * len(SPACE)))
    rep = run_gate(bundle, min_decisions=60, players=6, hand_cap=25, log=lambda s: None, p99_limit_us=10_000,
                   p999_limit_us=10_000, cost_tables=4)
    assert rep["decisions"] >= 60
    assert rep["internal_fallbacks"] == 0 and rep["restarts"] == 0
    assert set(rep["fallbacks"]) <= {"timeout"}  # shared CI machines may miss the 1 ms deadline
    assert rep["max_state_bytes"] < 2048
    first = rep["attempts"][0]
    assert first["control"]["tables"] == first["candidate"]["tables"] and first["control"]["decisions"] > 0
    assert rep["control"]["policy"] == "cpu_burner" and rep["control"]["calibration_decisions"] > 0
    assert sum(rep["timeout_classes"].values()) == rep["fallbacks"].get("timeout", 0)
    assert rep["selection"]["action"] in ("use", "fail", "invalid") and len(rep["attempts"]) <= 3
    assert [a["multiplier"] for a in rep["points_cost"]["arms"]] == [1, 10]
    assert rep["points_cost"]["tables"] == 4


def test_gate_bot_caused_ceiling_control_test_headroom_and_reruns(monkeypatch):
    from strategy.policy import gate

    class Seat:
        def __init__(self, timeouts, wall=100_000):
            self.stats = type("S", (), {"decisions": 100, "worker_wall_ns": [wall] * 100,
                                        "e2e_ns": [300_000] * 100, "fallbacks": {"timeout": len(timeouts)} if timeouts
                                        else {}, "restarts": [], "late_discarded": 0, "max_state_bytes": 10})()
            self.records = timeouts

    slow = {"outcome": "late_reply", "worker_parse_ns": 0, "worker_wall_ns": 2_000_000,
            "replay": {"runs": [{"fallback": None, "worker_wall_ns": 900_000}] * 3}}
    stall = {"outcome": "late_reply", "worker_parse_ns": 0, "worker_wall_ns": 50_000}
    computing = {"outcome": "late_reply", "worker_parse_ns": 0, "worker_wall_ns": 1_200_000,
                 "worker_cpu_ns": 1_190_000}
    script = []

    def attempt(*_a, **_k):
        # 500 candidate and 500 control tables of 100 decisions; the scripted
        # records all land on the first candidate table.
        cand, ctrl = gate._Side(), gate._Side()
        records, wall = script.pop(0)
        for i in range(500):
            s = Seat(records if i == 0 else [], wall)
            cand.add("t", s, s.records)
            c = Seat([])
            ctrl.add("t", c, [])
        return cand, ctrl, 0

    def run(**kw):
        return gate.run_gate(Path("x"), min_decisions=100, log=lambda s: None, **kw)

    monkeypatch.setattr(gate, "_attempt", attempt)
    monkeypatch.setattr(gate, "load_population", lambda: None)
    monkeypatch.setattr(gate, "calibrate", lambda *_a, **_k: [300_000] * 9 + [5_000_000])
    monkeypatch.setattr(gate, "matched_iterations", lambda walls, **_k: {"iterations": len(walls) * 100,
                                                                          "target_median_ns": 300_000})
    # 11 bot-caused in 50,000 decisions is above the 0.02% ceiling: final, no rerun.
    script[:] = [([slow] * 11, 100_000)]
    rep = run()
    assert not rep["passed"] and rep["selection"]["action"] == "fail" and len(rep["attempts"]) == 1
    assert rep["timeout_classes"]["bot_caused"] == 11 and not rep["checks"]["bot_caused_ceiling"]
    assert rep["control"]["policy"] == "cpu_burner" and rep["control"]["iterations"] == 1000
    # 8 CPU-bound against none in the control at equal exposure: p = 0.5**8 < 0.01, a
    # bot excess, rerun; the rerun passes and is used.
    script[:] = [([computing] * 8, 100_000), ([computing] * 2, 100_000)]
    rep = run(control_iterations=2_000)
    assert rep["passed"] and [a["verdict"] for a in rep["attempts"]] == ["bot_excess", "pass"]
    first = rep["attempts"][0]
    assert first["checks"]["bot_caused_ceiling"] and not first["checks"]["bot_caused_within_control"]
    assert rep["timeout_rules"] == {"cpu_bound": 2} and rep["control"]["iterations"] == 2_000
    # A bot excess in every attempt is a fail once reruns are exhausted.
    script[:] = [([computing] * 8, 100_000)] * 3
    assert run(control_iterations=2_000)["selection"]["action"] == "fail"
    # Stall excess reruns.
    script[:] = [([stall] * 10, 100_000), ([stall] * 10, 100_000), ([], 100_000)]
    rep = run(control_iterations=2_000)
    assert rep["passed"] and [a["verdict"] for a in rep["attempts"]] == ["stall_excess", "stall_excess", "pass"]
    assert rep["selection"]["used"] == 2
    # Headroom is a hard gate: decide p99 above 400 us fails with no timeouts at all.
    script[:] = [([], 450_000)]
    rep = run(control_iterations=2_000)
    assert not rep["passed"] and not rep["checks"]["decide_p99_headroom"] and len(rep["attempts"]) == 1
    assert rep["p99_limit_us"] == 400 and rep["p999_limit_us"] == 600
