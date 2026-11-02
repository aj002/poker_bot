"""Timeout records, timeout replay, the parent's GC window and interleaved control tables."""

from __future__ import annotations

import gc
import json

from arena.agents.baselines import BASELINES
from arena.agents.worker import ReceivedFrame, play_table
from arena.benchmark.capacity import safe_jobs
from arena.benchmark.runner import Entrant, TableRunner, interleave_controls, run_benchmark, schedule
from tests.agents.conftest import bot, needs_sandbox, small_config
from tests.agents.test_deadline import BUDGET, ScriptedChannel, World, decide_once, make_seat


class LateChannel(ScriptedChannel):
    """Hands over a queued reply even when it arrived after the deadline, as a pipe
    read that finishes late does."""

    def recv(self, deadline_ns, spin_ns: int = 0):
        if self.queue and deadline_ns is not None and self.queue[0][0] > deadline_ns:
            arrival, meta, blob = self.queue.pop(0)
            self.clock.set(arrival)
            return ReceivedFrame(meta, blob, arrival)
        return super().recv(deadline_ns, spin_ns)


class GcProbeChannel(ScriptedChannel):
    def __init__(self, world: World) -> None:
        super().__init__(world)
        self.gc_during: list[bool] = []

    def send(self, frame: bytes) -> None:
        self.gc_during.append(gc.isenabled())
        super().send(frame)

    def recv(self, deadline_ns, spin_ns: int = 0):
        self.gc_during.append(gc.isenabled())
        return super().recv(deadline_ns, spin_ns)


def _spawn_with(world: World, cls) -> None:
    def spawn():
        ch = cls(world)
        world.channels.append(ch)
        return ch
    world.spawn = spawn  # type: ignore[method-assign]


def _next_turn(seat, table) -> None:
    while table.actor_seat != seat.seat:
        table.apply_fallback(table.actor_seat, "test")


def test_late_reply_is_timed_when_it_is_drained():
    world = World(lambda k: BUDGET + 400_000 if k == 1 else 0)
    seat, table = make_seat(world)
    assert decide_once(seat, table).fallback == "timeout"
    seat.reject()
    [rec] = seat.stats.timeouts
    assert rec.outcome == "unsettled" and rec.e2e_ns == BUDGET and rec.decision_index == 0
    table.apply_fallback(table.actor_seat, "timeout")
    _next_turn(seat, table)
    assert decide_once(seat, table).fallback is None
    assert rec.as_dict() == {"decision_index": 0, "seq": 1, "budget_ns": BUDGET, "e2e_ns": BUDGET,
                             "outcome": "late_reply", "late_by_ns": None, "drained_after_ns": 400_000,
                             "reply_kind": "action",
                             "worker_parse_ns": 5, "worker_wall_ns": 10, "worker_cpu_ns": 9,
                             "worker_parse_cpu_ns": 4, "worker_parse_gc": 0, "worker_threads": 1}


def test_reply_read_after_the_deadline_is_recorded_at_once():
    world = World(lambda k: BUDGET + 7)
    _spawn_with(world, LateChannel)
    seat, table = make_seat(world)
    assert decide_once(seat, table).fallback == "timeout"
    [rec] = seat.stats.timeouts
    assert rec.outcome == "late_reply" and rec.late_by_ns == 7 and rec.worker_wall_ns == 10


def test_unresponsive_worker_marks_the_record():
    world = World(lambda k: "never" if k == 1 else 0)
    seat, table = make_seat(world, stale_grace_ns=20_000_000)
    decide_once(seat, table)
    seat.reject()
    decide_once(seat, table)
    assert [r.outcome for r in seat.stats.timeouts] == ["unresponsive"]
    assert seat.stats.restarts[0].cause == "unresponsive"


def test_explicit_state_inputs_are_kept_up_to_the_cap():
    world = World(lambda k: 0 if k == 1 else BUDGET + 1)
    seat, table = make_seat(world, timeout_inputs=1)
    assert decide_once(seat, table).fallback is None
    seat.accept()
    for _ in range(2):
        assert decide_once(seat, table).fallback == "timeout"
        seat.reject()
    first, second = seat.stats.timeouts
    assert first.replayable and not second.replayable
    assert json.loads(first.committed) == {"n": 1}
    requests = [blob for meta, blob in world.channels[0].sent if meta["t"] == "decide"]
    assert first.observation == requests[1]
    assert "observation" not in first.as_dict() and "committed" not in first.as_dict()


def test_instance_state_inputs_are_never_kept():
    world = World(lambda k: BUDGET + 1, mode="instance")
    seat, table = make_seat(world, timeout_inputs=5)
    decide_once(seat, table)
    assert seat.stats.timeouts and not seat.stats.timeouts[0].replayable


def test_parent_gc_is_off_from_dispatch_until_the_reply_is_read():
    world = World(lambda k: 0 if k == 1 else BUDGET + 1)
    _spawn_with(world, GcProbeChannel)
    seat, table = make_seat(world)
    ch = world.channels[0]
    assert gc.isenabled()
    ch.gc_during.clear()
    decide_once(seat, table)
    seat.accept()
    decide_once(seat, table)
    # prepare's wake notice is sent with GC on; the decide request and its reads run with GC off.
    assert ch.gc_during == [True, False, False, True, False, False]
    assert gc.isenabled()
    gc.disable()
    try:
        decide_once(seat, table)
        assert not gc.isenabled()
    finally:
        gc.enable()


def test_play_table_freezes_the_heap_for_the_table_only():
    world = World(lambda k: 0)
    seat, table = make_seat(world)
    others = {s: make_seat(World(lambda k: 0))[0] for s in small_config(3).resolved_seats if s != seat.seat}
    for s, agent in others.items():
        agent.seat = s
    frozen: list[int] = []
    before = gc.get_freeze_count()
    play_table(table, {seat.seat: seat, **others}, on_decision=lambda *_: frozen.append(gc.get_freeze_count()))
    assert table.complete and frozen and min(frozen) > before
    assert gc.get_freeze_count() == 0


def test_controls_are_interleaved_without_touching_the_plan():
    config = small_config(3)
    plan = schedule(config, 3, blocks=3, base_seed="c")
    full = interleave_controls(config, plan, 3, every=4, base_seed="c")
    controls = [a for a in full if a.control]
    assert [a for a in full if not a.control] == plan
    assert [full.index(a) for a in controls] == [0, 5, 10]
    assert all(set(a.seats.values()) == {3} and a.index >= len(plan) for a in controls)
    assert len({a.seed for a in controls}) == 3 and not {a.seed for a in controls} & {a.seed for a in plan}
    assert interleave_controls(config, plan, 3, every=0, base_seed="c") == plan


def test_in_process_control_tables_leave_scores_unchanged(tmp_path):
    config = small_config(3, hand_cap=6)
    entrants = [Entrant(n, BASELINES[n], True) for n in ("calling_station", "legal_random", "check_fold")]
    plain = run_benchmark(config=config, entrants=entrants, blocks=2, base_seed="ctl", isolation="in_process")
    seen: list[tuple[int, int]] = []
    run = run_benchmark(config=config, entrants=entrants, blocks=2, base_seed="ctl", isolation="in_process",
                        control=Entrant("calling_station", BASELINES["calling_station"], True), control_every=3,
                        out_dir=tmp_path / "run", save_histories=True,
                        progress=lambda done, total, r: seen.append((done, total)))
    assert [r.points for r in run.results] == [r.points for r in plain.results]
    untimed = lambda run: [{k: v for k, v in e.items() if k != "timing"}  # noqa: E731
                           for e in run.manifest.summary["entrants"]]
    assert untimed(run) == untimed(plain)
    assert seen == [(k, 6) for k in range(1, 7)]
    assert len(run.control) == 2 and all(r.valid for r in run.control)
    c = run.manifest.summary["control"]
    assert c["tables"] == 2 and c["every_tables"] == 3 and c["decisions"] > 0 and c["timeout_fallbacks"] == 0
    assert c["decisions"] == sum(r.decisions[3] for r in run.control)
    assert len(list((tmp_path / "run" / "tables").glob("*.json"))) == 6
    assert "control" not in plain.manifest.summary


def test_safe_jobs_is_one_sandboxed_job_per_four_physical_cores():
    cap = {"usable_cores": 8}
    assert safe_jobs("sandbox", cap) == 1 and safe_jobs("in_process", cap) == 7
    assert safe_jobs("sandbox", {"usable_cores": 16}) == 3 and safe_jobs("sandbox", {"usable_cores": 2}) == 1


@needs_sandbox
def test_slow_decisions_are_replayed_after_the_table():
    config = small_config(3, hand_cap=4)
    entrants = [Entrant("slow", bot("slow_bot")),
                Entrant("calling_station", BASELINES["calling_station"], True),
                Entrant("check_fold", BASELINES["check_fold"], True)]
    runner = TableRunner(config, entrants, timeout_inputs=2, replay_repeats=2)
    try:
        result = runner.run(schedule(config, 3, 1, "replay")[0])
    finally:
        runner.close()
    records = result.timeouts[0]
    replayed = [r for r in records if "replay" in r]
    assert result.fallbacks[0].get("timeout", 0) == len(records) >= 1
    assert 1 <= len(replayed) <= 2
    for r in replayed:
        assert r["seat"] in result.seats and r["outcome"] in ("late_reply", "unsettled")
        runs = r["replay"]["runs"]
        assert len(runs) == 2 and all(x["fallback"] is None and x["worker_wall_ns"] >= 3_000_000 for x in runs)
    assert all("replay" not in r for e, rs in result.timeouts.items() if e != 0 for r in rs)
