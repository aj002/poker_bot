"""Deadline, ordering and commit rules with a fake clock (spec 6.3 and 7).

The real :class:`WorkerSeat` logic runs against a scripted channel that behaves like
a worker (including the commit protocol) but whose replies arrive at chosen fake
times. Nothing here sleeps or depends on the machine's speed.
"""

from __future__ import annotations

import json

import pytest

from arena.agents.protocol import HEADER, FrameError
from arena.agents.worker import ChannelClosed, FakeClock, ReceivedFrame, WorkerSeat, on_time, play_table
from arena.agents.worker import Artifact
from arena.domain.observations import Observation
from arena.table.projections import observation
from arena.table.service import seeded_table
from tests.agents.conftest import info_for, small_config

BUDGET = 1_000_000
ART = Artifact("scripted", entry=None, root=None, sha256="0" * 64)  # type: ignore[arg-type]


class ScriptedChannel:
    """Acts like a worker: answers load, keeps cached/pending state like the real one,
    and replies to the k-th decide request according to ``plan(k)``.

    ``plan(k)`` returns a delay in ns, ``"never"``, ``"crash"``, ``("error", code)``,
    ``("action", dict)`` (on time, immediate) or a ``FrameError`` to raise on recv.
    """

    def __init__(self, world: World) -> None:
        self.world = world
        self.clock = world.clock
        self.queue: list[tuple[int, dict, bytes]] = []
        self.sent: list[tuple[dict, bytes]] = []
        self.cached = b"null"
        self.pending: tuple[int, bytes] | None = None
        self.closed = False
        self.killed = False
        self.raise_on_recv: Exception | None = None

    def send(self, frame: bytes) -> None:
        if self.closed:
            raise ChannelClosed("closed")
        meta_len, _ = HEADER.unpack_from(frame)
        meta = json.loads(frame[HEADER.size:HEADER.size + meta_len])
        blob = frame[HEADER.size + meta_len:]
        self.sent.append((meta, blob))
        now = self.clock.now_ns()
        if meta["t"] == "load":
            if meta["restore"]:
                self.cached, state = blob, b""
            else:
                self.cached = state = json.dumps({"n": 0}).encode()
            self.queue.append((now, {"t": "loaded", "info": self.world.info}, state))
        elif meta["t"] == "decide":
            if self.pending is not None and self.pending[0] == meta["commit"]:
                self.cached = self.pending[1]
            self.pending = None
            self.world.k += 1
            k = self.world.k
            obs = Observation.model_validate_json(blob)
            self.world.seen_states.append(json.loads(self.cached))
            plan = self.world.plan(k)
            n = json.loads(self.cached)["n"] if self.world.info["state_mode"] == "explicit" else 0
            state = json.dumps({"n": n + 1}).encode() if self.world.info["state_mode"] == "explicit" else b""
            action = {"type": "check"} if obs.legal.check else {"type": "fold"}
            reply = {"t": "action", "seq": meta["seq"], "action": action, "wall_ns": 10, "cpu_ns": 9, "parse_ns": 5,
                     "parse_cpu_ns": 4}
            if plan == "never":
                return
            if plan == "crash":
                self.closed = True
                return
            if isinstance(plan, FrameError):
                self.raise_on_recv = plan
                return
            if isinstance(plan, tuple) and plan[0] == "error":
                self.queue.append((now, {"t": "error", "seq": meta["seq"], "code": plan[1], "detail": "x"}, b""))
                return
            if isinstance(plan, tuple) and plan[0] == "action":
                reply["action"] = plan[1]
                plan = 0
            if self.world.info["state_mode"] == "explicit":
                self.pending = (meta["seq"], state)
            self.queue.append((now + plan, reply, state))

    def recv(self, deadline_ns, spin_ns: int = 0):
        if self.raise_on_recv is not None:
            exc, self.raise_on_recv = self.raise_on_recv, None
            raise exc
        if self.queue and (deadline_ns is None or self.queue[0][0] <= deadline_ns):
            arrival, meta, blob = self.queue.pop(0)
            self.clock.set(arrival + self.world.processing_lag_ns)
            return ReceivedFrame(meta, blob, arrival)
        if self.closed:
            raise ChannelClosed("worker exited")
        assert deadline_ns is not None, "would block forever"
        self.clock.set(deadline_ns + self.world.wake_lag_ns)
        return None

    def kill(self) -> None:
        self.killed = True
        self.closed = True

    def log_excerpt(self) -> str:
        return ""


class World:
    def __init__(self, plan, *, mode: str = "explicit", processing_lag_ns: int = 0, wake_lag_ns: int = 0) -> None:
        self.clock = FakeClock()
        self.plan = plan
        self.k = 0
        self.info = {"name": "scripted", "rulesets": ["practice_nlhe"], "state_mode": mode}
        self.processing_lag_ns = processing_lag_ns
        self.wake_lag_ns = wake_lag_ns
        self.channels: list[ScriptedChannel] = []
        self.seen_states: list[dict] = []

    def spawn(self) -> ScriptedChannel:
        ch = ScriptedChannel(self)
        self.channels.append(ch)
        return ch


def make_seat(world: World, **kw) -> tuple[WorkerSeat, object]:
    config = small_config(3)
    table = seeded_table(config, "deadline", table_id="t_test")
    seat = WorkerSeat(table.actor_seat, ART, info_for(config, table.actor_seat), world.spawn, clock=world.clock,
                      budget_ns=BUDGET, **kw)
    seat.start()
    return seat, table


def decide_once(seat: WorkerSeat, table) -> object:
    seat.prepare()
    return seat.decide(observation(table, table.actor_seat))


# --------------------------------------------------------------- the boundary


def test_on_time_rule_is_inclusive():
    assert on_time(999, 1000) and on_time(1000, 1000) and not on_time(1001, 1000)


@pytest.mark.parametrize("delay,expect", [(BUDGET - 1, None), (BUDGET, None), (BUDGET + 1, "timeout")])
def test_boundary_just_before_exactly_at_just_after(delay, expect):
    world = World(lambda k: delay)
    seat, table = make_seat(world)
    d = decide_once(seat, table)
    assert d.fallback == expect
    if expect is None:
        assert d.e2e_ns == delay
        assert d.action in ({"type": "check"}, {"type": "fold"})
        seat.accept()
        assert json.loads(seat.committed) == {"n": 1}
    else:
        seat.reject()
        assert json.loads(seat.committed) == {"n": 0}
        assert seat.stale_seq == 1


def test_on_time_reply_processed_after_the_deadline_still_wins():
    # The reply is completely received exactly at the deadline, but the parent only
    # gets to it 5 ms later (slow queue processing). Received time decides.
    world = World(lambda k: BUDGET, processing_lag_ns=5_000_000)
    seat, table = make_seat(world)
    start = world.clock.now_ns()
    d = decide_once(seat, table)
    assert world.clock.now_ns() - start > BUDGET
    assert d.fallback is None and d.e2e_ns == BUDGET


def test_watchdog_waking_late_does_not_accept_a_late_reply():
    # The reply arrives 1 ns late; the watchdog wakes 3 ms after the deadline and the
    # reply is already queued. It is still late.
    world = World(lambda k: BUDGET + 1 if k == 1 else 0, wake_lag_ns=3_000_000)
    seat, table = make_seat(world)
    d = decide_once(seat, table)
    assert d.fallback == "timeout"


def test_late_reply_never_applies_to_the_next_decision():
    world = World(lambda k: BUDGET + 500_000 if k == 1 else 0)
    seat, table = make_seat(world)
    first = decide_once(seat, table)
    assert first.fallback == "timeout"
    seat.reject()
    table.apply_fallback(table.actor_seat, "timeout")
    # Advance to the same seat's next decision.
    while table.actor_seat != seat.seat:
        table.apply_fallback(table.actor_seat, "test")
    second = decide_once(seat, table)
    assert second.fallback is None
    assert seat.stats.late_discarded == 1
    # The late reply's state (n=1) was never committed; the next decision saw n=0.
    assert world.seen_states == [{"n": 0}, {"n": 0}]
    assert world.channels[0].sent[-1][0]["commit"] is None
    assert seat.stats.restarts == []


def test_unresponsive_worker_is_killed_and_restored_from_committed_state():
    world = World(lambda k: "never" if k == 2 else 0)
    seat, table = make_seat(world, stale_grace_ns=20_000_000)
    d1 = decide_once(seat, table)
    assert d1.fallback is None
    seat.accept()
    assert json.loads(seat.committed) == {"n": 1}
    d2 = decide_once(seat, table)
    assert d2.fallback == "timeout"
    seat.reject()
    d3 = decide_once(seat, table)
    assert d3.fallback is None
    assert world.channels[0].killed
    assert len(world.channels) == 2
    load_meta, load_blob = world.channels[1].sent[0]
    assert load_meta["t"] == "load" and load_meta["restore"] is True and json.loads(load_blob) == {"n": 1}
    assert [(r.cause, r.restored) for r in seat.stats.restarts] == [("unresponsive", True)]
    seat.accept()
    assert json.loads(seat.committed) == {"n": 2}


def test_crash_falls_back_once_and_restores():
    world = World(lambda k: "crash" if k == 1 else 0)
    seat, table = make_seat(world)
    d = decide_once(seat, table)
    assert d.fallback == "worker_crash"
    seat.reject()
    d2 = decide_once(seat, table)
    assert d2.fallback is None and len(world.channels) == 2
    assert seat.stats.restarts[0].cause == "worker_crash" and seat.stats.restarts[0].restored


@pytest.mark.parametrize("code", ["bot_exception", "state_too_large", "bot_memory_error"])
def test_bot_error_codes_pass_through(code):
    world = World(lambda k: ("error", code))
    seat, table = make_seat(world)
    assert decide_once(seat, table).fallback == code
    assert seat.channel is world.channels[0] and seat.stats.restarts == []


def test_unknown_error_code_is_normalised():
    world = World(lambda k: ("error", "made_up"))
    seat, table = make_seat(world)
    assert decide_once(seat, table).fallback == "bot_exception"


def test_oversized_frame_replaces_worker():
    world = World(lambda k: FrameError("oversized_output", "too big") if k == 1 else 0)
    seat, table = make_seat(world)
    assert decide_once(seat, table).fallback == "oversized_output"
    assert world.channels[0].killed
    assert decide_once(seat, table).fallback is None and len(world.channels) == 2


def test_instance_mode_restart_is_marked_not_restorable():
    world = World(lambda k: "crash" if k == 1 else 0, mode="instance")
    seat, table = make_seat(world)
    decide_once(seat, table)
    seat.reject()
    decide_once(seat, table)
    assert seat.stats.restarts[0].restored is False and not seat.stats.restorable


def test_restart_limit_leads_to_worker_unavailable():
    world = World(lambda k: "crash")
    seat, table = make_seat(world, max_restarts=2)
    reasons = []
    for _ in range(5):
        reasons.append(decide_once(seat, table).fallback)
        seat.reject()
    assert reasons[:3] == ["worker_crash"] * 3 and reasons[3:] == ["worker_unavailable"] * 2


# ------------------------------------------------------------ through a table


def test_illegal_action_and_state_commit_through_play_table():
    # Decisions alternate: legal check/fold, then an illegal raise. Only accepted
    # actions commit state, and each illegal one becomes exactly one fallback.
    world = World(lambda k: ("action", {"type": "raise_to", "amount": 1}) if k % 2 == 0 else 0)
    config = small_config(3)
    seats = {}
    table = seeded_table(config, "deadline", table_id="t_test")
    for s in config.resolved_seats:
        w = world if s == table.actor_seat else World(lambda k: 0)
        w.clock = world.clock
        seats[s] = WorkerSeat(s, ART, info_for(config, s), w.spawn, clock=world.clock, budget_ns=BUDGET)
        seats[s].start()
    hero = table.actor_seat
    play_table(table, seats)
    st = seats[hero].stats
    assert st.decisions == world.k
    assert world.k >= 2
    assert st.fallbacks == {"illegal_action": world.k // 2}
    assert json.loads(seats[hero].committed) == {"n": world.k - world.k // 2}
    fallbacks = [e for e in table.log if e.type == "ActionFallback" and e.data["seat"] == hero]
    assert len(fallbacks) == world.k // 2
    assert all(e.data["reason"] == "illegal_action" for e in fallbacks)
    assert table.complete


def test_every_decision_consumed_exactly_once_under_mixed_faults():
    plans = {0: lambda k: [0, BUDGET + 1, "never", 0, ("error", "bot_exception")][k % 5],
             1: lambda k: ["crash", 0, 0][k % 3],
             2: lambda k: 0}
    config = small_config(3, hand_cap=4)
    clock = FakeClock()
    seats = {}
    for s in config.resolved_seats:
        w = World(plans[s])
        w.clock = clock
        seats[s] = WorkerSeat(s, ART, info_for(config, s), w.spawn, clock=clock, budget_ns=BUDGET)
        seats[s].start()
    table = seeded_table(config, "mixed", table_id="t_test")
    play_table(table, seats)
    assert table.complete
    requested = [e for e in table.log if e.type == "DecisionRequested"]
    applied = [e for e in table.log if e.type == "ActionApplied" and not e.data.get("auto")]
    assert len(requested) == len(applied) == sum(a.stats.decisions for a in seats.values())
    fallbacks = sum(1 for e in table.log if e.type == "ActionFallback")
    assert fallbacks == sum(sum(a.stats.fallbacks.values()) for a in seats.values())
