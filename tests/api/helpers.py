"""Shared fixtures for the in-memory API tests: a fake clock and an HTTP harness."""

from __future__ import annotations

import itertools
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from arena.api.app import create_app
from arena.api.clock import FakeClock
from arena.api.manager import TableManager
from arena.api.runs import RunManager

ORIGIN = "http://localhost:5317"
OPERATOR = "operator-test-token"


def auth(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


@dataclass
class Harness:
    client: TestClient
    manager: TableManager
    clock: FakeClock
    runs: RunManager | None = None
    table_id: str = ""
    host: str = ""
    tokens: dict[int, str] = field(default_factory=dict)
    _ids: Iterator[int] = field(default_factory=itertools.count)

    def create(self, seats: list[dict[str, Any]], *, seed: str | None = "api-test", operator: bool = False,
               **config: Any):
        body: dict[str, Any] = {"config": config, "seats": seats}
        if seed is not None:
            body["seed"] = seed
        r = self.client.post("/tables", json=body, headers=auth(OPERATOR) if operator else {})
        assert r.status_code == 201, r.text
        self.table_id, self.host = r.json()["table_id"], r.json()["host_token"]
        return r

    def join_all(self) -> None:
        live = self.manager.tables[self.table_id]
        for seat in sorted(s for s, slot in live.slots.items() if slot.kind == "human"):
            r = self.client.post(f"/tables/{self.table_id}/join", json={"seat": seat})
            assert r.status_code == 200, r.text
            self.tokens[seat] = r.json()["seat_token"]

    def start(self):
        r = self.client.post(f"/tables/{self.table_id}/start", headers=auth(self.host))
        assert r.status_code == 200, r.text
        return r.json()

    def setup(self, seats: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
        self.create(seats, **kw)
        self.join_all()
        return self.start()

    def get(self, token: str | None = None):
        return self.client.get(f"/tables/{self.table_id}", headers=auth(token))

    def snap(self, seat: int | None = None) -> dict[str, Any]:
        r = self.get(self.tokens[seat] if seat is not None else None)
        assert r.status_code == 200, r.text
        return r.json()

    def envelope(self, view: dict[str, Any], action: dict[str, Any], action_id: str | None = None) -> dict[str, Any]:
        return {
            "action_id": action_id or f"a_{next(self._ids)}",
            "table_id": self.table_id,
            "hand_id": view["hand_id"],
            "decision_id": view["decision_id"],
            "expected_state_version": view["state_version"],
            "action": action,
        }

    def post_action(self, seat: int | None, body: Any, token: str | None = None):
        tok = token if token is not None else (self.tokens[seat] if seat is not None else None)
        return self.client.post(f"/tables/{self.table_id}/actions", headers=auth(tok), json=body)

    def act(self, seat: int, action: dict[str, Any], action_id: str | None = None):
        view = self.snap(seat)["view"]
        return self.post_action(seat, self.envelope(view, action, action_id))

    def actor(self) -> int | None:
        turn = self.snap()["turn"]
        return turn["seat"] if turn else None


HUMAN = {"kind": "human"}


def bot(policy: str = "legal_random") -> dict[str, Any]:
    return {"kind": "bot", "policy": policy}


def _api_test_store():
    """``ARENA_API_TEST_STORE=postgres`` runs the API suite on a freshly truncated test database."""
    if os.environ.get("ARENA_API_TEST_STORE", "memory") != "postgres":
        return None
    from arena.persistence.postgres import PostgresStore

    store = PostgresStore(os.environ.get("ARENA_TEST_DATABASE_URL",
                                         "postgresql://arena:arena@127.0.0.1:54329/arena_test"))
    with store._conn() as conn:
        conn.execute("TRUNCATE arena_tables CASCADE")
    return store


@contextmanager
def harness(*, bot_delay_ms: int = 0, hand_pause_ms: int = 0, runs_root: Path | None = None) -> Iterator[Harness]:
    clock = FakeClock()
    store = _api_test_store()
    manager = TableManager(clock, bot_delay_ms=bot_delay_ms, hand_pause_ms=hand_pause_ms, store=store)
    runs = RunManager(manager.bots, runs_root) if runs_root is not None else None
    app = create_app(manager, background_pump=False, operator_token=OPERATOR, runs=runs, advisor_dir=None)
    try:
        with TestClient(app) as client:
            yield Harness(client, manager, clock, app.state.runs)
    finally:
        manager.bots.close()
        if store is not None:
            store.close()
