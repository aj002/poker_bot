"""Advisor HTTP surface: create, operate, error shape, persistence across a server restart."""

from __future__ import annotations

from fastapi.testclient import TestClient

from arena.api.app import create_app
from arena.api.manager import TableManager
from tests.api.helpers import FakeClock, harness

SETUP = {"players": 3, "hero": 0, "sb": 50, "bb": 100, "mode": "cash", "stacks": [5000, 5000, 5000]}


def test_a_hand_can_be_played_over_http_and_errors_use_the_api_shape():
    with harness() as h:
        r = h.client.post("/advisor/sessions", json=SETUP)
        assert r.status_code == 201
        v = r.json()
        sid = v["session"]
        assert v["stage"] == "hole" and v["hand_number"] == 1 and v["advice"] is None
        url = f"/advisor/sessions/{sid}/do"
        bad = h.client.post(url, json={"op": "hole", "cards": ["As", "As"]})
        assert bad.status_code == 422 and bad.json()["error"]["code"] == "duplicate_card"
        v = h.client.post(url, json={"op": "hole", "cards": ["As", "Ad"]}).json()
        while v["stage"] == "action" and v["actor"] != 0:
            v = h.client.post(url, json={"op": "action", "seat": v["actor"], "type": "fold"}).json()
        if v["stage"] == "action":
            assert v["advice"]["headline"] and v["legal"]["can_raise"]
        assert h.client.get(f"/advisor/sessions/{sid}").json()["log"] == v["log"]
        assert h.client.post(url, json={"op": "nonsense"}).status_code == 422
        assert h.client.get("/advisor/sessions/nope").status_code == 404
        assert h.client.post("/advisor/sessions", json={**SETUP, "hero": 9}).status_code == 422


def test_sessions_survive_a_server_restart(tmp_path):
    def server():
        return TestClient(create_app(TableManager(FakeClock()), background_pump=False, advisor_dir=tmp_path))

    with server() as a:
        sid = a.post("/advisor/sessions", json=SETUP).json()["session"]
        before = a.post(f"/advisor/sessions/{sid}/do", json={"op": "hole", "cards": ["Kc", "Kd"]}).json()
    with server() as b:
        after = b.get(f"/advisor/sessions/{sid}").json()
    assert after["hole"] == ["Kc", "Kd"] and after["log"] == before["log"] and after["actor"] == before["actor"]
