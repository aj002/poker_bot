"""LAN use: the final bot as an in-process seat, the Mac's LAN addresses, LAN WebSocket origins."""

from __future__ import annotations

import json

import pytest
from starlette.websockets import WebSocketDisconnect

from arena.api import party
from arena.api.seats import FINAL_BOT_ENTRY, FINAL_BOT_RULESETS, FINAL_SUBMISSION
from tests.api.helpers import HUMAN, bot, harness
from tests.no_numba import run_without_numba


def test_final_bot_is_a_public_builtin_seat_with_the_submission_hash():
    with harness() as h:
        body = h.client.get("/catalog").json()
        spec = next(p for p in body["policies"] if p["id"] == "final_bot")
        bundle = json.loads(FINAL_SUBMISSION.read_text())["submission_bundle_sha256"]
        assert body["operator"] is False
        assert spec["source"] == "strategy" and spec["artifact_hash"] == bundle and bundle.startswith("bec8892d")
        assert spec["rulesets"] == ["practice_nlhe", "uncertainty_shortdeck", "uncertainty_plo4", "reveal_variant"]
        assert spec["state_mode"] == "explicit" and spec["bot_name"] == "final_bot"


def test_final_bot_metadata_matches_the_bot_itself():
    from strategy.final.runtime import NAME, RULESETS, FinalBot

    assert FINAL_BOT_RULESETS == RULESETS and FinalBot.state_mode == "explicit" and NAME == "final_bot"
    assert FINAL_BOT_ENTRY.is_file()


def test_two_final_bot_copies_play_a_table_with_the_right_labels():
    with harness() as h:
        snap = h.setup([bot("final_bot"), bot("final_bot"), bot("calling_station")], hand_cap=8, hands_per_level=4)
        assert snap["status"] == "complete"
        seats = snap["setup"]["seats"]
        assert [s["label"] for s in seats[:2]] == ["Final bot", "Final bot"]
        assert {s["source"] for s in seats[:2]} == {"strategy"}


def test_two_final_bot_copies_play_a_table_without_numba():
    done = run_without_numba("""
        from tests.api.helpers import bot, harness
        with harness() as h:
            snap = h.setup([bot("final_bot"), bot("final_bot"), bot("calling_station")], hand_cap=4, hands_per_level=4)
            assert snap["status"] == "complete", snap["status"]
    """)
    assert done.returncode == 0, done.stderr[-2000:]


def test_final_bot_plays_the_other_rulesets_too():
    for ruleset in ("uncertainty_shortdeck", "uncertainty_plo4"):
        with harness() as h:
            snap = h.setup([bot("final_bot"), bot("legal_random")], ruleset_id=ruleset, hand_cap=4, hands_per_level=2)
            assert snap["status"] == "complete"


def test_each_human_seat_sees_only_its_own_cards_next_to_final_bots():
    with harness() as h:
        h.setup([HUMAN, HUMAN, HUMAN, bot("final_bot")], hand_cap=10)
        views = {seat: h.snap(seat)["view"] for seat in (0, 1, 2)}
        for seat, view in views.items():
            assert len(view["hole_cards"]) == 2
            others = {c for s, v in views.items() if s != seat for c in v["hole_cards"]}
            assert not others & set(view["hole_cards"])
            rest = json.dumps({k: v for k, v in view.items() if k != "hole_cards"})
            assert all(f'"{c}"' not in rest for c in others)
        assert h.snap()["view"]["hole_cards"] is None


def test_party_urls_prefer_the_configured_address(monkeypatch):
    monkeypatch.setenv("ARENA_PUBLIC_URL", "http://192.168.1.23:5317/")
    assert party.public_urls() == ["http://192.168.1.23:5317"]
    monkeypatch.delenv("ARENA_PUBLIC_URL")
    monkeypatch.setattr(party, "lan_addresses", lambda: ["10.0.0.7"])
    monkeypatch.setenv("ARENA_WEB_PORT", "6000")
    assert party.public_urls() == ["http://10.0.0.7:6000"]


@pytest.mark.parametrize(("addr", "ok"), [("192.168.1.9", True), ("10.1.2.3", True), ("172.20.0.4", True),
                                          ("127.0.0.1", False), ("169.254.3.4", False), ("8.8.8.8", False),
                                          ("fe80::1", False), ("nonsense", False)])
def test_only_private_ipv4_addresses_are_offered(addr, ok):
    assert party._private(addr) is ok


def test_websocket_accepts_the_pages_own_lan_origin_only():
    with harness() as h:
        h.setup([HUMAN, bot("legal_random")])
        lan = "192.168.1.23:5317"
        with h.client.websocket_connect(f"/tables/{h.table_id}/stream",
                                        headers={"origin": f"http://{lan}", "host": lan}) as ws:
            ws.send_json({"type": "auth", "token": h.tokens[0]})
            assert ws.receive_json()["type"] == "snapshot"
        with pytest.raises(WebSocketDisconnect) as closed:
            with h.client.websocket_connect(f"/tables/{h.table_id}/stream",
                                            headers={"origin": "http://evil.example", "host": lan}):
                pass
        assert closed.value.code == 4403
