"""Slumbot adapter: action strings, the mirror table, the client's pacing, the statistics, and a
full session against an in-process fake server through the sandboxed worker path."""

from __future__ import annotations

import json
import random
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from arena.domain.cards import RANKS, SUITS
from arena.domain.models import Action
from arena.table.projections import observation
from strategy.eval import slumbot as S
from strategy.eval import slumbot_run as R
from strategy.eval import slumbot_stats as ST
from tests.strategy.slumbot_fake import FakeSlumbot

CARDS = [r + s for r in RANKS for s in SUITS]


# ---------------------------------------------------------------------- action strings


@pytest.mark.parametrize("text,expected", [
    ("", []),
    ("b200c/kk/kk/kb200", ["raise_to", "call", "check", "check", "check", "check", "check", "raise_to"]),
    ("b20000c///", ["raise_to", "call"]),
    ("ck/b300f", ["call", "check", "raise_to", "fold"]),
])
def test_parse_actions(text, expected):
    assert [a.type for a in S.parse_actions(text)] == expected


def test_parse_and_format_round_trip():
    text = "b200b600c/kb400b1200c/kk/b3000b20000c"
    acts = S.parse_actions(text)
    assert [S.format_action(a) for a in acts] == [t for t in S.tokenize(text) if t != "/"]
    assert acts[0] == Action(type="raise_to", amount=200)


@pytest.mark.parametrize("bad", ["x", "b", "bb200", "c k", "b2.5", "b-3", "kk?"])
def test_malformed_strings_are_rejected(bad):
    with pytest.raises(S.ProtocolMismatch):
        S.parse_actions(bad)


def test_format_action_covers_every_type():
    assert S.format_action(Action(type="fold")) == "f"
    assert S.format_action(Action(type="check")) == "k"
    assert S.format_action(Action(type="call")) == "c"
    assert S.format_action(Action(type="raise_to", amount=450)) == "b450"


# ----------------------------------------------------------------------- mirror table


@pytest.mark.parametrize("client_pos", [0, 1])
def test_deal_layout_matches_the_engine(client_pos):
    """Known cards land exactly where the engine deals them: hero and villain holes, flop,
    turn and river, with burns in between."""
    hero, villain = ("As", "Kd"), ("Qh", "Qd")
    board = ("2c", "7h", "9d", "Jc", "3s")
    t = S.replay(client_pos, hero, board, "c" + "k" + "/kk/kk/kk", first_hand=7, villain_hole=villain)
    res = t.hands[0].result
    assert t.complete and res is not None
    assert res.board == board
    assert res.revealed[S.HERO] == hero and res.revealed[S.VILLAIN] == villain
    assert S.hero_net(t) == -100  # queens beat ace-king on that board
    assert t.hands[0].hand_id == "h_007"


@pytest.mark.parametrize("client_pos,first_actor,call_amount", [(1, S.HERO, 50), (0, S.VILLAIN, 50)])
def test_positions_and_blinds(client_pos, first_actor, call_amount):
    t = S.replay(client_pos, ("As", "Kd"), (), "", first_hand=1)
    assert t.actor_seat == first_actor
    assert t.legal_actions().call_amount == call_amount
    assert t.legal_actions().max_raise_to == S.STACK


def test_replay_state_after_a_street():
    t = S.replay(0, ("As", "Kd"), ("2c", "7h", "9d", "Jc"), "b200c/kk/kb400", first_hand=3)
    assert t.actor_seat == S.HERO and t.hand.street == "turn"
    o = observation(t, S.HERO)
    assert o.pot_total == 800 and o.legal.call_amount == 400 and o.board == ["2c", "7h", "9d", "Jc"]
    assert o.legal.min_full_raise_to == 800


def test_bot_sees_hero_cards_only_and_stable_ids():
    t = S.replay(0, ("As", "Kd"), ("2c", "7h", "9d"), "b200c/", first_hand=12)
    assert t.actor_seat == S.HERO
    o = observation(t, S.HERO)
    assert o.hole_cards == ["As", "Kd"] and o.board == ["2c", "7h", "9d"]
    assert o.hand_id == "h_012" and o.hand_number == 12 and o.table_id == S.TABLE_ID
    assert o.hand_cap == S.HAND_CAP and o.payouts == [1, 0]
    assert all(p.shown_cards is None for p in o.players)
    assert [p.stack + p.hand_committed for p in o.players] == [S.STACK, S.STACK]


@pytest.mark.parametrize("action", [
    "b50", "k", "b100c/b50", "b20001", "b150c/kk/kk/kk/k", "cc", "ck/kb20000c/b100",
    "b200c", "b200c//", "c/",
])
def test_illegal_or_inconsistent_strings_raise(action):
    with pytest.raises(S.ProtocolMismatch):
        S.replay(1, ("As", "Kd"), ("2c", "7h", "9d", "Jc", "3s"), action, first_hand=1)


def test_replay_accepts_all_in_runouts_with_empty_streets():
    t = S.replay(1, ("As", "Ah"), ("2c", "7h", "9d", "Jc", "3s"), "b20000c///", first_hand=1,
                 villain_hole=("Kd", "Kc"))
    assert t.complete and S.hero_net(t) == S.STACK


def test_duplicate_or_bad_cards_are_rejected():
    with pytest.raises(S.ProtocolMismatch):
        S.build_deck(1, ("As", "As"))
    with pytest.raises(S.ProtocolMismatch):
        S.build_deck(1, ("As", "Kd"), ("As", "2c", "3c"))
    with pytest.raises(S.ProtocolMismatch):
        S.build_deck(1, ("Ax", "Kd"))


def test_fold_pays_blinds():
    t = S.replay(1, ("7s", "2d"), (), "f", first_hand=1)
    assert S.hero_net(t) == -50
    t = S.replay(0, ("7s", "2d"), (), "b200f", first_hand=1)
    assert S.hero_net(t) == -100


# ------------------------------------------------------------------------------- client


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


class FakeResponse:
    def __init__(self, body: dict) -> None:
        self.body = body

    def read(self) -> bytes:
        return json.dumps(self.body).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _api(monkeypatch, outcomes, **kw):
    clock = FakeClock()
    seen: list[dict] = []

    def fake_urlopen(req, timeout=None):
        seen.append({"url": req.full_url, "body": json.loads(req.data), "ua": req.headers.get("User-agent")})
        out = outcomes.pop(0)
        if isinstance(out, BaseException):
            raise out
        return FakeResponse(out)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return R.Api(sleep=clock.sleep, clock=clock.now, **kw), clock, seen


def test_api_paces_requests_and_sends_identifying_agent(monkeypatch):
    api, clock, seen = _api(monkeypatch, [{"token": "a", "action": ""}, {"token": "b", "action": "b200"}], min_interval=0.5)
    api.new_hand()
    api.act("c")
    assert clock.sleeps == [0.5]
    assert seen[1]["body"] == {"token": "a", "incr": "c"} and api.token == "b"
    assert seen[0]["url"].endswith("/slumbot/api/new_hand") and "poker-bot-arena" in seen[0]["ua"]


def test_api_backs_off_exponentially_then_succeeds(monkeypatch):
    err = urllib.error.URLError("down")
    api, clock, _ = _api(monkeypatch, [err, TimeoutError(), err, {"token": "t"}], min_interval=0.0)
    assert api.new_hand()["token"] == "t"
    assert clock.sleeps == [2.0, 4.0, 8.0] and api.retries == 3


def test_api_gives_up_and_does_not_retry_protocol_errors(monkeypatch):
    api, clock, _ = _api(monkeypatch, [urllib.error.URLError("x")] * 3, min_interval=0.0, max_retries=2)
    with pytest.raises(R.ApiError, match="gave up"):
        api.new_hand()
    assert len(clock.sleeps) == 2
    api, clock, seen = _api(monkeypatch, [{"error_msg": "Illegal call", "old_action": "b200c/"}], min_interval=0.0)
    with pytest.raises(R.ApiError, match="Illegal call"):
        api.act("c")
    assert len(seen) == 1 and not clock.sleeps
    api, clock, seen = _api(monkeypatch, [urllib.error.HTTPError("u", 404, "nf", {}, None)], min_interval=0.0)
    with pytest.raises(R.ApiError, match="HTTP 404"):
        api.new_hand()
    assert len(seen) == 1


# ----------------------------------------------------------------------------- stats


def test_hand_evaluator_agrees_with_pokerkit():
    from pokerkit import StandardHighHand

    rng = random.Random(5)
    for i in range(3000):
        pool = CARDS if i % 2 else [c for c in CARDS if c[0] in "2345TJQKA"]
        c = rng.sample(pool, 9)
        h1, h2, b = c[:2], c[2:4], c[4:]
        a = StandardHighHand.from_game("".join(h1), "".join(b))
        z = StandardHighHand.from_game("".join(h2), "".join(b))
        s1 = ST.score7(np.array([ST._CARD[x] for x in h1 + b]))
        s2 = ST.score7(np.array([ST._CARD[x] for x in h2 + b]))
        assert (s1 > s2) - (s1 < s2) == (a > z) - (a < z)


def test_equity_known_values():
    assert ST.equity(["As", "Ah"], ["Kd", "Kc"], ["2c", "7h", "9d", "Jc", "3s"]) == 1.0
    assert ST.equity(["As", "Kh"], ["Qd", "Qc"], ["2c", "7d", "9s", "3h"]) == pytest.approx(6 / 44, abs=1e-9)
    assert ST.equity(["Ad", "Kd"], ["As", "Ks"], ["2c", "7h", "9d", "Jc", "3s"]) == 0.5
    pre = ST.equity(["As", "Ah"], ["Kd", "Kc"], [])
    assert 0.80 < pre < 0.83


@pytest.mark.parametrize("action,street", [
    ("b20000c///", 0), ("b300b900b20000c///", 0), ("b300c/b19700c//", 1), ("ck/kb20000c//", 1),
    ("b300c/kk/kb20000c/", 2), ("b300c/kk/kk/kb20000c", 3),
    ("b300c/kb1000c/", None), ("b20000f", None), ("b300c/kk/kk/kb1000c", None), ("b200c/kk/kk/kk", None),
])
def test_all_in_street(action, street):
    assert ST.all_in_street(action) == street


def test_adjusted_winnings_replaces_only_the_runout():
    rec = {"action": "b300c/kb20000c//", "hole": ["As", "Ah"], "bot_hole": ["Kd", "Kc"],
           "board": ["2c", "7h", "9d", "Jc", "3s"], "winnings": 20000, "n": 1}
    eq = ST.equity(rec["hole"], rec["bot_hole"], rec["board"][:3])
    assert ST.adjusted_winnings(rec) == pytest.approx((2 * eq - 1) * S.STACK)
    assert 0 < ST.adjusted_winnings(rec) < S.STACK
    rec["action"] = "b300c/kk/kk/kb20000c"
    assert ST.adjusted_winnings(rec) == 20000
    rec["action"] = "b300c/kk/kk/kb500c"
    assert ST.adjusted_winnings(rec) == 20000
    rec["bot_hole"] = None
    rec["action"] = "b20000c///"
    assert ST.adjusted_winnings(rec) == 20000


def test_summary_and_difference():
    s = ST.summarize([0.0, 2.0, 4.0, 6.0])
    assert s["mean"] == 3 and s["n"] == 4
    assert s["se"] == pytest.approx(s["sd"] / 2)
    assert s["lo"] == pytest.approx(3 - 1.96 * s["se"])
    d = ST.difference([1.0, 3.0], [0.0, 2.0])
    assert d["diff"] == 1.0 and d["lo"] < 1.0 < d["hi"]
    assert ST.mbb(100) == 1000.0
    assert ST.summarize([])["n"] == 0


# ------------------------------------------------------------ full session, fake server


BOT_SRC = '''
from arena.agents.sdk import ExplicitStateBot, call, check, clamp_raise_to, fold, raise_to

class TinyBot(ExplicitStateBot):
    rulesets = ("practice_nlhe",)
    name = "tiny"
    def initial_state(self, table):
        return {"i": 0}
    def decide(self, obs, state):
        i = state["i"] + 1
        legal = obs.legal
        k = (i * 7 + len(obs.board)) % 10
        if legal.can_raise and k in (0, 1):
            act = clamp_raise_to(legal, legal.min_full_raise_to * 3) or call()
        elif legal.can_raise and k == 2:
            act = raise_to(legal.max_raise_to)
        elif legal.check:
            act = check()
        elif k == 9:
            act = fold()
        else:
            act = call()
        return act, {"i": i}

def make_bot():
    return TinyBot()
'''


def test_session_against_fake_server_reconciles_every_hand(tmp_path: Path):
    bot_dir = tmp_path / "tiny"
    bot_dir.mkdir()
    (bot_dir / "bot.py").write_text(BOT_SRC)
    out = tmp_path / "run.jsonl"
    fake = FakeSlumbot(seed=11)
    res = R.play({"tiny": bot_dir}, 150, out, block=40, api=fake, log=lambda *a, **k: None)
    recs = [json.loads(line) for line in out.read_text().splitlines()]
    assert res["done"] == {"tiny": 150} and len(recs) == 150
    assert all(r["reconciled"] for r in recs), [r for r in recs if not r["reconciled"]][:2]
    assert all(r["winnings"] == r["engine_net"] for r in recs)
    kinds = {r["client_pos"] for r in recs}
    assert kinds == {0, 1}
    assert any(S_ends_all_in(r["action"]) for r in recs)
    assert any("f" in r["action"] for r in recs)
    assert sum(1 for r in recs if r["bot_hole"]) > 10
    assert [r["serial"] for r in recs] == list(range(1, 151))
    fb = sum(1 for r in recs for d in r["decisions"] if d["fb"])
    assert fb <= 3  # a late worker reply on a loaded machine is a legal fallback, never an error
    # resume: counts what is already there and plays nothing more
    res2 = R.play({"tiny": bot_dir}, 150, out, api=FakeSlumbot(seed=12), log=lambda *a, **k: None)
    assert res2["done"] == {"tiny": 150} and len(out.read_text().splitlines()) == 150


def S_ends_all_in(action: str) -> bool:
    return ST.all_in_street(action) is not None


def test_two_bundles_alternate_blocks(tmp_path: Path):
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "bot.py").write_text(BOT_SRC)
    out = tmp_path / "two.jsonl"
    R.play({"a": tmp_path / "a", "b": tmp_path / "b"}, 20, out, block=5, api=FakeSlumbot(seed=3),
           log=lambda *a, **k: None)
    seq = [json.loads(line)["cfg"] for line in out.read_text().splitlines()]
    assert seq == (["a"] * 5 + ["b"] * 5) * 4


def test_illegal_bot_action_falls_back_to_check_or_fold(tmp_path: Path):
    src = BOT_SRC.replace("return act, {\"i\": i}", "return {'type': 'raise_to', 'amount': 1}, {\"i\": i}")
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "bot.py").write_text(src)
    out = tmp_path / "bad.jsonl"
    R.play({"bad": tmp_path / "bad"}, 10, out, api=FakeSlumbot(seed=5), log=lambda *a, **k: None)
    recs = [json.loads(line) for line in out.read_text().splitlines()]
    decs = [d for r in recs for d in r["decisions"]]
    assert decs and all(d["fb"] == "illegal_action" for d in decs)
    assert all(d["act"] in ("k", "f") for d in decs)
    assert all(r["reconciled"] for r in recs)


def test_off_bundle_patches_only_exploit_weights(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "bot.py").write_text(
        'def make_bot():\n    return FinalBot(pack=DATA / "blueprint-practice_nlhe.npz", '
        'exploit_pack=DATA / "exploit-practice_nlhe.npz")\n')
    R.build_off_bundle(src, tmp_path / "dst")
    text = (tmp_path / "dst" / "bot.py").read_text()
    assert 'overrides={"lam_scale": 0.0}' in text and 'blueprint-practice_nlhe.npz' in text
    compile(text, "bot.py", "exec")


def test_report_summarises_a_log(tmp_path: Path):
    from strategy.eval import slumbot_report as RP

    bot_dir = tmp_path / "tiny"
    bot_dir.mkdir()
    (bot_dir / "bot.py").write_text(BOT_SRC)
    out = tmp_path / "r.jsonl"
    R.play({"tiny": bot_dir, "twin": bot_dir}, 30, out, block=10, api=FakeSlumbot(seed=9), log=lambda *a, **k: None)
    rep = RP.report(RP.load(out))
    assert set(rep["configs"]) == {"tiny", "twin"}
    c = rep["configs"]["tiny"]
    assert c["hands"] == 30 and c["reconciled"] == 30
    assert c["adjusted_mbb"]["n"] == 30
    assert rep["difference"]["raw"]["lo"] <= rep["difference"]["raw"]["diff"] <= rep["difference"]["raw"]["hi"]
    assert "tiny minus twin" in RP.markdown(rep)


def test_luck_terms_shapes_and_zero_without_villain_cards():
    rec = {"n": 1, "action": "b300c/kk/kb400f", "hole": ["As", "Kd"], "bot_hole": ["Qh", "Qd"],
           "board": ["2c", "7h", "9d", "Jc", "3s"], "winnings": 700}
    chips, terms = ST.luck_terms(rec)
    assert chips == 700 and terms[0] != 0 and terms[1] != 0 and terms[2] != 0 and terms[3] == 0
    rec["action"] = "f"
    assert ST.luck_terms(rec)[1][1:] == [0.0, 0.0, 0.0]
    rec["bot_hole"] = None
    assert ST.luck_terms(rec) == (700.0, [0.0] * 4)
    # all-in on the flop: the result becomes an expectation and only terms up to the flop remain
    rec = {"n": 2, "action": "b300c/kb20000c//", "hole": ["As", "Ah"], "bot_hole": ["Kd", "Kc"],
           "board": ["2c", "7h", "9d", "Jc", "3s"], "winnings": 20000}
    chips, terms = ST.luck_terms(rec)
    assert 0 < chips < S.STACK and terms[1] != 0 and terms[2:] == [0.0, 0.0]


def test_luck_terms_have_zero_mean_over_random_deals():
    rng = random.Random(4)
    xs = []
    for i in range(400):
        c = rng.sample(CARDS, 9)
        rec = {"n": i, "action": "ck/kk/kk/kk", "hole": c[:2], "bot_hole": c[2:4], "board": c[4:], "winnings": 0}
        xs.append(ST.luck_terms(rec)[1])
    mean = [sum(x[j] for x in xs) / len(xs) for j in range(4)]
    sd = [(sum(x[j] ** 2 for x in xs) / len(xs)) ** 0.5 for j in range(4)]
    assert all(abs(m) < 4 * s / len(xs) ** 0.5 for m, s in zip(mean, sd))


def test_regression_adjust_removes_explained_noise_and_keeps_the_mean():
    rng = random.Random(8)
    x = [[rng.gauss(0, 1), 0.0, 0.0, 0.0] for _ in range(4000)]
    y = [100.0 + 3000 * xi[0] + rng.gauss(0, 300) for xi in x]
    raw = ST.summarize([ST.mbb(v) for v in y])
    adj = ST.regression_adjust(y, x)
    assert adj["beta"][0] == pytest.approx(3000, rel=0.05)
    assert adj["se"] < raw["se"] / 5
    assert adj["lo"] < ST.mbb(100.0) < adj["hi"]


def test_decision_ids_are_unique_across_hands_and_stable_within_one():
    """The bot draws its mixed-strategy random number from the decision id, so a rebuilt table
    must give each hand its own ids; the same decision rebuilt twice must repeat its id."""
    ids = []
    for h in range(1, 40):
        t = S.replay(1, ("As", "Kd"), (), "", first_hand=h)
        ids.append(observation(t, S.HERO).decision_id)
        again = S.replay(1, ("As", "Kd"), (), "", first_hand=h)
        assert observation(again, S.HERO).decision_id == ids[-1]
    assert len(set(ids)) == len(ids)
    t = S.replay(0, ("As", "Kd"), ("2c", "7h", "9d"), "b200c/", first_hand=3)
    t2 = S.replay(0, ("As", "Kd"), ("2c", "7h", "9d"), "b200c/", first_hand=4)
    assert observation(t, S.HERO).decision_id != observation(t2, S.HERO).decision_id


def test_mixed_strategy_draws_vary_across_hands():
    from strategy.blueprint.runtime import _uniform

    us = []
    for h in range(1, 200):
        t = S.replay(1, ("As", "Kd"), (), "", first_hand=h)
        us.append(_uniform("seed", observation(t, S.HERO).decision_id))
    assert len({round(u, 6) for u in us}) == len(us) and 0.3 < sum(us) / len(us) < 0.7


def test_state_snapshots_are_logged_every_fifty_hands(tmp_path: Path):
    bot_dir = tmp_path / "b"
    bot_dir.mkdir()
    (bot_dir / "bot.py").write_text(BOT_SRC)
    out = tmp_path / "s.jsonl"
    R.play({"b": bot_dir}, 100, out, api=FakeSlumbot(seed=2, style="station"), log=lambda *a, **k: None)
    recs = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["serial"] for r in recs if "bot_state" in r] == [50, 100]
    assert all(r["reconciled"] for r in recs)
    station = [r for r in recs if r["action"].count("b") == sum(1 for d in r["decisions"] if d["act"][0] == "b")]
    assert len(station) == len(recs)  # the station never raises, so every bet in the log is ours


def test_report_tolerates_repeated_hand_indices(tmp_path: Path):
    from strategy.eval import slumbot_report as RP

    bot_dir = tmp_path / "t"
    bot_dir.mkdir()
    (bot_dir / "bot.py").write_text(BOT_SRC)
    out = tmp_path / "r.jsonl"
    R.play({"t": bot_dir}, 20, out, api=FakeSlumbot(seed=1), log=lambda *a, **k: None)
    recs = RP.load(out)
    joined = recs + [{**r, "n": r["n"]} for r in recs]
    rep = RP.report(joined)
    assert rep["configs"]["t"]["hands"] == 40
