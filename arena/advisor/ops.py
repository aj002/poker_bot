"""Everything the user can do to a session: enter cards and actions, undo, deal the next hand, fix chips."""

from __future__ import annotations

import copy
from typing import Any

from arena.advisor.render import end_stacks
from arena.advisor.session import Advisor, AdvisorError, _fresh_hand, _int, _ring, clean_card, clean_setup


def _check(adv: Advisor, session: dict[str, Any], index: int) -> Any:
    state = adv.state_in(session, index)
    return adv.simulate(session, index, state)


def _apply_entry(adv: Advisor, session: dict[str, Any], entry: dict[str, Any]) -> None:
    hand = session["hands"][-1]
    hand["log"].append(entry)
    try:
        _check(adv, session, len(session["hands"]) - 1)
    except Exception:
        hand["log"].pop()
        raise


def _cards(raw: Any, count: int | None = None) -> list[str]:
    if not isinstance(raw, list) or (count is not None and len(raw) != count) or not raw:
        raise AdvisorError("invalid_card", f"pick {count if count else 'some'} card{'s' if count != 1 else ''}")
    cards = [clean_card(c) for c in raw]
    if len(set(cards)) != len(cards):
        raise AdvisorError("duplicate_card", "the same card cannot be picked twice")
    return cards


def _used(adv: Advisor, session: dict[str, Any]) -> set[str]:
    hand = session["hands"][-1]
    return {c for e in hand["log"] if e["k"] in ("hole", "board") for c in e["cards"]}


def do(adv: Advisor, session: dict[str, Any], op: dict[str, Any]) -> None:
    kind = op.get("op")
    players = session["setup"]["players"]
    hand = session["hands"][-1]
    index = len(session["hands"]) - 1
    if kind == "hole":
        if hand["log"]:
            raise AdvisorError("already_dealt", "your cards are already entered; undo to change them")
        _apply_entry(adv, session, {"k": "hole", "cards": _cards(op.get("cards"), 2)})
    elif kind == "board":
        cards = _cards(op.get("cards"))
        clash = set(cards) & _used(adv, session)
        if clash:
            raise AdvisorError("duplicate_card", f"{', '.join(sorted(clash))} is already on the table or in your hand")
        _apply_entry(adv, session, {"k": "board", "cards": cards})
    elif kind == "action":
        seat = _int(op.get("seat"), "seat", low=0, high=players - 1)
        type_ = op.get("type")
        if type_ not in ("fold", "check", "call", "raise_to"):
            raise AdvisorError("invalid_action", "action must be fold, check, call or raise_to")
        entry: dict[str, Any] = {"k": "act", "seat": seat, "type": type_}
        if type_ == "raise_to":
            entry["amount"] = _int(op.get("amount"), "the amount", low=1)
        _apply_entry(adv, session, entry)
    elif kind == "award":
        pots = op.get("pots")
        if not isinstance(pots, list) or not all(isinstance(w, list) and w for w in pots):
            raise AdvisorError("invalid_award", "say who won each pot")
        sim = _check(adv, session, index)
        if sim.result is None or sim.result.ended_by_fold:
            raise AdvisorError("no_showdown", "there is nothing to award yet")
        if len(pots) != len(sim.result.pots):
            raise AdvisorError("invalid_award", f"there are {len(sim.result.pots)} pot(s)")
        for winners, pot in zip(pots, sim.result.pots, strict=True):
            if any(w not in pot.eligible for w in winners):
                raise AdvisorError("invalid_award", "that player is not in this pot")
        if any(e["k"] == "award" for e in hand["log"]):
            hand["log"] = [e for e in hand["log"] if e["k"] != "award"]
        _apply_entry(adv, session, {"k": "award", "pots": [sorted(set(w)) for w in pots]})
    elif kind == "undo":
        if hand["log"]:
            hand["log"].pop()
        elif len(session["hands"]) > 1:
            session["hands"].pop()
        else:
            raise AdvisorError("nothing_to_undo", "nothing to undo")
    elif kind == "new_hand":
        if not hand["log"]:
            raise AdvisorError("hand_not_started", "enter this hand first, or just edit it")
        sim = _check(adv, session, index)
        stacks, resolved = end_stacks(session, index, sim)
        over = op.get("stacks")
        if not sim.complete:
            stacks = {s: v for s, v in hand["stacks"].items()}
        elif not resolved and over is None:
            raise AdvisorError("award_first", "say who won the pot first, or type everyone's chips")
        if over is not None:
            if not isinstance(over, list) or len(over) != players:
                raise AdvisorError("invalid_stacks", "give one stack per player")
            stacks = {str(i): _int(v, "a stack", low=0) for i, v in enumerate(over)}
        if int(stacks[str(session["setup"]["hero"])]) <= 0:
            raise AdvisorError("hero_out", "you are out of chips: the session is over (start a new one)")
        if sum(1 for v in stacks.values() if v > 0) < 2:
            raise AdvisorError("table_over", "only one player has chips left")
        nxt = _fresh_hand(session["setup"], hand["n"] + 1, stacks, hand["button"])
        nxt["sb"], nxt["bb"], nxt["ante"], nxt["ante_posting"] = hand["sb"], hand["bb"], hand["ante"], hand["ante_posting"]
        session["hands"].append(nxt)
    elif kind == "edit_hand":
        if hand["log"]:
            raise AdvisorError("hand_started", "this hand has started; undo first or edit the next hand")
        edited = copy.deepcopy(hand)
        if "stacks" in op:
            stacks = op["stacks"]
            if not isinstance(stacks, list) or len(stacks) != players:
                raise AdvisorError("invalid_stacks", "give one stack per player")
            edited["stacks"] = {str(i): _int(v, "a stack", low=0) for i, v in enumerate(stacks)}
        for key, name in (("sb", "small blind"), ("bb", "big blind")):
            if key in op:
                edited[key] = _int(op[key], name, low=1)
        if edited["sb"] > edited["bb"]:
            raise AdvisorError("invalid_blinds", "the small blind cannot be bigger than the big blind")
        if "ante" in op:
            edited["ante"] = _int(op["ante"], "ante", low=0)
            posting = op.get("ante_posting", edited.get("ante_posting")) or "classic"
            edited["ante_posting"] = posting if edited["ante"] else None
            if edited["ante"] and posting not in ("classic", "big_blind"):
                raise AdvisorError("invalid_ante", "ante_posting must be classic or big_blind")
        if "button" in op:
            edited["button"] = _int(op["button"], "button", low=0, high=players - 1)
        ring = _ring(edited)
        if len(ring) < 2 or session["setup"]["hero"] not in ring:
            raise AdvisorError("invalid_stacks", "you and at least one other player need chips")
        if edited["button"] not in ring:
            later = [s for s in ring if s > edited["button"]]
            edited["button"] = later[0] if later else ring[0]
        session["hands"][-1] = edited
        try:
            _check(adv, session, index)
        except Exception:
            session["hands"][-1] = hand
            raise
    elif kind == "edit_setup":
        merged = {**session["setup"], **{k: v for k, v in op.items() if k in ("mode", "payouts")}}
        session["setup"] = clean_setup(merged)
    else:
        raise AdvisorError("unknown_op", f"unknown operation {kind!r}")
