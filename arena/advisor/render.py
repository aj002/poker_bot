"""What the advisor page shows: the hand so far, whose turn it is, and the bot's advice at the hero's turn."""

from __future__ import annotations

from collections import Counter
from typing import Any

from arena.advisor.session import MIX_STEP
from arena.api.styles import build_aids
from arena.table.projections import Viewer, table_view

POSITIONS = {
    2: ["BTN/SB", "BB"],
    3: ["BTN", "SB", "BB"],
    4: ["UTG", "BTN", "SB", "BB"],
    5: ["UTG", "CO", "BTN", "SB", "BB"],
    6: ["UTG", "HJ", "CO", "BTN", "SB", "BB"],
    7: ["UTG", "MP", "HJ", "CO", "BTN", "SB", "BB"],
    8: ["UTG", "UTG+1", "MP", "HJ", "CO", "BTN", "SB", "BB"],
}


def position_labels(ring: list[int], button: int, big_blind: int) -> dict[int, str]:
    """Clockwise from the seat after the big blind (under the gun) to the big blind."""
    n = len(ring)
    start = ring.index(big_blind)
    order = [ring[(start + 1 + i) % n] for i in range(n)]
    if n == 2:
        return {button: "BTN/SB", big_blind: "BB"}
    return dict(zip(order, POSITIONS[n], strict=True))


def chip_text(n: int) -> str:
    return f"{n:,}"


def bb_text(amount: int, bb: int) -> str:
    x = amount / bb
    return f"{x:.0f}" if abs(x - round(x)) < 0.05 else f"{x:.1f}"


def action_label(a: dict[str, Any], obs: Any, bb: int) -> str:
    kind, amount = a["type"], a.get("amount")
    legal = obs.legal
    if kind == "fold":
        return "Fold"
    if kind == "check":
        return "Check"
    if kind == "call":
        return f"Call {chip_text(legal.call_amount or 0)}"
    if amount is not None and amount == legal.all_in_raise_to:
        return f"All-in {chip_text(amount)} ({bb_text(amount, bb)} BB)"
    verb = "Bet" if obs.current_bet_to == 0 else "Raise to"
    return f"{verb} {chip_text(amount)} ({bb_text(amount, bb)} BB)"


def advise(adv: Any, sim: Any, bb: int, ring_names: dict[int, str]) -> dict[str, Any]:
    obs, state = sim.pending
    pick, note, _ = adv._decide(obs, state)
    counts: Counter = Counter()
    for i in range(adv.samples):
        action, _, _ = adv._decide(obs.model_copy(update={"decision_id": f"{obs.decision_id}~{i}"}), state)
        counts[(action["type"], action.get("amount"))] += 1
    key = lambda a: (a["type"], a.get("amount"))  # noqa: E731
    mix = []
    for (kind, amount), n in counts.most_common():
        action = {"type": kind, **({"amount": amount} if amount is not None else {})}
        mix.append({"label": action_label(action, obs, bb), "action": action,
                    "pct": max(MIX_FLOOR, round(100 * n / adv.samples / MIX_STEP) * MIX_STEP)})
    mix = [m for m in mix if m["pct"] >= MIX_STEP]
    pick_label = action_label(pick, obs, bb)
    if key(pick) not in {key(m["action"]) for m in mix}:
        mix.append({"label": pick_label, "action": pick, "pct": MIX_STEP})
    top = mix[0]
    mixed = len(mix) > 1 and top["pct"] < 95
    view = table_view(sim.table, Viewer.for_seat(obs.hero_seat))
    aids = build_aids(view, obs)["hero"]
    me = next(p for p in view.players if p.seat == obs.hero_seat)
    parts = [aids["hand_in_words"]]
    if aids["pot_odds"]:
        parts.append(aids["pot_odds"]["words"])
    eq = aids["equity_vs_random"]
    parts.append(f"about {round(eq['vs_field'] * 100)}% against random hands")
    depth = (me.stack + me.street_committed) / bb
    source = "blueprint"
    if note == "push/fold table":
        source = "push/fold chart"
        parts.append(f"short stack ({depth:.0f} BB): the push/fold chart decides")
    elif note and note.startswith("blueprint miss"):
        source = "fallback rules"
    reason = "; ".join(parts) + "."
    return {
        "decision": obs.decision_id,
        "headline": pick_label,
        "pick": pick,
        "mix": mix,
        "mixed": mixed,
        "mix_text": ", ".join(f"{m['label']} ({m['pct']}%)" for m in mix),
        "reason": reason[0].upper() + reason[1:],
        "source": source,
        "position": ring_names.get(obs.hero_seat),
        "effective_bb": round(depth, 1),
    }


MIX_FLOOR = 5


def _describe_events(sim: Any, bb: int) -> list[str]:
    out: list[str] = []
    for e in sim.table.hands[-1].events:
        d = e.data
        if e.type == "AntePosted":
            out.append(f"Seat {d['seat'] + 1} antes {chip_text(d['amount'])}")
        elif e.type == "BlindPosted":
            out.append(f"Seat {d['seat'] + 1} posts {chip_text(d['amount'])}")
        elif e.type == "ActionApplied" and not d.get("auto"):
            a = d["action"]
            who = f"Seat {d['seat'] + 1}"
            if a["type"] == "raise_to":
                out.append(f"{who} raises to {chip_text(a['amount'])}")
            elif a["type"] == "call":
                to = d["street_committed"]
                out.append(f"{who} calls {chip_text(to)}" + (f" (adds {chip_text(d['added'])})" if d["added"] != to else ""))
            else:
                out.append(f"{who} {'folds' if a['type'] == 'fold' else 'checks'}")
        elif e.type == "ExcessReturned":
            out.append(f"{chip_text(d['amount'])} returned to seat {d['seat'] + 1}")
    return out


def end_stacks(session: dict[str, Any], index: int, sim: Any) -> tuple[dict[str, int], bool]:
    """Stacks after this hand, and whether the pot is fully accounted for."""
    hand = session["hands"][index]
    start = {int(s): v for s, v in hand["stacks"].items()}
    out = dict(start)
    res = sim.result
    if res is None:
        view = table_view(sim.table, Viewer.spectator())
        for p in view.players:
            out[p.seat] = p.stack
        return {str(s): v for s, v in out.items()}, False
    if res.ended_by_fold:
        out.update(res.final_stacks)
        return {str(s): v for s, v in out.items()}, True
    for seat, paid in res.contributions.items():
        out[seat] = start[seat] - paid
    if sim.awards is None:
        return {str(s): v for s, v in out.items()}, False
    for pot, winners in zip(res.pots, sim.awards, strict=True):
        share, extra = divmod(pot.amount, len(winners))
        for i, seat in enumerate(sorted(winners)):
            out[seat] += share + (1 if i < extra else 0)
    return {str(s): v for s, v in out.items()}, True


def render(adv: Any, session: dict[str, Any], index: int, sim: Any) -> dict[str, Any]:
    setup, hand = session["setup"], session["hands"][index]
    hero, bb = setup["hero"], hand["bb"]
    table, res = sim.table, sim.result
    view = table_view(table, Viewer.spectator())
    pos = table.positions
    labels = position_labels(sim.ring, pos.button, pos.big_blind)
    players = {p.seat: p for p in view.players}
    started = bool(hand["log"]) and hand["log"][0]["k"] == "hole"
    stage = "hole" if not started else "complete" if sim.complete else "board" if sim.need_board else "action"
    actor = table.actor_seat if stage == "action" else None
    legal = None
    if actor is not None:
        lg = table.legal_actions()
        legal = {"fold": lg.fold, "check": lg.check, "call_amount": lg.call_amount, "can_raise": lg.can_raise,
                 "min_raise_to": lg.min_full_raise_to, "max_raise_to": lg.max_raise_to,
                 "short_all_in_raise_to": lg.short_all_in_raise_to, "pot_raise_to": lg.pot_raise_to,
                 "all_in_raise_to": lg.all_in_raise_to, "blocked": lg.raise_blocked_reason}
    seats = []
    for s in range(setup["players"]):
        stack0 = hand["stacks"].get(str(s), 0)
        p = players.get(s)
        seats.append({
            "seat": s, "hero": s == hero, "button": s == pos.button and s in sim.ring,
            "position": labels.get(s), "in_hand": s in sim.ring, "stack_start": stack0,
            "stack": p.stack if p else stack0,
            "street_committed": p.street_committed if p else 0,
            "status": (p.status if p else "out") if started else ("ready" if s in sim.ring else "out"),
        })
    stacks_after, resolved = end_stacks(session, index, sim)
    result = None
    if res is not None:
        result = {
            "ended_by_fold": res.ended_by_fold, "resolved": resolved,
            "winner": next((s for s in sim.ring if s not in res.folded), None) if res.ended_by_fold else None,
            "pots": [{"amount": p.amount, "eligible": sorted(p.eligible)} for p in res.pots],
            "awards": sim.awards, "stacks_after": stacks_after,
        }
    advice = None
    if sim.pending is not None and sim.hole:
        names = {s: labels.get(s) for s in sim.ring}
        advice = advise(adv, sim, bb, names)
    street = ["preflop", "flop", "turn", "river"][min(3, max(0, {0: 0, 3: 1, 4: 2, 5: 3}.get(len(sim.board), 0)))]
    need = sim.need_board
    used = list(sim.hole or []) + list(sim.board)
    return {
        "session": session["id"],
        "setup": setup,
        "hand_number": hand["n"],
        "hands_total": len(session["hands"]),
        "button": pos.button,
        "blinds": {"sb": hand["sb"], "bb": hand["bb"], "ante": hand["ante"], "ante_posting": hand["ante_posting"]},
        "stage": stage,
        "street": street,
        "board": sim.board,
        "hole": sim.hole,
        "used_cards": used,
        "pot": view.pot_total,
        "seats": seats,
        "actor": actor,
        "legal": legal,
        "need_board": need,
        "street_needed": ("flop" if need == 3 else "turn" if len(sim.board) == 3 else "river") if need else None,
        "advice": advice,
        "log": _describe_events(sim, bb),
        "result": result,
        "can_undo": bool(hand["log"]) or len(session["hands"]) > 1,
        "editable": not hand["log"],
        "next_stacks": stacks_after if started else hand["stacks"],
    }
