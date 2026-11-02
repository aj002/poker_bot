"""PHH (Poker Hand History, TOML) export: secondary interchange only.

The canonical record is the JSON history; PHH cannot hold burns, the deck,
payout points, ICM or event timing. Each complete hand exports to one PHH
document. Players are listed in PokerKit index order (heads-up: big blind first),
so ``blinds_or_straddles`` is the vector PokerKit expects. ``seats`` holds
1-based physical seats.

Practice-rule notes carried as PHH user-defined fields (``_arena_*``):

* A call priced at the nominal big blind over a short posted blind is written as
  ``cbr <total> # arena:nominal_bb_call``. Generic PHH readers will treat it as a
  completion.
* An automatic check (nobody left who could respond) is ``cc # arena:auto``.
* When the big blind is posted short, the arena prices calls and raise increments
  from the nominal big blind; ``_arena_short_big_blind_correction`` flags those
  hands, which generic readers may reject or price differently.
* Odd chips follow the practice rule (one each, clockwise after the button), so
  ``finishing_stacks`` can differ from a reader that awards the whole remainder to
  one winner. ``_arena_odd_chip_correction`` flags those hands.

:func:`parse_phh` and :func:`replay_phh` read PHH back without PokerKit's notation
module and re-run the hand through the arena engine. Round-trip tests compare
cards, actions and final stacks.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from typing import Any

from arena.domain.cards import full_deck
from arena.domain.models import Action, LocalRules
from arena.history.canonical import TableHistory
from arena.rules.base import HandEvent
from arena.rules.pokerkit_adapter import HandSpec, PokerKitHand, engine_order, seat_antes
from arena.rules.profiles import PROFILES, get_profile

PHH_VARIANTS = {"practice_nlhe": "NT", "uncertainty_shortdeck": "NS", "uncertainty_plo4": "PO"}


class PhhError(ValueError):
    pass


@dataclass(frozen=True)
class PhhHandMeta:
    table_id: str
    hand_id: str
    hand_number: int
    level: int
    ruleset_id: str
    seat_ring: tuple[int, ...]
    button: int
    small_blind_seat: int | None
    big_blind_seat: int
    small_blind: int
    big_blind: int
    start_stacks: dict[int, int]
    local_rules: dict[str, Any]
    ante: int = 0
    ante_posting: str | None = None


def _spec(meta: PhhHandMeta, deck: tuple[str, ...]) -> HandSpec:
    return HandSpec(
        ante=meta.ante,
        ante_posting=meta.ante_posting,  # type: ignore[arg-type]
        profile=get_profile(meta.ruleset_id),
        seat_ring=meta.seat_ring,
        stacks=dict(meta.start_stacks),
        button=meta.button,
        small_blind_seat=meta.small_blind_seat,
        big_blind_seat=meta.big_blind_seat,
        small_blind=meta.small_blind,
        big_blind=meta.big_blind,
        deck=deck,
        local_rules=LocalRules(**meta.local_rules),
    )


def _blind_vector(meta: PhhHandMeta, order: tuple[int, ...]) -> list[int]:
    if len(order) == 2:
        return [meta.small_blind, meta.big_blind]
    blinds = [0] * len(order)
    if meta.small_blind_seat is not None:
        blinds[order.index(meta.small_blind_seat)] = meta.small_blind
    blinds[order.index(meta.big_blind_seat)] = meta.big_blind
    return blinds


def _ante_fields(meta: PhhHandMeta, order: tuple[int, ...]) -> tuple[bool, list[int]]:
    """``(ante_trimming_status, antes)`` in PokerKit's convention (heads-up: big blind first,
    reading entry 1), with each seat's ante as actually owed (``seat_antes``)."""
    if not meta.ante:
        return False, [0] * len(order)
    owed = seat_antes(_spec(meta, ()))
    by_index = [owed.get(s, 0) for s in order]
    if len(order) == 2:
        by_index.reverse()
    return meta.ante_posting == "classic", by_index


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise TypeError(f"cannot write {type(value).__name__} to PHH")


def export_hand(meta: PhhHandMeta, events: list[Any]) -> str:
    """PHH text for one complete hand, from its ``HandStarted``..``HandCompleted`` events."""
    order = engine_order(_spec(meta, tuple(full_deck(get_profile(meta.ruleset_id).ranks))))
    player = {seat: f"p{i + 1}" for i, seat in enumerate(order)}
    holes: dict[int, list[str]] = {s: [] for s in order}
    actions: list[str] = []
    final: dict[int, int] | None = None
    corrected = False
    posted: list[int] = []
    holes_written = False

    def flush_holes() -> None:
        nonlocal holes_written
        if not holes_written:
            actions.extend(f"d dh {player[s]} {''.join(holes[s])}" for s in order)
            holes_written = True

    for event in events:
        kind, data = event.type, event.data
        if kind == "BlindPosted":
            posted.append(data["amount"])
        elif kind == "HoleDealt":
            holes[data["seat"]].append(data["card"])
        elif kind == "BoardDealt":
            flush_holes()
            actions.append(f"d db {''.join(data['cards'])}")
        elif kind == "ActionApplied":
            flush_holes()
            p, act = player[data["seat"]], data["action"]
            if act["type"] == "fold":
                actions.append(f"{p} f")
            elif data.get("auto"):
                actions.append(f"{p} cc # arena:auto")
            elif data.get("nominal_big_blind_call"):
                actions.append(f"{p} cbr {data['street_committed']} # arena:nominal_bb_call")
            elif act["type"] in ("check", "call"):
                actions.append(f"{p} cc")
            else:
                actions.append(f"{p} cbr {act['amount']}")
        elif kind == "CardsRevealed":
            flush_holes()
            actions.append(f"{player[data['seat']]} sm {''.join(data['cards'])}")
        elif kind == "OddChipCorrection":
            corrected = True
        elif kind == "HandCompleted":
            final = dict(data["final_stacks"])
    if final is None:
        raise PhhError(f"hand {meta.hand_id} is not complete")
    flush_holes()

    trimming, antes = _ante_fields(meta, order)
    fields: list[tuple[str, Any]] = [
        ("variant", PHH_VARIANTS[meta.ruleset_id]),
        ("ante_trimming_status", trimming),
        ("antes", antes),
        ("blinds_or_straddles", _blind_vector(meta, order)),
        ("min_bet", meta.big_blind),
        ("starting_stacks", [meta.start_stacks[s] for s in order]),
        ("actions", actions),
        ("table", meta.table_id),
        ("hand", meta.hand_number),
        ("level", meta.level + 1),
        ("seats", [s + 1 for s in order]),
        ("seat_count", 8),
        ("players", [f"seat {s}" for s in order]),
        ("finishing_stacks", [final[s] for s in order]),
        ("_arena_compatibility_status", "practice_assumptions"),
        ("_arena_ruleset_id", meta.ruleset_id),
        ("_arena_hand_id", meta.hand_id),
        ("_arena_seat_ring", list(meta.seat_ring)),
        ("_arena_button_seat", meta.button),
        ("_arena_big_blind_seat", meta.big_blind_seat),
        ("_arena_small_blind", meta.small_blind),
        ("_arena_big_blind", meta.big_blind),
        ("_arena_short_bet_reopens_checker", bool(meta.local_rules.get("short_bet_reopens_checker", False))),
        ("_arena_odd_chip_correction", corrected),
        ("_arena_short_big_blind_correction", 0 < max(posted, default=0) < meta.big_blind),
    ]
    if meta.small_blind_seat is not None:
        fields.append(("_arena_small_blind_seat", meta.small_blind_seat))
    if meta.ante:
        fields += [("_arena_ante", meta.ante), ("_arena_ante_posting", meta.ante_posting)]
    return "".join(f"{key} = {_toml_value(value)}\n" for key, value in fields)


def hand_meta(history: TableHistory, hand_number: int) -> PhhHandMeta:
    started = history.table_started.data
    hs = next(e for e in history.events if e.type == "HandStarted" and e.data["hand_number"] == hand_number)
    d = hs.data
    return PhhHandMeta(
        table_id=history.table_id,
        hand_id=hs.hand_id or "",
        hand_number=hand_number,
        level=d["level"],
        ruleset_id=started["config"]["ruleset_id"],
        seat_ring=tuple(started["seat_ring"]),
        button=d["positions"]["button"],
        small_blind_seat=d["small_blind_seat"],
        big_blind_seat=d["positions"]["big_blind"],
        small_blind=d["small_blind"],
        big_blind=d["big_blind"],
        start_stacks=dict(d["start_stacks"]),
        local_rules=dict(started["config"]["local_rules"]),
        ante=d.get("ante", 0),
        ante_posting=d.get("ante_posting"),
    )


def export_history_phh(history: TableHistory) -> list[str]:
    """One PHH document per complete hand of the table."""
    out = []
    for e in history.events:
        if e.type != "HandStarted":
            continue
        events = history.hand_events(e.data["hand_number"])
        if any(x.type == "HandCompleted" for x in events):
            out.append(export_hand(hand_meta(history, e.data["hand_number"]), events))
    return out


def export_history_phhs(history: TableHistory) -> str:
    """All complete hands as one ``.phhs`` file (TOML tables named by hand number)."""
    parts = []
    for doc in export_history_phh(history):
        number = tomllib.loads(doc)["hand"]
        parts.append(f"[{number}]\n{doc}")
    return "\n".join(parts)


# ---------------------------------------------------------------- reading back


@dataclass(frozen=True)
class ParsedPhh:
    meta: PhhHandMeta
    order: tuple[int, ...]
    holes: dict[int, list[str]]
    boards: list[list[str]]
    actions: list[tuple[int, str, int | None, str | None]]
    shows: dict[int, list[str]]
    finishing_stacks: dict[int, int]


def _split_cards(text: str) -> list[str]:
    if len(text) % 2:
        raise PhhError(f"bad card string {text!r}")
    return [text[i : i + 2] for i in range(0, len(text), 2)]


def parse_phh(text: str) -> ParsedPhh:
    doc = tomllib.loads(text)
    ruleset = doc.get("_arena_ruleset_id")
    if ruleset not in PROFILES or PHH_VARIANTS[ruleset] != doc.get("variant"):
        raise PhhError("not an arena PHH export (missing or inconsistent _arena_ruleset_id/variant)")
    order = tuple(s - 1 for s in doc["seats"])
    stacks = dict(zip(order, doc["starting_stacks"]))
    meta = PhhHandMeta(
        table_id=doc["table"],
        hand_id=doc["_arena_hand_id"],
        hand_number=doc["hand"],
        level=doc["level"] - 1,
        ruleset_id=ruleset,
        seat_ring=tuple(doc["_arena_seat_ring"]),
        button=doc["_arena_button_seat"],
        small_blind_seat=doc.get("_arena_small_blind_seat"),
        big_blind_seat=doc["_arena_big_blind_seat"],
        small_blind=doc["_arena_small_blind"],
        big_blind=doc["_arena_big_blind"],
        start_stacks=stacks,
        local_rules={"short_bet_reopens_checker": doc["_arena_short_bet_reopens_checker"]},
        ante=doc.get("_arena_ante", 0),
        ante_posting=doc.get("_arena_ante_posting"),
    )
    if _blind_vector(meta, order) != doc["blinds_or_straddles"]:
        raise PhhError("blinds_or_straddles disagrees with the arena blind markers")
    if list(_ante_fields(meta, order)) != [doc["ante_trimming_status"], doc["antes"]]:
        raise PhhError("antes disagree with the arena ante structure")
    holes: dict[int, list[str]] = {}
    boards: list[list[str]] = []
    actions: list[tuple[int, str, int | None, str | None]] = []
    shows: dict[int, list[str]] = {}
    for raw in doc["actions"]:
        body, _, comment = raw.partition(" # ")
        words = body.split()
        note = comment.strip() or None
        if words[:2] == ["d", "dh"]:
            holes[order[int(words[2][1:]) - 1]] = _split_cards(words[3])
        elif words[:2] == ["d", "db"]:
            boards.append(_split_cards(words[2]))
        elif len(words) >= 2 and words[0].startswith("p"):
            seat = order[int(words[0][1:]) - 1]
            if words[1] == "sm":
                shows[seat] = _split_cards(words[2])
            elif words[1] in ("f", "cc"):
                actions.append((seat, words[1], None, note))
            elif words[1] == "cbr":
                actions.append((seat, "cbr", int(words[2]), note))
            else:
                raise PhhError(f"unsupported PHH action {raw!r}")
        else:
            raise PhhError(f"unsupported PHH action {raw!r}")
    return ParsedPhh(meta, order, holes, boards, actions, shows, dict(zip(order, doc["finishing_stacks"])))


def _rebuild_deck(parsed: ParsedPhh) -> tuple[str, ...]:
    """A deck that deals the recorded cards; burns and undealt cards are filler."""
    profile = get_profile(parsed.meta.ruleset_id)
    used = [c for s in parsed.order for c in parsed.holes[s]] + [c for b in parsed.boards for c in b]
    if len(set(used)) != len(used):
        raise PhhError("a card appears twice")
    filler = iter(c for c in full_deck(profile.ranks) if c not in set(used))
    deck: list[str] = []
    for r in range(profile.hole_card_count):
        deck.extend(parsed.holes[s][r] for s in parsed.order)
    for board in parsed.boards:
        deck.append(next(filler))
        deck.extend(board)
    deck.extend(filler)
    return tuple(deck)


def replay_phh(text: str) -> tuple[ParsedPhh, PokerKitHand]:
    """Re-run a PHH export through the arena engine; raise :class:`PhhError` on any mismatch."""
    parsed = parse_phh(text)
    hand = PokerKitHand(_spec(parsed.meta, _rebuild_deck(parsed)))
    for seat, verb, amount, note in parsed.actions:
        if note == "arena:auto":
            continue
        if hand.is_complete or hand.actor_seat != seat:
            raise PhhError(f"PHH action by seat {seat} does not match the engine's actor {hand.actor_seat}")
        legal = hand.legal_actions()
        if verb == "f":
            action = Action(type="fold")
        elif verb == "cc":
            action = Action(type="check") if legal.check else Action(type="call")
        elif note == "arena:nominal_bb_call":
            action = Action(type="call")
        else:
            action = Action(type="raise_to", amount=amount)
        hand.apply(action)
    if not hand.is_complete:
        raise PhhError("PHH actions end before the hand is complete")
    auto_expected = sum(1 for a in parsed.actions if a[3] == "arena:auto")
    auto_seen = sum(1 for e in hand.events if e.type == "ActionApplied" and e.data.get("auto"))
    if auto_expected != auto_seen:
        raise PhhError("automatic checks differ from the PHH record")
    if {s: list(c) for s, c in hand.revealed.items()} != parsed.shows:
        raise PhhError("revealed cards differ from the PHH record")
    if hand.final_stacks() != parsed.finishing_stacks:
        raise PhhError("finishing stacks differ from the PHH record")
    return parsed, hand


def reexport(parsed: ParsedPhh, hand: PokerKitHand) -> str:
    events = [HandEvent("HandStarted", {})] + list(hand.events)
    return export_hand(parsed.meta, events)
