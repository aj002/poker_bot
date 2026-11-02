"""Canonical game state at stable boundaries, and its hash.

The hashed state contains everything that decides what happens next: stacks,
commitments, statuses, every private card and the undealt deck order, the
betting-rights overlay, the legal-action descriptor, standings and blinds. It
excludes table/decision IDs, state versions and timestamps, so a replay with
freshly assigned IDs reproduces the same hashes. If a PokerKit upgrade changed
a legal action or a settlement, the hash changes and replay halts.

These hashes are private: with few unknown cards a hash of private state could
be brute-forced, so they never appear in seat or public projections.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from arena.domain.canonical import canonical_hash

if TYPE_CHECKING:
    from arena.table.service import Table

STATE_HASH_DOMAIN = "arena-state-v1"
HAND_START_DOMAIN = "arena-hand-start-v1"


def _standings(table: Table) -> dict[int, list[int]]:
    return {seat: list(p.positions) for seat, p in sorted(table.standings.placements.items())}


def hand_start_state(table: Table, *, hand_number: int, positions: dict, blinds: tuple[int, int],
                     start_stacks: dict[int, int], deck: tuple[str, ...],
                     ante: tuple[int, str | None] = (0, None)) -> dict[str, Any]:
    state = {
        "config_hash": table.config_hash,
        "hand_number": hand_number,
        "positions": positions,
        "blinds": list(blinds),
        "table_stacks": dict(sorted(table.stacks.items())),
        "start_stacks": dict(sorted(start_stacks.items())),
        "deck": list(deck),
        "standings": _standings(table),
    }
    if ante[0]:
        state["ante"] = list(ante)
    return state


def decision_state(table: Table) -> dict[str, Any]:
    hand = table.hand
    assert hand is not None and table.actor_seat is not None
    record = table.hands[-1]
    ledger = hand.card_ledger()
    state = {
        "config_hash": table.config_hash,
        "hand_number": record.hand_number,
        "positions": record.positions.as_dict(),
        "blinds": list(record.blinds),
        "standings": _standings(table),
        "street": hand.street,
        "actor": table.actor_seat,
        "level": hand.current_level(),
        "players": [
            {
                "seat": p.seat,
                "stack": p.stack,
                "street_committed": p.street_committed,
                "hand_committed": p.hand_committed,
                "status": p.status,
            }
            for p in sorted(hand.players(), key=lambda p: p.seat)
        ],
        "collected": dict(sorted(hand.collected().items())),
        "folded": sorted(hand.folded),
        "hole": {s: list(c) for s, c in sorted(hand.hole_cards.items())},
        "board": list(hand.board),
        "burns": list(hand.burns),
        "undealt": list(ledger["undealt"]),
        "betting_rights": hand.betting_rights(),
        "legal": hand.legal_actions().model_dump(),
    }
    if record.ante:
        state["ante"] = [record.ante, record.ante_posting]
        state["dead_money"] = dict(sorted(hand.dead_money().items()))
    return state


def decision_hash(table: Table) -> str:
    return canonical_hash(decision_state(table), domain=STATE_HASH_DOMAIN)


def hand_start_hash(state: dict[str, Any]) -> str:
    return canonical_hash(state, domain=HAND_START_DOMAIN)
