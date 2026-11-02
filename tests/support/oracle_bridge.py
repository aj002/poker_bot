"""Cross-check completed engine hands against the independent oracles.

Hand strength comes from the oracle evaluators (never PokerKit), and the pot
ledger is re-settled with ``tests.oracles.pots.settle`` under the spec rules:
one pot per contribution level, odd chips one each clockwise after the button.
"""

from __future__ import annotations

from arena.rules.pokerkit_adapter import PokerKitHand
from tests.oracles import cards as ocards
from tests.oracles.omaha import best_omaha
from tests.oracles.pots import settle
from tests.oracles.shortdeck import best_shortdeck
from tests.oracles.standard import best_nlhe


def oracle_strength(ruleset_id: str, hole: list[str], board: list[str]):
    h = ocards.parse_cards(" ".join(hole))
    b = ocards.parse_cards(" ".join(board))
    if ruleset_id == "practice_nlhe":
        return best_nlhe(h, b).value
    if ruleset_id == "uncertainty_shortdeck":
        return best_shortdeck(h, b).value
    if ruleset_id == "uncertainty_plo4":
        return best_omaha(h, b).value
    raise ValueError(ruleset_id)


def oracle_antes(hand: PokerKitHand) -> dict[int, int]:
    """Antes each seat should post, re-derived from the spec: classic antes before the blinds,
    a big-blind ante after the big blind."""
    spec = hand.spec
    if not spec.ante:
        return {}
    if spec.ante_posting == "classic":
        return {s: spec.ante if v >= spec.ante else v for s, v in spec.stacks.items()}
    behind = spec.stacks[spec.big_blind_seat] - spec.big_blind
    return {spec.big_blind_seat: spec.ante if behind >= spec.ante else max(behind, 0)}


def oracle_dead_money(hand: PokerKitHand) -> dict[int, int]:
    return oracle_antes(hand) if hand.spec.ante_posting == "big_blind" else {}


def assert_matches_oracles(hand: PokerKitHand) -> None:
    result = hand.result()
    posted = {e.data["seat"]: e.data["amount"] for e in hand.events if e.type == "AntePosted"}
    assert posted == {s: a for s, a in oracle_antes(hand).items() if a}, f"antes {posted} != oracle"
    live = [s for s in hand.seats if s not in result.folded]
    engine_pots = [(p.amount, tuple(sorted(p.eligible)), dict(sorted(p.awards.items()))) for p in result.pots]
    if len(live) == 1:
        # Fold win: PokerKit returns the winner's uncalled street bet before collection,
        # so the whole collected pot goes to the sole live seat and no card is shown.
        total = sum(result.contributions.values())
        assert engine_pots == ([(total, (live[0],), {live[0]: total})] if total else [])
        expected_final = {
            s: result.starting_stacks[s] - result.contributions[s] + (total if s == live[0] else 0)
            for s in hand.seats
        }
        assert result.final_stacks == expected_final
        assert not result.revealed, "fold win must not reveal cards"
        return
    assert len(result.board) == 5, "showdown without a full board"
    strengths = {s: oracle_strength(hand.profile.ruleset_id, hand.hole_cards[s], list(result.board)) for s in live}
    settlement = settle(result.contributions, result.folded, strengths, hand.spec.button, dead=oracle_dead_money(hand))
    oracle_pots = [(p.amount, tuple(sorted(p.eligible)), dict(sorted(p.shares.items()))) for p in settlement.pots]
    assert engine_pots == oracle_pots, f"pots differ from oracle:\nengine {engine_pots}\noracle {oracle_pots}"
    expected_final = {
        s: result.starting_stacks[s] - result.contributions[s] + settlement.totals.get(s, 0) for s in hand.seats
    }
    assert result.final_stacks == expected_final, f"final stacks {result.final_stacks} != oracle {expected_final}"
    assert sorted(result.revealed) == sorted(live), "every live hand is revealed at showdown"
