"""Generated hands and invariant checks shared by the fast and release gates."""

from __future__ import annotations

import dataclasses

from arena.domain.cards import full_deck
from arena.domain.models import Action, LocalRules, default_blind_levels
from arena.domain.rng import SeededRng, shuffled
from arena.rules.base import IllegalAction
from arena.rules.pokerkit_adapter import HandSpec, PokerKitHand
from arena.rules.profiles import get_profile
from arena.table.driver import legal_random
from arena.table.positions import next_positions, positions_from_button
from tests.support.oracle_bridge import assert_matches_oracles

RULESETS = ("practice_nlhe", "uncertainty_shortdeck", "uncertainty_plo4")


def awkward_stack(rng: SeededRng, bb: int) -> int:
    kind = rng.randbelow(8)
    if kind == 0:
        return 1
    if kind == 1:
        return 1 + rng.randbelow(bb)
    if kind == 2:
        return bb
    if kind == 3:
        return bb + 1 + rng.randbelow(2 * bb)
    if kind == 4:
        return 5 * bb + rng.randbelow(10 * bb)
    return 1 + rng.randbelow(6000)


def random_spec(seed: int, ruleset: str, n: int, ante_posting: str | None = None) -> HandSpec:
    """``ante_posting`` adds an ante drawn from its own stream, so ante-free specs are unchanged."""
    spec = _random_spec(seed, ruleset, n)
    if ante_posting is None:
        return spec
    arng = SeededRng(seed, f"gen-ante-{ruleset}-{n}-{ante_posting}")
    bb = spec.big_blind
    ante = (1 + arng.randbelow(bb // 4)) if ante_posting == "classic" else (bb // 2 + arng.randbelow(bb))
    return dataclasses.replace(spec, ante=ante, ante_posting=ante_posting)


def _random_spec(seed: int, ruleset: str, n: int) -> HandSpec:
    rng = SeededRng(seed, f"gen-{ruleset}-{n}")
    profile = get_profile(ruleset)
    dead = rng.randbelow(3) if n >= 3 and n < 8 else 0
    dead = min(dead, 8 - n)
    ring = sorted(shuffled(list(range(8)), rng)[: n + dead])
    live = sorted(shuffled(ring, rng)[:n])
    sb, bb = default_blind_levels()[rng.randbelow(8)]
    if dead:
        # The previous hand had every ring seat live; the dead seats busted on it.
        prev = positions_from_button(ring, ring, ring[rng.randbelow(len(ring))])
        pos = next_positions(prev, ring, live)
    else:
        pos = positions_from_button(ring, live, live[rng.randbelow(n)])
    stacks = {s: awkward_stack(rng, bb) for s in live}
    return HandSpec(
        profile=profile,
        seat_ring=tuple(ring),
        stacks=stacks,
        button=pos.button,
        small_blind_seat=pos.small_blind_seat(live),
        big_blind_seat=pos.big_blind,
        small_blind=sb,
        big_blind=bb,
        deck=tuple(shuffled(full_deck(profile.ranks), rng)),
        local_rules=LocalRules(short_bet_reopens_checker=bool(rng.randbelow(2))),
    )


def check_invariants(hand: PokerKitHand) -> None:
    players = hand.players()
    for p in players:
        assert type(p.stack) is int and p.stack >= 0
        assert p.street_committed >= 0
    assert hand.chips_in_play() == hand.total_chips
    ledger = hand.card_ledger()
    cards = [c for part in ledger.values() for c in part]
    assert sorted(cards) == sorted(hand.deck) and len(set(cards)) == len(cards)
    assert all(c[0] in hand.profile.ranks for c in cards)
    if hand.is_complete:
        return
    actor = hand.actor_seat
    assert actor is not None
    me = next(p for p in players if p.seat == actor)
    assert me.status == "active" and me.stack > 0 and actor not in hand.folded
    legal = hand.legal_actions()
    assert legal.check != legal.fold
    if legal.call_amount is not None:
        assert 0 < legal.call_amount <= me.stack
    if legal.min_full_raise_to is not None:
        assert legal.min_full_raise_to <= legal.max_raise_to <= me.stack + me.street_committed
    if legal.short_all_in_raise_to is not None:
        assert legal.short_all_in_raise_to == me.stack + me.street_committed
    assert legal.can_raise == (legal.min_full_raise_to is not None or legal.short_all_in_raise_to is not None)


def illegal_probe(hand: PokerKitHand, rng: SeededRng) -> Action:
    legal = hand.legal_actions()
    candidates = []
    if not legal.check:
        candidates.append(Action(type="check"))
    if legal.call_amount is None:
        candidates.append(Action(type="call"))
    if not legal.fold:
        candidates.append(Action(type="fold"))
    if legal.min_full_raise_to is not None and legal.min_full_raise_to - 1 != legal.short_all_in_raise_to:
        if legal.min_full_raise_to > 1:
            candidates.append(Action(type="raise_to", amount=legal.min_full_raise_to - 1))
    if legal.max_raise_to is not None:
        candidates.append(Action(type="raise_to", amount=legal.max_raise_to + 1))
    if not legal.can_raise:
        candidates.append(Action(type="raise_to", amount=10**6))
    return candidates[rng.randbelow(len(candidates))]


def play_generated_hand(seed: int, ruleset: str, n: int, stats: dict[str, int] | None = None,
                        ante_posting: str | None = None) -> PokerKitHand:
    spec = random_spec(seed, ruleset, n, ante_posting)
    hand = PokerKitHand(spec)
    rng = SeededRng(seed, "policy")
    street_order = ["preflop", "flop", "turn", "river"]
    last_street = 0
    submitted = 0
    check_invariants(hand)
    while not hand.is_complete:
        if rng.randbelow(5) == 0:
            probe = illegal_probe(hand, rng)
            before = (hand.actor_seat, hand.players(), len(hand.events))
            try:
                hand.apply(probe)
            except IllegalAction:
                pass
            else:
                raise AssertionError(f"illegal probe accepted: {probe}")
            assert (hand.actor_seat, hand.players(), len(hand.events)) == before
            if stats is not None:
                stats["illegal_probes_rejected"] = stats.get("illegal_probes_rejected", 0) + 1
        action = legal_random(hand.actor_seat, hand.legal_actions(), rng)
        hand.apply(action)
        submitted += 1
        if hand.street is not None:
            idx = street_order.index(hand.street)
            assert idx >= last_street
            last_street = idx
        check_invariants(hand)
    manual = [e for e in hand.events if e.type == "ActionApplied" and "auto" not in e.data]
    assert len(manual) == submitted
    result = hand.result()
    assert sum(result.final_stacks.values()) == sum(result.starting_stacks.values())
    assert_matches_oracles(hand)
    return hand
