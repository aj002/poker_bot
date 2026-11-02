"""The three bounded rule scenarios (spec section 3.2), plus the build-day reveal slot. Practice assumptions only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pokerkit

from arena.domain.cards import RANKS, SHORT_DECK_RANKS
from arena.domain.models import REVEAL_RULESET_ID

ShortDeckRanking = Literal["pokerkit_native"]


@dataclass(frozen=True)
class RuleProfile:
    ruleset_id: str
    title: str
    game_class: type[pokerkit.Poker]
    ranks: str
    hole_card_count: int
    betting: Literal["no_limit", "pot_limit"]
    hand_selection: Literal["best_five_of_seven", "two_hole_three_board"]
    # PokerKit's ShortDeckHoldemLookup orders ... trips < straight < full house < flush < quads.
    # That already matches the spec; a custom layer is only needed if a different order is chosen.
    short_deck_ranking: ShortDeckRanking | None = None
    five_card_hand: type[pokerkit.Hand] | None = None
    """Five-card evaluator for conformance probes (None: the practice default for the ruleset)."""

    @property
    def deck_size(self) -> int:
        return len(self.ranks) * 4


PROFILES: dict[str, RuleProfile] = {
    "practice_nlhe": RuleProfile(
        ruleset_id="practice_nlhe",
        title="No-limit Texas Hold'em",
        game_class=pokerkit.NoLimitTexasHoldem,
        ranks=RANKS,
        hole_card_count=2,
        betting="no_limit",
        hand_selection="best_five_of_seven",
    ),
    "uncertainty_shortdeck": RuleProfile(
        ruleset_id="uncertainty_shortdeck",
        title="Short-deck Hold'em (uncertainty scenario)",
        game_class=pokerkit.NoLimitShortDeckHoldem,
        ranks=SHORT_DECK_RANKS,
        hole_card_count=2,
        betting="no_limit",
        hand_selection="best_five_of_seven",
        short_deck_ranking="pokerkit_native",
    ),
    "uncertainty_plo4": RuleProfile(
        ruleset_id="uncertainty_plo4",
        title="Four-card pot-limit Omaha (uncertainty scenario)",
        game_class=pokerkit.PotLimitOmahaHoldem,
        ranks=RANKS,
        hole_card_count=4,
        betting="pot_limit",
        hand_selection="two_hole_three_board",
    ),
}


def get_profile(ruleset_id: str) -> RuleProfile:
    if ruleset_id == REVEAL_RULESET_ID:
        from arena.rules.reveal import reveal_profile

        return reveal_profile()
    try:
        return PROFILES[ruleset_id]
    except KeyError:
        raise ValueError(f"unsupported ruleset {ruleset_id!r}") from None
