"""The build-day reveal slot: a ruleset whose rules come from a variant spec file.

At the competition the variant may be announced only on build day. The arena plays it
under the single ruleset id :data:`~arena.domain.models.REVEAL_RULESET_ID`; what that id
means is read from ``$ARENA_REVEAL_VARIANT/variant.json``, a variant description in the
:class:`strategy.rules.variants.VariantSpec` dictionary form (deck ranks and suits, hole
cards, board streets, burns, showdown selection, category order, lowest straight and
betting). Only one reveal variant is active per process, and the rules hash of every
table records which one it was.

The builder assembles a PokerKit game from PokerKit's own parts (a lookup whose entries
follow the declared category order, a combination or exactly-two-hole hand, a hold'em
street structure with the declared hole-card count, the no-limit or pot-limit mixin), so
settlement still goes through the same adapter and its checks. Anything the parts
cannot express raises :class:`UnsupportedRules` before a hand is dealt; nothing falls
back to a practice profile.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pokerkit
from pokerkit import Deck, RankOrder
from pokerkit.hands import CombinationHand, HoleBoardCombinationHand
from pokerkit.lookups import Label, Lookup

from arena.domain.models import REVEAL_ENV, REVEAL_RULESET_ID
from arena.rules.profiles import RuleProfile

ENV = REVEAL_ENV
"""Directory holding ``variant.json`` (and, for bots, the reveal tables)."""
VARIANT_FILE = "variant.json"
SUPPORTED_SCHEMA = 1

_DECKS = {"23456789TJQKA": (RankOrder.STANDARD, Deck.STANDARD),
          "6789TJQKA": (RankOrder.SHORT_DECK_HOLDEM, Deck.SHORT_DECK_HOLDEM)}
_CATEGORIES = {
    "high_card": ("multiset", {1: 5}, False, Label.HIGH_CARD),
    "one_pair": ("multiset", {2: 1, 1: 3}, False, Label.ONE_PAIR),
    "two_pair": ("multiset", {2: 2, 1: 1}, False, Label.TWO_PAIR),
    "three_of_a_kind": ("multiset", {3: 1, 1: 2}, False, Label.THREE_OF_A_KIND),
    "straight": ("straight", None, False, Label.STRAIGHT),
    "flush": ("multiset", {1: 5}, True, Label.FLUSH),
    "full_house": ("multiset", {3: 1, 2: 1}, False, Label.FULL_HOUSE),
    "four_of_a_kind": ("multiset", {4: 1, 1: 1}, False, Label.FOUR_OF_A_KIND),
    "straight_flush": ("straight", None, True, Label.STRAIGHT_FLUSH),
}


class UnsupportedRules(ValueError):
    """The revealed variant needs something the engine cannot build."""


def reveal_dir() -> Path | None:
    raw = os.environ.get(ENV)
    return Path(raw) if raw else None


def load_variant_dict(directory: Path | None = None) -> dict[str, Any]:
    d = directory or reveal_dir()
    if d is None:
        raise UnsupportedRules(f"{REVEAL_RULESET_ID} needs ${ENV} pointing at a directory with {VARIANT_FILE}")
    path = Path(d) / VARIANT_FILE
    if not path.exists():
        raise UnsupportedRules(f"no reveal variant at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _lookup_class(rank_order: RankOrder, categories: tuple[str, ...], wrap: bool) -> type[Lookup]:
    def _add_entries(self: Lookup) -> None:
        for name in categories:
            kind, counts, suited, label = _CATEGORIES[name]
            if kind == "straight":
                self._add_straights(5, (suited,), label, wrap)
            else:
                self._add_multisets(Counter(counts), (suited,), label)

    from dataclasses import dataclass

    return dataclass(type("RevealLookup", (Lookup,), {"rank_order": rank_order, "_add_entries": _add_entries}))


def build_profile(data: Mapping[str, Any]) -> RuleProfile:
    """A :class:`RuleProfile` for the variant ``data`` describes, or :class:`UnsupportedRules`."""
    if int(data.get("schema_version", SUPPORTED_SCHEMA)) != SUPPORTED_SCHEMA:
        raise UnsupportedRules(f"variant schema_version {data.get('schema_version')!r}")
    ranks, suits = str(data["ranks"]), str(data["suits"])
    if suits != "cdhs" or ranks not in _DECKS:
        raise UnsupportedRules(f"deck {ranks}/{suits}: the engine deals the 52-card or the 36-card (6-A) deck")
    if list(data.get("streets", ["preflop", "flop", "turn", "river"])) != ["preflop", "flop", "turn", "river"] \
            or list(data["board_streets"]) != [3, 1, 1] or not data["burn_before_board"]:
        raise UnsupportedRules("the engine deals hold'em streets only: flop 3, turn 1, river 1, a burn before each")
    hole = int(data["hole_cards"])
    if not 2 <= hole <= 5:
        raise UnsupportedRules(f"{hole} hole cards: the engine supports 2 to 5")
    categories = tuple(data["category_order"])
    if sorted(categories) != sorted(_CATEGORIES):
        raise UnsupportedRules(f"category_order must order exactly {sorted(_CATEGORIES)}")
    rank_order, deck = _DECKS[ranks]
    lowest = str(data["lowest_straight"])
    if lowest == ranks[-1] + ranks[:4]:
        wrap = True
    elif lowest == ranks[:5]:
        wrap = False
    else:
        raise UnsupportedRules(f"lowest straight {lowest!r} is neither the wheel nor the five lowest ranks")
    lookup = _lookup_class(rank_order, categories, wrap)()
    selection = data["hand_selection"]
    if selection == "best_five_of_seven":
        hand = type("RevealHand", (CombinationHand,), {"lookup": lookup, "low": False, "card_count": 5})
    elif selection == "two_hole_three_board":
        hand = type("RevealHand", (HoleBoardCombinationHand,), {"lookup": lookup, "low": False, "card_count": 5,
                                                                "board_card_count": 3, "hole_card_count": 2})
    else:
        raise UnsupportedRules(f"hand selection {selection!r}")
    betting = data["betting"]
    mixin = {"no_limit": pokerkit.games.NoLimitPokerMixin, "pot_limit": pokerkit.games.PotLimitPokerMixin}.get(betting)
    if mixin is None:
        raise UnsupportedRules(f"betting {betting!r}: the engine supports no_limit and pot_limit")
    game = type("RevealGame", (mixin, pokerkit.games.UnfixedLimitHoldem),
                {"deck": deck, "hand_types": (hand,), "hole_dealing_count": hole})
    return RuleProfile(ruleset_id=REVEAL_RULESET_ID, title=str(data.get("title", "Revealed variant")),
                       game_class=game, ranks=ranks, hole_card_count=hole, betting=betting,
                       hand_selection=selection, five_card_hand=type(
                           "RevealFive", (CombinationHand,), {"lookup": lookup, "low": False, "card_count": 5}))


_CACHE: dict[str, RuleProfile] = {}


def reveal_profile() -> RuleProfile:
    """The active reveal variant's profile (cached by the spec's content)."""
    data = load_variant_dict()
    key = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    profile = _CACHE.get(key)
    if profile is None:
        profile = _CACHE[key] = build_profile(data)
    return profile
