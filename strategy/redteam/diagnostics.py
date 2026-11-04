"""Per-table behaviour diagnostics from a table's own event log, for attributing a loss to a red-team behaviour.

Everything is read from the public events plus the final stacks (no hidden cards). For one
table it returns a small JSON-friendly dict of counts and sums that add up exactly across
tables, so any slice can be pooled and bootstrapped over tables later:

``phase_eq``
    the hero's points-equity change over hands 1-50, 51-125 and 126-200 (the points the
    scorer would award if the table stopped at that hand; the three parts sum to the
    table's points minus the 12.5 start).
``alive_eq`` / ``alive_hands``
    the same equity change summed by how many players were alive at the start of the hand
    (``8-6``, ``5``, ``4`` = the bubble, ``3``, ``2``): where in the tournament the hero's
    points were won or lost.
``agg``
    ``tag -> class -> count`` of every non-automatic raise or bet by each opponent tag,
    classed ``pre_open``, ``pre_3bet``, ``pre_4bet`` (4-bet or more), ``post_small`` (up to
    half the pot), ``post_mid`` (up to the pot), ``post_over`` (more than the pot after
    calling, an overbet) and ``shove`` (all-in raise), plus ``hands`` (hands dealt) and
    ``open_bb`` (sum of first-raise sizes in big blinds, with ``opens``).
``resp``
    ``tag|class -> [fold, call, raise]``: the hero's response when it faced that class of
    aggression from that tag and had to act.
``pots``
    hero chips (in big blinds of the hand) in the hands where the hero was live when a
    red-team seat overbet postflop (``overbet``), when one shoved (``shove``), and in
    hands where the hero saw a flop without either (``other_flop``): ``[hands, sum_bb]``.
``image``
    the same ``[hands, sum_bb]`` in hands where an adaptive opponent was dealt in with the
    hero, split by that opponent's image phase (``tight`` / ``loose``).
``bandagg``
    ``band|dealt`` red-team seat-hands and ``band|agg`` their preflop 3-bets, 4-bets and
    shoves, by players alive at the hand's start.
``bust``
    the hero's elimination: hand, players alive at its start, and the tag that won most
    of the chips it lost (``None`` when it survived).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

PHASES = (50, 125, 10 ** 9)
ALIVE_BANDS = ("8-6", "5", "4", "3", "2")
AGG_CLASSES = ("pre_open", "pre_3bet", "pre_4bet", "post_small", "post_mid", "post_over", "shove")
HALF, ONE = 0.5, 1.0


def alive_band(alive: int) -> str:
    return "8-6" if alive >= 6 else str(max(alive, 2))


def _classify(street: str, raises_before: int, ratio: float, shove: bool) -> str:
    if shove:
        return "shove"
    if street == "preflop":
        return "pre_open" if raises_before <= 1 else "pre_3bet" if raises_before == 2 else "pre_4bet"
    return "post_small" if ratio <= HALF else "post_mid" if ratio <= ONE + 1e-9 else "post_over"


def adaptive_phase(phase0: int, hand_number: int) -> str:
    """The adaptive bot's image in a hand (see ``redteam/adaptive/bot.py``): 0 tight, 1 loose."""
    return "loose" if (phase0 + (hand_number - 1) // 18) % 2 else "tight"


def table_diagnostics(events: Sequence[Mapping[str, Any]], hero: int, tags: Mapping[int, str], *,
                      phase0: Mapping[int, int] | None = None) -> dict[str, Any]:
    """``tags`` maps every seat to its tag (the hero's too); ``phase0`` maps adaptive seats to their image offset."""
    from strategy.objective.points import rescore

    phase0 = phase0 or {}
    payouts: list[int] = []
    ring: list[int] = []
    start0: dict[int, int] = {}
    cap = 0
    hands: list[tuple[int, dict[int, int], dict[int, int]]] = []
    hand: dict[str, Any] | None = None
    agg: dict[str, Counter] = defaultdict(Counter)
    resp: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    bandagg: Counter = Counter()
    per_hand: list[dict[str, Any]] = []
    bust: dict[str, Any] | None = None

    for e in events:
        kind, d = e["type"], e["data"]
        if kind == "TableStarted":
            payouts, ring = list(d["config"]["payout_points"]), list(d["seat_ring"])
            cap = int(d["config"]["hand_cap"])
            start0 = {int(s): int(v) for s, v in d["start_stacks"].items()}
        elif kind == "HandStarted":
            start = {int(s): int(v) for s, v in d["start_stacks"].items()}
            hand = {"n": int(d["hand_number"]), "start": start, "bb": int(d["big_blind"]),
                    "committed": dict.fromkeys(start, 0), "street_bets": dict.fromkeys(start, 0), "street": "preflop",
                    "folded": set(), "raises": 0, "last_agg": None, "classes": set(), "hero_live_over": False,
                    "hero_live_shove": False, "hero_flop": False, "dealt": sorted(start), "agg_count": 0}
            hand["band"] = alive_band(sum(1 for v in start.values() if v > 0))
            for s in start:
                agg[tags[s]]["hands"] += 1
                if tags[s].startswith("rt-") and start[s] > 0:
                    bandagg[f"{hand['band']}|dealt"] += 1
        elif hand is not None and kind in ("BlindPosted", "AntePosted"):
            s = int(d["seat"])
            hand["committed"][s] += int(d["amount"])
            if kind == "BlindPosted":
                hand["street_bets"][s] += int(d["amount"])
        elif hand is not None and kind == "BoardDealt":
            hand["street"] = d["street"]
            hand["street_bets"] = dict.fromkeys(hand["street_bets"], 0)
            hand["raises"] = 0
            hand["last_agg"] = None
            if hero not in hand["folded"] and hero in hand["start"]:
                hand["hero_flop"] = True if d["street"] == "flop" else hand["hero_flop"]
        elif hand is not None and kind == "ExcessReturned":
            hand["committed"][int(d["seat"])] -= int(d["amount"])
        elif hand is not None and kind == "ActionApplied" and not d.get("auto"):
            s = int(d["seat"])
            act = d["action"]
            sb = hand["street_bets"]
            top = max(sb.values())
            to_call = top - sb[s]
            pot = sum(hand["committed"].values())
            stack_before = hand["start"][s] - hand["committed"][s]
            if act["type"] == "fold":
                hand["folded"].add(s)
            if s == hero and to_call > 0 and hand["last_agg"] is not None:
                tag, cls = hand["last_agg"]
                resp[f"{tag}|{cls}"][{"fold": 0, "call": 1}.get(act["type"], 2)] += 1
            if act["type"] == "raise_to":
                new_top = int(d["street_committed"])
                if hand["street"] == "preflop":
                    ratio = 0.0
                else:
                    base = pot + to_call if top > 0 else pot
                    ratio = (new_top - top) / base if base else 0.0
                shove = int(d["added"]) >= stack_before
                hand["raises"] += 1
                cls = _classify(hand["street"], hand["raises"], ratio, shove)
                tag = tags[s]
                if s != hero:
                    agg[tag][cls] += 1
                    if tag.startswith("rt-") and cls in ("pre_3bet", "pre_4bet", "shove") \
                            and hand["street"] == "preflop":
                        bandagg[f"{hand['band']}|agg"] += 1
                    if cls == "pre_open":
                        agg[tag]["opens"] += 1
                        agg[tag]["open_bb"] += new_top / hand["bb"]
                    if tag.startswith("rt-") and hero not in hand["folded"] and hero in hand["start"]:
                        if cls == "post_over":
                            hand["hero_live_over"] = True
                        if cls == "shove":
                            hand["hero_live_shove"] = True
                    hand["last_agg"] = (tag, cls)
                else:
                    hand["last_agg"] = None
            hand["committed"][s] += int(d["added"])
            sb[s] = int(d["street_committed"])
        elif kind == "PlayerEliminated" and int(d["seat"]) == hero:
            bust = {"hand": int(d["hand_number"]), "alive": None, "by": None}
        elif kind == "PotAwarded" and hand is not None:
            hand.setdefault("awards", Counter()).update({int(s): int(v) for s, v in d["awards"].items()})
        elif kind == "HandCompleted" and hand is not None:
            after = {int(s): int(v) for s, v in d["final_stacks"].items()}
            hands.append((hand["n"], hand["start"], after))
            per_hand.append({"n": hand["n"], "start": hand["start"], "after": after, "bb": hand["bb"],
                             "over": hand["hero_live_over"], "shove": hand["hero_live_shove"],
                             "flop": hand["hero_flop"], "awards": dict(hand.get("awards", {})),
                             "folded": sorted(hand["folded"])})
            hand = None

    equity = [rescore(payouts, ring, start0, [h for h in hands if h[0] <= b], cap).points(hero)
              for b in range(len(hands) + 1)] if hands else [0.0]
    out_phase = [0.0, 0.0, 0.0]
    alive_eq: Counter = Counter()
    alive_n: Counter = Counter()
    pots: dict[str, list[float]] = {"overbet": [0, 0.0], "shove": [0, 0.0], "other_flop": [0, 0.0]}
    image: dict[str, list[float]] = {"tight": [0, 0.0], "loose": [0, 0.0]}
    for i, h in enumerate(per_hand):
        if hero not in h["start"]:
            continue
        delta = equity[i + 1] - equity[i]
        out_phase[0 if h["n"] <= PHASES[0] else 1 if h["n"] <= PHASES[1] else 2] += delta
        band = alive_band(sum(1 for v in h["start"].values() if v > 0))
        alive_eq[band] += delta
        alive_n[band] += 1
        chips_bb = (h["after"][hero] - h["start"][hero]) / h["bb"]
        key = "overbet" if h["over"] else "shove" if h["shove"] else "other_flop" if h["flop"] else None
        if key:
            pots[key][0] += 1
            pots[key][1] += chips_bb
        for s, p0 in phase0.items():
            if s in h["start"] and s != hero:
                row = image[adaptive_phase(p0, h["n"])]
                row[0] += 1
                row[1] += chips_bb
        if bust is not None and h["n"] == bust["hand"]:
            bust["alive"] = sum(1 for v in h["start"].values() if v > 0)
            lost = h["start"][hero]
            winners = Counter({s: v for s, v in h["awards"].items() if s != hero})
            bust["by"] = tags[winners.most_common(1)[0][0]] if winners and lost else None
    return {"phase_eq": out_phase, "alive_eq": dict(alive_eq), "alive_hands": dict(alive_n),
            "agg": {t: dict(c) for t, c in agg.items() if t != tags[hero]}, "resp": dict(resp), "pots": pots,
            "image": image, "bust": bust, "hands": len(per_hand), "bandagg": dict(bandagg)}


def merge_counts(rows: Sequence[Mapping[str, Any]], path: Callable[[Mapping[str, Any]], Mapping[str, float]]
                 ) -> dict[str, float]:
    out: Counter = Counter()
    for r in rows:
        out.update(path(r))
    return dict(out)
