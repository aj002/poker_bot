"""Proof of concept: replay HandHQ online NLHE hands (PHH dataset) in the arena engine.

HandHQ is 21.6 million anonymised real-money cash-game hands from six 2009 online sites
(25NL to 1000NL, two to ten handed), stored in the same phh-dataset repository as the
Pluribus hands, as ``.phhs`` files of about 1,000 hands each. They differ from the
Pluribus records in four ways, each handled here before the strict replay of
:mod:`strategy.data.phh_import`:

- amounts are dollars, converted exactly to integer cents;
- stacks are unequal, so the table is built with per-seat ``start_stacks``;
- hole cards are ``????`` unless shown. Shown cards are used; the rest are filled with
  placeholder cards drawn deterministically from the unused deck. Placeholders decide
  nothing except a showdown where a loser mucked, so the payoff is only checked when
  every player who reached showdown showed;
- there are no ``finishing_stacks``; ``winnings`` is the pot each player collected after
  rake, so the check is that engine net minus recorded net is the rake: never negative,
  and on the winners only.

Only six-handed hands at six-max tables with plain blinds are replayed, the shape the
importer and Pluribus already use.

    python -m strategy.data.handhq PATH.phhs [PATH.phhs ...] [--limit N] [--out report.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from arena.domain.cards import full_deck
from strategy.rules.variants import get_variant
from strategy.data.phh_import import (SIX_MAX, ImportedHand, ImportRejected, PhhHand, Provenance, _replay,
                                      parse_actions)

DATASET = "phh-dataset/handhq"
REPO_URL = "https://github.com/uoftcprg/phh-dataset"
COMMIT = "e47fbd5816372360bade4de5d712346fe1bb70f6"
LICENCE = "MIT (phh-dataset GitHub repository); CC BY 4.0 (Zenodo record 10.5281/zenodo.10796885)"
CITATION = "HandHQ anonymised hand histories, 2009-07-01 to 2009-07-23, converted to PHH by the phh-dataset project"


def cents(x: Any) -> int:
    c = Decimal(str(x)) * 100
    if c != c.to_integral_value():
        raise ImportRejected("amounts", f"{x} is not a whole number of cents")
    return int(c)


def load_phhs(path: Path) -> dict[str, dict[str, Any]]:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ImportRejected("toml", str(exc)) from None


@dataclass
class Converted:
    phh: PhhHand
    known: frozenset[int]
    """Players (zero-based) whose hole cards the record shows."""
    winnings: tuple[int, ...] | None
    showdown_known: bool
    """Every player still in at the end showed, or only one player was left."""


def convert(doc: dict[str, Any], key: str) -> Converted:
    """A HandHQ record as an integer-chip :class:`PhhHand` the strict replay accepts."""
    try:
        n = len(doc["starting_stacks"])
        variant, antes, blinds = str(doc["variant"]), doc["antes"], doc["blinds_or_straddles"]
        actions = [str(a) for a in doc["actions"]]
    except (KeyError, TypeError) as exc:
        raise ImportRejected("fields", f"missing or malformed field: {exc}") from None
    if variant != "NT":
        raise ImportRejected("variant", variant)
    if n != SIX_MAX or doc.get("seat_count") != SIX_MAX:
        raise ImportRejected("not_six_max", f"{n} players at a {doc.get('seat_count')}-seat table")
    if any(antes) or any(blinds[2:]):
        raise ImportRejected("blinds", f"antes {antes}, blinds_or_straddles {blinds}")

    board: list[str] = []
    shown: dict[int, str] = {}
    folded: set[int] = set()
    for raw in actions:
        body = raw.split()
        if body[:2] == ["d", "db"]:
            board += [body[2][i:i + 2] for i in range(0, len(body[2]), 2)]
        elif len(body) >= 2 and body[1] == "f":
            folded.add(int(body[0][1:]) - 1)
        elif len(body) == 3 and body[1] == "sm" and "?" not in body[2]:
            shown[int(body[0][1:]) - 1] = body[2]
    live = set(range(n)) - folded
    known = frozenset(shown)
    used = set(board) | {c[i:i + 2] for c in shown.values() for i in (0, 2)}
    spare = [c for c in full_deck(get_variant("practice_nlhe").ranks) if c not in used]
    random.Random(hashlib.sha256(key.encode()).digest()).shuffle(spare)

    out = []
    for raw in actions:
        body = raw.split()
        if body[:2] == ["d", "dh"]:
            p = int(body[2][1:]) - 1
            cards = shown.get(p) or spare.pop() + spare.pop()
            out.append(f"d dh {body[2]} {cards}")
        elif len(body) >= 2 and body[1] == "sm":
            p = int(body[0][1:]) - 1
            if len(live) > 1 and p in live:
                out.append(f"{body[0]} sm {shown[p]}" if p in shown else f"{body[0]} sm")
        elif len(body) == 3 and body[1] == "cbr":
            out.append(f"{body[0]} cbr {cents(body[2])}")
        else:
            out.append(raw)
    winnings = tuple(cents(w) for w in doc["winnings"]) if "winnings" in doc else None
    phh = PhhHand(variant="NT", antes=(0,) * n, blinds=tuple(cents(b) for b in blinds), min_bet=cents(doc["min_bet"]),
                  starting_stacks=tuple(cents(s) for s in doc["starting_stacks"]), actions=tuple(out),
                  players=tuple(str(p) for p in doc.get("players", [f"p{i + 1}" for i in range(n)])),
                  hand=int(doc["hand"]) if "hand" in doc else None, finishing_stacks=None)
    return Converted(phh, known, winnings, len(live) == 1 or live <= known)


def contributions(phh: PhhHand) -> list[int]:
    """Chips each player put in the pot, with uncalled bets returned."""
    n = len(phh.starting_stacks)
    total = [0] * n
    street = list(phh.blinds)

    def close() -> None:
        top, second = sorted(street, reverse=True)[:2]
        for i in range(n):
            total[i] += min(street[i], second) if street[i] == top else street[i]
            street[i] = 0

    for step in parse_actions(phh.actions, n):
        if step.verb == "db":
            close()
        elif step.verb == "cbr":
            assert step.player is not None and step.amount is not None
            street[step.player] = step.amount
        elif step.verb == "cc":
            assert step.player is not None
            p = step.player
            street[p] = min(max(street), phh.starting_stacks[p] - total[p])
    close()
    return total


def rake_of(hand: ImportedHand, conv: Converted) -> int:
    """Rake implied by engine net minus recorded net; raises if the payoff disagrees."""
    assert conv.winnings is not None
    phh = conv.phh
    paid = contributions(phh)
    seats = sorted(hand.seat_names)
    diff = [hand.finishing_stacks[s] - phh.starting_stacks[i] - (conv.winnings[i] - paid[i])
            for i, s in enumerate(seats)]
    if any(d < 0 for d in diff) or any(d and not conv.winnings[i] for i, d in enumerate(diff)):
        raise ImportRejected("payoff", f"engine net minus recorded net {diff}, winnings {list(conv.winnings)}")
    return sum(diff)


@dataclass
class Report:
    files: list[str] = field(default_factory=list)
    hands: int = 0
    eligible: int = 0
    replayed: int = 0
    skipped: Counter = field(default_factory=Counter)
    rejected: Counter = field(default_factory=Counter)
    examples: dict[str, list[str]] = field(default_factory=dict)
    payoff: Counter = field(default_factory=Counter)
    rake_cents: int = 0
    decisions: int = 0
    by_category: Counter = field(default_factory=Counter)
    by_street: Counter = field(default_factory=Counter)
    players: set[str] = field(default_factory=set)
    shown_hands: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": DATASET, "commit": COMMIT, "licence": LICENCE, "files": self.files, "hands": self.hands,
            "six_handed_six_max_plain_blinds": self.eligible, "replayed": self.replayed,
            "skipped": dict(self.skipped.most_common()), "rejected": dict(self.rejected.most_common()),
            "rejected_examples": self.examples, "payoff_check": dict(self.payoff), "rake_cents": self.rake_cents,
            "decisions": self.decisions, "by_category": dict(self.by_category.most_common()),
            "by_street": {s: self.by_street[s] for s in ("preflop", "flop", "turn", "river")},
            "distinct_players": len(self.players), "hands_with_shown_cards": self.shown_hands,
        }


SKIP_CODES = frozenset({"variant", "not_six_max", "blinds"})


def run(paths: list[Path], *, limit: int | None = None) -> Report:
    report = Report()
    for path in paths:
        report.files.append(path.name)
        for key, doc in load_phhs(path).items():
            if limit is not None and report.hands >= limit:
                return report
            report.hands += 1
            rel = f"{path.parent.parent.name}/{path.parent.name}/{path.stem}/{key}"
            try:
                conv = convert(doc, rel)
            except ImportRejected as exc:
                if exc.code in SKIP_CODES:
                    report.skipped[exc.code] += 1
                    continue
                conv = None
                err = exc
            if conv is not None:
                report.eligible += 1
                prov = Provenance(dataset=DATASET, path=f"data/handhq/{rel}", sha256="", url=REPO_URL, commit=COMMIT,
                                  licence=LICENCE, citation=CITATION, dataset_doi="10.5281/zenodo.10796885",
                                  paper_doi="")
                try:
                    hand = _replay(conv.phh, prov, None, mixed_stacks=True)
                    if conv.winnings is None:
                        report.payoff["no_winnings_field"] += 1
                    elif not any(conv.winnings):
                        report.payoff["winnings_all_zero_unchecked"] += 1
                    elif not conv.showdown_known:
                        report.payoff["mucked_showdown_unchecked"] += 1
                    else:
                        report.rake_cents += rake_of(hand, conv)
                        report.payoff["checked"] += 1
                except ImportRejected as exc:
                    err = exc
                else:
                    report.replayed += 1
                    report.decisions += len(hand.decisions)
                    report.players.update(hand.seat_names.values())
                    report.shown_hands += bool(conv.known)
                    for d in hand.decisions:
                        report.by_category[d.category] += 1
                        report.by_street[d.street] += 1
                    continue
            report.rejected[err.code] += 1
            ex = report.examples.setdefault(err.code, [])
            if len(ex) < 3:
                ex.append(f"{rel}: {err.detail}")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Replay HandHQ .phhs files in the arena engine and report the results.")
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    body = run(args.paths, limit=args.limit).as_dict()
    text = json.dumps(body, indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if body["replayed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
