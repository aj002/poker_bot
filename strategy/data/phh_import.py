"""Import PHH hands (the Pluribus corpus) and validate each one in the arena engine.

Every hand is re-played through a real :class:`arena.table.service.Table` built for
its format (six seats, the recorded blinds and stacks, a deck rebuilt from the
recorded cards by :func:`strategy.rules.conformance.build_deck`). Each PHH action
must be the engine's actor and a legal action there, every dealt card and board
must match, shown cards must match the engine's showdown, and the engine's final
stacks must equal ``finishing_stacks``. Anything else rejects the hand with a
reason code; nothing is repaired.

At each player decision the acting seat's :class:`~arena.domain.observations.Observation`
is frozen *before* the action (so it holds no future action or hidden card) and the
recorded action is mapped onto the canonical menu of
:func:`strategy.rules.adapter.canonical_actions`: fold/check/call stay as they are; a
raise maps to the nearest menu entry by raise increment (log ratio), with the exact
amount and its pot fraction kept alongside.

The structure is a cash-game proxy: one blind level for the whole cap, and
winner-take-all points, under which ICM is exactly the chip share (chip EV), so
point-pressure features carry no tournament premium.

    python -m strategy.data.phh_import                     # the cached corpus (strategy.data.fetch)
    python -m strategy.data.phh_import --source fixture    # the committed fixture hands
    python -m strategy.data.phh_import --out artifacts/phh/pluribus-import.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tomllib
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from arena.domain.cards import full_deck
from arena.domain.models import Action, LegalActions, TournamentConfig
from arena.domain.observations import Observation
from arena.rules.profiles import RuleProfile
from arena.table.projections import observation
from arena.table.service import ActionRejected, Table
from strategy.data.fetch import PLURIBUS, FetchError, PinnedSource, verify
from strategy.rules.adapter import CanonicalAction, RulesAdapter, canonical_actions
from strategy.rules.conformance import build_deck, deal_order
from strategy.rules.variants import CAP_NLHE_ASSETS, get_variant

TRANSFORM_VERSION = "phh-import-v1"
RULESET_ID = "practice_nlhe"
SIZE_FRACTIONS = (0.33, 0.5, 0.75, 1.0, 1.5, 2.5)
"""Pot fractions in the canonical menu (with min raise, pot and all-in)."""
CASH_HAND_CAP = 1000
SIX_MAX = 6
POSITIONS_6MAX = ("SB", "BB", "UTG", "MP", "CO", "BTN")
"""PHH player order is the engine order: p1 small blind ... p6 button."""

REPO = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO / "strategy" / "data" / "fixtures" / "pluribus"


class ImportRejected(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


# ------------------------------------------------------------------ parsing


@dataclass(frozen=True)
class Provenance:
    dataset: str
    path: str
    """Path inside the source repository, e.g. ``data/pluribus/100/0.phh``."""
    sha256: str
    url: str
    commit: str
    licence: str
    citation: str
    dataset_doi: str
    paper_doi: str
    transform: str = TRANSFORM_VERSION

    @property
    def hand_id(self) -> str:
        rel = self.path.removeprefix("data/").removesuffix(".phh")
        return rel.replace("/", "-")

    def as_dict(self) -> dict[str, Any]:
        return {"dataset": self.dataset, "path": self.path, "sha256": self.sha256, "url": self.url,
                "commit": self.commit, "licence": self.licence, "citation": self.citation,
                "dataset_doi": self.dataset_doi, "paper_doi": self.paper_doi, "transform": self.transform}


@dataclass(frozen=True)
class PhhHand:
    variant: str
    antes: tuple[int, ...]
    blinds: tuple[int, ...]
    min_bet: int
    starting_stacks: tuple[int, ...]
    actions: tuple[str, ...]
    players: tuple[str, ...]
    hand: int | None
    finishing_stacks: tuple[int, ...] | None


def parse_phh(text: str) -> PhhHand:
    try:
        doc = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ImportRejected("toml", str(exc)) from None
    try:
        n = len(doc["starting_stacks"])
        return PhhHand(
            variant=str(doc["variant"]),
            antes=tuple(int(x) for x in doc.get("antes", [0] * n)),
            blinds=tuple(int(x) for x in doc["blinds_or_straddles"]),
            min_bet=int(doc["min_bet"]),
            starting_stacks=tuple(int(x) for x in doc["starting_stacks"]),
            actions=tuple(str(a) for a in doc["actions"]),
            players=tuple(str(p) for p in doc.get("players", [f"p{i + 1}" for i in range(n)])),
            hand=int(doc["hand"]) if "hand" in doc else None,
            finishing_stacks=tuple(int(x) for x in doc["finishing_stacks"]) if "finishing_stacks" in doc else None,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ImportRejected("fields", f"missing or malformed field: {exc}") from None


def _cards(text: str) -> list[str]:
    if len(text) % 2:
        raise ImportRejected("cards", f"bad card string {text!r}")
    return [text[i:i + 2] for i in range(0, len(text), 2)]


@dataclass(frozen=True)
class PhhStep:
    index: int
    """Position in the PHH ``actions`` list."""
    verb: str
    """``dh`` / ``db`` (dealer) or ``f`` / ``cc`` / ``cbr`` / ``sm`` (player)."""
    player: int | None
    """Zero-based player index for player actions and hole dealing."""
    cards: tuple[str, ...] = ()
    amount: int | None = None


def parse_actions(actions: Sequence[str], players: int) -> list[PhhStep]:
    out = []
    for i, raw in enumerate(actions):
        body = raw.partition("#")[0].split()
        if len(body) < 2:
            raise ImportRejected("action_syntax", f"action {i} {raw!r}")
        who, verb, rest = body[0], body[1], body[2:]
        if who == "d":
            if verb == "dh" and len(rest) >= 2 and rest[0].startswith("p"):
                out.append(PhhStep(i, "dh", int(rest[0][1:]) - 1, tuple(_cards(rest[1]))))
            elif verb == "db" and rest:
                out.append(PhhStep(i, "db", None, tuple(_cards(rest[0]))))
            else:
                raise ImportRejected("action_syntax", f"unsupported dealer action {i} {raw!r}")
            continue
        if not who.startswith("p") or not who[1:].isdigit() or not 1 <= int(who[1:]) <= players:
            raise ImportRejected("action_syntax", f"unknown actor in action {i} {raw!r}")
        p = int(who[1:]) - 1
        if verb in ("f", "cc") and not rest:
            out.append(PhhStep(i, verb, p))
        elif verb == "cbr" and len(rest) == 1 and rest[0].isdigit():
            out.append(PhhStep(i, "cbr", p, amount=int(rest[0])))
        elif verb == "sm":
            out.append(PhhStep(i, "sm", p, tuple(_cards(rest[0])) if rest else ()))
        else:
            raise ImportRejected("action_syntax", f"unsupported player action {i} {raw!r}")
    return out


# --------------------------------------------------------------- the engine


def cash_config(players: int, stack: int, sb: int, bb: int) -> TournamentConfig:
    return TournamentConfig(
        ruleset_id=RULESET_ID, initial_player_count=players, seats=list(range(players)), starting_stack=stack,
        hand_cap=CASH_HAND_CAP, hands_per_level=CASH_HAND_CAP, blind_levels=[[sb, bb]],
        payout_points=[1] + [0] * (players - 1),
    )


class _HandTable(Table):
    """A table that snapshots itself as the first hand completes, before results are recorded."""

    def __init__(self, *args: Any, on_hand_end: Any = None, **kwargs: Any) -> None:
        self._on_hand_end = on_hand_end
        super().__init__(*args, **kwargs)

    def _record_result(self) -> None:
        if self._on_hand_end is not None and self.hand_number == 1:
            self._on_hand_end(self)
        super()._record_result()


@dataclass(frozen=True)
class SizeMapping:
    option: CanonicalAction
    fraction: float
    """Raise increment over the current bet, as a share of the pot after calling."""
    log_error: float
    """``|log(increment / option increment)|``; 0 for an exact menu amount."""


@dataclass
class Decision:
    phh_index: int
    seat: int
    player: str
    street: str
    observation: Observation
    menu: tuple[CanonicalAction, ...]
    action: Action
    category: str
    """fold, check, call or raise."""
    size: SizeMapping | None
    frame: int
    """Index of the frame (state before this action) in :attr:`ImportedHand.frames`."""


@dataclass
class ImportedHand:
    hand_id: str
    phh: PhhHand
    provenance: Provenance
    config: TournamentConfig
    rules_hash: str
    seat_names: dict[int, str]
    positions: dict[int, str]
    hole_cards: dict[int, list[str]]
    board: list[str]
    decisions: list[Decision]
    finishing_stacks: dict[int, int]
    showdown: dict[int, list[str]]
    phh_shown: dict[int, list[str]]
    table: Table
    frames: list[Any] = field(default_factory=list)
    """``(kind, table snapshot callback output)`` collected by :func:`import_hand`'s ``on_frame``."""
    notes: list[str] = field(default_factory=list)

    def seat_of(self, name: str) -> int | None:
        return next((s for s, n in self.seat_names.items() if n == name), None)


def category_of(action: Action) -> str:
    return "raise" if action.type == "raise_to" else action.type


def menu_for(obs: Observation) -> tuple[CanonicalAction, ...]:
    return canonical_actions(obs.legal, pot_total=obs.pot_total, current_bet_to=obs.current_bet_to,
                             fractions=SIZE_FRACTIONS)


def map_size(amount: int, obs: Observation, menu: Sequence[CanonicalAction]) -> SizeMapping:
    """The menu raise nearest to ``amount`` by raise increment over the current bet (log ratio)."""
    raises = [c for c in menu if c.amount is not None]
    if not raises:
        raise ImportRejected("size_mapping", f"raise to {amount} but the canonical menu has no raise")
    level = obs.current_bet_to
    after_call = max(1, obs.pot_total + (obs.legal.call_amount or 0))
    inc = max(1, amount - level)

    def err(c: CanonicalAction) -> float:
        assert c.amount is not None
        return abs(math.log(inc / max(1, c.amount - level)))

    best = min(raises, key=lambda c: (err(c), c.amount))
    return SizeMapping(best, (amount - level) / after_call, 0.0 if best.amount == amount else err(best))


def _action_for(step: PhhStep, legal: LegalActions) -> Action:
    if step.verb == "f":
        return Action(type="fold")
    if step.verb == "cc":
        return Action(type="check") if legal.check else Action(type="call")
    assert step.amount is not None
    return Action(type="raise_to", amount=step.amount)


def check_format(phh: PhhHand, *, players: int = SIX_MAX, mixed_stacks: bool = False) -> tuple[int, int, int]:
    """``(stack, small blind, big blind)`` for a hand the arena can host as six-max NLHE.

    ``mixed_stacks`` accepts unequal starting stacks (``stack`` is then the largest) and
    records without ``finishing_stacks``, as in online cash-game logs.
    """
    n = len(phh.starting_stacks)
    if phh.variant != "NT":
        raise ImportRejected("variant", f"PHH variant {phh.variant!r} is not no-limit Texas hold'em (NT)")
    if n != players:
        raise ImportRejected("not_six_max", f"{n} players")
    if len(phh.players) != n or len(phh.blinds) != n or len(phh.antes) != n:
        raise ImportRejected("fields", "players, blinds and antes must list every player")
    if any(phh.antes):
        raise ImportRejected("antes", f"antes {list(phh.antes)}; the arena rules have no ante")
    sb, bb, *rest = phh.blinds
    if any(rest) or not 0 < sb <= bb:
        raise ImportRejected("blinds", f"blinds_or_straddles {list(phh.blinds)} is not small/big blind only")
    if phh.min_bet != bb:
        raise ImportRejected("blinds", f"min_bet {phh.min_bet} differs from the big blind {bb}")
    if mixed_stacks:
        if min(phh.starting_stacks) <= bb:
            raise ImportRejected("stacks", f"a starting stack in {list(phh.starting_stacks)} does not cover the big "
                                           f"blind")
        return max(phh.starting_stacks), sb, bb
    stacks = set(phh.starting_stacks)
    if len(stacks) != 1:
        raise ImportRejected("stacks", f"unequal starting stacks {list(phh.starting_stacks)}")
    stack = stacks.pop()
    if stack <= bb:
        raise ImportRejected("stacks", f"starting stack {stack} does not cover the big blind")
    if phh.finishing_stacks is None or len(phh.finishing_stacks) != n:
        raise ImportRejected("fields", "finishing_stacks missing")
    return stack, sb, bb


def scaled(phh: PhhHand, k: int) -> PhhHand:
    """The same hand with every chip amount multiplied by ``k``."""
    actions = []
    for raw in phh.actions:
        body = raw.partition("#")[0].split()
        actions.append(f"{body[0]} cbr {int(body[2]) * k}" if len(body) == 3 and body[1] == "cbr" else raw)
    return PhhHand(phh.variant, tuple(a * k for a in phh.antes), tuple(b * k for b in phh.blinds), phh.min_bet * k,
                   tuple(s * k for s in phh.starting_stacks), tuple(actions), phh.players, phh.hand,
                   tuple(s * k for s in phh.finishing_stacks) if phh.finishing_stacks is not None else None)


def import_hand(text: str, provenance: Provenance, *, on_frame: Any = None) -> ImportedHand:
    """Validate one PHH hand in the arena engine; raise :class:`ImportRejected` on any mismatch.

    ``on_frame(table, kind)`` is called with the live table after the deal (``deal``),
    after every action (``action``) and as the hand completes (``hand_end``); its
    return values are kept in :attr:`ImportedHand.frames`.

    A record whose ``finishing_stacks`` lose chips (a split pot with an odd chip, written
    as truncated half chips) is re-played with every amount doubled, where the split is
    exact: each seat must then finish at twice its recorded stack, plus one for each
    truncated half chip. The hand is kept at its original scale, with the arena's
    odd-chip rule deciding the final stacks.
    """
    phh = parse_phh(text)
    hand = _replay(phh, provenance, on_frame)
    want = dict(zip(sorted(hand.seat_names), phh.finishing_stacks or ()))
    got = hand.finishing_stacks
    if got == want:
        return hand
    lost = sum(phh.starting_stacks) - sum(want.values())
    if lost <= 0:
        raise ImportRejected("finishing_stacks", f"engine {got}, PHH {want}")
    doubled = _replay(scaled(phh, 2), provenance, None).finishing_stacks
    halves = {s: doubled[s] - 2 * want[s] for s in want}
    if any(h not in (0, 1) for h in halves.values()) or sum(halves.values()) != 2 * lost \
            or any(got[s] != want[s] for s in want if halves[s] == 0):
        raise ImportRejected("finishing_stacks", f"engine {got}, PHH {want} (x2 engine {doubled})")
    split = sorted(s for s, h in halves.items() if h)
    hand.notes.append(f"half-chip split between seats {split}: validated at chips x2; the arena odd-chip rule "
                      f"gives {[got[s] for s in split]}")
    return hand


def _replay(phh: PhhHand, provenance: Provenance, on_frame: Any, *, mixed_stacks: bool = False) -> ImportedHand:
    stack, sb, bb = check_format(phh, mixed_stacks=mixed_stacks)
    n = len(phh.starting_stacks)
    steps = parse_actions(phh.actions, n)
    variant = get_variant(RULESET_ID)
    variant.require(CAP_NLHE_ASSETS, asset="PHH import")
    config = cash_config(n, stack, sb, bb)
    adapter = RulesAdapter.from_config(config)

    seats = tuple(range(n))
    button, sb_seat, bb_seat = seats[-1], seats[0], seats[1]
    hole = {s.player: list(s.cards) for s in steps if s.verb == "dh"}
    if sorted(hole) != list(seats) or any(len(c) != variant.hole_cards for c in hole.values()):
        raise ImportRejected("hole_cards", "every player needs exactly two dealt hole cards")
    boards = [list(s.cards) for s in steps if s.verb == "db"]
    if [len(b) for b in boards] != list(variant.board_streets[:len(boards)]):
        raise ImportRejected("board", f"board deals {[len(b) for b in boards]} do not follow 3/1/1")
    board = [c for b in boards for c in b]
    try:
        deck = build_deck(variant, deal_order(seats, seats, button, bb_seat), hole, board)
    except ValueError as exc:
        raise ImportRejected("cards", str(exc)) from None

    def deck_source(hand_number: int, profile: RuleProfile) -> tuple[str, ...]:
        return deck if hand_number == 1 else tuple(full_deck(profile.ranks))

    frames: list[Any] = []
    result: dict[str, Any] = {}

    def at_end(t: Table) -> None:
        assert t.hand is not None
        result["revealed"] = {s: list(c) for s, c in t.hand.revealed.items()}
        result["final"] = t.hand.final_stacks()
        result["board"] = list(t.hand.board)
        if on_frame is not None:
            frames.append(on_frame(t, "hand_end"))

    table = _HandTable(config, table_id=f"phh_{provenance.hand_id}", deck_source=deck_source, initial_button=button,
                       start_stacks=dict(zip(seats, phh.starting_stacks)) if mixed_stacks else None,
                       on_hand_end=at_end)
    hand = table.hand
    if hand is None or table.positions is None:
        raise ImportRejected("engine", "the engine did not start the hand")
    if (table.positions.small_blind, table.positions.big_blind) != (sb_seat, bb_seat):
        raise ImportRejected("positions", "engine blinds are not on PHH players 1 and 2")
    for s in seats:
        if list(hand.hole_cards[s]) != hole[s]:
            raise ImportRejected("hole_cards", f"engine dealt seat {s} {hand.hole_cards[s]}, PHH has {hole[s]}")
    if on_frame is not None:
        frames.append(on_frame(table, "deal"))

    names = {s: phh.players[s] for s in seats}
    decisions: list[Decision] = []
    shown: dict[int, list[str]] = {}
    boards_seen = 0
    for step in steps:
        if step.verb == "dh":
            continue
        if step.verb == "db":
            boards_seen += len(step.cards)
            live = table.hand if table.hand_number == 1 else None
            dealt = list(live.board) if live is not None else result.get("board", [])
            if dealt[:boards_seen] != board[:boards_seen]:
                raise ImportRejected("board", f"PHH action {step.index} deals {list(step.cards)}; engine board "
                                              f"{dealt}")
            continue
        assert step.player is not None
        seat = seats[step.player]
        if step.verb == "sm":
            shown[seat] = list(step.cards)
            continue
        if table.hand_number != 1 or table.actor_seat is None:
            raise ImportRejected("actor", f"PHH action {step.index} by seat {seat} after the hand ended")
        if table.actor_seat != seat:
            raise ImportRejected("actor", f"PHH action {step.index} by seat {seat}; the engine's actor is "
                                          f"{table.actor_seat}")
        obs = observation(table, seat)
        action = _action_for(step, obs.legal)
        if not obs.legal.allows(action):
            raise ImportRejected("illegal_action", f"PHH action {step.index} {action.model_dump(exclude_none=True)} "
                                                   f"is not legal: {obs.legal.model_dump()}")
        menu = adapter.actions(obs, SIZE_FRACTIONS)
        size = map_size(action.amount, obs, menu) if action.amount is not None else None
        if size is not None and not obs.legal.allows(size.option.action()):
            raise ImportRejected("size_mapping", f"mapped option {size.option.key} is not legal")
        decisions.append(Decision(step.index, seat, names[seat], obs.street, obs, menu, action, category_of(action),
                                  size, len(frames) - 1))
        try:
            table.submit(seat, action)
        except ActionRejected as exc:
            raise ImportRejected("rejected", f"PHH action {step.index}: {exc}") from None
        if on_frame is not None and table.hand_number == 1:
            frames.append(on_frame(table, "action"))

    if "final" not in result:
        raise ImportRejected("incomplete", "PHH actions end before the hand is complete")
    revealed = result["revealed"]
    for seat, cards in shown.items():
        if cards and cards != hole[seat]:
            raise ImportRejected("showdown", f"seat {seat} shows {cards} but was dealt {hole[seat]}")
        if seat not in revealed:
            raise ImportRejected("showdown", f"PHH shows seat {seat}; the engine reached no showdown for it")
    notes = []
    mucked = sorted(set(revealed) - set(shown))
    if mucked:
        notes.append(f"engine shows seats {mucked} that the PHH record lets muck")
    return ImportedHand(
        hand_id=provenance.hand_id, phh=phh, provenance=provenance, config=config, rules_hash=adapter.rules_hash,
        seat_names=names, positions=dict(zip(seats, POSITIONS_6MAX)), hole_cards=hole, board=board,
        decisions=decisions, finishing_stacks=dict(result["final"]), showdown=revealed, phh_shown=shown, table=table,
        frames=frames, notes=notes,
    )


# ------------------------------------------------------------------ sources


def provenance_for(rel: str, data: bytes, source: PinnedSource = PLURIBUS) -> Provenance:
    return Provenance(dataset="phh-dataset/pluribus", path=f"{source.subdir}/{rel}",
                      sha256=hashlib.sha256(data).hexdigest(), url=source.file_url(rel), commit=source.commit,
                      licence=source.licence, citation=source.citation, dataset_doi=source.dataset_doi,
                      paper_doi=source.paper_doi)


def _hand_key(rel: str) -> tuple[str, int]:
    session, _, name = rel.partition("/")
    stem = name.removesuffix(".phh")
    return session, int(stem) if stem.isdigit() else -1


def iter_source(root: Path) -> Iterator[tuple[str, bytes]]:
    """``(path relative to the pluribus directory, bytes)`` in session/hand order."""
    rels = sorted((p.relative_to(root).as_posix() for p in root.rglob("*.phh")), key=_hand_key)
    for rel in rels:
        yield rel, (root / rel).read_bytes()


def source_dir(name: str) -> Path:
    if name == "fixture":
        return FIXTURE_DIR
    if name == "cache":
        return verify()
    return Path(name)


@dataclass
class ImportReport:
    source: str
    hands: int = 0
    imported: int = 0
    rejected: Counter = field(default_factory=Counter)
    examples: dict[str, list[str]] = field(default_factory=dict)
    decisions: int = 0
    pluribus_decisions: int = 0
    by_category: Counter = field(default_factory=Counter)
    by_street: Counter = field(default_factory=Counter)
    raises: int = 0
    exact_sizes: int = 0
    size_options: Counter = field(default_factory=Counter)
    size_log_error_sum: float = 0.0
    size_log_error_max: float = 0.0
    notes: Counter = field(default_factory=Counter)

    def add(self, hand: ImportedHand, hero: str = "Pluribus") -> None:
        self.imported += 1
        self.decisions += len(hand.decisions)
        for d in hand.decisions:
            if d.player != hero:
                continue
            self.pluribus_decisions += 1
            self.by_category[d.category] += 1
            self.by_street[d.street] += 1
            if d.size is not None:
                self.raises += 1
                self.exact_sizes += d.size.log_error == 0.0
                self.size_options[d.size.option.key] += 1
                self.size_log_error_sum += d.size.log_error
                self.size_log_error_max = max(self.size_log_error_max, d.size.log_error)
        for note in hand.notes:
            self.notes[note.split(" seats ")[0].split(" between")[0]] += 1

    def reject(self, rel: str, exc: ImportRejected) -> None:
        self.rejected[exc.code] += 1
        ex = self.examples.setdefault(exc.code, [])
        if len(ex) < 5:
            ex.append(f"{rel}: {exc.detail}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source, "transform": TRANSFORM_VERSION, "hands": self.hands, "imported": self.imported,
            "rejected": dict(sorted(self.rejected.items())), "rejected_examples": self.examples,
            "decisions": self.decisions, "pluribus_decisions": self.pluribus_decisions,
            "pluribus_by_category": dict(sorted(self.by_category.items())),
            "pluribus_by_street": {s: self.by_street[s] for s in ("preflop", "flop", "turn", "river")},
            "pluribus_raises": self.raises, "pluribus_exact_menu_sizes": self.exact_sizes,
            "pluribus_size_options": dict(self.size_options.most_common()),
            "pluribus_size_mean_log_error": round(self.size_log_error_sum / self.raises, 4) if self.raises else 0.0,
            "pluribus_size_max_log_error": round(self.size_log_error_max, 4),
            "notes": dict(self.notes),
        }


def import_all(root: Path, *, label: str | None = None, limit: int | None = None,
               keep: Any = None) -> ImportReport:
    """Import every hand under ``root``; ``keep(hand)`` sees each imported hand."""
    report = ImportReport(label or str(root))
    for k, (rel, data) in enumerate(iter_source(root)):
        if limit is not None and k >= limit:
            break
        report.hands += 1
        try:
            hand = import_hand(data.decode("utf-8"), provenance_for(rel, data))
        except ImportRejected as exc:
            report.reject(rel, exc)
            continue
        report.add(hand)
        if keep is not None:
            keep(hand)
    return report


def iter_imported(root: Path, *, on_frame: Any = None, limit: int | None = None) -> Iterable[ImportedHand]:
    for k, (rel, data) in enumerate(iter_source(root)):
        if limit is not None and k >= limit:
            return
        try:
            yield import_hand(data.decode("utf-8"), provenance_for(rel, data), on_frame=on_frame)
        except ImportRejected:
            continue


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate PHH hands in the arena engine and report the results.")
    ap.add_argument("--source", default="cache", help="cache (default), fixture, or a directory of .phh files")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None, help="write the report as JSON")
    args = ap.parse_args(argv)
    try:
        root = source_dir(args.source)
    except FetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    report = import_all(root, label=args.source, limit=args.limit)
    body = report.as_dict()
    text = json.dumps(body, indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report.imported else 1


if __name__ == "__main__":
    raise SystemExit(main())
