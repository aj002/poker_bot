"""Offline conformance: does a simulator implement the rules our variant metadata declares?

A simulator is anything with :class:`Simulator`'s methods; the arena engine is the
reference (:class:`ArenaSimulator`) and an official starter-kit shim is the intended
second implementation. The suite checks, per variant:

``metadata``      deck, deal, showdown selection and betting match the :class:`VariantSpec`.
``ranking``       one five-card probe per category sorts in the declared order, and the
                  declared lowest straight sits between trips and the next straight.
``fixtures``      the arena golden hand fixtures (``tests/golden/fixtures/{core,eval}``)
                  replay with the expected actors, legal actions, rejections, pots and
                  stacks. Decks are rebuilt from *our* metadata, not the engine's.
``differential``  seeded random hands run in lockstep through a reference and a
                  candidate simulator; legal actions, actors and results must agree, every
                  canonical action must be legal and accepted, and chips are conserved.
``scoring``       the ICM fixtures (``tests/golden/fixtures/tournament/icm.json``) through
                  the scorer the objective uses.

Engine-only fixture keys (event counts, automatic actions) are checked only when the
simulator exposes ``events``; otherwise they are reported as skipped, never as passes.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol

from arena.domain.models import Action, LegalActions, LocalRules
from strategy.rules.adapter import canonical_actions
from strategy.rules.reveal import get_variant
from strategy.rules.variants import VariantSpec

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE_ROOT = REPO_ROOT / "tests" / "golden" / "fixtures"
DIFFERENTIAL_FRACTIONS = (0.33, 0.5, 0.75, 1.0, 1.5, 2.5)

_PROBES = {
    "23456789TJQKA": {
        "high_card": "2c4d6h8sTc", "one_pair": "2c2d6h8sTc", "two_pair": "2c2d6h6sTc",
        "three_of_a_kind": "2c2d2h8sTc", "straight": "3c4d5h6s7c", "flush": "2c4c6c8cTc",
        "full_house": "2c2d2h8s8c", "four_of_a_kind": "2c2d2h2s8c", "straight_flush": "2c3c4c5c6c",
    },
    "6789TJQKA": {
        "high_card": "6c8dThQsAc", "one_pair": "6c6dThQsAc", "two_pair": "6c6dThTsAc",
        "three_of_a_kind": "6c6d6hQsAc", "straight": "7c8d9hTsJc", "flush": "6c8cTcQcAc",
        "full_house": "6c6d6hQsQc", "four_of_a_kind": "6c6d6h6sAc", "straight_flush": "6c7c8c9cTc",
    },
}


# ------------------------------------------------------------------ protocols


@dataclass(frozen=True)
class HandSetup:
    ruleset_id: str
    seat_ring: tuple[int, ...]
    stacks: Mapping[int, int]
    button: int
    small_blind_seat: int | None
    big_blind_seat: int
    small_blind: int
    big_blind: int
    deck: tuple[str, ...]
    local_rules: Mapping[str, Any] = field(default_factory=dict)


class SimHand(Protocol):
    """One hand. ``apply`` raises ``ValueError`` (or a subclass) for an illegal action."""

    @property
    def actor_seat(self) -> int | None: ...
    @property
    def is_complete(self) -> bool: ...
    @property
    def street(self) -> str | None: ...
    @property
    def board(self) -> Sequence[str]: ...
    @property
    def hole_cards(self) -> Mapping[int, Sequence[str]]: ...
    def pot_total(self) -> int: ...
    def current_level(self) -> int:
        """The street total a call matches (``Observation.current_bet_to``)."""
    def players(self) -> Sequence[Any]: ...
    def legal_actions(self) -> LegalActions: ...
    def apply(self, action: Action) -> Any: ...
    def result(self) -> Any: ...


class Simulator(Protocol):
    name: str

    def metadata(self, ruleset_id: str) -> dict[str, Any]:
        """``ranks``, ``hole_cards``, ``board_streets``, ``burn_before_board``, ``betting``, ``hand_selection``."""

    def new_hand(self, setup: HandSetup) -> SimHand: ...

    def rank_five(self, ruleset_id: str, cards: str) -> Any:
        """A comparable strength for five cards; raise NotImplementedError if unavailable."""


class Scorer(Protocol):
    def icm(self, stacks: Sequence[int], prizes: Sequence[float]) -> list[float]: ...


class ArenaSimulator:
    """The arena engine (PokerKit plus the betting-rights overlay) as a :class:`Simulator`."""

    name = "arena"

    def metadata(self, ruleset_id: str) -> dict[str, Any]:
        from pokerkit import Mode

        from arena.rules.profiles import get_profile

        profile = get_profile(ruleset_id)
        game = profile.game_class((), True, 0, (1, 2), 2, mode=Mode.TOURNAMENT)
        streets = game.streets
        return {
            "ranks": profile.ranks,
            "hole_cards": len(streets[0].hole_dealing_statuses),
            "board_streets": [s.board_dealing_count for s in streets[1:]],
            "burn_before_board": all(s.card_burning_status for s in streets[1:]),
            "betting": profile.betting,
            "hand_selection": profile.hand_selection,
        }

    def new_hand(self, setup: HandSetup) -> SimHand:
        from arena.rules.pokerkit_adapter import HandSpec, PokerKitHand
        from arena.rules.profiles import get_profile

        spec = HandSpec(
            profile=get_profile(setup.ruleset_id), seat_ring=tuple(setup.seat_ring), stacks=dict(setup.stacks),
            button=setup.button, small_blind_seat=setup.small_blind_seat, big_blind_seat=setup.big_blind_seat,
            small_blind=setup.small_blind, big_blind=setup.big_blind, deck=tuple(setup.deck),
            local_rules=LocalRules(**dict(setup.local_rules)),
        )
        return PokerKitHand(spec)

    def rank_five(self, ruleset_id: str, cards: str) -> Any:
        from pokerkit import ShortDeckHoldemHand, StandardHighHand

        from arena.rules.profiles import get_profile

        custom = get_profile(ruleset_id).five_card_hand
        if custom is not None:
            return custom(cards)
        return (ShortDeckHoldemHand if ruleset_id == "uncertainty_shortdeck" else StandardHighHand)(cards)


class ArenaScorer:
    def icm(self, stacks: Sequence[int], prizes: Sequence[float]) -> list[float]:
        from strategy.objective.points import icm_points

        return icm_points(stacks, prizes)


# --------------------------------------------------------------------- report


@dataclass
class CheckResult:
    name: str
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.failed == 0 and self.passed > 0

    def fail(self, message: str) -> None:
        self.failed += 1
        if len(self.failures) < 50:
            self.failures.append(message)

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "passed": self.passed, "failed": self.failed,
                "skipped": self.skipped, "failures": list(self.failures), "notes": list(self.notes)}


@dataclass
class ConformanceReport:
    simulator: str
    ruleset_id: str
    variant_hash: str
    checks: list[CheckResult]

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {"simulator": self.simulator, "ruleset_id": self.ruleset_id, "variant_hash": self.variant_hash,
                "ok": self.ok, "checks": [c.as_dict() for c in self.checks]}


# ----------------------------------------------------------- metadata, ranking


def check_metadata(sim: Simulator, spec: VariantSpec) -> CheckResult:
    out = CheckResult("metadata")
    got = sim.metadata(spec.ruleset_id)
    want = {"ranks": spec.ranks, "hole_cards": spec.hole_cards, "board_streets": list(spec.board_streets),
            "burn_before_board": spec.burn_before_board, "betting": spec.betting,
            "hand_selection": spec.hand_selection}
    for key, value in want.items():
        if key not in got:
            out.skipped += 1
            out.notes.append(f"{sim.name} does not report {key}")
        elif got[key] != value:
            out.fail(f"{key}: variant declares {value!r}, {sim.name} has {got[key]!r}")
        else:
            out.passed += 1
    return out


def check_ranking(sim: Simulator, spec: VariantSpec) -> CheckResult:
    out = CheckResult("ranking")
    probes = _PROBES.get(spec.ranks)
    if probes is None:
        out.skipped += 1
        out.notes.append(f"no ranking probes for ranks {spec.ranks}")
        return out
    try:
        strength = [sim.rank_five(spec.ruleset_id, probes[c]) for c in spec.category_order]
    except NotImplementedError:
        out.skipped += 1
        out.notes.append(f"{sim.name} exposes no five-card evaluator")
        return out
    pairs = list(zip(spec.category_order, strength))
    for (lo, a), (hi, b) in zip(pairs, pairs[1:]):
        if a < b:
            out.passed += 1
        else:
            out.fail(f"{hi} ({probes[hi]}) should beat {lo} ({probes[lo]})")
    suits = "cdhsc"
    lowest = "".join(r + suits[k] for k, r in enumerate(spec.lowest_straight))
    order = spec.ranks
    start = order.index(spec.lowest_straight[1])
    next_straight = "".join(order[start + k] + suits[k] for k in range(5))
    below_name = spec.category_order[spec.category_order.index("straight") - 1]
    below = probes[below_name]
    low_s, next_s, below_s = (sim.rank_five(spec.ruleset_id, c) for c in (lowest, next_straight, below))
    if below_s < low_s < next_s:
        out.passed += 1
    else:
        out.fail(f"lowest straight {lowest} should beat {below_name} {below} and lose to {next_straight}")
    return out


# ------------------------------------------------------------------- fixtures


def load_fixture_group(group: str, root: Path = DEFAULT_FIXTURE_ROOT) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted((root / group).glob("*.json"))]


def deal_order(ring: Sequence[int], live: Sequence[int], button: int, big_blind: int) -> list[int]:
    """The spec's dealing rule: clockwise from the seat left of the button; heads-up, the BB first."""
    if len(live) == 2:
        return [big_blind, button]
    i = list(ring).index(button)
    rotated = list(ring[i + 1:]) + list(ring[: i + 1])
    return [s for s in rotated if s in live]


def build_deck(spec: VariantSpec, order: Sequence[int], hole: Mapping[int, Sequence[str]],
               board: Sequence[str]) -> tuple[str, ...]:
    """A full deck that deals ``hole`` and ``board`` under the variant's declared deal and burns."""
    full = [r + s for r in spec.ranks for s in spec.suits]
    reserved = [c for cards in hole.values() for c in cards] + list(board)
    if len(set(reserved)) != len(reserved):
        raise ValueError("fixture repeats a card")
    filler = [c for c in full if c not in reserved]
    filler.reverse()
    deck: list[str] = []
    for k in range(spec.hole_cards):
        for seat in order:
            given = hole.get(seat)
            deck.append(given[k] if given else filler.pop())
    board_left = list(board)
    for count in spec.board_streets:
        if spec.burn_before_board:
            deck.append(filler.pop())
        for _ in range(count):
            deck.append(board_left.pop(0) if board_left else filler.pop())
    deck.extend(reversed(filler))
    return tuple(deck)


def _cards(text: str) -> tuple[str, ...]:
    compact = text.replace(" ", "")
    return tuple(compact[i:i + 2] for i in range(0, len(compact), 2))


def fixture_setup(fx: Mapping[str, Any]) -> HandSetup:
    setup = fx["setup"]
    ruleset_id = fx.get("ruleset", "practice_nlhe")
    spec = get_variant(ruleset_id)
    ring = sorted(setup.get("seats", [int(s) for s in setup["stacks"]]))
    stacks = {int(s): v for s, v in setup["stacks"].items()}
    live = [s for s in ring if s in stacks]
    hole = {int(s): _cards(c) for s, c in setup.get("hole", {}).items()}
    order = deal_order(ring, live, setup["button"], setup["big_blind_seat"])
    sb, bb = setup.get("blinds", [10, 20])
    return HandSetup(ruleset_id=ruleset_id, seat_ring=tuple(ring), stacks=stacks, button=setup["button"],
                     small_blind_seat=setup.get("small_blind_seat"), big_blind_seat=setup["big_blind_seat"],
                     small_blind=sb, big_blind=bb, deck=build_deck(spec, order, hole, _cards(setup.get("board", ""))),
                     local_rules=dict(fx.get("local_rules", {})))


_ENGINE_EVENT_KEYS = ("excess_returned", "event_counts", "auto_actions", "events_include", "events_exclude",
                      "first_hole_seat")


def _seat_map(d: Mapping[str, Any]) -> dict[int, Any]:
    return {int(k): v for k, v in d.items()}


def replay_fixture(sim: Simulator, fx: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """Replay one golden hand fixture. Returns (mismatches, skipped keys)."""
    fid = fx["id"]
    errors: list[str] = []
    skipped: list[str] = []
    hand = sim.new_hand(fixture_setup(fx))
    for seat, cards in fx["setup"].get("hole", {}).items():
        if tuple(hand.hole_cards[int(seat)]) != _cards(cards):
            errors.append(f"{fid}: seat {seat} dealt {list(hand.hole_cards[int(seat)])}, expected {cards}")
    for n, step in enumerate(fx.get("steps", [])):
        where = f"{fid} step {n}"
        if "state" in step:
            errors += _check_state(hand, step["state"], where)
        if "actor" in step and hand.actor_seat != step["actor"]:
            errors.append(f"{where}: actor {hand.actor_seat}, expected {step['actor']}")
        if "legal" in step:
            legal = hand.legal_actions().model_dump()
            for key, want in step["legal"].items():
                if legal.get(key) != want:
                    errors.append(f"{where}: legal.{key} {legal.get(key)!r}, expected {want!r}")
        if "reject" in step:
            before = (hand.actor_seat, hand.pot_total(), list(hand.players()))
            try:
                hand.apply(Action.model_validate(step["reject"]))
            except ValueError:
                if before != (hand.actor_seat, hand.pot_total(), list(hand.players())):
                    errors.append(f"{where}: rejected action changed the state")
            else:
                errors.append(f"{where}: {step['reject']} was accepted")
                return errors, skipped
        if "do" in step:
            try:
                hand.apply(Action.model_validate(step["do"]))
            except ValueError as exc:
                errors.append(f"{where}: {step['do']} rejected: {exc}")
                return errors, skipped
        if step.get("check_down"):
            while not hand.is_complete:
                legal = hand.legal_actions()
                hand.apply(Action(type="check") if legal.check else Action(type="call"))
    expect = fx.get("expect", {})
    if expect:
        if not hand.is_complete:
            return errors + [f"{fid}: hand not complete"], skipped
        errors += _check_result(hand, expect, fid, skipped)
    return errors, skipped


def _check_state(hand: SimHand, expected: Mapping[str, Any], where: str) -> list[str]:
    errors = []
    players = {p.seat: p for p in hand.players()}
    for key, want in expected.items():
        if key == "street":
            got: Any = hand.street
        elif key == "actor":
            got = hand.actor_seat
        elif key == "pot_total":
            got = hand.pot_total()
        elif key == "board":
            got = " ".join(hand.board)
        elif key == "complete":
            got = hand.is_complete
        elif key in ("street_committed", "hand_committed", "stack", "status"):
            want = _seat_map(want)
            got = {s: getattr(players[s], key) for s in want}
        else:
            errors.append(f"{where}: unknown state key {key}")
            continue
        if got != want:
            errors.append(f"{where}: {key} {got!r}, expected {want!r}")
    return errors


def _check_result(hand: SimHand, expect: Mapping[str, Any], fid: str, skipped: list[str]) -> list[str]:
    errors = []
    result = hand.result()
    if "final_stacks" in expect and dict(result.final_stacks) != _seat_map(expect["final_stacks"]):
        errors.append(f"{fid}: final stacks {dict(result.final_stacks)}")
    if "contributions" in expect and dict(result.contributions) != _seat_map(expect["contributions"]):
        errors.append(f"{fid}: contributions {dict(result.contributions)}")
    if "pots" in expect:
        got = [{"amount": p.amount, "eligible": sorted(p.eligible), "awards": dict(sorted(p.awards.items()))}
               for p in result.pots]
        want = [{"amount": p["amount"], "eligible": sorted(p["eligible"]),
                 **({"awards": dict(sorted(_seat_map(p["awards"]).items()))} if "awards" in p else {})}
                for p in expect["pots"]]
        for g, w in zip(got, want):
            if "awards" not in w:
                g.pop("awards")
        if got != want:
            errors.append(f"{fid}: pots {got}, expected {want}")
    if "revealed" in expect and sorted(result.revealed) != sorted(expect["revealed"]):
        errors.append(f"{fid}: revealed {sorted(result.revealed)}")
    if "board_cards" in expect and len(result.board) != expect["board_cards"]:
        errors.append(f"{fid}: board {list(result.board)}")
    if "ended_by_fold" in expect and result.ended_by_fold != expect["ended_by_fold"]:
        errors.append(f"{fid}: ended_by_fold {result.ended_by_fold}")
    events = getattr(hand, "events", None)
    for key in _ENGINE_EVENT_KEYS:
        if key not in expect:
            continue
        if events is None:
            skipped.append(f"{fid}: {key}")
            continue
        errors += _check_events(events, key, expect[key], fid)
    return errors


def _check_events(events: Sequence[Any], key: str, want: Any, fid: str) -> list[str]:
    types = [e.type for e in events]
    if key == "excess_returned":
        got = {e.data["seat"]: e.data["amount"] for e in events if e.type == "ExcessReturned"}
        return [] if got == _seat_map(want) else [f"{fid}: excess returned {got}"]
    if key == "event_counts":
        return [f"{fid}: {t} count {types.count(t)} != {n}" for t, n in want.items() if types.count(t) != n]
    if key == "auto_actions":
        got = sum(1 for e in events if e.type == "ActionApplied" and "auto" in e.data)
        return [] if got == want else [f"{fid}: auto actions {got}"]
    if key == "events_include":
        return [f"{fid}: missing event {t}" for t in want if t not in types]
    if key == "events_exclude":
        return [f"{fid}: unexpected event {t}" for t in want if t in types]
    first = next(e for e in events if e.type == "HoleDealt")
    return [] if first.data["seat"] == want else [f"{fid}: first hole card to {first.data['seat']}"]


def check_fixtures(sim: Simulator, ruleset_id: str, fixtures: Iterable[Mapping[str, Any]]) -> CheckResult:
    out = CheckResult("fixtures")
    for fx in fixtures:
        if fx.get("ruleset", "practice_nlhe") != ruleset_id:
            continue
        try:
            errors, skipped = replay_fixture(sim, fx)
        except Exception as exc:  # a crash is a mismatch, not a harness error
            errors, skipped = [f"{fx['id']}: {type(exc).__name__}: {exc}"], []
        if errors:
            out.failed += 1
            out.failures.extend(errors[:5])
        else:
            out.passed += 1
        out.skipped += len(skipped)
        out.notes.extend(f"skipped engine-event check {s}" for s in skipped)
    if out.passed + out.failed == 0:
        out.notes.append(f"no fixtures for {ruleset_id}")
    return out


# --------------------------------------------------------------- differential


def random_setup(spec: VariantSpec, rng: random.Random) -> HandSetup:
    n = rng.choice((2, 2, 3, 4, 6, 8))
    ring = tuple(sorted(rng.sample(range(8), n)))
    sb, bb = rng.choice(((10, 20), (25, 50), (100, 200), (150, 300)))
    stacks = {}
    for s in ring:
        r = rng.random()
        stacks[s] = rng.randint(1, bb) if r < 0.12 else rng.randint(bb, 8 * bb) if r < 0.4 else rng.randint(
            8 * bb, 150 * bb)
    button = rng.choice(ring)
    i = ring.index(button)
    if n == 2:
        small, big = button, ring[(i + 1) % 2]
    else:
        small, big = ring[(i + 1) % n], ring[(i + 2) % n]
    deck = [r + s for r in spec.ranks for s in spec.suits]
    rng.shuffle(deck)
    return HandSetup(spec.ruleset_id, ring, stacks, button, small, big, sb, bb, tuple(deck))


def _result_view(result: Any) -> dict[str, Any]:
    return {
        "final_stacks": dict(sorted(dict(result.final_stacks).items())),
        "contributions": dict(sorted(dict(result.contributions).items())),
        "pots": [(p.amount, sorted(p.eligible), dict(sorted(p.awards.items()))) for p in result.pots],
        "board": list(result.board),
        "ended_by_fold": bool(result.ended_by_fold),
    }


def check_differential(reference: Simulator, candidate: Simulator, spec: VariantSpec, *, hands: int = 200,
                       seed: str = "conformance-v1") -> CheckResult:
    """Lockstep seeded hands. ``passed`` counts agreeing hands; decisions are noted."""
    out = CheckResult("differential")
    rng = random.Random(f"{seed}|{spec.ruleset_id}")
    decisions = 0
    kinds_seen: set[str] = set()
    for h in range(hands):
        setup = random_setup(spec, rng)
        try:
            a, b = reference.new_hand(setup), candidate.new_hand(setup)
        except Exception as exc:
            out.fail(f"hand {h}: could not start: {type(exc).__name__}: {exc}")
            continue
        error = None
        steps = 0
        while error is None and not a.is_complete:
            steps += 1
            if steps > 400:
                error = "hand did not finish in 400 decisions"
                break
            if b.is_complete or a.actor_seat != b.actor_seat:
                error = f"actor {a.actor_seat} vs {None if b.is_complete else b.actor_seat}"
                break
            la, lb = a.legal_actions(), b.legal_actions()
            if la != lb:
                error = f"legal actions differ at seat {a.actor_seat}: {la.model_dump()} vs {lb.model_dump()}"
                break
            menu = canonical_actions(la, pot_total=a.pot_total(), current_bet_to=_level(a, la),
                                     fractions=DIFFERENTIAL_FRACTIONS)
            illegal = [c.key for c in menu if not la.allows(c.action())]
            if illegal or not menu:
                error = f"canonical actions not legal: {illegal or 'empty menu'}"
                break
            choice = rng.choice(menu)
            if choice.kind.startswith("raise") and la.min_full_raise_to is not None and rng.random() < 0.2:
                action = Action(type="raise_to", amount=rng.randint(la.min_full_raise_to, la.max_raise_to or 0))
            else:
                action = choice.action()
                kinds_seen.add(choice.kind)
            decisions += 1
            try:
                a.apply(action)
            except ValueError as exc:
                error = f"reference rejected legal {action.model_dump(exclude_none=True)}: {exc}"
                break
            try:
                b.apply(action)
            except ValueError as exc:
                error = f"{candidate.name} rejected {action.model_dump(exclude_none=True)}: {exc}"
        if error is None and not b.is_complete:
            error = f"{candidate.name} hand still running after the reference finished"
        if error is None:
            ra, rb = _result_view(a.result()), _result_view(b.result())
            if ra != rb:
                error = f"results differ: {ra} vs {rb}"
            elif sum(ra["final_stacks"].values()) != sum(setup.stacks.values()):
                error = "chips not conserved"
        if error is None:
            out.passed += 1
        else:
            out.fail(f"hand {h} ({len(setup.stacks)} players): {error}")
    out.notes.append(f"{decisions} decisions; canonical kinds exercised: {', '.join(sorted(kinds_seen))}")
    return out


def _level(hand: SimHand, legal: LegalActions) -> int:
    return hand.current_level()


# -------------------------------------------------------------------- scoring


def check_scoring(scorer: Scorer, fixtures: Sequence[Mapping[str, Any]], tolerance: float = 1e-9) -> CheckResult:
    out = CheckResult("scoring")
    for fx in fixtures:
        got = scorer.icm(fx["stacks"], fx["prizes"])
        if "expect" in fx:
            want = [float(x) for x in fx["expect"]]
        elif "expect_fraction" in fx:
            want = [float(Fraction(n, d)) for n, d in fx["expect_fraction"]]
        else:
            want = None
        if abs(sum(got) - sum(fx["prizes"][: len(fx["stacks"])])) > tolerance * max(1.0, sum(fx["prizes"])):
            out.fail(f"{fx['id']}: awards {sum(got)} do not sum to the prizes")
        elif want is not None and any(abs(g - w) > tolerance * max(1.0, abs(w)) for g, w in zip(got, want)):
            out.fail(f"{fx['id']}: {got} != {want}")
        else:
            out.passed += 1
    return out


# ---------------------------------------------------------------------- suite


def run_conformance(ruleset_id: str, *, simulator: Simulator | None = None, reference: Simulator | None = None,
                    hands: int = 200, seed: str = "conformance-v1", fixture_root: Path = DEFAULT_FIXTURE_ROOT,
                    spec: VariantSpec | None = None, scorer: Scorer | None = None) -> ConformanceReport:
    sim = simulator or ArenaSimulator()
    ref = reference or ArenaSimulator()
    spec = spec or get_variant(ruleset_id)
    checks = [check_metadata(sim, spec), check_ranking(sim, spec)]
    if fixture_root.exists():
        fixtures = load_fixture_group("core", fixture_root) + load_fixture_group("eval", fixture_root)
        checks.append(check_fixtures(sim, ruleset_id, fixtures))
        icm_path = fixture_root / "tournament" / "icm.json"
        if icm_path.exists():
            checks.append(check_scoring(scorer or ArenaScorer(), json.loads(icm_path.read_text())))
    else:
        missing = CheckResult("fixtures", skipped=1)
        missing.notes.append(f"fixture root {fixture_root} not found")
        checks.append(missing)
    checks.append(check_differential(ref, sim, spec, hands=hands, seed=seed))
    return ConformanceReport(sim.name, ruleset_id, spec.variant_hash(), checks)


def run_all(rulesets: Sequence[str], **kwargs: Any) -> list[ConformanceReport]:
    return [run_conformance(r, **kwargs) for r in rulesets]


SimulatorFactory = Callable[[], Simulator]
