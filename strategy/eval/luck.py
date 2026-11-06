"""Zero-mean luck terms of a logged NLHE tournament, for variance-reduced paired evaluation.

In our own runs every hole card is in the private history, so chance can be separated
from choice. Every term is a sum, over the table's chance events, of
``w * (f(after) - E[f(after) | before])``, where the weight ``w`` and the function ``f``
are fixed before the cards arrive and the expectation is over the cards still unseen.
Each hand's deck is shuffled before the hand, independently of play, and no bot sees an
undealt card, so given everything logged before a card is dealt the card is uniform over
the unseen ones. Every term therefore has mean exactly zero whatever the bots do, and
subtracting any fixed multiple of it from the points leaves the expectation unchanged
(:mod:`strategy.eval.variance`). The one approximation is Monte Carlo for expectations
over five unseen board cards, seeded by what is already known.

Value: the focus seat's *points-equity* after a hand. Finished places keep their
points and survivors split the remaining prizes by ICM, which is the arena's own cap
rule, so the hand-by-hand changes add up to the official points.

Terms (points per tournament):

``allin``
    Board cards dealt after the hand's last decision are a pure runout: the seat's
    points-equity at the realized runout minus its mean over every runout (exact for
    one or two cards to come, Monte Carlo for five). Hands the seat is not in count
    too, because other players' busts move its equity; ``allin_in`` and ``allin_out``
    split the two.
``flop``, ``turn``, ``river`` (suffix ``_pot`` or ``_stack``)
    Board cards dealt while decisions remain, with the seat still in the hand: the
    change of its showdown share against the hands it still faces. ``_pot`` weights by
    the points the seat would gain by winning the current pot rather than losing it;
    ``_stack`` by the swing of winning or losing the effective stack.
``deal_stack``, ``deal_raw``
    The seat's hole cards: its showdown share against every hand dealt minus ``1/k``
    (all ``k`` hands are exchangeable), weighted by the effective-stack swing or not at all.
"""

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from arena.domain.cards import RANKS, SUITS
from strategy.endgame.points import icm_batch

RULESETS = ("practice_nlhe",)
STREETS = ("flop", "turn", "river")
TERMS = ("allin", "allin_in", "allin_out", *(f"{s}_{w}" for s in STREETS for w in ("pot", "stack")),
         "deal_stack", "deal_raw")
COUNTS = ("hands", "allin_hands", "street_cards")
EXACT_CARDS = 2
"""Expectations over at most this many unseen board cards are exact enumerations."""
_CARD = {r + s: 4 * i + j for i, r in enumerate(RANKS) for j, s in enumerate(SUITS)}
_EPS = 1e-9


@dataclass(frozen=True)
class LuckSettings:
    runout_samples: int = 2000
    """Monte Carlo boards for a preflop all-in runout."""
    equity_samples: int = 600
    """Monte Carlo boards for a showdown share with five cards to come."""


@dataclass
class _Hand:
    number: int
    start: dict[int, int]
    holes: dict[int, list[int]] = field(default_factory=dict)
    board: list[int] = field(default_factory=list)
    burned: list[int] = field(default_factory=list)
    deals: list[dict[str, Any]] = field(default_factory=list)
    """Board deals: index, cards, board and burns before, live seats and commitments."""
    last_decision: int = -1
    pots: list[tuple[int, list[int], dict[int, int]]] = field(default_factory=list)
    final: dict[int, int] = field(default_factory=dict)


def _hands(events: Iterable[Mapping[str, Any]]) -> Iterable[_Hand]:
    hand: _Hand | None = None
    committed: dict[int, int] = {}
    folded: set[int] = set()
    pending_burn: list[int] = []
    for i, e in enumerate(events):
        kind, d = e["type"], e["data"]
        if kind == "HandStarted":
            hand = _Hand(int(d["hand_number"]), {int(s): int(v) for s, v in d["start_stacks"].items()})
            committed = dict.fromkeys(hand.start, 0)
            folded = set()
            pending_burn = []
        elif hand is None:
            continue
        elif kind == "HoleDealt":
            hand.holes.setdefault(int(d["seat"]), []).append(_CARD[d["card"]])
        elif kind in ("BlindPosted", "AntePosted"):
            committed[int(d["seat"])] += int(d["amount"])
        elif kind == "ExcessReturned":
            committed[int(d["seat"])] -= int(d["amount"])
        elif kind == "ActionApplied":
            committed[int(d["seat"])] += int(d["added"])
            if d["action"]["type"] == "fold":
                folded.add(int(d["seat"]))
        elif kind == "DecisionRequested":
            hand.last_decision = i
        elif kind == "CardBurned":
            pending_burn.append(_CARD[d["card"]])
        elif kind == "BoardDealt":
            cards = [_CARD[c] for c in d["cards"]]
            hand.deals.append({"index": i, "street": d["street"], "cards": cards, "board": list(hand.board),
                               "burned": list(hand.burned), "new_burns": list(pending_burn),
                               "live": [s for s in hand.start if s not in folded and s in hand.holes],
                               "committed": dict(committed)})
            hand.burned += pending_burn
            pending_burn = []
            hand.board += cards
        elif kind == "PotAwarded":
            hand.pots.append((int(d["amount"]), [int(s) for s in d["eligible"]],
                              {int(s): int(v) for s, v in d["awards"].items()}))
        elif kind == "HandCompleted":
            hand.final = {int(s): int(v) for s, v in d["final_stacks"].items()}
            yield hand
            hand = None


def _seed(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, ("strategy-luck-v1", *parts))).encode()).digest()[:8], "big")


def _boards(unseen: Sequence[int], m: int, samples: int, key: tuple) -> np.ndarray:
    """Every completion of ``m`` cards from ``unseen`` (exact up to :data:`EXACT_CARDS`), else a seeded sample."""
    rest = np.asarray(sorted(unseen), dtype=np.int64)
    if m == 0:
        return np.zeros((1, 0), dtype=np.int64)
    if m <= EXACT_CARDS:
        return np.array(list(itertools.combinations(rest.tolist(), m)), dtype=np.int64)
    rng = np.random.default_rng(_seed(*key))
    return rest[np.argsort(rng.random((samples, len(rest))), axis=1)[:, :m]]


def _scores(holes: Sequence[Sequence[int]], board: Sequence[int], extra: np.ndarray) -> np.ndarray:
    from strategy.eval.kernels import score_boards

    return score_boards(np.asarray(holes, dtype=np.int64).reshape(-1, 2), np.asarray(board, dtype=np.int64), extra)


def _shares(scores: np.ndarray) -> np.ndarray:
    win = scores == scores.max(1, keepdims=True)
    return win / win.sum(1, keepdims=True)


def _bust_points(ci: int, bust: Sequence[int], order: Sequence[int], start: Mapping[int, int],
                 payouts: Sequence[float]) -> float:
    """Finish points of column ``ci`` busting with the columns ``bust`` (as
    :class:`arena.tournament.scoring.Standings`: larger start stacks finish higher, ties share)."""
    pos = len(order) - len(bust) + 1
    for stack in sorted({start[order[j]] for j in bust}, reverse=True):
        group = [j for j in bust if start[order[j]] == stack]
        if ci in group:
            return sum(payouts[p - 1] for p in range(pos, pos + len(group))) / len(group)
        pos += len(group)
    raise ValueError("column is not among the busted")


class _Values:
    """Deferred points-equity evaluations of one seat, solved with one ICM call per survivor set."""

    def __init__(self, seat: int, payouts: Sequence[float]) -> None:
        self.seat = seat
        self.payouts = [float(p) for p in payouts]
        self.jobs: list[tuple[tuple[int, ...], Mapping[int, int], np.ndarray]] = []

    def add(self, start: Mapping[int, int], finals: np.ndarray) -> int:
        """Queue end-of-hand stack rows (columns: the seats of ``start`` in sorted order)."""
        self.jobs.append((tuple(sorted(start)), start, np.atleast_2d(np.asarray(finals, dtype=np.float64))))
        return len(self.jobs) - 1

    def solve(self) -> list[np.ndarray]:
        out = [np.empty(len(f)) for _, _, f in self.jobs]
        groups: dict[tuple[int, ...], list[tuple[int, np.ndarray, np.ndarray]]] = {}
        for n, (order, start, finals) in enumerate(self.jobs):
            k = len(order)
            ci = order.index(self.seat)
            keys = (finals <= _EPS).astype(np.int64) @ (1 << np.arange(k, dtype=np.int64))
            for key in np.unique(keys):
                rows = np.flatnonzero(keys == key)
                bust = [j for j in range(k) if key >> j & 1]
                if ci in bust:
                    out[n][rows] = _bust_points(ci, bust, order, start, self.payouts)
                    continue
                surv = [j for j in range(k) if j not in bust]
                if len(surv) == 1:
                    out[n][rows] = self.payouts[0]
                    continue
                survivors = tuple(order[j] for j in surv)
                groups.setdefault(survivors, []).append((n, rows, finals[np.ix_(rows, surv)]))
        for survivors, parts in groups.items():
            ci = survivors.index(self.seat)
            v = icm_batch(np.concatenate([p[2] for p in parts]), self.payouts[: len(survivors)])[:, ci]
            at = 0
            for n, rows, block in parts:
                out[n][rows] = v[at: at + len(block)]
                at += len(block)
        return out


def points_equity(seat: int, start: Mapping[int, int], finals: np.ndarray, payouts: Sequence[float]) -> np.ndarray:
    """``seat``'s points-equity for each row of end-of-hand stacks ``finals`` (columns: the seats
    of ``start``, live at hand start, in sorted order): its finish points if it busts in the
    hand, else its ICM share of the top ``len(survivors)`` prizes."""
    v = _Values(seat, payouts)
    v.add(start, finals)
    return v.solve()[0]


class _Table:
    def __init__(self, seat: int, payouts: Sequence[float], settings: LuckSettings) -> None:
        self.seat = seat
        self.settings = settings
        self.values = _Values(seat, payouts)
        self.pending: list[tuple[str, int, Any]] = []
        """``(term, job, f)``: the term gains ``f(values of the job)``."""
        self.counts = dict.fromkeys(COUNTS, 0)
        self.raw = 0.0

    def rows(self, h: _Hand, stacks: Sequence[Mapping[int, float]]) -> np.ndarray:
        order = sorted(h.start)
        return np.array([[s[x] for x in order] for s in stacks], dtype=np.float64)

    def swing(self, h: _Hand, terms: Sequence[str], x: float, win: Mapping[int, float],
              lose: Mapping[int, float]) -> None:
        """Each term gains ``(V(win) - V(lose)) * x``."""
        job = self.values.add(h.start, self.rows(h, [win, lose]))
        for term in terms:
            self.pending.append((term, job, lambda v, x=x: (v[0] - v[1]) * x))

    def effective(self, h: _Hand, live: Sequence[int]) -> tuple[dict[int, float], dict[int, float]]:
        """Start stacks after the seat wins, or loses, the effective stack against the largest live stack."""
        big = max((s for s in live if s != self.seat), key=lambda s: (h.start[s], -s))
        eff = min(h.start[self.seat], h.start[big])
        win, lose = {s: float(v) for s, v in h.start.items()}, {s: float(v) for s, v in h.start.items()}
        win[self.seat] += eff
        win[big] -= eff
        lose[self.seat] -= eff
        lose[big] += eff
        return win, lose

    def share(self, h: _Hand, live: Sequence[int], board: Sequence[int], dead: Iterable[int], samples: int,
              tag: str) -> float:
        """Expected showdown share of the seat against ``live`` given ``board``."""
        known = set(dead) | set(board) | {c for cs in h.holes.values() for c in cs}
        unseen = [c for c in range(52) if c not in known]
        live = sorted(live)
        extra = _boards(unseen, 5 - len(board), samples, (tag, h.number, tuple(live), tuple(board),
                                                           tuple(sorted(known))))
        return float(_shares(_scores([h.holes[s] for s in live], board, extra))[:, live.index(self.seat)].mean())

    def hand(self, h: _Hand) -> None:
        if self.seat not in h.start or self.seat not in h.holes:
            return
        self.counts["hands"] += 1
        dealt = sorted(h.holes)
        x = self.share(h, dealt, [], [], self.settings.equity_samples, "deal") - 1.0 / len(dealt)
        self.raw += x
        self.swing(h, ["deal_stack"], x, *self.effective(h, dealt))
        runout = [d for d in h.deals if d["index"] > h.last_decision]
        for d in h.deals:
            if d not in runout and self.seat in d["live"] and len(d["live"]) >= 2:
                self.street(h, d)
        if runout:
            self.allin(h, runout[0])

    def street(self, h: _Hand, d: Mapping[str, Any]) -> None:
        live, committed = d["live"], d["committed"]
        samples = self.settings.equity_samples
        before = self.share(h, live, d["board"], d["burned"], samples, "street")
        after = self.share(h, live, d["board"] + d["cards"], d["burned"] + d["new_burns"], samples, "street")
        behind = {s: float(h.start[s] - committed[s]) for s in h.start}
        others = [s for s in live if s != self.seat]
        pot = sum(committed.values())
        win, lose = dict(behind), dict(behind)
        win[self.seat] += pot
        for s in others:
            lose[s] += pot / len(others)
        street = d["street"]
        self.swing(h, [f"{street}_pot"], after - before, win, lose)
        self.swing(h, [f"{street}_stack"], after - before, *self.effective(h, live))
        self.counts["street_cards"] += 1

    def allin(self, h: _Hand, first: Mapping[str, Any]) -> None:
        live = sorted(first["live"])
        if len(live) < 2:
            return
        board = first["board"]
        known = set(board) | set(first["burned"]) | {c for cs in h.holes.values() for c in cs}
        unseen = [c for c in range(52) if c not in known]
        extra = _boards(unseen, 5 - len(board), self.settings.runout_samples,
                        ("runout", h.number, tuple(live), tuple(board), tuple(sorted(known))))
        holes = [h.holes[s] for s in live]
        realized = self.finals(h, live, _scores(holes, board, np.asarray([h.board[len(board):]], dtype=np.int64)))
        alt, counts = np.unique(self.finals(h, live, _scores(holes, board, extra)), axis=0, return_counts=True)
        job = self.values.add(h.start, np.concatenate([realized, alt]))
        mean = lambda v, c=counts / counts.sum(): v[0] - float(v[1:] @ c)  # noqa: E731
        for term in ("allin", "allin_in" if self.seat in live else "allin_out"):
            self.pending.append((term, job, mean))
        self.counts["allin_hands"] += 1

    def finals(self, h: _Hand, live: Sequence[int], scores: np.ndarray) -> np.ndarray:
        """End-of-hand stack rows for each runout: the realized stacks minus the realized awards
        plus each pot split (fractionally) among its best eligible hands."""
        order = sorted(h.start)
        idx = {s: j for j, s in enumerate(order)}
        base = np.array([float(h.final.get(s, 0)) for s in order])
        out = np.tile(base, (len(scores), 1))
        col = {s: j for j, s in enumerate(live)}
        for amount, eligible, awards in h.pots:
            for s, x in awards.items():
                out[:, idx[s]] -= x
            elig = [s for s in eligible if s in col]
            if len(elig) < 2:
                for s, x in awards.items():
                    out[:, idx[s]] += x
                continue
            share = _shares(scores[:, [col[s] for s in elig]]) * amount
            for j, s in enumerate(elig):
                out[:, idx[s]] += share[:, j]
        return out

    def result(self) -> dict[str, Any]:
        terms = dict.fromkeys(TERMS, 0.0)
        terms["deal_raw"] = self.raw
        solved = self.values.solve()
        for term, job, f in self.pending:
            terms[term] += float(f(solved[job]))
        return {"terms": terms, "counts": self.counts}


def table_luck(events: Sequence[Mapping[str, Any]], seat: int,
               settings: LuckSettings | None = None) -> dict[str, Any]:
    """Every term of :data:`TERMS` for ``seat`` over one logged table, plus counts."""
    config = next(e["data"]["config"] for e in events if e["type"] == "TableStarted")
    t = _Table(seat, config["payout_points"], settings or LuckSettings())
    for h in _hands(events):
        t.hand(h)
    return t.result()
