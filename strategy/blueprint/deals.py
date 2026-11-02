"""Deals for blueprint training: card buckets on every street and showdown values.

Pure Python over :mod:`strategy.blueprint.cards`, the module the deployed bot uses, so
training sees exactly the buckets the bot will compute. A chunk of deals is a function
of ``(ruleset_id, seed)`` only, which makes resumed runs replay the same deal stream.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from strategy.blueprint.cards import CardAbstraction
from strategy.endgame.classes import hand_classes
from strategy.features.cards import capabilities, evaluate

MAX_PLAYERS = 8

_CTX: dict[str, tuple] = {}


def _context(ruleset_id: str):
    ctx = _CTX.get(ruleset_id)
    if ctx is None:
        caps = capabilities(ruleset_id)
        ctx = _CTX[ruleset_id] = (caps, CardAbstraction(caps), hand_classes(ruleset_id))
    return ctx


def class_index(ruleset_id: str, hole) -> int:
    _, _, classes = _context(ruleset_id)
    return classes.index(tuple(hole))


def make_deals(ruleset_id: str, seed: int, count: int, players: int = MAX_PLAYERS) -> dict[str, np.ndarray]:
    caps, cards, classes = _context(ruleset_id)
    rng = random.Random(f"blueprint-deals|{ruleset_id}|{seed}")
    deck = list(caps.deck)
    buckets = np.zeros((count, MAX_PLAYERS, 4), dtype=np.uint8)
    ranks = np.zeros((count, MAX_PLAYERS), dtype=np.int64)
    cls = np.zeros((count, MAX_PLAYERS), dtype=np.int16)
    need = 2 * players + 5
    for j in range(count):
        drawn = rng.sample(deck, need)
        board = drawn[2 * players:]
        for p in range(players):
            hole = drawn[2 * p: 2 * p + 2]
            buckets[j, p] = cards.street_buckets(hole, board)
            ranks[j, p] = evaluate(hole, board, caps)
            cls[j, p] = classes.index(tuple(hole))
    return {"buckets": buckets, "ranks": ranks, "classes": cls}


def _chunk(args) -> dict[str, np.ndarray]:
    ruleset_id, seed, count, players = args
    return make_deals(ruleset_id, seed, count, players)


def deal_stream(ruleset_id: str, first_chunk: int, chunk_size: int, *, jobs: int,
                players: int = MAX_PLAYERS, base_seed: int = 0) -> Iterator[tuple[int, dict[str, np.ndarray]]]:
    """``(chunk index, deals)`` for chunk ``first_chunk`` onward, produced by ``jobs`` processes."""
    def args(i: int):
        return ruleset_id, base_seed * 1_000_003 + i, chunk_size, players

    if jobs <= 1:
        i = first_chunk
        while True:
            yield i, _chunk(args(i))
            i += 1
    ex = ProcessPoolExecutor(jobs)
    try:
        ahead = 2 * jobs
        pending = {}
        i = first_chunk
        for k in range(ahead):
            pending[i + k] = ex.submit(_chunk, args(i + k))
        nxt = i + ahead
        while True:
            fut = pending.pop(i)
            pending[nxt] = ex.submit(_chunk, args(nxt))
            nxt += 1
            yield i, fut.result()
            i += 1
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
