"""Behaviour-cloned opponent tables: real-engine bots read through the blueprint abstraction (offline).

The league learner trains in the abstract game, where every seat must play a policy
table over blueprint keys. Frozen checkpoints are such tables already. The opponent
population and the interpretable baseline are not, so they are *cloned*: complete
in-process tournaments are played on the authoritative engine, and at every decision of
a recorded seat the key the blueprint runtime would compute for that seat
(:meth:`~strategy.blueprint.runtime.BlueprintPolicy.features`) and the abstract slot of
the action actually taken are counted. A raise maps to the legal raise slot whose
target is nearest the real amount.

Counts become a table by hierarchical Dirichlet smoothing over the pack's back-off
levels: from the coarsest level (pressure, acted, initiative and depth pooled) to the
trained key, each level's estimate is ``(counts + alpha * coarser estimate) / (n +
alpha)``, and the prior above the coarsest level is the blueprint's own row. A family
model is therefore the blueprint wherever the family was never observed and the
family's measured frequencies where it was.

Only training-partition members are recorded (:func:`training_lineups`); validation
and holdout members never shape a table.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from strategy.blueprint import abstraction as A
from strategy.blueprint.policy import LEVEL_DIVISORS, quantize

ROW = A.NUM_ACTIONS
RAISE_SLOTS = (A.MIN_RAISE, A.RAISE_A, A.RAISE_B, A.ALL_IN)
DEFAULT_ALPHA = 2.0


def action_slot(action: Mapping, mask: int, targets: Sequence[int]) -> int | None:
    """The abstract slot of a real action at a decision with ``(mask, targets)``, or None."""
    kind = action.get("type")
    if kind == "fold":
        return A.FOLD if (mask >> A.FOLD) & 1 else None
    if kind in ("check", "call"):
        return A.CALL
    if kind not in ("raise_to", "bet", "raise"):
        return None
    amount = int(action.get("amount") or 0)
    legal = [a for a in RAISE_SLOTS if (mask >> a) & 1]
    if not legal:
        return None
    if (mask >> A.ALL_IN) & 1 and amount >= targets[A.ALL_IN]:
        return A.ALL_IN
    return min(legal, key=lambda a: (abs(amount - targets[a]), a))


@dataclass
class Counts:
    keys: np.ndarray
    """Sorted ``int64`` keys."""
    counts: np.ndarray
    """``(len(keys), ROW)`` ``uint32`` action counts."""

    @classmethod
    def from_dict(cls, d: Mapping[int, Sequence[int]]) -> Counts:
        keys = np.array(sorted(d), np.int64)
        counts = np.array([d[k] for k in keys], np.uint32).reshape(-1, ROW)
        return cls(keys, counts)

    @classmethod
    def merge(cls, parts: Sequence[Counts]) -> Counts:
        keys = np.concatenate([p.keys for p in parts]) if parts else np.zeros(0, np.int64)
        counts = np.concatenate([p.counts for p in parts]) if parts else np.zeros((0, ROW), np.uint32)
        uk, inv = np.unique(keys, return_inverse=True)
        out = np.zeros((uk.size, ROW), np.uint64)
        np.add.at(out, inv, counts.astype(np.uint64))
        return cls(uk, out.astype(np.uint32))

    @property
    def decisions(self) -> int:
        return int(self.counts.sum())

    def street_decisions(self) -> list[int]:
        streets = np.searchsorted(np.asarray(A.OFFSETS[1:4], np.int64), self.keys, side="right")
        return np.bincount(streets, weights=self.counts.sum(axis=1), minlength=A.STREETS).astype(int).tolist()

    def action_shares(self) -> dict[str, float]:
        tot = self.counts.sum(axis=0).astype(float)
        s = tot.sum() or 1.0
        return {A.ACTION_NAMES[a]: round(float(tot[a] / s), 4) for a in range(ROW)}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, keys=self.keys, counts=self.counts)

    @classmethod
    def load(cls, path: Path) -> Counts:
        with np.load(path) as z:
            return cls(z["keys"].astype(np.int64), z["counts"].astype(np.uint32))


def model_table(c: Counts, prior: np.ndarray, *, alpha: float = DEFAULT_ALPHA) -> np.ndarray:
    """Hierarchically smoothed ``uint8`` table from counts, shrunk toward ``prior`` (a table)."""
    est = prior.reshape(-1, ROW).astype(np.float32)
    tot = est.sum(axis=1, keepdims=True)
    est = np.where(tot > 0, est / np.where(tot > 0, tot, 1.0), np.float32(1.0 / ROW)).astype(np.float32)
    dense = np.zeros((A.NUM_KEYS, ROW), np.float32)
    dense[c.keys] = c.counts
    for level in reversed(range(len(LEVEL_DIVISORS))):
        div = LEVEL_DIVISORS[level]
        g = dense.reshape(-1, div, ROW).sum(axis=1) if div > 1 else dense
        n = g.sum(axis=1, keepdims=True)
        if div > 1:
            g = np.repeat(g, div, axis=0)
            n = np.repeat(n, div, axis=0)
        est = (g + np.float32(alpha) * est) / (n + np.float32(alpha))
        del g, n
    return quantize_fast(est)


def quantize_fast(p: np.ndarray, chunk: int = 1 << 20) -> np.ndarray:
    out = np.empty((p.shape[0], ROW), np.uint8)
    for i in range(0, p.shape[0], chunk):
        out[i:i + chunk] = quantize(p[i:i + chunk])
    return out.reshape(-1)


# ------------------------------------------------------------------ recording


def _record_lineup(args) -> tuple[dict[str, Counts], dict[str, int]]:
    from arena.agents.worker import play_table
    from arena.benchmark.runner import Entrant, TableRunner, schedule, table_info
    from arena.domain.models import TournamentConfig
    from arena.table.projections import observation
    from arena.table.service import seeded_table
    from strategy.blueprint.runtime import BlueprintPolicy
    from strategy.eval.policies import resolve_policy

    config_json, policies, labels, blocks, seed = args
    config = TournamentConfig.model_validate_json(config_json)
    entrants = []
    for i, spec in enumerate(policies):
        p = resolve_policy(spec)
        entrants.append(Entrant(f"e{i}:{p.name}", p.path / "bot.py" if p.path.is_dir() else p.path, p.builtin))
    runner = TableRunner(config, entrants, isolation="in_process")
    feat = BlueprintPolicy(config.ruleset_id)
    acc: dict[str, dict[int, list[int]]] = {lbl: {} for lbl in labels if lbl}
    stats = {"recorded": 0, "unmapped": 0, "outside": 0, "fallbacks": 0}
    for a in schedule(config, len(policies), blocks, seed):
        seat_label = {seat: labels[e] for seat, e in a.seats.items()}
        agents = {}
        for seat, entrant in a.seats.items():
            agents[seat] = runner._agent(seat, entrant, table_info(config, a, seat, runner.budget_ns))
            agents[seat].start()
        table = seeded_table(config, a.seed, table_id=a.table_id)
        pending: dict[str, object] = {}

        def before(_index: int, seat: int, table=table, seat_label=seat_label, pending=pending) -> None:
            pending["f"] = feat.features(observation(table, seat)) if seat_label.get(seat) else None

        def after(seat: int, decision, _table, seat_label=seat_label, pending=pending) -> None:
            lbl = seat_label.get(seat)
            if not lbl:
                return
            f = pending.get("f")
            if f is None:
                stats["outside"] += 1
                return
            if decision.fallback is not None or decision.action is None:
                stats["fallbacks"] += 1
                return
            key, mask, targets = f
            slot = action_slot(decision.action, mask, targets)
            if slot is None:
                stats["unmapped"] += 1
                return
            row = acc[lbl].setdefault(int(key), [0] * ROW)
            row[slot] += 1
            stats["recorded"] += 1

        try:
            play_table(table, agents, before_decision=before, on_decision=after)
        finally:
            for agent in agents.values():
                agent.close()
    return {lbl: Counts.from_dict(d) for lbl, d in acc.items()}, stats


def record(lineups: Sequence[Sequence[str]], labels: Sequence[Sequence[str | None]], *, blocks: int, base_seed: str,
           jobs: int, ruleset_id: str = "practice_nlhe") -> tuple[dict[str, Counts], dict[str, int]]:
    """Counts per label over ``blocks`` seat-rotated blocks of each lineup (``None`` labels are not recorded)."""
    from arena.domain.models import TournamentConfig

    config = TournamentConfig(ruleset_id=ruleset_id).model_dump_json()
    tasks = [(config, list(lu), list(lb), blocks, f"{base_seed}|lineup{i}") for i, (lu, lb) in
             enumerate(zip(lineups, labels, strict=True))]
    parts: dict[str, list[Counts]] = {}
    stats = {"recorded": 0, "unmapped": 0, "outside": 0, "fallbacks": 0}
    if jobs > 1:
        with ProcessPoolExecutor(jobs) as ex:
            results = list(ex.map(_record_lineup, tasks))
    else:
        results = [_record_lineup(t) for t in tasks]
    for counts, st in results:
        for lbl, c in counts.items():
            parts.setdefault(lbl, []).append(c)
        for k, v in st.items():
            stats[k] += v
    return {lbl: Counts.merge(ps) for lbl, ps in sorted(parts.items())}, stats


def training_lineups(ruleset_id: str, count: int, seed: int, *, fixed: Sequence[str] = (),
                     fixed_labels: Sequence[str | None] = ()) -> tuple[list[list[str]], list[list[str | None]]]:
    """Lineups of distinct-family training members after the ``fixed`` seats, and each seat's label.

    A member's label is its family; ``fixed_labels`` label the fixed seats (None: not recorded).
    """
    from strategy.population import PopulationSource
    from strategy.population.families import CHECKPOINT_FAMILY
    from strategy.population.registry import VALIDATION_ONLY_PAIRS

    members = sorted(PopulationSource().partition("train").eligible(ruleset_id), key=lambda m: m.member_id)
    by_family: dict[str, list] = {}
    for m in members:
        if m.family != CHECKPOINT_FAMILY:
            by_family.setdefault(m.family, []).append(m)
    families = sorted(by_family)
    seats = 8 - len(fixed)
    rng = random.Random(f"league-clone-lineups|{ruleset_id}|{seed}")
    lineups, labels = [], []
    for _ in range(count):
        fams = rng.sample(families, seats)
        while any(a in fams and b in fams for a, b in VALIDATION_ONLY_PAIRS):
            fams = rng.sample(families, seats)
        picks = [by_family[f][rng.randrange(len(by_family[f]))] for f in fams]
        lineups.append(list(fixed) + [m.policy for m in picks])
        labels.append(list(fixed_labels or [None] * len(fixed)) + [m.family for m in picks])
    return lineups, labels
