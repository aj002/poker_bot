"""Dense policy tables for the league kernels, and new checkpoint packs as overrides (numpy only).

A *table* is a ``uint8`` array of shape ``(NUM_KEYS * NUM_ACTIONS,)``: the row the
deployed bot would look up for every blueprint key (back-off included), as
:meth:`~strategy.blueprint.policy.PolicyPack.resolved` gives it but kept in bytes
(42 MB rather than 168 MB). An all-zero row is a miss: the kernels play it uniformly.

A league checkpoint is its parent's pack with level-0 rows replaced or inserted at the
keys where the learner deviated (:func:`write_override_pack`). Lookup takes the finest
level holding a key, so an inserted level-0 row overrides only that key; every other
lookup is the parent's. The deployed bot and the pack format are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from strategy.blueprint import abstraction as A
from strategy.blueprint.policy import LEVEL_DIVISORS, PolicyPack, quantize, write_levels

ROW = A.NUM_ACTIONS
SLOTS = A.NUM_KEYS * A.NUM_ACTIONS


def pack_table(pack: PolicyPack) -> np.ndarray:
    """The resolved ``uint8`` row of every key (zeros where every level misses)."""
    out = np.zeros((A.NUM_KEYS, ROW), np.uint8)
    for level in reversed(range(len(LEVEL_DIVISORS))):
        div = LEVEL_DIVISORS[level]
        keys = pack.keys[level].astype(np.int64)
        if not keys.size:
            continue
        coarse = np.full(A.NUM_KEYS // div, -1, np.int64)
        coarse[keys] = np.arange(keys.size)
        idx = np.repeat(coarse, div) if div > 1 else coarse
        hit = idx >= 0
        out[hit] = pack.probs[level][idx[hit]]
    return out.reshape(-1)


def uniform_table() -> np.ndarray:
    """Every legal action equally likely: the kernels' reading of an all-zero row."""
    return np.zeros(SLOTS, np.uint8)


@dataclass(frozen=True)
class Deviations:
    keys: np.ndarray
    """Sorted ``int64`` keys where the learner leaves its base table."""
    rows: np.ndarray
    """``uint8`` rows (sum 255) at those keys."""
    best: np.ndarray
    """The greedy action at each key."""

    def __len__(self) -> int:
        return int(self.keys.size)

    def by_street(self) -> list[int]:
        streets = np.searchsorted(np.asarray(A.OFFSETS[1:4], np.int64), self.keys, side="right")
        return np.bincount(streets, minlength=A.STREETS).astype(int).tolist()

    def by_action(self) -> dict[str, int]:
        counts = np.bincount(self.best, minlength=ROW)
        return {A.ACTION_NAMES[a]: int(counts[a]) for a in range(ROW)}


def deviations(regret: np.ndarray, sq: np.ndarray, base: np.ndarray, *, z: float, mix: float) -> Deviations:
    """Keys whose largest cumulative regret clears ``z`` standard errors, and their new rows.

    The new row is ``(1 - mix) * base + mix * onehot(best)``; ``mix = 1`` is the greedy
    response itself. A key whose base row is a miss gets the one-hot row.
    """
    if not 0.0 < mix <= 1.0:
        raise ValueError("mix must be in (0, 1]")
    r = regret.reshape(-1, ROW)
    s = sq.reshape(-1, ROW)
    best = r.argmax(axis=1)
    idx = np.arange(r.shape[0])
    top = r[idx, best].astype(np.float64)
    gate = (top > 0.0) & (top * top > (z * z) * s[idx, best].astype(np.float64))
    keys = np.nonzero(gate)[0].astype(np.int64)
    b = best[keys]
    rows = base.reshape(-1, ROW)[keys].astype(np.float64)
    tot = rows.sum(axis=1, keepdims=True)
    rows = np.where(tot > 0, rows / np.where(tot > 0, tot, 1.0), 0.0) * (1.0 - mix)
    rows[np.arange(keys.size), b] += mix
    return Deviations(keys, quantize(rows), b.astype(np.int64))


def apply_deviations(base: np.ndarray, dev: Deviations) -> np.ndarray:
    out = base.copy().reshape(-1, ROW)
    out[dev.keys] = dev.rows
    return out.reshape(-1)


def write_override_pack(parent: Path, dev: Deviations, out: Path, provenance: dict) -> dict:
    """``parent``'s pack with ``dev`` as level-0 rows (replacing or inserting), written to ``out``."""
    pack = PolicyPack(parent)
    with np.load(parent) as z:
        arrays = {k: np.array(z[k]) for k in z.files if k != "__provenance__"}
    keys0 = arrays["keys0"].astype(np.int64)
    probs0 = arrays["probs0"]
    mass0 = arrays["mass0"]
    keep = ~np.isin(keys0, dev.keys)
    keys = np.concatenate([keys0[keep], dev.keys])
    order = np.argsort(keys, kind="stable")
    arrays["keys0"] = keys[order].astype(np.int32)
    arrays["probs0"] = np.concatenate([probs0[keep], dev.rows])[order]
    arrays["mass0"] = np.concatenate([mass0[keep], np.full(len(dev), 255, np.uint8)])[order]
    prov = dict(provenance, parent={"path": str(parent), "data_sha256": pack.provenance["data_sha256"]},
                overrides=len(dev), overrides_by_street=dev.by_street(), overrides_by_action=dev.by_action())
    min_mass = tuple(pack.provenance.get("min_mass", (0, 0, 0, 0)))
    return write_levels(out, arrays, prov, min_mass)
