"""The compact blueprint policy table and its builder.

A pack is one ``.npz``: for each back-off level a sorted ``int32`` key array and a
``uint8`` probability row per key (six action slots, summing to 255), plus provenance.
Level 0 is the trained information set; coarser levels pool the strategy sums of
information sets that differ only in

* level 1: the pressure bucket,
* level 2: also the acted and initiative flags,
* level 3: also the depth / SPR bucket,

which are contiguous blocks of the key space (those fields are its lowest digits), so a
coarse key is ``key // LEVEL_DIVISORS[level]``. A level keeps a row only when its
reach-weighted strategy mass is at least that level's threshold. Lookup takes the
finest level present; a miss on every level returns ``None`` and the bot falls back.

Loading needs only numpy; a lookup is a binary search per level (a few microseconds).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from strategy.blueprint import abstraction as A

LEVEL_DIVISORS = (1, A.R_PRESSURE, A.R_PRESSURE * A.R_INIT * A.R_ACTED,
                  A.R_PRESSURE * A.R_INIT * A.R_ACTED * A.R_DEPTH)
DEFAULT_MIN_MASS = (6.0, 12.0, 24.0, 24.0)
PACK_FORMAT = 1


def quantize(probs: np.ndarray) -> np.ndarray:
    """Rows of probabilities to uint8 rows summing to 255 (largest remainder)."""
    p = np.clip(probs, 0.0, None)
    tot = p.sum(axis=1, keepdims=True)
    p = np.where(tot > 0, p / np.where(tot > 0, tot, 1.0), 1.0 / p.shape[1])
    scaled = p * 255.0
    q = np.floor(scaled).astype(np.int64)
    short = 255 - q.sum(axis=1)
    rem = scaled - q
    order = np.argsort(-rem, axis=1, kind="stable")
    for k in range(p.shape[1]):
        add = short > k
        q[np.arange(len(q))[add], order[add, k]] += 1
    return q.astype(np.uint8)


def build_levels(strat: np.ndarray, min_mass=DEFAULT_MIN_MASS) -> dict[str, np.ndarray]:
    """Back-off levels from strategy sums over keys ``0 .. strat.size / NUM_ACTIONS``."""
    s = np.asarray(strat, dtype=np.float64).reshape(-1, A.NUM_ACTIONS)
    n = s.shape[0]
    if n % LEVEL_DIVISORS[-1]:
        raise ValueError("key count must be a multiple of the coarsest level divisor")
    out: dict[str, np.ndarray] = {}
    for level, div in enumerate(LEVEL_DIVISORS):
        g = s.reshape(n // div, div, A.NUM_ACTIONS).sum(axis=1) if div > 1 else s
        mass = g.sum(axis=1)
        keep = np.nonzero(mass >= min_mass[level])[0]
        out[f"keys{level}"] = keep.astype(np.int32)
        out[f"probs{level}"] = quantize(g[keep])
        out[f"mass{level}"] = np.minimum(255, np.round(np.log2(1 + mass[keep]) * 8)).astype(np.uint8)
    return out


def array_digest(arrays: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for k in sorted(arrays):
        h.update(k.encode())
        h.update(np.ascontiguousarray(arrays[k]).tobytes())
    return h.hexdigest()


def write_pack(path: Path, strat: np.ndarray, provenance: dict, min_mass=DEFAULT_MIN_MASS) -> dict:
    return write_levels(path, build_levels(strat, min_mass), provenance, min_mass)


def write_levels(path: Path, arrays: dict[str, np.ndarray], provenance: dict, min_mass=DEFAULT_MIN_MASS) -> dict:
    prov = dict(provenance, format=PACK_FORMAT, abstraction=A.describe(), level_divisors=list(LEVEL_DIVISORS),
                min_mass=list(min_mass), rows=[int(arrays[f"keys{k}"].size) for k in range(len(LEVEL_DIVISORS))],
                data_sha256=array_digest(arrays))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, __provenance__=np.array(json.dumps(prov, sort_keys=True)), **arrays)
    tmp.replace(path)
    return prov


class PolicyPack:
    __slots__ = ("provenance", "keys", "probs", "path")

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with np.load(self.path) as z:
            self.provenance = json.loads(str(z["__provenance__"]))
            arrays = {k: z[k] for k in z.files if k != "__provenance__"}
        if self.provenance.get("format") != PACK_FORMAT:
            raise ValueError(f"{path}: unsupported pack format")
        if array_digest(arrays) != self.provenance["data_sha256"]:
            raise ValueError(f"{path}: pack data hash mismatch")
        if self.provenance["abstraction"] != json.loads(json.dumps(A.describe())):
            raise ValueError(f"{path}: pack was built for another abstraction")
        n = len(LEVEL_DIVISORS)
        self.keys = [arrays[f"keys{k}"] for k in range(n)]
        self.probs = [arrays[f"probs{k}"] for k in range(n)]

    def lookup(self, key: int) -> tuple[tuple[int, ...], int] | None:
        """``(uint8 row, level)`` for the finest level holding ``key``, or None."""
        for level, div in enumerate(LEVEL_DIVISORS):
            k = key // div
            keys = self.keys[level]
            i = int(np.searchsorted(keys, k))
            if i < keys.size and int(keys[i]) == k:
                return tuple(int(x) for x in self.probs[level][i]), level
        return None

    def dense(self, level: int = 0) -> np.ndarray:
        """Float probabilities over the full key space at one level (zeros where absent)."""
        div = LEVEL_DIVISORS[level]
        out = np.zeros((A.NUM_KEYS // div, A.NUM_ACTIONS), np.float32)
        out[self.keys[level]] = self.probs[level] / 255.0
        return out

    def nbytes(self) -> int:
        return sum(a.nbytes for a in self.keys) + sum(a.nbytes for a in self.probs)


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Build a blueprint policy pack from a run checkpoint or milestone")
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--milestone", help="label under <run>/milestones (default: the latest checkpoint)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    cfg = json.loads((a.run / "config.json").read_text())
    if a.milestone:
        with np.load(a.run / "milestones" / f"{a.milestone}.npz") as z:
            strat = z["strat"]
            progress = json.loads(str(z["progress"]))
    else:
        strat = np.load(a.run / "ckpt" / "strat.npy")
        progress = json.loads((a.run / "ckpt" / "meta.json").read_text())["progress"]
    prov = write_pack(a.out, strat, {"ruleset_id": cfg["ruleset_id"], "run": a.run.name, "config": cfg,
                                     "progress": progress, "milestone": a.milestone})
    print(f"{a.out}: rows per level {prov['rows']}, {a.out.stat().st_size / 1e6:.1f} MB on disk")


if __name__ == "__main__":
    main()
