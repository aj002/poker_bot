"""Distillation study: a table student of the teacher's combined policy, and the deadline hybrid.

The teacher is :class:`~strategy.exploit.runtime.ExploitBot`. Its decision at a blueprint
information set ``k`` is the blueprint row, blended toward expert actions with weights
that depend on per-seat beliefs carried in the explicit state. Its extra cost over the
blueprint is that belief layer, so the only student that buys headroom is one without it.

**Student.** For each key where the teacher ever blended during train-partition
rollouts, the student's row is the teacher's distribution averaged over every visit of
``k`` (blended or not), shrunk toward the blueprint row by :data:`PRIOR_VISITS` pseudo-visits.
That is the teacher's combined policy with the beliefs marginalised out over the states
the teacher reaches. Rows are bytes summing to 255, and every action the average gives
positive probability keeps at least 1/255, so stochastic tail actions survive. The student
pack is the blueprint pack with those rows added at level 0; it runs the blueprint's own
decision path, so its cost and its 43-byte state are the blueprint's.

Steps (``collect`` and ``build`` read only the train partition):

* ``collect --partition train|validation``: teacher rollouts (one teacher, seven partition
  members), recording at every table decision the key, the legal-action mask, the
  teacher's full action distribution and whether it blended;
* ``build``: the student pack from train rollouts;
* ``agree``: on validation rollouts, the student's and the blueprint's distributions at
  the teacher's own decisions (total variation, KL, argmax and common-random-number action
  agreement, tail preservation), overall and on blended decisions;
* ``hybrid``: the deadline-hybrid counterfactual from a deployed-gate report: how many of
  the teacher's timeouts a switch to the cheaper path at decide entry could have saved.

    python -m strategy.final.distill collect --partition train --tables 3000 --out artifacts/final/distill/train.npz
    python -m strategy.final.distill build --data artifacts/final/distill/train.npz --out artifacts/final/distill/student.npz
    python -m strategy.final.distill agree --data artifacts/final/distill/validation.npz --student artifacts/final/distill/student.npz
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from pathlib import Path
from typing import Any

import numpy as np

from strategy.blueprint import abstraction as A

PRIOR_VISITS = 1.0
TAIL = 0.15
"""Teacher actions with probability in ``(0, TAIL)`` are the stochastic tail."""
SEEDS = {"train": "final-distill-train-v1", "validation": "final-distill-validation-v1"}
ROW_SUM = 255


def blend_probs(bot: Any, lams: list[float], f: Any, got: Any) -> list[float] | None:
    """The distribution :meth:`ExploitBot._blend` samples from, or None where it defers to the blueprint."""
    key, mask, _ = f
    if got is None:
        return None
    row, _ = got
    devs = []
    for e, lam in enumerate(lams):
        if lam > 0.0:
            a = bot.xpack.deviation(e, key)
            if a >= 0 and (mask >> a) & 1:
                devs.append((lam, a))
    if not devs:
        return None
    weights = [row[a] if (mask >> a) & 1 else 0 for a in range(A.NUM_ACTIONS)]
    total = sum(weights)
    if total <= 0:
        return None
    probs = [w / total for w in weights]
    for lam, a in devs:
        probs = [(p - lam * p) + (lam if i == a else 0.0) for i, p in enumerate(probs)]
    return probs


def recording_teacher(**kwargs: Any) -> Any:
    """An :class:`ExploitBot` that appends ``(key, mask, probs, blended)`` for each table decision to ``bot.rec``."""
    from strategy.exploit.runtime import ExploitBot

    class RecordingTeacher(ExploitBot):
        def __init__(self) -> None:
            self.rec: list[tuple[int, int, list[float], int]] = []
            super().__init__(**kwargs)
            self.rec.clear()
            policy = self.policy("practice_nlhe")
            orig = policy.act_row
            rec = self.rec

            def act_row(obs, u, f, hit):
                if hit is not None:
                    key, mask, _ = f
                    w = [hit[0][a] if (mask >> a) & 1 else 0 for a in range(A.NUM_ACTIONS)]
                    t = sum(w)
                    if t > 0:
                        rec.append((key, mask, [x / t for x in w], 0))
                return orig(obs, u, f, hit)

            policy.act_row = act_row

        def _blend(self, obs, u, lams, state, f, got):
            probs = blend_probs(self, lams, f, got)
            hit = super()._blend(obs, u, lams, state, f, got)
            if hit is not None and probs is not None:
                self.rec.append((f[0], f[1], probs, 1))
            return hit

    return RecordingTeacher()


def _tables(args: tuple[list[int], str, str, int]) -> dict[str, np.ndarray]:
    from strategy.population.measure import play
    from strategy.population.registry import default_registry

    indices, partition, seed, lineups = args
    pool = default_registry().lineup_pool(partition, "practice_nlhe", lineups=lineups, checkpoints=False)
    hero = recording_teacher()
    keys, masks, probs, blended, tables = [], [], [], [], []
    for t in indices:
        lineup = pool.lineups[t % len(pool.lineups)]
        bots = pool.bots(lineup)
        bots.insert(t % 8, ("teacher", hero))
        hero.rec.clear()
        play(bots, f"{seed}|{t}")
        for k, m, p, b in hero.rec:
            keys.append(k)
            masks.append(m)
            probs.append(p)
            blended.append(b)
            tables.append(t)
    return {"key": np.array(keys, np.int64), "mask": np.array(masks, np.uint8),
            "probs": np.array(probs, np.float32).reshape(-1, A.NUM_ACTIONS), "blended": np.array(blended, np.int8),
            "table": np.array(tables, np.int32)}


def collect(partition: str, tables: int, *, seed: str | None = None, lineups: int = 200, jobs: int = 1,
            log: Any = print) -> dict[str, np.ndarray]:
    if partition not in ("train", "validation"):
        raise ValueError("distillation rollouts use the train or validation partition, never holdout")
    seed = seed or SEEDS[partition]
    chunks = [c for c in (list(range(i, tables, jobs * 4)) for i in range(jobs * 4)) if c]
    t0 = time.time()
    parts = []
    with multiprocessing.get_context("fork").Pool(jobs) as pool:
        for i, part in enumerate(pool.imap_unordered(_tables, [(c, partition, seed, lineups) for c in chunks])):
            parts.append(part)
            log(f"collect {partition}: {i + 1}/{len(chunks)} chunks, {time.time() - t0:.0f}s")
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    order = np.lexsort((np.arange(out["table"].size), out["table"]))
    return {k: v[order] for k, v in out.items()}


# ------------------------------------------------------------------ student


def quantize_keep_tail(p: np.ndarray) -> np.ndarray:
    """Rows to uint8 summing to 255 in which every positive probability keeps at least 1."""
    from strategy.blueprint.policy import quantize

    q = quantize(p).astype(np.int64)
    lift = (p > 0) & (q == 0)
    if lift.any():
        q[lift] = 1
        for i in np.nonzero(lift.any(axis=1))[0]:
            j = int(np.argmax(q[i]))
            q[i, j] -= int(q[i].sum()) - ROW_SUM
    assert (q.sum(axis=1) == ROW_SUM).all() and (q >= 0).all()
    return q.astype(np.uint8)


def student_rows(data: dict[str, np.ndarray], bp: Any, prior: float = PRIOR_VISITS
                 ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """``(keys, uint8 rows, stats)`` for every key where the teacher blended at least once."""
    keys, probs, blended = data["key"], data["probs"].astype(np.float64), data["blended"]
    uk, inv = np.unique(keys, return_inverse=True)
    n = np.bincount(inv, minlength=uk.size).astype(np.float64)
    nb = np.bincount(inv, weights=blended.astype(np.float64), minlength=uk.size)
    sums = np.zeros((uk.size, A.NUM_ACTIONS))
    np.add.at(sums, inv, probs)
    sel = np.nonzero(nb > 0)[0]
    rows = np.zeros((sel.size, A.NUM_ACTIONS))
    for j, i in enumerate(sel):
        hit = bp.lookup(int(uk[i]))
        base = np.array(hit[0], np.float64) / ROW_SUM if hit is not None else sums[i] / n[i]
        rows[j] = (sums[i] + prior * base) / (n[i] + prior)
    stats = {"decisions": int(keys.size), "blended_decisions": int(blended.sum()), "keys_visited": int(uk.size),
             "keys_blended": int(sel.size), "visits_per_blended_key_median": float(np.median(n[sel])) if sel.size else 0,
             "blended_share_at_blended_keys": float(nb[sel].sum() / n[sel].sum()) if sel.size else 0.0,
             "prior_visits": prior}
    return uk[sel], quantize_keep_tail(rows), stats


def build_student(data_path: Path, out: Path, *, pack: Path | None = None, prior: float = PRIOR_VISITS) -> dict:
    from strategy.blueprint.policy import DEFAULT_MIN_MASS, LEVEL_DIVISORS, PolicyPack, write_levels
    from strategy.blueprint.runtime import pack_path

    pack = Path(pack or pack_path("practice_nlhe"))
    bp = PolicyPack(pack)
    with np.load(data_path) as z:
        data = {k: z[k] for k in z.files}
    if str(data.get("partition", "train")) != "train":
        raise ValueError("the student is built from train-partition rollouts only")
    keys, rows, stats = student_rows(data, bp, prior)
    with np.load(pack) as z:
        arrays = {k: z[k] for k in z.files if k != "__provenance__"}
    k0, p0, m0 = arrays["keys0"].astype(np.int64), arrays["probs0"], arrays["mass0"]
    replace = np.isin(k0, keys)
    keep = ~replace
    merged_k = np.concatenate([k0[keep], keys])
    merged_p = np.concatenate([p0[keep], rows])
    merged_m = np.concatenate([m0[keep], np.full(keys.size, 255, np.uint8)])
    order = np.argsort(merged_k, kind="stable")
    arrays["keys0"] = merged_k[order].astype(np.int32)
    arrays["probs0"] = np.ascontiguousarray(merged_p[order])
    arrays["mass0"] = np.ascontiguousarray(merged_m[order])
    stats |= {"rows_replaced": int(replace.sum()), "rows_added": int(keys.size - replace.sum())}
    prov = dict(bp.provenance, kind="final_student", teacher="strategy.exploit.runtime.ExploitBot",
                blueprint_data_sha256=bp.provenance["data_sha256"], distill=stats,
                rollouts=str(data.get("seed", "")))
    for k in ("format", "abstraction", "level_divisors", "min_mass", "rows", "data_sha256"):
        prov.pop(k, None)
    assert len(LEVEL_DIVISORS) == len(DEFAULT_MIN_MASS)
    return write_levels(out, arrays, prov)


# ------------------------------------------------------------------ agreement


def _masked(rows: np.ndarray, masks: np.ndarray) -> np.ndarray:
    bits = ((masks[:, None].astype(np.int64) >> np.arange(A.NUM_ACTIONS)) & 1).astype(np.float64)
    w = rows * bits
    t = w.sum(axis=1, keepdims=True)
    return np.where(t > 0, w / np.where(t > 0, t, 1.0), 0.0)


def pack_distributions(pack: Any, keys: np.ndarray, masks: np.ndarray) -> np.ndarray:
    rows = np.zeros((keys.size, A.NUM_ACTIONS))
    cache: dict[int, Any] = {}
    for i, k in enumerate(keys.tolist()):
        hit = cache.get(k)
        if hit is None:
            got = pack.lookup(k)
            hit = cache[k] = np.array(got[0], np.float64) if got is not None else np.zeros(A.NUM_ACTIONS)
        rows[i] = hit
    return _masked(rows, masks)


def agreement(t: np.ndarray, s: np.ndarray, floor: float = 0.5 / ROW_SUM) -> dict[str, float]:
    """How closely distributions ``s`` follow the teacher's ``t`` (rows over the same legal actions)."""
    if t.shape[0] == 0:
        return {"n": 0}
    tv = 0.5 * np.abs(t - s).sum(axis=1)
    sf = np.maximum(s, floor)
    kl = np.where(t > 0, t * np.log(np.where(t > 0, t, 1.0) / sf), 0.0).sum(axis=1)
    ct, cs = np.cumsum(t, axis=1), np.cumsum(s, axis=1)
    lo_t, lo_s = ct - t, cs - s
    crn = np.clip(np.minimum(ct, cs) - np.maximum(lo_t, lo_s), 0.0, None).sum(axis=1)
    tail = (t > 0) & (t < TAIL)
    return {"n": int(t.shape[0]), "tv_mean": float(tv.mean()), "tv_p90": float(np.quantile(tv, 0.9)),
            "kl_mean": float(kl.mean()), "argmax_agreement": float((t.argmax(axis=1) == s.argmax(axis=1)).mean()),
            "same_action_crn": float(crn.mean()), "identical_share": float((tv < 1e-6).mean()),
            "tail_actions": int(tail.sum()),
            "tail_kept": float((s[tail] > 0).mean()) if tail.any() else None,
            "tail_mass_teacher": float(t[tail].sum() / t.shape[0]),
            "tail_mass_other": float(s[tail].sum() / t.shape[0])}


def agree(data_path: Path, student: Path, *, pack: Path | None = None) -> dict[str, Any]:
    from strategy.blueprint.policy import PolicyPack
    from strategy.blueprint.runtime import pack_path

    with np.load(data_path) as z:
        data = {k: z[k] for k in z.files}
    t = data["probs"].astype(np.float64)
    keys, masks, blended = data["key"], data["mask"], data["blended"].astype(bool)
    out: dict[str, Any] = {"partition": str(data.get("partition", "")), "decisions": int(keys.size),
                           "blended_share": float(blended.mean()) if keys.size else 0.0}
    for name, path in (("student", student), ("blueprint", Path(pack or pack_path("practice_nlhe")))):
        s = pack_distributions(PolicyPack(path), keys, masks)
        out[name] = {"all": agreement(t, s), "blended": agreement(t[blended], s[blended]),
                     "not_blended": agreement(t[~blended], s[~blended])}
    return out


# ------------------------------------------------------------------ deadline hybrid


def hybrid(gate: Path, cheap: Path | None = None, *, cost_ratio: float | None = None) -> dict[str, Any]:
    """Counterfactual for a hybrid that switches to a cheaper path (ratio ``r`` of the teacher's decide time).

    A late reply's lateness is bracketed. The upper bound is ``late_by``, when the reply was
    read late, or ``drained_after``, when it was drained later. (The end-to-end time of a
    timeout is when the parent gave up, not when the reply arrived.) The lower bound is the
    worker's own parse plus decide time past the budget, ignoring transport. Saving
    ``(1 - r) * decide wall`` rescues the decision if it covers the upper bound, and cannot
    rescue it if it falls short of the lower bound; in between it is undetermined.
    Even "rescued" is an upper bound on a real hybrid's benefit. It assumes a perfect
    switch at decide entry. A real switch can react only to time the bot measures itself,
    not to the worker's parse, the transport or a stall before decide starts, and it makes
    play depend on the clock. ``r`` is the cheap gate's decide p50 over the teacher's,
    unless given."""
    rep = json.loads(Path(gate).read_text())
    if cost_ratio is None:
        if cheap is None:
            raise ValueError("give a cheap-path gate report or a cost ratio")
        other = json.loads(Path(cheap).read_text())
        cost_ratio = other["worker_wall"]["p50_us"] / rep["worker_wall"]["p50_us"]
    used = rep["attempts"][rep["selection"]["used"]] if rep["selection"].get("used") is not None else rep["attempts"][-1]
    recs = used["candidate"]["timeouts"]["records"]
    rows = []
    for r in recs:
        upper = r.get("late_by_ns") if r.get("late_by_ns") is not None else r.get("drained_after_ns")
        wall, parse, budget = r.get("worker_wall_ns"), r.get("worker_parse_ns"), r.get("budget_ns")
        lower = (max(0, wall + (parse or 0) - budget) if wall is not None and budget is not None else None)
        saved = (1.0 - cost_ratio) * wall if wall is not None else None
        if saved is None:
            verdict = "unknown"
        elif upper is not None and saved >= upper:
            verdict = "rescued"
        elif lower is not None and saved < lower:
            verdict = "not_rescued"
        else:
            verdict = "undetermined"
        rows.append({"class": r["class"], "late_upper_us": None if upper is None else upper / 1e3,
                     "late_lower_us": None if lower is None else lower / 1e3,
                     "parse_us": None if parse is None else parse / 1e3,
                     "decide_us": None if wall is None else wall / 1e3,
                     "saved_us": None if saved is None else saved / 1e3, "verdict": verdict})
    by_class: dict[str, dict[str, int]] = {}
    for r in rows:
        c = by_class.setdefault(r["class"], {"timeouts": 0, "rescued": 0, "not_rescued": 0, "undetermined": 0,
                                             "unknown": 0})
        c["timeouts"] += 1
        c[r["verdict"]] += 1
    return {"gate": str(gate), "cost_ratio": cost_ratio, "decisions": rep["decisions"],
            "timeouts": used["candidate"]["timeouts"]["timeouts"], "records": len(rows),
            "rescued": sum(r["verdict"] == "rescued" for r in rows),
            "not_rescued": sum(r["verdict"] == "not_rescued" for r in rows), "by_class": by_class, "rows": rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--partition", choices=("train", "validation"), required=True)
    c.add_argument("--tables", type=int, default=3000)
    c.add_argument("--lineups", type=int, default=200)
    c.add_argument("--jobs", type=int, default=max(1, (multiprocessing.cpu_count() or 2) - 1))
    c.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("build")
    b.add_argument("--data", type=Path, required=True)
    b.add_argument("--prior", type=float, default=PRIOR_VISITS)
    b.add_argument("--out", type=Path, required=True)
    g = sub.add_parser("agree")
    g.add_argument("--data", type=Path, required=True)
    g.add_argument("--student", type=Path, required=True)
    g.add_argument("--out", type=Path)
    h = sub.add_parser("hybrid")
    h.add_argument("--gate", type=Path, required=True)
    h.add_argument("--cheap", type=Path)
    h.add_argument("--cost-ratio", type=float)
    h.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "collect":
        data = collect(a.partition, a.tables, lineups=a.lineups, jobs=a.jobs)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.out, partition=np.array(a.partition), seed=np.array(SEEDS[a.partition]),
                            tables=np.array(a.tables), **data)
        print(f"{a.out}: {data['key'].size:,} decisions, {int(data['blended'].sum()):,} blended")
        return 0
    if a.cmd == "build":
        prov = build_student(a.data, a.out, prior=a.prior)
        print(json.dumps({"out": str(a.out), "data_sha256": prov["data_sha256"], "rows": prov["rows"],
                          **prov["distill"]}, indent=1))
        return 0
    out = agree(a.data, a.student) if a.cmd == "agree" else hybrid(a.gate, a.cheap, cost_ratio=a.cost_ratio)
    text = json.dumps(out, indent=1, sort_keys=True)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
