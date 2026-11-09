"""Two bundles through the real sandbox worker on the same tables: identical play or not.

In-process evaluation imports a bundle's ``bot.py`` but resolves ``strategy`` to the
repository, so it compares packs, not code. Here each bundle runs in a sandboxed
:class:`~arena.agents.worker.WorkerSeat` with its own staged ``strategy`` copy, as in a
tournament, against in-process train-partition opponents. The deadline is raised to
:data:`BUDGET_NS` so no timeout fallback can make the two runs diverge. Per table the
public action log (every seat), the hero's committed explicit state and its fallbacks
are compared.

``--until-note MARK`` checks a candidate that changes play only at some decisions: a table
must be identical until B's first hero decision whose note contains ``MARK`` (the public
log before that decision and every earlier hero note), and fully identical when B never
writes ``MARK``. ``--structure`` plays a ``TournamentConfig`` structure (antes, blinds).

    python -m strategy.final.equivalence --a artifacts/exploit/bundles/exploit-be17e35 \
        --b artifacts/final/bundle --tables 40
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

BUDGET_NS = 1_000_000_000
SEED = "final-equivalence-v1"


def _play(pool: Any, artifact: Any, config: Any, task: Any, mark: str | None = None) -> dict[str, Any]:
    from arena.agents.worker import WorkerSeat, play_table
    from strategy.search.evaluate import seat_agents

    agents, cs, table = seat_agents(config, task, lambda seat, info: WorkerSeat(
        seat, artifact, info, pool.acquire, budget_ns=BUDGET_NS))
    seat = agents[cs]
    notes: list[str] = []
    before = [0]
    marked: list[int] = []

    def count(index: int, s: int) -> None:
        if s == cs:
            before[0] = sum(1 for e in table.log if e.type == "ActionApplied")

    def record(s: int, d: Any, tbl: Any) -> None:
        if s != cs:
            return
        notes.append(d.note or "")
        if mark is not None and not marked and mark in (d.note or ""):
            marked.extend((before[0], len(notes) - 1))

    try:
        play_table(table, agents, on_decision=record, before_decision=count)
    finally:
        seat.close()
    actions = [[e.data["seat"], e.data["street"], e.data["action"]] for e in table.log if e.type == "ActionApplied"]
    return {"actions": actions, "hero_decisions": seat.stats.decisions, "fallbacks": dict(seat.stats.fallbacks),
            "committed_sha256": hashlib.sha256(seat.committed or b"").hexdigest(),
            "log_sha256": hashlib.sha256(json.dumps(actions, sort_keys=True).encode()).hexdigest(),
            "notes": notes, "marked": marked or None}


def compare(a: Path, b: Path, *, tables: int = 40, ruleset_id: str = "practice_nlhe", seed: str = SEED,
            log: Any = print, structure: dict[str, Any] | None = None, until_note: str | None = None) -> dict[str, Any]:
    from arena.agents.worker import WorkerPool
    from arena.domain.models import TournamentConfig
    from strategy.search.evaluate import _init, make_tasks
    from strategy.search.population import load_population

    shape = dict(structure) if structure is not None else {"initial_player_count": 8, "hand_cap": 200}
    config = TournamentConfig(ruleset_id=ruleset_id, **shape)
    _init(config.model_dump_json())
    tasks = make_tasks(load_population(), "train", ruleset_id=ruleset_id, players=config.initial_player_count,
                       count=tables, namespace=f"{seed}|{ruleset_id}")
    rows = []
    with WorkerPool(size=2) as pool:
        arts = {"A": pool.stage(Path(a)), "B": pool.stage(Path(b))}
        for i, task in enumerate(tasks):
            ra = _play(pool, arts["A"], config, task)
            rb = _play(pool, arts["B"], config, task, until_note)
            first = next((k for k, (x, y) in enumerate(zip(ra["actions"], rb["actions"], strict=False)) if x != y),
                         None if len(ra["actions"]) == len(rb["actions"]) else min(len(ra["actions"]),
                                                                                    len(rb["actions"])))
            row = {"table": i, "hero_decisions": [ra["hero_decisions"], rb["hero_decisions"]],
                   "actions": len(ra["actions"]), "identical_log": ra["log_sha256"] == rb["log_sha256"],
                   "identical_state": ra["committed_sha256"] == rb["committed_sha256"],
                   "first_difference": first, "fallbacks": [ra["fallbacks"], rb["fallbacks"]]}
            if until_note is not None:
                m = rb["marked"]
                row["marked"] = m
                if m is None:
                    row["ok"] = row["identical_log"] and row["identical_state"]
                else:
                    pos, k = m
                    row["ok"] = (ra["actions"][:pos] == rb["actions"][:pos] and ra["notes"][:k] == rb["notes"][:k])
                row["ok"] = row["ok"] and not any(row["fallbacks"])
            rows.append(row)
            if (i + 1) % 10 == 0:
                log(f"{i + 1}/{tables} tables, identical so far: {sum(r['identical_log'] for r in rows)}")
    out = {"a": str(a), "b": str(b), "ruleset_id": ruleset_id, "seed": seed, "tables": len(rows),
           "hero_decisions": sum(r["hero_decisions"][0] for r in rows),
           "public_actions": sum(r["actions"] for r in rows),
           "identical_tables": sum(r["identical_log"] and r["identical_state"] for r in rows),
           "fallbacks": sum(sum(f.values()) for r in rows for f in r["fallbacks"]), "rows": rows}
    if structure is not None:
        out["structure"] = structure
    if until_note is None:
        out["identical"] = all(r["identical_log"] and r["identical_state"] and not any(r["fallbacks"]) for r in rows)
    else:
        out["until_note"] = until_note
        out["marked_tables"] = sum(r["marked"] is not None for r in rows)
        out["identical_until_marked"] = sum(r["ok"] for r in rows)
        out["identical"] = all(r["ok"] for r in rows)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--tables", type=int, default=40)
    ap.add_argument("--ruleset", default="practice_nlhe")
    ap.add_argument("--structure", type=Path, help="JSON file of TournamentConfig fields (antes, blinds, seats, cap)")
    ap.add_argument("--until-note", help="identical until B's first hero decision whose note contains this")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    structure = json.loads(a.structure.read_text()) if a.structure else None
    rep = compare(a.a, a.b, tables=a.tables, ruleset_id=a.ruleset, structure=structure, until_note=a.until_note,
                  log=lambda s: print(s, file=sys.stderr, flush=True))
    text = json.dumps(rep, indent=1, sort_keys=True)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "rows"}, indent=1, sort_keys=True))
    return 0 if rep["identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
