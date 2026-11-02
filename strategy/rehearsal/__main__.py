"""``python -m strategy.rehearsal run <announcement.json> [--budget fast]`` (see :mod:`strategy.rehearsal.runner`)."""

from __future__ import annotations

import argparse
import json
import sys

from strategy.rehearsal.runner import BUDGETS, HELDBACK, run


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m strategy.rehearsal")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="play build day for one announced variant")
    r.add_argument("spec", help="announcement JSON (variant and structure), e.g. strategy/rehearsal/heldback/*.json")
    r.add_argument("--budget", default="fast", choices=sorted(BUDGETS))
    r.add_argument("--out", help="output directory (default artifacts/rehearsal/<name>-<budget>)")
    r.add_argument("--jobs", type=int, help="worker processes (default: CPUs minus one)")
    r.add_argument("--no-gate", action="store_true", help="skip the deployed gate (smoke runs only)")
    sub.add_parser("list", help="the held-back announcements")
    a = p.parse_args(argv)
    if a.cmd == "list":
        for f in sorted(HELDBACK.glob("*.json")):
            print(f"{f}: {json.loads(f.read_text()).get('announcement', '')}")
        return 0
    rep = run(a.spec, a.out, budget=a.budget, jobs=a.jobs, gate=not a.no_gate)
    print(json.dumps({"name": rep["name"], "chosen": rep["chosen"], "failure": rep["failure"],
                      "total_wall_s": rep["total_wall_s"], "total_active_s": rep.get("total_active_s"),
                      "first_artifact_s": rep["first_artifact_s"],
                      "steps": {s["name"]: [s["status"], s["wall_s"]] for s in rep["steps"]}}, indent=1))
    return 0 if rep["chosen"] and not rep["failure"] else 1


if __name__ == "__main__":
    sys.exit(main())
