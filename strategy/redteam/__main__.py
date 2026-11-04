"""``python -m strategy.redteam {run,report,build-previous,lineups,gate}``."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from strategy.redteam import registry, suite

GATE_ARGS = {"rt-pressure": "redteam/pressure/bot.py", "rt-equilibrium": "redteam/equilibrium",
             "rt-adaptive": "redteam/adaptive/bot.py"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run every pre-registered cell against a bundle, then write the report")
    r.add_argument("--out", required=True)
    r.add_argument("--bundle", help="the bundle to test (python -m strategy.final.package output)")
    r.add_argument("--pack", help="new blueprint pack: builds the final bundle and the blueprint-only reference from it")
    r.add_argument("--exploit-pack", help="exploit pack for the final bundle (default: the committed default pack)")
    r.add_argument("--ref", action="append", metavar="ARM=PATH", help="override a reference arm's bundle")
    r.add_argument("--jobs", type=int, default=8)
    r.add_argument("--seed", help="override the pre-registered base seed (needed after tuning on these seeds)")
    r.add_argument("--blocks", type=int, help="override every cell's block count (smoke runs only)")
    r.add_argument("--reuse", action="append", metavar="RUN_DIR",
                   help="copy cell/arm results with an identical key (same bundle, lineup, blocks, seed) from a "
                        "finished run, so only the changed arms are replayed")
    r.add_argument("--only", action="append", help="cell id (repeatable)")
    r.add_argument("--arm", action="append", help="arm label (repeatable)")
    p = sub.add_parser("report", help="re-render the report of a run directory")
    p.add_argument("--out", required=True)
    sub.add_parser("build-previous", help="rebuild the pinned previous bots (blueprint-only, nlhe-v2, previous final)")
    sub.add_parser("lineups", help="print the cells")
    g = sub.add_parser("gate", help="the deployed sandbox gate on each red-team bot (idle machine)")
    g.add_argument("--decisions", type=int, default=50_000)
    g.add_argument("--out", default="artifacts/redteam/gate")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        if not (a.bundle or a.pack or a.exploit_pack):
            ap.error("give --bundle, or --pack / --exploit-pack to build one")
        return suite.main_run(a)
    if a.cmd == "report":
        from strategy.redteam.report import write_report

        out = Path(a.out)
        write_report(out, suite.analyze(out))
        print(out / "report.md")
        return 0
    if a.cmd == "build-previous":
        suite.build_previous()
        return 0
    if a.cmd == "lineups":
        for c in suite.build_cells(suite.prereg()):
            print(c.cell_id, c.set, c.blocks, [o.member_id for o in c.lineup.opponents])
        return 0
    rc = 0
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for member, path in GATE_ARGS.items():
        short = registry.SHORT[member]
        cmd = [sys.executable, "-m", "strategy.policy.gate", "--bundle", path, "--decisions", str(a.decisions),
               "--cost-tables", "0", "--out", f"{a.out}/{short}.json"]
        print(" ".join(cmd), file=sys.stderr)
        rc |= subprocess.run(cmd, cwd=suite.REPO).returncode
    print(json.dumps({"exit": rc}))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
