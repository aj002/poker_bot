"""The one-shot final holdout evaluation (``docs/final-holdout-preregistration.md``).

Four pre-registered comparisons of the final bundle on the holdout partition, one-sided
paired t on block differences, Holm across all four at alpha 0.05:

* H1-H3 superiority (``H0: mean(B - A) <= 0``) against the blueprint, the baseline
  ``nlhe-v2`` and the endgame-capped bot, on sampled holdout lineups (each holds one
  holdout-only family member);
* H4 non-inferiority (``H0: mean(B - A) <= -0.5``) against the blueprint in lineups made
  only of the two holdout-only families.

``run`` refuses to start unless the bundle's hash is the pre-registered one and the
pre-registration commit is an ancestor of HEAD, and it refuses to run a comparison a second
time: :data:`LEDGER` records each start and finish. An interrupted comparison resumes the
same manifest (the harness never replays a finished block); nothing is re-seeded or re-run.

    python -m strategy.final.holdout run --jobs 8
    python -m strategy.final.holdout confirm
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
PREREG = Path(__file__).resolve().parent / "holdout-preregistration.json"
OUT = REPO / "artifacts" / "final" / "holdout"
LEDGER = OUT / "ledger.json"


def preregistration() -> dict[str, Any]:
    return json.loads(PREREG.read_text(encoding="ascii"))


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def check_ready(prereg: dict[str, Any]) -> None:
    from strategy.final.package import verify

    bundle = REPO / prereg["bundle"]
    got = verify(bundle)["bundle_sha256"]
    if got != prereg["bundle_sha256"]:
        raise SystemExit(f"{bundle} hash {got[:16]} is not the pre-registered {prereg['bundle_sha256'][:16]}")
    tag = prereg["tag"]
    try:
        commit = _git("rev-list", "-n", "1", tag)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"pre-registration tag {tag} is missing") from exc
    if subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=REPO).returncode:
        raise SystemExit(f"pre-registration commit {commit[:10]} is not an ancestor of HEAD")


def _ledger() -> dict[str, Any]:
    return json.loads(LEDGER.read_text()) if LEDGER.exists() else {}


def _write_ledger(ledger: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(ledger, indent=1, sort_keys=True) + "\n")


def run(jobs: int, *, only: list[str] | None = None) -> dict[str, Any]:
    from strategy.exploit.evaluate import run as run_spec

    from strategy.population import PopulationSource

    prereg = preregistration()
    check_ready(prereg)
    PopulationSource().partition("holdout")  # writes the member bundles H4's explicit lineups name
    ledger = _ledger()
    for h in prereg["hypotheses"]:
        if only and h["id"] not in only:
            continue
        entry = ledger.get(h["id"], {})
        if entry.get("finished"):
            print(f"{h['id']}: already run ({entry['run']}); the holdout is never run twice", file=sys.stderr)
            continue
        ledger[h["id"]] = {"spec": h["spec"], "started": entry.get("started") or dt.datetime.now(dt.UTC).isoformat(),
                           "head": _git("rev-parse", "HEAD"), "resumed": bool(entry)}
        _write_ledger(ledger)
        run_dir, _ = run_spec(h["spec"], jobs=jobs)
        run_dir = Path(run_dir).resolve()
        ledger[h["id"]] |= {"run": str(run_dir.relative_to(REPO)), "finished": dt.datetime.now(dt.UTC).isoformat()}
        _write_ledger(ledger)
    return ledger


def confirm() -> dict[str, Any]:
    from strategy.exploit.evaluate import confirm as holm_confirm

    prereg = preregistration()
    ledger = _ledger()
    missing = [h["id"] for h in prereg["hypotheses"] if not ledger.get(h["id"], {}).get("finished")]
    if missing:
        raise SystemExit(f"not run yet: {missing}")
    sup = {h["id"]: REPO / ledger[h["id"]]["run"] for h in prereg["hypotheses"] if h["kind"] == "superiority"}
    ni = {h["id"]: REPO / ledger[h["id"]]["run"] for h in prereg["hypotheses"] if h["kind"] == "non_inferiority"}
    out = holm_confirm(sup, ni, margin=prereg["margin"], alpha=prereg["alpha"])
    out["bundle_sha256"] = prereg["bundle_sha256"]
    for h in prereg["hypotheses"]:
        row = out["hypotheses"][h["id"]]
        rep = json.loads((Path(row["run"]) / "report.json").read_text())
        row["comparison"] = h["comparison"]
        p = rep["analysis"]["primary"]
        row["points"] = {"A": p["mean_points_A"], "B": p["mean_points_B"]}
        row["mde"] = p["minimum_detectable_effect"]
        ex = rep["analysis"]["execution"]
        row["execution"] = {arm: {k: ex[arm].get(k) for k in ("decisions", "errors", "illegal_actions",
                                                              "fallback_total", "timeouts")}
                            for arm in ("A", "B")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "confirm.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--jobs", type=int, default=1)
    r.add_argument("--only", action="append", help="hypothesis id (default: all, in pre-registered order)")
    sub.add_parser("check")
    sub.add_parser("confirm")
    a = ap.parse_args(argv)
    if a.cmd == "check":
        check_ready(preregistration())
        print("ready: bundle hash and pre-registration commit match")
        return 0
    out = run(a.jobs, only=a.only) if a.cmd == "run" else confirm()
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
