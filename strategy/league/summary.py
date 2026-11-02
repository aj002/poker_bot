"""Markdown summary of a league run: iterations, payoff matrices, exploitability, gates, curve, confirmations.

    python -m strategy.league.summary artifacts/league/main-v1 > summary.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _ci(r: dict) -> str:
    if r.get("withheld") or r.get("lo") is None:
        return "n/a"
    return f"{r['diff']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]"


def matrix_md(meta: dict) -> list[str]:
    names = meta["names"]
    G, S = meta["matrix"]["gain"], meta["matrix"]["se"]
    out = ["| a \\ b | " + " | ".join(names) + " |", "| --- |" + " --- |" * len(names)]
    for i, a in enumerate(names):
        cells = ["0" if i == j else f"{G[i][j]:+.4f} ± {S[i][j]:.4f}" for j in range(len(names))]
        out.append(f"| {a} | " + " | ".join(cells) + " |")
    return out


def iteration_md(d: Path, k: int) -> list[str]:
    out = [f"### Iteration {k}", ""]
    ex, meta, learn, gates = (_load(d / f) for f in ("exploiter.json", "meta.json", "learner.json", "gates.json"))
    if ex:
        if "reused" in ex:
            out.append(f"Exploiter: incumbent unchanged, {ex['reused']} reused.")
        else:
            g = ex["gain_vs_target"]
            out.append(f"Exploiter {ex['name']} against {ex['target']}: {ex['deviations']:,} deviations, held-out gain "
                       f"{g['gain_points']:+.4f} ± {g['se']:.4f} points per player-hand -> {ex['status']}.")
        out.append("")
    if meta:
        out += ["Payoff matrix `G[a][b]` (points per player-hand of `a` over `b`, other seats `b`):", ""]
        out += matrix_md(meta)
        sig = ", ".join(f"{n} {p:.3f}" for n, p in meta["sigma"].items())
        out += ["", f"σ: {sig}. NashConv {meta['nash_conv']:.4f} (unfloored {meta['nash_conv_raw']:.4f}); exact "
                f"mixed-seat {meta['exact']['nash_conv_exact']:.4f}. Cycles: {meta['cycles'] or 'none'}.", ""]
    if learn:
        out += ["| learner z | mix | deviations | mixture gain | population gain | worst pool world |",
                "| --- | --- | --- | --- | --- | --- |"]
        for g in learn["grid"]:
            m = g["gains"]["mixture"]
            out.append(f"| {g['z']} | {g['mix']} | {g['deviations']:,} | {m['gain_points']:+.4f} ± {m['se']:.4f} | "
                       f"{g['gains']['population']['gain_points']:+.4f} | {g['worst']:+.4f} |")
        out += ["", f"Chosen: z {learn['chosen']['z']}, mix {learn['chosen']['mix']}; screen "
                f"{'passed' if learn['screen']['passed'] else 'failed: ' + learn['screen']['reason']}.", ""]
    if gates and not gates.get("skipped"):
        out += ["| gate | blocks | B − A [95%] | worst family | p5 lineup | verdict |", "| --- | --- | --- | --- | --- | --- |"]
        for name, r in gates["runs"].items():
            wf = r.get("worst_family") or {}
            wfs = f"{wf.get('family')} {wf['mean']:+.2f}" if wf.get("mean") is not None else "n/a"
            p5 = f"{r['lineup_p5']:+.2f}" if r.get("lineup_p5") is not None else "n/a"
            out.append(f"| {name} | {r.get('blocks')} | {_ci(r)} | {wfs} | {p5} | {r.get('verdict')} |")
        failed = [c["check"] for c in gates["decision"]["checks"] if not c["passed"]]
        pool = gates["decision"].get("pooled_regress")
        if pool:
            out += ["", f"Pooled regression {pool['mean']:+.2f} [{pool['lo']:+.2f}, {pool['hi']:+.2f}] over {pool['k']}."]
        out += ["", f"Decision: {'promote' if gates['promote'] else 'retire'}"
                + (f" (failed: {', '.join(failed)})" if failed else "") + ".", ""]
    elif gates:
        out += [f"Gates skipped: {gates.get('reason')}.", ""]
    return out


def summary(run: Path) -> str:
    state = json.loads((run / "league.json").read_text())
    out = [f"# League {state['run']}", "", f"Incumbent: {state['incumbent']}.", "",
           "| k | incumbent before | candidate | promoted | incumbent after |", "| --- | --- | --- | --- | --- |"]
    for it in state["iterations"]:
        out.append(f"| {it['k']} | {it['incumbent_before']} | {it['candidate']} | {it['promoted']} | "
                   f"{it['incumbent_after']} |")
    out.append("")
    for it in state["iterations"]:
        out += iteration_md(run / "iterations" / f"it{it['k']}", it["k"])
    curve = _load(run / "curve.json")
    if curve:
        out += ["## Learning curve (vs blueprint, validation)", "", "| checkpoint | blocks | B − A [95%] |",
                "| --- | --- | --- |"]
        out += [f"| {n} | {r.get('blocks')} | {_ci(r)} |" for n, r in curve.items()] + [""]
    sel = _load(run / "selection.json")
    if sel:
        out += ["## Selection", "", f"Selected: {sel['selected']}.", ""] + matrix_md(
            {"names": sel["matrix"]["names"], "matrix": sel["matrix"]}) + [""]
    conf = _load(run / "confirm.json")
    if conf and not conf.get("skipped"):
        out += ["## Confirmations (Holm, one-sided α = 0.05)", "",
                "| id | blocks | B − A [95%] | margin | p | Holm p | outcome |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for k, v in conf["holm"].items():
            out.append(f"| {k} | {v['blocks']} | {_ci(v)} | {v['margin']:+g} | {v['p_one_sided']:.2g} | "
                       f"{v['p_holm']:.2g} | {'confirmed' if v['confirmed'] else 'null'} |")
        out.append("")
    elif conf:
        out += ["## Confirmations", "", conf["reason"], ""]
    return "\n".join(out)


if __name__ == "__main__":
    print(summary(Path(sys.argv[1])))
