"""Markdown report for a finished (or partial) suite run: ``<out>/report.md`` and ``<out>/results.json``."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

HERO = "ours"
ARM_LABEL = {"ours": "final+HandHQ", "blueprint-only": "blueprint-only", "nlhe-v2": "nlhe-v2",
             "previous-final": "previous final"}


def f(x: float | None, d: int = 2, sign: bool = False) -> str:
    if x is None:
        return "n/a"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def ci(c: dict[str, Any], d: int = 2, sign: bool = False) -> str:
    return f"{f(c.get('mean'), d, sign)} [{f(c.get('lo'), d, sign)}, {f(c.get('hi'), d, sign)}]"


def write_report(out: Path, res: dict[str, Any]) -> None:
    (out / "results.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    (out / "report.md").write_text(render(res))


def render(res: dict[str, Any]) -> str:
    par = res["par"]
    meta = res["meta"]
    lines = [f"# Red-team suite run ({meta['commit'][:10]}, seed `{meta['base_seed']}`)", "",
             f"Pre-registration sha256 `{meta['prereg_sha256'][:12]}`; par is {par} points per tournament; "
             "points are the arena's official terminal points; intervals are 95%.", ""]
    lines += ["## Arms", "", "| arm | bundle | sha256 |", "| --- | --- | --- |"]
    for a, v in meta["arms"].items():
        lines.append(f"| {ARM_LABEL.get(a, a)} | `{v['path']}` | `{v['bundle_sha256'][:12]}` |")
    lines += ["", "## Integrity", "", "| arm | tables | invalid | illegal | errors | fallbacks | timeouts | opp. timeouts |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for a, i in res["integrity"].items():
        lines.append(f"| {ARM_LABEL.get(a, a)} | {i['tables']} | {i['invalid']} | {i['illegal']} | {i['errors']} | "
                     f"{i['fallbacks']} | {i['timeouts']} | {i['opp_timeouts']} |")
    refs = [a for a in ("blueprint-only", "nlhe-v2", "previous-final") if any(a in r["arms"] for r in res["cells"].values())]
    lines += ["", "## Points per tournament by lineup", "",
              "Final bot's mean points (95% interval), its class against par, and the same on the references "
              "(mean only) with the final bot's paired difference.", "",
              "| cell | opponents | final+HandHQ | class | Holm p | " + " | ".join(
                  f"{ARM_LABEL[a]} (diff)" for a in refs) + " |",
              "| --- | --- | --- | --- | --- | " + " | ".join("---" for _ in refs) + " |"]
    for cid, r in res["cells"].items():
        h = r["arms"].get(HERO)
        if not h:
            continue
        opp = _opp_label(r["lineup"])
        cols = []
        for a in refs:
            if a in r["arms"] and a in r["diff"]:
                cols.append(f"{f(r['arms'][a]['mean'])} ({ci(r['diff'][a], 2, True)})")
            else:
                cols.append("n/a")
        holm = f"{h['p_holm']:.3g}" if h.get("p_holm") is not None else "n/a"
        flag = "*" if h.get("holm_significant") else ""
        lines.append(f"| {cid} | {opp} | {ci(h)} | {h['class']}{flag} | {holm} | " + " | ".join(cols) + " |")
    lines += ["", "`*` = the deviation from par is significant after Holm correction across all cells.", ""]
    lines += ["## Pooled by lineup set", "",
              "Equal-weight mean of the cells' means; the interval uses the within-cell variances.", "",
              "| set | cells | final+HandHQ | class | " + " | ".join(f"minus {ARM_LABEL[a]}" for a in refs) + " |",
              "| --- | --- | --- | --- | " + " | ".join("---" for _ in refs) + " |"]
    for name, s in res["sets"].items():
        h = s["arms"].get(HERO)
        if not h:
            continue
        cols = [ci(s["diff"][a], 2, True) if a in s["diff"] else "n/a" for a in refs]
        lines.append(f"| {name} | {len(s['cells'])} | {ci(h)} | {h['class']} | " + " | ".join(cols) + " |")
    lines += ["", *attribution_section(res), "", *census_section(res)]
    return "\n".join(lines) + "\n"


def _opp_label(lineup: list[str]) -> str:
    c = Counter(m.replace("rt-", "").replace("prev.", "prev:") if not m.startswith("pm-") and not m.startswith("hq-")
                else "pop" for m in lineup)
    return " ".join(f"{k}x{v}" if v > 1 else k for k, v in c.items())


def attribution_section(res: dict[str, Any]) -> list[str]:
    att = res["attribution"]["cells"]
    lines = ["## Behaviour attribution (exploratory)", "",
             "Rules are in `preregistration.json`. A flag marks a behaviour as implicated; it is an association on "
             "these tables, not a causal claim. The same rules run on every arm.", "",
             "| cell | arm | tables | fold to overbet / smaller | hero bb per overbet hand | bubble+money equity per table "
             "| red-team 3bet+/shove rate 8-6 -> 4 -> 3 | image tight - loose (bb/hand) | flags |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for cid, arms in att.items():
        for a, r in arms.items():
            ob = r["overbet_chips_bb_per_hand"]
            bm = r["bubble_money_eq_per_table"]
            rt = r["redteam_agg_rate_by_band"]
            img = r["image_tight_minus_loose"]
            lines.append(
                f"| {cid} | {ARM_LABEL.get(a, a)} | {r['tables']} | "
                f"{_pct(r['fold_to_overbet'])} / {_pct(r['fold_to_smaller'])} (n={sum(r['overbet_faced'])}) | "
                f"{f(ob[0], 2, True)} [{f(ob[1], 2, True)}, {f(ob[2], 2, True)}] (n={r['overbet_hands']}) | "
                f"{f(bm[0], 2, True)} [{f(bm[1], 2, True)}, {f(bm[2], 2, True)}] | "
                f"{_pct(rt['8-6'])} -> {_pct(rt['4'])} -> {_pct(rt['3'])} | "
                f"{'n/a' if img is None else f(img['est'], 2, True) + ' [' + f(img['lo'], 2, True) + ', ' + f(img['hi'], 2, True) + ']'} | "
                f"{', '.join(k for k, v in r['flags'].items() if v) or '-'} |")
    return lines


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def census_section(res: dict[str, Any]) -> list[str]:
    c = res.get("census", {}).get(HERO)
    if not c:
        return []
    lines = ["## Opponent behaviour seen by the final bot", "",
             f"Pooled over all {c['tables']} tables of the final bot. Rates are per hand dealt in; `open size` is the "
             "first raise in big blinds; an overbet is a post-flop bet or raise of more than the pot after calling.", "",
             "| opponent | hands | open rate | open size (bb) | 3-bet+ per 100 | shoves per 100 | post-flop bets per 100 | overbet share |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for tag, r in c["behaviour"].items():
        lines.append(f"| {tag} | {r['hands']} | {_pct(r['open_rate'])} | {f(r['open_bb'], 1)} | "
                     f"{f(r['three_bet_plus_per_100'], 1)} | {f(r['shoves_per_100'], 1)} | "
                     f"{f(r['postflop_aggressive_per_100'], 1)} | {_pct(r['overbet_share_of_postflop'])} |")
    total = sum(c["hero_bust"].values()) or 1
    lines += ["", "How the final bot's tournaments ended (eliminated by whom, at how many players alive):", "",
              "| outcome | share of tables |", "| --- | --- |"]
    for k, v in sorted(c["hero_bust"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {100 * v / total:.1f}% |")
    return lines
