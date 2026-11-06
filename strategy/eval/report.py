"""``report.json`` and a concise ``report.md`` for one experiment run."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from strategy.rules.canonical import canonical_json


def _f(v: Any, digits: int = 3) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _ci(lo: Any, hi: Any, digits: int = 3) -> str:
    return "-" if lo is None or hi is None else f"[{lo:.{digits}f}, {hi:.{digits}f}]"


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return out


def _slice_rows(rows: Mapping[str, Mapping[str, Any]]) -> list[list[str]]:
    return [[k, r.get("n", 0), _f(r.get("mean_difference")), _ci(r.get("lo"), r.get("hi")),
             _f(r.get("p_holm"))] for k, r in rows.items()]


def _integrity_lines(check: Mapping[str, Any] | None) -> list[str]:
    if not check:
        return []
    rows = []
    for arm, t in check["arms"].items():
        worst = max(t["by_entrant"].items(), key=lambda kv: kv[1]["timeout_rate"], default=(None, None))
        rows.append([arm, t["decisions"], t["timeouts"], f"{t['timeout_rate']:.3%}",
                     f"{worst[0]} {worst[1]['timeout_rate']:.3%}" if worst[0] else "-"])
    lines = ["", f"Deadline integrity (every seat, both arms): {'ok' if check['passed'] else 'FAILED'}; limits "
             f"{_pct(check['max_timeout_rate'])} pooled, {_pct(check['max_entrant_timeout_rate'])} per entrant "
             f"(at least {check['entrant_min_count']} timeouts).", ""]
    lines += _table(["arm", "decisions (all seats)", "timeouts", "rate", "worst entrant"], rows)
    return lines


def _timeout_class_lines(classes: Mapping[str, Any] | None) -> list[str]:
    if not classes:
        return []
    rows = []
    for arm, c in classes.items():
        k, ctl = c["candidate"], c["control"]
        attempts = [f"{cid} a{row['attempt']} {row['verdict']}" for cid, cell in c["cells"].items()
                    for row in cell["selection"]["attempts"]]
        rows.append([arm + (" (strict)" if c["strict"] else ""), k["decisions"], k["timeouts"],
                     k["classes"]["bot_caused"], k["classes"]["stall_like"], k["classes"]["unclassified"],
                     f"{ctl['timeouts']}/{ctl['decisions']} ({_pct3(ctl['timeout_rate'])}; "
                     f"bot-caused {ctl['classes']['bot_caused']})", c["reruns"],
                     "; ".join(attempts[:6]) + (" ..." if len(attempts) > 6 else "")])
    lines = ["", "Classified timeouts (candidate seat; control = interleaved workload-matched burner tables, "
                 "whose bot-caused timeouts are host events misattributed by the same rules; "
                 "see strategy.eval.stalls):", ""]
    lines += _table(["arm", "decisions", "timeouts", "bot-caused", "stall-like", "unclassified", "control",
                     "reruns", "attempts"], rows)
    for arm, c in classes.items():
        lines += [f"- arm {arm}: {x}" for x in c["failed"] + c["invalid"]]
    return lines


def _pct3(v: Any) -> str:
    return "-" if v is None else f"{v:.3%}"


def _pct(v: Any) -> str:
    return "-" if v is None else f"{v:.2%}"


def _header(report: Mapping[str, Any]) -> list[str]:
    m, dec = report["manifest"], report["analysis"]["decision"]
    spec = m["spec"]
    lines = [f"# {spec['experiment_id']}: {spec['title']}", ""]
    if report["analysis"].get("withheld"):
        banner = "RUN INVALID: POINTS WITHHELD." if dec["verdict"] == "invalid" else "POINTS WITHHELD."
        lines += [f"> **{banner}** " + dec["reason"] + ".", ">", f"> {dec['note']}.", ""]
        lines += [f"**Verdict: {dec['verdict']}** ({dec['stage']})."]
    else:
        lines += [f"**Verdict: {dec['verdict']}** ({dec['stage']}). {dec['reason']}."]
        if dec.get("note"):
            lines += ["", f"_{dec['note']}._"]
    run = report.get("run", {})
    jobs = f" Jobs {run['jobs']} (safe default {run['safe_jobs']})." if "safe_jobs" in run else ""
    lines += ["", f"Arm A `{m['policies']['A']['spec']}` (control), arm B `{m['policies']['B']['spec']}` (candidate). "
              f"Objective: {m['objective']['id']} (chips are diagnostics only). Isolation {m['isolation']}.{jobs} "
              f"Manifest `{m['manifest_id'][:16]}`.", ""]
    return lines


def _provenance(m: Mapping[str, Any]) -> list[str]:
    return ["", "## Provenance", "",
            "- Rules: " + ", ".join(f"{r['ruleset_id']} `{r['rules_hash'][:12]}`" for r in m["rules"]),
            f"- Population: {m['population']['source']} `{m['population']['registry_hash'][:12]}`, partition "
            f"{m['population']['partition']['name']} `{m['population']['partition']['hash'][:12]}`"
            + (" (STUB: not a locked registry)" if m["population"]["stub"] else ""),
            f"- Seeds: {m['seeds']['blocks_total']} blocks, list hash `{m['seeds']['seed_list_sha256'][:12]}`",
            f"- Code: {m['code']['commit'] or 'unknown'}{' (dirty)' if m['code']['dirty'] else ''}",
            "", "Practice assumptions; not official Sequoia rules."]


def _execution_lines(ex: Mapping[str, Any]) -> list[str]:
    lines = ["", "## Execution", ""]
    rows = []
    for arm in ("A", "B"):
        e = ex.get(arm) or {}
        if not e:
            continue
        lat = e["latency_end_to_end_s"]
        rows.append([arm, e["decisions"], _f(lat["p50"], 6), _f(lat["p95"], 6), _f(lat["p99"], 6),
                     _f(lat["p99_9"], 6), _f(lat["max"], 6), e["fallback_total"], e["timeouts"], e["illegal_actions"],
                     e["errors"], e["artifact_bytes"], e["max_state_bytes"], "yes" if e["competitive"] else "no"])
    lines += _table(["arm", "candidate decisions", "p50 s", "p95 s", "p99 s", "p99.9 s", "max s", "fallbacks",
                     "timeouts", "illegal", "errors", "artifact bytes", "max state bytes", "competitive timing"], rows)
    lines += _integrity_lines(ex.get("timeout_integrity"))
    lines += _timeout_class_lines(ex.get("timeout_classes"))
    if ex.get("gates"):
        lines += ["", "Gates (arm B): " + ", ".join(
            f"{g['gate']} {'ok' if g['passed'] else 'FAILED'} ({_f(g['observed'], 6)} vs {g['limit']})"
            for g in ex["gates"])]
    lines += ["", "Memory: not measured (the sandbox enforces a 1 GiB address-space limit)."]
    return lines


def render_markdown(report: Mapping[str, Any]) -> str:
    m, a = report["manifest"], report["analysis"]
    lines = _header(report)
    if a.get("withheld"):
        lines += _execution_lines(a["execution"])
        lines += _provenance(m)
        return "\n".join(lines) + "\n"
    p = a["primary"]
    lines += ["## Primary result", ""]
    if p.get("blocks"):
        t, b = p["paired_t"], p["cluster_bootstrap"]
        lines += _table(["metric", "value"], [
            ["blocks (independent units)", p["blocks"]],
            ["tables per arm", p["tables_per_arm"]],
            ["mean points A", _f(p["mean_points_A"])],
            ["mean points B", _f(p["mean_points_B"])],
            ["mean difference B - A", _f(p["mean_difference"])],
            ["SD of block differences", _f(p["sd_difference"])],
            [f"{int(p['level'] * 100)}% paired t interval", _ci(t["lo"], t["hi"])],
            [f"{int(p['level'] * 100)}% cluster bootstrap interval", _ci(b["lo"], b["hi"])],
            ["decision interval", "luck-adjusted (below)" if a.get("decision", {}).get("estimator") == "luck_cv"
             else p["interval_method"]],
            ["declared minimum effect", _f(p["min_effect"])],
            ["minimum detectable effect (80% power)", _f(p["minimum_detectable_effect"])],
            ["blocks B better / tied / A better", f"{p['blocks_B_better']} / {p['blocks_tied']} / {p['blocks_A_better']}"],
        ])
    else:
        lines += ["No complete blocks."]
    if a.get("variance_reduced"):
        lines += render_variance_section(a["variance_reduced"])
    if p["blocks_excluded"]:
        lines += ["", f"**{len(p['blocks_excluded'])} blocks excluded** (incomplete in an arm):"]
        lines += [f"- {x['cell_id']} block {x['block']}: {'; '.join(x['reasons'][:3])}" for x in p["blocks_excluded"][:10]]
    lines += _execution_lines(a["execution"])
    s = a["slices"]
    for name, title in (("variant", "By variant"), ("family", "By opponent family (blocks whose lineup includes it)"),
                        ("seat", "By candidate seat"), ("lineup", "By lineup")):
        if s[name]:
            lines += ["", f"## {title}", ""]
            lines += _table(["slice", "blocks", "mean B - A", "interval", "Holm p"], _slice_rows(s[name]))
    lines += ["", "## Finishes and the cap (descriptive, per arm)", ""]
    da, db = s["descriptive_by_arm"]["A"], s["descriptive_by_arm"]["B"]
    if da.get("tables") and db.get("tables"):
        lines += _table(["", "A", "B"], [
            ["tables", da["tables"], db["tables"]],
            ["tables ended at the cap", _f(da["tables_ended_at_cap"]), _f(db["tables_ended_at_cap"])],
            ["candidate survived to cap (ICM)", _f(da["finish_method"].get("icm", 0.0)),
             _f(db["finish_method"].get("icm", 0.0))],
            ["candidate won outright", _f(da["finish_method"].get("winner", 0.0)),
             _f(db["finish_method"].get("winner", 0.0))],
            ["candidate eliminated", _f(da["finish_method"].get("elimination", 0.0)),
             _f(db["finish_method"].get("elimination", 0.0))],
            ["cap ICM points per table", _f(da["cap_icm_points_mean"]), _f(db["cap_icm_points_mean"])],
            ["lower-tail CVaR10 points", _f(da["lower_tail_cvar10_points"]), _f(db["lower_tail_cvar10_points"])],
            ["mean final chips (diagnostic)", _f(da["chips_diagnostic"]["mean_final_chips"], 1),
             _f(db["chips_diagnostic"]["mean_final_chips"], 1)],
        ])
        keys = sorted(set(da["finish_position_distribution"]) | set(db["finish_position_distribution"]),
                      key=lambda k: (k == "survived_icm", int(k) if k.isdigit() else 0))
        lines += ["", "Finish position distribution:", ""]
        lines += _table(["position", "A", "B"], [[k, _f(da["finish_position_distribution"].get(k, 0.0)),
                                                  _f(db["finish_position_distribution"].get(k, 0.0))] for k in keys])
    opp = [r for r in s["opportunity_rates"] if r["opportunities_A"] >= 30 and r["opportunities_B"] >= 30]
    if opp:
        lines += ["", "## Changed action rates (nominal Wilson/Newcombe; decisions are clustered)", ""]
        rows = []
        for r in opp[:40]:
            for cls in ("raise", "fold"):
                x = r["rates"][cls]
                if x["A"]["k"] or x["B"]["k"]:
                    rows.append([r["slice"], cls, f"{x['A']['k']}/{x['A']['n']}", f"{x['B']['k']}/{x['B']['n']}",
                                 _f(x["difference"]), _ci(*(x["newcombe"] or (None, None)))])
        lines += _table(["opportunity", "action", "A", "B", "B - A", "interval"], rows)
    rb = a["robustness"]
    lines += ["", "## Robustness", ""]
    wf = rb["worst_family"]
    lines += [f"- Worst family: {wf['family']} mean {_f(wf.get('mean_difference'))} "
              f"{_ci(wf.get('lo'), wf.get('hi'))}" if wf else "- Worst family: -",
              f"- 5th-percentile lineup mean difference: {_f(rb['lineup_p5_mean_difference'])} "
              f"over {rb['lineups']} lineups",
              f"- 5th-percentile block difference: {_f(rb['block_difference_p5'])}",
              f"- Lower-tail CVaR10 points: A {_f(rb['lower_tail_cvar10_points']['A'])}, "
              f"B {_f(rb['lower_tail_cvar10_points']['B'])}"]
    lines += ["", "Not measured: " + "; ".join(f"{k.replace('_', ' ')} ({v})" for k, v in a["not_measured"].items())
              + "."]
    if "null_calibration" in report:
        lines += ["", render_null_section(report["null_calibration"])]
    lines += _provenance(m)
    return "\n".join(lines) + "\n"


def render_variance_section(v: Mapping[str, Any]) -> list[str]:
    role = "the decision interval" if v.get("primary") else "reported alongside; the decision uses the raw interval"
    lines = ["", "## Luck-adjusted estimate", ""]
    if not v.get("available"):
        return lines + [f"Unavailable: {v.get('reason')}."]
    coef = (f"cross-fitted over {v['folds']} folds" if v["coefficients"] == "crossfit"
            else "fixed before the run")
    lines += [f"Block differences minus zero-mean luck terms ({', '.join(v['terms'])}), coefficients {coef}"
              + ("; within-lineup variance (conditional on these lineups)" if v["stratify"] else "")
              + f"; {role} (strategy.eval.variance).", ""]
    lines += _table(["metric", "value"], [
        ["mean difference B - A (adjusted)", _f(v["mean_difference"])],
        ["mean difference B - A (raw)", _f(v["raw"]["mean_difference"])],
        [f"{int(v['level'] * 100)}% interval (adjusted)", _ci(v["lo"], v["hi"])],
        ["SD of block differences: raw / adjusted", f"{_f(v['raw']['sd'])} / {_f(v['sd_adjusted'])}"],
        ["variance reduction factor", _f(v["variance_reduction_factor"], 2)],
        ["minimum detectable effect (80% power): raw / adjusted",
         f"{_f(v['minimum_detectable_effect_raw'])} / {_f(v['minimum_detectable_effect'])}"],
        ["mean correction (raw - adjusted)", f"{_f(v['correction']['mean'])} (SE {_f(v['correction']['se'])})"],
    ])
    lines += ["", "Luck differences (each has mean exactly zero by construction):", ""]
    lines += _table(["term", "mean B - A", "SE", "t", "corr. with block difference"],
                    [[t, _f(d["mean_difference"]), _f(d["se"]), _f(d["t_stat"], 2), _f(d["corr_with_difference"], 2)]
                     for t, d in v["luck_differences"].items()])
    return lines


def render_null_section(n: Mapping[str, Any]) -> str:
    lines = ["## Null calibration (identical policies)", ""]
    cells = n.get("cells")
    size = (f"{n['blocks_per_replicate']} blocks per cell x {cells} cells" if cells
            else f"{n['blocks_per_replicate']} blocks")
    lines += [f"{n['replicates']} disjoint replicate experiments of {size}. "
              f"Declared one-sided false-gain rate {n['declared_false_gain_rate']:.3f} "
              f"(alpha {n['alpha']:g}, two-sided interval).", ""]
    rows = []
    for name, r in n["methods"].items():
        rows.append([name, f"{r['false_gains']}/{n['replicates']}", _f(r["false_gain_rate"]),
                     _ci(*r["false_gain_rate_ci"]), f"{r['false_promotions']}", _f(r["mean_interval_width"]),
                     _f(r["binomial_p_vs_declared"]), "ok" if r["calibrated"] else "TOO MANY"])
    lines += _table(["method", "false gains", "rate", "exact 95% CI", f"false promotions (> {n['min_effect']:g})",
                     "mean interval width", "P(X >= observed)", "calibration"], rows)
    if n.get("exact_coupling") is not None:
        c = n["exact_coupling"]
        timed = c.get("tables_with_timeouts", 0)
        lines += ["", f"Exact coupling (same artifact, same policy seed): {c['blocks']} blocks, max |points "
                  f"difference| {_f(c['max_abs_difference'], 9)} over {c.get('tables', '?')} - {timed} tables "
                  f"without a timeout -> {'identical' if c['identical'] else 'NOT IDENTICAL'}."]
        if timed:
            lines += [f"{timed} tables had a timeout in either arm (a forced fold in that arm only); "
                      f"{c['diverged_after_timeout']} of them diverged."]
    return "\n".join(lines)


def finite(obj: Any) -> Any:
    """JSON-safe copy: non-finite floats become null, tuples become lists."""
    if isinstance(obj, Mapping):
        return {str(k): finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def write_report(out_dir: Path, report: Mapping[str, Any]) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = finite(report)
    jpath, mpath = out_dir / "report.json", out_dir / "report.md"
    jpath.write_text(canonical_json(report) + "\n", encoding="ascii")
    mpath.write_text(render_markdown(report), encoding="utf-8")
    return jpath, mpath


def load_report(out_dir: Path) -> dict[str, Any]:
    return json.loads((out_dir / "report.json").read_text(encoding="ascii"))
