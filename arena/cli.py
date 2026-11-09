"""Headless command line for the practice engine.

    python -m arena.cli hand --ruleset practice_nlhe --players 6 --seed 1
    python -m arena.cli tournament --ruleset uncertainty_plo4 --players 8 --seed 7 --save
    python -m arena.cli replay --table-id t_demo --verify-hashes
    python -m arena.cli export-phh --table-id t_demo --out artifacts/t_demo.phhs
    python -m arena.cli history --table-id t_demo --viewer seat:3
    python -m arena.cli run-baseline --ruleset practice_nlhe --players 6 --tables 20 --seed demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from arena.domain.canonical import canonical_json, canonical_hash
from arena.domain.models import RULESET_IDS, TournamentConfig
from arena.history.canonical import DEFAULT_HISTORY_DIR, history_path, load_history, write_history
from arena.table.driver import POLICIES, play_out, policy_rngs
from arena.table.service import seeded_table

RUNS_DIR = Path("artifacts/runs")


def _config(args: argparse.Namespace, **extra) -> TournamentConfig:
    return TournamentConfig(
        ruleset_id=args.ruleset, initial_player_count=args.players, starting_stack=args.stack, **extra
    )


def _table_id(args: argparse.Namespace, config: TournamentConfig) -> str:
    if args.table_id:
        return args.table_id
    return "t_" + canonical_hash([config.config_hash(), str(args.seed)], domain="arena-cli-table-v1")[:12]


def _print_events(record) -> None:
    print(f"== {record.hand_id}  blinds {record.blinds[0]}/{record.blinds[1]}  {record.positions.as_dict()}")
    print(f"   start stacks {record.start_stacks}")
    for event in record.events:
        print(f"   {event.type:<18} {json.dumps(event.data, sort_keys=True, default=str)}")


def _play(args: argparse.Namespace, config: TournamentConfig):
    table = seeded_table(config, args.seed, table_id=_table_id(args, config))
    play_out(table, POLICIES[args.policy], policy_rngs(args.seed, config.resolved_seats))
    if args.save:
        print(f"history written to {write_history(table, Path(args.history_dir))}")
    return table


def cmd_hand(args: argparse.Namespace) -> int:
    table = _play(args, _config(args, hand_cap=1, hands_per_level=1))
    _print_events(table.hands[0])
    return 0


def cmd_tournament(args: argparse.Namespace) -> int:
    config = _config(args)
    table = _play(args, config)
    if args.trace:
        for record in table.hands:
            _print_events(record)
    print(f"table {table.table_id}  ruleset {config.ruleset_id}  players {config.initial_player_count}  "
          f"config {config.config_hash()[:12]}")
    print(f"hands played {len(table.hands)}  (cap {config.hand_cap})")
    for record in table.hands:
        for p in record.eliminations:
            print(f"   {record.hand_id}: seat {p.seat} out, positions {p.positions}, {p.points:g} points")
    for p in table.final_placements:
        print(f"   end: seat {p.seat} positions {p.positions}, {p.points:.6f} points ({p.method})")
    print("practice assumptions; not official Sequoia rules")
    return 0


def _find_history(args: argparse.Namespace) -> Path:
    if args.file:
        return Path(args.file)
    if args.history_dir:
        return history_path(args.table_id, Path(args.history_dir))
    candidates = [history_path(args.table_id, DEFAULT_HISTORY_DIR)]
    candidates += sorted(RUNS_DIR.glob(f"*/tables/{args.table_id}.json"))
    for path in candidates:
        if path.exists():
            return path
    raise SystemExit(f"no stored history for table {args.table_id!r} (looked in {DEFAULT_HISTORY_DIR} and {RUNS_DIR})")


def cmd_replay(args: argparse.Namespace) -> int:
    from arena.history.replay import ReplayIncompatible, ReplayMismatch, replay

    path = _find_history(args)
    try:
        report = replay(load_history(path), allow_engine_mismatch=args.allow_engine_mismatch)
    except ReplayIncompatible as exc:
        print(f"HALT {path}: incompatible engine: {exc}", file=sys.stderr)
        return 3
    except (ReplayMismatch, ValueError) as exc:
        print(f"HALT {path}: {exc}", file=sys.stderr)
        return 2
    print(f"table {report.table_id}: replay matches the stored record ({path})")
    print(f"   events {report.events}  hands {report.hands}  complete {report.complete}")
    if args.verify_hashes:
        print(f"   hand-start hashes verified {report.hand_starts_verified}")
        print(f"   decision-boundary state hashes verified {report.decisions_verified}")
    print(f"   final stacks {report.final_stacks}")
    if report.points is not None:
        print(f"   points {report.points}")
    for note in report.notes:
        print(f"   note: {note}")
    return 0


def cmd_export_phh(args: argparse.Namespace) -> int:
    from arena.history.phh import export_history_phhs

    text = export_history_phhs(load_history(_find_history(args)))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="ascii")
        print(f"PHH written to {out}")
    else:
        sys.stdout.write(text)
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    from arena.table.projections import ProjectionDenied, Viewer, history_view

    history = load_history(_find_history(args))
    if args.viewer == "spectator":
        viewer = Viewer.spectator()
    elif args.viewer == "reviewer":
        viewer = Viewer.reviewer()
    elif args.viewer.startswith("seat:"):
        viewer = Viewer.for_seat(int(args.viewer[5:]))
    else:
        raise SystemExit("--viewer is spectator, reviewer or seat:N")
    try:
        view = history_view(history.events, viewer, table_complete=history.summary.complete)
    except ProjectionDenied as exc:
        print(f"denied: {exc}", file=sys.stderr)
        return 4
    print(canonical_json(view.model_dump(mode="json")))
    return 0


def cmd_run_baseline(args: argparse.Namespace) -> int:
    from arena.benchmark.manifest import default_run_id, run_baseline

    config = _config(args)
    policies = {s: (args.policy, POLICIES[args.policy]) for s in config.resolved_seats}
    run_id = args.run_id or default_run_id(config, str(args.seed), args.tables)
    out = Path(args.out) if args.out else RUNS_DIR / run_id
    manifest = run_baseline(config=config, tables=args.tables, base_seed=str(args.seed), policies=policies,
                            out_dir=out, run_id=run_id)
    r = manifest.results
    print(f"run {manifest.run_id}: {len(r.completed_table_ids)} completed, {len(r.failed_tables)} failed, "
          f"{len(r.incomplete_table_ids)} incomplete of {manifest.scheduled_tables}")
    print(f"   manifest {out / 'manifest.json'}  (content hash {manifest.content_hash()[:16]})")
    print(f"   mean points by seat {r.mean_points_by_seat}")
    return 0 if not r.failed_tables else 1


DEFAULT_FIELDS = {
    "practice_nlhe": ["nlhe_heuristic", "calling_station", "legal_random", "check_fold"],
    "uncertainty_shortdeck": ["calling_station", "legal_random", "check_fold"],
    "uncertainty_plo4": ["calling_station", "legal_random", "check_fold"],
}


def _entrant(spec: str):
    from arena.agents.baselines import BASELINES
    from arena.agents.styles import STYLES, resolve_style
    from arena.benchmark.runner import Entrant

    name = spec.removeprefix("baseline:")
    if name in BASELINES:
        return Entrant(name=name, path=BASELINES[name], builtin=True)
    style = resolve_style(spec) if not spec.endswith(".py") else None
    if style is not None:
        return Entrant(name=f"style_{style}", path=STYLES[style], builtin=True)
    path = Path(spec)
    if not path.exists():
        raise SystemExit(f"bot {spec!r} is neither a file, a baseline ({', '.join(sorted(BASELINES))}) "
                         f"nor a style ({', '.join(STYLES)})")
    return Entrant(name=path.stem if path.is_file() else path.name, path=path.resolve())


def _fmt_s(v) -> str:
    return "-" if v is None else f"{v:.6f}"


def _print_benchmark(run, focus: list[int]) -> None:
    m = run.manifest
    s = m.summary
    label = "" if m.competitive else "  [DIAGNOSTIC: not competitive]"
    print(f"run {m.run_id}  ruleset {m.ruleset_id}  {m.initial_player_count} seats  "
          f"{m.scheduled_tables} tables = {m.seed_blocks} paired seed blocks x {m.rotations_per_block} seat rotations")
    print(f"isolation {m.isolation}  budget {m.budget_ns / 1e9:.6f} s  jobs {m.process_layout['jobs']}{label}")
    print(f"completed {s['tables_completed']}/{s['tables_scheduled']}  failed {len(s['tables_failed'])}  "
          f"incomplete {len(s['tables_incomplete'])}  mean table wall {_fmt_s(s['table_wall_s_mean'])} s")
    for f in s["tables_failed"][:5]:
        print(f"   failed {f['table_id']}: {f['error']}")
    if s.get("attempts_total", s["tables_scheduled"]) != s["tables_scheduled"] or s.get("tables_invalid"):
        print(f"attempts {s['attempts_total']} (reruns {s['reruns']}, invalidated by infrastructure "
              f"{len(s['invalid_attempts'])}, still invalid {len(s['tables_invalid'])}; max "
              f"{m.retry.get('max_attempts')} attempts per table, same seed)")
        for x in s["invalid_attempts"][:5]:
            print(f"   invalid {x['attempt_id']}: {x['reason']}")
    payout = sum(m.config["payout_points"])
    print(f"\nmean points per table (payout total {payout}; 95% bootstrap CI over seed blocks)")
    for e in s["entrants"]:
        ci = e["points_ci95"]
        ci_text = f"[{ci[0]:7.3f}, {ci[1]:7.3f}]" if ci else "[not enough blocks]"
        mark = "*" if e["entrant"] in focus else " "
        print(f" {mark} {e['name']:<22} {e['mean_points'] if e['mean_points'] is not None else float('nan'):8.3f}  {ci_text}"
              f"   fallbacks {e['fallback_total']}/{e['decisions']}")
    for e in s["entrants"]:
        if e["entrant"] not in focus:
            continue
        t = e["timing"]
        print(f"\n{e['name']} ({e['state_mode']} state): {e['decisions']} decisions, timing in seconds")
        for key, title in (("end_to_end_s", "end-to-end round trip"), ("worker_wall_s", "worker decide() wall"),
                           ("worker_cpu_s", "worker decide() CPU")):
            q = t[key]
            print(f"   {title:<24} p50 {_fmt_s(q['p50'])}  p95 {_fmt_s(q['p95'])}  p99 {_fmt_s(q['p99'])}  "
                  f"max {_fmt_s(q['max'])}")
        rate = e["fallback_total"] / e["decisions"] if e["decisions"] else 0.0
        reasons = ", ".join(f"{k} {v}" for k, v in e["fallbacks"].items()) or "none"
        print(f"   fallbacks {e['fallback_total']} ({rate:.4%}): {reasons}")
        print(f"   worker restarts {e['restarts']} (not restorable {e['restarts_not_restorable']}, tables with "
              f"lost instance state {e['tables_with_state_loss']}), late replies discarded "
              f"{e['late_replies_discarded']}, max committed state {e['max_state_bytes']} bytes")
    b = s["baselines_combined"]
    if b["decisions"]:
        q = b["timing"]["end_to_end_s"]
        print(f"\nbaselines combined: {b['decisions']} decisions, 1 ms timeout fallbacks {b['timeout_fallbacks']} "
              f"({(b['timeout_rate'] or 0):.4%}); end-to-end p50 {_fmt_s(q['p50'])}  p95 {_fmt_s(q['p95'])}  "
              f"p99 {_fmt_s(q['p99'])}  max {_fmt_s(q['max'])} s")
    if run.out_dir is not None:
        print(f"\nmanifest {run.out_dir / 'manifest.json'}")
    print("practice assumptions; not official Sequoia rules. Results against baselines say little about strength.")


TIMEOUT_WARN_SHARE = 0.01


def _timeout_report(summary: dict, focus: list[int]) -> tuple[list[str], bool]:
    """Timeout and fallback share of the run, and whether timeouts exceed 1% anywhere."""
    lines: list[str] = []
    warn = False
    rows = [(e["name"], e["decisions"], e["fallback_total"], e["fallbacks"].get("timeout", 0))
            for e in summary["entrants"] if e["entrant"] in focus]
    b = summary["baselines_combined"]
    if b["decisions"]:
        rows.append(("baselines combined", b["decisions"], b["timeout_fallbacks"], b["timeout_fallbacks"]))
    for name, decisions, fallbacks, timeouts in rows:
        if not decisions:
            continue
        t, f = timeouts / decisions, fallbacks / decisions
        over = t > TIMEOUT_WARN_SHARE
        warn = warn or over
        lines.append(f"   {name:<22} timeouts {timeouts}/{decisions} ({t:.2%})  all fallbacks {fallbacks} ({f:.2%})"
                     + ("   <-- above 1%" if over else ""))
    return lines, warn


def _print_timeout_report(summary: dict, focus: list[int], *, file) -> None:
    lines, warn = _timeout_report(summary, focus)
    if not lines:
        return
    print("\ntimeout / fallback share (a timed-out decision is played as check or fold)", file=file)
    print("\n".join(lines), file=file)
    if warn:
        bar = "!" * 78
        print(f"\n{bar}\nWARNING: more than 1% of decisions timed out. The 1 ms deadline was missed because the "
              "machine\nwas busy, not because the bot was slow, and results are biased against the late bot. Re-run "
              "with\nfewer parallel jobs (--jobs) or on an otherwise idle machine before trusting these numbers.\n"
              f"{bar}", file=file)


def _default_jobs(isolation: str) -> int:
    """One sandboxed table per four physical cores, one core kept for this process
    (:mod:`arena.benchmark.capacity`): oversubscription shows up as timeouts."""
    from arena.benchmark.capacity import safe_jobs

    return safe_jobs(isolation)


def _run_benchmark_cli(args: argparse.Namespace, entrants, focus: list[int]) -> int:
    import math

    from arena.agents.sdk import UnsupportedRuleset
    from arena.agents.worker import AgentLoadError, WorkerStartError
    from arena.benchmark.runner import run_benchmark

    config = _config(args)
    if len(entrants) != config.initial_player_count:
        raise SystemExit(f"need exactly {config.initial_player_count} entrants, got {len(entrants)}")
    blocks = max(1, math.ceil(args.tables / config.initial_player_count))
    out = Path(args.out) if args.out else None
    step = max(1, (blocks * config.initial_player_count) // 10)

    def progress(done: int, total: int, r) -> None:
        if not args.quiet and (done % step == 0 or done == total):
            print(f"   {done}/{total} tables", file=sys.stderr, flush=True)

    isolation = "in_process" if args.in_process else "sandbox"
    try:
        run = run_benchmark(config=config, entrants=entrants, blocks=blocks, base_seed=str(args.seed),
                            isolation=isolation, budget_ns=args.budget_ns, jobs=args.jobs or _default_jobs(isolation),
                            out_dir=out,
                            save_histories=args.save_histories, progress=progress)
    except UnsupportedRuleset as exc:
        print(f"rejected before the table started: {exc}", file=sys.stderr)
        return 5
    except (AgentLoadError, WorkerStartError) as exc:
        print(f"bot failed to load: {exc}", file=sys.stderr)
        return 6
    if args.json:
        print(canonical_json(run.manifest.model_dump(mode="json")))
    else:
        _print_benchmark(run, focus)
    s = run.manifest.summary
    _print_timeout_report(s, focus, file=sys.stderr if args.json else sys.stdout)
    return 0 if not (s["tables_failed"] or s["tables_invalid"]) else 1


def cmd_match(args: argparse.Namespace) -> int:
    field = args.opponents.split(",") if args.opponents else DEFAULT_FIELDS[args.ruleset]
    opponents = [field[k % len(field)] for k in range(args.players - 1)]
    entrants = [_entrant(args.bot)] + [_entrant(o) for o in opponents]
    return _run_benchmark_cli(args, entrants, focus=[0])


def cmd_benchmark(args: argparse.Namespace) -> int:
    entrants = [_entrant(b) for b in args.bots.split(",")]
    return _run_benchmark_cli(args, entrants, focus=list(range(len(entrants))))


def cmd_schemas(args: argparse.Namespace) -> int:
    from arena.schemas import write_schemas

    for path in write_schemas(Path(args.out)):
        print(path)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="arena")
    sub = parser.add_subparsers(dest="command", required=True)

    def play_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--ruleset", choices=RULESET_IDS, default="practice_nlhe")
        p.add_argument("--players", type=int, default=6)
        p.add_argument("--stack", type=int, default=2000)
        p.add_argument("--seed", default="1")
        p.add_argument("--policy", choices=sorted(POLICIES), default="legal_random")

    def table_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--table-id", required=True)
        p.add_argument("--history-dir", default=None, help=f"default: {DEFAULT_HISTORY_DIR}, then {RUNS_DIR}/*/tables")
        p.add_argument("--file", default=None, help="explicit history file")

    for name, fn in (("hand", cmd_hand), ("tournament", cmd_tournament)):
        p = sub.add_parser(name)
        play_args(p)
        p.add_argument("--table-id", default=None)
        p.add_argument("--save", action="store_true", help="write the private history")
        p.add_argument("--history-dir", default=str(DEFAULT_HISTORY_DIR))
        if name == "tournament":
            p.add_argument("--trace", action="store_true")
        p.set_defaults(fn=fn)

    p = sub.add_parser("replay", help="replay a stored history and verify it")
    table_args(p)
    p.add_argument("--verify-hashes", action="store_true", help="report hand-start and decision hash checks")
    p.add_argument("--allow-engine-mismatch", action="store_true")
    p.set_defaults(fn=cmd_replay)

    p = sub.add_parser("export-phh", help="export complete hands as PHH")
    table_args(p)
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_export_phh)

    p = sub.add_parser("history", help="print a viewer-authorized history")
    table_args(p)
    p.add_argument("--viewer", default="spectator", help="spectator, reviewer or seat:N")
    p.set_defaults(fn=cmd_history)

    p = sub.add_parser("run-baseline", help="play seeded in-process tables and write a run manifest")
    play_args(p)
    p.add_argument("--tables", type=int, default=10)
    p.add_argument("--run-id", default=None)
    p.add_argument("--out", default=None, help=f"run directory (default {RUNS_DIR}/<run_id>)")
    p.set_defaults(fn=cmd_run_baseline)

    def bench_args(p: argparse.ArgumentParser, tables: int) -> None:
        p.add_argument("--ruleset", choices=RULESET_IDS, default="practice_nlhe")
        p.add_argument("--players", type=int, default=6)
        p.add_argument("--stack", type=int, default=2000)
        p.add_argument("--seed", default="1")
        p.add_argument("--tables", type=int, default=tables, help="rounded up to whole seed blocks of --players")
        p.add_argument("--jobs", type=int, default=None,
                       help="tables played in parallel (default: one per 4 physical cores, one core reserved)")
        p.add_argument("--in-process", action="store_true", help="diagnostic: no sandbox, no enforced deadline")
        p.add_argument("--budget-ns", type=int, default=None, help="diagnostic budget (labelled not competitive)")
        p.add_argument("--out", default=None, help="write manifest.json (and private seeds) to this directory")
        p.add_argument("--save-histories", action="store_true", help="with --out, also write private histories")
        p.add_argument("--json", action="store_true", help="print the manifest as JSON")
        p.add_argument("--quiet", action="store_true")

    p = sub.add_parser("match", help="play your bot against the baselines and report points and timing")
    p.add_argument("--bot", required=True, help="bot file (or directory with bot.py) defining make_bot()")
    p.add_argument("--opponents", default=None,
                   help="comma-separated baselines, styles (rock, aggressive, super_aggressive, station, weak_tight, "
                        "switcher) or bot paths, cycled to fill seats")
    bench_args(p, tables=240)
    p.set_defaults(fn=cmd_match)

    p = sub.add_parser("benchmark", help="seat-balanced paired-seed benchmark of N bots")
    p.add_argument("--bots", required=True, help="comma-separated baselines, styles or bot paths, one per seat")
    bench_args(p, tables=480)
    p.set_defaults(fn=cmd_benchmark)

    p = sub.add_parser("schemas", help="write the JSON Schemas of the public contracts")
    p.add_argument("--out", default="docs/schemas")
    p.set_defaults(fn=cmd_schemas)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
