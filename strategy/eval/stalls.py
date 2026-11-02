"""Classify deadline timeouts as bot-caused or stall-like, and gate each class.

Why
    On a virtualised host, a no-op bot still misses the 1 ms deadline on about 0.04%
    of decisions: the host deschedules a vCPU for a millisecond or more. A rate
    limit alone cannot separate those stalls from a bot that is too slow, so every
    timeout is classified and the two classes are gated differently.

Classes (:func:`classify`, first matching rule wins)
    The late reply carries the worker's own times: ``parse`` (observation decode,
    harness code identical for every bot) and the bot's ``decide`` as wall time and
    thread CPU time. ``allowance`` is the budget minus the entrant's median harness
    overhead (``median e2e - median decide wall``, i.e. typical parse plus transport).

    ``bot_caused``
        * cpu_bound: the decide alone exceeded the allowance and its CPU time is at
          least ``CPU_WALL_RATIO`` (0.9) of its wall time. The bot was computing, so
          with typical parse and transport the deadline would still have been missed.
        * replay_timeout / replay_slow: the decision, replayed three times from its
          exact observation and committed state in a fresh worker (explicit-state
          bots only), missed the 100 ms replay budget or had a median ``parse +
          decide`` of at least ``STALL_REPLAY_FRACTION`` (half) of the budget.
          The first replay in the fresh worker is a cold start (lazy loads, caches)
          and is left out of the median when there are at least three.
    ``stall_like``
        * transport: ``parse + decide`` fitted in the allowance; the time was lost
          outside the worker.
        * descheduled: the decide's or the parse's CPU time is under 0.9 of its
          wall time; the thread was off-CPU for the difference.
        * parse: the decide fitted in the allowance but ``parse + decide`` did not;
          the harness's own decode was slow.
        * replay_fast: no reply timings (unresponsive), and the replay median is
          under half the budget.
    ``unclassified``
        * none of the above evidence: instance-state bots without a late reply,
          timeouts past the per-table replay cap, or a replay that could not run.
          Counted with the stall-like ones (never exempted).

    Caveat. On a virtualised host a stalled vCPU's lost time is mostly charged to
    the running thread's CPU clock (steal is not subtracted from thread time), so
    CPU ~ wall does not prove the bot was computing. A host stall that lands in a
    decide is then classified cpu_bound. The workload-matched control measures how
    often that happens: its work is identical on every decision, so all of its
    ``bot_caused`` timeouts are host events misattributed by these rules.

Control
    A workload-matched burner (:mod:`strategy.eval.burn`) that does a fixed amount of
    CPU work per decision, calibrated to the candidate's median decide time, so host
    stalls and slowdowns stretch its decides as they stretch the candidate's. The
    calling station (about 20 us per decision) is the fallback when no calibration
    is possible.

Gates (:func:`assess`, strict entrants; first failing gate wins)
    1. Bot-caused ceiling: a bot-caused rate above ``max_bot_timeout_rate`` (0.02%)
       fails the entrant outright, with no rerun, so a bot that is truly slow can
       never pass however noisy the control was. ``max_bot_timeouts`` (default
       off) is an optional absolute count with the same effect.
    2. Host ceiling: the control's timeouts (all classes) must not run significantly
       above ``max_control_timeout_rate`` (0.05%): one-sided Poisson test,
       ``P(X >= k | n * rate) < CEILING_ALPHA`` means the host was invalid.
    3. Bot-caused against the control: the entrant's bot-caused timeouts are compared
       with the control's bot-caused timeouts (host events the same rules
       misattribute) by a one-sided conditional two-Poisson test (the C-test).
       Exposure is the time spent waiting for replies (sum of end-to-end times), so
       under H0 the entrant's share of the pooled count is
       Binomial(n, E_entrant / (E_entrant + E_control)). p < ``bot_p_min`` (0.01) is
       a bot excess.
    4. Stall-like plus unclassified timeouts are compared with the control's
       stall-like plus unclassified timeouts by the same C-test; p < ``stall_p_min``
       (0.01) is a stall excess.
    Both C-tests pool every interleaved table of the attempt. Non-strict entrants
    get only gates 2 and 4.

Reruns (:func:`select_attempt`)
    A host-invalid, bot-excess or stall-excess attempt is rerun with the same seeds,
    up to ``max_reruns`` (2) times. Every attempt is reported; the first passing
    attempt is used; a ceiling failure in any attempt is final.
"""

from __future__ import annotations

import math
import statistics
from array import array
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

CONTROL_POLICY = "baseline:calling_station"
"""Fallback control when the workload-matched burner cannot be calibrated."""
STALL_REPLAY_FRACTION = 0.5
CPU_WALL_RATIO = 0.9
CEILING_ALPHA = 0.01
MAX_CONTROL_TIMEOUT_RATE = 0.0005
MAX_BOT_TIMEOUT_RATE = 0.0002
BOT_P_MIN = 0.01
STALL_P_MIN = 0.01
MAX_RERUNS = 2
CONTROL_EVERY = 4
TIMEOUT_REPLAYS_PER_SEAT = 3
RECORDS_KEPT = 50

CLASSES = ("bot_caused", "stall_like", "unclassified")


def overhead_ns(e2e: Sequence[int], wall: Sequence[int]) -> int | None:
    """Median parent-observed round trip minus median worker wall time."""
    if not e2e or not wall:
        return None
    return max(0, int(statistics.median(e2e) - statistics.median(wall)))


def _worker_ns(row: Mapping[str, Any]) -> int | None:
    wall = row.get("worker_wall_ns")
    if not isinstance(wall, int):
        return None
    parse = row.get("worker_parse_ns")
    return wall + (parse if isinstance(parse, int) else 0)


def _int(row: Mapping[str, Any], key: str) -> int | None:
    v = row.get(key)
    return v if isinstance(v, int) else None


_TIMES = {"decide": ("worker_wall_ns", "worker_cpu_ns"), "parse": ("worker_parse_ns", "worker_parse_cpu_ns")}


def cpu_bound(record: Mapping[str, Any], part: str = "decide") -> bool | None:
    """Whether the decide's (or the parse's) thread CPU time is at least
    ``CPU_WALL_RATIO`` of its wall time; None without both."""
    wall, cpu = (_int(record, k) for k in _TIMES[part])
    if wall is None or cpu is None:
        return None
    return cpu >= CPU_WALL_RATIO * wall


def classify(record: Mapping[str, Any], budget_ns: int, overhead: int | None) -> tuple[str, str]:
    """``(class, rule)`` for one timeout record from ``TableResult.timeouts``."""
    timed = record.get("outcome") == "late_reply" and overhead is not None
    own, decide = _worker_ns(record), _int(record, "worker_wall_ns")
    allowance = budget_ns - (overhead or 0)
    bound = cpu_bound(record)
    if timed and bound and decide is not None and decide > allowance:
        return "bot_caused", "cpu_bound"
    replay = record.get("replay") or {}
    runs = replay.get("runs") or []
    warm = runs[1:] if len(runs) >= 3 else runs
    times = [t for t in (_worker_ns(r) for r in warm) if t is not None]
    if any(r.get("fallback") == "timeout" for r in runs):
        return "bot_caused", "replay_timeout"
    if times and statistics.median(times) >= STALL_REPLAY_FRACTION * budget_ns:
        return "bot_caused", "replay_slow"
    if timed and own is not None:
        if own <= allowance:
            return "stall_like", "transport"
        if bound is False or cpu_bound(record, "parse") is False:
            return "stall_like", "descheduled"
        if bound:
            return "stall_like", "parse"
    if times:
        return "stall_like", "replay_fast"
    if not replay:
        return "unclassified", "no_replay"
    if "error" in replay:
        return "unclassified", "replay_error"
    return "unclassified", "replay_untimed"


def tally(attempts: Sequence[Any], entrant: int, budget_ns: int) -> dict[str, Any]:
    """Decisions, exposure and classified timeouts of one entrant over runner ``TableResult``s."""
    return tally_samples(
        decisions=sum(r.decisions.get(entrant, 0) for r in attempts),
        timeouts=sum(r.fallbacks.get(entrant, {}).get("timeout", 0) for r in attempts),
        e2e=[x for r in attempts for x in array("q", r.e2e_ns.get(entrant, b""))],
        wall=[x for r in attempts for x in array("q", r.worker_wall_ns.get(entrant, b""))],
        records=[(f"{r.table_id}-a{r.attempt}", rec) for r in attempts for rec in r.timeouts.get(entrant, [])],
        budget_ns=budget_ns)


def tally_samples(*, decisions: int, timeouts: int, e2e: Sequence[int], wall: Sequence[int],
                  records: Sequence[tuple[str, Mapping[str, Any]]], budget_ns: int) -> dict[str, Any]:
    """:func:`tally` from raw samples; ``records`` pairs a table label with each timeout record."""
    over = overhead_ns(e2e, wall)
    classes: Counter = Counter()
    rules: Counter = Counter()
    kept: list[dict[str, Any]] = []
    for table, rec in records:
        cls, rule = classify(rec, budget_ns, over)
        classes[cls] += 1
        rules[rule] += 1
        if len(kept) < RECORDS_KEPT:
            kept.append({"table": table, "class": cls, "rule": rule,
                         **{k: v for k, v in rec.items() if k != "replay"},
                         "replay_worker_ns": [_worker_ns(x) for x in (rec.get("replay") or {}).get("runs", [])]})
    return {
        "decisions": decisions,
        "timeouts": timeouts,
        "timeout_rate": timeouts / decisions if decisions else None,
        "exposure_s": sum(min(x, budget_ns) for x in e2e) / 1e9,
        "overhead_ns": over,
        "classes": {c: classes.get(c, 0) for c in CLASSES},
        "rules": dict(sorted(rules.items())),
        "unrecorded": max(0, timeouts - sum(classes.values())),
        "records": kept,
    }


def merge_tallies(parts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"decisions": 0, "timeouts": 0, "exposure_s": 0.0, "classes": Counter(),
                           "rules": Counter(), "unrecorded": 0, "records": []}
    for p in parts:
        for k in ("decisions", "timeouts", "exposure_s", "unrecorded"):
            out[k] += p[k]
        out["classes"].update(p["classes"])
        out["rules"].update(p["rules"])
        out["records"] += p["records"][: max(0, RECORDS_KEPT - len(out["records"]))]
    out["classes"] = {c: out["classes"].get(c, 0) for c in CLASSES}
    out["rules"] = dict(sorted(out["rules"].items()))
    out["timeout_rate"] = out["timeouts"] / out["decisions"] if out["decisions"] else None
    return out


def _log_binom_pmf(k: int, n: int, p: float) -> float:
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p) + (n - k) * math.log1p(-p))


def binomial_upper_tail(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    if k <= 0:
        return 1.0
    if k > n or p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    return min(1.0, sum(math.exp(_log_binom_pmf(i, n, p)) for i in range(k, n + 1)))


def poisson_upper_tail(k: int, mu: float) -> float:
    """P(X >= k) for X ~ Poisson(mu)."""
    if k <= 0:
        return 1.0
    if mu <= 0.0:
        return 0.0
    below = sum(math.exp(i * math.log(mu) - mu - math.lgamma(i + 1)) for i in range(k))
    return max(0.0, 1.0 - below)


def excess_p_value(k_entrant: int, exposure_entrant: float, k_control: int, exposure_control: float) -> float | None:
    """One-sided C-test p-value that the entrant's timeout rate per unit exposure
    exceeds the control's; None without exposure on both sides."""
    if exposure_entrant <= 0 or exposure_control <= 0:
        return None
    share = exposure_entrant / (exposure_entrant + exposure_control)
    return binomial_upper_tail(k_entrant, k_entrant + k_control, share)


def _stall(t: Mapping[str, Any]) -> int:
    return t["classes"]["stall_like"] + t["classes"]["unclassified"] + t.get("unrecorded", 0)


def _host_and_stalls(entrant: Mapping[str, Any], control: Mapping[str, Any] | None,
                     max_control_timeout_rate: float | None, stall_p_min: float) -> dict[str, Any]:
    stall = _stall(entrant)
    if control is None:
        return {"stall_verdict": "pass" if stall == 0 else "stall_excess",
                "stall_reason": "no control run" if stall == 0
                else f"{stall} stall-like timeouts and no control to test them"}
    out: dict[str, Any] = {}
    kc, nc = control["timeouts"], control["decisions"]
    ks = _stall(control)
    p_excess = excess_p_value(stall, entrant["exposure_s"], ks, control["exposure_s"])
    out["excess_p_value"] = p_excess
    if max_control_timeout_rate is not None:
        p_ceiling = poisson_upper_tail(kc, nc * max_control_timeout_rate)
        out["ceiling_p_value"] = p_ceiling
        if p_ceiling < CEILING_ALPHA:
            return {**out, "stall_verdict": "host_invalid",
                    "stall_reason": f"control timed out on {kc}/{nc} decisions, significantly above the "
                                    f"{max_control_timeout_rate:.3%} host ceiling (p={p_ceiling:.2g})"}
    if stall and (p_excess is None or p_excess < stall_p_min):
        return {**out, "stall_verdict": "stall_excess",
                "stall_reason": f"{stall} stall-like or unclassified timeouts against {ks} in the control "
                                f"(exposure {entrant['exposure_s']:.3g} s vs {control['exposure_s']:.3g} s, "
                                f"p={'n/a' if p_excess is None else f'{p_excess:.2g}'} < {stall_p_min})"}
    return {**out, "stall_verdict": "pass", "stall_reason": "within the host-stall rate of the interleaved control"}


def assess(entrant: Mapping[str, Any], control: Mapping[str, Any] | None, *, strict: bool,
           max_bot_timeout_rate: float | None = MAX_BOT_TIMEOUT_RATE, bot_p_min: float = BOT_P_MIN,
           max_bot_timeouts: int | None = None, max_control_timeout_rate: float | None = MAX_CONTROL_TIMEOUT_RATE,
           stall_p_min: float = STALL_P_MIN) -> dict[str, Any]:
    """Verdict for one entrant tally against the control tally of the same run:
    ``fail`` (bot-caused ceiling, strict entrants only), ``host_invalid``,
    ``bot_excess`` (strict only), ``stall_excess`` or ``pass``. Every test is
    evaluated and reported whatever the verdict (``stall_verdict`` is the host and
    stall part alone)."""
    bot, n = entrant["classes"]["bot_caused"], entrant["decisions"]
    rate = bot / n if n else None
    out: dict[str, Any] = {"strict": strict, "bot_caused": bot, "bot_caused_rate": rate,
                           "stall_like_or_unclassified": _stall(entrant),
                           "max_bot_timeout_rate": max_bot_timeout_rate, "bot_p_min": bot_p_min,
                           "max_bot_timeouts": max_bot_timeouts, "max_control_timeout_rate": max_control_timeout_rate,
                           "stall_p_min": stall_p_min, "ceiling_p_value": None, "excess_p_value": None,
                           "control_bot_caused": None, "control_stall_like_or_unclassified": None,
                           "bot_vs_control_p_value": None}
    if control is not None:
        out["control_bot_caused"] = control["classes"]["bot_caused"]
        out["control_stall_like_or_unclassified"] = _stall(control)
        out["bot_vs_control_p_value"] = excess_p_value(bot, entrant["exposure_s"], control["classes"]["bot_caused"],
                                                       control["exposure_s"])
    out |= _host_and_stalls(entrant, control, max_control_timeout_rate, stall_p_min)
    rules = ", ".join(f"{r} {k}" for r, k in sorted(entrant.get("rules", {}).items())
                      if r in ("cpu_bound", "replay_slow", "replay_timeout"))
    if strict and bot and max_bot_timeout_rate is not None and rate is not None and rate > max_bot_timeout_rate:
        return {**out, "verdict": "fail",
                "reason": f"{bot} bot-caused timeouts in {n} decisions ({rate:.3%}; {rules}), above the "
                          f"{max_bot_timeout_rate:.3%} ceiling"}
    if strict and max_bot_timeouts is not None and bot > max_bot_timeouts:
        return {**out, "verdict": "fail", "reason": f"{bot} bot-caused timeouts ({rules}; limit {max_bot_timeouts})"}
    if out["stall_verdict"] == "host_invalid":
        return {**out, "verdict": "host_invalid", "reason": out["stall_reason"]}
    p_bot = out["bot_vs_control_p_value"]
    if strict and bot and (p_bot is None or p_bot < bot_p_min):
        kc = "no control" if control is None else (
            f"{control['classes']['bot_caused']} in the control (exposure {entrant['exposure_s']:.3g} s vs "
            f"{control['exposure_s']:.3g} s)")
        return {**out, "verdict": "bot_excess",
                "reason": f"{bot} bot-caused timeouts ({rules}) against {kc}, "
                          f"p={'n/a' if p_bot is None else f'{p_bot:.2g}'} < {bot_p_min}"}
    return {**out, "verdict": out["stall_verdict"], "reason": out["stall_reason"]}


def needs_rerun(assessment: Mapping[str, Any]) -> bool:
    return assessment["verdict"] in ("host_invalid", "bot_excess", "stall_excess")


def select_attempt(assessments: Sequence[Mapping[str, Any]], max_reruns: int = MAX_RERUNS) -> dict[str, Any]:
    """Apply the rerun policy to the attempts so far (oldest first).

    ``action`` is ``use`` (with ``used``, the attempt index), ``rerun``, ``fail`` or
    ``invalid`` (reruns exhausted). Exhausted reruns with a bot excess in any attempt
    are a ``fail``: the excess was measured on a host within its ceiling."""
    for i, a in enumerate(assessments):
        if a["verdict"] == "fail":
            return {"action": "fail", "used": i, "reason": a["reason"]}
    for i, a in enumerate(assessments):
        if a["verdict"] == "pass":
            return {"action": "use", "used": i, "reason": a["reason"]}
    if len(assessments) <= max_reruns:
        return {"action": "rerun", "used": None, "reason": assessments[-1]["reason"] if assessments else ""}
    excess = [i for i, a in enumerate(assessments) if a["verdict"] == "bot_excess"]
    if excess:
        return {"action": "fail", "used": excess[-1],
                "reason": f"{len(assessments)} attempts, none passed: {assessments[excess[-1]]['reason']}"}
    return {"action": "invalid", "used": None,
            "reason": f"{len(assessments)} attempts, none passed: {assessments[-1]['reason']}"}


__all__ = ["BOT_P_MIN", "CLASSES", "CONTROL_POLICY", "CPU_WALL_RATIO", "MAX_BOT_TIMEOUT_RATE", "STALL_REPLAY_FRACTION", "assess", "binomial_upper_tail",
           "classify", "cpu_bound", "excess_p_value", "merge_tallies", "needs_rerun", "overhead_ns",
           "poisson_upper_tail", "select_attempt", "tally", "tally_samples"]
