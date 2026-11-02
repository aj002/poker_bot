# Baseline policy, parameter search and deployed-path gate

Stage 1 of the self-play plan (workstreams B and D). The pieces are:

- an interpretable, stochastic rule policy with 100 named parameters;
- a cross-entropy-method search over those numbers, with a confirmed incumbent;
- a packaged bot that runs through the real sandbox worker;
- a promotion check on the validation partition, and a latency and legality gate.

| Module | Role |
| --- | --- |
| `strategy/policy/params.py` | `Param`, `ParamSpace` (bounds, unit-cube map, `space_hash`), `vector_hash` |
| `strategy/policy/baseline.py` | `PARAMS`, `SPACE`, `BaselinePolicy.decide(spot, draw) -> Intent` |
| `strategy/policy/spot.py` | the slim decision features (`TableMemory.spot(obs) -> Spot`) |
| `strategy/policy/runtime.py` | `RuntimeBot`, the single deployed path (an explicit-state SDK bot) |
| `strategy/policy/fallback.py` | constant-time legal fallback; intent legalization |
| `strategy/policy/package.py` | self-contained, sandbox-ready bundle for one vector |
| `strategy/policy/gate.py` | deployed-path gate through `WorkerSeat` (Landlock, seccomp, 1 ms deadline) |
| `strategy/policy/bench.py` | worker-style replay with GC attribution |
| `strategy/search/` | CEM / CMA-ES search, CRN evaluation, run store, validation |

## Policy

Each decision compares one strength number with one threshold made from named
parameters. Near the threshold the policy mixes with a logistic of width `temp_pre`
or `temp_post`; elsewhere it plays a named frequency. The inputs are:

- **Strength.** Preflop, the hand's percentile from the variant's own offline table.
  Postflop, equity against the live opponents' random hands, minus a named range tax
  when facing a bet.
- **Pressure.** The ICM risk premium against the opponent who covers or is covered
  (hand-start stacks, exact Malmuth-Harville), the hands left to the cap, and the
  stack share.
- **Opponents.** Integer counters per opponent (hands, VPIP, PFR, folds to bets,
  aggression). These move thresholds and bluff frequencies by at most `exploit_cap`.

The parameter groups (`SPACE.groups()`) are:

| Group | Count | What it sets |
| --- | --- | --- |
| open | 12 | opening percentile by position group, depth adjustments, isolation, over-limp band, SB complete, BB isolation, steal |
| size | 14 | open, 3-bet, value, c-bet and bluff sizes; all-in commit share; jam SPR; raise fraction |
| jam | 10 | push/fold depth, shove thresholds by position, slope, re-steal and re-shove |
| 3bet / 4bet | 11 | value and bluff 3-bet bands, 4-bet and cold-4-bet thresholds, prices |
| call | 5 | gap penalty, call-price coefficient, BB defence, position and multiway bonuses |
| allin | 6 | shove-width model, range tax, all-in margin, commitment |
| icm | 4 | risk-premium weight, ICM tightening for opens and calls, big-stack pressure |
| post | 6 | value-bet thresholds by street, multiway penalty, traps |
| bluff | 10 | c-bet by texture and player count, barrels, river bluff ratio, semi-bluff bonus, stabs |
| face | 12 | facing-bet range tax by street, raise thresholds, semi and bluff raises, call margins |
| cap | 3 | cap urgency window and the open and call shifts near the cap |
| exploit | 5 | counter prior, bluff, call and steal exploits, `exploit_cap` |
| mix | 2 | logistic temperatures |

The defaults are the **hand-written control**. The search only changes numbers.

### Runtime

`RuntimeBot` is the only path that plays: in search, validation, the gate and
submission.

- **Committed state** is at most about 260 bytes (255 measured over 50k gate
  decisions): the policy seed, the draw counter, the
  decision count, the last counted event `seq`, the fallback count, and the
  per-opponent counters.
- **Restore.** A restarted worker rebuilds its memory from that state plus the hand's
  public events and decides identically. `test_restore_from_committed_state_replays_exactly`
  checks this.
- **Fallbacks.** Any exception or illegal intent plays the legal fallback (check, else
  fold) and increments `state["fb"]`, so the gate can require zero internal fallbacks
  as well as zero arena fallbacks.
- **Imports.** The decision path imports only `arena.agents.sdk` from the arena, plus
  the strategy modules in `strategy.policy.package.RUNTIME_FILES`. Those are:
  - rules: `adapter`, `canonical`, `variants`;
  - features: `cards`, `equity`, `state`;
  - tables: `store`, `preflop`, `postflop`;
  - all of `strategy.policy`.

  It does not use `FeatureEngine`, the belief posteriors or
  `arena.table.player_stats`. `tests/strategy/test_policy.py` checks the static
  imports and the import closure while playing all three variants.

## Search (`python -m strategy.search`)

```bash
make search-baseline SEARCH_CONFIG=strategy/search/configs/nlhe-cem.json   # v1, best challenger
make search-baseline            # nlhe-cem-v2.json: resumes v1's state, mean challenger
make search-status              # SEARCH_RUN=artifacts/search/nlhe-cem-v2 by default
```

Each generation runs these steps:

1. **Fresh tasks.** Draw table seeds, candidate rotations, and train-partition lineups
   from `strategy.population:PopulationSource`. A share of seats (`mirror_rate`) is
   played by the current incumbent.
2. **Common random numbers.** The incumbent, the distribution mean and `population`
   mirrored samples all play the *same* tasks.
3. **Update.** CEM uses the smoothed diagonal Gaussian over the elite fraction, with a
   sigma floor. `"algorithm": "cmaes"` uses standard CMA-ES instead.
4. **Confirm.** A challenger replays against the incumbent on a second, fresh task
   set. It replaces the incumbent only when the paired gain has `t >= accept_t`.
   Selection and confirmation never share tables. The challenger is set by
   `"challenger"`:
   - `"mean"`: the distribution mean. Its selection score is unbiased, because the
     mean was fixed before the generation's tables were drawn.
   - `"best"`: the top-scoring vector. With a per-table standard deviation of about
     22 points, the best of 16 samples is inflated by several points and rarely
     confirms; `nlhe-cem-v1` shows this.

   `"resume_from"` starts a new run from another run's incumbent and distribution.

Everything is in the unit cube of `SPACE`, and `frozen` holds parameters fixed. The
search trains only on `train`; `SearchConfig` rejects any other partition. Every
evaluated vector is persisted in `artifacts/search/<run_id>/` as follows:

- `vectors/<vector_hash>.json`: the parameter values;
- `tasks/<seed_set_hash>.json`: the task sets;
- `evaluations.jsonl`: one row per vector and stage, with the seed-set hash, population
  hash, registry hash and every per-table point result;
- `generations.jsonl`: one row per generation, with selection and confirmation
  statistics;
- `state.json`, for resuming;
- `incumbent.json`.

## Validation (`python -m strategy.search.validate`)

```bash
make validate-baseline          # spec baseline-validation-nlhe, validation partition
make validate-baseline-variants # rehearsal on short deck and PLO4 (discovery stage)
```

This packages the control (defaults) and the candidate, then runs the experiment
harness. Each lineup is played in paired seed blocks with every seat rotation, scored
in official terminal points, with a paired-t 95% interval. The spec declares
`min_effect` before the run. The harness report gives the opponent-family and
candidate-seat slices.

`phases.json` adds a split by tournament phase. A seat's *points-equity* after hand
`h` is what the table's scorer would award if the tournament stopped there:
finished places keep their points, and survivors get the ICM share of the rest. It
starts at `100 / players` and ends at the official points, so the changes over levels
1–2, 3–5 and 6–8 add up exactly to the primary metric.

## Deployed-path gate (`python -m strategy.policy.gate`)

```bash
make gate-baseline              # >= 50,000 decisions, one table at a time
make gate-control               # same gate, calling station: the machine's timeout floor
make bench-policy               # GC attribution by replay
```

The packaged bundle is staged and played by `arena.agents.worker.WorkerSeat`, as in a
sandboxed tournament. Only the candidate's seat is sandboxed; opponents run
in-process. Every candidate table is followed by a **control table**: the same
seed, seat and opponents, with a workload-matched burner (`strategy.eval.burn`)
sandboxed in the candidate's seat. The burner does a fixed amount of CPU work per
decision, calibrated before the first attempt so its median decide time matches the
candidate's over 2 tasks that no attempt plays (`--control-iterations` fixes it).
The control measures the host's stall rate at the candidate's workload while the
gate runs.

Each timeout, candidate and control alike, is classified (`strategy.eval.stalls`,
described in [strategy-evaluation.md](strategy-evaluation.md)):

- *bot-caused*: the decide alone ran past the budget minus the median harness
  overhead with CPU time at least 0.9 of its wall time (`cpu_bound`), or three
  replays of the exact decision in a fresh worker have a median of 500 µs or more;
- *stall-like*: the worker's own time fitted (`transport`), the decide was off-CPU
  (`descheduled`), only the harness parse was slow (`parse`), or the replay was fast.

The control's bot-caused timeouts are host stalls charged to the burner's CPU clock;
the report gives them as the rule's false-attribution floor.

The gate passes, in the attempt used, when all of the following hold:

- at least 50,000 decisions;
- worker wall time p99 below 500 µs;
- zero arena fallbacks other than timeouts, and zero restarts;
- zero bot-caused timeouts;
- stall-like timeouts within the control's stall-like rate (C-test, p ≥ 0.01), on a
  host whose control is not significantly above the 0.05% ceiling;
- zero internal fallbacks.

A host-invalid or stall-excess attempt is rerun with the same seeds, up to 2 times
(`--max-reruns`). Every attempt is in the report under `attempts`, and the first
passing one decides. A bot-caused timeout, or any other failed check, ends the gate
with a failure. The report gives p50, p95, p99, p99.9 and max for worker time and
end-to-end time, for the candidate and the control. The harness process's collector
is off from dispatch until each reply is read, and its heap is frozen per table.
`make gate-control` still runs the plain gate with the calling station as the
candidate.

Run it on an otherwise idle machine. Concurrent CPU load shows up as sandbox
timeouts while the bot's own time stays flat: on 8 CPUs with 7 search workers
running, 60% of decisions timed out at a worker p99 of 254 µs.

Even idle, this VM has a timeout floor. Its host pauses it, and the pauses show up
as thread CPU time rather than steal time. `make gate-control` timed out on 3 of
50,011 decisions with a calling station whose decisions take 16 µs (p50). On the
same machine the baseline candidate timed out on 13 of 50,111 and then 5 of 50,046
decisions, at a worker p99 of 240 µs and p99.9 of 298 µs. The late replies split
between `bot_time` stalls (CPU time equal to a 0.7–2.1 ms wall time on decisions
that replay in under 350 µs) and `transport`. None overlaps a GC pause, and none
reproduces in replay.

With the classified gate (`make gate-baseline`, nlhe-v2, bundle `203360239e2a…`, idle
VM, 9.3 min), the gate **passed on its first attempt**. Over 369 tables:

- **Candidate:** 50,118 decisions with worker wall p99 267 µs, p99.9 359 µs and max
  641 µs. End-to-end p99 was 579 µs. There were no other fallbacks, no restarts and no
  internal fallbacks, and the maximum state was 255 bytes.
- **Timeouts:** 34 in total, 0 bot-caused. 24 were stall-like: 9 by transport (the
  worker's own time fitted the budget) and 15 by a fast replay (replay medians of
  150–450 µs). The other 10 were unclassified, past the 3-replays-per-table cap in
  tables that had a burst of timeouts.
- **Control:** 2 timeouts in 12,876 decisions (0.016%; ceiling p = 0.99). The C-test on
  the candidate's 34 against exposure of 17.9 s versus 3.0 s gave p = 0.094.

Both rates were about 2.5 times those of the earlier runs, so the host was noisier
that day. The first replay in a fresh worker takes about 23 ms because the tables
load lazily; the replay median of 3 absorbs it.

**CPU-aware classifier and matched control.** Under the `cpu_bound` rule and the
workload-matched burner, nlhe-v2 passed one of three gate runs on an idle VM. That run had
20 timeouts, all stall-like, against 6 of 12,860 in the control. The other two runs
failed. One failed on 3 CPU-bound decides, after a first attempt the host invalidated;
the other on 1. Those decides took 821–933 µs. In the same replies the harness parse
ran 3–6 times its usual time, and the replays took 200–360 µs. Over seven gate
attempts, the controls had 1 CPU-bound timeout in 86,000 decisions. Pooled, nlhe-v2's
4 CPU-bound timeouts in 200,000 decisions are not distinguishable from the controls'
(C-test p = 0.48), so a strict gate fails a bot at this workload in a large share of
50,000-decision runs.

**GC.** The worker collects and freezes the heap after loading the bot, and packaged
bots also freeze after loading the variant tables (`gc_mode="freeze"`). The decision
path keeps no garbage cycles, so the bench sees no collection inside a decision.
`--gc fresh` replays each table in a new process to show cold-start cost (the first
decision takes 330–390 µs; the rest match the warm replay). The
~1.5 ms spikes in the full feature bench come from collections that rescan large
recorded heaps, which this path does not create.
