# Strategy lab: rules, objective, paired evaluation and experiments

The `strategy/` package holds the self-play learning work. This document covers the
variant-agnostic layers every later stage builds on:

- `strategy/rules/`: variant metadata, the rules hash, the canonical action menu, and conformance.
- `strategy/objective/`: the terminal-points objective.
- `strategy/eval/`: paired A/B evaluation, statistics, slices, execution metrics, and the report.
- `strategy/experiments/`: experiment specs, immutable run manifests, and the runner.
- `strategy/cli.py`: the `python -m strategy.cli` entry point.

All results are practice assumptions, not official Sequoia rules.

## Decision-time boundary

At decision time a bot may import only the public SDK (`arena.agents.sdk`) and the
observation contracts (`arena.domain.observations`). Inside this package only
`strategy.rules.variants`, `strategy.rules.adapter` and `strategy.rules.canonical`
are decision-time safe. `tests/strategy/test_rules.py` enforces that with an import
check. Everything else (objective, eval, experiments, conformance) is offline and may
use arena internals.

## Rules (`strategy.rules`)

```python
from strategy.rules import RulesAdapter, get_variant, canonical_actions

adapter = RulesAdapter.from_table_info(table_info)  # or .from_config(TournamentConfig | resolved dict)
adapter.rules_hash         # variant + everything that changes terminal points
adapter.variant_hash       # the game alone
adapter.capabilities       # e.g. {"betting.no_limit", "deck.standard_52", "assets.nlhe", ...}
adapter.variant.require("assets.nlhe", asset="preflop charts")  # raises UnsupportedVariant elsewhere
adapter.actions(observation, fractions=(0.5, 1.0))  # canonical, always-legal menu
```

- **`VariantSpec`** is a frozen description of one variant: deck ranks and suits,
  hole cards, board streets, burns, showdown hand selection, category order, lowest
  straight, and betting structure. `KNOWN_VARIANTS` covers `practice_nlhe`,
  `uncertainty_shortdeck` and `uncertainty_plo4`. A surprise variant is a new
  `VariantSpec`, built with `VariantSpec.from_dict` if needed. It needs no new code
  paths.
- **Capabilities** gate asset reuse. Hold'em charts, priors and tables must call
  `require(CAP_NLHE_ASSETS)`. Only exact NLHE has that capability.
- **The rules hash** covers the variant plus seats, stacks, blind schedule, hand cap,
  payouts, button policy, runouts, rake and local rules. Decision budgets and UI
  timers are excluded.
- **The canonical action menu** contains fold, check, call, min raise, pot raise,
  all-in, and pot-fraction raises. A pot fraction `f` raises to
  `current_bet_to + f * (pot_total + to_call)`. Fraction raises outside the legal
  interval are dropped. When two raises land on the same amount, one survives, in
  priority order all-in, then pot, then min, then fraction. Every entry satisfies
  `legal.allows`.

### Conformance (`strategy.rules.conformance`, offline)

`run_conformance(ruleset_id, simulator=...)` checks any object that implements the
`Simulator` and `SimHand` protocols against the variant metadata and the arena
fixtures:

- **metadata**: deck, deal and betting structure.
- **ranking**: probe hands per deck.
- **fixtures**: replay of `tests/golden/fixtures` core and eval fixtures. Decks are
  rebuilt from our own metadata.
- **scoring**: the ICM fixtures.
- **differential**: lockstep random play against the arena reference, with every
  canonical-menu entry checked for legality.

`ArenaSimulator` is the reference implementation. A fast simulator written for
training must pass this before use. Run `make lab-conformance`.

## Objective (`strategy.objective`)

The objective is official terminal points, and only that:

- **Eliminated**: the finishing position's points. Simultaneous busts share
  positions.
- **Last alive**: the first prize.
- **Survivor at the hand cap**: the exact Malmuth-Harville ICM share of the
  unassigned top prizes.

All of this goes through `arena.tournament.scoring`, the code the table itself uses.
Chips appear only in `chip_diagnostics(...)`, labelled `diagnostic_only`. The
experiment schema rejects any other primary metric.

```python
from strategy.objective import outcome_from_events, rescore, cap_allocation, icm_points
outcome = outcome_from_events(table.log)        # or outcome_from_table(table)
outcome.utility(seat); outcome.finishes[seat].method  # "elimination" | "winner" | "icm"
rescore(payouts, seats, start_stacks, [(hand_number, before, after), ...], hand_cap)  # for rollouts
```

## Paired evaluation (`strategy.eval`)

- **Cell**: one ruleset, table config and opponent lineup. The candidate is
  entrant 0 and the opponents are entrants 1 to N−1.
- **Block**: one table seed. Within a block the candidate rotates through every
  seat.
- **Arms**: arms A and B run `arena.benchmark.runner.run_benchmark` with the same
  seeds, so cards, buttons and opponents' seeds match (common random numbers).

Block utility `U = mean candidate points over rotations`, and `D = U(B) − U(A)`. The
block is the only inferential unit. Hands, decisions and tables within a block are
never treated as independent.

- **Statistics** (`strategy.eval.stats`, standard library only): paired t;
  stratified cluster bootstrap; Welch; Wilson and Newcombe for rates;
  Clopper-Pearson; Holm; `sample_size(sigma_d, delta)` (N ≈ (z₁₋α/₂ + z₁₋β)² σ²/δ²,
  t-refined); and `minimum_detectable_effect`.
- **Decision**: `pass` when the interval's lower bound exceeds `min_effect` and every
  gate passes. `fail` when an execution gate breaks or the interval lies entirely
  below zero. `invalid` when any block is incomplete, for example a failed table.
  Otherwise `inconclusive`. Discovery results are labelled as needing fresh locked
  data.
- **Slices**: candidate seat, variant, lineup, and opponent family, each with Holm
  p-values. Descriptive stats per arm cover:
  - finish distribution and finish method;
  - fraction of tables ended at the cap, and cap ICM points;
  - bust level, and points by survivors at the end;
  - lower-tail CVaR10, and chip diagnostics.

  Changed action rates are reported by street and facing, preflop effective depth,
  blind level, hands to the cap, and players in the hand. These come from public
  quantities only and use nominal Newcombe intervals.
- **Execution** (per arm): latency p50, p95, p99, p99.9 and max; fallbacks,
  timeouts, illegal actions, errors and reruns; artifact bytes and state bytes;
  whether timing was competitive (sandboxed). Gates come from the spec.
- **Timeout integrity gate**: checked before anything else, over every seat
  (candidate and opponents) in both arms. A timed-out decision is a forced check or
  fold, so a run whose deadline did not hold measured a different game. The gate
  fails when an arm's pooled timeout rate exceeds `gates.max_timeout_rate` (default
  0.5%), or when any one entrant has at least 5 timeouts and a rate above
  `gates.max_entrant_timeout_rate` (default 2%). The run is then `invalid`: no
  primary result, slices, robustness or block pairs; `report.md` opens with
  **RUN INVALID: POINTS WITHHELD**; a null calibration is withheld; and
  `strategy.cli experiment` exits 3. In-process runs enforce no deadline and record
  no timeouts. Set either limit to `null` to switch it off.
- **Classified timeout gate** (sandboxed runs, `strategy.eval.stalls`): each arm
  plays a control table (the control in every seat) before its first table and after
  every `gates.control_every` (4) tables. The control is a workload-matched burner
  (`strategy.eval.burn`): a bot that runs a fixed number of loop iterations per
  decision and then checks or calls. Before each arm, one sandboxed probe table on
  its own seed (`<cell seed>|calibration`, never counted) measures the candidate's
  median decide wall time. Two burner probe tables (0 and 3,000 iterations) give the
  burner's base cost (about 15 µs) and cost per iteration (about 55 ns), and the
  iteration count is set so the burner's median matches the candidate's. The work is
  fixed, not the time, so host stalls and host slowdowns stretch the burner's decides
  as they stretch the candidate's. `gates.control_iterations` fixes the count
  instead. The calling station is the fallback when the probe sees no candidate
  decision. Every timeout, candidate and
  control alike, is classified, first matching rule first. The *allowance* is the
  budget minus the entrant's median harness overhead (median end-to-end minus median
  decide wall, which is typical parse plus transport):
  - *bot-caused*: the late reply's decide alone exceeded the allowance with thread
    CPU time at least 0.9 of its wall time (`cpu_bound`: the bot was computing); or a
    replay of the exact decision (explicit-state bots, first
    `gates.timeout_replays_per_seat` (3) timeouts per seat per table, 3 repeats in a
    fresh worker) has a median of at least half the budget or misses even its 100 ms
    budget;
  - *stall-like*: parse plus decide fitted the allowance (`transport`); the decide
    or the parse was off-CPU for over a tenth of its wall time (`descheduled`); the
    decide fitted but the harness's parse did not (`parse`); or, with no reply
    timings, the replay median is under half the budget (`replay_fast`);
  - *unclassified*: none of that evidence. These count with the stall-like ones.

  On this VM a host stall is mostly charged to the running thread's CPU clock, so
  CPU ≈ wall does not prove the bot was computing: a stall that lands in a decide is
  classified `cpu_bound`. The burner's work is identical on every decision, so the
  control's own bot-caused timeouts are host events misattributed by the same rule.
  They are reported as the floor (`control_bot_caused`, with a C-test p-value of the
  candidate's bot-caused count against it, `bot_vs_control_p_value`).

  Arm B is strict. More than `gates.max_bot_timeouts` (0) bot-caused timeouts fail
  the run with verdict `fail`, whatever the floor: a timeout is a fold in
  competition. The points are still reported, and the arm is not rerun. In each
  arm, the control's timeouts (every class) must not run significantly above the
  `gates.max_control_timeout_rate` host ceiling (0.05%; one-sided Poisson test at
  0.01). Otherwise the attempt is *host-invalid*.

  Stall-like and unclassified timeouts are compared with the control's stall-like
  and unclassified timeouts by a conditional two-Poisson C-test. Exposure is the
  summed end-to-end time, and p < `gates.stall_p_min` (0.01) is a *stall excess*.

  A host-invalid or stall-excess arm is rerun with the same seeds under
  `cells/<cell>/<arm>/rerun-<n>/`, up to `gates.max_reruns` (2) times, against the
  same calibrated control. The first passing attempt is used, and every attempt is
  listed in `execution.timeout_classes` (with each cell's `control_calibration`). If
  no attempt passes, the run is `invalid` with points withheld, as for the integrity
  gate. `gates.control_every: 0` switches the gate off. Control tables add a quarter
  more tables to a sandboxed run, plus one probe table per arm.
- **Report**: `report.md` (concise) and `report.json` (full, including every block
  pair).

### Partition interface (population registry stub)

The registry belongs to `strategy.population`. Evaluation needs only this protocol
from `strategy.eval.population`:

```python
class PartitionSource(Protocol):
    registry_id: str
    def registry_hash(self) -> str: ...
    def partition(self, name: str) -> Partition: ...   # "train" | "validation" | "holdout"
    def sample_lineups(self, partition, *, ruleset_id, opponents, count, seed) -> list[Lineup]: ...
```

- A `Partition` carries `registry_id`, `registry_hash`, `seed_namespace`,
  `members: tuple[OpponentRef, ...]` and `metadata`, and exposes
  `partition_hash()`.
- An `OpponentRef` has `member_id`, `family`, a `policy` spec (`baseline:<name>`,
  `style:<name>` or a bot path), the `rulesets` it plays, and `params_hash`.
- `sample_lineups(...)` is a deterministic default sampler.
- A spec names a source as `"stub"` or as `"package.module:attribute"`, which can be
  a source, a class, or a factory.

`StubPartitionSource` splits the arena's baselines and styles by family. Every
manifest marks it `stub: true`, and it is not valid for promotion claims.

The frozen opponent registry implements the protocol as
`"source": "strategy.population:PopulationSource"`. It has 17 sampled families with
train, validation and holdout partitions, plus promoted checkpoint slots; see
[opponent-population.md](opponent-population.md). It is the default
`population.source`. The stub stays available as `"source": "stub"`, which the
published null calibrations (`null-aa-nlhe`, `null-aa-variants`) set explicitly so
their numbers reproduce.

## Experiments (`strategy.experiments`)

A spec, in JSON under `strategy/experiments/specs/` or at any path, declares
everything before the run:

- the hypothesis id and stage (`discovery`, `confirmation` or `final_holdout`);
- arms, including an optional `salt` per arm;
- variants, table structure, the population partition and lineups;
- blocks per lineup and the base seed;
- `min_effect`, `alpha`, `power` and the interval method;
- isolation, the decision budget, and execution gates;
- for `kind: null_calibration`, the replicate layout.

Two rules are enforced. The holdout partition is used by, and only by,
`final_holdout` runs. A null calibration must compare identical policies.

`run_experiment(spec)` materializes an **immutable manifest**. Its id,
`manifest_id = "m_" + hash(content)`, pins:

- the objective, including a hash of the scorer source;
- the engine versions;
- each variant's rules hash, variant hash, arena config hash and capabilities;
- each arm's artifact hash (before and after salting);
- the registry and partition hashes and member ids;
- every cell's lineup with opponent artifact hashes;
- a hash of every block seed.

The manifest is written once with mode 0444 and verified on every load.
`ManifestTampered` means the content changed after the fact. `ManifestConflict`
means a different manifest was written to the same path. Seeds stay in
`private/seeds.json`, and the manifest carries only their hashes. A finished run
directory is never re-run (`AlreadyRun`).

```
artifacts/lab/<experiment_id>/<manifest_id>/
  manifest.json  private/seeds.json  cells/<cell>/<arm>/manifest.json  report.json  report.md
```

### Parallel jobs and the decision deadline (`strategy.eval.parallel`)

The 1 ms deadline is wall clock: a reply counts only if the parent has it within the
budget, and a late one is a forced check or fold. A sandboxed benchmark job keeps two
to three cores busy at once: the parent busy-spins while it waits for each reply; the
deciding worker needs its own core; and, because workers are never reused across
tables, every table starts one fresh sandboxed Python per seat (about 130 ms each)
during its first hands. The old default of `min(4, cpus // 2)` jobs (and `make`'s
`--jobs $(nproc)`) oversubscribed the host. This was not a CPU-time versus wall-clock
mix-up, and steal time was negligible.

Pooled timeout rates over every seat, measured on this 8-CPU VM (6 players, 1 ms):

| jobs | arena styles, 24 tables | `strategy.population` lineups, ~25k decisions |
|---|---|---|
| 1 | 0–0.02% | 0.02% |
| 2 | 0.03–0.3% | 0.12–0.24% (unpinned or pinned to 3 cores each) |
| 4 (old default) | 30–33% | — |
| 8 (old `make`) | 66% | ~65% |

One job averaged 1.8 busy cores and peaked at 2.4. Confined to 2 CPUs it timed out
43% of the time, and under 0.1% with 3. With 2 jobs the timeouts cluster in the first
75 decisions of each table (0.49%, against 0.08% mid-table). That is where the
replacement start-ups compete with the deciding worker, so pinning jobs to disjoint
cores did not help and the harness does not pin.

Policy: a sandboxed job gets 4 physical cores, and 1 core is reserved for the harness
process. That gives 1 job on 8 cores, 3 on 16 and 7 on 32. In-process runs, which
enforce no deadline, use every core but one. The count honours CPU affinity and any
cgroup v2 `cpu.max` quota, and counts physical cores rather than SMT threads.
`python -m strategy.cli capacity` prints it. `--jobs` above the safe count is allowed
with a warning; the timeout integrity gate catches the result. The arena CLI
(`arena.benchmark.capacity`) uses the same policy.

**The floor at 1 job is host stalls.** Six sandboxed calling stations (16 µs
decisions), one table at a time, timed out on 0.035% of 360,000 decisions. With the
pool quiesced so no worker started during a table, the rate was 0.039%. Two spinning
monitor processes saw 4–6 stalls of at least 1 ms per minute, and 8–13 of at least
500 µs, with nothing else running. A timeout needs a stall of about 780 µs or more on
the parent's or the worker's vCPU within the decision window, and at that stall rate
the expected timeout rate is about 0.03%.

The harness adds three things, none of them a source of stalls:

- **Fixed overhead.** About 220 µs at the median: 110 µs for the worker to parse the
  observation, and 95 µs for serialization and the pipe.
- **Parent GC pauses.** Gen2 collections in the harness process take 30–50 ms. They
  landed between dispatch and receive on 1–4 of 264 timeouts. The arena now disables
  the parent's collector from dispatch until the reply is read, and freezes the heap
  per table, so none can.
- **Worker start-ups.** They matter only when jobs share cores.

Timer reads (kvm-clock) cost 45 ns. Neither process sleeps on the critical path: the
parent spins, and the worker spins after a wake notice. The guest has no cpufreq
governor or cpuidle driver to tune.

### Null A/B calibration

Arms A and B are the same artifact. A **salted** arm wraps the bot so that
`TableInfo.policy_seed` becomes `sha256(salt | seed)`. This makes policy noise
independent between arms while card luck stays shared, so the true difference is
exactly zero. The pooled blocks are split into disjoint replicate experiments, and
each replicate is analysed like a real A/B. The false-gain rate, meaning replicates
whose interval lies above zero, must be consistent with `alpha/2` (binomial
p ≥ 0.01). An extra cell runs the same salt in both arms, and every block difference
must be exactly 0.

Measured on this VM (8 CPUs). The in-process runs use the stub population at the
in-process default of 7 jobs. The sandboxed run uses the registry at the safe default
of 1 job.

| spec | blocks | full-sample B − A (95% paired t) | paired t false gains | two-sided rejections: paired t, bootstrap | exact coupling | wall time |
|---|---|---|---|---|---|---|
| `null-aa-nlhe` (6 players, cap 50) | 1200 | +0.077 [−0.228, 0.382] | 4/100 (p = 0.24) | 6.0%, 13.0% | identical | 7 min |
| `null-aa-variants` (4 players, cap 40, 3 variants) | 720 | −0.634 [−1.385, 0.117] | 1/60 (p = 0.78) | 6.7%, 15.0% | identical | 2 min |
| `null-aa-sandbox` (6 players, cap 50, sandboxed, registry) | 400 | −0.055 [−0.428, 0.318] | 0/50 (p = 1.0) | 2.0%, 24.0% | identical on 8 of 12 tables; 4 had a timeout | 82 min |

Run directories: `artifacts/lab/null-aa-nlhe/m_adeeb5bfbae46f1155d45ee6a9b1b6c2`,
`null-aa-variants/m_b9825ab6f2e5cc540ddbe8ffb365173b` and
`null-aa-sandbox/m_63d904e9230af8c3fb657977696865d5`. The in-process points match the
runs made before the timeout gate existed, digit for digit. Their manifest ids changed
because the gate fields are part of the spec, and the bootstrap column moved because
its resampling seed is the manifest id.

In the sandboxed run the deadline held. Over every seat, 0.054% (arm A) and 0.038%
(arm B) of about 756,000 decisions per arm timed out, and no single entrant exceeded
0.08%. The candidate's end-to-end p99 was 0.46 ms. Exact coupling compares only
tables without a timeout in either arm, because a forced fold in one arm alone is a
legitimate divergence.

The paired t interval holds its declared rate. The percentile cluster bootstrap is
too liberal with few blocks per replicate (12 in the in-process runs, 4 in the
sandboxed one), so `paired_t` stays the default decision interval. Use the bootstrap
only as a robustness check, or for runs with many more blocks. Pairing narrows the
interval: the unpaired Welch interval on tables was about 1.6 times as wide in the
NLHE run (10.4 against 6.7 points).

## Commands

| command | what it does |
|---|---|
| `make lab-conformance` | conformance of the arena engine for all variants |
| `make lab-report` | run the `lab-smoke` A/B and write its report |
| `make lab-experiment LAB_SPEC=<id or path>` | run any spec |
| `make lab-null` | real A/A null calibration on NLHE (`null-aa-nlhe`) |
| `make lab-null-variants` | the same across NLHE, short deck and PLO4 |
| `make lab-null-sandbox` | sandboxed A/A against the registry at the safe jobs default (`null-aa-sandbox`) |
| `make lab-capacity` | usable cores and the safe `--jobs` for this host |
| `make lab-null-synthetic` | 4000-replicate synthetic null calibration |
| `make verify-strategy` | `pytest tests/strategy` (also part of `make verify-fast`) |

Lab targets use the safe jobs default; `LAB_JOBS=<n>` overrides it.

More commands are available directly: `python -m strategy.cli rules --ruleset uncertainty_plo4`,
`sample-size --sigma 9.2 --delta 2` (or `--pilot <run dir>`), `report <run dir>`,
`experiment <spec> --dry-run`, and `specs`.
