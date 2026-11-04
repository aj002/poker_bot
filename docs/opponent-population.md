# Opponent population

`strategy/population/` is the opponent population for training and evaluation. It has
four parts:

- 17 rule-based families built on the six arena styles;
- a hierarchical, correlated parameter sampler;
- a frozen registry with disjoint **train / validation / holdout** partitions;
- checkpoint slots for frozen learner checkpoints (family 18).

Evaluation uses it through the harness's partition protocol:

```json
"population": {"source": "strategy.population:PopulationSource", "partition": "validation", "lineups_per_variant": 3}
```

Every member plays No-limit Hold'em, short deck and PLO4 legally. The behaviour is
calibrated in depth for No-limit Hold'em only.

## Families

Each family is an **anchor** style from `arena/agents/styles/` plus sampled knobs.
Families 7 to 17 also add one behaviour: a hook that takes some decisions first, or a
rewrite of bet sizes. Any decision a hook leaves alone goes to the anchor's rule table,
with the anchor's numbers and ranges replaced by the member's knobs. The arena styles
keep their exact behaviour (a knob defaults to the old constant).

"Signature" lists the measured statistics that tell a family apart. The ranges are
declared in `Family.expected`; the tests check those ranges and the contrasts. All
numbers come from official-format NLHE tables: 8 seats, 2,000 chips, 200-hand cap.

| # | id | anchor | behaviour | signature (pooled, NLHE) | regime switch |
|---|----|--------|-----------|--------------------------|---------------|
| 1 | `selective_aggressor` | rock | tight raise-or-fold, wider late, rarely calls a raise | VPIP ≈ 0.13, cold-call ≈ 0.02, folds to a reraise ≈ 0.8 | opens up |
| 2 | `wide_pressure` | aggressive | high raise-first, c-bets and barrels, light reraises | VPIP ≈ 0.22, c-bet ≈ 0.86, heads-up bluff ≈ 0.3 | stops barrelling |
| 3 | `chaotic_aggressor` | super_aggressive | very wide, big varied sizes; persistent, tilting or stack-triggered "hot" mode | VPIP ≈ 0.40, 3-bet ≈ 0.17, bottom-range entries ≈ 0.34 | calms down |
| 4 | `loose_passive` | station | limps and calls wide, sticky with draws, rarely raises | PFR ≈ 0.004, limp ≈ 0.45, cold-call ≈ 0.5, aggression ≈ 0.07 | tightens up |
| 5 | `tight_surrender` | weak_tight | low entry, folds to pressure, rare river bluffs | VPIP ≈ 0.08, calls big bets ≈ 0.06 | gets braver |
| 6 | `smooth_switcher` | switcher | soft gates on M, the next blind level and hands to the cap | push/fold entries outside the top 35% ≈ 0.2 | becomes patient |
| 7 | `limp_iso` | aggressive | limps, over-limps, isolates, limp-reraises premiums | limp ≈ 0.27, PFR ≈ 0.04 | stops limping |
| 8 | `squeezer` | aggressive | reraises much more after a raise and a call | squeeze ≈ 0.17 (anchor ≈ 0.06) | stops squeezing |
| 9 | `trapper` | aggressive | flats premiums, checks strong hands, check-raises | checks strong hands ≈ 0.45 (anchor 0), check-raise share ≈ 0.47 | abandons traps |
| 10 | `texture_cbettor` | aggressive | c-bet frequency and size by texture and field size | c-bet dry heads-up ≈ 0.9 vs wet ≈ 0.5 | autopilot c-bets |
| 11 | `draw_player` | aggressive | chases draws at poor prices or attacks with them (sampled attack level) | continues with draws ≈ 0.65 (anchor ≈ 0.35) | only chases |
| 12 | `sizing_tells` | aggressive | size rises with strength, falls with it, or stays flat | size–strength correlation ≈ −0.99 / ≈ 0 / ≈ +0.98 by mode | the tell fades |
| 13 | `image_responsive` | aggressive | per-seat open-rate memory; reraises loose openers wider | reraise vs loose / vs tight openers ≈ 2.9× (control ≈ 1.4×) | stops adjusting |
| 14 | `icm_cap` | aggressive | ICM risk premium on big calls (correct, overweighted, ignored or inverted), big-stack pressure, short jams near the cap | short-stack raises all-in ≈ 0.99 (anchor ≈ 0.5) | tightens near the money |
| 15 | `multiway_specialist` | aggressive | overcalls multiway, suppresses multiway bluffs, side-pot aware | multiway bluff ≈ 0.03 (anchor ≈ 0.3) | bluffs multiway |
| 16 | `odd_sizer` | super_aggressive | min-raises, overbets, irregular amounts, frequent all-ins | min-raise share ≈ 0.13 (everyone else 0), size CV ≈ 0.8 | sizes normally |
| 17 | `variant_noisy` | aggressive | generic stack and price play over noisy hand reading | weak-hand entries ≈ 3.5× its noise-free twin | gets noisier |
| 18 | `frozen_checkpoint` | – | registry slot for frozen learner checkpoints and approximate responses | – | – |

Families 13 and 16 are **holdout-only**: they never appear in train or validation.

## Sampling

Each family has a `Prior` (`strategy/population/params.py`). For every parameter it
declares bounds `[lo, hi]`, a centre and a spread, plus pairwise correlations. These
form the family-level centre and covariance.

An individual member is drawn as follows:

1. Draw \(z \sim N(0, I)\), truncated at \(|z_i| \le 2.5\).
2. Correlate the draw through the Cholesky factor \(L\) of the correlation matrix.
3. Map each coordinate onto its bounds with a logit-normal:
   \(x_i = lo + (hi - lo)\,\sigma(\mathrm{logit}(c_i) + s_i (Lz)_i)\).

Categorical choices are real values in `[0, k)` read with `int()`. Examples are the
chaotic mode, the sizing-tell sign and the ICM calibration. The options are close to
equally likely.

There are two kinds of within-member variation:

- **Table-local regime switch.** With probability `regime_p` (0.25), a member's table
  draws a switch hand in 30–169 from its policy seed. From that hand on, the member
  plays with its parameters shifted by the family's `regime` vector (in logit units).
- **Hot modes.** These are state-triggered shifts: the chaotic aggressor's tilt after a
  big loss, or its stack-triggered mode as the big stack.

Labels are never exposed:

- A bot's name is its opaque member id, `pm-<12 hex>`.
- Actions carry only a type and an amount, with no notes.
- The family, regime and mode live only in the bot object and its private state.
- The tests check that no family id appears anywhere in a table's event log.

## Partitions and the registry

`Registry` (`registry.py`) is a pure function of `REGISTRY_VERSION` (`pop-v1`), the
priors and the counts.

- **Members.** Each regular family has 14 train, 3 validation and 3 holdout members
  (70/15/15). Holdout-only families have 6 holdout members. That is 312 members in
  total. Member `i` of family `f` in partition `P` is drawn with seed `pop-v1|P|f|i`.
- **Disjoint parameter regions.** Each family's latent draw has a fixed random direction
  \(w\). The value \(u = \Phi(w \cdot z)\) is cut into 20 bins:
  - validation owns bins {3, 10, 17};
  - holdout owns bins {6, 13, 19};
  - train owns the other 14.

  A draw is kept only if it lands in its partition's bins, at least 0.002 from a bin
  edge. The sampling map is invertible, so `region_of(prior, params)` recovers any
  stored member's bin. The tests check every member.
- **Disjoint seeds.** Member seeds, lineup draws and table seeds are all namespaced by
  partition and variant.
- **Withheld families.**
  - Two families are holdout-only: `image_responsive` and `odd_sizer`.
  - Three family pairs are validation-only combinations: (`squeezer`, `limp_iso`),
    (`trapper`, `texture_cbettor`) and (`icm_cap`, `smooth_switcher`). No train lineup
    contains both families of a pair, and every validation lineup contains one pair.
  - Every holdout lineup contains a holdout-only family.

### The lock

`strategy/population/registry.lock.json` freezes the following:

- every member id, partition and spec hash;
- the registry hash;
- the hash of the generated card tables;
- every default lineup-pool hash;
- a **behaviour fingerprint** per variant: the hash of every action at short seeded
  tables seating one member of each family.

The fingerprint catches code changes that alter decisions; member hashes only freeze
parameters. `tests/strategy/population/test_registry.py::test_lock_matches` fails when
anything moves. `PopulationSource` also refuses to start if the registry hash differs
from the lock.

To change a prior, a count, a rule or the tables:

1. Bump `REGISTRY_VERSION`.
2. Run `uv run python -m strategy.population.registry --write-lock`.
3. Commit the new lock.

Never re-lock under the same version: old manifests name that version.

## Interfaces

### `PopulationSource` (for `strategy.eval` and experiment specs)

`strategy.population.source.PopulationSource` implements
`strategy.eval.population.PartitionSource`:

```python
source = PopulationSource()        # or load_partition_source("strategy.population:PopulationSource")
source.registry_id                 # "strategy.population/pop-v1"
source.registry_hash()             # frozen registry hash (== lock)
part = source.partition("train")   # Partition: members as OpponentRef, seed namespace "pop-v1|train"
source.sample_lineups(part, ruleset_id="practice_nlhe", opponents=7, count=16, seed="exp-seed")
```

The `OpponentRef` fields are filled as follows:

- `member_id`: the opaque member id;
- `family`: the family id, for evaluation slices and reports only;
- `policy`: a generated bot bundle, `artifacts/population/pop-v1/<member_id>/`;
- `rulesets`: all three variants;
- `params_hash`: the member's spec hash.

A bundle holds `bot.py` (the member's frozen spec) plus a copy of the decision-time
runtime: the `policy`, `families`, `behaviours`, `cards` and `params` modules and the card
tables. It is written on first use and is byte-identical on every machine. The harness
resolves, hashes, stages and sandboxes it like any other bot. Its artifact hash
therefore covers the code that plays the member, not only the parameters. The policy
path is relative to the working directory when the bundle lies under it; the harness
already runs from the repo root.

`sample_lineups` behaves as follows:

- It is deterministic in (partition hash, ruleset, opponent count, seed), with one
  stream per lineup.
- It draws distinct families under the partition rules above, then one member of each.
- When the partition has promoted checkpoint slots, the last seat is a checkpoint with
  probability 0.5. The draw never displaces the forced validation pair or the
  holdout-only family.

`strategy/experiments/specs/population-smoke.json` is a worked example covering all
three variants.

### `get_lineup_pool` (for training loops)

```python
from strategy.population import get_lineup_pool
pool = get_lineup_pool("train", "nlhe", opponents=7, lineups=64)   # aliases: nlhe, shortdeck, plo4
pool.lineups                 # tuple[Lineup(lineup_id, partition, variant, entries)]
pool.bots(lineup)            # [(entrant name, bot)] for in-process play
pool.entrants(lineup, dir)   # [(entrant name, bot bundle)] for sandboxed runs
pool.table_seeds(n)          # table seeds reserved for this partition and variant
pool.label(entry_id)         # family or checkpoint kind (reports only)
pool.hash, pool.manifest()
```

Pools come from the same registry and partition rules as `PopulationSource`.

### Checkpoint slots (family 18)

`CheckpointLedger` (`strategy/population/checkpoints.json`) is an append-only list of
`CheckpointSlot` entries. Each entry records:

- the slot id (`ck-<artifact hash prefix>`);
- the kind (`learner_checkpoint` or `approximate_response`);
- the partition;
- the artifact path and its `artifact_hash`;
- the rulesets it plays;
- its status (`candidate`, `promoted` or `retired`);
- the promotion evidence.

The ledger is used as follows:

```python
ledger = CheckpointLedger()
slot = ledger.register("runs/ck-42/bot.py", kind="learner_checkpoint", partition="train",
                       rulesets=["practice_nlhe"])
ledger.promote(slot.slot_id, "promotion-run-id")   # only promoted slots enter pools
```

A slot whose artifact has changed fails `verify()` and is refused. Checkpoints are
outside the lock; the partition and pool hashes include them.

## Variants

`cards.py` gives every variant the Hold'em vocabulary the style rules use.

**Preflop.** A hand is placed at its *top fraction* within that variant's own starting
hands, using the generated `data/preflop.json`. The tables are:

- No-limit Hold'em: heads-up equity for all 169 classes;
- short deck: heads-up equity for all 81 classes;
- PLO4: 400 score quantiles.

The hand is then mapped to the NLHE class at the same fraction.

**Post-flop.** The evaluator depends on the variant:

- Hold'em uses the style evaluator.
- Short deck ranks a flush above a full house and counts A-6-7-8-9 as a straight.
- PLO uses exactly two hole cards and three board cards, through a prime-product lookup
  (about 12 µs at the river).

Regenerate the tables with `uv run python -m strategy.population.gen_tables`, then
re-lock.

## HandHQ extension (`pop-v1+handhq-v1`)

Two more families, calibrated on real six-max players from the HandHQ corpus
(PokerStars, July 2009, 25NL–1000NL):

- `handhq_regular` (19) is the raise-or-fold regular.
- `handhq_recreational` (20) limps, cold-calls, calls down light and min-raises.

They live in `families.HANDHQ_FAMILY_LIST`. They are registered only by
`strategy/population/handhq.py`, in train (14 members each) and validation (3 each),
never in holdout. pop-v1's members, lock and partitions are unchanged.

- Experiment specs reach the extension through
  `"source": "strategy.population.handhq:HandhqPopulationSource"`.
- Its lock is `handhq.lock.json`. After a family change, rewrite it with
  `python -m strategy.population.handhq --write-lock`.

The real-player frequencies and Dirichlet priors are in `data/handhq-priors.json`. The
pipeline that produces them:

```bash
uv run python -m strategy.data.handhq_fetch            # pinned sparse checkout, about 6 GB, outside git
uv run python -m strategy.data.handhq_stats --root "$(uv run python -m strategy.data.handhq_fetch --where)" \
  --jobs 7 --files-per-stake 1250 --out artifacts/handhq/stats
uv run python -m strategy.data.handhq_priors fit --stats artifacts/handhq/stats --out strategy/population/data/handhq-priors.json
uv run python -m strategy.data.handhq_priors census --tables 900 --jobs 7 --out artifacts/handhq/census.json
```

The census compares every style bot and family with the real players' 5–95% bands.

## Measurement and tests

`measure.census` seats entries at seeded official-format tables and returns one
mergeable `Tally` per entry. It can fork worker processes, and the result does not
depend on how many. The statistics are documented in `measure.py`. They use each seat's
own hole cards, so they are diagnostics, never learner inputs.

`tests/strategy/population/` covers:

- **Declared ranges.** Family ranges are checked on a census of five registry members
  per family. Each member's VPIP and PFR must also fall inside its family's range.
- **Signatures.** These are checked as contrasts against reference families, and with
  per-mode or ablation censuses:
  - the sizing-tell sign by mode;
  - variant noise against its noise-free twin;
  - image response against a control.
- **Determinism.** Play is identical under the same seed on all three variants and
  differs under a different seed.
- **Label hiding.**
- **Sampling.** Draws are bounded and correlated as declared, and the truncation holds.
- **Regime frequency.**
- **ICM math.**
- **Card adapters.**
- **Partition disjointness.** This covers parameter regions, ids, hashes, seeds and
  lineup rules.
- **The lock.**
- **The checkpoint ledger.**
- **The `PartitionSource` contract,** including a materialized experiment plan.
- **The sandbox.** Every family and a promoted checkpoint slot play all three variants
  in sandboxed workers with the 1 ms budget. No decision is accepted over budget, no
  fallback other than a timeout occurs, and timeouts stay at or below 1%.

The arena never imports `strategy` (`tests/strategy/test_isolation.py`). Members reach
sandboxed workers as self-contained bundles. Inside the sandbox they import only their
bundled runtime and the arena SDK and style engine, which the worker preloads.

**Sandbox timing.** A single sandboxed run of the population is fine: in a 4,155-decision
run, p99 end-to-end latency was 0.49 ms and 3 decisions timed out. Under the harness
CLI's default parallel jobs on this 8-CPU VM, however, both arms timed out on 20–32% of
decisions with the stub population as well, so this comes from contention, not from
population bots. With `--jobs 1`, the same three-variant run had no timeouts.

## Red-team extension (`redteam-v1`)

Three bots written blind (by workers who never saw `strategy/`) are registered as the family
`redteam-v1` in **validation only**: `rt-pressure`, `rt-equilibrium` and `rt-adaptive`
(`redteam/`, `strategy/redteam/registry.py`, lock `strategy/redteam/redteam.lock.json`).
They are fixed code, pinned by artifact hash. Specs name the source as
`"strategy.redteam.registry:RedteamPopulationSource"`; asking it for `train` or `holdout`
raises `PermissionError`. pop-v1 and `pop-v1+handhq-v1` are unchanged and do not contain
them. The tournament suite that plays a bot against them is described in
`docs/redteam-suite.md`.
