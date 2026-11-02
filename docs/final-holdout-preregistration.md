# Pre-registration: the final bot on the holdout partition (Stage 8 / M8)

Written and committed, and tagged `prereg-final-holdout`, before any holdout run. Nothing
that produced the final bot touched the holdout partition of `strategy.population`
(pop-v1):

- the blueprint, the exploitation layer and the distillation study used train data;
- the selection between teacher and student used validation data.

The holdout is run once. Nothing is tuned after it.

## Question

The final bot is the exploiting blueprint on NLHE, with the generic fallback on every other
ruleset (`strategy/final`). Against holdout lineups, including the six holdout-only
families, does it score more tournament points than the bots it would replace? Does it
avoid losing meaningfully to the plain blueprint against the two holdout-only families the
exploitation layer has no model of?

## What is fixed before the runs

- **The final bot.** Arm B in every comparison is `artifacts/final/bundle`, built by
  `make final-package` at the tagged commit. Its `bundle_sha256` is recorded in
  `strategy/final/holdout-preregistration.json`. `python -m strategy.final.holdout run`
  refuses to start if the built bundle hash differs, or if the tag is not an ancestor of
  HEAD. The bundle plays identically to the pre-registered Stage 6 exploit bundle: the same
  public actions, committed state and zero fallbacks on 160 sandboxed train-partition tables
  (24,355 decisions; `strategy.final.equivalence`). Both packs are unchanged:
  - blueprint pack data sha256 `c21c9d781ba3a00e…`;
  - exploit pack data sha256 `3f74721a3cb10774…`.
- **The distillation decision.** It is fixed and is not revisited on holdout data. The
  table student was non-inferior to the teacher on 16 sampled validation lineups (+0.38
  [−0.52, +1.29], 400 blocks). It lost to the teacher against blueprint-like lineups (−1.68
  [−2.87, −0.50], 150 blocks). So the teacher is shipped and the student is not evaluated
  on holdout.
- **Comparison arms (A).**
  - The MCCFR blueprint with the point-aware leaf: `artifacts/blueprint/bundles/final`,
    the same pack, bundle sha256 `71bb31686969…`.
  - The tuned baseline nlhe-v2: `artifacts/bots/baseline-candidate` from
    `strategy/policy/vectors/nlhe-v2.json`, bundle hash `9b08dc5390f1…`.
  - The Stage 3 endgame-capped bot: `strategy/endgame/bots/capped.py`, file sha256
    `038113bea3eb…`.
- **Harness.** Every comparison uses:
  - `strategy.experiments` stage `final_holdout` and partition `holdout`;
  - 8-player `practice_nlhe` tables with 2,000 chips and a 200-hand cap;
  - every seat rotation per block, with paired arms;
  - in-process isolation and 7 jobs.

  In-process runs import each bundle's `bot.py` with the repository's `strategy` package.
  The sandbox equivalence above is what ties the repository code to the bundle's code.
  Seeds are fixed by each spec's `base_seed`.

## Primary analysis

Four co-primary hypotheses, each on B − A in mean terminal points per candidate seat (an
average seat scores 12.5):

| id | spec | A | opponents | hypothesis | blocks |
| --- | --- | --- | --- | --- | --- |
| H1 | `final-holdout-vs-blueprint` | blueprint | 16 sampled holdout lineups | superiority: H0 mean ≤ 0 | 16 × 40 |
| H2 | `final-holdout-vs-baseline` | baseline nlhe-v2 | 16 sampled holdout lineups | superiority: H0 mean ≤ 0 | 16 × 15 |
| H3 | `final-holdout-vs-endgame-capped` | endgame-capped bot | 16 sampled holdout lineups | superiority: H0 mean ≤ 0 | 16 × 15 |
| H4 | `final-holdout-only-families` | blueprint | 7 × image_responsive; 7 × odd_sizer; 4 image_responsive + 3 odd_sizer (holdout-only families, members listed in the spec) | non-inferiority: H0 mean ≤ −0.5 | 3 × 200 |

The test is a one-sided paired t-test on block differences. The four p-values are
Holm-corrected at family-wise α = 0.05 (`python -m strategy.final.holdout confirm`, via
`strategy.exploit.evaluate.confirm`). H1 to H3 are wins if rejected. H4 passes if rejected,
meaning the final bot loses less than 0.5 points per tournament against families it has no
model of.

**Expected precision.** The block standard deviation was 8.5 to 9.3 points in the Stage 6
and distillation runs. That gives these minimum detectable effects at 80% power and
one-sided α = 0.05:

- H1: about 0.9 points (Stage 6 measured +1.85 on validation);
- H2 and H3: about 1.5 points (the blueprint measured about +5 against the endgame bots);
- H4: its power against a true difference of 0 is between about 40% (block standard
  deviation 9) and 65% (deviation 6, as in the Stage 6 deceptive run).

A null H4 is therefore weak evidence either way, and is reported as such.

The M8 exit holds if all of the following are true:

- the final bot passes the deployed gate (`make gate-final`);
- H4 passes;
- at least H1 is a win.

H2 and H3 are expected wins and are reported regardless.

**Gate status at registration.** Two gate runs of this bundle on this VM passed every
latency check. Both failed the absolute bot-caused ceiling of 0.020%:

- run 1: 30 bot-caused timeouts, in two bursts on two tables;
- run 2: 11 bot-caused timeouts (0.022%), within the control (p = 0.16).

The same code passed the gate in Stage 6 with 6. One more gate run follows the holdout.
Every run is reported, and nothing about the bot changes in between.

## Secondary (exploratory, no claims)

- The harness's per-family slices of H1 to H3, which include each holdout-only family that
  appears in the sampled lineups, with its Holm-adjusted p-values. The per-lineup results
  of H4, and phase slices.
- Execution counts per arm: fallbacks, timeouts, errors and illegal actions. Every count
  must be 0.

## Reporting

All four results are reported with intervals, including nulls and failures. The runner
records a ledger (`artifacts/final/holdout/ledger.json`) and refuses a second run of any
comparison. No spec is re-run with new seeds. If a result prompts a change, the changed bot
is a new experiment and cannot use this partition as unseen evidence again.
