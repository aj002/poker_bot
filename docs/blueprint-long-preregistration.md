# Pre-registration: the 3.2B-pair blueprint vs the Stage 4 blueprint and the final bot

Written and committed before any validation run of the specs below. The candidate was
trained on the training partition's start states only. The holdout partition is not used.

## Question

Does training the blueprint for about 35 times as many deal pairs (300 CPU-hours in total,
against about 30 for Stage 4) make it score more tournament points, on its own and inside
the final bot?

## What is fixed before the runs

- **Training.** The Stage 4 configuration unchanged (linear discounting for 100 intervals of
  1M pairs, then plain sums; pruning after 5M pairs; regret floor −6), continued from the
  Stage 4 run at 110.64M pairs and sharded over three 8-core machines
  (`strategy/blueprint/shards.py`). Five merged epochs; the final merge is 3,195,920,000 pairs
  and 650.5B nodes touched. The CPU ledger stopped training at 292.6 CPU-hours.
- **Candidate pack.** `strategy/tables/data/blueprint-long/blueprint-practice_nlhe.npz`, data
  sha256 `6e805280a4738db6`, file sha256 `49a7e607c58c5403`, 684,048 / 199,279 / 79,920 /
  15,504 rows per level (the Stage 4 pack `c21c9d78` has 490,539 / 147,184 / 63,758 /
  13,672). It was built with the unchanged pack builder and mass thresholds.
- **Arms.**
  - H1, blueprint only. A is the Stage 4 pack in the blueprint bundle, `python -m
    strategy.blueprint.package --pack strategy/tables/data/blueprint-practice_nlhe.npz --out
    artifacts/blueprint/bundles/stage4` (bundle `da5ed399c9d5`). B is the candidate pack in
    the same bundle, `--pack strategy/tables/data/blueprint-long/blueprint-practice_nlhe.npz
    --out artifacts/blueprint/bundles/long-final` (bundle `c4b5f1a6a28d`). Both use the
    point-aware push/fold leaf.
  - H2, inside the final bot. A is the final bot, `python -m strategy.final.package --out
    artifacts/final/bundle-8f4fd15e` at commit `2fbec14`, bundle sha256
    `8f4fd15ecb57875a6e5173307b91c6262ea590030cbd302f636243f38750ba03` (reproduced byte for
    byte). B is the same package on this commit with only the blueprint pack swapped,
    `--pack strategy/tables/data/blueprint-long/blueprint-practice_nlhe.npz --out
    artifacts/final/bundle-newbp`, bundle `c67c404265ed2e08`. The exploit pack (`3f74721a`)
    and its experts are A's. They were trained against the Stage 4 blueprint, so H2 measures
    a drop-in swap. Retraining the experts is the separate combination step.
- **Train-partition probe.** One timing probe of each arm pair on the train partition (4
  blocks each) was run to estimate cost. Its results were discarded.

## Primary analysis

Two co-primary superiority hypotheses on B − A in mean terminal points per candidate seat.
Every spec uses 8-player NLHE tables with 2,000 chips, a 200-hand cap and every seat rotation
per block, in-process:

| id | spec | opponents | hypothesis | blocks |
| --- | --- | --- | --- | --- |
| H1 | `blueprint-long-vs-stage4` | 16 sampled pop-v1 validation lineups | superiority: H0 mean ≤ 0 | 16 × 40 |
| H2 | `final-newbp-vs-final` | 16 sampled `pop-v1+handhq-v1` validation lineups | superiority: H0 mean ≤ 0 | 16 × 40 |

The test is the one-sided paired t-test on block differences,
`python -m strategy.exploit.evaluate confirm --sup H1=… --sup H2=…`, with Holm correction
at family-wise α = 0.05.

**Deployed gates** on B of H2 (`artifacts/final/bundle-newbp`), on an otherwise idle
machine:

- `python -m strategy.policy.gate --bundle artifacts/final/bundle-newbp --decisions 50000`:
  every check ok (1 ms deadline).
- `python -m strategy.final.package --verify --out artifacts/final/bundle-newbp`: ok. The
  bundle and one sandboxed worker's peak RSS are each at most 100 MiB, and numba is not
  loaded.

**Decision.** The candidate pack is adopted for the combination step only if all three of
these hold:

- H1 is rejected after Holm;
- H2's 95% paired interval lies above −0.5;
- both gates pass.

Otherwise the final bot keeps the Stage 4 pack and nothing is combined. Whether anything
lands in `strategy/final` is decided by the combination's own pre-registered test.

## Secondary (exploratory, no claims)

- **Improvement curve.** The merged epoch packs at 791M, 1.09B and 2.76B pairs, each against
  the Stage 4 pack, 16 × 8 blocks each: `blueprint-long-curve-e2`, `-e3`, `-e4` (discovery,
  validation partition, their own seeds). The final pack's point is H1.
- **Training signals.** Per merged epoch: reached keys, average positive regret per pair, and
  mass-weighted strategy movement since the previous epoch (`shards.py curve`).
- **Slices.** Per-lineup results and phase slices of H1 and H2.

## Reporting

All results are reported with intervals, including nulls and failures. No spec is re-run
with new seeds to change a result. If a result prompts a change, the changed bot is a new
experiment with new specs.
