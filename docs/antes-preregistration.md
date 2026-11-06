# Pre-registration: first-class antes (`antes=tables`)

Written and committed before any run of the confirmation below. The design used the train
partition (HandHQ train lineups and the pop-v1 census, seeds `…-train-…`) and a dev draw of
train lineups (seeds `…-dev-…`) only. The validation partition has not been played by any
ante candidate, and the holdout partition is not used.

## The ante structure is assumed

No official ante schedule has been given. Both posting modes are taken to occur, and the bot
must be good in each. The schedules are working assumptions (`strategy/antes/structures.py`),
on the practice tournament: 8 seats, 2,000 chips, 25 hands per level, blinds 10/20 to 200/400,
a 200-hand cap and 50/30/20 payouts.

| mode | antes by level (level 1 → 8) | who posts |
| --- | --- | --- |
| `classic` | 0, 3, 5, 10, 15, 25, 40, 50 | every dealt seat, before the blinds; a contribution, so a short ante caps a side pot |
| `big_blind` | 0, 30, 50, 100, 150, 200, 300, 400 | the big blind, for the table; dead money in the main pot |

The solved tables are not tied to these schedules. They are parameterised by the ante-to-BB
ratio and the posting mode:

- one table per posting mode;
- states are solved at schedule rollouts plus synthetic states whose ante is drawn as a ratio
  of the big blind (per seat 0.05–0.2 for classic, 0.5–1.25 for the big-blind ante);
- the key carries the bucket of the total ante money over the big blind (edges 0.15 to 2.3).

A different official schedule within these ranges needs new specs, not new tables or code.

## Question

With antes in the pot, does the final bot lose points that ante-aware play recovers? The
paired census (final bot with antes minus without, 200 blocks) shows a deficit of −1.05 and
−0.90 points (classic, big blind; train draw), not significant on its own. The blueprint's
price bucket reads the antes as a limper's chips, so the bot opens less (28% → 21–23%) and
limps about three times as often.

## Candidate

`antes=tables` (`strategy/antes/`, behind `strategy.final.package --override antes=tables`).
It changes only hands with an ante; hands without one play exactly as the final bot.

- **Preflop pot read (`stripped`).** In a hand with an ante, the blueprint keys preflop
  decisions on the pot without the antes, the context the trainer used for the same blinds
  and bets.
- **Ante push/fold tables.** Preflop push/fold spots of a hand with an ante play from the
  posting mode's table: ICM, solved with the antes posted (classic antes as contribution
  levels, a big-blind ante as dead money). Spots outside the table's support fall back to the
  final bot's own push/fold leaf. The tables are in the endgame format
  (`strategy.endgame.lookup.PushFoldTable` over `store.stream_arrays`), cut to the rows a
  lookup uses: `strategy/tables/data/antes-pushfold-{classic,big_blind}-practice_nlhe.npz`.
  The cut equals the full solve on 383,532 lookups.
- **Warm-up.** It replays 48 recorded preflop ante observations
  (`strategy/antes/data/warm-practice_nlhe.json.gz`) after the normal warm-up.

Not carried forward:
- the ante-trained preflop blueprint overlay (`antes=overlay`, a 100M-pair fork of the final
  shards): overlay minus tables was −0.42 and −0.13 on train and −0.36 and +0.07 on dev;
- a longer overlay run.

## Design evidence (train and dev only)

Points per tournament, B − A, 95% CI:

| measurement | classic | big blind |
| --- | --- | --- |
| census, 200 blocks, train draw: `stripped` − final | +1.18 (−0.13, +2.50) | +2.14 (+0.76, +3.51) |
| census, train draw: `tables` − final | +2.01 (+0.79, +3.23) | +2.80 (+1.51, +4.09) |
| census, dev draw: `tables` − final | +1.08 (−0.26, +2.43) | +1.73 (+0.16, +3.30) |
| HandHQ train lineups, 80 blocks, raw | +0.93 (−1.38, +3.24) | +1.05 (−1.27, +3.37) |
| HandHQ train, 240 blocks, standard deal, luck-adjusted | +0.71 [−0.17, +1.59] | +1.05 [+0.15, +1.96] |
| HandHQ train, 240 blocks, seat-keyed, luck-adjusted | +0.82 [−0.01, +1.66] | +1.61 [+0.74, +2.47] |

**The luck-adjusted estimator and seat-keyed dealing with antes**
(`tests/strategy/test_antes_variance.py`, and the four 240-block runs above):

- The all-in term averages to exactly zero over every one- and two-card runout of real ante
  hands.
- The re-split of the realized runout reproduces the table's final stacks, side pots
  included, so the antes (and the big-blind ante's dead money) are in the pots the terms
  re-split.
- Points-equity telescopes to the official points.
- Seat-keyed dealing keeps every seat's cards across arms that play differently.
- In each run every luck term's mean (A, B, B − A) is within sampling error of zero: the
  largest |t| was 2.64 over 99 statistics. The correction is −0.00 ± 0.44, +0.02 ± 0.42,
  +0.23 ± 0.42 and −0.13 ± 0.38.
- Variance factor of the four default terms: 1.91 and 1.91 (classic), 1.80 and 1.72 (big blind).
  Seat-keyed dealing adds 1.10× [0.84, 1.42] after the luck terms in both modes, for 2.11×
  and 1.97× against the raw standard deal. The adjusted block SD with both levers is 6.57
  (classic) and 6.79 (big blind).

**Other checks:**
- Ante-free play: B equals A action-for-action in 40 of 40 sandboxed tables
  (`strategy.final.equivalence`; 44,155 public actions, 6,216 hero decisions, 0 fallbacks).
- Worker peak RSS of a tables bundle: 92.8 MB, against 77.6 MB for A.

**Sizing.** At a block SD of 6.8 and α = 0.025 (Holm's first step), 960 blocks give 88%
power at a 0.75-point effect (93% at α = 0.05) and an 80%-power MDE of 0.68.

## What is fixed before the runs

- **Arms.**
  - **A** is the shipped final bot, `artifacts/antes/bundle-final`, bundle sha256
    `59f371c60f15d98c03539af8431eaf23b6749cceff4048407018686904593395`. This is
    `strategy/final/submission.json`, and the default build of this commit is identical.
  - **B** is `python -m strategy.final.package --out artifacts/antes/bundle-candidate
    --override antes=tables`, bundle sha256
    `bec8892d62329e6f66ce137ffe55862383306a2c851eff9a6082ae83228e15f7`, 10,140,021 bytes.
    It differs from A in `bot.py`, which builds `AnteFinalBot(..., overrides={'antes': 'tables'})`,
    and adds seven files:
    - `strategy/antes/{__init__,lookup,pushfold,runtime}.py`;
    - `strategy/antes/data/warm-practice_nlhe.json.gz`;
    - the two ante tables.
- **Specs**, `strategy/antes/specs/antes-validation-{classic,big-blind}.json`.
  - `confirmation` stage, HandHQ population source, **validation** partition, 16 sampled
    `pop-v1+handhq-v1` lineups × 60 blocks (960 blocks per posting mode).
  - The table: 8 players, 2,000 chips, 200-hand cap, the mode's schedule above,
    `"deal": "seat_keyed"`.
  - `"variance_reduction": {"terms": ["allin", "flop_pot", "turn_pot", "river_pot"]}`
    (cross-fitted, 10 folds, primary).
  - Seeds `antes-validation-classic-v1` and `antes-validation-big-blind-v1`. No run has used them.
- **Machine.** In-process, 7 jobs, numba installed (the luck terms).

## Primary analysis

Two hypotheses in mean terminal points per tournament, B − A:

| id | spec | hypothesis | blocks |
| --- | --- | --- | --- |
| H1 | `antes-validation-classic` | superiority: H0 mean ≤ 0 | 16 × 60 |
| H2 | `antes-validation-big-blind` | superiority: H0 mean ≤ 0 | 16 × 60 |

The estimate is the luck-adjusted one, and the test is one-sided t on it (its SE and df).
Holm correction over H1 and H2, at family-wise α = 0.05:

```bash
uv run python -m strategy.exploit.evaluate run strategy/antes/specs/antes-validation-classic.json --jobs 7
uv run python -m strategy.exploit.evaluate run strategy/antes/specs/antes-validation-big-blind.json --jobs 7
uv run python -m strategy.exploit.evaluate confirm --sup H1=<classic run> --sup H2=<big-blind run> \
    --out artifacts/antes/confirm.json
```

The raw paired-t result is reported next to it, and is not used for the decision.

**Side conditions**, on B's bundle, run only if the statistics pass, on an otherwise idle
machine:

- **Ante-free play.** `python -m strategy.final.equivalence --a artifacts/antes/bundle-final
  --b artifacts/antes/bundle-candidate --tables 40`: identical.
- **Deployed gate, once per structure.** `python -m strategy.policy.gate --bundle
  artifacts/antes/bundle-candidate --decisions 50000`, once with no `--structure` and once
  with each of `--structure strategy/antes/specs/structure-{classic,big_blind}.json`: every
  check ok (the 1 ms budget).
- **Bundle limits.** `python -m strategy.final.package --verify --out
  artifacts/antes/bundle-candidate`: ok, with the bundle and one sandboxed worker's peak RSS
  under 100,000,000 bytes, and numba not loaded.

**Decision.** Adopt B only if H1 and H2 are **both** rejected and every side condition passes.

Adopting makes `antes=tables` the default of `strategy.final.package`. The default bundle is
then B's, with B's hash. In the same change, `strategy/final/submission.json` and its README
record the new bundle and its gate runs, and the drift test pins the new hash. Otherwise the
default bundle stays `59f371c6…` and the mode stays available behind `--override`.

## Secondary (exploratory, no claims)

- raw paired-t estimates;
- per-lineup results;
- the luck-term diagnostics (term means, fitted slopes);
- the variance factor with antes on validation lineups.

## Reporting

All results are reported with intervals, including nulls and failures. No spec is re-run
with new seeds to change a result. If a result prompts a change, the changed bot is a new
experiment with new specs.

## Results (added after the runs)
