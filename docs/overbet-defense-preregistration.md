# Pre-registration: reading postflop overbets as a pot-sized raise

Written and committed before any run of the confirmation below. The design used train
lineups, exploiter proxies on the target's own blueprint and dev seeds only. The redteam-v1
bots were not played during design: the motivation is the red-team report's finding, not
its tables. The holdout partition is not used.

## Question

The red-team report found that the final bot folds 66–76% to postflop overbets but about 50%
to bets up to a pot. Against the pressure bot, which overbets in about half of its postflop
bets, that costs about −2.5 bb per overbet hand. The v1 translation (`translate=postflop`,
docs/redteam-translation-preregistration.md) won against bet-size exploiters but lost −0.80
against the validation population. Does a translation of off-tree sizes that avoids the v1
loss make the final bot better against overbet pressure, without losing elsewhere?

## Why v1 lost, and what the design keeps

On train lineups (dev seeds), v1 moved the key on 61% of plain postflop facing-bet decisions.
Most population bets fall between trained sizes (0.5–0.75 pot: 548 of 771; 0.25–0.5: 176),
and almost none are overbets. Those bets are value-heavy, so the default reading as the next
trained size up protects the bot. v1 also anchored its lower interval at a price edge
(0.214 pot) instead of a trained size, which read a third-pot bet as a min-bet about half
the time.

Design candidates (all in `strategy/blueprint/translate.py`, outside the default bundle).
`size_map` takes the bettor's pot fraction `x` in the tree's own terms
(`cur + x · (pot + to_call)`), and `read_as` keys the decision as if the mapped bet had been
made: price, stack depth and bet level change together. All-ins, bets that put the hero
all-in, and raises after the last sized raise are never mapped.

| mode | mapping |
| --- | --- |
| `overbet` | `1 < x < x_allin`: pot-sized raise or all-in, pseudo-harmonic |
| `harmonic` | as `overbet`, plus `0.5 < x < 1`: half pot or pot, pseudo-harmonic |
| `shove` | every overbet as the all-in |
| `pot` | every overbet as the pot-sized raise |

Proxies (`strategy/exploiters/bots.py`, `OverbetExploitBot`): the target's blueprint, with
its postflop pot-sized bets made two-pot overbets (`overbet-pot`). `overbet-polar` also
overbets the half-pot bets of its weakest and strongest hands. Dev results (points per
tournament, 95% CI):

| measurement | current bot | `overbet` | `harmonic` | `shove` | `pot` |
| --- | --- | --- | --- | --- | --- |
| overbet-pot's edge, table of 7 × target, seed A, 48 blocks | +1.67 [+0.18, +3.16] | | | | |
| overbet-polar's edge, same | −0.37 [−2.12, +1.38] | | | | |
| overbet-pot's edge, table of 7 × target, seed B, 48 blocks | +0.25 [−1.18, +1.69] | +0.07 | +0.07 | +0.44 | +0.03 |
| sizing-all's edge, table of 7 × target, 48 blocks | −1.06 [−2.79, +0.67] | | −0.11 | | |
| B − A, hero seat vs 7 × overbet-pot, 32 blocks | | −1.41 [−3.14, +0.33] | −0.23 | | −0.59 [−2.07, +0.90] |
| B − A, hero seat vs 7 × overbet-polar, 32 blocks | | −0.74 | −0.14 | | +1.24 [−1.12, +3.60] |
| B − A, hero seat, 16 train lineups × 8 blocks | | +0.05 | +0.28 | | +0.01 [−0.03, +0.05] |

Hero fold rate facing overbet-pot's non-all-in overbets (12 tables each): current 78%,
`overbet` 85%, `shove` 92%, `pot` 74%.

Reading: no design difference is significant. Mapping half-pot-to-pot bets down
(`harmonic`) helps the sizing exploiter. That matches v1's failure: bets read down get
defended as cheaper than they are. Overbet reads that fold more (`shove`, the all-in side of
`overbet`) do worse against both proxies. Reading overbets as a pot defends more and is the
only variant pointing the right way against the bluff-heavy proxy, which resembles the
pressure bot. It also leaves the population almost untouched, because the population almost
never overbets. **The candidate is `translate=pot`**, and bets up to a pot keep the default
reading.

## What is fixed before the runs

- **Arms.**
  - **A** is the shipped final bot, `artifacts/redteam/target-handhq`, bundle sha256
    `c3e0fa540ee27a66873c3e53608f2f0b8d1d9f10c1ddf6b3869b6c90aef09ab3`. This is
    `strategy/final/submission.json`, and the default build of this commit is identical.
  - **B** is `python -m strategy.final.package --out artifacts/redteam/target-pot --override
    translate=pot`, bundle sha256
    `7ce7e5e5a0b01aa1203ed3dd5f231a1ee2e66b9c544fdcf2e4d29abee7e31dee`. B differs from A in
    `bot.py`, which builds `TranslatingFinalBot(..., overrides={'translate': 'pot'})`, and in
    two added files, `strategy/blueprint/translate.py` and `strategy/final/translated.py`.
- **Candidate.** Postflop only. When the hero faces the current street's last raise, the
  raise is not an all-in, calling does not put the hero all-in, and the bettor still had
  sized raises (at most `MAX_SIZED_RAISES` raises on the street), let `x` be the bettor's
  fraction of the pot and `x_allin` the effective all-in's. A bet with `1 < x < x_allin` is
  keyed as the pot-sized raise. Every other decision is keyed as before. The hero's actions
  and their sizes are the real ones.
- **Exploiter for H1.** `artifacts/redteam/handhq-v1/bots/overbet-pot.py`, sha256
  `ce1e17a9ef7fde80550dd5a6d80cf18a547fd3a4086fe131d58a279c3fd5c53a`:
  `OverbetExploitBot(mode="pot", size=2.0)` on A's blueprint pack.
- **Seeds.** The spec seeds are `overbet-defense-exploiter-v1` (both H1 runs) and
  `overbet-defense-population-v1`, and the red-team suite seed is
  `--seed overbet-defense-suite-v1`. No run has used them. The suite's pre-registered seed
  `redteam-v1-suite-a` is not used.

## Primary analysis

Four hypotheses in mean terminal points per tournament. Every table is 8-player NLHE with
2,000 chips, a 200-hand cap and every seat rotation per block, in-process, 7 jobs:

| id | runs | measurement | hypothesis | blocks |
| --- | --- | --- | --- | --- |
| H1 | `overbet-defense-exploiter-vs-a`, `-vs-b` | overbet-pot's edge at a table of 7 × A minus at 7 × B | superiority: H0 ≤ 0 | 2 × 192 |
| H2 | red-team suite, A and B on one seed | B − A in the `single-pressure` cell (7 × pressure bot) | superiority: H0 ≤ 0 | 100 |
| H3 | `overbet-defense-population` | B − A, 16 sampled `pop-v1+handhq-v1` validation lineups | non-inferiority: H0 ≤ −0.5 | 16 × 50 |
| H4 | red-team suite, A and B on one seed | B − A over all 21 suite cells, equal-weight mean of per-cell block differences | non-inferiority: H0 ≤ −0.5 | 1,700 |

Runs:

```bash
uv run python -m strategy.exploit.evaluate run overbet-defense-exploiter-vs-a --jobs 7
uv run python -m strategy.exploit.evaluate run overbet-defense-exploiter-vs-b --jobs 7
uv run python -m strategy.exploit.evaluate run overbet-defense-population --jobs 7
make redteam-suite BUNDLE=artifacts/redteam/target-handhq RT_OUT=artifacts/redteam/runs/overbet-a \
    RT_JOBS=7 RT_ARGS="--seed overbet-defense-suite-v1 --arm ours"
make redteam-suite BUNDLE=artifacts/redteam/target-pot RT_OUT=artifacts/redteam/runs/overbet-b \
    RT_JOBS=7 RT_ARGS="--seed overbet-defense-suite-v1 --arm ours"
uv run python -m strategy.exploiters.validate --h1 <vs-a run> <vs-b run> \
    --suite artifacts/redteam/runs/overbet-a artifacts/redteam/runs/overbet-b --h3 <population run> \
    --out artifacts/redteam/overbet-confirm.json
```

`strategy.exploiters.validate` computes each one-sided p-value:

- **H1:** a t-test on the difference of the two runs' mean edges, with the two variances
  added. The runs share a seed stream, so the test is conservative.
- **H2 and H3:** paired t-tests on block differences.
- **H4:** a t-test on the stratified mean of the suite's `stratified` statistic.

Holm correction over H1–H4 at family-wise α = 0.05.

**Deployed gates** on B, run only if the statistics pass, on an otherwise idle machine:

- `python -m strategy.policy.gate --bundle artifacts/redteam/target-pot --decisions 50000`:
  every check ok (the 1 ms budget).
- `python -m strategy.final.package --verify --out artifacts/redteam/target-pot`: ok, with
  worker peak RSS at most 97.5 MB for NLHE and PLO4. The shipped bundle measures
  97.11–97.29 MB.

**Decision.** Adopt only if:

- H1 or H2 is rejected,
- H3 and H4 are both rejected, and
- both gates pass.

Adopting makes `translate=pot` the default of `strategy.final.package`: the default bundle
is B's, with a new hash. In the same change, `strategy/final/submission.json` and its README
record the new bundle and its gate runs, and the drift test then pins the new hash.
Otherwise the default bundle stays `c3e0fa54…` and the mode stays available behind
`--override`.

## Results

The runs were in-process with 7 jobs, from commit `3653a74` (tag `prereg-overbet-defense`,
pushed before the first run), with the pre-registered bundles, seeds and block counts
unchanged. The suite runs record `commit 3653a74`, seed `overbet-defense-suite-v1`, and
arms `c3e0fa540ee2` (A) and `7ce7e5e5a0b0` (B). The analysis is
`python -m strategy.exploiters.validate` exactly as written above, saved as
`artifacts/redteam/overbet-confirm.json`.

| id | B − A (points), 95% CI | one-sided p | Holm p | outcome |
| --- | --- | --- | --- | --- |
| H1 | +0.38 [−0.68, +1.44] (exploiter edge −0.11 vs A, −0.49 vs B) | 0.24 | 0.48 | not shown |
| H2 | −1.62 [−2.89, −0.34] | 0.99 | 0.99 | not shown: B is worse |
| H3 | +0.02 [−0.06, +0.11] | <0.0001 | <0.0001 | non-inferior |
| H4 | −0.19 [−0.35, −0.02] | 0.0001 | 0.0003 | non-inferior |

**Decision: fail.** Neither H1 nor H2 is rejected, so the decision rule is not met. The
gates were not run, as the pre-registration specifies. `translate=pot` is not adopted, the
default bundle stays `c3e0fa54…`, and the mode stays available behind `--override`.

Descriptive, not part of the test: the suite's attribution rows for `single-pressure`
show that B lowers the fold rate to pressure-bot overbets from 67% to 54%, against 51% to
smaller bets, which clears the suite's OVERBET flag. Chips per overbet hand fall from −1.83
to −2.62 bb, and the hero survives 353 of 800 tables against 379. The per-cell differences
are negative mainly where the pressure bot sits: `mult-P4E3` −1.08, `mult-A4P3` −0.45,
`mult-P3E2A2` −0.37. The cells without it are about 0. The equilibrium and adaptive cells are
exactly 0.00, because neither bot overbets, so the mode never fires there.
