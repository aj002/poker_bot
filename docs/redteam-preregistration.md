# Pre-registration: the final bot against three blind red-team bots

Written and committed (tag `redteam-prereg-v1`) before any red-team lineup was played at
validation seeds. The machine-readable version is `strategy/redteam/preregistration.json`;
this page explains it. If the two disagree, the JSON is what the suite runs.

## Question

Where does our final bot lose, or only break even, against opponents built by workers who
never saw our code, and which red-team behaviour causes it?

## What is fixed before the runs

- **The opponents.** Three bots from Origin branches `cursor/redteam-pressure-4be9`
  (`29aac75`, aggressive), `cursor/redteam-equilibrium-f733` (`3e41cf6`, balanced) and
  `cursor/redteam-adaptive-bot-e0e8` (`a082f3b`, an adaptive exploiter that switches its
  own image), merged unchanged under `redteam/`. They are registered as the opponent family
  `redteam-v1` in **validation only** (`strategy/redteam/registry.py`, lock
  `redteam.lock.json`). Train and holdout are refused by the source; no red-team bot is in
  pop-v1 or pop-v1+handhq-v1; the holdout is never read.
- **The bot under test.** The final bot with the HandHQ exploit pack: `python -m
  strategy.final.package --out artifacts/redteam/bundles/final-handhq --exploit-pack
  strategy/tables/data/handhq/exploit-practice_nlhe.npz`, bundle sha256 `c3e0fa540ee2…`.
- **References**, each played in the same seat on the same deals:
  `blueprint-only` (`da5ed399c9d5…`), the baseline `nlhe-v2` (`32d7dfd9b9f7…`) and the
  `previous-final` bot, i.e. the final bot with the default exploit pack (`a2dbd67f84b0…`).
  They are also the "previous bots" in the fourth lineup set.
- **Tournaments.** Official format, 8 players, 2,000 chips, 200-hand cap, official terminal
  points (100 per table, ICM share at the cap), in process. A *block* is one deal seeded
  from the cell's seed, played once per seat rotation (8 tables) with the bot under test
  rotating through the seats; every arm sees the same deals and the same opponent policy
  seeds. The unit of analysis is the block's mean points, so tables sharing a deal are not
  treated as independent. Par for 8 equal seats is 12.5 points.

## Lineups (21 cells)

| set | cells | blocks each |
| --- | --- | --- |
| `single`: seven copies of one red-team bot | pressure, equilibrium, adaptive | 100 |
| `multiples`: red-team bots only, in multiples | 4+3 (P4E3, E4A3, A4P3) and 3+2+2 (P3E2A2, E3A2P2, A3P2E2) | 100 |
| `previous`: red-team bots plus our previous bots | P4, E4, A4 each with blueprint-only, nlhe-v2 and previous-final; and one each of P, E, A with blueprint-only twice, nlhe-v2 and previous-final | 100 |
| `population`: one of each red-team bot plus four population members | 8 draws of four pop-v1+handhq-v1 **validation** members (seed `redteam-v1-population-a`) | 50 |

Seeds are `sha256("redteam-cell-seed-v1|<base seed>|<cell id>")`, independent of the
bundle, so a new bundle is evaluated on the same deals (paired across bundles).

## Hypotheses and decision rules

1. **Primary (per cell).** H0: the final bot's mean points per tournament equal 12.5. Two-sided
   t-test on block means, Holm across the 21 cells at α = 0.05.
2. **Classification (per cell and per set, no multiplicity claim).** From the 95% interval of
   the final bot's mean: **LOSES** if the upper bound is below 12.5, **WINS** if the lower
   bound is above 12.5, otherwise **BREAK-EVEN** (not distinguishable from par at this size).
   Set-level intervals use the equal-weight mean of the cells' means and within-cell variances.
3. **Secondary.** The final bot minus each reference on the same deals, per cell and per set,
   with 95% paired intervals.
4. **Attribution (exploratory, descriptive).** Three behaviour flags, computed from each
   table's event log (`strategy/redteam/diagnostics.py`) and applied to every arm:
   - **OVERBET**: the hero folds at least 10 points more often to a red-team post-flop overbet
     (more than the pot after calling) than to a bet of up to one pot, and the hero's chips per
     hand (big blinds) in hands where it was live when a red-team seat overbet have a 95%
     bootstrap interval below zero (at least 30 overbets faced).
   - **BUBBLE**: the hero's summed points-equity change over hands that start with 4 or 3
     players alive is negative with a 95% bootstrap interval below zero, and the red-team
     seats' preflop 3-bet/4-bet/shove rate per hand dealt is higher at 4 or 3 players than at
     6 or more.
   - **IMAGE**: with an adaptive opponent in the cell, the hero's chips per hand (bb) differ
     between that opponent's tight and loose phases with a bootstrap interval excluding zero.
   A flag is an association on these tables, not a causal claim; because the rules run on
   the references too, a behaviour that hurts every bot equally shows up as such.

## Gates

Each red-team bot is checked with the deployed sandbox gate before use
(`python -m strategy.policy.gate --bundle <bot> --cost-tables 0`, 50,000 decisions, idle
machine; `python -m strategy.redteam gate` runs all three). The tournaments are in process
(no deadline is enforced), so a bot that fails the gate on timeouts still plays at its
intended strength here; the gate result is reported as a property of the opponent.
Integrity for every arm: zero illegal actions, errors, fallbacks and invalid tables.

## Stopping

Block counts are fixed. No cell is rerun with new seeds to change a result. Smoke runs
(timing, bug-finding) use base seed `redteam-smoke`; their results are not read for
hypotheses. A bot tuned on these results must be evaluated with a new `--seed`.

## Results

Added to `docs/redteam-report.md` in the project store after the runs.
