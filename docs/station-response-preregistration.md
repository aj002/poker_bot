# Pre-registration: the calling-station response (`station=value`)

Written and committed before any run of the confirmation below. The design used the train
partition only: the HandHQ train lineups, the pop-v1 census, the arena's fixed fields, and the
pilots `station-pilot-*` and `station-pilot2-*` (seeds `…-pilot-…`, `…-pilot2-…`, `it2`,
`nl94`). The validation partition has not been played by any station candidate, and the
holdout partition is not used.

## Question

The losing diagnosis (NL-94) found that the final bot donates chips to calling stations. It
bluffs into players who do not fold, and stacks off with medium hands. Against the arena's
calling-station field, the shipped bot busts by hand 50 in 69% of tournaments.

Does value-only play against a *detected* station win points where stations sit, without
costing points in the population field?

## Candidate

`station=value` (`strategy/exploit/station.py`, behind `strategy.final.package --override
station=value`; with antes it builds `StationFinalBot`, an `AnteFinalBot` whose NLHE teacher is
`StationExploitBot`). It changes only decisions where a live opponent is a detected station:
the seat's `station` posterior is at least 0.8 after at least 6 observations, and the seat is
not cooling down after a switch alarm. Every other decision takes the exploit bot's path
unchanged.

At a detected-station decision, the blueprint row (after any expert blend) is reshaped before
the bot's usual random number samples it:

- **No bluffs.** A postflop bet or raise needs equity bin ≥ 5, or ≥ 6 with two or more
  opponents. Otherwise its mass checks, or, facing a bet, calls with bin ≥ 3 and folds below.
- **No stacking off light.** A raise committing ≥ 40% of the remaining stack needs bin 7.
  Below that, a value hand bets the largest size under 40%. Calling a station's bet of that
  size needs bin 6.
- **Thinner value.** With a value hand and nothing to call, half of the check mass bets.
- **Preflop.** A jam needs strength bucket 13, or 9 at 15 BB or less effective. A weaker jam
  raises the smallest size instead. The blueprint's other raises, isolation included, stand
  (`pre_raise` 0). On a blueprint miss, preflop raises start at bucket 8.

The push/fold leaf is unchanged. Settings are frozen in `StationConfig` at this commit. The
`max_bb` depth gate exists but is off.

## Design evidence (train only)

**Iteration (harness, raw points per tournament, B − A, classic antes).** The first candidate
(`pre_raise` 8: no sized preflop raise below bucket 8 against a station) gained a lot against
the arena's stations: +15.99 (+13.2, +18.8) over 320 blocks, with busts by hand 50 falling
from 68% to 28%. Its luck-adjusted pilot on the station-heavy train field was −0.26
[−0.81, +0.29], with −0.27 [−0.50, −0.05] in levels 1–2. It was not taken to validation.

On a fresh seed, in chips against A on the same deals, it was −101 ± 79 after hand 50 and
−266 ± 191 after hand 100. Letting the preflop raises stand removed that cost: −4 ± 81 and
+14 ± 192. Station-heavy field, 1,600 blocks, raw:

| variant | B − A (95% CI) |
| --- | --- |
| first candidate | +1.01 (−0.14, +2.16) |
| **`pre_raise` 0 (this candidate)** | **+1.21 (+0.08, +2.34)** |
| no preflop rules | +1.31 (+0.17, +2.44) |
| depth gate 40 BB / 25 BB | −0.22 (−1.18, +0.73) / +0.34 (−0.50, +1.18) |

The jam rule is kept: against the arena stations it was +14.78 (+12.99, +16.57) with the rule
and +11.23 without it (800 blocks). On 800 blocks the population fields were −0.10
(−0.92, +0.72) (HandHQ) and +0.29 (−0.48, +1.07) (training rollouts).

**Pilot of this candidate.** Luck-adjusted, train, fresh seeds `station-pilot2-*`, 16 lineups
× 30 blocks, seat-keyed. The specs are committed with this document. Their hashes are
`14721af8…` (population) and `aa4a4431…` (station-heavy).

| field | B − A, luck-adjusted, 95% CI | adjusted block SD | variance factor |
| --- | --- | --- | --- |
| population | +0.10 [−0.10, +0.31] | 2.27 | 1.50 |
| station-heavy | +0.67 [+0.10, +1.25] | 6.38 | 1.58 |

The first-candidate pilots `station-pilot-*` (hashes `b84df26a…`, `8d8bab02…`) ran with A =
`bec8892d` and B = the first candidate `3ecddb2f…`, under the same artifact paths.

**Other checks.**
- In harness blocks where station mode never fired, B − A was exactly 0.00 in every block.
- Worker peak RSS: 91,643,904 bytes for B, against 91,492,352 for A.

**Sizing.** One-sided α = 0.025, Holm's first step:
- **H1.** 960 blocks give 0.98 power for non-inferiority at a true −0.2 (0.93 with the SD 20%
  higher).
- **H2.** 1,440 blocks give 0.98 power at the pilot's +0.67 and 0.66 at +0.4. The 80%-power
  MDE is 0.47, or 0.56 with the SD 20% higher.

## What is fixed before the runs

- **Arms.**
  - **A** is the shipped final bot, `artifacts/station/bundle-final`, bundle sha256
    `fd6b00cab4ccf309850a60725c0c5ab8bd5cf40cff7a95887beb5f3e0cf7cd46`. This is
    `strategy/final/submission.json`, and the default build of this commit is identical.
  - **B** is `python -m strategy.final.package --out artifacts/station/bundle-candidate
    --override antes=tables --override station=value`, bundle sha256
    `e19be8ff1beec6a34a30ec3c3d674b5de152f40004e4f8ac6213274f7760e8f1`, 10,155,546 bytes.
    It differs from A in `bot.py`, which builds `StationFinalBot(..., overrides={'antes':
    'tables', 'station': 'value'})`, and adds `strategy/exploit/station.py`.
- **Specs**, `strategy/exploit/specs/station-validation-{population,stations}.json`, with
  spec hashes `d42128ca…` and `cbcd9e62…`. Both are `confirmation` stage, HandHQ population
  source, **validation** partition, classic antes `0, 3, 5, 10, 15, 25, 40, 50`, 8 players,
  2,000 chips, 200-hand cap, and `"deal": "seat_keyed"`.
  - **Population:** 16 sampled `pop-v1+handhq-v1` lineups × 60 blocks (960).
  - **Station-heavy:** 16 explicit lineups × 90 blocks (1,440). Each lineup is the
    partition's three `loose_passive` members plus four members sampled from the partition
    (`sample_lineups`, 4 opponents, seed `station-validation-stations-fill-v1`).
  - `"variance_reduction": {"terms": ["allin", "flop_pot", "turn_pot", "river_pot"]}`
    (cross-fitted, primary).
  - Seeds `station-validation-population-v1` and `station-validation-stations-v1`. No run has
    used them.
- **Machine.** In-process, 7 jobs, numba installed (the luck terms).

## Primary analysis

Two hypotheses in mean terminal points per tournament, B − A:

| id | spec | hypothesis | blocks |
| --- | --- | --- | --- |
| H1 | `station-validation-population` | non-inferiority: H0 mean ≤ −0.5 | 16 × 60 |
| H2 | `station-validation-stations` | superiority: H0 mean ≤ 0 | 16 × 90 |

The estimate is the luck-adjusted one, and the test is a one-sided t on it (its SE and df). The
margin is 0.5 points (`strategy.exploit.evaluate.MARGIN`; an average seat scores 12.5). Holm
correction over H1 and H2, at family-wise α = 0.05:

```bash
uv run python -m strategy.exploit.evaluate run strategy/exploit/specs/station-validation-population.json --jobs 7
uv run python -m strategy.exploit.evaluate run strategy/exploit/specs/station-validation-stations.json --jobs 7
uv run python -m strategy.exploit.evaluate confirm --ni H1=<population run> --sup H2=<station-heavy run> \
    --margin 0.5 --out artifacts/station/confirm.json
```

The raw paired-t result is reported next to it, and is not used for the decision.

**Side conditions**, on B's bundle, run only if the statistics pass, on an otherwise idle
machine:

- **Identical play without a detected station.** `python -m strategy.final.equivalence --a
  artifacts/station/bundle-final --b artifacts/station/bundle-candidate --tables 40
  --until-note "st["`, once with no structure and once with `--structure
  strategy/antes/specs/structure-classic.json`. Every hero decision before B's first
  station-mode decision must be identical.
- **Deployed gate, once per structure.** `python -m strategy.policy.gate --bundle
  artifacts/station/bundle-candidate --decisions 50000`, once with no `--structure` and once
  with each of `--structure strategy/antes/specs/structure-{classic,big_blind}.json`. Every
  check must be ok.
- **Bundle limits.** `python -m strategy.final.package --verify --out
  artifacts/station/bundle-candidate` must be ok, with the bundle and one sandboxed worker's
  peak RSS under 100,000,000 bytes, and numba not loaded.

**Decision.** Adopt B only if H1 and H2 are **both** rejected and every side condition passes.

Adopting makes `station=value` part of the default build of `strategy.final.package`
(`DEFAULT_OVERRIDES`, with `antes=tables`). The default bundle is then B's, with B's hash. In
the same change, `strategy/final/submission.json` and its README record the new bundle and its
gate runs, and the drift test pins the new hash. Otherwise the default bundle stays
`fd6b00ca…` and the mode stays available behind `--override`.

**If Origin `main` moves before adoption.** The branch is rebased onto it, and A′ (the new
default build without `station`) must equal the new `submission.json`. The result above
carries over only if that change is recorded as play-identical to `fd6b00ca` on fresh
instances. B′ is then rebuilt on the new base, and every side condition is re-run on B′
against A′. Any other change to the base stops the adoption, and a new experiment is needed.

## Secondary (exploratory, no claims)

- raw paired-t estimates;
- per-lineup and per-family results, and results by level phase;
- the share of decisions in station mode;
- the luck-term diagnostics (term means, fitted slopes) and the variance factor on validation.

## Reporting

All results are reported with intervals, including nulls and failures. No spec is re-run with
new seeds to change a result. If a result prompts a change, the changed bot is a new
experiment with new specs.

## Results (added after the runs)

The runs were in-process with 7 jobs, from commit `8d5fdb0` (tag `station-prereg-v1`, pushed
before the first run, clean tree), with the pre-registered bundles, specs, seeds and block
counts unchanged. The spec hashes are `d42128ca…` (population) and `cbcd9e62…` (station-heavy).
The analysis is `python -m strategy.exploit.evaluate confirm` exactly as written above, saved
as `artifacts/station/confirm.json`.

| id | B − A, luck-adjusted, 95% CI | one-sided p | Holm p | raw paired t, 95% | outcome |
| --- | --- | --- | --- | --- | --- |
| H1 (population, NI −0.5) | −0.07 [−0.20, +0.07] | 2.0e-10 | 4.0e-10 | −0.22 [−0.41, −0.04] | rejected: B is non-inferior |
| H2 (station-heavy) | +0.85 [+0.54, +1.16] | 5.3e-8 | 5.3e-8 | +0.72 [+0.32, +1.11] | rejected: B is better |

**Side conditions** (run after the statistics passed, on an otherwise idle machine):

- **Identical play without a detected station:** 40 of 40 tables identical up to B's first
  station-mode decision, with no structure and with classic antes (6,216 hero decisions,
  0 fallbacks). Station mode fired on 16 tables in each run, and 31 tables were fully identical.
- **Deployed gate:** passed, every check ok, under every structure:

  | structure | decisions | worker p99 / p99.9 / max (µs) | timeouts (bot-caused) | control bot-caused |
  | --- | --- | --- | --- | --- |
  | none | 50,081 | 353 / 443 / 752 | 54 (5) | 3 / 11,384 |
  | classic | 50,003 | 390 / 489 / 794 | 44 (3) | 2 / 12,141 |
  | big blind | 50,194 | 388 / 469 / 686 | 26 (4) | 1 / 12,519 |

  No decision took 1 ms of worker time, and there were 0 restarts and 0 internal fallbacks.
  The PLO4 path of the same bundle (needed for the submission record) also passed: 50,095
  decisions, p99 90 µs, 0 bot-caused timeouts.
- **`package --verify`:** ok. 10,155,546 bytes; worker peak RSS 91,684,864 bytes (NLHE) and
  91,725,824 (PLO4); numba not loaded.

**Decision: pass.** H1 and H2 are both rejected and every side condition passed.
`station=value` is adopted: the default bundle of `strategy.final.package` is B's, `e19be8ff…`.
Origin `main` had not moved from `f192a61`, so A was still the submission at adoption.

Secondary, descriptive, not part of the test:

- **The population effect is small and probably slightly negative.** The raw paired estimate
  excludes zero: −0.22 [−0.41, −0.04]. By family, the luck-adjusted point estimates are mostly
  below zero. `smooth_switcher` is −0.53 [−0.89, −0.18] (Holm p 0.059), and `loose_passive`
  seated in the population lineups is −0.85 [−1.55, −0.14] (Holm p 0.29). No family is
  significant after Holm. By level: −0.04, −0.09 and −0.09.
- **Station-heavy by level:** levels 1–2 +0.19 [+0.06, +0.33], levels 3–5 +0.25 [−0.01, +0.51],
  levels 6–8 +0.27 [+0.06, +0.48].
- **Variance factor** of the luck terms with seat-keyed dealing: 1.90 (population) and 1.59
  (station-heavy).
