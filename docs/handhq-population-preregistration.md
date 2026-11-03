# Pre-registration: exploit bot with HandHQ real-player classes vs the final bot

Written and committed before any validation run of the specs below. Everything that
produced the candidate was fitted and tuned on real HandHQ hands and on the train
partition of `pop-v1+handhq-v1` (`strategy/population/handhq.py`). The holdout partition is
not used, and no HandHQ family is registered in it.

## Question

Does adding two real-player classes (and their capped response experts) to the exploit
pack make the final bot score more tournament points against real-player-like opponents,
without losing meaningfully elsewhere?

## What is fixed before the runs

- **Arms.** A is the final bot, `python -m strategy.final.package --out
  artifacts/final/bundle`, bundle sha256 `8f4fd15ecb57875a6e5173307b91c6262ea590030cbd302f636243f38750ba03`.
  B is the candidate, `python -m strategy.final.package --out artifacts/handhq/bundle
  --exploit-pack artifacts/handhq/exploit/exploit-practice_nlhe.npz`, bundle sha256
  `bf31d67822fda2ebe7dfae8cfb4b2c0a9f02826e22a3afe4b987f42da1e1afb1`. B is A with one
  file changed: the exploit pack (data sha256 `a83ba9af5609f480`; A's is `3f74721a`), committed as
  `strategy/tables/data/handhq/exploit-practice_nlhe.npz` (file sha256 `db7b1a3cbdd35835`).
- **Real-player data.** HandHQ (phh-dataset commit `e47fbd58`), PokerStars July 2009
  six-max NLHE at 25, 50, 200, 400, 600 and 1000NL, seats with starting stacks of 20–150
  big blinds. `strategy/population/data/handhq-priors.json` holds the per-stake and
  per-type frequencies and the Dirichlet fits.
- **Opponent families.** `handhq_regular` and `handhq_recreational`
  (`strategy/population/families.py`), calibrated against the real regular and passive
  player types. They are registered in train (14 members each) and validation (3 each)
  only, by the `pop-v1+handhq-v1` extension, whose lock is
  `strategy/population/handhq.lock.json` (registry `e57b1eaeabcb`). pop-v1's members, lock
  and holdout are unchanged.
- **Candidate pack.** The committed pack plus classes `human_regular` and
  `human_recreational`. Their likelihood rows come from 1,200 training tables (one blueprint,
  four HandHQ training members, three pop-v1 training members), shrunk with the
  real-player empirical-Bayes pseudo-counts (11.8, 118.0, 28.8) and base measure. Each
  class has one expert, restriction p = 0.5, selected by the unchanged rule in
  `strategy/exploit/build.py`:

| expert | lam_max | gain vs own model | vs blueprint-like | risk per deviation |
| --- | --- | --- | --- | --- |
| human_regular | 0.5 | +0.1223 | −0.0406 | 0.0750 |
| human_recreational | 0.5 | +0.0945 | −0.0350 | 0.0712 |

  Belief settings, budget and every existing class row and expert are A's.
- **Tuning on train.** A single pooled `human` class was tried first. On training tables
  it was almost never the top posterior for HandHQ seats, so it was split into one class
  per family (`artifacts/handhq/diagnose-train.json`). Two train-partition discovery runs
  of `handhq-exploit-train` (3 explicit HandHQ training lineups × 20 blocks) gave
  B − A = +0.21 [−1.14, +1.57] with the pooled class, and +0.25 [−1.30, +1.80] with the
  split classes. Their per-block SD (about 6 points) set the block counts below. No
  validation-partition data was used.

## Primary analysis

Four co-primary hypotheses, each on B − A in mean terminal points per candidate seat.
Every spec uses 8-player NLHE tables with 2,000 chips, a 200-hand cap and every seat
rotation per block, in-process, 7 jobs:

| id | spec | opponents | hypothesis | blocks |
| --- | --- | --- | --- | --- |
| H1 | `handhq-exploit-targeted-validation` | HandHQ validation members: 6 regulars + 1 recreational; 4 + 3; 2 + 5; 2 + 2 + pop-v1 loose_passive, tight_surrender and wide_pressure | superiority: H0 mean ≤ 0 | 4 × 400 |
| H2 | `handhq-exploit-population-validation` | 16 sampled `pop-v1+handhq-v1` validation lineups | non-inferiority: H0 mean ≤ −0.5 | 16 × 50 |
| H3 | `handhq-exploit-robust-equilibrium` | the R1 lineups (blueprints, capped-leaf blueprints, endgame bots) | non-inferiority: H0 mean ≤ −0.5 | 3 × 100 |
| H4 | `handhq-exploit-robust-deceptive` | the R2 lineups (bait, image, flip, mix, arena switcher) | non-inferiority: H0 mean ≤ −0.5 | 6 × 50 |

The test is the one-sided paired t-test on block differences in `python -m
strategy.exploit.evaluate confirm --sup H1=… --ni H2=… --ni H3=… --ni H4=… --margin 0.5`,
with Holm correction at family-wise α = 0.05.

**Deployed gates** (on an otherwise idle machine, B's bundle):

- `python -m strategy.policy.gate --bundle artifacts/handhq/bundle --decisions 50000`:
  every check ok. That means worker decide p99 ≤ 400 µs and p99.9 ≤ 600 µs against the
  1 ms deadline, bot-caused timeouts ≤ 0.02%, and no restarts or internal fallbacks.
- `python -m strategy.final.package --verify --out artifacts/handhq/bundle`: ok. That
  means the bundle is under 100 MB, one sandboxed worker's peak RSS is under 100 MB, and
  numba is not loaded.

**Decision.** Adopt B only if H1 is a win, H2, H3 and H4 all pass, and both gates pass.
Otherwise the final bot keeps A's pack.

## Secondary (exploratory, no claims)

Per-lineup results, phase slices, the family slices of H2, the train-partition diagnose
labels, and the comparison of style bots with real-player frequencies
(`artifacts/handhq/census.json`).

## Reporting

All results are reported with intervals, including nulls and failures. No spec is re-run
with new seeds to change a result. If a result prompts a change, the changed bot is a new
experiment with new specs.
