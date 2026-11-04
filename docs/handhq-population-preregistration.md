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

## Results (added after the runs)

| id | spec | B − A (points) | 95% paired t | one-sided p | Holm p | outcome |
| --- | --- | --- | --- | --- | --- | --- |
| H1 | `handhq-exploit-targeted-validation` | +0.476 | [+0.181, +0.771] | 7.9e-4 | 1.6e-3 | win |
| H2 | `handhq-exploit-population-validation` | +0.104 | [−0.273, +0.481] | 8.7e-4 (vs −0.5) | 1.6e-3 | non-inferior |
| H3 | `handhq-exploit-robust-equilibrium` | +0.300 | [−0.004, +0.603] | 2.0e-7 (vs −0.5) | 8.0e-7 | non-inferior |
| H4 | `handhq-exploit-robust-deceptive` | +0.247 | [−0.179, +0.673] | 3.2e-4 (vs −0.5) | 9.7e-4 | non-inferior |

Per lineup (exploratory, unadjusted):

- H1, in the lineup order above: +0.31 [−0.27, +0.90], +0.55 [−0.01, +1.11],
  +0.47 [−0.13, +1.06] and +0.57 [−0.04, +1.19].
- H3: +0.77 [+0.16, +1.37] in the mixed blueprint/endgame lineup.
- H4: +1.75 [+0.91, +2.59] in the `image` lineup, and −0.75 [−2.35, +0.85] in the mixed
  lineup.
- In H1, the gain is concentrated in blind levels 3–5: +0.39 [+0.22, +0.56].

**Gates** (idle machine, B's bundle):

- `strategy.policy.gate` passed every check on 50,024 decisions. Worker decide p99 was
  295 µs, p99.9 321 µs and the maximum 396 µs, with no decision over 1 ms. There were no
  timeouts (control: 0), no restarts and no internal fallbacks. End-to-end p99 was 509 µs.
- `strategy.final.package --verify` was ok. The bundle is 7,874,851 bytes, and one
  sandboxed worker's peak RSS is 97.05 MB on NLHE and 97.12 MB on PLO4, against a
  100 MiB limit (A, measured the same day: 93.54 MB). numba was not loaded.

**Decision.** Every pre-registered criterion holds, so the candidate exploit pack is
adopted. As instructed, the final bot is not rebuilt here: the committed default pack is
unchanged, and the candidate is `strategy/tables/data/handhq/exploit-practice_nlhe.npz`,
for combination with the blueprint work. The experts were trained against the current
blueprint pack and states. If the blueprint changes, retrain both HandHQ experts and the
existing ones with the commands in the HandHQ report, and recheck the RSS headroom (about
7.8 MB under 100 MiB).

Runs were in-process with 7 jobs, from commit `e43fb74` (tag `handhq-prereg-v1`), with the
pre-registered bundles unchanged.
