# Pre-registration: the Stage 5 checkpoint league (`main-v1`)

Written and committed before the production league run `artifacts/league/main-v1` and
before any of its gates or confirmations ran. Every real-engine run below uses the
validation partition of `strategy.population` (pop-v1) on fresh seeds; the holdout partition
is not used. Training and the pilot use only the train partition.

## What was fixed by the pilot (train partition only)

A pilot league (`artifacts/league/pilot`, the same init artifacts) ran one iteration up to
the abstract screen and then tested five learner gate/blend settings with the real engine
on the **train** partition: candidate against the blueprint in train-population lineups
(8 lineups × 6 blocks) and in blueprint-rich lineups (three blueprint seats, 8 × 4).

| learner (z, mix) | deviations | abstract mixture gain | train pop. B − A | blueprint-rich B − A |
| --- | --- | --- | --- | --- |
| 2.0, 0.5 | 75,253 | +0.028 | +1.82 [−1.13, +4.78] | +1.70 [−2.42, +5.81] |
| 2.0, 1.0 | 75,253 | +0.066 | **+9.87 [+6.32, +13.42]** | +2.68 [−1.46, +6.82] |
| 2.5, 0.5 | 6,993 | +0.005 | +0.53 [−1.93, +2.99] | +1.25 [−1.33, +3.83] |
| 2.5, 1.0 | 6,993 | +0.009 | −1.63 [−4.58, +1.32] | +3.25 [−0.56, +7.06] |
| 3.0, 1.0 | 2,469 | +0.003 | +1.06 [−1.49, +3.60] | −0.94 [−3.35, +1.47] |

The abstract gains are points per player-hand on held-out abstract deals. The real
differences are points per tournament (95% paired t). Two decisions follow. The learner grid
is z ∈ {2.0, 2.5, 3.0} × mix ∈ {0.5, 1.0}. The abstract tolerance, the loss a learner may
take in any one pool member's homogeneous world, is 0.15 points per player-hand: the z = 2
learner lost 0.12 to the blueprint in an all-blueprint abstract world but did not regress
in real blueprint-rich lineups. The block standard deviation was 7–12 points.

## The league config (frozen at `init`)

`strategy/league/runner.py:Config` and `strategy/league/gates.py:Rule` defaults at this
commit: 4 iterations; 2M learner pairs and 1M exploiter pairs; exploiter gate z = 3;
seat mixture 45% training-population models, 2% uniform legal policy, the rest drawn
from the meta-solved mixture σ (regret matching, 5% uniform floor).

## Promotion rule (each iteration, discovery stage)

A candidate is promoted only if all of these hold:

- `improve`: candidate (B) against the incumbent (A), 24 validation lineups × 12 blocks.
  The 95% paired-t lower bound is above 0, the worst validation family's mean is above
  −5, and the execution gates pass (no illegal actions, errors, fallbacks or failed tables;
  bundle under 100 MB).
- `regress-<j>` for every earlier deployable pool member `j` (anchors and promoted league
  checkpoints) and the latest promoted exploiter: candidate (B) against `j` (A) in
  `j`-rich lineups (`j` in three opponent seats, validation members in four), 12 × 8
  blocks. Each point estimate is above −2, each lower bound above −5, and the
  inverse-variance pooled lower bound across all `regress-*` gates is above −1.5.

A candidate that fails is retired and the incumbent stays. No gate spec is edited or
re-run with new seeds; specs are pinned by their manifests.

## Selection (declared before the run)

`python -m strategy.league.runner curve` runs every promoted league checkpoint against the
blueprint (8 validation lineups × 12 blocks). `select` takes, among promoted league
checkpoints whose abstract antisymmetric advantage over every other strategic pool member
is at least −0.15, the one with the highest curve difference. If none is promoted, the
selection is the blueprint and the confirmations below are not run (a null result).

## Confirmations (co-primary, Holm, one-sided α = 0.05)

For the selected checkpoint S, with fresh seeds (`league-main-v1-confirm-*`):

| id | comparison | lineups × blocks | H1 |
| --- | --- | --- | --- |
| `vs-blueprint` | S (B) vs blueprint (A), validation population | 40 × 10 = 400 | B − A > 0 |
| `ni-<j>` | S (B) vs `j` (A) in `j`-rich lineups, for every other deployable pool member `j` (blueprint, baseline, earlier promoted checkpoints) | 25 × 8 = 200 | B − A > −1.0 |

Each test is a one-sided paired t on block differences
(`strategy.league.gates.holm_family`). The family's p-values are Holm-corrected together. A
hypothesis is confirmed only if its Holm-adjusted p < 0.05. Every result is reported,
including nulls, with intervals.

## Deployment gate

The selected pack is exported to `strategy/tables/data/league-practice_nlhe.npz` and played by
`strategy/league/bots/league.py` (the blueprint runtime, no numba). Its bundle must pass
`python -m strategy.policy.gate` (50,000 sandboxed decisions, worker p99 below 500 µs, no
bot-caused timeouts or fallbacks) and stay under 100 MB.

## Amendment 1 (written after selection, before any confirmation run)

The declared selection chose the **blueprint**. C1, the only promoted league checkpoint,
was ineligible: its abstract antisymmetric advantage over the exploiter E2 is −0.350
(tolerance −0.15). So the pre-registered confirmations above were not run, a null result
that stands. Its learning-curve score against the blueprint was +11.15 [+8.86, +13.43].

This amendment runs the same confirmation family for C1 (`confirm --name C1`):
`vs-blueprint` (400 blocks, H1 > 0), `ni-blueprint` and `ni-baseline` (200 blocks each,
H1 > −1.0). The family is Holm-corrected together, with one-sided α = 0.05. These are
the spec ids declared above (`league-main-v1-confirm-*`), and they had never run. The
results are reported as an amendment family. They do not change the declared selection:
the default deployed bot stays the blueprint. C1 ships only as the optional league bot
(`strategy/league/bots/league.py`), which must pass the deployment gate.

A secondary real-engine exploitability check, added after iteration 2 and not
pre-registered (`artifacts/league/main-v1/exploitability.py`), runs each exploiter
against its target in target-rich lineups: E1 vs the blueprint and E2 vs C1, 96 blocks
each.

## Secondary (descriptive, no claims)

Payoff matrices and σ per iteration, NashConv in the linearised meta-game and the exact
mixed-seat check, exploiter gains (the exploitability proxy), intransitive cycles, family
slices and fifth-percentile lineups of the gates.

## Results (added after the runs)

The pre-registration commit is tag `prereg-league-main-v1` (`62dd7fb`) and amendment 1 is
tag `prereg-league-main-v1-amendment-1` (`dac4667`). Both were pushed before the runs they
govern. Rebasing onto main turned them into `cbcc009` and `e5fdcb7`, with the same code
and text.

| iteration | candidate | improve (B − A vs incumbent) | decision |
| --- | --- | --- | --- |
| 1 | C1 | +9.12 [+7.81, +10.44] | promoted |
| 2 | C2 | −0.37 [−1.52, +0.77] | retired (also regress-E2 −4.65) |
| 3 | C3 | +0.44 [−0.77, +1.65] | retired (also regress-E2 −2.67) |
| 4 | C4 | −0.82 [−1.90, +0.26] | retired (also regress-E2 −5.06) |

Declared selection: **blueprint**. C1 was ineligible, with an antisymmetric advantage of
−0.350 over E2. The pre-registered confirmations were not run (null).

Amendment 1, C1's family (Holm, one-sided α = 0.05):

| id | blocks | B − A | 95% paired t | Holm p | outcome |
| --- | --- | --- | --- | --- | --- |
| `vs-blueprint` | 400 | +10.38 | [+9.21, +11.55] | < 1e-15 | confirmed |
| `ni-blueprint` (margin −1) | 200 | +3.40 | [+2.04, +4.76] | 6.3e-10 | confirmed |
| `ni-baseline` (margin −1) | 200 | +16.37 | [+14.96, +17.77] | < 1e-15 | confirmed |

Secondary exploitability, real engine (exploiter vs target in target-rich lineups, 96 blocks):

- E1 vs the blueprint: +0.68 [−1.22, +2.59];
- E2 vs C1: +2.35 [+0.30, +4.40].

Deployment gate for the league bot: passed on attempt 1. 50,197 decisions, worker p99
229 µs, 2 bot-caused timeouts within the control's rate, 6.4 MB bundle.

## Amendment 2: blueprint–C1 blend frontier (written before any blend run)

**Question.** Is there a blend of C1 and the blueprint that keeps a real population gain and
passes the selection's robust-score clause that C1 failed?

**Family.** `B_λ` is the blueprint pack with level-0 rows at C1's 75,407 override keys
set to `(1 − λ) · blueprint row + λ · C1 row` (`strategy.league.blend`). Every key has a
blueprint row, so no key is a miss. The grid is λ ∈ {0.25, 0.5, 0.7, 0.85, 0.95}. λ = 0 is
the blueprint, with exploiter E1, and λ = 1 is C1, with exploiter E2.

**Train-only measurements, per λ:**

1. On held-out abstract deals (train-derived states and clones):
   - the population gain of `B_λ` over the blueprint;
   - an exploiter `X_λ`, a gated best response to `B_λ` alone (the league exploiter
     config: 1M pairs, z 3, seed 3000 + 100λ);
   - the antisymmetric advantage `H(B_λ, j)` for `j` ∈ {blueprint, baseline, C1, E2, X_λ}.
2. With the real engine on the **train** partition:
   - `B_λ` vs the blueprint, 12 lineups × 8 blocks;
   - `X_λ` vs `B_λ` in `B_λ`-rich lineups (three `B_λ` seats), 8 × 6 blocks.

**Choosing λ (declared now).**

- **Eligible:** `H(B_λ, j) ≥ −0.15` for every `j` above. `X_λ` counts whether or not its gain
  is significant, which is stricter than the selection clause.
- **Choice:** among eligible λ in (0, 1), the one with the largest real-engine train
  population gain.
- **Stopping rule:** if none is eligible, or the chosen λ's train lower bound is ≤ 0, the
  answer is a null and no validation run is made.

The chosen λ is committed before any validation run.

**Validation, as originally declared (incumbent: the blueprint, the declared selection):**

- `improve` vs the blueprint, 24 × 12.
- `regress-j` for `j` ∈ {blueprint, baseline, C1, E2, X_λ}, 12 × 8 each. Each point
  estimate > −2, each lower bound > −5, and the pooled lower bound > −1.5.
- `X_λ` enters the strategic pool only if its abstract gain is significant (the league's
  exploiter rule).
- If promoted: `curve`, then `select` (the same −0.15 clause over the strategic pool).
  If selected: the confirmation family with fresh ids `league-main-v1-confirm-<B>-*`:
  `vs-blueprint` (400 blocks, > 0) and `ni-j` for blueprint, baseline and C1 (200 blocks
  each, > −1), Holm-corrected at one-sided α = 0.05.
- Descriptive: `X_λ` vs `B_λ` in `B_λ`-rich validation lineups, 12 × 8.

**The blend replaces the blueprint as the default bot only if** it is promoted, selected,
confirmed on every member of its family, and passes the deployment gate. Otherwise the
blueprint stays the default and the result is reported as a null.
