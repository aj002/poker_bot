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
