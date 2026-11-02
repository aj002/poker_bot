# League training (Stage 5 / M6)

A league of frozen checkpoints, an empirical opponent mixture solved over the payoff
matrix, a learner per iteration, and promotion only through paired real-engine gates on
the validation population. Pre-registration: `docs/league-training-preregistration.md`.

## Pieces

| module | role | numba |
| --- | --- | --- |
| `runner.py` | the league: config, checkpoint ledger, iterations, curve, selection, confirmations, export | lazy |
| `clone.py` | behaviour-cloned tables of the training population (and the CEM baseline) from real-engine play | no |
| `tables.py` | dense `uint8` policy tables; learner deviations; checkpoint packs as level-0 overrides of the parent pack | no |
| `kernels.py` | abstract mixture game: every seat its own table, one gated-response learner seat | yes |
| `learner.py` | `AbstractLeague`: seat mixtures, response training, frozen paired evaluation, payoff matrix | via kernels |
| `meta.py` | regret-matching meta-solver, NashConv, antisymmetric part, intransitive cycles | no |
| `gates.py` | promotion specs (`improve`, `regress-<j>`), the declared rule, Holm confirmations | no |
| `bots/league.py` | deployed bot: the blueprint runtime on `strategy/tables/data/league-practice_nlhe.npz` | no |

## One iteration

1. Start-of-hand states from training lineups with the incumbent seated.
2. Exploiter: a greedy gated response to the incumbent alone (approximate best
   response). Its held-out gain is the incumbent's exploitability proxy. It joins the pool
   only when that gain is significant.
3. Meta-solve: payoff matrix `G[a][b]` over the strategic pool on frozen held-out abstract
   deals. σ comes from regret matching with a 5% floor, and is checked with NashConv and exact
   mixed-seat values.
4. Learner: gated response to the seat mixture (45% population models, 2% uniform legal
   policy, the rest drawn from σ), starting from the incumbent's table. The gate z and blend
   come from a held-out abstract grid. A learner that does not beat the incumbent there is
   rejected without real-engine runs.
5. Gates: `improve` against the incumbent and `regress-<j>` against every earlier deployable
   checkpoint and the latest promoted exploiter, on official tournament points.

## Why response fine-tuning

Continued MCCFR against a fixed mixture would retrain 7M infosets for a small shift.
CEM on the interpretable layer cannot touch the blueprint. A sparse gated response
changes only the keys whose regret evidence clears z standard errors. It is cheap,
since 2M pairs take about 9 minutes on 6 threads, and it stays inside the deployed pack
format as level-0 overrides. The bot, latency and size are unchanged.

## Commands

```bash
uv run --group train python -m strategy.league.runner init    --run artifacts/league/main-v1
uv run --group train python -m strategy.league.runner iterate --run artifacts/league/main-v1
uv run --group train python -m strategy.league.runner curve   --run artifacts/league/main-v1
uv run --group train python -m strategy.league.runner select  --run artifacts/league/main-v1
uv run --group train python -m strategy.league.runner confirm --run artifacts/league/main-v1
uv run --group train python -m strategy.league.runner export  --run artifacts/league/main-v1
uv run --group train python -m strategy.league.runner status  --run artifacts/league/main-v1
```

`init --smoke` builds a tiny league for checking the plumbing. Every step writes its
result and is skipped when the result exists, so `iterate` resumes.
