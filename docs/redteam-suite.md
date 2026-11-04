# Red-team suite

Three blind red-team bots (`redteam/pressure`, `redteam/equilibrium`, `redteam/adaptive`),
written by workers who never saw `strategy/`, are registered as the opponent family
`redteam-v1`, **validation only**, and a re-runnable suite plays a bot against them.
The pre-registration is `docs/redteam-preregistration.md` (machine-readable:
`strategy/redteam/preregistration.json`).

| file | role |
| --- | --- |
| `redteam/<bot>/` | the three bots, unchanged from their branches |
| `strategy/redteam/registry.py`, `redteam.lock.json` | the `redteam-v1` source: three members, content-hashed; `train` and `holdout` raise `PermissionError` |
| `strategy/redteam/preregistration.json` | lineups, block counts, seeds, hypotheses, pinned previous bots |
| `strategy/redteam/suite.py` | builds the 21 cells, plays every (cell, arm), analyses |
| `strategy/redteam/diagnostics.py` | per-table behaviour counts from the event log (overbets, shoves, phase and bubble equity, image phase) |
| `strategy/redteam/report.py` | `report.md` and `results.json` |

## One command against a new bundle

```bash
make redteam-suite BUNDLE=artifacts/final/bundle RT_OUT=artifacts/redteam/runs/new-bundle
# retrained blueprint: build the final bundle and the blueprint-only reference from the new packs, then run
make redteam-suite-pack PACK=path/to/blueprint-practice_nlhe.npz XPACK=path/to/exploit-practice_nlhe.npz RT_OUT=...
```

Equivalent: `uv run python -m strategy.redteam run --bundle DIR --out DIR` (or `--pack`/`--exploit-pack`).
The run resumes where it stopped (per cell and arm results are cached under `<out>/cells`
with a key of bundle hash, lineup, blocks and seed), then writes `<out>/report.md` and
`results.json`. `python -m strategy.redteam report --out DIR` re-renders it.

- The same seeds are used for every bundle, so two bundles are paired. A bot tuned on a
  run's results must be re-evaluated with a new `--seed`.
- `--reuse RUN_DIR` copies every (cell, arm) result whose key is identical (same bundle bytes,
  lineup, blocks and seed) from an earlier run, so a new bundle replays only its own arm
  (about a quarter of the cost when the references are unchanged).
- `--ref ARM=PATH` replaces a reference arm. With `--pack` the blueprint-only reference is
  rebuilt from the new pack automatically. The "previous bots" that sit in the lineups
  (blueprint-only, nlhe-v2, previous final) stay pinned by hash; `make redteam-previous`
  rebuilds them from tag `redteam-prereg-v1` if `artifacts/` was lost.
- `--only CELL`, `--arm ARM` and `--blocks N` restrict a run (smoke only; the pre-registered
  numbers need the defaults).
- `make redteam-gate` runs the deployed sandbox gate on each red-team bot. It needs an
  otherwise idle machine, and takes about 10 to 30 minutes per bot.

Cost: about one second per table single-threaded, 8 tables per block, 4 arms, 2,040
blocks per arm, so roughly two hours on 8 cores.

## Reading the report

- **Points** are official terminal points per tournament; par is 12.5. A cell is `LOSES`,
  `WINS` or `BREAK-EVEN` by the 95% interval of the bot's mean against 12.5; `*` marks a
  deviation that survives Holm across all 21 cells.
- **Pooled by set** uses the equal-weight mean of cells, so a set with more cells does not
  dominate it.
- **Attribution** rows give, per cell and arm: the hero's fold rate to red-team post-flop
  overbets against smaller bets, its chips per hand in overbet hands, its points-equity
  change over the bubble and in-the-money hands, the red-team preflop 3-bet/4-bet/shove
  rate by players alive, and (with an adaptive opponent) the hero's result in the opponent's
  tight against loose phase. Flags follow the rules in the pre-registration.
