# MCCFR blueprint (Stage 4 / M5)

An abstracted, point-aware external-sampling MCCFR blueprint for the official NLHE
format (8 seats, 2,000 chips, blinds 10/20 to 200/400, 25 hands per level, 200-hand
cap, 50/30/20 payouts), and a deployed bot that plays it. The measured results are in
the M5 report (`docs/mccfr-blueprint-report.md` in the project store); this file says
what the pieces are and how to rebuild them.

## Pieces

| Module | Role | numba |
|---|---|---|
| `abstraction.py` | Action set, bet sizes and the information-set key, plain Python shared by the trainer and the bot | no |
| `cards.py` | Card buckets per street (same code offline and at decision time) | no |
| `states.py` | Start-of-hand states harvested from training-partition rollouts | no |
| `deals.py` | Deal chunks (buckets, ranks, classes), a function of the seed only | no |
| `kernels.py` | The abstract hand simulator, ES-MCCFR walk, payoffs, traces | yes |
| `train.py` | Training driver: variants, discounting, pruning, checkpoints, resume, milestones | yes |
| `policy.py` | The compact policy pack (uint8 rows, four back-off levels) and its builder | no |
| `runtime.py` | `BlueprintBot`: push/fold leaf, then blueprint, then the base policy on a miss | no |
| `package.py` | Builds a self-contained submission bundle and checks its import closure | no |
| `exploit.py` | Heads-up preflop reduced games: exact best response, vector CFR+, sampled variants | yes |
| `response.py` | Approximate best responses to a pack in the full abstract game | yes |
| `audit.py` | Unabstracted boundary audit against exact point-aware push/fold solutions | no |

The deployed path (`runtime`, `policy`, `cards`, `abstraction`, the endgame leaf and
the features it reads) never imports numba, `strategy.solve`, `kernels` or `train`;
`package.py` refuses to build a bundle that does, and `tests/strategy/test_isolation.py`
confines numba to the offline modules.

## Abstraction

* **Actions:** fold, check/call, minimum raise, a pot-fraction raise (0.6 of pot
  preflop, 0.5 postflop), a pot-size raise, all-in. Sized raises only strictly
  between the minimum raise and all-in; from the third raise of a street only all-in.
* **Cards:** 15 preflop buckets (strength percentile of the 169 classes), 32 flop,
  32 turn and 16 river buckets (equity of the made-hand x draw x texture cell in 8
  bins, times a draw / hole-strength split). Equity only: no blockers.
* **Context:** players behind (6), active players (4), raises this street (4), price
  (6), depth (8: effective start stack in BB unraised preflop, bet level over
  effective start stack once raised, SPR postflop), acted (2), initiative or
  preflop commitment (2), ICM pressure at hand start (4): 73,728 contexts.
* **Keys:** 15 x 73,728 + (32 + 32 + 16) x 73,728 = 7,004,160 information sets,
  6 action slots each. Imperfect recall: betting history is summarised.

The preflop depth and commitment fields were chosen by exact exploitability in
heads-up preflop games (`exploit.py`); four candidate edge sets were tried.

## Training

External-sampling MCCFR in numba with dense float32 regret and strategy-sum arrays
(168 MB each), lock-free across threads. Each deal is paired with start states from
the training partition; every live seat traverses once. Utility is tournament
points: ICM of the survivors plus finishing prizes for players busted in the hand.
Variants: `vanilla`, `linear` (interval discounting), `dcfr` (1.5, 0, 2) and `cfr+`;
optional regret-based pruning. Checkpoints are written atomically every 15 minutes,
and a resumed run replays the same deal stream.

## Policy pack and bot

The pack keeps the average strategy as uint8 rows over four back-off levels (the
trained key, then pooling pressure, acted/initiative and depth), dropping rows with
little reach. Lookup is a binary search per level; the bot warms its caches at
load. Inside the push/fold range the point-aware endgame tables decide, as in the
Stage 3 bot.

## Rebuilding

```bash
make blueprint-states     # start-state pool (training partition)
make blueprint-train      # BP_HOURS=4; resumes when the run directory exists
make blueprint-pack       # final average strategy -> strategy/tables/data/blueprint-practice_nlhe.npz
make blueprint-bundles    # final, final-noleaf and milestone bundles under artifacts/blueprint/bundles
make blueprint-exploit    # heads-up exploitability by abstraction and variant
make blueprint-response   # approximate best responses to the milestone and final packs
make blueprint-audit      # unabstracted boundary audit of the final pack
make lab-blueprint        # paired points evaluations on the validation population
```

Training, `blueprint-exploit` and `blueprint-response` need `--group train`, which the
Makefile targets pass. Nothing here reads the holdout partition.
