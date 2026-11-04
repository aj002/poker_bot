# Pressure

`Pressure` is a self-contained NLHE tournament bot for the arena SDK. It is built
around smart, polarized aggression:

- large opens, high polarized 3-bet frequency, and decisive short-stack jams;
- large continuation bets, frequent barrels, and heads-up overbets;
- extra pressure when it covers medium stacks near the bubble or hand cap;
- a tournament-life risk premium when shorter stacks remain;
- confidence-weighted attacks on opponents whose public history shows excessive
  folding to reraises.

The bot uses only public observations, its per-seat policy seed, arena SDK/style
helpers, and `StatsTracker`. Its explicit JSON state is restorable and capped by the
SDK. It supports `practice_nlhe`.

Run a sandboxed match:

```bash
uv run python -m arena.cli match \
  --bot redteam/pressure/bot.py \
  --players 8 --tables 80 --jobs 1 \
  --opponents aggressive,super_aggressive,switcher
```

The policy intentionally calls infrequently. Its bluffs and value bets use similar
large sizes, but its aggression is reduced when ICM survival pressure is high.
