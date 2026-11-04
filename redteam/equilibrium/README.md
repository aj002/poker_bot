# Red-Team Equilibrium Bot (`redteam_equilibrium`)

An independent, solver-derived, ICM-aware, and adaptive tournament poker bot built under the blind red-team rules using only the arena SDK (`arena.agents.sdk`, `arena.agents.styles.*`) and standard scientific Python (`numpy`).

## Design Overview

1. **Offline CFR+ Preflop & Push/Fold Equilibrium Tables (`precompute.py`, `tables.npz`, `tables_data.py`)**
   - **169×169 Hand-Class Equity & Card-Removal Matrix**: Computed by Monte Carlo rollouts over 2,000 random 5-card boards evaluating all 1,326 two-card combinations simultaneously and aggregating exact disjoint combo-pair showdowns across the 169 canonical Hold'em hand classes (`AA`, `AKs`, ..., `22`), paired with a postflop playability/realization adjustment vector.
   - **CFR+ Push/Fold Solver**: Solves two-player and late-position push/fold Nash equilibria across 10 effective stack depths (`2, 3, 4, 5, 6, 8, 10, 12, 15, 20` BB) and 5 ICM bubble-factor tiers (`1.0, 1.15, 1.35, 1.65, 2.1`) using Linear CFR+ averaging.
   - **CFR+ Multi-Player Open / Call / 3-Bet / 4-Bet Solver**: Solves the full preflop betting tree across all 7 active-players-behind counts (`n_behind` 1..7, corresponding to SB through UTG in 8-max) and 5 ICM bubble-factor tiers. Accounts for the compound probability that at least one of the `k` players behind wakes up with a flat or 3-bet (`1 - (1 - b_1)^k`), out-of-position equity realization shifts, and multiway squeeze hazards.
   - **Self-Contained Dual Packaging**: The solved tables are stored both in `tables.npz` (for directory staging: `--bot redteam/equilibrium`) and quantized/base85-embedded in `tables_data.py` (so single-file staging works seamlessly).

2. **Fast Malmuth-Harville ICM & Bubble Factor Engine**
   - Computes exact Malmuth-Harville expected tournament points via bitmask dynamic programming for $\le 4$ survivors and an exact $O(n^2)$ 3-prize summation for $5\text{--}8$ players (since top 3 places pay `50 / 30 / 20`), taking $< 8\,\mu\text{s}$ per decision.
   - Evaluates thehero's marginal chip gain (`+delta`) versus marginal chip loss (`-delta`) against the primary active opponent to obtain the exact pairwise Bubble Factor $\text{BF} = \Delta \text{EV}_{\text{loss}} / \Delta \text{EV}_{\text{gain}} \in [1.0, 2.1]$, weighted by proximity to the 200-hand cap.

3. **Balanced Mixed Postflop Engine**
   - Uses a zero-allocation bitmask 7-card evaluator (`_fast_nlhe_postflop`) that computes made-hand category, relative board texture (flush/straight completion, pairedness, wetness, high-card density), and exact flush/straight draw outs in $< 10\,\mu\text{s}$.
   - **When Checked To**: Chooses geometric bet sizes (`35%` pot on dry heads-up flops, `55–62%` on wet flops/turns, `68%` polar river value) and mixes value bets, range continuation bets, and semi-bluffs/blocker bluffs at the game-theoretic bluff-to-value ratio $\alpha = \text{bet} / (\text{pot} + \text{bet})$ scaled by $1 / (\text{BF} \cdot n_{\text{opp}}^{0.9})$.
   - **When Facing a Bet/Raise**: Applies ICM-adjusted pot odds and Minimum Defense Frequency (MDF) mixing around the indifference threshold, while narrowing opponent ranges across multi-street barrels and check-raises so one-pair hands are never stacked off against polarized turn/river raises.

4. **Compact Empirical-Bayes Opponent Adaptation**
   - Tracks a 15-integer per-seat summary in `ExplicitStateBot` state (`< 450` bytes total): hands dealt, VPIP, PFR, RFI opportunities/opens, 3-bets vs opens, fold-to-steal in the blinds, postflop aggression, postflop fold-to-bet, and an exponentially decayed recent-16-hand PFR window (to catch gear-switching opponents like `switcher`).
   - Shrinks noisy observations toward equilibrium priors (`N_prior = 12`) and shifts preflop steal/3-bet defense frequencies and postflop value-sizing/bluff frequencies within bounded safety rails.

5. **Multi-Variant Support (`uncertainty_shortdeck`, `uncertainty_plo4`, `reveal_variant`)**
   - Includes a fast bitmask evaluator for Short-Deck Hold'em, 4-card Pot-Limit Omaha (exact 2-from-hole + 3-from-board bitmask evaluation), and arbitrary `reveal_variant` configurations.

## Running in the Sandbox

```bash
# 8-player sandboxed match against the built-in style opponents
uv run python -m arena.cli match --bot redteam/equilibrium \
    --opponents rock,aggressive,super_aggressive,station,weak_tight,switcher,nlhe_heuristic \
    --players 8 --tables 80 --seed eval1

# Regenerate precomputed CFR+ tables from scratch (~35s)
uv run python redteam/equilibrium/precompute.py
uv run python redteam/equilibrium/export_tables.py
```

## Sandbox Timing, Memory, and Legality Summary

- **Decision latency (`worker decide() CPU`)**: `p50 = 91 µs`, `p95 = 128 µs`, `p99 = 137 µs`, `max = 177 µs` (over 11,583 sandboxed decisions; >7× headroom below the 1,000 µs limit).
- **End-to-end parent round trip**: `p50 = 273 µs`, `p95 = 338 µs`, `p99 = 532 µs`.
- **Memory footprint**: Worker peak RSS is `~56 MB` (well below the `100 MB` limit); explicit JSON state is `< 450 bytes` (`64 KiB` limit).
- **Legality**: `0` illegal actions, `0` schema errors, `0` bot exceptions, `0` worker crashes across >50,000 decisions across all rulesets.
