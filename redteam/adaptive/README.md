# adaptive (red-team bot)

`bot.py` is a self-contained `ExplicitStateBot` (stdlib + `arena.agents.sdk` only) for `practice_nlhe`.

Design:
- **Opponent model.** Each decision it reads `action_history` and keeps per-seat counters in explicit JSON state (~400 bytes): VPIP, PFR, RFI, fold-to-raise preflop, postflop bet/call (aggression) and fold-to-bet. Rates are shrunk to priors, so reads move after about a dozen hands.
- **Exploits.** Steal range widens when the players behind fold a lot; it 3-bets light against loose openers who fold to 3-bets; it tightens against nits. It c-bets and bluffs in proportion to the opponents' fold-to-bet rate, stops bluffing stations and thin-value-bets them, and calls bettors who show high aggression lighter than nits. Bet-size pressure adds a margin to calls.
- **Image switching.** A tight/loose phase flips every 18 hands (seeded phase offset), shifting open and defend thresholds and bluff frequency.
- **ICM / points.** With payouts 50/30/20 it tightens by stack count (most near the bubble), switches to push/fold under ~11 bb and shove-or-fold under ~14 bb when facing raises, and prices big calls with a Malmuth-Harville ICM equity (about 45 µs per call) blended into the pot-odds requirement.
- **Hand strength.** Chen score preflop; postflop a made-hand/draw heuristic with no sampling, so `decide()` is about 90 µs median.

Run: `uv run python -m arena.cli match --bot redteam/adaptive/bot.py --players 8 --opponents rock,aggressive,super_aggressive,station,weak_tight,switcher,nlhe_heuristic`
