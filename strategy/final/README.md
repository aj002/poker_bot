# Final competition bot (Stage 8 / M8)

One entry point, `bot.py:make_bot` in the bundle (`runtime.FinalBot`):

* `practice_nlhe` is played by the exploiting blueprint (`strategy.exploit.runtime.ExploitBot`):
  the MCCFR blueprint, its point-aware push/fold leaf and the confidence-capped exploitation
  layer. Its actions, notes and explicit state are the same as the exploit bundle's.
* Every other ruleset is played by the variant-agnostic fallback
  (`strategy.policy.generic.GenericBot`): short deck, PLO4, and a build-day `reveal_variant`
  bundled with `--reveal`.
* If the NLHE path raises, the bot plays the constant-time legal fallback and counts it in
  `state["fb"]`. The deployed gate requires that count to be 0.

The bot never reads the clock, so the same seeds give the same actions.

The report is in `docs/final-bot-report.md` in the agent store.
The holdout pre-registration is in `docs/final-holdout-preregistration.md`, and the
submission procedure is in `docs/submission-checklist.md`.

## Modules

| module | role |
| --- | --- |
| `runtime` | `FinalBot`: ruleset dispatch, the guarded teacher and the generic fallback |
| `bots/final.py` | the repository bot (in-process arenas and experiments) |
| `package` | the deterministic, hashed bundle; `--verify` checks hashes, runs a smoke load with the repository hidden and checks the size and worker-memory limits; `--submit DIR` writes the tarball |
| `equivalence` | two bundles through the sandbox worker on the same tables: identical play or not |
| `distill` | the distillation study: teacher rollouts, the table student, agreement, and the hybrid counterfactual |
| `holdout` | the one-shot holdout runner: refuses a changed bundle or a comparison already run, and writes the Holm confirmation |

Only `runtime` and the modules it imports are shipped. The probe that records the closure
refuses numba, llvmlite, the solvers, the exploit trainers and the four tool modules
above.

```bash
make final-arms final-package final-verify   # comparison arms, the bundle, the limits and the tarball
make final-equivalence                       # sandboxed identity against the exploit bundle (160 tables)
make gate-final gate-final-generic           # the deployed gate, NLHE and PLO4 (idle machine)
make distill-collect distill-student distill-agree distill-validate gate-student   # the distillation study
make final-holdout final-holdout-confirm     # once, after the pre-registration tag is pushed
```

## Distillation study (validation partition)

The question was whether a belief-free student could replace the teacher at lower latency.
The student is a table. Each blueprint key where the teacher ever blended on 4,000
train-partition tables (13,007 keys) gets the teacher's average distribution over its visits
there, shrunk toward the blueprint row by one visit. Rows are quantised to uint8 and keep
every positive action. The table is inserted at back-off level 0 of the blueprint pack, so
the student runs on the plain blueprint path with 43 bytes of state.

| measure (validation) | student | blueprint |
| --- | --- | --- |
| total variation to the teacher, decisions where the teacher blended (13%) | 0.270 | 0.366 |
| argmax agreement there | 69.0% | 63.2% |
| total variation to the teacher, other decisions | 0.033 | 0 |
| points vs the teacher, 16 sampled lineups, 400 blocks | +0.38 [−0.52, +1.29] | |
| points vs the teacher, blueprint-like lineups (R1), 150 blocks | **−1.68 [−2.87, −0.50]** | |

The student holds its own against the exploitable population but loses to the teacher
where the teacher's confidence gate keeps it on the blueprint. **The teacher is shipped as
is.** It passes the deployed gate, and a deadline hybrid would make play depend on the
clock (see the report).
