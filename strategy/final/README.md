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

**Submission bundle.** `c3e0fa540ee27a66873c3e53608f2f0b8d1d9f10c1ddf6b3869b6c90aef09ab3`
(7,875,285 bytes; tarball `final-bot-c3e0fa540ee2.tar.gz`, sha256
`894d25c405f36d5649b8e4110d90d016c79601ade7eaaee86c5d94a380970710`; worker peak RSS 97.28 MB
NLHE, 97.15 MB PLO4). It is the previous final bot `8f4fd15e` with the HandHQ exploit pack
(`human_regular` and `human_recreational` classes and experts; see
`docs/handhq-population-preregistration.md`). Details are in `submission.json`.

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
make final-holdout final-holdout-confirm     # done once (see below); refuses to run again
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
is.** A deadline hybrid would make play depend on the clock, and a perfect switch would have
rescued none of the gate's bot-caused timeouts for certain (see the report).

## Holdout (run once, `docs/final-holdout-preregistration.md`)

B is the final bot. The values are mean terminal points per candidate seat (an average seat
scores 12.5). The p-values are one-sided and Holm-corrected over the four hypotheses.

| id | A | blocks | A | B | B − A | 95% CI | Holm p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H1 | blueprint, 16 sampled holdout lineups | 640 | 19.07 | 21.17 | **+2.10** | [+1.44, +2.75] | 8e-10 |
| H2 | baseline nlhe-v2 | 240 | 17.52 | 21.03 | **+3.51** | [+2.30, +4.71] | 3e-8 |
| H3 | endgame-capped bot | 240 | 14.82 | 22.68 | **+7.86** | [+6.64, +9.09] | < 1e-15 |
| H4 | blueprint, holdout-only families, non-inferiority at −0.5 | 600 | 20.98 | 21.55 | +0.57 | [+0.02, +1.12] | 8e-5 |

All four are rejected. Every arm had 0 errors, fallbacks, timeouts and illegal actions. The
evidence is `artifacts/final/holdout/` (the ledger and `confirm.json`).

## Submission bundle

Main was rebased onto later commits after the holdout (antes in the engine and rules
adapter, and the burst-aware gate classifier). The rebuilt bundle's bytes therefore differ
from the evaluated one's. `submission.json` records both hashes, the sandbox equivalence
between them on NLHE and PLO4, and every deployed-gate run of either bundle.

| gate run | bundle | classifier | verdict | decide p99 / p99.9 | bot-caused |
| --- | --- | --- | --- | --- | --- |
| 1–3, NLHE | evaluated | `cpu_bound` (ee67639) | fail (ceiling) | 351–368 / 435–507 µs | 30, 11, 23 per 50k |
| 4, NLHE | submission | burst-aware (main) | pass | 361 / 466 µs | 5 per 50k (0.010%), 0 cleared by host rules |
| PLO4 | both | both | pass | 87–106 / 132–165 µs | 0 |

The ceiling failures came mostly with harness parses at 3–23 times their median, which is
the host-burst pattern main's classifier now attributes to the host (see the report). The
decide p99 is the package's tightest margin against the 400 µs limit.
