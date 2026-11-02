# Submission checklist (final bot, Stage 8 / M8)

The competition bundle is `artifacts/final/bundle`, built by `strategy.final.package`
(`strategy/final/README.md`). It has one entry point, `bot.py:make_bot`. On
`practice_nlhe` it plays the exploiting blueprint (`strategy.exploit`). On every other
ruleset, including a build-day `reveal_variant`, it plays the variant-agnostic fallback
(`strategy.policy.generic`). This list covers an NLHE submission. For a variant announced
on build day, follow [`docs/build-day-playbook.md`](build-day-playbook.md). Its selection
rule still applies, and this bundle is not a substitute for its gated fallback.

Everything here is a practice assumption. The arena is not the official simulator. Check
the official kit's interface, sandbox, memory rule and scoring against these steps before
trusting them.

## Before the day

- [ ] Check out the tagged commit `prereg-final-holdout`, or a descendant that changes
      nothing under the bundle's closure. Then run `uv sync --frozen`. numba is not
      needed, so leave out `--group train`.
- [ ] `make verify-fast` is green.
- [ ] Record the commit SHA. The bundle manifest does not embed it; the bundle hash
      identifies the bytes.

## Build and verify

- [ ] `make final-arms final-package`. The printed `bundle_sha256` must be the
      pre-registered one in `strategy/final/holdout-preregistration.json`. If it differs,
      the code or a pack changed, so this is not the bot that was evaluated. Stop.
- [ ] `make final-verify` must report `"ok": true`. It checks, in order:
  - every file hash and the bundle hash against `MANIFEST.json`;
  - a smoke load from a scratch directory with the repository hidden: the runtime, the
    generic fallback and both packs load from the bundle, the `practice_nlhe`,
    `uncertainty_shortdeck` and `uncertainty_plo4` rulesets are declared, and numba and
    llvmlite are absent;
  - artifact bytes at most 100 MB;
  - the peak resident memory of one sandboxed worker process playing a table, at most
    100 MB, on NLHE and on PLO4. The value is about 93 MB, of which about 38 MB is the
    interpreter and arena worker.
- [ ] `artifacts/final/submission/final-bot-<hash12>.tar.gz` and `SHA256SUMS` are written.
      The tarball is deterministic: sorted entries, zero mtimes and owners.

## Gate (on an otherwise idle machine)

- [ ] Close every other job, browser and indexer. The gate measures latency.
- [ ] `make gate-final` must pass. That means:
  - worker decide p99 ≤ 400 µs and p99.9 ≤ 600 µs;
  - bot-caused timeouts ≤ 0.02% and within the matched control;
  - 0 internal fallbacks and 0 restarts.

  The gate retries once on a host-invalid or stall-excess verdict. A bot-caused failure
  on an idle machine means the bundle is not safe to submit; fall back (see below).
- [ ] `make gate-final-generic` must pass. This is the PLO4 path of the same bundle.
- [ ] Optional: `make final-equivalence` replays the bundle and
      `artifacts/exploit/bundles/exploit` through the sandbox worker on 160 tables. It must
      report `identical: true`.

## Upload

- [ ] Upload the tarball. Keep `SHA256SUMS`, `MANIFEST.json`, `artifacts/gate/final.json`
      and the holdout evidence (`artifacts/final/holdout/`) as the dispute record.
- [ ] Rollbacks, in order:
  1. The blueprint bundle (`artifacts/blueprint/bundles/final`). Gate it first with
     `strategy.policy.gate --bundle artifacts/blueprint/bundles/final`.
  2. The fallback (`make package-fallback gate-fallback`), which is 68 KB and loads no
     numpy.

## What is not measured

- **The official sandbox.** The gate measures this repository's sandbox: a 1 ms
  parent-observed deadline, 1 GiB `RLIMIT_AS` and 64 KiB of bot state. The bot's
  explicit state is at most about 0.75 KB per table.
- **Memory.** The official 100 MB rule is read as an artifact limit (7.6 MB here). The
  runtime check above is a stricter local reading. It assumes one process per seat. Eight
  instances in one process use about 210 MB, because each instance holds its own copy of
  the packs.
- **Structure.** The blueprint and the exploitation layer were trained and evaluated on
  8-player, 2,000-chip tables with the practice blind schedule and a 200-hand cap. Other
  NLHE structures play legally but are not evaluated.
- **Opponents.** The holdout partition of the frozen population is the last unseen
  evidence. Real entrants are not that population.
