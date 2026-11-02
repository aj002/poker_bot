# Build-day playbook

At the competition the variant may be announced only on build day, with about 60
minutes to build. This playbook turns the announcement into a gated submission with
`strategy.rehearsal` (Stage 9 of the self-play learning plan). The rule is: **submit a
legal, gated fallback first, then improve it only when validation and the gate say so.**
Deeper training never blocks the first accepted bot.

Everything here is a practice assumption. The arena is not the official simulator. The
official kit's interface, sandbox and scoring must be checked against these steps
before they are trusted (see "Open organiser questions" below).

## What exists before the day

| Piece | Where | Variant coverage |
|---|---|---|
| Reveal slot in the engine | `arena/rules/reveal.py`, ruleset id `reveal_variant` | Any variant made of PokerKit parts (see "What the slot can describe") |
| Reveal-aware rules layer | `strategy/rules/reveal.py` (`get_variant`, `reveal_dir`) | Same |
| Card layer | `strategy/features/cards.py` | Any category order, 52- or 36-card deck, wheel-style lowest straight, 2–5 hole cards |
| Variant-agnostic fallback | `strategy/policy/generic.py` (`GenericBot`) | Anything the slot can describe; card-blind if the card layer cannot read it |
| Tuned baseline | `strategy/policy/runtime.py` plus `strategy/policy/vectors/nlhe-v2.json` | Anything the card layer reads |
| Equity tables | `strategy/tables/postflop.py`, `preflop.py` | Postflop always. Preflop when C(deck, hole) ≤ 300,000 |
| Push/fold tables | `strategy/tables/endgame.py` | Only 2-hole, best-five, no-limit, hold'em category priority (not regenerated on the day) |
| Pipeline | `strategy/rehearsal/runner.py`, `python -m strategy.rehearsal` | — |
| Rehearsal specs | `strategy/rehearsal/heldback/*.json` | PLO5 8-max, short-deck trips-over-straight 6-max turbo, NLHE with antes (must be rejected) |

### What the slot can describe

- Ranks `23456789TJQKA` (52 cards) or `6789TJQKA` (36 cards), suits `cdhs`.
- Hold'em streets: flop 3, turn 1, river 1, each with a burn.
- 2 to 5 hole cards.
- Best five of all cards, or exactly two hole cards plus three board cards.
- Any order of the nine five-card categories.
- The lowest straight is either the wheel (ace low) or the five lowest ranks.
- No-limit or pot-limit betting.
- Any structure `TournamentConfig` accepts: 2–8 seats, any stack, any increasing blind
  schedule, any hand cap and any non-increasing point vector.

The slot **cannot** describe these, and the pipeline rejects them at its `load` step:

- antes
- 9 or more seats
- draw, stud and high-low games
- fixed-limit betting
- other decks or street layouts

That leaves no safe automatic path. Escalate at once (see "When the variant is out of scope").

## The operator's checklist

Print this. Tick each line with the time.

**Before the day (T−1)**

- [ ] `git pull` on the build machine. Then run `uv sync --frozen` (add `--group train` only for push/fold work).
- [ ] `make verify-fast` is green.
- [ ] `make rehearse-heldback REHEARSAL_BUDGET=smoke` passes on the build machine.
- [ ] Record the commit SHA. Every bundle manifest records it too.
- [ ] Machine is idle: no other jobs, browsers or indexers. The deployed gate measures latency.
- [ ] No network or live AI is needed after this point. The pipeline is offline.

**T+0: announcement**

- [ ] Write the announcement into a spec file. Copy the closest `strategy/rehearsal/heldback/*.json`
      and edit `variant` (VariantSpec fields) and `structure` (TournamentConfig fields).
      Two people read the announcement against the file. Check the category order, the lowest
      straight, hole cards, hand selection, betting, the blind schedule and the points.
- [ ] Start the run: `make rehearse REVEAL_SPEC=<file>` (budget `fast`). It logs every step with a
      timestamp. The report lands in `artifacts/rehearsal/<name>-fast/report.md` and is
      rewritten after every step.

**T+1 min: load and conformance**

- [ ] `load` is `ok`. A `fatal` here means the engine cannot play the variant; see below.
- [ ] `conformance` shows metadata, ranking, differential, scoring, evaluator and smoke as `ok`.
      Fixtures show `na`, because a new variant has no fixtures. `evaluator` must show 0
      disagreements. Otherwise the card layer misreads hands and only the fallback's card-blind
      mode is trustworthy.

**About T+1 to T+20 min: first window**

- [ ] `fallback` writes `bots/fallback` within about 20 s.
- [ ] `gate_fallback` runs 50,000 sandboxed decisions against a workload-matched control
      (11–13 minutes active per attempt in the rehearsals; short deck needed a second attempt
      after a stall-excess verdict). It must pass.
- [ ] `submit_early` writes `submission/early/<name>-fallback-<hash>.tar.gz` and `SHA256SUMS`.
      **Upload it now.** It is the rollback for the rest of the day.

**About T+20 to T+45 min: improvement path**

- [ ] `tables` writes `reveal/tables/` (postflop always, preflop when feasible) and its
      `variant_hash` marker. It also records why push/fold was not regenerated.
- [ ] `search` runs 6 CEM generations on the train partition from the NLHE vector (4.5 minutes
      for short deck 6-max, 10.5 for PLO5 8-max). Check that `fallbacks` is 0.
- [ ] `validation` runs three paired in-process A/B experiments on the validation partition
      (3–4 minutes; two when the search kept the start vector). Read the "Validation" table
      in `report.md`.
- [ ] `select` applies the pre-declared rule (below). Do not override it by eye.

**About T+45 to T+60 min: final window**

- [ ] `gate` runs the deployed gate for the selected bundle if it is not the fallback
      (7–10 minutes active).
- [ ] `limits` checks the size (≤ 100 MB), the resident memory the bot adds, and that no
      numpy or numba is loaded. It also checks the bot was imported from its own bundle.
- [ ] `submit` writes `submission/final/` with the selected bundle and the fallback as its
      rollback. **Upload** the selected tarball when it differs from the early one. Keep both
      tarballs and `submission.json`.
- [ ] Archive `artifacts/rehearsal/<name>-fast/` (reports, gate JSON, experiment manifests,
      search logs). This is the evidence bundle for any dispute.

**Deadline pressure.** If the clock runs out after `submit_early`, stop. The early
window is a gated, legal submission. If `search` is the bottleneck, rerun with
`--budget smoke` into a new `--out` directory. It is not a substitute for the fast
budget: its validation has a single block and no confidence interval, so `select`
falls back to the fallback.

## Selection rule (declared before any result)

Candidates, in order:

1. **tuned**, the CEM incumbent.
2. **transferred**, the NLHE vector, untuned.
3. **fallback**.

- Tuned is selected when two things hold on validation. Its paired 95% lower bound
  against the fallback is above 0, and its mean difference against transferred is at
  least 0.
- Otherwise transferred is selected, when its 95% lower bound against the fallback is
  above 0.
- Otherwise the fallback is selected.
- A selected bundle that fails the deployed gate drops to the next eligible candidate.
- The fallback must pass the gate, or there is no submission.

The rule is `SELECTION_RULE` in `strategy/rehearsal/runner.py` and is printed in every
report.

## Running it by hand

```bash
uv run python -m strategy.rehearsal list                        # the held-back announcements
uv run python -m strategy.rehearsal run SPEC.json --budget fast  # the whole day; --out DIR, --jobs N
uv run python -m strategy.rehearsal run SPEC.json --budget smoke --no-gate   # a 15 s dry run

# Individual pieces, for a manual recovery:
export ARENA_REVEAL_VARIANT=artifacts/rehearsal/NAME-fast/reveal   # variant.json (+ tables/)
uv run python -m strategy.policy.package --kind fallback --reveal $ARENA_REVEAL_VARIANT --out artifacts/bots/fb
uv run python -m strategy.policy.gate --bundle artifacts/bots/fb --ruleset reveal_variant \
    --structure structure.json --decisions 50000 --cost-tables 0 --out artifacts/gate/fb.json
uv run python -m strategy.policy.package --params INCUMBENT.json --reveal $ARENA_REVEAL_VARIANT --out artifacts/bots/x
```

`structure.json` holds the `TournamentConfig` fields of the announcement. These are
`initial_player_count`, `starting_stack`, `hand_cap`, `hands_per_level`,
`blind_levels` and `payout_points`.

A packaged bot never reads the environment. The sandbox gives a bot only `PATH`,
`LANG` and `PYTHONHASHSEED`, and file reads only inside its bundle. So the package step
copies the variant into the bundle at `strategy/reveal/variant.json`, with its tables
under `strategy/reveal/tables/`. The arena and the offline tools read
`$ARENA_REVEAL_VARIANT`. Reveal tables carry a `variant_hash` marker, and tables for
another variant raise instead of loading.

## When something fails

| Symptom | Meaning | Action |
|---|---|---|
| `load` fatal: `UnsupportedRules` or `ValueError: ante` | The engine cannot express the variant or structure | Out of scope; see the next section |
| `conformance` evaluator disagreements > 0 | The card layer misranks hands | Submit the early fallback only. The baseline's strength features are wrong |
| `conformance` smoke fatal | The engine cannot finish a tournament | Stop and debug the reveal profile. Nothing downstream is meaningful |
| `gate_fallback` fail | The fallback misses the deadline on this machine | Read `gates/fallback.json` `timeout_classes`. If timeouts are stall-like or host-invalid, rerun on an idle machine. If bot-caused, there is no safe submission: escalate |
| `tables` fail | The card layer cannot read the variant (it says why) | The baseline plays card-blind priors; prefer the fallback |
| `search` fail or 0 accepted generations | No improvement found in the budget | Expected sometimes; `select` handles it |
| `validation` withheld | Timeout integrity gate fired in-process | Rerun on an idle machine; do not read points |
| `limits` fail | Too large, too much memory, numpy loaded or not loaded from the bundle | Do not submit that bundle; the rollback stands |

## When the variant is out of scope

Antes, 9 or more seats, draw, stud, high-low, fixed-limit and other decks all fail at
`load`, in under a second. That is deliberate: nothing falls back to a practice profile.
The options, in order:

1. If only the **structure** is out of scope (antes, more than 8 seats), the game itself
   can still be rehearsed. Run the pipeline with the nearest legal structure, for example
   no ante or 8 seats, to get a legal bot. Say in the submission notes that the
   rehearsal did not match the structure.
2. If the **game** is out of scope, the engine needs a new PokerKit game class in
   `arena/rules/reveal.py`. That is code work, not a pipeline run; nothing here is
   pre-tested for it.

## What is not measured

- **The official sandbox.** The gate measures this repository's sandbox: a 1 ms
  parent-observed deadline, 1 GiB `RLIMIT_AS` and 64 KiB of bot state. The official
  sandbox and its timing semantics are unknown.
- **Memory.** "100 MB" is checked twice: as artifact bytes, and as the resident memory
  the bot adds in a probe process (every seat at the table plays it). The official limit
  may measure something else.
- **Population.** Validation uses the frozen opponent population, read through the
  nearest practice card adapter (`strategy.population.cards.card_ruleset`). On a revealed
  variant every opponent is a transfer bot. Points against them say nothing about the
  real field.
- **Sandbox validation.** Validation runs in-process. Population members can play the
  reveal variant only where they see `$ARENA_REVEAL_VARIANT`, which sandboxed members do
  not. Only the candidate is sandboxed, in the deployed gate.
- **Push/fold.** Push/fold and endgame tables are never regenerated on the day. The
  generator needs numba, a hold'em category priority, two-card no-limit showdowns, the
  default 8-seat structure and 10–20 minutes per mode, and neither bundled bot reads
  them.

## Open organiser questions that change this playbook

The plan's list applies in full. These matter most here:

- The exact Agent interface.
- How 100 MB is measured.
- Whether antes or more than 8 seats are possible.
- What falls inside the 1 ms deadline.
- How submission windows and rollback work.
