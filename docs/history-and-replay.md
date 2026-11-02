# Event log, history, replay, PHH and run manifests

## Private event log (event schema version 1)

Every transition appends `EventEnvelope` records to `Table.log`:

```json
{"schema_version": 1, "table_id": "t_…", "hand_id": "h_003", "seq": 117,
 "type": "ActionApplied", "data": {"seat": 4, "street": "flop", "action": {"type": "raise_to", "amount": 120}, …}}
```

- Each `type` has a strict data model in `arena/domain/events.py`. Unknown types,
  extra fields, booleans or floats where integers belong, and bad schema versions are
  rejected when an event is appended and again when a history is loaded.
- The log is private. Players and spectators get views only through
  `arena.table.projections` (see `observations-and-projections.md`).

What it records:
- `TableStarted`: config, config hash, seats, stacks, RNG implementation, seed and
  stream, arena and PokerKit versions.
- Per hand:
  - `HandStarted`, with the **explicit deck**, start stacks, positions and a
    hand-start hash;
  - every blind, hole card, burn and board card;
  - each `DecisionRequested` with its state hash;
  - accepted actions, fallbacks and automatic checks;
  - bet collection, refunds, reveals, PokerKit's raw chip pushes, the odd-chip
    correction and the authoritative `PotAwarded` pots;
  - `HandCompleted`.
- Then `PlayerEliminated` and `TableScored`.

## State hashes

- `DecisionRequested.state_hash` is the SHA-256 of the canonical JSON of the full
  decision state, with domain tag `arena-state-v1`. That state is:
  - the config hash, hand number, positions and blinds, and standings;
  - street, actor and call level;
  - every player's stack, commitments and status;
  - collected chips and the folded list;
  - all hole cards, the board, burns and the undealt deck order;
  - the betting-rights overlay and the legal-action descriptor.
- `HandStarted.start_hash` covers the hand-start snapshot: stacks, positions, blinds,
  deck and standings.
- Table IDs, decision IDs, state versions and timestamps are excluded. Two tables with
  different IDs and the same play hash identically (tested).
- Hashes are private, because a hash of a small private state could be brute-forced.

## Canonical history files

`arena.history.canonical.write_history(table, dir)` writes
`<dir>/<table_id>.json`. The document holds:
- `format = "arena.table-history"`, `schema_version = 1`, `privacy = "private"`;
- the full event log;
- a summary: complete, hands, decisions, final stacks, points.

The JSON is canonical: sorted keys, fixed separators, ASCII, no NaN. Loading and
re-dumping gives identical bytes. A history is an operator/reviewer artifact. Players
download `HistoryView`s instead (`python -m arena.cli history --viewer seat:3`).

## Replay (`python -m arena.cli replay --table-id ID --verify-hashes`)

Replay works from the file-based history; Postgres comes in milestone 5.

1. Load and validate the history. Refuse it if its PokerKit version differs from the
   pinned one (exit 3), unless `--allow-engine-mismatch` is given, in which case the
   difference is reported.
2. Rebuild the table from `TableStarted`, feeding each hand its stored deck. Nothing
   is reshuffled.
3. Re-submit only accepted commands: manual `ActionApplied` and `ActionFallback`.
   Automatic checks, dealing and settlement must come out of the engine by
   themselves. No agent is re-run.
4. Require the regenerated log to equal the stored log event for event. That
   includes every hand-start and decision hash, every payout, elimination and score.
   The first difference halts with the event index, hand and decision (exit 2).
   Nothing is repaired.
5. Compare the summary (final stacks, points). An incomplete table verifies up to its
   last committed event.

The history is found at `artifacts/tables/<id>.json`, then
`artifacts/runs/*/tables/<id>.json`. Override with `--history-dir` or `--file`.

## PHH export (`python -m arena.cli export-phh --table-id ID --out file.phhs`)

PHH is secondary interchange. Each complete hand becomes a PHH TOML document:
- players are in PokerKit index order;
- `seats` holds 1-based physical seats;
- `blinds_or_straddles` is PokerKit's vector;
- `antes` is PokerKit's vector of the amounts actually posted, and
  `ante_trimming_status` is true for classic antes (live contributions) and false for a
  big-blind ante (dead money); `_arena_ante` and `_arena_ante_posting` record the
  configured ante and are written only for hands with an ante;
- practice-rule details are `_arena_*` user fields.

Rules:
- Burns, the deck, ICM and timing are not representable. The canonical JSON keeps them.
- A nominal-big-blind call is `cbr <total> # arena:nominal_bb_call`.
- An engine auto-check is `cc # arena:auto`.
- Hands with a short-big-blind correction (D1) or an odd-chip correction (D5) are
  flagged. A generic PHH reader prices those differently.

`arena.history.phh.replay_phh` reads PHH with `tomllib` (not PokerKit), rebuilds a deck
that deals the recorded cards, re-runs the hand in the arena engine and checks:
- revealed cards;
- automatic checks;
- final stacks.

Re-exporting gives the same text. As a secondary check, PokerKit's own reader must
agree with every unflagged hand; flagged hands are counted as exclusions, not passes.
There is no claim of PokerStars or PokerTracker compatibility.

## Randomness

- Interactive tables default to `SystemRng` (`secrets`).
- Tests and benchmarks use `SeededRng` (`sha256-ctr-v1`, unbiased Fisher–Yates).
  Streams are separate: `env` for decks and the first button, and `policy-seat-N` for
  each baseline seat (`arena.table.driver.policy_rngs`).
- Participants never receive the environment seed.
- Fixtures store explicit decks.

## Run manifest (`python -m arena.cli run-baseline …`)

`arena.benchmark.manifest.run_baseline` plays seeded in-process tables and writes:

```
artifacts/runs/<run_id>/manifest.json      # RunManifest (no seeds)
artifacts/runs/<run_id>/tables/<id>.json   # private histories, replayable
artifacts/runs/<run_id>/private/seeds.json # the seed list
```

The manifest records:
- the code commit and dirty flag, the `uv.lock` SHA-256, and the arena, PokerKit and
  Python versions;
- the ruleset, every config value and the config hash;
- per-seat participant artifact hashes;
- the seat count, the RNG implementation, the seed count and the seed-list hash;
- the seat-allocation policy, budget semantics, host OS, CPU and container details,
  the scheduled table count and the retry policy;
- completed, failed and incomplete table IDs, and mean points by seat.

`content_hash()` excludes only `created_at`, and reruns of the same manifest produce the
same hash and byte-identical histories (tested). Milestone 4's benchmark runner extends
this with real agent artifacts, seat rotation and timing.
