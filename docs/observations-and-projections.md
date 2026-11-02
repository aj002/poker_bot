# Observations, views and projections (contract version 1)

This is the stable interface for everything that shows table state to someone: the
agent SDK and workers (milestone 4), the in-memory server and table UI (milestone 3b),
and replay and review later. **Build on these types; do not read the engine, the
`Table` internals or the private event log to render anything for a player.**

The JSON Schemas in [`docs/schemas/`](schemas/) are generated from the models.
`tests/property/test_schemas_stable.py` fails if a model drifts from its committed
schema. To change a contract on purpose:
1. Bump `schema_version`.
2. Run `python -m arena.cli schemas`.
3. Update this page in the same commit.

## Module paths

| What | Where |
| --- | --- |
| View and observation types | `arena/domain/observations.py`: `Observation`, `TableView`, `PlayerView`, `PotView`, `BlindsView`, `NextLevelView`, `DecisionBudget`, `PublicEvent`, `HistoryView` |
| Action types | `arena/domain/models.py`: `Action`, `ActionEnvelope`, `LegalActions`, `TournamentConfig` |
| Projector (the only way to build views) | `arena/table/projections.py`: `Viewer`, `observation`, `table_view`, `table_history_view`, `history_view`, `project_event`, `ProjectionDenied`, `PUBLIC_FIELDS`, `OWNER_FIELDS` |
| Private event schema | `arena/domain/events.py`: `EventEnvelope`, `EVENT_MODELS`, `EVENT_SCHEMA_VERSION` |
| Table service (commands) | `arena/table/service.py`: `Table`, `ActionRejected`, `Receipt`, `seeded_table` |
| Canonical history, replay, PHH | `arena/history/canonical.py`, `arena/history/replay.py`, `arena/history/phh.py` |
| Run manifest | `arena/benchmark/manifest.py` |
| JSON Schemas | `arena/schemas.py` writes `docs/schemas/*.v1.json` |

## Viewers

```python
from arena.table.projections import Viewer, observation, table_view, table_history_view

observation(table, seat)                       # decision request for the acting seat only
table_view(table, Viewer.for_seat(seat))       # what that seat sees right now
table_view(table, Viewer.spectator())          # public view
table_history_view(table, Viewer.for_seat(s))  # downloadable history for a seat
table_history_view(table, Viewer.reviewer())   # full private log; completed tables only
```

| Viewer | Current state (`table_view`) | History (`history_view`) |
| --- | --- | --- |
| `Viewer.for_seat(s)` | Public state plus seat `s`'s own hole cards, and `legal` when `s` is to act | Public events plus `HoleDealt.card` for seat `s` |
| `Viewer.spectator()` | Public state only (`hero_seat` and `hole_cards` are null) | Public events only |
| `Viewer.reviewer()` | Denied (`ProjectionDenied`) | Every event and field, **only when the table is complete** |

Denials raise `ProjectionDenied` (a `PermissionError`). It is raised for:
- an observation for a seat that is not to act;
- a seat that is not at the table;
- a reviewer view while the table is still running.

A server should map it to 403. **Who** a caller is (session, seat credential,
reviewer permission) is decided by the caller's layer. The projector only enforces
what each viewer kind may see.

## `Observation` / `TableView` fields

`Observation` is a `TableView` in which `perspective="seat"`, `hero_seat == actor_seat`,
and `hole_cards`, `hand_id`, `decision_id`, `street` and `legal` are all present. Humans
and bots get the same `Observation`. UI conveniences must not add hidden state.

| Field | Meaning |
| --- | --- |
| `schema_version` | `1` |
| `perspective` | `"seat"` or `"spectator"` |
| `compatibility_status` | Always `"practice_assumptions"`. Show the label. |
| `table_id`, `hand_id`, `decision_id`, `state_version` | Identifiers. Echo `hand_id`, `decision_id` and `state_version` in the action envelope. |
| `ruleset_id`, `config_hash` | Frozen configuration identity |
| `hero_seat`, `hole_cards` | The viewing seat and its own cards (null for spectators, and for an eliminated hero) |
| `seats` | Physical seat ring (0..7). Seat numbers are anonymous. |
| `initial_player_count`, `surviving_player_count` | Players at the start, and players with chips |
| `hand_number` | 1-based number of the current (or last) hand |
| `button_seat`, `small_blind_seat`, `big_blind_seat` | Dead-button markers. `button_seat` may be an empty seat; `small_blind_seat` is null when the small blind is dead. |
| `street`, `actor_seat` | Null between hands and after the table ends |
| `board` | Public board cards |
| `pot_total` | All chips committed this hand (collected pots plus current street bets) |
| `settled_pot` | Chips collected into pots from earlier streets |
| `current_bet_to` | The street total a call matches. It is the nominal big blind preflop even when the big blind is posted short. |
| `pots` | Collected pots, `{amount, eligible}`, merged where eligible sets match. Current street bets are not in `pots`. At settlement `PotAwarded` events are authoritative (one pot per contribution level). |
| `players[]` | Every seat in the ring: `seat`, `stack`, `street_committed`, `hand_committed`, `status` (`active`, `folded`, `all_in`, `eliminated`), and `shown_cards` (null unless revealed at showdown this hand) |
| `blinds`, `next_level` | Current blinds; next level and the hand it starts at (null if none before the cap). With antes configured both also carry `ante` and `ante_posting` (`classic` or `big_blind`); the keys are absent when the ante is 0. |
| `payouts` | Initial payout points vector |
| `hands_played`, `hands_remaining`, `hand_cap` | Completed hands; `hand_cap - hands_played` (counts the current hand) |
| `decision_budget` | `bot_budget_ns` and `human_turn_ms` from the config |
| `legal` | `LegalActions`, only for the acting seat (see below) |
| `action_history` | Current hand's events as `PublicEvent`s for this viewer, from `HandStarted` onward |
| `table_complete`, `points` | When complete: final points by seat (sum = payout total) |

The spec §6.2 accounting example is a test (`tests/golden/test_observation_example.py`).

## `LegalActions` and actions

`LegalActions` fields:
- `fold`, `check`
- `call_amount`: the amount added; null if there is nothing to call
- `can_raise`
- `min_full_raise_to`, `max_raise_to`: an inclusive interval of street totals
- `short_all_in_raise_to`: the only legal raise below a full raise, when set
- `all_in_raise_to`
- `pot_raise_to`: the pot-sized raise, when one is legal
- `raise_blocked_reason`: human-readable text, e.g. why a short all-in did not reopen betting

`LegalActions.allows(action)` is the server's check.

`Action`:
- `{"type": "fold" | "check" | "call"}`, or
- `{"type": "raise_to", "amount": <street total>}`.

`call` never takes an amount, because the server prices calls. `raise_to` is a total,
not an increment. The UI shows "Raise to 100 · add 80" by subtracting `street_committed`.

`ActionEnvelope` is the request body:
- `action_id`: the client idempotency key, `[A-Za-z0-9_.:-]{1,128}`
- `table_id`, `hand_id`, `decision_id`, `expected_state_version`
- `action`

The acting seat comes from the session or worker lease, never the body. The
idempotency registry (same id + same payload returns the receipt, a changed payload is
an error) belongs to the server layer, which keys it by `action_id`.

`Table.submit(seat, action, hand_id=, decision_id=, expected_state_version=)` either
applies the action atomically and returns a `Receipt`, or raises `ActionRejected(code)`
without changing anything. The codes are:
- `table_complete`
- `wrong_hand`
- `stale_decision`
- `stale_version`
- `out_of_turn`
- `invalid_schema`
- `illegal_action`
- `no_decision`

`Table.apply_fallback(seat, reason, rejected)` consumes a decision with check, or fold
when check is not legal. The reason code is public; the rejected payload is private.

## Public events (`PublicEvent`, `HistoryView`)

A `PublicEvent` is `{seq, hand_id, type, data}`. `data` contains only the fields
allowlisted for the viewer, already in JSON form (map keys are strings). `seq` is the
private log position, so it has gaps where hidden events were dropped.

| Event | Public `data` fields | Notes |
| --- | --- | --- |
| `TableStarted` | `config`, `config_hash`, `seat_ring`, `start_stacks` | RNG seed and engine build stay private |
| `HandStarted` | `hand_number`, `level`, `small_blind`, `big_blind`, `positions`, `small_blind_seat`, `start_stacks`, and `ante`, `ante_posting` when the hand has an ante | Deck and start hash stay private |
| `AntePosted` | `seat`, `amount` | Before the blinds; one per posting seat |
| `BlindPosted` | `seat`, `amount` | |
| `HoleDealt` | `seat` (the owner also gets `card`) | One event per card |
| `BoardDealt` | `street`, `cards` | |
| `DecisionRequested` | `decision_id`, `seat`, `state_version` | State hash stays private |
| `ActionFallback` | `seat`, `reason` | Precedes the `ActionApplied` it caused |
| `ActionApplied` | `seat`, `street`, `action`, `added`, `street_committed`, `nominal_big_blind_call`, `auto` | `auto: "no_contestable_action"` marks engine checks |
| `BetsCollected` | `bets` | |
| `ExcessReturned` | `seat`, `amount` | Unmatched chips refunded |
| `CardsRevealed` | `seat`, `cards` | Showdown; every live hand is shown |
| `PotAwarded` | `pot_index`, `amount`, `eligible`, `awards` | One per contribution level |
| `HandCompleted` | `final_stacks` | |
| `PlayerEliminated` | `seat`, `hand_number`, `positions`, `points` | |
| `TableScored` | `placements`, `points` | |
| `CardBurned`, `HandKilled`, `EngineChipsPushed`, `ChipsPulled`, `OddChipCorrection` | hidden | Engine bookkeeping or private cards |

An event type without a rule raises `UnknownEventType`. Projection fails closed.

## Privacy guarantees and how they are tested

- Views never contain:
  - opponents' unrevealed or folded cards;
  - burns or the deck;
  - RNG seeds or state;
  - state hashes;
  - other seats' fallback payloads;
  - engine objects.
- Unavailable cards are `null` or absent, never placeholders.
- `tests/property/test_information_sets.py` builds pairs of tables that differ only in
  cards the hero cannot see. Until a reveal, it checks that the canonical JSON of the
  hero's `Observation`, its `TableView` and `HistoryView`, and the spectator's views are
  byte-identical, and that a deterministic policy picks the same action. It also scans
  every view, history and rejection message for card strings that viewer is not
  entitled to.
- HTTP bodies, WebSocket frames and the DOM must be checked the same way by the
  server and UI work (spec §12.2).
