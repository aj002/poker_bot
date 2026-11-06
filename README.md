# arena: Sequoia-style pokerbot practice engine

This is the authoritative poker engine for a practice arena where bots play short
single-table tournaments. It covers rulesets, dealing and betting, pot settlement,
dead-button seating, eliminations and ICM scoring. It also has a private event log with
explicit decks and state hashes, privacy-safe seat and spectator views, canonical JSON
histories with hash-verified replay, and PHH export. A playable table (FastAPI plus a
React table) lets humans and simple baseline bots play on the same engine, in memory or
durably on PostgreSQL with crash recovery. A local Python bot SDK runs bots in sandboxed
workers with a 1 ms decision deadline, with baseline bots and a paired-seed benchmark
runner. The browser also replays finished tables and launches and reports batch runs.

All rules are **practice assumptions**, not official Sequoia rules. See
[`docs/assumptions.md`](docs/assumptions.md) for every local rule choice and every place
where the engine corrects PokerKit.

## Rulesets

| `ruleset_id` | Game |
| --- | --- |
| `practice_nlhe` | No-limit Texas Hold'em |
| `uncertainty_shortdeck` | No-limit short-deck Hold'em (6–A, PokerKit's native hand order) |
| `uncertainty_plo4` | Pot-limit Omaha, four hole cards |
| `reveal_variant` | The build-day slot: whatever `$ARENA_REVEAL_VARIANT/variant.json` describes (see below) |

Tables seat 2–8 players. They use a dead button, escalating blinds, optional antes per
level (`classic` or `big_blind`; none by default), and a hand cap (default 200). If the cap is reached, the survivors are scored by ICM.

`reveal_variant` is for a variant announced only on build day. The engine builds it at
run time from a variant description (deck ranks, 2–5 hole cards, best-five or
exactly-two-hole showdown, any order of the nine hand categories, the lowest straight,
no-limit or pot-limit) out of PokerKit's own parts, and rejects anything it cannot express
before a hand is dealt (`arena/rules/reveal.py`). Practice registries, sweeps and
fingerprints still iterate only the three practice rulesets.

## Layout

```
arena/domain/      cards, seeded RNG, canonical JSON, strict models: config, actions,
                   event schema (events.py), observation and view types (observations.py)
arena/rules/       rule profiles and the PokerKit 0.7.7 adapter (betting overlay, settlement)
arena/table/       dead-button positions, the table service, state hashes, projections, drivers
arena/tournament/  standings, same-hand bust ranking, Malmuth-Harville ICM
arena/history/     canonical JSON histories, replay verification, PHH export
arena/agents/      bot SDK (sdk.py), sandboxed worker pool and deadlines (worker.py),
                   process confinement (sandbox.py), baselines/, styles/ (opponent styles)
arena/benchmark/   run manifests, paired seed-block benchmark runner (runner.py)
arena/api/         FastAPI server: seat credentials, per-table command writer, deadlines, WebSocket
arena/persistence/ memory and PostgreSQL stores, migrations, snapshot + replay recovery
web/               React + TypeScript + Vite: setup, lobby, table, results, replay, batch runs; Playwright e2e
strategy/solve/    offline solvers on a dense extensive-form tree: regret matching, CFR, CFR+,
                   linear/discounted CFR, external-sampling MCCFR, best response, NashConv
strategy/games/    Stage 2 toy games with analytic solutions (Kuhn, Leduc, AKQ, [0,1] games,
                   clairvoyance, marked draw, dry side pot) and the fixture gate registry
strategy/endgame/  Stage 3 push/fold subgames over the full stack vector, class-vector CFR+ with
                   exact best responses, limp/min-raise gaps, table lookup, push/fold bots (bots/)
strategy/value/    tournament rollouts and the continuation value V(s) (ICM + fitted residual)
strategy/blueprint/ Stage 4 abstracted point-aware MCCFR blueprint for NLHE: abstraction, numba
                   trainer, policy pack, deployed bot (no numba), responses and audits (README.md)
strategy/policy/   Stage 1 baseline policy (runtime.py, bundles, deployed gate) and generic.py, the
                   variant-agnostic fallback bot
strategy/rehearsal/ Stage 9 build-day pipeline: announcement -> conformance -> tables -> CEM re-tune
                   -> validation -> gate -> submission, every step timed (heldback/: rehearsal specs)
strategy/data/     pinned Pluribus PHH download, import through the arena engine, study packs,
                   bot-vs-Pluribus agreement diagnostic (fixtures/: 24 upstream hands)
examples/          template_bot.py: copy it to start a bot
tests/agents/      fault-injection bots, sandbox probes, fake-clock deadline tests, restore
tests/api/         API behaviour and HTTP/WebSocket leak checks
tests/persistence/ resume equivalence, store contract, crash/race/outage/fencing faults
tests/golden/      spec fixtures as JSON (G, E and T series), re-settled by the oracles
tests/property/    generated hands and tournaments, information-set and leak tests
tests/history/     event schema, replay and tampering, PHH round trips, manifests
tests/oracles/     independent evaluators, pot ledger and ICM (stdlib only)
docs/              assumptions and discrepancies, G20/G21 states, view contracts,
                   history/replay notes, JSON Schemas (docs/schemas/)
```

## Running locally

You need [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 if it is missing)
and Node 20+ for the web table.

```bash
make setup         # uv sync --frozen, npm ci in web/
make dev           # API on :8317, table on http://127.0.0.1:5317 (tables in memory)
make party         # advisor mode served to the Wi-Fi, for use from a phone (see below)
make dev-postgres  # same, with tables in PostgreSQL (docker compose, port 54329)
make verify-postgres  # persistence and API suites against PostgreSQL
make verify-web    # web typecheck + Playwright suite (npx playwright install chromium once)
make verify-fast   # pytest -m "not release and not slow and not oracle"
make verify-release  # every release gate as resumable shards -> artifacts/test-report.{json,md}
make verify-release-status  # shard states per gate
make cli-demo      # one seeded 6-player NLHE tournament
```

**Advisor mode.** `make party` builds the web app once and serves it on every network interface (port 5317; the
API stays on loopback behind it). It prints the Mac's address; open `http://<address>:5317/#/advisor` on a phone
on the same Wi-Fi. You type a real-life hand in as it happens (your cards, each player's action, the board) and the
final bot (competition bundle `bec8892d`, run in the server process with no 1 ms deadline and no numba) shows what it
would play, with its mix and a one-line reason. It remembers each seat across hands like the arena does. Sessions
are saved under `artifacts/advisor`. The `final_bot` is also a normal opponent in the bot list. Set `ARENA_PUBLIC_URL` if the printed address is wrong, and `ARENA_NO_OPEN=1` to stop it opening a browser.

Release verification (spec sections 12 and 13), its gates, checkpoints, failing-seed
minimizer and mutation checks are described in `docs/release-verification.md`.

The offline solvers (`strategy/solve`) need numba, which is in the opt-in `train` group:
`make setup` leaves it out, so the arena installs anywhere, including Intel Macs where
numba has no wheels. The targets below add it on demand (`uv run --group train`).
Without numba, pytest does not collect any test module that imports numba, `strategy.solve`
or `strategy.games`, directly or through a repository module (`tests/conftest.py`), so
`make verify-fast` stays green. With `ARENA_REQUIRE_NUMBA=1`, set by the solver targets and
the release `fast` gate, a missing numba stops pytest with a usage error. OpenSpiel is a
dev-only oracle in the `oracle` group; nothing outside `tests/strategy/oracle` and
`tests/strategy/test_openspiel_oracle.py` imports it.

```bash
make verify-strategy    # every fast strategy test, solver tests included
make gates-stage2       # every Stage 2 fixture gate at n = 200 as a Markdown table
make verify-solve-slow  # full-resolution gates, ES-MCCFR agreement over seeds, long Leduc runs
make verify-oracle      # live Kuhn/Leduc cross-checks against OpenSpiel 2.0.2
make bench-solve        # NashConv vs nodes touched and wall time -> artifacts/bench-solve/
uv run --group oracle python -m tests.strategy.oracle.openspiel_reference  # refresh the frozen file
```

Stage 3 endgame tables are generated per variant through the rules adapter (NLHE and
short deck; PLO4 is pot-limit and has no push/fold tables):

```bash
make tables-showdown   # all-in showdown matrices per variant (about 20 s)
make tables-endgame    # point-aware, chip-EV and chart push/fold tables (about 30 min on 8 cores)
make gates-endgame     # exact two-player regrets, multiway gains, limp gaps, table audit
make value-rollouts value-fit  # rollouts and the fitted continuation value
make lab-endgame       # paired SNG experiments: point-aware vs chip-EV, chart and no layer
make tables-response tables-exploit  # population response curves (train partition) and tables responding to them
make lab-response      # the pre-registered response-vs-equilibrium comparisons (docs/endgame-response-preregistration.md)
```

The Stage 4 blueprint (`strategy/blueprint/README.md`) trains with numba and deploys
as a numba-free bot over a committed 4.4 MB pack
(`strategy/tables/data/blueprint-practice_nlhe.npz`). Its bots are
`strategy/blueprint/bots/blueprint.py` (point-aware push/fold leaf),
`blueprint_capped.py`, `blueprint_chips.py` and `blueprint_noleaf.py`:

```bash
make blueprint-train      # BP_HOURS=4 of ES-MCCFR; resumes from the run directory's checkpoint
make blueprint-pack blueprint-bundles  # pack and self-contained submission bundles
make blueprint-exploit blueprint-response blueprint-audit  # HU exploitability, approximate responses, boundary audit
make lab-blueprint        # paired points evaluations on the validation population, then a sandbox smoke
```

The Stage 5 league (`strategy/league/README.md`, `docs/league-training-preregistration.md`)
keeps frozen checkpoints, meta-solves an opponent mixture over their payoff matrix
and promotes learners only through paired validation gates. Its optional bot
`strategy/league/bots/league.py` plays the league checkpoint C1 (pack
`strategy/tables/data/league-practice_nlhe.npz`) on the blueprint runtime. The
blueprint stays the default: C1 and its blueprint blends (`strategy/league/blend.py`)
fail the declared robustness clauses against the league's exploiters.

```bash
uv run --group train python -m strategy.league.runner init --run artifacts/league/main-v1
uv run --group train python -m strategy.league.runner iterate --run artifacts/league/main-v1
uv run python -m strategy.league.summary artifacts/league/main-v1
```

The Stage 6 exploitation layer (`strategy/exploit/README.md`,
`docs/opponent-exploitation-preregistration.md`) adds per-seat archetype beliefs and
confidence-capped response experts to the blueprint, within a per-table loss budget. Its bot
`strategy/exploit/bots/exploit.py` loads the pack `strategy/tables/data/exploit-practice_nlhe.npz`:

```bash
make exploit-bundles exploit-validate   # the pre-registered paired validation runs
make gate-exploit                       # the sandboxed deployed-path gate
```

The final competition bot (Stage 8, `strategy/final/README.md`) has one entry point. It plays
the exploiting blueprint on NLHE and the variant-agnostic fallback (`strategy/policy/generic.py`)
on every other ruleset. Its bundle is deterministic and hashed (7.6 MB, no numba) and plays
identically to the exploit bundle through the sandbox worker. A distilled table student was
measured and rejected: it loses to the teacher against blueprint-like opponents. The holdout
evaluation was pre-registered in `docs/final-holdout-preregistration.md` and run once.
Holm-corrected, the final bot beat the blueprint by +2.10 points [+1.44, +2.75], the baseline
nlhe-v2 by +3.51 and the endgame-capped bot by +7.86. It was non-inferior to the blueprint
against the two holdout-only families it has no model of (+0.57 [+0.02, +1.12]). The
submission bundle passed the deployed gate on NLHE (decide p99 361 µs, 0.010% bot-caused
timeouts) and PLO4. Three earlier runs, under the older timeout classifier, failed the
0.02% ceiling. The bundle and every gate run are recorded in
`strategy/final/submission.json`. Submit
by following [`docs/submission-checklist.md`](docs/submission-checklist.md):

```bash
make final-arms final-package final-verify   # bundle, hashes, size and memory limits, tarball
make gate-final gate-final-generic           # the deployed gate on NLHE and PLO4 (idle machine)
make final-holdout final-holdout-confirm     # the one-shot holdout run and its Holm confirmation
```

Build day (Stage 9, [`docs/build-day-playbook.md`](docs/build-day-playbook.md)): write the
announced variant and structure into a spec like `strategy/rehearsal/heldback/*.json`, then

```bash
make rehearse REVEAL_SPEC=path/to/announcement.json   # timed pipeline -> artifacts/rehearsal/<name>-fast/
make gate-fallback FALLBACK_RULESET=uncertainty_plo4  # the variant-agnostic fallback through the deployed gate
```

The CLI plays seeded headless hands and tournaments with simple built-in policies
(`legal_random`, `check_or_call`, `check_or_fold`, `shove`):

```bash
uv run python -m arena.cli hand --ruleset uncertainty_plo4 --players 4 --seed 3
uv run python -m arena.cli tournament --ruleset uncertainty_shortdeck --players 8 --seed 7 --trace
```

The same seed always produces the same decks, positions, events and scores.

Histories, replay and exports work from files under `artifacts/` (git-ignored):

```bash
uv run python -m arena.cli tournament --players 6 --seed 9 --save --table-id t_demo
uv run python -m arena.cli replay --table-id t_demo --verify-hashes
uv run python -m arena.cli history --table-id t_demo --viewer seat:2     # or spectator / reviewer
uv run python -m arena.cli export-phh --table-id t_demo --out artifacts/t_demo.phhs
uv run python -m arena.cli run-baseline --players 6 --tables 20 --seed demo   # manifest + histories
uv run python -m arena.cli schemas                                         # regenerate docs/schemas
```

## Playing in the browser

Run `make dev` and open http://127.0.0.1:5317. Choose a rule scenario, 2–8 seats
and who sits where (a human or a baseline bot; bots that do not support the ruleset
are disabled), then create the table. You take the first human seat. Every other human seat gets an
"Open in new tab" link (each tab holds its own private seat credential), and the host
starts the table once every human seat is taken. With `make dev` tables live in memory,
so restarting the server clears them; `make dev-postgres` keeps them and recovers
them after a restart or crash ([`docs/persistence.md`](docs/persistence.md)).

After a table, **Replay this table** steps through every action and hand at 1–16×,
from your seat or as a spectator; the host can switch to the reviewer view with every
hole card. `make dev` also prints an **operator token** (saved to
`artifacts/operator-token`). Paste it under **Operator** to seat registered bot files
and to launch batch benchmark runs from **Batch runs**, whose reports show config and
artifact hashes, mean points with confidence intervals, timing and fallbacks per
scenario and seat count. The endpoints, deadlines and privacy rules are in
[`docs/playable-table.md`](docs/playable-table.md).

New to poker? Open **Learn** for five short practice tables, from heads-up against a
labelled calling station to a full table of hidden random styles, with progress kept in
this browser. On No-limit Hold'em tables, setup also has an **Opponent styles** mode
(labelled, hidden, mixed or random) and a **Learning aids** switch. With aids on, the
table shows your hand in words, position labels and tips, the opening chart for your
seat, pot odds, equity against random hands, an optional coach hint and a recap after
each hand. All of these use only your own cards and public information. A **Notes and
guess** panel tracks every opponent's stats with sample sizes and records your style
guesses. When the table ends, the **Reveal** scores those guesses against each seat's
true style and shows the evidence. `ARENA_SCREENSHOTS=/some/dir npx playwright test
screenshots` (in `web/`) captures a walkthrough of these screens.

**Study Pluribus** (next to Learn) replays real six-max hands played by Pluribus and
pauses at each of its decisions: pick an action and a size, then see what it did, with
pot odds, equity and the spot's features alongside and a running agreement score kept
in this browser. `make dev` builds 24 sample hands from the committed fixture; for all
10,000 hands run `make study-pack`, which first downloads the dataset into a cache
outside git and checks it against a pinned hash. See [`docs/pluribus-study.md`](docs/pluribus-study.md).

## Bots

Write a bot by copying [`examples/template_bot.py`](examples/template_bot.py), then play
it against the built-in baselines (`check_fold`, `calling_station`, `legal_random`, and
the NLHE-only `nlhe_heuristic`). Each run prints mean points with a 95% interval,
p50/p95/p99/max decision timing in seconds, and fallback counts:

There are also six **opponent styles** modelled on the playing styles of *Harrington on
Hold'em, Vol. I*: `rock`, `aggressive`, `super_aggressive`, `station`, `weak_tight` and
the advanced `switcher`. Each has declared stat ranges that a test checks. Tables can show
them labelled, hide them, mix both or deal random hidden styles, then reveal each seat
with the evidence once the table ends. They also serve table-reading stats and seat-only
learning aids (position, hand in words, outs, equity, pot odds and a coach hint). See
[`docs/opponent-styles.md`](docs/opponent-styles.md).

```bash
uv run python -m arena.cli match --bot examples/template_bot.py              # 240 tables
uv run python -m arena.cli match --bot my_bot.py --opponents rock,aggressive,station
uv run python -m arena.cli match --bot my_bot.py --ruleset uncertainty_shortdeck --tables 60
uv run python -m arena.cli benchmark --bots nlhe_heuristic,calling_station,legal_random,check_fold \
    --out artifacts/runs/demo
```

Sandboxed workers need Linux with util-linux `unshare`, unprivileged user namespaces,
Landlock and x86-64 seccomp. Tests that need them skip themselves elsewhere. See
[`docs/agent-sdk.md`](docs/agent-sdk.md) for the bot contract and
[`docs/agent-sandbox.md`](docs/agent-sandbox.md) for what is and is not hardened, with
timing measured on this VM.

The view types other components build on are documented in
[`docs/observations-and-projections.md`](docs/observations-and-projections.md). History,
replay, PHH and manifests are described in
[`docs/history-and-replay.md`](docs/history-and-replay.md).
