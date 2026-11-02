UV ?= uv

.PHONY: setup dev dev-postgres db-up db-down verify-fast verify-postgres verify-release verify-release-status \
	verify-release-report verify-release-smoke verify-web test cli-demo \
	verify-solve-slow verify-oracle gates-stage2 bench-solve

setup:
	$(UV) sync --frozen
	cd web && npm ci

dev:
	UV=$(UV) ./scripts/dev.sh

db-up:
	docker compose up -d --wait postgres

db-down:
	docker compose down

dev-postgres: db-up
	ARENA_STORE=postgres UV=$(UV) ./scripts/dev.sh

# Solver test modules (anything importing numba, strategy.solve or strategy.games) are not
# collected unless numba is installed; verify-strategy runs them.
verify-fast:
	$(UV) run pytest -m "not release and not slow and not oracle" -q

# The offline solvers need numba from the opt-in `train` group (no Intel-macOS wheels).
# ARENA_REQUIRE_NUMBA=1 makes pytest fail at startup instead of dropping the solver tests.
TRAIN = --group train
REQUIRE_NUMBA = ARENA_REQUIRE_NUMBA=1

# Stage 2 solver gates at full resolution, long ES-MCCFR runs and sampled-vs-full agreement.
verify-solve-slow:
	$(REQUIRE_NUMBA) $(UV) run $(TRAIN) pytest -m slow tests/strategy -q

# Kuhn/Leduc cross-checks against OpenSpiel (dev-only dependency group).
verify-oracle:
	$(REQUIRE_NUMBA) $(UV) run $(TRAIN) --group oracle pytest -m oracle tests/strategy -q

gates-stage2:
	$(UV) run $(TRAIN) python -m strategy.games.fixtures --resolution 200

bench-solve:
	$(UV) run $(TRAIN) python -m strategy.solve.bench

verify-postgres: db-up
	ARENA_REQUIRE_POSTGRES=1 $(UV) run pytest tests/persistence -q
	ARENA_API_TEST_STORE=postgres $(UV) run pytest tests/api -q

# Every section 13 gate as checkpointed shards; resumes, then writes artifacts/test-report.{json,md}.
# Exit 1 unless every applicable gate passed. JOBS defaults to the CPU count; see docs/release-verification.md.
JOBS ?= $(shell nproc 2>/dev/null || echo 4)
verify-release:
	$(UV) run python -m tests.release run --jobs $(JOBS) --exclusive-last

verify-release-status:
	$(UV) run python -m tests.release status

verify-release-report:
	$(UV) run python -m tests.release report

# The older marker-based release smoke tests (a subset of the gates above).
verify-release-smoke:
	$(UV) run pytest -m release -q

verify-web:
	cd web && npx tsc -b && npx playwright test

test: verify-fast

cli-demo:
	$(UV) run python -m arena.cli tournament --ruleset practice_nlhe --players 6 --seed 1

# Strategy lab (docs/strategy-evaluation.md). Runs land in artifacts/lab/<experiment>/<manifest_id>/;
# a manifest that already ran is not re-run. Jobs default to a safe count for the host and isolation
# (python -m strategy.cli capacity); set LAB_JOBS only to override it.
.PHONY: lab-conformance lab-capacity lab-report lab-experiment lab-null lab-null-variants lab-null-sandbox \
	lab-null-synthetic verify-strategy
LAB_JOBS ?=
LAB_SPEC ?= lab-smoke
LAB_JOBS_FLAG = $(if $(LAB_JOBS),--jobs $(LAB_JOBS))

lab-conformance:
	$(UV) run python -m strategy.cli conformance

lab-report:
	$(UV) run python -m strategy.cli experiment lab-smoke $(LAB_JOBS_FLAG)

lab-experiment:
	$(UV) run python -m strategy.cli experiment $(LAB_SPEC) $(LAB_JOBS_FLAG)

lab-null:
	$(UV) run python -m strategy.cli experiment null-aa-nlhe $(LAB_JOBS_FLAG)

lab-null-variants:
	$(UV) run python -m strategy.cli experiment null-aa-variants $(LAB_JOBS_FLAG)

lab-null-sandbox:
	$(UV) run python -m strategy.cli experiment null-aa-sandbox $(LAB_JOBS_FLAG)

lab-capacity:
	$(UV) run python -m strategy.cli capacity

lab-null-synthetic:
	$(UV) run python -m strategy.cli null-synthetic --replicates 4000 --blocks 20

verify-strategy:
	$(REQUIRE_NUMBA) $(UV) run $(TRAIN) pytest tests/strategy -m "not slow and not oracle" -q

# Pluribus hands (Stage 7; docs/pluribus-study.md). The corpus is cached outside git by a pinned,
# hash-checked download; only a small fixture is committed.
.PHONY: pluribus-fetch pluribus-import study-sample study-pack pluribus-agreement
pluribus-fetch:
	$(UV) run python -m strategy.data.fetch

pluribus-import: pluribus-fetch
	$(UV) run python -m strategy.data.phh_import --out artifacts/phh/pluribus-import.json

study-sample:
	$(UV) run python -m strategy.data.study --source fixture

study-pack: pluribus-fetch
	$(UV) run python -m strategy.data.study

pluribus-agreement: pluribus-fetch
	$(UV) run python -m strategy.data.agreement --jobs $(JOBS) --out artifacts/phh/pluribus-agreement.json

# Stage 3 endgame (docs: strategy/tables/README.md). Heavy: about 30 minutes on 8 cores.
.PHONY: tables-showdown tables-endgame gates-endgame value-rollouts value-fit lab-endgame tables-response \
	tables-exploit lab-response
ENDGAME_RULESETS ?= practice_nlhe uncertainty_shortdeck
ENDGAME_JOBS ?= $(JOBS)

# Only the showdown generator needs numba (the evaluator kernel); everything else runs without it.
tables-showdown:
	for r in $(ENDGAME_RULESETS); do $(UV) run $(TRAIN) python -m strategy.tables.showdown --ruleset $$r --jobs $(ENDGAME_JOBS) || exit 1; done

tables-endgame:
	for r in $(ENDGAME_RULESETS); do for m in points chips chart; do \
		$(UV) run python -m strategy.tables.endgame --ruleset $$r --mode $$m --jobs $(ENDGAME_JOBS) || exit 1; done; done

gates-endgame:
	$(UV) run python -m strategy.endgame.gates --jobs $(ENDGAME_JOBS)

value-rollouts:
	for r in $(ENDGAME_RULESETS); do $(UV) run python -m strategy.value.model rollouts --ruleset $$r --jobs $(ENDGAME_JOBS) || exit 1; done

value-fit:
	for r in $(ENDGAME_RULESETS); do $(UV) run python -m strategy.value.model fit --ruleset $$r || exit 1; done

# Population response curves from train-partition tournaments, then tables that best-respond to them.
tables-response:
	for r in $(ENDGAME_RULESETS); do $(UV) run python -m strategy.endgame.response collect --ruleset $$r --jobs $(ENDGAME_JOBS) && \
		$(UV) run python -m strategy.endgame.response fit --ruleset $$r || exit 1; done

tables-exploit:
	for m in response capped; do for r in $(ENDGAME_RULESETS); do \
		$(UV) run python -m strategy.tables.exploit --ruleset $$r --mode $$m --jobs $(ENDGAME_JOBS) || exit 1; done; done

# The eight pre-registered comparisons (docs/endgame-response-preregistration.md).
lab-response:
	for c in response capped; do for a in points chips; do for v in "" -shortdeck; do \
		$(UV) run python -m strategy.cli experiment endgame-$$c-vs-$$a$$v $(LAB_JOBS_FLAG) || exit 1; done; done; done

lab-endgame:
	for s in endgame-points-vs-chips endgame-points-vs-chart endgame-points-vs-chips-shortdeck endgame-points-vs-none \
		endgame-points-vs-chips-confirm; do \
		$(UV) run python -m strategy.cli experiment $$s $(LAB_JOBS_FLAG) || exit 1; done

# Baseline policy search (Stage 1 B/D; docs/baseline-policy.md). Long runs: never in CI.
# Runs land in artifacts/search/<run_id>/ and resume where they stopped. nlhe-cem-v2 starts from
# nlhe-cem-v1's state, so run SEARCH_CONFIG=strategy/search/configs/nlhe-cem.json first.
.PHONY: search-baseline search-status package-baseline validate-baseline validate-baseline-variants gate-baseline \
	gate-control bench-policy
SEARCH_CONFIG ?= strategy/search/configs/nlhe-cem-v2.json
SEARCH_RUN ?= artifacts/search/nlhe-cem-v2
SEARCH_JOBS ?= $(shell n=$$(nproc 2>/dev/null || echo 4); echo $$((n > 1 ? n - 1 : 1)))
# The promoted vector (provenance inside). Use BASELINE_CANDIDATE=$(SEARCH_RUN)/incumbent.json for a new run.
BASELINE_CANDIDATE ?= strategy/policy/vectors/nlhe-v2.json
GATE_DECISIONS ?= 50000

search-baseline:
	$(UV) run python -m strategy.search run $(SEARCH_CONFIG) --jobs $(SEARCH_JOBS)

search-status:
	$(UV) run python -m strategy.search status $(SEARCH_RUN)

package-baseline:
	$(UV) run python -m strategy.policy.package --params $(BASELINE_CANDIDATE) --out artifacts/bots/baseline-candidate \
		--name strategy_baseline

# Paired promotion check on the validation partition (in-process; official points; phases.json added).
validate-baseline:
	$(UV) run python -m strategy.search.validate baseline-validation-nlhe --candidate $(BASELINE_CANDIDATE) \
		--jobs $(or $(LAB_JOBS),$(SEARCH_JOBS))

# Rehearsal: the same NLHE-searched vector on short deck and PLO4 (no claims, discovery stage).
validate-baseline-variants:
	$(UV) run python -m strategy.search.validate baseline-rehearsal-variants --candidate $(BASELINE_CANDIDATE) \
		--jobs $(or $(LAB_JOBS),$(SEARCH_JOBS))

# The packaged bot through the real sandbox worker, one table at a time (run on an otherwise idle machine).
gate-baseline: package-baseline
	$(UV) run python -m strategy.policy.gate --bundle artifacts/bots/baseline-candidate --decisions $(GATE_DECISIONS) \
		--out artifacts/gate/baseline-candidate.json

# The same gate with a trivial bot: the machine's own timeout floor.
gate-control:
	$(UV) run python -m strategy.policy.gate --bundle arena/agents/baselines/calling_station.py \
		--decisions $(GATE_DECISIONS) --out artifacts/gate/control-calling-station.json

bench-policy: package-baseline
	$(UV) run python -m strategy.policy.bench --bundle artifacts/bots/baseline-candidate --tables 60 \
		--out artifacts/gate/baseline-bench.json

# Stage 4 MCCFR blueprint (docs: strategy/blueprint/README.md). Training is offline and needs
# numba (--group train); the deployed bot and the evaluation targets do not.
.PHONY: blueprint-states blueprint-train blueprint-pack blueprint-bundles blueprint-exploit blueprint-response \
	blueprint-audit lab-blueprint
BP = artifacts/blueprint
BP_RUN ?= $(BP)/runs/main-v3
BP_STATES = $(BP)/states-practice_nlhe-s1.npz
BP_HOURS ?= 4
BP_THREADS ?= 4
BP_PACK = strategy/tables/data/blueprint-practice_nlhe.npz
BP_MILESTONES = 1000000 3000000 10000000

blueprint-states:
	$(UV) run python -m strategy.blueprint.states --lineups 40 --seed 1 --jobs $(JOBS) --out $(BP_STATES)

# Resumes from the last checkpoint when $(BP_RUN) exists.
blueprint-train:
	$(UV) run $(TRAIN) python -m strategy.blueprint.train --run $(BP_RUN) --states $(BP_STATES) --variant linear \
		--chunk 20000 --interval-pairs 1000000 --linear-intervals 100 --prune-after 5000000 --threads $(BP_THREADS) \
		--producers 3 --hours $(BP_HOURS) --checkpoint-minutes 15 $(foreach m,$(BP_MILESTONES) 20000000 30000000,--milestone $(m))

blueprint-pack:
	$(UV) run python -m strategy.blueprint.policy --run $(BP_RUN) --out $(BP_PACK)

blueprint-bundles:
	$(UV) run python -m strategy.blueprint.package --pack $(BP_PACK) --out $(BP)/bundles/final > /dev/null
	$(UV) run python -m strategy.blueprint.package --pack $(BP_PACK) --leaf capped --out $(BP)/bundles/final-capped > /dev/null
	$(UV) run python -m strategy.blueprint.package --pack $(BP_PACK) --leaf none --out $(BP)/bundles/final-noleaf > /dev/null
	for m in $(BP_MILESTONES); do \
		$(UV) run python -m strategy.blueprint.policy --run $(BP_RUN) --milestone pairs-$$m --out $(BP)/packs/main-v3-$$m.npz && \
		$(UV) run python -m strategy.blueprint.package --pack $(BP)/packs/main-v3-$$m.npz --out $(BP)/bundles/main-v3-$$m > /dev/null \
		|| exit 1; done

blueprint-exploit:
	$(UV) run $(TRAIN) python -m strategy.blueprint.exploit --stacks 10 25 50 100 --out $(BP)/exploit-hu.json

blueprint-response:
	$(UV) run $(TRAIN) python -m strategy.blueprint.response $(foreach m,$(BP_MILESTONES),--pack $(BP)/packs/main-v3-$(m).npz) \
		--pack $(BP_PACK) --pairs 3000000 --checkpoint 1000000 --out $(BP)/response.json

blueprint-audit:
	$(UV) run python -m strategy.blueprint.audit --pack $(BP_PACK) --jobs $(JOBS) --out $(BP)/audit.json

# Needs `make blueprint-bundles`. The sandboxed smoke runs with one job: default parallelism times out.
lab-blueprint:
	for s in blueprint-vs-endgame-points blueprint-capped-vs-endgame-capped blueprint-leaf-capped-vs-points \
		blueprint-vs-endgame-chips blueprint-vs-heuristic blueprint-leaf-ablation \
		blueprint-progress-1m blueprint-progress-10m blueprint-vs-style-aggressive blueprint-vs-style-rock \
		blueprint-vs-style-station blueprint-vs-style-super-aggressive blueprint-vs-style-switcher \
		blueprint-vs-style-weak-tight; do \
		$(UV) run python -m strategy.cli experiment $$s $(LAB_JOBS_FLAG) || exit 1; done
	$(UV) run python -m strategy.cli experiment blueprint-sandbox-smoke --jobs 1

# Opponent exploitation (Stage 6 / M7; strategy/exploit/README.md). Training partition only up to the
# pack; exploit-validate runs the pre-registered specs (docs/opponent-exploitation-preregistration.md).
.PHONY: exploit-collect exploit-models exploit-calibrate exploit-experts exploit-pack exploit-bundles \
	exploit-validate gate-exploit
EX = artifacts/exploit
EX_PACK = strategy/tables/data/exploit-practice_nlhe.npz
EX_EXPERTS = --archetype station --archetype nit --archetype maniac --archetype limper --archetype regular \
	--archetype population
EX_JOBS ?= 7

exploit-collect:
	$(UV) run python -m strategy.exploit.collect --tables 3000 --jobs $(EX_JOBS) --out $(EX)/collect-train.npz

exploit-models:
	$(UV) run python -m strategy.exploit.models --data $(EX)/collect-train.npz --out $(EX)/models

exploit-calibrate:
	$(UV) run python -m strategy.exploit.calibrate --data $(EX)/collect-train.npz --models $(EX)/models \
		--out $(EX)/calibration.json

exploit-experts:
	$(UV) run $(TRAIN) python -m strategy.exploit.experts deals --pairs 3000000 --seed 910 --out $(EX)/deals-train.npz
	$(UV) run $(TRAIN) python -m strategy.exploit.experts deals --pairs 100000 --seed 911 --out $(EX)/deals-eval.npz
	$(UV) run $(TRAIN) python -m strategy.exploit.experts train $(EX_EXPERTS) --p 0.5 \
		--all-archetypes station nit maniac limper regular --pairs 2000000 --deals $(EX)/deals-train.npz \
		--eval-deals $(EX)/deals-eval.npz --models $(EX)/models --threads $(EX_JOBS) --out $(EX)/experts

exploit-pack:
	$(UV) run python -m strategy.exploit.build --experts $(EX)/experts --models $(EX)/models \
		--calibration $(EX)/calibration.json --out $(EX_PACK)

exploit-bundles: blueprint-bundles
	$(UV) run python -m strategy.exploit.package --out $(EX)/bundles/exploit > /dev/null

exploit-validate:
	for s in exploit-vs-blueprint-validation exploit-targeted-validation exploit-robust-equilibrium \
		exploit-robust-deceptive; do \
		$(UV) run python -m strategy.exploit.evaluate run $$s --jobs $(EX_JOBS) || exit 1; done

# The packaged exploiting bot through the real sandbox worker (run on an otherwise idle machine).
gate-exploit: exploit-bundles
	$(UV) run python -m strategy.policy.gate --bundle $(EX)/bundles/exploit --decisions $(GATE_DECISIONS) \
		--out artifacts/gate/exploit.json
