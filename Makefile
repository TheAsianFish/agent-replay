# Targets for later milestones (eval, bench, ui, serve, dashboard lint/type)
# are added with the code they run.

.PHONY: eval-judge-smoke setup lint format type test test-live check cassettes retail-cassettes demo ci-demo \
	bench-sprt flake-demo eval-mutations openapi serve demo-data ui ui-check \
	golden record-golden eval-judge eval-judge-objective \
	data-llmbar eval-llmbar eval-llmbar-dry eval-llmbar-heuristics

TOY_RECORD = uv run replay record examples.toy_agent.agent:run \
	--client examples.toy_agent.fake_model:make_client --force

setup:
	uv sync --all-extras

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

type:
	uv run mypy agent_replay examples eval

test:
	uv run pytest -q

# Calls the real Anthropic API; needs ANTHROPIC_API_KEY. Costs a few cents.
test-live:
	uv run pytest -q -m live

check: lint type test

# Re-record the toy agent's committed cassettes with the deterministic fake model.
cassettes:
	$(TOY_RECORD) -t "Order A1001 arrived damaged. Can I get a refund?" \
		-o examples/toy_agent/cassettes/refund_eligible.jsonl
	$(TOY_RECORD) -t "I'd like a refund for order A1002, please." \
		-o examples/toy_agent/cassettes/refund_denied.jsonl
	uv run replay record examples.toy_agent.ollama_agent:run \
		--client examples.toy_agent.ollama_fake:make_client --force \
		-t "Order A1001 arrived damaged. Can I get a refund?" \
		-o examples/toy_agent/cassettes/ollama_refund_eligible.jsonl

# Re-record the retail agent's committed cassettes (one per task) with the fake model.
retail-cassettes:
	uv run python -m examples.retail_agent.record_all

# The quickstart: replay passes, then the mutant fails (expected exit 1).
demo:
	uv run replay run examples/toy_agent/cassettes/refund_eligible.jsonl
	uv run replay run examples/toy_agent/cassettes/refund_eligible.jsonl \
		--agent examples.toy_agent.mutants:first_tool_only; test $$? -eq 1

RETAIL_SUITE = examples/retail_agent/replay.toml

# replay ci on the retail suite: passes in every mode (the fake model plays the
# live one), the live recordings stay within their cost and latency baseline,
# and each seeded regression fails it (expected exit 1).
ci-demo:
	uv run replay ci $(RETAIL_SUITE) --no-store
	REPLAY_JUDGE_OFFLINE=1 uv run replay ci examples/retail_agent/replay-live.toml --no-store
	uv run replay ci $(RETAIL_SUITE) --no-store --mode tools-live
	uv run replay ci $(RETAIL_SUITE) --no-store --mode model-live \
		--client examples.retail_agent.fake_model:make_client
	for mutant in wrong_write_argument skips_order_lookup first_tool_only; do \
		uv run replay ci $(RETAIL_SUITE) --no-store \
			--agent examples.retail_agent.mutants:$$mutant > /dev/null; \
		test $$? -eq 1 || { echo "mutant $$mutant was not caught"; exit 1; }; \
	done

# replay flake offline: the fake model passes every scenario's gate; a seeded flaky
# model is classified passing, flaky or failing per scenario and fails the gate (exit 1).
flake-demo:
	uv run replay flake $(RETAIL_SUITE) --no-store --client examples.retail_agent.fake_model:make_client
	uv run replay flake $(RETAIL_SUITE) --no-store \
		--client examples.retail_agent.flaky_model:make_client; test $$? -eq 1

# Simulate the sequential CI gate over a grid of true pass rates (offline, ~seconds).
bench-sprt:
	uv run python eval/sprt_sim.py --out eval/results/sprt_sim.json

# The judge against the deterministic checks on the live retail recordings and the
# seeded mutants. Offline from the committed verdict cache; drop --offline to judge
# cache misses with claude-opus-5-5 (needs ANTHROPIC_API_KEY, about $0.015 a run).
eval-judge-smoke:
	uv run python eval/judge_smoke.py --offline --out eval/results/judge_smoke.json

# Calibrate the judge against human labels on the retail golden set: kappa per criterion
# with a bootstrap CI, self-consistency over 3 verdicts per item, and the position,
# verbosity and self-preference probes. Needs human labels (`replay label`), an
# ANTHROPIC_API_KEY for uncached verdicts, and REPLAY_STATS_IMPL=core for a report that
# can gate. Prints its call count and cost estimate first (84 calls, about $1.30, on the
# 21 retail items: their first verdicts are already in the committed cache); add --dry-run
# to stop there. New verdicts land in the same cache, so commit it with the report.
eval-judge:
	uv run replay calibrate eval/golden/retail --rubric examples/retail_agent/rubric.yaml \
		--reference human --repeats 3 --cache-dir examples/retail_agent/judge-cache \
		-o eval/results/calibration_human.json

# The judge against the objective (deterministic-check) labels of the retail golden set,
# offline from the committed verdict cache. Informative only: such a report never gates.
# created_at is the newest cached verdict used, so the JSON is byte-stable across runs.
eval-judge-objective:
	REPLAY_JUDGE_OFFLINE=1 uv run replay calibrate eval/golden/retail \
		--rubric examples/retail_agent/rubric.yaml --reference objective --repeats 1 \
		--no-probes --cache-dir examples/retail_agent/judge-cache \
		-o eval/results/calibration_objective.json --no-store

# The mutation study: 32 known-bad variants and 5 benign refactors of the retail agent,
# replayed in full, tools-live, model-live, model-live with fallback and tools-live with
# fail_on_approximate; detection rate per check type and per configuration (offline,
# ~25 s). The JSON is byte-stable across runs.
eval-mutations:
	uv run python eval/mutation_study.py --out eval/results/mutations.json

# Regenerate the API schema the dashboard's TypeScript types come from.
openapi:
	uv run python -m agent_replay.server.openapi dashboard/openapi.json

# The dashboard and its API over the local run store (or $REPLAY_DB_URL) on :8000.
serve:
	uv run replay serve

# The dashboard's demo data: real replay ci / replay flake runs of the retail example
# (~25 s, ~40 MB) into $(DEMO_DB). `make serve REPLAY_DB_URL=$(DEMO_DB)` to browse it.
DEMO_DB ?= sqlite:///.replay/demo.db
demo-data:
	uv run python -m examples.demo_data --db $(DEMO_DB)

# Build the dashboard into agent_replay/server/static, where replay serve finds it.
ui:
	npm --prefix dashboard ci --no-audit --no-fund
	npm --prefix dashboard run build

# Dashboard checks: generated API types current, tsc, eslint, vitest.
ui-check:
	npm --prefix dashboard run check:types
	npm --prefix dashboard run typecheck
	npm --prefix dashboard run lint
	npm --prefix dashboard test

# The retail golden set for judge calibration: the six live recordings replayed with
# the real agent and the three seeded mutants (24 runs, 21 distinct transcripts), with
# objective labels from the suite's deterministic checks. Offline and judge-free (~10 s);
# re-running changes nothing. Label it by hand with `uv run replay label eval/golden/retail
# --rubric examples/retail_agent/rubric.yaml --labeller NAME`.
GOLDEN_RETAIL = eval/golden/retail
golden:
	uv run replay golden add examples/retail_agent/replay-live.toml --out $(GOLDEN_RETAIL)
	for mutant in wrong_write_argument skips_order_lookup first_tool_only; do \
		uv run replay golden add examples/retail_agent/replay-live.toml --out $(GOLDEN_RETAIL) \
			--variant $$mutant --agent examples.retail_agent.mutants:$$mutant || exit 1; \
	done

# D1, the larger golden set: 10 model-live runs per retail task with claude-opus-5-5 and
# 10 with claude-haiku-5-5 (120 runs), saved as cassettes and built into golden items with
# objective labels. Needs ANTHROPIC_API_KEY; about $2.19 at the measured cost per run (the
# estimate prints first). Resumes: recorded cassettes are kept. Then label it with
# `uv run replay label eval/golden/retail-d1 --rubric examples/retail_agent/rubric.yaml`.
record-golden:
	uv run python eval/record_golden.py --runs 10 --out eval/golden/retail-d1

# Re-download LLMBar (princeton-nlp/LLMBar, MIT) at the pinned commit into eval/data/llmbar
# and verify every file's sha256 (network; stdlib only). `--verify` checks the vendored copy.
data-llmbar:
	python3 -I eval/llmbar_fetch.py

# The judge on LLMBar's 419 Natural + Adversarial pairs in both orders: accuracy and
# position-bias rate. 838 calls to claude-opus-5-5 at effort low; needs ANTHROPIC_API_KEY,
# about $13 (dry-run estimate; stops past $25). Verdicts are cached in eval/llmbar-cache,
# so an interrupted run resumes.
eval-llmbar:
	uv run python eval/llmbar.py --max-cost-usd 25 --out eval/results/llmbar.json

# Offline: the calls eval-llmbar still needs after cache hits and their estimated cost.
eval-llmbar-dry:
	uv run python eval/llmbar.py --dry-run

# Offline: three heuristic judges on LLMBar (always the first output, the longer, the
# shorter), the floor any real judge's numbers are read against (~5 s, no key).
eval-llmbar-heuristics:
	for judge in first longer shorter; do \
		cache=$$(mktemp -d) && \
		uv run python eval/llmbar.py --provider fake-$$judge --cache-dir $$cache \
			--out eval/results/llmbar_heuristic_$$judge.json > /dev/null; \
		status=$$?; rm -rf $$cache; test $$status -eq 0 || exit 1; \
	done
