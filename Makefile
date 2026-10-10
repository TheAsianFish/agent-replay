# Targets for later milestones (eval, bench, ui, serve, dashboard lint/type)
# are added with the code they run.

.PHONY: eval-judge-smoke setup lint format type test test-live check cassettes retail-cassettes demo ci-demo \
	bench-sprt flake-demo eval-mutations openapi serve

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
