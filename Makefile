# Targets for later milestones (eval, bench, ui, serve, dashboard lint/type)
# are added with the code they run.

.PHONY: setup lint format type test test-live check cassettes demo

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
	uv run mypy agent_replay examples

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

# The quickstart: replay passes, then the mutant fails (expected exit 1).
demo:
	uv run replay run examples/toy_agent/cassettes/refund_eligible.jsonl
	uv run replay run examples/toy_agent/cassettes/refund_eligible.jsonl \
		--agent examples.toy_agent.mutants:first_tool_only; test $$? -eq 1
