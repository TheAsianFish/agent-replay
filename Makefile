# Targets for later milestones (eval, bench, ui, serve, dashboard lint/type)
# are added with the code they run.

.PHONY: eval-judge-smoke setup lint format type test test-live check cassettes retail-cassettes demo ci-demo \
	bench-sprt flake-demo eval-mutations openapi serve demo-data ui ui-check dist \
	golden record-golden eval-judge eval-judge-objective token-rates \
	data-llmbar eval-llmbar eval-llmbar-dry eval-llmbar-heuristics \
	bench-replay bench-replay-record eval bench regression-pr regression-pr-rehearse walkthrough \
	pydantic-ai-cassettes record-pydantic-ai eval-pydantic-ai eval-pydantic-ai-offline \
	eval-pydantic-ai-arms eval-pydantic-ai-swap eval-verdict-curve bench-scale bench-scale-record bench-upstream

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
	uv run mypy agent_replay examples eval action

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
# and each seeded regression fails it (expected exit 1). In tools-live the live recordings'
# tools return what they returned when recorded, and a tool that drifted fails the suite.
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
	REPLAY_JUDGE_OFFLINE=1 uv run replay ci examples/retail_agent/replay-live.toml --no-store \
		--mode tools-live --fail-on-tool-drift
	uv run replay ci $(RETAIL_SUITE) --no-store --mode tools-live --fail-on-tool-drift \
		--agent eval.mutations.tools:order_details_stale_status > /dev/null; \
		test $$? -eq 1 || { echo "the stale-status tool drifted unnoticed"; exit 1; }

# replay flake offline: the fake model passes every scenario's gate; a seeded flaky
# model is classified passing, flaky or failing per scenario and fails the gate (exit 1).
flake-demo:
	uv run replay flake $(RETAIL_SUITE) --no-store --client examples.retail_agent.fake_model:make_client
	uv run replay flake $(RETAIL_SUITE) --no-store \
		--client examples.retail_agent.flaky_model:make_client; test $$? -eq 1

# Simulate the sequential CI gate over a grid of true pass rates (offline, ~seconds).
bench-sprt:
	uv run python eval/sprt_sim.py --out eval/results/sprt_sim.json

# Recorder overhead per boundary (model, tool, clock, random: paired bare vs recorded calls)
# and full replay vs the recorded live duration of the six live retail cassettes (offline,
# 20 to 35 s). Timings depend on the machine, so the JSON is not byte-stable: a run writes
# to .replay/ (untracked) and leaves the tree clean; `make bench-replay-record` rewrites the
# committed copy that RESULTS.md quotes.
BENCH_REPLAY_OUT ?= .replay/bench_replay.json
bench-replay:
	uv run python eval/bench_replay.py --out $(BENCH_REPLAY_OUT)

bench-replay-record:
	$(MAKE) bench-replay BENCH_REPLAY_OUT=eval/results/bench_replay.json

# Every offline evaluation: the mutation study, the judge against the checks and against the
# golden set's objective labels, the LLMBar heuristic floors, the token-rate check, the
# verdict-stability curve over the committed flake reports. No key.
eval: eval-mutations eval-judge-smoke eval-judge-objective eval-llmbar-heuristics token-rates \
	eval-pydantic-ai-offline eval-verdict-curve

# Every offline benchmark: the sequential gate simulation, replay speed and cassette bytes
# at scale. No key.
bench: bench-sprt bench-replay bench-scale

# M13: cassette bytes for fake-model runs of 5 to 200 turns and for every committed real
# recording, through gzip, bz2 and xz and a shared-prefix encoding, and the decision rule;
# then load, replay and run-store timings (about a minute). The sizes report is committed and
# byte-stable apart from the compressed sizes (recorded with the codec versions); the test
# suite re-checks its raw and prefix-encoded sizes. Timings depend on the machine, so they go
# to .replay/ unless `make bench-scale-record` rewrites the committed copy.
BENCH_SCALE_OUT ?= .replay/bench_scale.json
bench-scale:
	uv run python -m eval.bench_scale --out $(BENCH_SCALE_OUT)

bench-scale-record:
	$(MAKE) bench-scale BENCH_SCALE_OUT=eval/results/bench_scale.json

# The pydantic-ai cassettes (recorded on 2.61.0) replayed in full on pydantic-ai-slim 2.58.0
# to 2.62.0, each in a throwaway venv under .replay/upstream/. Needs network (PyPI).
bench-upstream:
	uv run python -m eval.upstream_matrix --out eval/results/upstream_pydantic_ai.json

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
# replayed in full, tools-live, model-live, model-live with fallback, tools-live with
# fail_on_approximate and tools-live with fail_on_tool_drift; detection rate per check type
# and per configuration. Then everything again with the suite's identify keys, plus the two
# identity mutations, and five correct-but-different paths of the fake model in model-live
# under strict and fallback (offline, about two minutes). The JSON is byte-stable across runs.
eval-mutations:
	uv run python eval/mutation_study.py --out eval/results/mutations.json

# The pydantic-ai travel desk (examples/pydantic_ai_agent): re-record its fake-model
# cassettes (offline, instant).
pydantic-ai-cassettes:
	uv run python -m examples.pydantic_ai_agent.record_all

# Re-record its qwen2.5:7b cassettes through Ollama: up to 5 attempts per task, the first
# that passes the suite's checks is kept, every attempt is logged to
# eval/results/m9_qwen_recording.json. Needs `ollama serve` and `ollama pull qwen2.5:7b`;
# CPU only, several minutes. Locally or from the manual Ollama workflow, never on a PR.
record-pydantic-ai:
	uv run python -m examples.pydantic_ai_agent.record_all --live

# Both of its suites replayed in full (every step exact, 0 approximate); then, when Ollama
# serves qwen2.5:7b, `replay flake` in model-live, 10 runs per scenario, raw report and
# summary (passes, approximate steps per run, wall time, CPU) to eval/results/m9_flake_qwen*.
# Without Ollama the flake part is skipped with a message. The measured run took 17 minutes on 4 CPUs.
eval-pydantic-ai:
	uv run python -m eval.pydantic_ai --runs 10

# The offline half of eval-pydantic-ai (part of `make eval`, so CI runs it).
eval-pydantic-ai-offline:
	uv run python -m eval.pydantic_ai --offline

# M12: the two qwen2.5:7b-recorded scenarios in model-live, 20 runs each under the strict
# and the fallback policy (identify keys on), every run classified by how it ended, into
# eval/results/m12_arms_qwen.json. Needs Ollama serving qwen2.5:7b (skipped without it);
# CPU only, about an hour on 4 CPUs. Locally or from the manual Ollama workflow, never on a PR.
eval-pydantic-ai-arms:
	uv run python -m eval.pydantic_ai --arms strict,fallback --runs 20

# M15: the same two scenarios with the live side changed: qwen2.5:3b under strict and
# fallback, qwen2.5:7b without the prompt's guard line, the qwen2.5:7b control (20 runs
# each), and 10 runs of the booking at temperature 0 after probe calls of Ollama's
# /v1/messages, into eval/results/m15_swap_qwen.json (and the diff of the first failed 3b
# run beside it). Needs Ollama serving qwen2.5:7b and qwen2.5:3b (skipped without them);
# CPU only. Locally or from the manual Ollama workflow, never on a PR.
eval-pydantic-ai-swap:
	uv run python -m eval.pydantic_ai --swap

# M12: how often a subset of n runs gives the all-runs fixed-threshold verdict, for every
# committed flake report (seeded draws without replacement; offline, byte-stable).
eval-verdict-curve:
	uv run python -m eval.verdict_curve

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

# Question 3 on a real pull request: a branch that points the retail suite at a seeded
# mutant, a PR whose replay workflow runs the Action with dry-run off, the evidence
# (check run, wall time, failed scenarios, comment permalink and screenshot) captured
# into eval/results/regression_pr/, then the PR closed. Needs `gh auth login` as a
# maintainer and origin/main to have .github/workflows/replay.yml (FROM= overrides).
# The rehearsal does the same edit and replay ci locally, without GitHub.
MUTANT ?= wrong_write_argument
regression-pr:
	uv run python -m eval.regression_pr --mutant $(MUTANT) $(if $(FROM),--from $(FROM))
regression-pr-rehearse:
	uv run python -m eval.regression_pr --rehearse --mutant $(MUTANT)

# The README's dashboard walkthrough: seeds the demo store, serves the built dashboard,
# records runs, run detail, aligned diff and flakiness with Playwright, and converts
# the video with ffmpeg into docs/walkthrough/ (mp4 and GIF; needs ffmpeg, ~2 min).
walkthrough:
	scripts/walkthrough.sh

# Build the dashboard into agent_replay/server/static, where replay serve finds it.
ui:
	npm --prefix dashboard ci --no-audit --no-fund
	npm --prefix dashboard run build

# The release artifacts: the dashboard built into the package, then the wheel and sdist in
# dist/, each installed in a fresh venv outside the repo and smoke-tested (scripts/check_wheel.sh;
# needs python3.12 and the package index, ~1 min), then `twine check` on their metadata.
dist: ui
	rm -rf dist
	uv build
	scripts/check_wheel.sh --expect-dashboard dist
	uvx twine check --strict dist/*

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

# Offline: re-derive the calibrate cost estimate's constants (characters per input token,
# output tokens per call) from the committed retail verdicts and check them (exit 1 if stale).
token-rates:
	uv run python eval/token_rates.py

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
