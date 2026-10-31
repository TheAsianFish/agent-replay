# Contributing to agent-replay

Read PLAN.md first; it is the single source of truth. Tick a milestone only
when its definition of done is met and CI is green.

## What this is

`agent-replay`: record an LLM agent's run (model calls, tool calls, clock,
randomness) into a JSONL cassette, replay it with chosen cut points, assert on
the trajectory, gate CI statistically on flakiness, and calibrate the LLM judge
against human labels. Python SDK + CLI (`replay`), GitHub Action, FastAPI +
React dashboard. See PLAN.md "Architecture" for the component map.

## Layout

```
agent_replay/        Python package (record, cassette, replay, checks, judge,
                     stats, flake, calibrate, diff, store, cli)
agent_replay/migrations/  Alembic revisions for the run store (applied on open)
agent_replay/server/ FastAPI app over the store (`replay serve`); the dashboard
                     build lands in agent_replay/server/static (untracked)
dashboard/           React 19 + TypeScript strict (Vite, TanStack Query, Tailwind)
examples/            retail_agent (τ-bench retail subset, MIT, attributed, plus one hand-written
                     error task), a toy agent, and
                     pydantic_ai_agent (a pydantic-ai travel desk; fake and qwen2.5:7b cassettes)
eval/                datasets, mutation suite, SPRT simulation, eval scripts
tests/               offline, deterministic; no network ever
docs/                screenshots and the dashboard walkthrough (docs/walkthrough, `make walkthrough`)
action.yml           composite GitHub Action; its logic in action/replay_action.py (stdlib)
examples/github-workflow/  the consumer workflow users copy
scripts/             check_wheel.sh (clean-venv install check), release_notes.py
deploy/              fly.toml for the optional hosted demo (Dockerfile target `demo`)
RELEASE.md           every release step and the settings the maintainer adds
CHANGELOG.md         Keep a Changelog; the release build needs a dated section
```

## Commands

Python is managed with `uv` (Python 3.12). Node 22 for the dashboard.

```
make setup        uv sync --all-extras
make lint         ruff check . && ruff format --check .
make format       ruff check --fix . && ruff format .
make type         mypy --strict agent_replay examples eval action
make test         pytest -q (offline; sockets blocked; fake model + committed cassettes)
make test-live    pytest -m live (needs ANTHROPIC_API_KEY; records with claude-opus-5-5)
make check        lint + type + test (what CI runs)
make cassettes    re-record the toy agent's committed cassettes (Anthropic and Ollama fakes)
make retail-cassettes  re-record the retail example's cassettes with its fake model
make demo         the toy quickstart: replay passes, the mutant fails
make ci-demo      replay ci on the retail suite in every mode; each seeded regression must exit 1;
                  the live suite against its cost baseline, judged offline from the committed cache;
                  the live suite in tools-live with --fail-on-tool-drift (0 drift), a drifting tool exits 1
make bench-sprt   simulate the sequential CI gate over true pass rates (offline; core skipped until written)
make flake-demo   replay flake offline: the fake model passes; a seeded flaky model must exit 1
make eval-judge-smoke  judge vs deterministic checks on 24 retail runs, from the committed verdict cache
make eval-mutations    the mutation study: 32 retail mutations + 5 benign refactors in seven replay
                       configurations (the seventh, model-live+synth, since M17); detection per check type and mode; again with the suite's
                       identify keys plus 2 identity mutations; 5 correct-but-different paths in
                       model-live under strict, fallback and synth; the hand-written error task
                       (M18) unmutated and under every agent mutation (offline, byte-stable JSON)
make golden       rebuild eval/golden/retail from the live recordings and mutants (offline; 21 items from 24 runs)
make record-golden  D1: 120 model-live retail runs (Opus and Haiku) into eval/golden/retail-d1
                  (needs ANTHROPIC_API_KEY; ~$2.19 estimated)
make eval-judge   replay calibrate vs human labels on the retail golden set: kappa, self-consistency,
                  probes (needs a key and `replay label` answers; prints its estimate first, ~$1.30)
make eval-judge-objective  the judge vs the golden set's objective labels, offline from the committed
                  cache (informative; such a report never gates)
make token-rates  re-derive the calibrate cost estimate's constants from the committed verdicts
                  (offline; exits 1 if they drift)
make data-llmbar  re-download LLMBar at the pinned commit into eval/data/llmbar
make eval-llmbar  the real judge on LLMBar (needs a key; 838 calls, ~$12.73 dry-run estimate)
make eval-llmbar-dry  the calls eval-llmbar still needs and their estimated cost (offline)
make eval-llmbar-heuristics  LLMBar floors: first / longer / shorter-output judges (offline)
make pydantic-ai-cassettes  re-record the pydantic-ai example's fake-model cassettes
make record-pydantic-ai  re-record its qwen2.5:7b cassettes through Ollama (local, CPU, minutes)
make eval-pydantic-ai  replay ci in full on both of its suites; with Ollama, replay flake in
                  model-live on qwen2.5:7b (10 runs per scenario) into eval/results/m9_flake_qwen*
make eval-pydantic-ai-arms  M12: its qwen-recorded scenarios under strict and fallback, 20 runs each
make eval-pydantic-ai-swap  M15: the same with the live side changed (qwen2.5:3b, the prompt without
                  its guard line, temperature 0, a 7b control) into eval/results/m15_swap_qwen*
make record-pydantic-ai-chain  M16: the rebooking chain's qwen2.5:7b recording attempts (5, logged)
make eval-pydantic-ai-long  M16: the rebooking chain (9 sequential tool calls) under strict and fallback,
                  20 runs each, survival curve and clustering into eval/results/m16_long_qwen.json
make eval-pydantic-ai-no-guard  M17: qwen2.5:7b without the prompt's guard line under fallback and synth,
                  20 runs per scenario each, causes and answer figures into eval/results/m17_no_guard_qwen.json
make record-retail-error-qwen  M18: the retail error task recorded on qwen2.5:7b (up to 3 attempts,
                  each model call's input tokens logged to eval/results/m18_qwen_recording.json)
make eval-retail-error-qwen  M18: an is_error probe of Ollama, then the error task in model-live under
                  fallback, 10 runs unmutated and 10 with drops_error_flag, into eval/results/m18_error_qwen.json
make record-retail-lookup-qwen  M19: lookup_latest_order_total recorded on qwen2.5:7b (up to 3 attempts,
                  logged to eval/results/m19_qwen_recording.json)
make eval-retail-mutations-qwen  M19: a tool_use_id probe, then the lookup task in model-live under fallback,
                  10 runs unmutated and 5 of each of eight agent mutations, every prompt's size read from
                  the server log ($(OLLAMA_LOG), default /tmp/ollama.log; start `ollama serve` with
                  OLLAMA_CONTEXT_LENGTH=8192), into eval/results/m19_retail_mutations_qwen.json
make eval-retail-qwen-offline  the qwen retail recordings replayed in full, 0 approximate (in make eval)
make demo-data    real replay ci / replay flake runs of the retail example into .replay/demo.db
make openapi      regenerate dashboard/openapi.json (then `npm run gen:types` in dashboard/)
make ui           npm ci + build the dashboard into agent_replay/server/static
make ui-check     dashboard: check:types, typecheck, lint, vitest
make serve        replay serve over $REPLAY_DB_URL
make eval         every offline evaluation (mutations, judge smoke + objective, LLMBar floors, token rates)
make bench        every offline benchmark (bench-sprt, bench-replay, bench-scale)
make bench-replay recorder overhead per boundary and full replay vs live duration (to .replay/; not byte-stable)
make bench-replay-record  the same, rewriting the committed eval/results/bench_replay.json that RESULTS.md quotes
make bench-scale  cassette bytes for fake-model runs of 5 to 200 turns and every committed real recording,
                  through gzip, bz2, xz and a shared-prefix encoding, the M13 decision, then load, replay and
                  store timings (to .replay/); the sizes report is committed and its raw and prefix sizes are
                  re-checked by tests/test_bench_scale.py (`make bench-scale-record` rewrites the timings)
make bench-upstream  the pydantic-ai cassettes on pydantic-ai-slim 2.58.0 to 2.62.0, each in a venv (network)
make walkthrough  record docs/walkthrough (mp4 + GIF) from make demo-data with Playwright and ffmpeg
make regression-pr  open a PR that seeds a regression, capture its failing replay check and comment
                  into eval/results/regression_pr/ and docs/screenshots/pr-comment.png, close it
                  (needs `gh auth login` as a maintainer; `make regression-pr-rehearse` is offline)
make dist         dashboard + uv build; installs the wheel and sdist in clean venvs and smoke-tests them
docker compose up --build   PostgreSQL + demo seed + dashboard on :8000
```

PostgreSQL tests are marked `postgres` and run with `REPLAY_TEST_PG_URL` set to a
disposable database (`uv run pytest -m postgres`); they reset its schema. Playwright:
`npm run e2e` in `dashboard/` against a served build (`DASHBOARD_URL`, default :8000);
`npm run screenshots` rewrites `docs/screenshots/`.

A new store column or table needs an Alembic revision in `agent_replay/migrations/versions/`
(hand-written, both backends); `tests/test_migrations.py` fails if models and migrations
disagree.

Planned (not in the Makefile yet): `make bench-flake` (D3; needs a key and prints a cost
estimate first).

The Action is tested four ways: `tests/test_action.py` (offline, a fake HTTP layer), the
`action` CI job (`uses: ./`, dry run on the retail suite), for real by `replay.yml` on PRs
that change the retail agent (`make regression-pr` opens one built to fail), and locally with
[act](https://github.com/nektos/act): `act pull_request -j action -W .github/workflows/ci.yml
-P ubuntu-latest=catthehacker/ubuntu:act-latest --artifact-server-path /tmp/act-artifacts`
(`-j replay -W .github/workflows/replay.yml` runs the real, failing job on a seeded suite).
The local-model runs (`make record-pydantic-ai`, `make eval-pydantic-ai`, `make eval-retail-error-qwen`, `make eval-retail-mutations-qwen`) need `ollama serve`
and `ollama pull qwen2.5:7b` (`make eval-pydantic-ai-swap` also `qwen2.5:3b`); on GitHub they run only from the manual `ollama.yml` workflow.
Release steps are in RELEASE.md; never publish by hand.

Until a target exists, say so in the PR rather than inventing it.

## Quality gates (CI must pass all)

- `ruff check` and `ruff format --check` clean; `mypy --strict` clean for
  `agent_replay/` (the server included); TypeScript `strict: true`, `eslint` clean.
- `pytest` fully offline and deterministic: no network, no API key, no clock
  dependence (use the replay clock helpers), fixed seeds. Mock at the SDK
  wrapper boundary, not with HTTP mocking: `anthropic` 1.x uses `httpx2`, so
  `respx`/`httpx` mocks do not intercept it.
- Unit + integration tests for every milestone's features; Hypothesis property
  tests for `stats` against scikit-learn / statsmodels oracles (dev deps only;
  the package itself keeps `stats` stdlib-only).
- No number in README.md or RESULTS.md that was not produced by a command in
  this repo. Record dataset, method, command and raw output in RESULTS.md. Key-
  requiring measurements go under "To measure" with the exact command.
- No secrets, no personal data, no resume content. Config via env vars with the
  `REPLAY_` prefix (`REPLAY_JUDGE_PROVIDER`, `REPLAY_JUDGE_MODEL`,
  `REPLAY_JUDGE_MIN_KAPPA`, `REPLAY_JUDGE_MIN_ITEMS`, `REPLAY_DB_URL`, `REPLAY_STATS_IMPL`, …),
  documented in README.
- Structured logging (structlog) in the server and CLI; clear error messages
  that name the fix.

## Conventions

- Small, reviewable commits in imperative mood ("Add cursor fallback to replay
  engine"), one logical step per commit, each passing the quality gates.
- Typed everything: pydantic 2 models for cassette and reports; `Protocol`s for
  provider adapters; TypeScript types generated from the OpenAPI schema.
- Cassette schema is versioned (`schema_version`); never change v1 fields,
  add v2 with a migration.

## Claude API facts to rely on (verified 2026-10-08)

- Default model `claude-opus-5-5` via the official `anthropic` SDK (1.12.x,
  depends on `httpx2`, Python ≥ 3.10). Judge and example agent both use it;
  Ollama (`ollama` 0.6.x) is the offline fallback provider.
- Thinking cannot be disabled on Opus 5.5; control cost with
  `output_config={"effort": "low"|"medium"|...}` (default `medium`). Do not send
  `budget_tokens` or `thinking: {type: "disabled"}` (400).
- `tool_choice: {type: "any"}` / `{type: "tool"}` return 400 on Opus 5.5. Use
  `auto`, `strict: true` on tools, or structured outputs
  (`output_config={"format": {"type": "json_schema", ...}}` or
  `client.messages.parse(..., output_format=PydanticModel)`). The judge uses
  structured outputs.
- No assistant prefill. Parse tool inputs with `json.loads`, never string-match.
- Return all `tool_result` blocks for one assistant turn in a single user
  message, `is_error: true` for failures.
- Put stable content (rubric, tool list) first for prompt caching; verify with
  `usage.cache_read_input_tokens`.
- Thinking blocks are bound to the producing model and conversation. Replay
  never sends recorded assistant turns back to a live model.
- Pricing used for cost gates lives in `agent_replay/pricing.py` with the date
  it was captured; Opus 5.5 is $4 / $20 per MTok (input / output), cache reads
  $0.20.

## Prior art to stay differentiated from

`pytest-agentreplay`, `replayproof`, `chronicle` (cut-point replay,
arXiv:2609.20625), `pytest-agent-trace`, `agentprobe`. Replay's differentiators
are measured judge calibration, statistical flakiness gating, the mutation
study, and the dashboard with aligned trace diffs. Cite prior art where an idea
is borrowed.
