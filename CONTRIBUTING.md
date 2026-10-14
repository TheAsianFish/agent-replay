# Contributing to agent-replay

Read PLAN.md first; it is the single source of truth. Tick a milestone only
when its definition of done is met and CI is green.

## What this is

`agent-replay`: record an LLM agent's run (model calls, tool calls, clock,
randomness) into a JSONL cassette, replay it with chosen cut points, assert on
the trajectory, gate CI statistically on flakiness, and calibrate the LLM judge
against human labels. Python SDK + CLI (`replay`), GitHub Action, FastAPI +
React dashboard. See PLAN.md "Architecture" for the component map.

## Layout (target)

```
agent_replay/        Python package (record, cassette, replay, checks, judge,
                     stats, flake, calibrate, diff, store, cli)
agent_replay/migrations/  Alembic revisions for the run store (applied on open)
agent_replay/server/ FastAPI app over the store (`replay serve`); the dashboard
                     build lands in agent_replay/server/static (untracked)
dashboard/           React 19 + TypeScript strict (Vite, TanStack Query, Tailwind)
examples/            retail_agent (τ-bench retail subset, MIT, attributed) and a toy agent
eval/                datasets, mutation suite, SPRT simulation, eval scripts
tests/               offline, deterministic; no network ever
docs/                screenshots, architecture notes, RELEASE.md
action.yml           composite GitHub Action
```

## Commands

Python is managed with `uv` (Python 3.12). Node 22 for the dashboard.

```
make setup        uv sync --all-extras
make lint         ruff check . && ruff format --check .
make format       ruff check --fix . && ruff format .
make type         mypy --strict agent_replay examples eval
make test         pytest -q (offline; sockets blocked; fake model + committed cassettes)
make test-live    pytest -m live (needs ANTHROPIC_API_KEY; records with claude-opus-5-5)
make check        lint + type + test (what CI runs)
make cassettes    re-record the toy agent's committed cassettes (Anthropic and Ollama fakes)
make retail-cassettes  re-record the retail example's cassettes with its fake model
make demo         the toy quickstart: replay passes, the mutant fails
make ci-demo      replay ci on the retail suite in every mode; each seeded regression must exit 1;
                  the live suite against its cost baseline, judged offline from the committed cache
make bench-sprt   simulate the sequential CI gate over true pass rates (offline; core skipped until written)
make flake-demo   replay flake offline: the fake model passes; a seeded flaky model must exit 1
make eval-judge-smoke  judge vs deterministic checks on 24 retail runs, from the committed verdict cache
make eval-mutations    the mutation study: 32 retail mutations + 5 benign refactors in five replay
                       configurations; detection per check type and mode (offline, byte-stable JSON)
make demo-data    real replay ci / replay flake runs of the retail example into .replay/demo.db
make openapi      regenerate dashboard/openapi.json (then `npm run gen:types` in dashboard/)
make ui           npm ci + build the dashboard into agent_replay/server/static
make ui-check     dashboard: check:types, typecheck, lint, vitest
make serve        replay serve over $REPLAY_DB_URL
docker compose up --build   PostgreSQL + demo seed + dashboard on :8000
```

PostgreSQL tests are marked `postgres` and run with `REPLAY_TEST_PG_URL` set to a
disposable database (`uv run pytest -m postgres`); they reset its schema. Playwright:
`npm run e2e` in `dashboard/` against a served build (`DASHBOARD_URL`, default :8000);
`npm run screenshots` rewrites `docs/screenshots/`.

A new store column or table needs an Alembic revision in `agent_replay/migrations/versions/`
(hand-written, both backends); `tests/test_migrations.py` fails if models and migrations
disagree.

Planned with later milestones (not in the Makefile yet): `make eval`, `make bench`,
`make eval-judge`, `make eval-llmbar`, `make bench-flake` (key-requiring ones print a
cost estimate first).

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
  `REPLAY_JUDGE_MIN_KAPPA`, `REPLAY_DB_URL`, `REPLAY_STATS_IMPL`, …),
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
