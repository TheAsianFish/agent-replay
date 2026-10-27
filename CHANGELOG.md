# Changelog

All notable changes to `agent-replay` are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/). The cassette format is versioned
separately (`schema_version` in each cassette header); readers keep accepting
every older version.

## [Unreleased]

### Added

- Records and replays `client.beta.messages.create` (plain and `stream=True`) and
  `client.beta.messages.stream(...)` for `Anthropic` and `AsyncAnthropic`, under the
  call name `anthropic.beta.messages.create`. `betas` is part of the fingerprint;
  replays return `BetaMessage` and the beta stream event types. This is the surface
  pydantic-ai calls, so a pydantic-ai agent records with
  `AnthropicProvider(anthropic_client=replay.wrap(AsyncAnthropic(...)))`.
- `examples/pydantic_ai_agent`: a tool-using pydantic-ai agent with fake-model and
  qwen2.5:7b (Ollama) cassettes; `make eval-pydantic-ai` replays them and, with
  Ollama running, gates them with `replay flake` in `model-live`.

### Changed

- A run whose model has no price is reported as `unpriced` (or `$x + N unpriced
  calls`), not `$0.0000`.
- `count_tokens`, message batches and the beta `parse` / `tool_runner` raise a
  `ReplayError` that names them inside a session instead of going live.

### Fixed

- `client.with_options(...)` and `client.copy(...)` on a wrapped client return a
  wrapped client; they used to return the bare client, so calls through the copy
  (pydantic-ai makes them when `max_retries` is set) went live unrecorded.
- Replaying with no client gave the offline stand-in no `base_url`, which
  pydantic-ai reads on every response.

## [1.0.0] - 2026-10-16

The first release: the SDK, the `replay` CLI, the GitHub Action and the
self-hosted dashboard.

### Recording and replay

- Record a real agent run into a JSONL cassette: every model call (Anthropic SDK
  sync, async, streaming and `messages.parse`; Ollama), every `@tool` call, clock
  reads and random draws, with usage, cost and latency. Cassette schema v2, with
  v1 cassettes read through a migration.
- Replay with cut points: `full` (everything from the cassette, offline and
  bit-stable), `model-live` (the model runs live against frozen tools) and
  `tools-live` (tools run live against recorded model replies).
- Fingerprint matching with a cursor fallback that flags `approximate` steps;
  unrecorded-call policies `strict`, `fallback` and `synth` (stubs generated from
  a tool's JSON schema).

### Checks and CI

- Nine checks: `tool_sequence`, `required_tools`, `forbidden_tools`,
  `max_tool_calls`, `arg_schema`, `output_schema`, `rule`, `budget` and `judge`.
- `replay ci`: a TOML suite of scenarios, exit codes 0 / 1 / 2, JSON and Markdown
  reports, the Markdown summary written to the GitHub step summary.
- Cost and latency baselines (`replay baseline`) with relative budget limits;
  a dated pricing table (`agent_replay/pricing.py`, overridable with
  `REPLAY_PRICES_FILE`).
- The composite GitHub Action (`uses: TheAsianFish/agent-replay@v1`): runs
  `replay ci`, upserts one sticky PR comment per suite, uploads the reports as an
  artifact and fails the job on a regression; a dry-run mode reports without
  failing or commenting.

### Judge and calibration

- A rubric-driven LLM judge (`claude-opus-5-5` through structured outputs, Ollama
  fallback) with a content-addressed verdict cache and an offline mode.
- Judge criteria are advisory until calibrated: a criterion can fail a build only
  with a calibration report against human labels, computed with the `core`
  statistics, over at least `REPLAY_JUDGE_MIN_ITEMS` items, whose kappa lower
  bound clears `REPLAY_JUDGE_MIN_KAPPA`.
- Golden sets (`replay golden`), blind labelling (`replay label`) and
  `replay calibrate`: per-criterion kappa with a bootstrap interval, raw
  agreement, the majority baseline, confusion matrices, self-consistency and
  position, verbosity and self-preference probes, with a cost estimate printed
  before any call.
- A pairwise judge and an LLMBar harness with offline heuristic floors.

### Flakiness and statistics

- `replay flake`: repeated `model-live` replays with bounded concurrency and rate
  limiting, a sequential (SPRT) or fixed-sample gate on the pass rate, and
  passing / flaky / failing classification per scenario.
- A statistics interface (Cohen's kappa, bootstrap CI, Wilson interval, SPRT)
  with a naive baseline implementation, selected by `REPLAY_STATS_IMPL`.

### Store and dashboard

- A run store on SQLite or PostgreSQL with Alembic migrations applied on open.
- `replay diff`: an aligned trace diff (Needleman-Wunsch over steps).
- `replay serve`: a read-only FastAPI API and a React dashboard (runs, trace
  viewer, aligned diff, suites, flakiness trends, calibration reports), bundled
  in the wheel; `docker compose up` runs it on PostgreSQL with demo data.

### Evaluation

- A mutation study (32 injected regressions, 5 benign refactors), a simulation of
  the sequential gate, the judge against objective labels, LLMBar floors and a
  recorder-overhead and replay-speed benchmark, all offline. Results and the
  commands that need an API key are in `RESULTS.md`.

### Known limitations

- `agent_replay/stats.py` (the `core` statistics) is a specification with
  oracle tests; until it is implemented, calibration reports show raw agreement
  in the kappa column and no judge criterion can gate.
- The dashboard API has no authentication; run it on a trusted network.

[Unreleased]: https://github.com/TheAsianFish/agent-replay/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/TheAsianFish/agent-replay/releases/tag/v1.0.0
