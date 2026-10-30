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
- `fail_on_tool_drift` in suites (and `replay ci --fail-on-tool-drift`): in `tools-live`,
  every live tool result is compared with the one recorded for the same call, and a
  difference fails the scenario as `no_tool_drift`, naming the first differing path.
  Per-tool `ignore_fields` skip values that legitimately vary.
- The mutation study's sixth configuration, `tools-live+drift`.
- `identify` in suites (`[defaults.identify]`, per scenario, or `replay(identify=...)`):
  per tool, the arguments that name the entity a call reads or writes. A served call
  only falls back to a recording of the same entity; otherwise it is unrecorded and the
  policy decides. With `fail_on_tool_drift`, a live call whose identity no recording has
  is drift (`identity_unrecorded`). The retail and pydantic-ai suites declare theirs.
- The mutation study re-runs with the identify keys, adds two mutations that read an
  entity the recording never saw (`guesses_user_id`, `reads_wrong_order`), and replays
  five correct-but-different paths of the fake model in `model-live` under `strict` and
  `fallback`, with a termination cause per scenario.
- `make eval-pydantic-ai-arms` (`eval.pydantic_ai --arms strict,fallback`): the
  qwen2.5:7b-recorded travel-desk scenarios in `model-live` under each unrecorded-call
  policy, every run classified by how it ended (passed, unrecorded, approximate,
  model); an `arms` input on the manual Ollama workflow.
- `make eval-pydantic-ai-swap` (`eval.pydantic_ai --swap`): the qwen2.5:7b-recorded
  travel-desk scenarios in `model-live` with the live side changed (qwen2.5:3b under
  `strict` and `fallback`, the prompt without its guard line, a control, and
  `temperature=0` after probe calls), each run's tool calls and wire settings recorded,
  and the first failed 3b run diffed against its cassette; ad hoc `--model`,
  `--prompt-variant` and `--temperature` for `--arms`; a `swap` input on the manual
  Ollama workflow. The pydantic-ai example reads its live model from
  `REPLAY_OLLAMA_MODEL`.
- `make eval-verdict-curve` (in `make eval`): how often a subset of n runs gives the
  all-runs fixed-threshold verdict, for every committed flake report.

- xz-compressed cassettes: a cassette path ending in `.xz` (`run.jsonl.xz`) is read and
  written as xz everywhere a cassette goes (`replay record --out run.jsonl.xz`, `replay
  run`, `replay ci` suites, `replay diff`, `Cassette.load` / `dump`). The file is plain
  xz, so the `xz` tool converts either way. `.gz`, `.bz2`, `.zst` and other compression
  suffixes are refused with the rename that fixes them. `replay record` also takes `--out`.
- `make bench-scale`: cassette bytes, codecs, load, replay and run-store timings for runs
  of 5 to 200 turns and for every committed real recording. `make bench-upstream`: the
  pydantic-ai cassettes replayed on pydantic-ai-slim 2.58.0 to 2.62.0.
- Partial cassettes: a recording to a path (`replay record`, `record(..., out=...)`,
  `RecordingSession(out=...)`) appends each envelope to `<out>.partial` as it completes
  and moves the finished file into place at the end, with the same bytes `dump` writes.
  A `.partial` without a footer loads as interrupted (`Cassette.interrupted`; a torn last
  line is dropped and counted in `torn_bytes`). `replay run` replays it with a warning,
  `replay diff` compares it, and `replay ci` refuses it. `replay record` keeps the
  partial on Ctrl-C and refuses to overwrite one without `--force`.
- `make bench-scale` gains a 400-turn row, a streaming-xz recoverability count, peak
  RSS, load with the collector paused and on, `full` replay with `gc.freeze()`, and
  load timings on the committed real recordings; `make bench-replay` gains the
  per-envelope cost of appending to a partial.
- The pydantic-ai travel desk's `trace_rebooking_chain` task: an eight-booking
  rebooking chain (`RebookedBooking.rebooked_from`) traced with eight sequential
  `get_booking` calls and one conversion, with a fake-model cassette in both suites.
  Its step limits are per scenario (12 tool and 14 model calls; the short tasks keep
  8 and 6).
- `make eval-pydantic-ai-long` (`eval/pydantic_ai.py --long`, an input of the manual
  Ollama workflow): the chain in `model-live` under `strict` and `fallback`, with
  per-run step statuses, a survival curve by served-call index, a clustering rate and
  the identify keys' firings; `make record-pydantic-ai-chain` logs its live recording
  attempts.

- A seventh retail task, hand-written and not from τ-bench
  (`account_not_found_wrong_zip`, `examples/retail_agent/replay-error.toml`): a wrong zip
  code, so the user lookup fails with `is_error: true` and the right reply is that the
  account was not found, with nothing written; the rule `reports_account_not_found`; a
  fake-model cassette and a qwen2.5:7b one. The mutation study replays it unmutated and
  under every agent mutation in all seven configurations (`error_task`).
- `examples/retail_agent/ollama.py`: the retail agent on a local model through Ollama.
- `make record-retail-error-qwen` and `make eval-retail-error-qwen`
  (`eval/retail_error.py`, a `retail_error` input of the manual Ollama workflow): the
  task recorded on qwen2.5:7b, a probe of what Ollama does with `is_error`, and
  `model-live` arms with the unmutated agent and with `drops_error_flag`.

### Changed

- `make record-pydantic-ai` re-records only M9's three tasks; the rebooking chain
  records through `make record-pydantic-ai-chain` into its own attempts log.
- A run whose model has no price is reported as `unpriced` (or `$x + N unpriced
  calls`), not `$0.0000`.
- `count_tokens`, message batches and the beta `parse` / `tool_runner` raise a
  `ReplayError` that names them inside a session instead of going live.
- `Cassette.loads` pauses the cyclic garbage collector while it parses, and only then:
  loading 400 turns (50 MB) went from 1.84–2.11× to 1.22–1.33× linear in bytes against 100
  turns (two runs).

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
