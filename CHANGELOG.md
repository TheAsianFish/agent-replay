# Changelog

All notable changes to `agent-replay` are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/). The cassette format is versioned
separately (`schema_version` in each cassette header); readers keep accepting
every older version.

## [Unreleased]

Nothing yet.

## [1.0.0] - 2026-10-31

The first release: the SDK, the `replay` CLI, the GitHub Action, the
self-hosted dashboard, and the evaluations that measure them (RESULTS.md).

### Recording and replay

- Record a real agent run into a JSONL cassette: every model call (Anthropic SDK
  sync, async, streaming and `messages.parse`; `client.beta.messages.create`,
  plain and `stream=True`, and `client.beta.messages.stream(...)` under the call
  name `anthropic.beta.messages.create`, with `betas` in the fingerprint, the
  surface pydantic-ai calls; Ollama), every `@tool` call, clock reads and random
  draws, with usage, cost and latency. Cassette schema v2, with v1 cassettes read
  through a migration.
- `client.with_options(...)` and `client.copy(...)` on a wrapped client return a
  wrapped client, so calls through a copy (pydantic-ai makes them when
  `max_retries` is set) are recorded. `count_tokens`, message batches and the beta
  `parse` / `tool_runner` raise a `ReplayError` that names them inside a session
  instead of going live.
- Replay with cut points: `full` (everything from the cassette, offline and
  bit-stable), `model-live` (the model runs live against frozen tools) and
  `tools-live` (tools run live against recorded model replies). The offline
  stand-in client carries a `base_url`, which pydantic-ai reads on every response.
- Fingerprint matching with a cursor fallback that flags `approximate` steps;
  unrecorded-call policies `strict`, `fallback` and `synth` (stubs generated from
  a tool's JSON schema).
- `identify` keys (`[defaults.identify]`, per scenario, or `replay(identify=...)`):
  per tool, the arguments that name the entity a call reads or writes. A served
  call only falls back to a recording of the same entity; otherwise it is
  unrecorded and the policy decides.
- xz-compressed cassettes: a path ending in `.xz` (`run.jsonl.xz`) is read and
  written as xz everywhere a cassette goes (`replay record --out run.jsonl.xz`,
  `replay run`, `replay ci` suites, `replay diff`, `Cassette.load` / `dump`). The
  file is plain xz, so the `xz` tool converts either way. `.gz`, `.bz2`, `.zst`
  and other compression suffixes are refused with the rename that fixes them.
- Partial cassettes: a recording to a path (`replay record`, `record(..., out=...)`,
  `RecordingSession(out=...)`) appends each envelope to `<out>.partial` as it
  completes and moves the finished file into place at the end, with the same
  bytes `dump` writes. A `.partial` without a footer loads as interrupted
  (`Cassette.interrupted`; a torn last line is dropped and counted in
  `torn_bytes`). `replay run` replays it with a warning, `replay diff` compares
  it, and `replay ci` refuses it. `replay record` keeps the partial on Ctrl-C and
  refuses to overwrite one without `--force`.
- `Cassette.loads` pauses the cyclic garbage collector while it parses, and only
  then: loading 400 turns (50 MB) went from 1.84–2.11× to 1.22–1.33× linear in
  bytes against 100 turns (two runs).

### Checks and CI

- Nine checks: `tool_sequence`, `required_tools`, `forbidden_tools`,
  `max_tool_calls`, `arg_schema`, `output_schema`, `rule`, `budget` and `judge`.
- `replay ci`: a TOML suite of scenarios, exit codes 0 / 1 / 2, JSON and Markdown
  reports, the Markdown summary written to the GitHub step summary.
- `fail_on_tool_drift` in suites (and `replay ci --fail-on-tool-drift`): in
  `tools-live`, every live tool result is compared with the one recorded for the
  same call, and a difference fails the scenario as `no_tool_drift`, naming the
  first differing path. Per-tool `ignore_fields` skip values that legitimately
  vary; with `identify` keys, a live call whose identity no recording has is
  drift (`identity_unrecorded`).
- Cost and latency baselines (`replay baseline`) with relative budget limits;
  a dated pricing table (`agent_replay/pricing.py`, overridable with
  `REPLAY_PRICES_FILE`). A run whose model has no price is reported as
  `unpriced` (or `$x + N unpriced calls`), not `$0.0000`.
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

### Examples

- `examples/toy_agent`: the quickstart agent, with fake-model cassettes for the
  Anthropic and Ollama adapters.
- `examples/retail_agent`: the τ-bench retail subset (six tasks, MIT, attributed)
  plus a hand-written seventh task (`account_not_found_wrong_zip`,
  `replay-error.toml`) in which a wrong zip code makes the user lookup fail with
  `is_error: true` and the right reply is that the account was not found, with
  nothing written (rule `reports_account_not_found`); fake-model, live
  `claude-opus-5-5` and qwen2.5:7b cassettes; `ollama.py` runs the agent on a
  local model through Ollama; `replay-qwen.toml` holds the ground-truth checks for
  a local model (no exact tool sequence).
- `examples/pydantic_ai_agent`: a tool-using pydantic-ai travel desk with the
  client injected through
  `AnthropicProvider(anthropic_client=replay.wrap(AsyncAnthropic(...)))` and no
  other framework change; fake-model and qwen2.5:7b cassettes; the
  `trace_rebooking_chain` task (an eight-booking chain, `RebookedBooking.rebooked_from`,
  traced with eight sequential `get_booking` calls and one conversion), with
  per-scenario step limits.

### Evaluation

- Offline, re-run by CI against the committed results: the mutation study (32
  injected regressions, 5 benign refactors, 2 identity mutations, 5
  correct-but-different paths and the error task, in seven replay
  configurations), a simulation of the sequential gate, the judge against
  objective labels, LLMBar floors, `make eval-verdict-curve` (how often a subset
  of n runs gives the all-runs fixed-threshold verdict, for every committed flake
  report), a recorder-overhead and replay-speed benchmark, `make bench-scale`
  (cassette bytes, codecs, load, replay and run-store timings for runs of 5 to
  400 turns and for every committed real recording) and `make bench-upstream`
  (the pydantic-ai cassettes on pydantic-ai-slim 2.58.0 to 2.62.0).
- On a local model through Ollama, from `make` targets and the manual
  `ollama.yml` workflow, never on pull requests: `eval-pydantic-ai` (a
  `model-live` flake on qwen2.5:7b), `eval-pydantic-ai-arms` (`strict` against
  `fallback`), `eval-pydantic-ai-swap` (qwen2.5:3b, the prompt without its guard
  line, `temperature=0` after probe calls, and a control), `eval-pydantic-ai-long`
  (the rebooking chain, with a survival curve by served-call index),
  `eval-pydantic-ai-no-guard` (`fallback` against `synth`), `eval-retail-error-qwen`
  (the error task unmutated and with `drops_error_flag`, after a probe of what
  Ollama does with `is_error`) and `eval-retail-mutations-qwen` (eight agent
  mutations on `lookup_latest_order_total`, with a `tool_use_id` probe and every
  prompt's size read from the server log). Every run is classified by how it
  ended; the raw reports are committed under `eval/results/`. `make record-*`
  targets log each recording attempt.
- `make regression-pr` opens a pull request that seeds a regression so the
  Action fails it for real; `make walkthrough` records the README's dashboard
  walkthrough. Results and the commands that need an API key are in
  `RESULTS.md`.

### Known limitations

- `agent_replay/stats.py` (the `core` statistics) is a specification with
  oracle tests; until it is implemented, calibration reports show raw agreement
  in the kappa column and no judge criterion can gate.
- The dashboard API has no authentication; run it on a trusted network.
- The longest committed real recording is 6 turns; the cassette-bytes results
  past that are on synthetic runs.

[Unreleased]: https://github.com/TheAsianFish/agent-replay/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/TheAsianFish/agent-replay/releases/tag/v1.0.0
