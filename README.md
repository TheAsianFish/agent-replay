# Replay: record-and-replay regression testing for LLM agents

[![CI](https://github.com/TheAsianFish/agent-replay/actions/workflows/ci.yml/badge.svg)](https://github.com/TheAsianFish/agent-replay/actions/workflows/ci.yml)

Replay records a real run of a tool-using agent (every model call, tool call,
clock read and random draw) into a cassette. On each PR it replays that
cassette against the changed code, with the cut point you choose, and asserts
on what the agent did:

- `full`: everything comes from the cassette. Offline, bit-stable, and it
  tests the agent's own orchestration code.
- `model-live`: the model runs for real against tools frozen from the
  recording. This catches prompt and model regressions.
- `tools-live`: the tools run for real against recorded model replies. This
  catches tool regressions without paying for the model.

`replay ci` runs a suite of these scenarios with structural, schema, rule,
budget and LLM-judge checks, and fails the build on a regression.

Judges are not oracles, so a judge check is *advisory* until the judge has
been calibrated against human labels (`replay label`, `replay calibrate`): it
reports its verdict but cannot fail the build. Agents are not deterministic either, so `replay flake` re-runs a
scenario, estimates its pass probability with an interval, gates on it, and
tells flaky from failing.

![The dashboard: the runs list, a failing run's trace, the aligned diff against a passing run, and the flakiness page](docs/walkthrough/walkthrough.gif)

<!-- Scripted: `make walkthrough` seeds the demo store (`make demo-data`), serves the
     dashboard and records it with Playwright (dashboard/e2e/walkthrough.spec.ts), then
     ffmpeg writes docs/walkthrough/walkthrough.{mp4,gif}. Re-record after a UI change. -->

The walkthrough ([mp4](docs/walkthrough/walkthrough.mp4)) clicks through the
dashboard over `make demo-data`'s real `replay ci` and `replay flake` runs of
the retail example: the runs list, a failing run's checks and trace with its
`approximate` steps, the aligned diff against a passing run of the same
scenario (the order lookups a mutant skipped show up as removed steps), and the
flakiness page with a flaky scenario's pass rate and interval over reports.

**Status: 1.0.0, all seven milestones built** (not yet published to PyPI; see
[RELEASE.md](RELEASE.md)). Working today:
- recording through the Anthropic SDK (sync, async, streaming,
  `messages.parse`), Ollama, `@tool`, and the clock and random helpers;
- the three cut-point modes, fingerprint matching and the three
  unrecorded-call policies;
- nine checks, including a rubric-driven `judge` (`claude-opus-5-5` through
  structured outputs, Ollama fallback, verdict cache, advisory until
  calibrated);
- cost and latency baselines (`replay baseline`) and relative budget limits;
- `replay ci` with JSON and Markdown reports, and a run store on SQLite
  (local) or PostgreSQL (shared), versioned with Alembic migrations;
- a τ-bench retail example agent with fake-model and live `claude-opus-5-5`
  cassettes;
- the statistics interface (kappa, bootstrap CI, Wilson, SPRT) with a naive
  baseline implementation and oracle-tested specs;
- `replay flake`: repeated replays with bounded concurrency and rate
  limiting, a sequential or fixed-sample gate, and passing / flaky / failing
  classification;
- a mutation study: 32 injected regressions and 5 benign refactors replayed
  in every mode, with and without failing on `approximate` steps;
- an aligned trace diff (`replay diff`): two runs lined up by sequence
  alignment, so an inserted call reads as one added step;
- judge calibration: golden sets (`replay golden`), a blind labelling TUI
  (`replay label`), `replay calibrate` (kappa with a bootstrap interval per
  criterion, a majority baseline, self-consistency, three bias probes) and the
  gate that lets only a human-calibrated criterion fail a build, plus a
  pairwise judge and an LLMBar harness;
- a dashboard (`replay serve`, or `docker compose up`): runs, a trace viewer,
  the aligned diff, suites, pass-rate trends from `replay flake`, and stored
  calibration reports;
- a GitHub Action (`uses: TheAsianFish/agent-replay@v1`) that runs `replay ci`
  on a PR, keeps one summary comment up to date and fails the check on a
  regression; a wheel with the dashboard bundled, a guarded release workflow
  and a container image.

Measured so far ([RESULTS.md](RESULTS.md)):
- **Mutation study (offline).** Stubbing everything (`full`) caught 7 of
  32 injected regressions, including none of the 12 model regressions and
  none of the 8 tool bugs: 20 of the 32 live in a component it never runs.
  `model-live` caught all 12 model regressions, `tools-live` 5 of 8 tool
  bugs, with no false alarm on the 5 benign refactors. Failing on
  `approximate` steps in `tools-live` caught all 8 tool bugs and the
  stripped policy, and one benign refactor with them.
- **Flakiness on live models.** Replayed 20 times per task, the retail agent
  on `claude-haiku-5-5` passed 119 of 120 runs. The one failure was a refund
  to the wrong payment method, a policy breach. On `claude-opus-5-5` it
  passed 30 of 30 (5 per task), at 41× Haiku's cost per run.
- **Judge against the deterministic checks (offline).** Through the
  calibration pipeline on the 21 distinct runs of the retail golden set, the
  judge agreed with the checks' verdict on 20 (95%), against 71% for a judge
  that always says "fail". The one disagreement is a false pass. This is
  raw agreement with objective labels, not a calibration, and it cannot gate.
- **LLMBar floors (offline).** On the 319 Adversarial instances, always
  choosing the shorter output is right 82.0% of the time and the longer one
  18.0%; always choosing the first output is right 50.0% with a position bias
  of 100%. A real judge's score means little until read against these.

Earlier milestones: on a live `model-live` run, a
25% relative cost limit passed the unchanged agent (within 5% of baseline)
and failed all six tasks of a variant with prompt caching removed (+76% to
+155% cost at unchanged token counts). The uncalibrated judge agreed with the
deterministic checks on 23 of 24 runs; that is a wiring test, not a
calibration.

**Not measured yet:** judge-human agreement (kappa) and the real judge's LLMBar
accuracy. Both need human labels or an API key, and kappa needs the statistics
core (`agent_replay/stats.py`), which is a specification until it is written.
Until then no criterion can gate, and the numbers above are the only ones we
claim (see [Calibrating the judge](#calibrating-the-judge) and
[Statistics](#statistics)). See [PLAN.md](PLAN.md).

## Results

Every number here comes from a command in this repo; the dataset, method and
raw output are in [RESULTS.md](RESULTS.md), which also lists the commands for
the rows still to measure.

| What | Result | Command |
|---|---|---|
| Regressions caught, stub everything (`full`) | 7 / 32 injected mutations; 0 of 12 model and 0 of 8 tool regressions | `make eval-mutations` |
| Regressions caught, cut-point replay | `model-live` 12 / 12 model regressions; `tools-live` 5 / 8 tool bugs (8 / 8 with `fail_on_approximate`) | `make eval-mutations` |
| False alarms on benign refactors | 0 / 5 (1 / 5 with `fail_on_approximate` in `tools-live`) | `make eval-mutations` |
| Live flakiness, retail agent | `claude-haiku-5-5` 119 / 120 runs passed; `claude-opus-5-5` 30 / 30 at 41× Haiku's cost per run | `replay flake` (RESULTS.md, M4) |
| Cost gate, live `model-live` | unchanged agent within 5% of baseline; prompt caching removed: 6 / 6 tasks failed the 25% limit (+76% to +155%) | `replay ci --baseline` (RESULTS.md, M3) |
| Judge vs objective labels, 21 golden items | 20 / 21 agree (95%); always-"fail" baseline 71% | `make eval-judge-objective` |
| LLMBar Adversarial floors (319 instances) | shorter output 82.0%, longer 18.0%, first output 50.0% | `make eval-llmbar-heuristics` |
| Fixed-sample gate meeting α = β = 0.05 | 24 runs (pass on ≥ 19); a fixed-20 majority gate false-passes 13.4% at p = 0.6 | `make bench-sprt` |
| Recorder overhead (median) | ~0.7 ms per model call (12.5 KB request), ~0.1 ms per tool call, ~50 µs per clock read | `make bench-replay` |
| Replay speed | `full` replay of the six live recordings with checks: 222 ms against 51 s spent in live calls (230×) | `make bench-replay` |
| The Action on a real failing PR | to measure: one maintainer command opens it (rehearsed locally: the seeded `wrong_write_argument` PR fails 4 of 6 scenarios, all on `rule:matches_ground_truth`) | `make regression-pr` |
| **To measure** (key or labels needed) | judge-human kappa (~$1.30), D1 golden set (~$2.19), real judge on LLMBar (~$12.73), SPRT runs saved (needs `stats.py`) | RESULTS.md, "To measure" |

## Install

```bash
pip install agent-replay                  # SDK and CLI, once 1.0.0 is on PyPI
pip install 'agent-replay[server]'        # + replay serve and the bundled dashboard
pip install 'agent-replay[postgres]'      # + the PostgreSQL run store
pip install 'agent-replay[ollama]'        # + the Ollama adapter and judge
```

Until the first PyPI release, install from GitHub:
`pip install 'agent-replay[server] @ git+https://github.com/TheAsianFish/agent-replay'`
(the dashboard is bundled only in release wheels; `make ui` builds it in a
checkout). Python 3.12 or newer.

## Quickstart

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). To run the
examples, work from a clone:

```bash
git clone https://github.com/TheAsianFish/agent-replay && cd agent-replay
uv sync
```

The repo ships a customer-service agent adapted from τ-bench's retail domain
(`examples/retail_agent`), with six tasks and one cassette per task. Run its
regression suite. No API key is needed and nothing touches the network:

```console
$ uv run replay ci examples/retail_agent/replay.toml
PASS cancel_pending_laptop (full; 11 exact)
PASS modify_address_suite (full; 8 exact)
PASS return_damaged_bicycle (full; 12 exact)
PASS exchange_waterproof_boots (full; 14 exact)
PASS lookup_latest_order_total (full; 7 exact)
PASS transfer_swapped_refunds (full; 10 exact)
6/6 scenario(s) passed: passed
```

Now replay the same suite against a broken variant of the agent. It changes
one argument of every write action, so the tool names are still right:

```console
$ uv run replay ci examples/retail_agent/replay.toml \
    --agent examples.retail_agent.mutants:wrong_write_argument
FAIL cancel_pending_laptop (full; 1 approximate, 10 exact)
    rule:matches_ground_truth: action #1: expected ('cancel_pending_order',
{'order_id': '#W2417020', 'reason': 'no longer needed'}), got
('cancel_pending_order', {'order_id': '#W2417020', 'reason': 'ordered by
mistake'})
...
2/6 scenario(s) passed: failed
$ echo $?
1
```

The same suite runs against six real `claude-opus-5-5` recordings, offline
(`--cassette-dir examples/retail_agent/cassettes/live`). It can also run with
the model live and only the tools frozen: `--mode model-live --client
anthropic` (needs `ANTHROPIC_API_KEY`). The two runs recorded in
[RESULTS.md](RESULTS.md) passed 6/6, for $0.10 with a warm prompt cache and
$0.29 with a cold one. `make ci-demo` runs the suite in all three modes and
asserts that each of the three seeded regressions makes it exit 1.

A single cassette, with the smaller toy agent:

```bash
uv run replay run examples/toy_agent/cassettes/refund_eligible.jsonl            # full
uv run replay run examples/toy_agent/cassettes/refund_eligible.jsonl \
  --mode model-live --client examples.toy_agent.fake_model:make_client          # model live
uv run replay run examples/toy_agent/cassettes/refund_eligible.jsonl \
  --agent examples.toy_agent.mutants:first_tool_only                            # fails, exit 1
```

Record your own cassette, offline with the deterministic fake model or live:

```bash
uv run replay record examples.toy_agent.agent:run \
  --task "Order A1001 arrived damaged. Can I get a refund?" \
  --client examples.toy_agent.fake_model:make_client -o my.jsonl
uv run replay record examples.toy_agent.agent:run \
  --task "Order A1001 arrived damaged. Can I get a refund?" -o live.jsonl   # needs a key
uv run replay show live.jsonl
```

## Instrumenting an agent

You wrap the client and decorate the tools. With no session active both call
straight through, so the same code runs in production.

```python
import anthropic
import agent_replay as replay


@replay.tool
def lookup_order(order_id: str) -> Order:
    """Look up an order by id."""
    ...


def run(client: anthropic.Anthropic, task: str) -> str:
    today = replay.now()  # recorded, and the same on every replay
    ...  # your usual messages.create / tool_use loop
    ...  # tools=[lookup_order.to_anthropic()] builds the tool definition


run(replay.wrap(anthropic.Anthropic()), "Refund order A1001")
```

- **Agents** are any callable `agent(client, task)`, or `async def`, referred
  to as `module:function` by the CLI.
- **Clients:** `anthropic.Anthropic` and `AsyncAnthropic` (`create`,
  `create(stream=True)`, `stream(...)`, `parse(...)`), and `ollama.Client` /
  `AsyncClient` (`chat`, streaming included; `pip install
  agent-replay[ollama]`).
- **Tools** are plain or `async def` functions decorated with `@tool`.
- **Clock and randomness:** `replay.now()`, `replay.random()`,
  `replay.randint()`, `replay.choice()`, `replay.uuid4()`, plus
  `replay.capture.time()`.

From Python:

```python
from agent_replay import Cassette, replay
from agent_replay.checks import budget, required_tools, tool_sequence

cassette = Cassette.load("cassettes/refund.jsonl")
result = replay(cassette, run, mode="model-live", client=anthropic.Anthropic())
for check in result.check(tool_sequence(cassette.tool_names()), budget(max_cost_usd=0.05)):
    assert check.passed, check.message
```

## Concepts

- **Cassette** (JSONL, schema v2). A header (agent, task, recorder versions),
  then one envelope per call, then a footer with the run's output and status.
  - Each envelope has `kind` (model, tool, clock or random), `name`, request,
    response or error, usage, cost, latency, span ids, and a SHA-256
    `fingerprint` of the canonical request JSON.
  - Transport options (`extra_headers`, `timeout`) are never recorded.
  - Files are written atomically.
  - v1 cassettes from M1 are read through frozen v1 models and migrated in
    memory. The live Opus toy cassette stays committed in v1 to prove it.
- **Cut points.** Each mode is a set of call kinds served from the cassette,
  and everything else runs for real:

  | Mode | Model | Tools | Clock, random |
  |---|---|---|---|
  | `full` | cassette | cassette | cassette |
  | `model-live` | live (`--client`) | cassette | cassette |
  | `tools-live` | cassette | live | cassette |

- **Matching.** A served call looks at the unconsumed recordings with the
  same kind and name.
  - It takes the first one with the same request fingerprint (`exact`).
  - Failing that, it takes the oldest one and flags the step `approximate`.
  - So a reordered run still replays exactly, and a changed request is served
    but visible.
  - A recorded call nested inside another call (a tool that calls the model)
    can only be asked for when its parent runs for real. It is servable only
    in modes where the parent's kind is live; otherwise the parent is served
    whole.
- **Unrecorded calls** follow `--policy`:
  - `strict` (the default) fails the run.
  - `fallback` performs the call for real.
  - `synth` answers a tool call with the smallest value that validates
    against the tool's return type, so a `-> Order` tool returns a real
    `Order`. It understands formats (dates, UUIDs, decimals) and numeric
    bounds. It refuses what it cannot build safely (regex patterns,
    multi-item sets), and the run fails with a message naming `fallback`.
    A model reply cannot be invented, so under `synth` an unrecorded model
    call fails as under `strict`.
- **Failures survive `except Exception`.** These all fail the run even if the
  agent swallows the exception:
  - an unrecorded call under `strict`;
  - a recorded response that no longer fits the tool's return type;
  - a value that cannot be recorded.

  Recorded exceptions are re-raised as instances of the original class.
- **Any replay is a recording too.** `replay run --save out.jsonl` writes the
  replayed run, served calls included, as a cassette that replays like any
  other. That turns a `model-live` run into a new recording.
- **Streaming and async.**
  - Streams are recorded as the final message.
  - Both recording and replay hand the agent a stream rebuilt from that
    message and wrapped in the SDK's own `MessageStream`. So `text_stream`
    and `get_final_message()` behave as usual.
  - The cost: during a recording the agent sees the stream only once the
    model has finished.
  - `stream` is not part of the fingerprint, so switching between streaming
    and non-streaming still matches `exact`.
  - Async tools and `asyncio.gather` keep their spans nested.
  - Thread pools lose the parent span; `replay.ContextThreadPoolExecutor`
    carries it into the workers. A plain pool still works with one active
    session. With several active (parallel replays in one process), a call
    from a plain pool worker cannot be attributed, so it fails every session
    instead of running live.
- **Not recorded:** `client.beta`, `with_raw_response`,
  `with_streaming_response`, and Ollama methods other than `chat`. Inside a
  session they raise a clear error instead of silently going live.

## Checks

| Check | Passes when |
|---|---|
| `tool_sequence(expected, mode="exact" \| "in_order")` | The agent's tool calls match the list (default in a suite: the recorded sequence); prints a diff |
| `required_tools(tools)` / `forbidden_tools(tools)` | Every listed tool was called / none was |
| `max_tool_calls(limit, tool=None)` | At most `limit` tool calls (or calls to `tool`) |
| `arg_schema(tool, schema, must_call=False)` | Every call to `tool` has arguments valid against a JSON Schema or pydantic model |
| `output_schema(schema, parse_json=False)` | The final output validates |
| `rule(fn, **args)` | `fn(run, **args)` returns `True`, `(True, message)` or a passing `CheckResult`; an exception fails the check, never the suite |
| `budget(max_cost_usd=, max_tokens=, max_model_calls=, max_p95_latency_ms=)` | The run is within every absolute limit given. Served steps carry their recorded cost and latency, so in `full` mode under `strict` this re-measures the recording; it bites in `model-live`, or when `fallback` / `synth` let the agent make calls the recording never saw |
| `budget(max_cost_increase_pct=, max_tokens_increase_pct=, max_model_calls_increase_pct=, max_p95_latency_increase_pct=)` | Each metric is at most `baseline × (1 + pct/100)` against the suite's baseline file (equality passes; a zero baseline allows only zero). See [Baselines](#baselines-and-cost-gating) |
| `judge(rubric, provider=, calibration=, min_kappa=)` | An LLM grades the run against a rubric, criterion by criterion. Advisory until a calibration report lets a criterion gate. See [Judge](#judge) |

## Judge

`judge` grades a run against a rubric. The judge sees the task, the agent's
tool calls (inputs, then results or errors, in start order), any agent error
and the final output. It never sees the agent's own model turns, which are
bound to the model that produced them. Each value is clipped to 2000
characters (`max_chars`) with a visible `…[truncated N chars]` marker.

```yaml
# examples/retail_agent/rubric.yaml (abridged)
name: retail-resolution
version: 1
instructions: |-
  The store policy the agent works under, in short: ...
criteria:                      # 1 to 10, unique ids
  - id: policy_followed
    question: Did the agent follow the store policy in the way it acted?
    pass_when: The user was authenticated ... before any consequential action ...
    fail_when: A consequential action came before authentication ...
```

```toml
checks = [{ type = "judge", rubric = "rubric.yaml", cache_dir = "judge-cache" }]
```

- **Providers.** `anthropic` (default): `claude-opus-5-5` at effort `low`,
  with the verdict constrained by a JSON schema
  (`output_config.format`). Instructions and rubric go first, in a cached
  system block; thinking, `budget_tokens` and forced tool choice are never
  sent. A refusal, a truncated answer or a verdict that misses or repeats a
  criterion is a judge error, not a verdict. `ollama` (`pip install
  'agent-replay[ollama]'`, default `qwen3:8b`) is the local fallback.
  `fake`, `FakeJudge(decide)` and `module:factory` cover tests.
- **Cache.** Every verdict is stored under a hash of the judge id (provider,
  model, effort), the rubric's fingerprint, the transcript's fingerprint and
  the prompt version. A `full` replay produces the same transcript, so it is
  judged once and then for free and identically. Commit the cache directory
  and set `REPLAY_JUDGE_OFFLINE=1` to judge in CI without a key. A miss then
  never calls the model: the scenario is reported as unable to run (exit 2),
  so a cache made stale by a rubric, judge or output change cannot pass
  unnoticed. Offline judging suits `full` replays; `model-live` output varies,
  so judge it live. The retail live
  suite does exactly that (`examples/retail_agent/judge-cache/`).
- **Advisory until calibrated.** A criterion can fail the build only if a
  calibration report exists whose rubric fingerprint, judge id and judge
  prompt version match this judge, it compares with *human* labels, it was
  computed with `REPLAY_STATS_IMPL=core`, it has at least
  `REPLAY_JUDGE_MIN_ITEMS` (20) labelled items for the criterion, and the
  lower bound of that criterion's kappa interval is at least
  `REPLAY_JUDGE_MIN_KAPPA` (0.6). Otherwise its failure is reported as
  advisory (`ADVISORY FAIL` in the CLI, an "Advisory" list in the Markdown
  summary), and the summary says why. If the judge cannot answer, an
  uncalibrated judge stays advisory and a calibrated one fails closed. Point
  the check at a report with `calibration = "report.json"`; see
  [Calibrating the judge](#calibrating-the-judge).

## Calibrating the judge

Calibration measures the judge against people, then lets only the criteria
that pass a threshold gate.

```bash
# 1. a golden set: judge transcripts from replaying a suite (offline, no judge calls)
uv run replay golden add examples/retail_agent/replay-live.toml --out eval/golden/retail
uv run replay golden show eval/golden/retail --rubric examples/retail_agent/rubric.yaml
# 2. label it by hand, one criterion at a time (blind)
uv run replay label eval/golden/retail --rubric examples/retail_agent/rubric.yaml --labeller NAME
# 3. measure the judge: prints its call count and cost estimate first
REPLAY_STATS_IMPL=core uv run replay calibrate eval/golden/retail \
    --rubric examples/retail_agent/rubric.yaml --repeats 3 -o report.json
```

then name the report in the suite, `{ type = "judge", rubric = "rubric.yaml",
calibration = "report.json" }`, and each criterion gates or stays advisory by
the rule above. `make golden` rebuilds the retail set (21 items from 24 runs).

- **Golden set.** A directory with `items.jsonl` (judge transcripts keyed by
  fingerprint, so runs the judge would see identically are one item) and an
  append-only `labels.jsonl`. A human label answers one criterion and is tied
  to that criterion's fingerprint, so editing one criterion invalidates only
  its labels (`replay golden show --rubric` counts the stale ones). Each item
  also gets an objective whole-run label (`_overall`) from the suite's
  checks of the run's behaviour (`judge` and `budget` checks excluded: cost is
  not something a transcript shows). `replay golden add` is idempotent from any
  working directory and reports a transcript whose runs got different verdicts
  as a conflict, leaving it out rather than guessing.
- **Labelling.** `replay label` shows the transcript and one criterion.
  It never shows the judge's verdict, the objective label, other labellers'
  answers or which agent variant produced the run. Keys: `p` pass, `f` fail,
  `s` skip, `n` note, `b` back, `q` quit. Each answer is appended at once,
  so quitting loses nothing and rerunning resumes. The labeller comes from
  `--labeller` or `$REPLAY_LABELLER`.
- **`replay calibrate`.** Prints the estimated judge calls and cost, then
  (`--dry-run` stops there; `--max-cost-usd` aborts above a cap) reports, per
  criterion, kappa with a bootstrap interval, raw agreement, the majority-label
  baseline, the confusion matrix with false passes highlighted, and
  self-consistency over `--repeats` verdicts. Reports are schema 2. The bias
  probes (`--probe` to pick one, `--no-probes` to skip) are:
  - *position*: the criteria are presented in reverse order and the verdicts
    compared (pairwise position bias is measured on LLMBar);
  - *verbosity*: a content-free paragraph is appended to the final output;
    flips to pass are the bias;
  - *self-preference*: leniency on runs driven by the judge's own model
    against runs from other models.

  `--reference human` (default) may gate; `--reference objective` compares
  with the suite's checks and never does. `--offline` / `--online` choose
  whether cache misses are filled, `--store` / `--no-store` whether the
  dashboard sees the report, `-o` writes JSON.

Sample, from [RESULTS.md](RESULTS.md) (`make eval-judge-objective`, the real
judge's committed verdicts against the objective labels, offline):

```console
criterion  kappa* [95% CI]    agreement  majority    pass/pass, false pass, false fail, fail/fail  n   gates
_overall   0.95 [0.86, 1.00]  95%        71% (fail)  6, 1, 0, 14                                   21  -
no criterion gates: the calibration report compares the judge with objective labels, not human ones; ...
```

The `*` marks raw agreement: the statistics are the `baseline` implementation,
which is also why that report cannot gate. The judge-versus-human run is
`make eval-judge` (84 calls, about $1.30 estimated; needs a key and the human
labels). `make record-golden` records the larger set D1 (120 `model-live`
runs with Opus and Haiku, about $2.19 estimated).

The LLMBar harness (`eval/llmbar.py`, `agent_replay/judge/pairwise.py`) tests
the judge's robustness on pairs of outputs, each judged in both orders:
`make eval-llmbar` (a key; 838 calls, about $12.73 by the dry-run estimate),
`make eval-llmbar-dry` (the estimate), `make eval-llmbar-heuristics` (the
offline floors above) and `make data-llmbar` (fetch the pinned dataset).

## Baselines and cost gating

An absolute budget (`max_cost_usd = 0.10`) is a number someone picked. A
relative budget compares each scenario with what it cost before:

```console
$ uv run replay baseline examples/retail_agent/replay-live.toml -o examples/retail_agent/baseline-live.json
```

`replay baseline` replays every scenario `--runs N` times (default 1)
*without running checks*, and stores per scenario the median cost, tokens,
model calls and p95 model latency, with the git SHA, mode and policy. Point
the suite at the file with `baseline = "baseline-live.json"` in `[defaults]`
(or `replay ci --baseline PATH`) and add relative limits:

```toml
{ type = "budget", max_cost_increase_pct = 25, max_tokens_increase_pct = 25, max_p95_latency_increase_pct = 50 }
```

A relative latency limit is skipped, and the report says so, when there are
no latency samples. Relative limits without a baseline file, or for a scenario
missing from it, make a broken suite (exit 2). The Markdown summary shows
each cost with its delta, such as `$0.0461 (+3.2%)`. In `full` replays,
served steps carry their recorded cost, so the gate bites when a scenario
runs `model-live` or a cassette is re-recorded. Prices come from
`agent_replay/pricing.py` (captured 2026-10-08); `REPLAY_PRICES_FILE` adds
or overrides models.

## Statistics

Every headline number of the later milestones rests on `agent_replay.stats`
(stdlib only):
- Cohen's kappa (unweighted, linear, quadratic) for judge-human agreement,
  with a paired percentile bootstrap interval;
- Wilson score intervals on pass rates;
- a Wald SPRT (`SequentialGate`) that stops re-running a scenario as soon as
  the evidence decides it, truncated at `max_runs`.

Its docstrings are the specification. The bodies are not written yet and
raise `NotImplementedError`. `agent_replay.stats_baseline` is the naive
reference behind the same interface: raw agreement, a normal-approximation
bootstrap, the Wald interval and a fixed-k majority gate. `REPLAY_STATS_IMPL`
selects one; the default stays `baseline` until `core` exists.
`tests/test_stats.py` checks both against scikit-learn and statsmodels with
Hypothesis, and marks the unimplemented module's tests as expected failures
(a wrong implementation fails them for real). `make bench-sprt` simulates the
gate over true pass rates 0.05 to 1.00 and compares its expected runs with
the smallest fixed-sample test that meets the same error targets. `replay
flake` gates through the same interface.

## `replay ci`

A suite is a TOML file of scenarios, each a cassette, an agent, a mode, a
policy and checks. Paths are relative to the suite. The retail suite
(`examples/retail_agent/replay.toml`) uses every check type:

```toml
[defaults]
agent = "examples.retail_agent.agent:run"
mode = "full"
checks = [
  { type = "tool_sequence" },
  { type = "rule", ref = "examples.retail_agent.rules:checked_before_writing" },
  { type = "budget", max_model_calls = 8 },
]

[[scenario]]
name = "cancel_pending_laptop"
cassette = "cassettes/cancel_pending_laptop.jsonl"
checks = [
  { type = "rule", ref = "examples.retail_agent.rules:matches_ground_truth", args = { task = "cancel_pending_laptop" } },
  { type = "arg_schema", tool = "cancel_pending_order", must_call = true, schema = { required = ["order_id", "reason"] } },
]
```

`replay ci SUITE`:
- replays every scenario and runs its checks;
- stores each run in the run store (`replay runs` lists them, `replay runs
  ID` shows one);
- writes `--json` and `--markdown` reports, and appends the Markdown summary
  to `$GITHUB_STEP_SUMMARY` when that is set;
- exits `0` when everything passed, `1` on any regression, `2` when the suite
  itself is broken.

A scenario that cannot run (a missing cassette, an unimportable agent, a bad
check, no client for `model-live`) is reported with the reason, the rest still
run, and the command exits `2`: that is a broken suite to fix, not a
regression to bisect. `--agent`, `--mode`, `--policy`, `--client` and
`--cassette-dir` override the suite for every scenario.

In `model-live`, an `approximate` tool step means the live model asked
something the recording never saw and got the oldest unused reply for that
tool instead. Set `fail_on_approximate = true` (or pass `--strict` to `replay
run`) when a wrong answer there would make the rest of the run meaningless;
leave it off when the drift is in free text, such as a summary argument.

## GitHub Action

`action.yml` is a composite action that runs `replay ci` on a pull request:

```yaml
# .github/workflows/replay.yml in the repository that holds your agent
on: pull_request
permissions:
  contents: read
  pull-requests: write          # the summary comment
jobs:
  replay:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: python -m pip install -e .          # your agent must be importable
      - uses: TheAsianFish/agent-replay@v1
        with:
          suite: replay.toml
          args: --no-store
```

The full example, with comments, is
[`examples/github-workflow/replay.yml`](examples/github-workflow/replay.yml).
The Action:
- runs in the workflow's own Python, because the agent under test must import
  there, so it does not set Python up itself;
- installs agent-replay from the ref you pinned (`install: action`, the
  default), so `@v1` and the code it runs always agree. Set `install:
  agent-replay==1.0.0` for a PyPI pin, or `install: skip` when your project
  already depends on it (with `replay-command: uv run replay` for a uv
  project);
- keeps **one** comment per suite on the PR (found by a hidden marker and
  updated on every push) holding `replay ci`'s Markdown summary and a link to
  the run. A marker comment the token cannot edit (someone else wrote it) is
  skipped. On a fork's PR the token is read-only, so it warns instead of
  failing;
- uploads the JSON and Markdown reports as an artifact, named
  `replay-report-<job>-<suite>` by default (`artifact-name`; artifact names
  are unique per workflow run, so give matrix legs their own). A failed upload
  is reported but never fails the job;
- fails the job with `replay ci`'s exit code: 1 for a regression, 2 for a
  broken suite, and 2 when the command exited 0 without writing a report (a
  `replay-command` that is not replay ci), so it is never green on no evidence.

Keep the trigger `pull_request`. The Action also comments on
`pull_request_target` events, but that event runs with a write token on the
base branch's behalf, so a workflow that checks out the PR's head there and
replays it executes the PR's own agent code with that token; never combine the
two. A fork's PR gets its verdict and artifact and a warning instead of a
comment.

`mode`, `policy`, `agent`, `client`, `baseline` and `cassette-dir` pass
through to `replay ci`, and `args` appends anything else. A `model-live` suite
needs `ANTHROPIC_API_KEY` in the step's `env`. With `dry-run: true` it runs,
uploads and prints the comment, but neither posts it nor fails. Outputs:
`exit-code`, `passed`, `scenarios`, `failed`, `report-json`, `report-markdown`,
`comment-body` and `comment-url`.

Every input reaches the shell through `env:`, never as `${{ }}` in a script,
so a suite path or argument cannot inject shell code. This repository is its
own first consumer, twice: the `action` CI job runs `uses: ./` in dry-run mode
on the retail suite and asserts that it passes and that a seeded mutant fails,
and [`.github/workflows/replay.yml`](.github/workflows/replay.yml) runs it for
real (dry-run off, `pull-requests: write`) on every pull request that changes
the retail agent. `make regression-pr` opens such a PR with a seeded
regression, waits for that check to fail, captures the check run, the failed
scenarios and the comment into `eval/results/regression_pr/`, and closes the
PR; it needs `gh auth login` as a maintainer and no new secret
(`make regression-pr-rehearse` runs the same edit and `replay ci` offline).

## `replay flake`

```bash
uv run replay flake examples/retail_agent/replay.toml --client anthropic \
    --cassette-dir examples/retail_agent/cassettes/live --max-runs 20 --concurrency 4
```

replays every scenario repeatedly (in `model-live` by default, where sampling
makes runs differ) and treats the outcomes as Bernoulli trials:

- **Gate.** `--gate sprt` (default) feeds each outcome to
  `SequentialGate`, which stops as soon as the evidence separates "pass rate
  `--p-pass`" from "pass rate `--p-fail`" (defaults 0.9 and 0.6, α = β = 0.05),
  or at `--max-runs`. `--gate wilson` runs exactly `--max-runs` times and passes
  when the interval's lower bound reaches `--min-pass-rate`. Both use the
  statistics implementation `REPLAY_STATS_IMPL` selects; with `baseline` the
  sequential gate is a fixed-k majority vote and the report says so.
- **Classification.** Over the runs the gate counted, a scenario is
  `passing` (all passed), `failing` (none did) or `flaky` (some did), with how
  often each check failed. A scenario can pass the gate and still be flaky;
  the report shows both.
- **Concurrency without bias.** Up to `--concurrency` runs are in flight
  (default 4, or 1 when tools run for real: `tools-live` or `--policy
  fallback`, since tools may share state such as the retail example's
  database; a larger value there is honoured with a warning), but
  outcomes reach the gate in run order: failing runs often finish first, and
  feeding them as they finish would bias the test towards failing. No run
  starts more than `--concurrency` places ahead of the gate, so at most
  `concurrency - 1` runs are wasted when it decides; they are reported as not
  counted. `--runs-per-minute` spaces run starts.
- **Reproducible simulations.** Each run gets a seed from `--seed`, the
  scenario and the run index; a client factory with a `seed` parameter
  receives it, so a simulated flaky model gives the same report at any
  concurrency (`make flake-demo`).

Exit codes match `replay ci`: `0` every scenario passed the gate, `1` one
failed, `2` a scenario could not run. `--json`, `--markdown`, the run store
and `$GITHUB_STEP_SUMMARY` work as in `replay ci`. Runs execute on threads, one
replay session each. An agent that hands tool calls to a plain
`ThreadPoolExecutor` makes calls no session can claim while several runs are
active; such a scenario is reported `broken` (exit 2) with the fix
(`agent_replay.ContextThreadPoolExecutor`, or `--concurrency 1`), never as a
flake. A run that crashes the harness is also `broken`, and the other runs are
kept.

## Dashboard

![Runs page](docs/screenshots/runs.png)

```bash
docker compose up --build        # PostgreSQL, a demo-data seed, the app on http://localhost:8000
```

Or locally, against the SQLite store your `replay ci` runs already fill:

```bash
make ui                          # build the React app into agent_replay/server/static
uv run replay serve              # http://127.0.0.1:8000 (API docs at /docs)
make demo-data && make serve REPLAY_DB_URL=sqlite:///.replay/demo.db   # the demo data instead
```

A pip install with the `server` extra serves the dashboard bundled in the
wheel (`replay serve`). For a single container holding only the demo data (the
optional hosted demo in [RELEASE.md](RELEASE.md)):
`docker build --target demo -t agent-replay-demo . && docker run --rm -p 8000:8000 agent-replay-demo`.

The dashboard is read-only and has no authentication: run it on a trusted
network. It has six pages:

- **Runs**: every stored run, filterable by scenario, mode and result.
- **Run detail**: the trace viewer. Each step shows its kind, status
  (`exact`, `approximate`, `live`, ...), latency, tokens and cost, and expands
  to its request and response. Nested calls are indented.
- **Diff**: two runs aligned step by step, with each changed request field.
  The run detail page links here, for any other run of the same scenario.
- **Suites**: each `replay ci` and `replay flake` invocation.
- **Flakiness**: each scenario's pass rate over successive `replay flake`
  reports, with its interval as a band, and the latest classification.
- **Calibration**: stored calibration reports. Each criterion shows kappa with
  its interval against `REPLAY_JUDGE_MIN_KAPPA`, badges for whether it would
  gate, the confusion matrix, the majority baseline, self-consistency and the
  bias probes. A banner says why a report is blocked from gating (objective
  reference, `baseline` statistics, too few items).

| Page | | |
|---|---|---|
| ![Run detail](docs/screenshots/run-detail.png) | ![Diff](docs/screenshots/diff.png) | ![Flakiness](docs/screenshots/flakiness.png) |
| ![Calibration](docs/screenshots/calibration.png) | ![Suites](docs/screenshots/suites.png) | |

`make demo-data` fills a store with about 480 real runs: `replay ci` on the
retail suite in every mode, on the live `claude-opus-5-5` recordings and
against each seeded regression, plus six `replay flake` reports and two real
`replay calibrate` reports. Both compare a judge with the suite's
deterministic verdicts (`--reference objective`), not with human labels, so
the page shows the blocked banner; one is the real judge from the committed
cache, the other a fake judge with repeats and every probe, so each part of
the page has data.

The API is plain JSON under `/api` (OpenAPI at `/docs`). The dashboard's
TypeScript types are generated from that schema (`make openapi`, then `npm run
gen:types` in `dashboard/`). CI fails if either is stale.

| Endpoint | Returns |
|---|---|
| `GET /api/health` | Status, version, backend, schema revision |
| `GET /api/runs?limit&offset&scenario&suite_id&mode&passed` | A page of runs and the total |
| `GET /api/runs/{id}` | One run with steps and checks (an id prefix works) |
| `GET /api/diff?base=&head=` | The aligned diff of two stored runs |
| `GET /api/scenarios`, `/api/suites`, `/api/suites/{id}` | Scenario names; suite invocations |
| `GET /api/flake`, `/api/flake/{id}`, `/api/flake/trend?scenario=` | Flake reports, one in full, and per-scenario points over time |
| `GET /api/calibration`, `/api/calibration/{id}` | Calibration reports, and one with which criteria would gate |

The store's schema is versioned with Alembic. Every command that opens a store
first migrates it to the newest revision, under a lock: `BEGIN IMMEDIATE` on
SQLite, an advisory lock on PostgreSQL. Two processes opening the same store at
once therefore neither collide nor migrate twice, and a failed migration rolls
back. Stores created before migrations are adopted in place.

## CLI

| Command | What it does |
|---|---|
| `replay record AGENT -t TASK -o FILE [--client anthropic\|module:factory] [--force]` | Run the agent for real and write a cassette |
| `replay run CASSETTE [--mode full\|model-live\|tools-live] [--policy strict\|fallback\|synth] [--client REF] [--agent REF] [--expect a,b] [--strict] [--save FILE] [--json]` | Replay one cassette and run `tool_sequence`; `--strict` also fails on `approximate` steps |
| `replay ci SUITE [--agent] [--mode] [--policy] [--client] [--cassette-dir DIR] [--baseline FILE] [--json FILE] [--markdown FILE] [--no-store] [--db URL]` | Replay a suite and gate on it |
| `replay baseline SUITE -o FILE [--runs N] [--agent] [--mode] [--policy] [--client] [--cassette-dir DIR]` | Measure every scenario (no checks) and write median cost, tokens, model calls and p95 latency; exit 2 if a scenario cannot run, 1 if the agent errors (nothing written) |
| `replay flake SUITE [--gate sprt\|wilson] [--max-runs N] [--p-fail P] [--p-pass P] [--alpha A] [--beta B] [--min-pass-rate R] [--concurrency N] [--runs-per-minute R] [--seed S] [-s SCENARIO] [--mode] [--agent] [--client] [--cassette-dir DIR] [--json FILE] [--markdown FILE] [--no-store]` | Replay each scenario repeatedly, gate on its pass rate, classify passing / flaky / failing |
| `replay runs [RUN_ID] [--limit N] [--scenario NAME] [--db URL]` | List stored runs, or show one |
| `replay diff BASE HEAD [--json FILE] [--markdown FILE] [--db URL]` | Align two runs (each a cassette or a stored run id) step by step: same, changed (with the differing request keys), added, removed. Always exits 0 once the diff is produced |
| `replay show CASSETTE` | Print a cassette's header, envelopes and outcome |
| `replay serve [--host H] [--port P] [--db URL] [--dashboard DIR]` | Serve the dashboard and its read-only API (needs the `server` extra) |
| `replay golden add SUITE --out DIR [--variant NAME] [--agent] [--mode] [--client]` | Replay a suite without its judge checks and add each run to the golden set at `DIR`, with an objective label; exit 2 if a scenario could not run |
| `replay golden show DIR [--rubric FILE]` | Summarise a golden set: items, labels per source, labeller and criterion; with `--rubric`, also stale and missing human labels |
| `replay label DIR --rubric FILE [--labeller NAME] [--criterion ID] [--relabel] [--shuffle/--no-shuffle] [--seed N] [--limit N]` | Blind labelling TUI (`p` `f` `s` `n` `b` `q`); never shows the judge's or the objective verdicts; answers are saved as given |
| `replay calibrate GOLDEN_DIR --rubric FILE [--provider] [--model] [--effort] [--reference human\|objective] [--labeller] [--repeats N] [--probe position\|verbosity\|self-preference \| --no-probes] [--offline\|--online] [--cache-dir DIR] [--dry-run] [--max-cost-usd X] [--min-kappa K] [--resamples N] [--seed S] [--store\|--no-store] [--db URL] [-o FILE]` | Print the call and cost estimate, then measure the judge against the golden set's labels: per-criterion kappa and interval, majority baseline, confusion matrix, self-consistency, bias probes, and which criteria would gate |
| `replay db upgrade [--db URL]` | Create the run store or migrate it to the newest revision |
| `replay db add-calibration REPORT [--db URL]` | Store a calibration report for the dashboard |

Exit codes: `0` passed, `1` a check failed or the agent raised, `2` bad input
(missing cassette or suite, unimportable agent, invalid config, missing
credentials).

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Used by `--client anthropic` (recording, `model-live`) and `make test-live` |
| `REPLAY_DB_URL` | `sqlite:///.replay/replay.db` | Run store (SQLAlchemy URL): SQLite, or PostgreSQL as `postgresql+psycopg://user:password@host/db` (the `postgres` extra) |
| `REPLAY_DASHBOARD_DIR` | the build in `agent_replay/server/static` | Built dashboard that `replay serve` serves at `/` |
| `REPLAY_LOG_LEVEL` | `WARNING` (`INFO` for `replay serve`) | structlog level (`DEBUG` shows every recorded or replayed step; `INFO` in `serve` logs each request) |
| `REPLAY_LOG_FORMAT` | `console` | `console` or `json` (one JSON object per line on stderr) |
| `REPLAY_JUDGE_PROVIDER` | `anthropic` | Judge provider: `anthropic`, `ollama`, `fake` or `module:factory` |
| `REPLAY_JUDGE_MODEL` | `claude-opus-5-5` (`qwen3:8b` for Ollama) | Judge model |
| `REPLAY_JUDGE_EFFORT` | `low` | Anthropic effort for the judge: `low`, `medium`, `high`, `xhigh`, `max` |
| `REPLAY_JUDGE_MIN_KAPPA` | `0.6` | Kappa lower bound (0 to 1) a calibrated criterion needs before it may gate |
| `REPLAY_JUDGE_MIN_ITEMS` | `20` | Labelled items (a whole number, at least 1) a criterion needs in a calibration report before it may gate |
| `REPLAY_LABELLER` | — | Who is labelling, for `replay label` when `--labeller` is not given (one of the two is required) |
| `REPLAY_JUDGE_CACHE` | `.replay/judge-cache` | Verdict cache directory when a judge check sets no `cache_dir` |
| `REPLAY_JUDGE_OFFLINE` | off | `1` / `true`: judge from the cache only; a miss breaks the scenario (exit 2) |
| `REPLAY_PRICES_FILE` | — | JSON `{"model": {"input", "output", "cache_read", "cache_write"}}` in USD per MTok, extending or overriding the built-in price table |
| `REPLAY_STATS_IMPL` | `baseline` | Statistics implementation: `baseline` (naive reference) or `core` (`agent_replay/stats.py`) |
| `REPLAY_TEST_PG_URL` | — | Tests only: a disposable PostgreSQL database for the `postgres`-marked tests (they reset its schema) |

`docker compose` also reads `POSTGRES_PASSWORD` (default `replay`, for the local
stack only), `REPLAY_PORT` (default `8000`) and `REPLAY_BIND` (default
`127.0.0.1`: the stack is reachable from this machine only, since the API has no
authentication and serves every stored request and response; `0.0.0.0` exposes it
to the network).

## Architecture

```mermaid
flowchart LR
  subgraph Agent code
    A[agent loop] -->|messages.create / stream / parse<br/>ollama chat| W["wrap(client)<br/>Anthropic · Ollama adapters"]
    A -->|tool call| T["@tool function"]
    A -->|now / random| N[capture helpers]
  end
  W --> S{Session engine}
  T --> S
  N --> S
  S -->|kind runs live in this mode| L[perform the call]
  S -->|kind is served| M[match: same fingerprint, else oldest<br/>exact / approximate]
  M -->|nothing left| P[policy: strict / fallback / synth]
  C[(cassette.jsonl)] --> M
  L --> TR[trajectory + output cassette]
  M --> TR
  TR --> K[checks]
  B[(baseline.json)] -.relative budgets.-> K
  K -->|transcript| J{judge cache}
  J -->|miss| JM[claude-opus-5-5 / Ollama<br/>structured verdict]
  CR[(calibration report)] -.gating criteria.-> K
  K --> CI[replay ci / replay flake: exit code,<br/>JSON / Markdown reports]
  CI --> DB[(run store<br/>SQLite / PostgreSQL · Alembic)]
  DB --> API[FastAPI read-only API<br/>replay serve]
  DF[diff.py: Needleman-Wunsch<br/>alignment] --> API
  API --> UI[React dashboard<br/>runs · trace · diff · flakiness · calibration]
```

The hard part of M2 is making replay trustworthy when only some of the world
is frozen. In `model-live` mode a live model reorders calls, asks for things
the recording never saw, and words arguments differently. Three mechanisms
deal with that, so a replay can follow a run that has drifted:
- fingerprint search with a flagged fallback;
- an explicit policy for unrecorded calls;
- the reachability rule for nested calls.

Every drift stays visible in the trajectory, and the checks decide whether it
matters.

M3's hard part is making a judge safe to put in CI before anyone has measured
it. Its verdicts are reproducible because they are cached by content: same
rubric, same judge, same transcript gives the same verdict, offline. Every
criterion is advisory until a calibration report for that exact rubric and
judge shows a kappa lower bound above the threshold. Cost is gated relative to
a measured baseline, because cost regressions often leave tokens unchanged.
M4's hard part is telling a real regression from noise without paying
for twenty runs every time, and knowing what the checks can and cannot see.
`replay flake` turns repeated runs into a sequential test fed in run order, so
concurrency does not bias it and early stopping stays valid. The mutation
study measures detection per check type and per mode, which is where the case
for cut-point replay comes from: the stub-everything mode missed every model
and tool regression.

M5's hard part is the trace diff and a store that several processes share.
Comparing two runs position by position shows "everything after step 3
changed" when the agent inserted one call. `diff.py` instead aligns the two
step lists with Needleman-Wunsch: an identical step costs 0, the same call with
different arguments 0.5 + 0.5 x (fraction of argument keys that differ), and a
step only one side has 1. Steps with different names are never paired. A
changed step therefore always costs less than removing and re-adding it, so
the diff reads "one call added, one argument changed". The store is now shared
by CI jobs, `replay flake` threads and the dashboard. Migrations run on open
under a database lock, inside one transaction.

M6's hard part is a judge measured against humans before it may fail a build,
without the measurement flattering the judge. Labelling is blind, so the
labeller is not anchored by the judge's verdict; labels are tied to each
criterion's fingerprint, so a reworded criterion cannot inherit old labels;
and the gate demands human labels, the `core` statistics and enough items,
because a kappa interval from a handful of runs, or raw agreement standing in
for kappa, would let an unmeasured judge fail builds. What the code can
compute offline is measured (the judge against the deterministic checks, the
LLMBar heuristic floors); the judge-human kappa waits for labels and a key.

M7 packages it for other repositories. The Action runs in the consumer's own
Python, since only there does the agent under test import, and installs
agent-replay from the ref the workflow pinned, so `@v1` works without PyPI and
never runs different code from the tag. It updates one marked comment instead
of adding one per push, and treats a failed comment as a warning: the replay
verdict is what gates, and a fork's read-only token must not turn a passing
suite red. The wheel bundles the dashboard build, which is gitignored, and CI
installs it in a clean venv outside the repo, so a packaging mistake fails
there instead of on a user's machine. PLAN.md has the full component map.

## Design decisions

- **Intercept at the SDK, not HTTP.** `anthropic` 1.x runs on `httpx2`, so
  HTTP-level recorders built for `httpx` never see its traffic. Wrapping the
  client gives typed messages and usage, and keeps credentials out of
  cassettes. Providers sit behind a `ProviderAdapter` protocol (Anthropic,
  Ollama).
- **One engine, modes as data.** Recording and the three replay modes are
  one session class. Each mode is just the set of kinds it serves, so every
  mode shares the same matching, policies, spans and output cassette. The
  alternative, a separate session class per mode, would duplicate the subtle
  parts.
- **Fingerprint search before cursor order.** Matching by fingerprint first
  makes reordered parallel calls `exact`, and falling back to the oldest
  unconsumed recording keeps a drifting live run going instead of crashing.
  The `approximate` flag (and `--strict` / `fail_on_approximate`) keeps that
  honest. Cut-point replay follows Chronicle (arXiv:2609.20625).
- **Streams recorded as the final message.** Recording the final message,
  not the raw events, keeps cassettes small and means the stream can be
  rebuilt and replayed whether or not the agent streamed. The price is that
  recording buffers the stream.
- **Schema versions are frozen.** v2 added the clock and random kinds as a
  new version with a migration, not by editing v1.
- **The fake model returns real SDK objects** and reports itself as
  `fake:<model>`, so its synthetic token counts are never priced as real
  spend.
- **The judge reports before it gates.** A judge nobody has compared with
  humans would make CI fail on a number nobody validated, so failures are
  advisory until calibrated, per criterion, and a report only counts for the
  rubric fingerprint, judge id and judge prompt version it measured (editing
  the rubric, the judge prompt, the model or the effort voids it). The alternative, trusting the judge, is what
  this project exists to avoid.
- **Labelling is blind.** `replay label` never shows the judge's verdict, the
  objective label or the agent variant, because seeing them anchors the answer
  and inflates the agreement being measured. Answers are appended as given, so
  an interrupted session loses nothing.
- **Labels are tied to criterion fingerprints.** A human label answers one
  criterion as worded when it was given. Editing a criterion invalidates only
  its labels (the others stay valid), and `replay golden show --rubric` counts
  the stale ones; the alternative, tying labels to the whole rubric, would
  throw away every label on a typo fix.
- **A criterion gates only on human labels, `core` statistics and enough
  items.** Objective labels come from the same checks the judge is meant to
  complement, so agreement with them does not show agreement with people. The
  `baseline` implementation puts raw agreement where kappa belongs, and raw
  agreement is high whenever one label dominates, so a report that does not
  name its statistics, or names `baseline`, does not gate. With fewer than
  `REPLAY_JUDGE_MIN_ITEMS` items a bootstrap interval says little (two agreeing
  items give [1, 1]). The `_overall` row is not a rubric criterion and never
  gates.
- **Structured outputs through `messages.create`.** The judge sends the
  same JSON-schema `output_config` that `messages.parse` would, but reads
  `stop_reason` itself. In SDK 1.12, `parse` raises a validation error on a
  refused or truncated answer, which hides why. Forced tool choice is
  rejected by Opus 5.5 and prefill is gone, so a schema is the robust path.
- **A content-addressed judge cache.** Keyed on what decides the verdict
  (judge id, rubric, transcript, prompt version), not on the run, so `full`
  replays are judged once and a committed cache makes CI judging free and
  deterministic. The alternative, recording judge calls into the agent's
  cassette, would tie verdicts to one recording.
- **Baselines are medians of replays without checks.** Running no checks
  avoids the chicken-and-egg of relative budgets needing a baseline; the
  median keeps one slow run from setting the bar.
- **The statistics ship as a spec plus a naive baseline.** The interface,
  oracle tests and a deliberately simple implementation land first, so every
  later number can be compared against "what the naive method would have
  said".
- **Flakiness is a sequential test, not a retry loop.** Retrying until green
  hides real failure rates; a fixed k wastes runs on clear cases and, at k =
  20 with a majority vote, still passes 13% of scenarios whose true pass rate
  is 0.6. The gate is a Wald SPRT behind the stats interface, compared against
  the smallest fixed-sample test meeting the same error targets (24 runs).
- **Outcomes feed the gate in run order.** Feeding by completion would let
  fast failures decide early and bias the gate; the price is that a slow run
  holds the others back, bounded by a window of `--concurrency` runs.
- **Mutations by kind and mode, with benign controls.** Each mutation is an
  agent, model or tool bug, so detection per mode shows what each cut point
  can see. Benign refactors measure false alarms. `full` mode is the
  stub-everything baseline, measured rather than assumed.
- **State checks only where state is real.** `rule:db_matches_ground_truth`
  (τ-bench's reward) passes as "not applicable" unless every tool call ran
  for real, because a served tool never touched the database.
- **Migrations on open, under a lock.** Every command migrates the store it
  opens, so a CI job never meets an old schema. SQLite takes its write lock
  before reading the schema version and PostgreSQL takes an advisory lock, so
  concurrent first opens are safe (a test runs eight processes at once on
  both backends). The alternative, a separate `migrate` step, is still there
  (`replay db upgrade`), but nothing depends on someone running it.
- **The server lives in the package, the dashboard is static.** `replay
  serve` is one process serving JSON and the built React app, so `pip install
  'agent-replay[server]'` and Docker Compose run the same thing. The API is
  read-only REST. The plan had server-sent events; for a page that shows
  finished runs, refetching on focus is enough, and SSE is on the roadmap.
- **Types generated from the API schema.** The dashboard's TypeScript types
  come from the FastAPI OpenAPI schema, and CI fails when either is stale, so
  a model change in Python breaks the TypeScript build rather than the page.
- **Alignment, not positional diff.** See M5's hard part above. A changed
  step's cost is capped below the cost of a gap pair, so steps with the same
  name always pair when the order allows. Ties break deterministically, front
  to back.
- **Committed cassettes are regenerated, not hand-edited.** Drift tests
  re-record the toy and retail cassettes and fail if they changed (`make
  cassettes`, `make retail-cassettes`).
- **The Action installs from its own ref, not PyPI.** `install: action`
  pip-installs the checkout GitHub made of the pinned ref, so a consumer on
  `@v1` gets exactly that code, and the Action worked before any PyPI release.
  The cost is a source build per run (8 to 10 s for the install,
  dependencies included, measured in a fresh venv); `install:
  agent-replay==X.Y.Z` uses the wheel instead.
- **A composite action, not a Docker or JavaScript one.** A Docker action
  would run in its own image, where the consumer's agent and its dependencies
  are not installed; a JavaScript action would need a bundled Node build. The
  logic that is more than a few lines (outputs, the comment) is a stdlib-only
  Python script with tests.
- **Trusted publishing, no tokens.** PyPI accepts the release workflow's OIDC
  identity for one repository, workflow and environment, so no upload token
  exists to leak, and every publishing job is guarded to tag pushes on this
  repository.

## Development

```bash
make check             # ruff check + format check, mypy --strict, pytest (offline, sockets blocked)
make demo              # toy quickstart: replay passes, the mutant fails
make ci-demo           # retail suite in all three modes; every seeded regression fails
make test-live         # live tests against claude-opus-5-5 (needs a key; a few cents)
make cassettes         # re-record the toy agent's cassettes (Anthropic and Ollama fakes)
make retail-cassettes  # re-record the retail agent's cassettes with the fake model
make bench-sprt        # simulate the sequential gate over true pass rates (offline)
make flake-demo        # replay flake offline: fake model passes, seeded flaky model fails
make eval-mutations    # the mutation study: 32 mutations + 5 benign refactors, 5 configurations (~25 s)
make eval-judge-smoke  # judge vs deterministic checks on 24 retail runs (offline, from the cache)
make golden            # rebuild the retail golden set: 21 items from 24 runs (offline)
make eval-judge-objective  # judge vs the golden set's objective labels (offline, from the cache)
make eval-judge        # judge vs human labels, kappa + probes (needs a key and human labels; ~$1.30)
make record-golden     # D1: 120 model-live runs, Opus and Haiku (needs a key; ~$2.19 estimated)
make token-rates       # re-derive the calibrate cost estimate's constants (offline)
make data-llmbar       # fetch LLMBar at the pinned commit into eval/data/llmbar
make eval-llmbar-heuristics  # LLMBar floors: first / longer / shorter output (offline)
make eval-llmbar-dry   # calls and estimated cost of the real judge's LLMBar run (offline)
make eval-llmbar       # the real judge on LLMBar (needs a key; 838 calls, ~$12.73 estimated)
make demo-data         # fill .replay/demo.db with real replay runs for the dashboard (~25 s)
make ui                # npm ci + build the dashboard into agent_replay/server/static
make ui-check          # dashboard: generated types current, tsc, eslint, vitest
make serve             # replay serve over $REPLAY_DB_URL (or .replay/replay.db)
make eval              # every offline evaluation (mutations, judge smoke and objective, LLMBar floors, token rates)
make bench             # every offline benchmark (bench-sprt, bench-replay)
make bench-replay      # recorder overhead per boundary and full replay vs live duration (~20 s, to .replay/)
make walkthrough       # record the README's dashboard walkthrough (Playwright + ffmpeg, ~1 min)
make regression-pr     # open a seeded-regression PR, capture its failing check and comment, close it (gh login)
make regression-pr-rehearse  # the same suite edit and replay ci, offline
make dist              # dashboard + wheel and sdist, each installed and smoke-tested in a clean venv
make openapi           # regenerate dashboard/openapi.json from the API
docker compose up --build   # PostgreSQL + demo seed + dashboard on :8000
```

PostgreSQL tests run when `REPLAY_TEST_PG_URL` points at a disposable database
(`uv run pytest -m postgres`); CI runs them against a PostgreSQL 16 service.
The Playwright smoke test runs against a served build (`npm run e2e` in
`dashboard/`, with `DASHBOARD_URL` if it is not on :8000).

See [CONTRIBUTING.md](CONTRIBUTING.md) for conventions, [RESULTS.md](RESULTS.md)
for every measured number, [RELEASE.md](RELEASE.md) for the release steps and
[CHANGELOG.md](CHANGELOG.md) for what each version changed.

## Roadmap

1. ~~Scaffolding, cassette v1, recorder, `full` replay, `tool_sequence`~~
2. ~~Cut-point modes, fingerprint matching, unrecorded-call policies, clock and
   random capture, async and streaming, Ollama, the checks library, SQLite
   store, `replay ci`, the τ-bench retail example~~
3. ~~LLM judge (structured outputs), cost and latency baselines, statistics
   interface~~
4. ~~Flakiness engine (`replay flake`, Wilson and SPRT gates through the stats
   interface) and a mutation study~~
5. ~~FastAPI + React dashboard with aligned trace diffs, PostgreSQL and
   migrations, Docker Compose~~
6. ~~Judge calibration against human labels (kappa, bootstrap CI, bias probes,
   LLMBar)~~ (built and tested offline; the judge-human kappa, the LLMBar run
   and `agent_replay/stats.py` are still to do: they need human labels, an API
   key and the implementation, see [RESULTS.md](RESULTS.md) "To measure")
7. ~~GitHub Action, PyPI packaging with the dashboard bundled, guarded release
   workflow, container image~~ (built; publishing is the maintainer steps in
   [RELEASE.md](RELEASE.md))

Next (evolution round 1, 2026-10-21; evidence and hypotheses in
[PLAN.md](PLAN.md), "Evolution round 1"):

8. The Action on a real failing pull request, and a recorded walkthrough of
   the dashboard (runs, aligned diff, flakiness) in place of the terminal GIF
9. Dogfood on [pydantic-ai](https://github.com/pydantic/pydantic-ai) through
   Ollama: record the `beta` messages surface the framework uses, commit a
   cassette from a local model (qwen2.5:7b), replay it in `full` and
   measure its pass rate over 10 `model-live` runs
10. A tool-result drift check (`fail_on_tool_drift`): in `tools-live`, compare
    each live tool's result with the recorded one, so that the three tool bugs
    only `fail_on_approximate` catches today are caught without failing the
    benign JSON-layout refactor, or the live task whose free-text argument is
    reworded on every run

Still to do, in order: implement `agent_replay/stats.py` so kappa and the
SPRT run for real; label the retail golden set and record D1 to measure
judge-human kappa; the real judge on LLMBar; `make bench-flake` (D3).

Later: replaying the pydantic-ai cassette across upstream releases, the
verdict-stability curve on a local model, a bytes-per-turn scale benchmark,
live updates in the dashboard (server-sent events), authentication for a
shared deployment, and running a flake suite's scenarios in parallel as well
as their runs.

## Prior art

VCR-style replay for agents exists in `pytest-agentreplay`, `replayproof`,
`chronicle` and `pytest-agent-trace`. The cut-point idea (choosing which
boundaries to serve from the recording) follows Chronicle (arXiv:2609.20625).
The retail example's data, tools and tasks come from τ-bench (Yao et al.,
2024, MIT). Replay aims to differ on measurement: judge calibration,
statistical flakiness gating and a mutation study.

## License

MIT, see [LICENSE](LICENSE). The vendored τ-bench subset keeps its MIT
license in `examples/retail_agent/data/LICENSE-tau-bench`.
