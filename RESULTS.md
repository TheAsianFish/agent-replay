# Results

Every number in README.md comes from this file. Each one lists its dataset,
method, the command that produced it and the raw output. Sections are added as
the evaluations land. PLAN.md, "Evaluation", lists the full plan.

## Live recording smoke test (M1)

Checks that the recorder works against the real Anthropic API and that a
real recording replays offline exactly. This is a single run, not a
benchmark: the latencies are one sample each.

- **Dataset:** one task for the toy agent, "Order A1001 arrived damaged. Can I
  get a refund?"
- **Model:** `claude-opus-5-5`, `output_config={"effort": "low"}`,
  `max_tokens=4096`. Cost uses the prices in `agent_replay/pricing.py`
  (captured 2026-10-08: $4 / $20 per MTok input / output).
- **Date:** 2026-10-09.
- **Command:** `uv run replay record examples.toy_agent.agent:run --task "Order A1001 arrived damaged. Can I get a refund?" -o examples/toy_agent/cassettes/live_opus_refund_eligible.jsonl`

Raw output:

```
#  kind   name                       status  request
0  model  anthropic.messages.create  live    "1 message(s)"
1  tool   lookup_order               live    {"order_id":"A1001"}
2  tool   get_refund_policy          live    {}
3  model  anthropic.messages.create  live    "3 message(s)"
4  tool   issue_refund               live    {"amount":42.5,"order_id":"A1001"}
5  model  anthropic.messages.create  live    "5 message(s)"
Recorded 6 envelope(s) to
examples/toy_agent/cassettes/live_opus_refund_eligible.jsonl (2687 tokens,
$0.0140)
Output: I'm sorry your order arrived damaged — I've refunded the full $42.50 for
order A1001 (refund ID RF-A1001).
```

Per-call usage, read from the committed cassette:

| Call | Content returned | Input tokens | Output tokens | Cost (USD) | Latency (ms) |
|---|---|---|---|---|---|
| 1 | 2 × `tool_use` (parallel) | 655 | 80 | 0.004220 | 11478.5 |
| 2 | 1 × `tool_use` | 856 | 77 | 0.004964 | 11233.9 |
| 3 | `text` | 972 | 47 | 0.004828 | 5156.2 |

Offline replay of that cassette (`uv run replay run examples/toy_agent/cassettes/live_opus_refund_eligible.jsonl`):
all 6 steps `exact`, `PASS tool_sequence`, exit 0. The same test runs as
`make test-live` (`tests/test_live_anthropic.py`). It records a fresh live run,
then replays it offline. It passed twice in this session, before and after
the review fixes (last run: `1 passed, 132 deselected in 5.71s`). Its cost was
not captured separately. The mutant
(`--agent examples.toy_agent.mutants:first_tool_only`) also fails on the live
cassette (exit 1).

## Retail agent: live recording and model-live replay (M2)

A smoke test of the M2 replay engine against the real API, not a benchmark:
one run per task, so it says nothing yet about run-to-run variance. Repeated
runs with confidence intervals are M4's job.

- **Dataset:** the six τ-bench retail test tasks vendored in
  `examples/retail_agent` (indices 10, 17, 27, 53, 67, 69), adapted to a
  single turn (see `examples/retail_agent/README.md`).
- **Model:** `claude-opus-5-5`, `output_config={"effort": "low"}`,
  `max_tokens=4096`, top-level `cache_control={"type": "ephemeral"}`. Cost uses
  `agent_replay/pricing.py` (captured 2026-10-08: $4 / $20 per MTok input /
  output, cache reads $0.20, 5-minute cache writes $5).
- **Date:** 2026-10-09. Total spend: $0.691 (recording $0.2950, first
  model-live run $0.1032, second model-live run $0.2933).

### Step 1: record each task live

**Command:** `uv run python -m examples.retail_agent.record_all --client anthropic --out examples/retail_agent/cassettes/live`

`db matches` is τ-bench's objective reward: after the run, the retail
database equals the database produced by applying τ-bench's ground-truth
actions (`tasks.expected_db_hash`). Raw output:

```
examples/retail_agent/cassettes/live/cancel_pending_laptop.jsonl: 11 envelope(s), ok, db matches, $0.0461
examples/retail_agent/cassettes/live/modify_address_suite.jsonl: 6 envelope(s), ok, db matches, $0.0403
examples/retail_agent/cassettes/live/return_damaged_bicycle.jsonl: 12 envelope(s), ok, db matches, $0.0527
examples/retail_agent/cassettes/live/exchange_waterproof_boots.jsonl: 14 envelope(s), ok, db matches, $0.0654
examples/retail_agent/cassettes/live/lookup_latest_order_total.jsonl: 9 envelope(s), ok, db matches, $0.0452
examples/retail_agent/cassettes/live/transfer_swapped_refunds.jsonl: 10 envelope(s), ok, db matches, $0.0453
```

Usage summed from the committed cassettes:

| Task | Model calls | Tool calls | Output tokens | Cache writes | Cache reads | Cost (USD) | Model latency (s) |
|---|---|---|---|---|---|---|---|
| cancel_pending_laptop | 5 | 6 | 494 | 6,445 | 19,938 | 0.0461 | 8.6 |
| modify_address_suite | 3 | 3 | 418 | 5,986 | 9,680 | 0.0403 | 6.6 |
| return_damaged_bicycle | 5 | 7 | 567 | 7,430 | 20,841 | 0.0527 | 8.8 |
| exchange_waterproof_boots | 6 | 8 | 703 | 9,096 | 29,147 | 0.0654 | 10.8 |
| lookup_latest_order_total | 4 | 5 | 580 | 6,156 | 13,751 | 0.0452 | 6.5 |
| transfer_swapped_refunds | 5 | 5 | 553 | 6,044 | 19,630 | 0.0453 | 9.7 |
| **Total** | **28** | **34** | **3,315** | **41,157** | **112,987** | **0.2950** | **51.0** |

Uncached input was 68 tokens in total: prompt caching served 73% of the
prompt tokens from cache (112,987 of 154,212).

### Step 2: replay the live recordings through `replay ci`

**Offline, `full` mode** (no network; this also runs in the test suite as
`tests/test_ci.py::test_live_opus_recordings_pass_the_suite_offline`):

```
$ uv run replay ci examples/retail_agent/replay.toml --no-store --cassette-dir examples/retail_agent/cassettes/live
PASS cancel_pending_laptop (full; 11 exact)
PASS modify_address_suite (full; 6 exact)
PASS return_damaged_bicycle (full; 12 exact)
PASS exchange_waterproof_boots (full; 14 exact)
PASS lookup_latest_order_total (full; 9 exact)
PASS transfer_swapped_refunds (full; 10 exact)
6/6 scenario(s) passed: passed
```

**`model-live`:** the model runs for real again and every tool call is served
from the live recording. Run right after step 1, so the prompt cache was warm.

```
$ uv run replay ci examples/retail_agent/replay.toml --no-store --cassette-dir examples/retail_agent/cassettes/live --mode model-live --client anthropic --json model_live.json
PASS cancel_pending_laptop (model-live; 6 exact, 5 live)
PASS modify_address_suite (model-live; 3 exact, 3 live)
PASS return_damaged_bicycle (model-live; 7 exact, 5 live)
PASS exchange_waterproof_boots (model-live; 8 exact, 6 live)
PASS lookup_latest_order_total (model-live; 5 exact, 4 live)
PASS transfer_swapped_refunds (model-live; 1 approximate, 4 exact, 5 live)
6/6 scenario(s) passed: passed
```

From that run's JSON report (not kept): $0.1032 and 157,409 tokens for the
six scenarios, 50.9 s wall clock. Every check passed on every scenario (6 or
7 per scenario: the exact recorded tool sequence, the two τ-bench rules, the
call and budget limits, and the scenario's schema or tool checks). The one
`approximate` step is a `transfer_to_human_agents` call whose free-text
`summary` argument was worded differently from the recording; it was still
served.

**Second `model-live` run**, made during review with the prompt cache cold,
so the report can be checked from the repo. Its JSON report is committed as
[`eval/results/m2_model_live.json`](eval/results/m2_model_live.json)
(the report's `git_sha` is the `$GITHUB_SHA` of the runner it ran on, not a
commit of this repository).

```
$ uv run replay ci examples/retail_agent/replay.toml --no-store --cassette-dir examples/retail_agent/cassettes/live --mode model-live --client anthropic --json eval/results/m2_model_live.json
PASS cancel_pending_laptop (model-live; 6 exact, 5 live)
PASS modify_address_suite (model-live; 3 exact, 3 live)
PASS return_damaged_bicycle (model-live; 7 exact, 5 live)
PASS exchange_waterproof_boots (model-live; 8 exact, 6 live)
PASS lookup_latest_order_total (model-live; 5 exact, 4 live)
PASS transfer_swapped_refunds (model-live; 1 approximate, 4 exact, 5 live)
6/6 scenario(s) passed: passed

real	1m9.436s
```

Summed from the committed report: $0.2933 and 157,442 tokens, 67.8 s for the
suite; every check passed. The cost is higher than the first run's because
the cache was cold (cache writes are billed at $5 per MTok, reads at $0.20).
The same scenario had the one `approximate` step again.

What this does and does not show: in these two samples Opus 5.5 made the same
tool calls, in the same order, as in its own recording on all six tasks.
How often that holds over repeated runs is the flakiness question, measured
with intervals in M4.

## Seeded regressions in `replay ci` (M2, offline)

`make ci-demo` (also a CI step) replays the retail suite against the
committed fake-model cassettes and against three deliberately broken agents
in `examples/retail_agent/mutants.py`. Each mutant must make `replay ci` exit 1.
Per-scenario outcomes are asserted in `tests/test_ci.py` (which checks `skips_order_lookup` catch `tool_sequence` as well is asserted in `tests/test_retail_agent.py`):

| Agent | Scenarios failed | Checks that caught it |
|---|---|---|
| `agent:run` (unchanged), full / tools-live / model-live with the fake model | 0 / 6 | none |
| `mutants:wrong_write_argument` (one write argument changed) | 4 / 6 (every task with a write) | `rule:matches_ground_truth` (`tool_sequence` passes: the tool names are right) |
| `mutants:skips_order_lookup` (drops `get_order_details`) | 6 / 6 | `tool_sequence`, and `rule:checked_before_writing` on the four write tasks |
| `mutants:first_tool_only` (drops parallel tool calls) | 5 / 6 | `tool_sequence` (the lookup task's recording has no parallel calls, so nothing is dropped) |

This is a demonstration on three mutants, not the mutation study: detection
rates per check type and per mode over 25+ mutations are M4.

## To measure

Nothing yet. Key-requiring evaluations (judge kappa, LLMBar, flake suite) are
listed here, with exact commands, when their harnesses exist (M3 to M6). The
M2 live smoke test above can be re-run with the two commands in its section
(about $0.30 each with `claude-opus-5-5` and a cold prompt cache).
