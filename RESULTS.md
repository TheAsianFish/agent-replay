# Results

Every number in README.md comes from this file. Each one lists its dataset,
method, the command that produced it and the raw output. Sections are added as
the evaluations land. PLAN.md, "Evaluation", lists the full plan.

## Index

Every number this repo claims, where its raw output is, and how to reproduce
it. "Offline" rows need no key and run in seconds to a minute; CI re-runs the
byte-stable ones and fails on any change.

| Claim | Number | Section | Command | Status |
|---|---|---|---|---|
| Recorder works against the real API, replays exactly | 6 envelopes, all `exact` on replay; $0.0140 | Live recording smoke test (M1) | `make test-live` | measured (key) |
| Live retail recordings replay and pass | 6/6 in `full`; 6/6 `model-live` twice ($0.10 warm cache, $0.29 cold) | Retail agent (M2) | `replay ci ... --mode model-live --client anthropic` | measured (key) |
| Seeded regressions fail `replay ci` | 3/3 mutants exit 1 | Seeded regressions (M2) | `make ci-demo` | measured, offline |
| Cost gate on a live run | unchanged agent within 5%; caching removed: 6/6 fail at +76% to +155% | Cost gate (M3) | commands in that section | measured (key) |
| Judge vs deterministic checks (wiring) | 23/24 agree | Judge smoke test (M3) | `make eval-judge-smoke` | measured, offline from cache |
| Fixed-sample gate at α = β = 0.05 | 24 runs (pass on ≥ 19); fixed-20 majority false-passes 13.4% at p = 0.6 | Sequential gate simulation | `make bench-sprt` | measured, offline |
| SPRT runs saved vs the fixed-sample test | — | Sequential gate simulation | `make bench-sprt` once `stats.py` exists | **to measure** |
| Mutation detection, per mode | `full` 7/32; `model-live` 12/12 model regressions; `tools-live` 5/8 tool bugs (8/8 with `fail_on_approximate`); false alarms 0/5 (1/5) | Mutation study (M4) | `make eval-mutations` | measured, offline |
| Live flakiness | Haiku 5.5 119/120; Opus 5.5 30/30 at 41× Haiku's cost per run | `replay flake` on live models (M4) | commands in that section | measured (key) |
| Wilson intervals for those runs | — | `replay flake` on live models | `REPLAY_STATS_IMPL=core` re-run | **to measure** |
| Offline flake gate | fake model passes every scenario; seeded flaky model fails the gate | `replay flake` offline (M4) | `make flake-demo` | measured, offline |
| Dashboard demo data | 480 runs, 14 suites, 6 flake reports, 2 calibration reports | Dashboard demo data (M5) | `make demo-data`, `docker compose up` | measured, offline |
| Judge vs objective labels | 20/21 (95%); majority baseline 71%; 1 false pass | Judge calibration pipeline (M6) | `make eval-judge-objective` | measured, offline from cache |
| LLMBar heuristic floors (Adversarial) | shorter 82.0%, longer 18.0%, first 50.0% (position bias 100%) | LLMBar heuristic baselines (M6) | `make eval-llmbar-heuristics` | measured, offline |
| Recorder overhead | model ~0.7 ms, tool ~0.1 ms, clock / random ~50 µs per call (median) | Recorder overhead and replay speed (M7) | `make bench-replay` | measured, offline (machine-dependent) |
| Full replay vs live call time | suite 222 ms with checks vs 51.0 s (230×) | Recorder overhead and replay speed (M7) | `make bench-replay` | measured, offline (machine-dependent) |
| Wheel installs in a clean venv | wheel 342,782 B with the dashboard; sdist 327,427 B; both install and pass the smoke test | Release readiness (M7) | `make dist` | measured |
| The Action, dry run on the retail suite | pass: exit 0, 0 failed; mutant: exit 1, 5/6 failed | Release readiness (M7) | CI job `action`; locally with `act` | measured |
| The Action fails a seeded regression, rehearsed | `replay ci` 2/6 passed, 4 failed on `rule:matches_ground_truth`; the `replay.yml` job under act fails with exit 1 | The Action on a real failing PR (M8) | `make regression-pr-rehearse`; act | measured, offline |
| The Action on a real PR, passing half | PR #11 (it changes `replay.yml`): 6/6 passed, sticky comment posted by `github-actions[bot]`, identical to a local run | The Action on a real failing PR (M8) | `replay.yml` on PR #11 | measured, observed on GitHub |
| The Action on a real failing PR | — | The Action on a real failing PR (M8) | `make regression-pr` (needs `gh auth login`) | **to measure** |
| Judge-human kappa, self-consistency, probes | — | To measure | `REPLAY_STATS_IMPL=core make eval-judge` (~$1.30) | **to measure** |
| D1 golden set | — | To measure | `make record-golden` (~$2.19) | **to measure** |
| Real judge on LLMBar | — | To measure | `make eval-llmbar` (~$12.73) | **to measure** |
| D3 flake benchmark | — | To measure | `make bench-flake` (not built; ~$21 Opus / ~$0.52 Haiku) | **to measure** |

API spend to date: $3.326 (M1 $0.014, M2 $0.691, M3 $1.408, M4 $1.213, M5 to
M8 $0).

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

## Cost gate on a live `model-live` run (M3)

Does a relative cost budget catch a cost regression that leaves behaviour
unchanged, without flagging the unchanged agent? One run per agent per task;
not a variance estimate (M4).

- **Dataset:** the six live retail recordings (`examples/retail_agent/cassettes/live`).
- **Baseline:** `examples/retail_agent/baseline-live.json`, measured from those
  recordings with
  `uv run replay baseline examples/retail_agent/replay-live.toml -o examples/retail_agent/baseline-live.json`
  (full mode, so it is the recorded cost: $0.2949 over the six tasks).
- **Gate:** the suite's default check
  `{ type = "budget", max_cost_increase_pct = 25, max_tokens_increase_pct = 25, max_p95_latency_increase_pct = 50 }`.
- **Regression:** `examples.retail_agent.mutants:no_prompt_cache`, the same
  agent with `cache_control` removed: same prompts, same tools, every turn pays
  the full input price for the prefix.
- **Model:** `claude-opus-5-5`, effort `low`, tools served from the cassette.
  The judge check ran offline (`REPLAY_JUDGE_OFFLINE=1`) and cost nothing. At
  the time an offline miss was reported as an advisory "verdict unknown"; it
  now breaks the scenario (exit 2), so to re-run these commands drop
  `REPLAY_JUDGE_OFFLINE=1` and the judge runs live (about $0.10 per six runs).
- **Date:** 2026-10-09. Spend: $0.2921 (agent) + $0.6816 (mutant).

**Commands:**

```
REPLAY_JUDGE_OFFLINE=1 uv run replay ci examples/retail_agent/replay-live.toml --no-store --mode model-live --client anthropic --json eval/results/m3_cost_gate_agent.json
REPLAY_JUDGE_OFFLINE=1 uv run replay ci examples/retail_agent/replay-live.toml --no-store --mode model-live --client anthropic --agent examples.retail_agent.mutants:no_prompt_cache --json eval/results/m3_cost_gate_no_prompt_cache.json
```

Exit codes 0 and 1. Raw reports: `eval/results/m3_cost_gate_*.json` (their
`git_sha`, like the baseline file's, is the `$GITHUB_SHA` of the runner they
were made on, not a commit of this repository; `replay` now ignores a SHA
that is not in the checkout). Deltas are against the baseline, from the
budget check's details:

| Task | agent cost | Δ cost | Δ tokens | no_prompt_cache cost | Δ cost | Δ tokens |
|---|---|---|---|---|---|---|
| cancel_pending_laptop | $0.0460 | −0.3% | −0.0% | $0.1155 | +150.4% | +0.0% |
| modify_address_suite | $0.0403 | +0.1% | +0.0% | $0.0710 | +76.3% | −0.0% |
| return_damaged_bicycle | $0.0526 | −0.1% | −0.0% | $0.1245 | +136.2% | +0.0% |
| exchange_waterproof_boots | $0.0656 | +0.3% | +0.0% | $0.1671 | +155.5% | +0.0% |
| lookup_latest_order_total | $0.0429 | −5.0% | −0.6% | $0.0901 | +99.5% | −0.3% |
| transfer_swapped_refunds | $0.0446 | −1.4% | −0.2% | $0.1133 | +150.4% | −0.2% |
| **Result** | **6/6 pass** | | | **6/6 fail (budget)** | | |

The unchanged agent stayed within 5% of its baseline cost. The mutant used the
same tokens (within 0.3% of the baseline) but cost 1.8× to 2.6× as much, and only the cost
limit caught it: a token budget would not have. p95 model latency moved between
−36% and +16% in the agent's run and between −21% and +7% in the mutant's,
inside the 50% limit both times; with one run per task, latency is too noisy to
gate tightly.

## Judge smoke test against the deterministic checks (M3)

Does the `judge` check, uncalibrated, agree with the suite's own rules on runs
where the answer is known? This is a wiring test on 24 runs, not a
calibration. The reference is the deterministic verdict (every non-judge check
passed, including τ-bench ground truth), not human labels, and there is one
judge sample per run. Judge-human kappa is M6.

- **Dataset:** the six live retail recordings replayed in `full` mode with the
  real agent and with the three M2 mutants (24 runs).
- **Judge:** `claude-opus-5-5`, effort `low`, structured outputs, rubric
  `examples/retail_agent/rubric.yaml` (three criteria: `resolved`,
  `policy_followed`, `accurate_report`), version 1.
- **Date:** 2026-10-09. Spend: $0.1011 for the six recordings plus $0.2572
  for 17 calls on the 18 mutant runs (one mutant transcript repeated another's
  and came from the cache). The mutants were judged into a scratch cache, so
  the two `wrong_write_argument` read-only runs, whose transcripts equal the
  real agent's, were judged again; those duplicate verdicts were not committed.
  The committed cache therefore holds 21 verdicts whose recorded cost sums to
  $0.3302 (each entry keeps its `cost_usd`). The rubric system block (2,243
  tokens) was read from the prompt cache on every call.

The verdicts are committed in `examples/retail_agent/judge-cache/`, so the
command below re-derives the table offline and for free, and
`tests/test_judge_smoke.py` asserts it.

**Command:** `make eval-judge-smoke` (= `uv run python eval/judge_smoke.py --offline --out eval/results/judge_smoke.json`)

```
| variant | scenario | deterministic | judge | resolved | policy_followed | accurate_report |
|---|---|---|---|---|---|---|
| agent | cancel_pending_laptop | pass | pass | pass | pass | pass |
| agent | modify_address_suite | pass | pass | pass | pass | pass |
| agent | return_damaged_bicycle | pass | pass | pass | pass | pass |
| agent | exchange_waterproof_boots | pass | pass | pass | pass | pass |
| agent | lookup_latest_order_total | pass | pass | pass | pass | pass |
| agent | transfer_swapped_refunds | pass | pass | pass | pass | pass |
| wrong_write_argument | cancel_pending_laptop | fail | fail | fail | pass | pass |
| wrong_write_argument | modify_address_suite | fail | fail | fail | pass | pass |
| wrong_write_argument | return_damaged_bicycle | fail | fail | fail | fail | pass |
| wrong_write_argument | exchange_waterproof_boots | fail | fail | pass | fail | pass |
| wrong_write_argument | lookup_latest_order_total | pass | pass | pass | pass | pass |
| wrong_write_argument | transfer_swapped_refunds | pass | pass | pass | pass | pass |
| skips_order_lookup | cancel_pending_laptop | fail | fail | pass | fail | pass |
| skips_order_lookup | modify_address_suite | fail | fail | pass | fail | pass |
| skips_order_lookup | return_damaged_bicycle | fail | fail | pass | fail | pass |
| skips_order_lookup | exchange_waterproof_boots | fail | fail | pass | fail | pass |
| skips_order_lookup | lookup_latest_order_total | fail | fail | fail | fail | fail |
| skips_order_lookup | transfer_swapped_refunds | fail | fail | pass | fail | pass |
| first_tool_only | cancel_pending_laptop | fail | pass | pass | pass | pass |
| first_tool_only | modify_address_suite | fail | fail | pass | fail | pass |
| first_tool_only | return_damaged_bicycle | fail | fail | pass | fail | pass |
| first_tool_only | exchange_waterproof_boots | fail | fail | pass | fail | pass |
| first_tool_only | lookup_latest_order_total | fail | fail | fail | fail | fail |
| first_tool_only | transfer_swapped_refunds | fail | fail | pass | fail | pass |

- judge answered 24/24 runs
- agreement with the deterministic verdict: 23/24
- runs the deterministic checks failed that the judge also failed: 15/16
- runs the deterministic checks passed that the judge also passed: 8/8
```

The one disagreement (`first_tool_only` on `cancel_pending_laptop`) failed
only `tool_sequence`: the run reached the right outcome with its parallel
lookups served one per turn, so the judge's pass is defensible. On the
`wrong_write_argument` exchange the judge put the wrong payment method under
`policy_followed`, not `resolved`, so per-criterion verdicts should not be read
as a taxonomy before calibration. With a deterministic reference this agreeable
on a small, easy set, the numbers say the wiring works, not that the judge is
accurate.

## Sequential gate simulation (M3, extended in M4; offline)

The CI gate's operating characteristic. `agent_replay/stats.py` (the SPRT,
Wilson and kappa) is still a specification with unimplemented bodies, so only
`agent_replay/stats_baseline.py` runs: `max_runs` runs, then pass on a
majority (pass rate ≥ (p_fail + p_pass) / 2). M4 adds the yardstick that
"runs saved" will be measured against: the smallest fixed-sample test that
meets the same error targets, computed exactly from the binomial distribution
(`fixed_sample_design` in `eval/sprt_sim.py`, checked against scipy in
`tests/test_sprt_sim.py`).

- **Method:** for each true pass rate p, 2000 simulated sequences of
  Bernoulli(p) run outcomes (seed 0) through a fresh gate; p_fail = 0.6,
  p_pass = 0.9, α = β = 0.05, max_runs = 20.
- **Command:** `make bench-sprt` (= `uv run python eval/sprt_sim.py --out eval/results/sprt_sim.json`), about 1.3 s. CI re-runs it and fails if
  `eval/results/sprt_sim.json` changes.

```
core: agent_replay/stats.py is not implemented yet; skipped
SPRT gate simulation: p_fail=0.6, p_pass=0.9, alpha=0.05, beta=0.05, max_runs=20, trials=2000, seed=0

| true p | baseline mean runs | baseline P(pass) |
|---:|---:|---:|
| 0.05 | 20.00 | 0.000 |
| 0.10 | 20.00 | 0.000 |
| 0.15 | 20.00 | 0.000 |
| 0.20 | 20.00 | 0.000 |
| 0.25 | 20.00 | 0.000 |
| 0.30 | 20.00 | 0.000 |
| 0.35 | 20.00 | 0.000 |
| 0.40 | 20.00 | 0.001 |
| 0.45 | 20.00 | 0.007 |
| 0.50 | 20.00 | 0.019 |
| 0.55 | 20.00 | 0.060 |
| 0.60 | 20.00 | 0.134 |
| 0.65 | 20.00 | 0.241 |
| 0.70 | 20.00 | 0.421 |
| 0.75 | 20.00 | 0.618 |
| 0.80 | 20.00 | 0.805 |
| 0.85 | 20.00 | 0.935 |
| 0.90 | 20.00 | 0.993 |
| 0.95 | 20.00 | 1.000 |
| 1.00 | 20.00 | 1.000 |

- baseline: expected runs 20.00 at p_fail, 20.00 at p_pass; false-pass rate at p_fail 0.134, false-fail rate at p_pass 0.007; runs vs the fixed-sample test: 83% at p_fail, 83% at p_pass
- fixed-sample test meeting alpha=0.05, beta=0.05 (exact binomial): 24 runs, pass on at least 19; false-pass rate 0.040, false-fail rate 0.028

wrote eval/results/sprt_sim.json
```

Two findings so far. The fixed-20 majority gate passes 13.4% of scenarios
whose true pass rate is 0.6 (exact binomial: P(X ≥ 15 | n = 20, p = 0.6) ≈
0.126), well over the 5% target, and always spends 20 runs. Meeting
α = β = 0.05 with a fixed sample takes **24 runs** (pass on at least 19; exact
error rates 4.0% and 2.8%). The SPRT's expected runs at p_fail and p_pass, as
a fraction of those 24, are the "runs saved" number; the same command prints
them once `stats.py` is implemented (see "To measure").

## Judge live test (M3)

`uv run pytest -q -s -m live -k judge` (`tests/test_live_judge.py`): the
`claude-opus-5-5` judge (effort `low`) on the replayed `cancel_pending_laptop`
run must return a well-formed verdict for every criterion at a positive cost.
Raw output of the run on 2026-10-09:

```
judge anthropic:claude-opus-5-5:effort=low: cost $0.025315, 3501.124 ms
usage {'input_tokens': 2300, 'output_tokens': 245, 'cache_creation_input_tokens': 2243, 'cache_read_input_tokens': 0}
resolved: pass - Laptop order #W2417020 was pending, so it couldn't be returned; the agent took the user's fallback and cancelled it with 'no longer needed' (call 6).
policy_followed: pass - The agent authenticated via find_user_id_by_name_zip (call 1) and fetched the order (call 3) before cancelling. The order was pending, the reason was valid, and the refund went to the original gift card.
accurate_report: pass - The reply says #W2417020 is cancelled, which matches the cancelled status in the call 6 result.
1 passed
```

## Mutation study (M4, offline)

Which injected regressions does each check type catch, and in which replay
mode? 32 scripted mutations of the retail agent, each one a realistic bug,
plus 5 behaviour-preserving refactors that must not fail anything.

- **Dataset (D4):** `eval/mutations/`, three kinds:
  - **agent** (12): orchestration-code bugs in a `RetailAgent` subclass:
    skipped lookups, a changed write argument, dropped parallel calls,
    truncated tool results, a turn limit too low, mismatched `tool_use` ids,
    a stripped policy, and so on (the three M2 mutants are among them);
  - **model** (12): simulated prompt or model regressions, written as
    wrappers around the deterministic fake model's policy: skipping
    authentication, a wrong cancel reason, refunding to the wrong payment
    method, transferring instead of acting, a rounded total, repeated
    lookups, and so on;
  - **tool** (8): tool-code bugs behind the same name and schema: a cancel
    that leaves the status unchanged, a return that ignores the item ids,
    stale order details, and so on.
  The **benign** refactors are an extracted loop, parallel tool execution, a
  different JSON layout for tool results, a higher turn limit and an async
  agent.
- **Suite:** `examples/retail_agent/replay.toml`, six τ-bench tasks, its
  committed fake-model cassettes and every check type except `judge` (no
  verdict cache exists for mutant transcripts). M4 added
  `rule:db_matches_ground_truth`, τ-bench's reward: the database after the
  run equals the one the ground-truth actions leave. It applies only when
  every tool call really ran; otherwise it passes as "not applicable".
- **Configurations:** `full` (everything from the cassette: the
  stub-everything baseline), `tools-live`, `model-live` (the fake model, or
  the mutated one, plays the live model), `model-live` with the `fallback`
  policy (a tool call with no recorded envelope left runs for real) and
  `tools-live+strict` (`tools-live` with `fail_on_approximate`: a served
  model step whose request differs from the recording fails the scenario).
  Strict policy otherwise.
- **Method:** a mutation is detected when the suite fails. A check type
  catches it when it is among the blocking failed checks of at least one
  scenario; `run_error:<Type>` is a run that ended in that exception; "alone"
  means no other check type caught that mutation in that configuration.
- **Command:** `make eval-mutations` (= `uv run python eval/mutation_study.py --out eval/results/mutations.json`), about 25 s. Deterministic: CI
  re-runs it and fails if `eval/results/mutations.json` changes.

Raw output (the per-mutation matrix is folded below):

```
## Mutation study: retail suite

32 mutations (12 agent, 12 model, 8 tool) and 5 benign refactors, each replayed against the six-scenario suite `examples/retail_agent/replay.toml` in 5 configurations. `full` serves every call from the cassette: it is the stub-everything baseline.

Detected: the suite fails. A check type catches a mutation when it is among the blocking failed checks of some scenario; `run_error:<Type>` is a run that ended in that exception. Alone: no other check type caught it in that configuration.

### Detection rate by configuration

| configuration | all | agent | model | tool |
|---|---|---|---|---|
| full | 7/32 (22%) | 7/12 (58%) | 0/12 (0%) | 0/8 (0%) |
| tools-live | 12/32 (38%) | 7/12 (58%) | 0/12 (0%) | 5/8 (62%) |
| model-live | 22/32 (69%) | 10/12 (83%) | 12/12 (100%) | 0/8 (0%) |
| model-live+fallback | 22/32 (69%) | 10/12 (83%) | 12/12 (100%) | 0/8 (0%) |
| tools-live+strict | 19/32 (59%) | 11/12 (92%) | 0/12 (0%) | 8/8 (100%) |

### Mutations caught per check type (caught / all mutations, alone)

| check type | full | tools-live | model-live | model-live+fallback | tools-live+strict |
|---|---|---|---|---|---|
| `arg_schema` | 1/32 (alone 0) | 1/32 (alone 0) | 6/32 (alone 0) | 6/32 (alone 0) | 1/32 (alone 0) |
| `budget` | 0/32 (alone 0) | 0/32 (alone 0) | 3/32 (alone 1) | 3/32 (alone 1) | 0/32 (alone 0) |
| `forbidden_tools` | 0/32 (alone 0) | 0/32 (alone 0) | 0/32 (alone 0) | 0/32 (alone 0) | 0/32 (alone 0) |
| `max_tool_calls` | 0/32 (alone 0) | 0/32 (alone 0) | 1/32 (alone 0) | 1/32 (alone 0) | 0/32 (alone 0) |
| `no_approximate_steps` | 0/32 (alone 0) | 0/32 (alone 0) | 0/32 (alone 0) | 0/32 (alone 0) | 14/32 (alone 7) |
| `output_schema` | 2/32 (alone 0) | 2/32 (alone 0) | 7/32 (alone 0) | 7/32 (alone 0) | 2/32 (alone 0) |
| `required_tools` | 4/32 (alone 0) | 4/32 (alone 0) | 9/32 (alone 0) | 9/32 (alone 0) | 4/32 (alone 0) |
| `rule:checked_before_writing` | 2/32 (alone 0) | 2/32 (alone 0) | 2/32 (alone 0) | 2/32 (alone 0) | 2/32 (alone 0) |
| `rule:db_matches_ground_truth` | 0/32 (alone 0) | 7/32 (alone 5) | 0/32 (alone 0) | 0/32 (alone 0) | 7/32 (alone 2) |
| `rule:matches_ground_truth` | 4/32 (alone 1) | 4/32 (alone 0) | 19/32 (alone 6) | 19/32 (alone 6) | 4/32 (alone 0) |
| `run_error:JSONDecodeError` | 0/32 (alone 0) | 0/32 (alone 0) | 2/32 (alone 0) | 2/32 (alone 0) | 0/32 (alone 0) |
| `run_error:KeyError` | 0/32 (alone 0) | 0/32 (alone 0) | 1/32 (alone 0) | 1/32 (alone 0) | 0/32 (alone 0) |
| `run_error:RuntimeError` | 1/32 (alone 0) | 1/32 (alone 0) | 2/32 (alone 0) | 3/32 (alone 0) | 1/32 (alone 0) |
| `run_error:UnrecordedCallError` | 1/32 (alone 0) | 0/32 (alone 0) | 4/32 (alone 0) | 0/32 (alone 0) | 0/32 (alone 0) |
| `tool_sequence` | 5/32 (alone 0) | 5/32 (alone 0) | 13/32 (alone 0) | 13/32 (alone 0) | 5/32 (alone 0) |

### Benign refactors (false alarms; approximate steps)

| refactor | full | tools-live | model-live | model-live+fallback | tools-live+strict |
|---|---|---|---|---|---|
| `extracted_loop` | pass; 0 approx | pass; 0 approx | pass; 0 approx | pass; 0 approx | pass; 0 approx |
| `parallel_tools` | pass; 0 approx | pass; 0 approx | pass; 0 approx | pass; 0 approx | pass; 0 approx |
| `compact_tool_json` | pass; 17 approx | pass; 17 approx | pass; 0 approx | pass; 0 approx | FALSE ALARM: no_approximate_steps; 17 approx |
| `more_turns` | pass; 0 approx | pass; 0 approx | pass; 0 approx | pass; 0 approx | pass; 0 approx |
| `async_agent` | pass; 0 approx | pass; 0 approx | pass; 0 approx | pass; 0 approx | pass; 0 approx |

False alarms: full 0/5, tools-live 0/5, model-live 0/5, model-live+fallback 0/5, tools-live+strict 1/5.
Survived every configuration: `drops_error_flag`.
```

<details><summary>Per-mutation matrix</summary>

### Per mutation (check types that caught it; — survived)

| mutation | target | full | tools-live | model-live | model-live+fallback | tools-live+strict |
|---|---|---|---|---|---|---|
| `skips_order_lookup` | agent | `required_tools`, `rule:checked_before_writing`, `tool_sequence` | `required_tools`, `rule:checked_before_writing`, `tool_sequence` | `arg_schema`, `budget`, `output_schema`, `required_tools`, `rule:matches_ground_truth`, `run_error:RuntimeError`, `tool_sequence` | `arg_schema`, `budget`, `output_schema`, `required_tools`, `rule:matches_ground_truth`, `run_error:RuntimeError`, `tool_sequence` | `no_approximate_steps`, `required_tools`, `rule:checked_before_writing`, `tool_sequence` |
| `wrong_write_argument` | agent | `rule:matches_ground_truth` | `rule:db_matches_ground_truth`, `rule:matches_ground_truth` | `rule:matches_ground_truth` | `rule:matches_ground_truth` | `no_approximate_steps`, `rule:db_matches_ground_truth`, `rule:matches_ground_truth` |
| `first_tool_only` | agent | `required_tools`, `rule:checked_before_writing`, `tool_sequence` | `required_tools`, `rule:checked_before_writing`, `tool_sequence` | `budget` | `budget` | `no_approximate_steps`, `required_tools`, `rule:checked_before_writing`, `tool_sequence` |
| `drops_error_flag` | agent | — | — | — | — | — |
| `truncates_tool_results` | agent | — | — | `arg_schema`, `output_schema`, `required_tools`, `rule:matches_ground_truth`, `run_error:JSONDecodeError`, `tool_sequence` | `arg_schema`, `output_schema`, `required_tools`, `rule:matches_ground_truth`, `run_error:JSONDecodeError`, `tool_sequence` | `no_approximate_steps` |
| `max_turns_too_low` | agent | `arg_schema`, `output_schema`, `required_tools`, `rule:matches_ground_truth`, `run_error:RuntimeError`, `tool_sequence` | `arg_schema`, `output_schema`, `required_tools`, `rule:db_matches_ground_truth`, `rule:matches_ground_truth`, `run_error:RuntimeError`, `tool_sequence` | `arg_schema`, `output_schema`, `required_tools`, `rule:matches_ground_truth`, `run_error:RuntimeError`, `tool_sequence` | `arg_schema`, `output_schema`, `required_tools`, `rule:matches_ground_truth`, `run_error:RuntimeError`, `tool_sequence` | `arg_schema`, `output_schema`, `required_tools`, `rule:db_matches_ground_truth`, `rule:matches_ground_truth`, `run_error:RuntimeError`, `tool_sequence` |
| `returns_empty_answer` | agent | `output_schema`, `rule:matches_ground_truth` | `output_schema`, `rule:matches_ground_truth` | `output_schema`, `rule:matches_ground_truth` | `output_schema`, `rule:matches_ground_truth` | `output_schema`, `rule:matches_ground_truth` |
| `repeats_write_calls` | agent | `rule:matches_ground_truth`, `run_error:UnrecordedCallError`, `tool_sequence` | `rule:matches_ground_truth`, `tool_sequence` | `rule:matches_ground_truth`, `run_error:UnrecordedCallError`, `tool_sequence` | `rule:matches_ground_truth`, `tool_sequence` | `rule:matches_ground_truth`, `tool_sequence` |
| `mismatches_tool_use_ids` | agent | — | — | `required_tools`, `rule:matches_ground_truth`, `run_error:KeyError`, `tool_sequence` | `required_tools`, `rule:matches_ground_truth`, `run_error:KeyError`, `tool_sequence` | `no_approximate_steps` |
| `unregisters_product_tool` | agent | `required_tools`, `tool_sequence` | `required_tools`, `tool_sequence` | `required_tools`, `rule:matches_ground_truth`, `run_error:JSONDecodeError`, `tool_sequence` | `required_tools`, `rule:matches_ground_truth`, `run_error:JSONDecodeError`, `tool_sequence` | `no_approximate_steps`, `required_tools`, `tool_sequence` |
| `strips_policy` | agent | — | — | — | — | `no_approximate_steps` |
| `truncates_history` | agent | — | — | `arg_schema`, `output_schema`, `required_tools`, `rule:matches_ground_truth`, `tool_sequence` | `arg_schema`, `output_schema`, `required_tools`, `rule:matches_ground_truth`, `tool_sequence` | `no_approximate_steps` |
| `skips_authentication` | model | — | — | `rule:checked_before_writing`, `tool_sequence` | `rule:checked_before_writing`, `tool_sequence` | — |
| `skips_order_check` | model | — | — | `required_tools`, `rule:checked_before_writing`, `tool_sequence` | `required_tools`, `rule:checked_before_writing`, `tool_sequence` | — |
| `wrong_cancel_reason` | model | — | — | `rule:matches_ground_truth` | `rule:matches_ground_truth` | — |
| `product_ids_as_item_ids` | model | — | — | `rule:matches_ground_truth` | `rule:matches_ground_truth` | — |
| `refunds_to_first_method` | model | — | — | `rule:matches_ground_truth` | `rule:matches_ground_truth` | — |
| `suite_in_address1` | model | — | — | `rule:matches_ground_truth` | `rule:matches_ground_truth` | — |
| `wrong_exchange_variant` | model | — | — | `rule:matches_ground_truth` | `rule:matches_ground_truth` | — |
| `transfers_instead_of_acting` | model | — | — | `arg_schema`, `required_tools`, `rule:matches_ground_truth`, `run_error:UnrecordedCallError`, `tool_sequence` | `arg_schema`, `required_tools`, `rule:matches_ground_truth`, `tool_sequence` | — |
| `transfers_after_acting` | model | — | — | `rule:matches_ground_truth`, `run_error:UnrecordedCallError`, `tool_sequence` | `rule:matches_ground_truth`, `tool_sequence` | — |
| `claims_without_acting` | model | — | — | `arg_schema`, `required_tools`, `rule:matches_ground_truth`, `tool_sequence` | `arg_schema`, `required_tools`, `rule:matches_ground_truth`, `tool_sequence` | — |
| `repeats_lookups` | model | — | — | `budget`, `max_tool_calls`, `output_schema`, `rule:matches_ground_truth`, `run_error:UnrecordedCallError`, `tool_sequence` | `budget`, `max_tool_calls`, `output_schema`, `rule:matches_ground_truth`, `run_error:RuntimeError`, `tool_sequence` | — |
| `rounds_reported_total` | model | — | — | `output_schema`, `rule:matches_ground_truth` | `output_schema`, `rule:matches_ground_truth` | — |
| `cancel_keeps_status` | tool | — | `rule:db_matches_ground_truth` | — | — | `rule:db_matches_ground_truth` |
| `cancel_skips_gift_card_refund` | tool | — | `rule:db_matches_ground_truth` | — | — | `rule:db_matches_ground_truth` |
| `address_drops_address2` | tool | — | `rule:db_matches_ground_truth` | — | — | `no_approximate_steps`, `rule:db_matches_ground_truth` |
| `return_ignores_item_ids` | tool | — | `rule:db_matches_ground_truth` | — | — | `no_approximate_steps`, `rule:db_matches_ground_truth` |
| `exchange_flips_price_difference` | tool | — | `rule:db_matches_ground_truth` | — | — | `no_approximate_steps`, `rule:db_matches_ground_truth` |
| `user_lookup_builds_id` | tool | — | — | — | — | `no_approximate_steps` |
| `product_details_all_unavailable` | tool | — | — | — | — | `no_approximate_steps` |
| `order_details_stale_status` | tool | — | — | — | — | `no_approximate_steps` |

</details>

What this shows:

- **Each mode catches what it runs live, and only that.** `full` serves the
  recording's model replies and tool results, so a model or tool bug cannot
  take effect there; a tool bug cannot take effect in `model-live` either.
  Of the 32 mutations, 20 live in a component `full` never runs, so a suite
  that only stubs everything passes them by construction: it caught 7/32,
  and 7/12 of the mutations that could take effect. Over the mutations that
  can take effect in each configuration: `tools-live` 12/20, `model-live`
  22/24, `tools-live+strict` 19/20. This is the measured case for cut-point
  replay: where the bug lives decides which cut point can see it.
- **Ground-truth rules carry most of the weight.**
  `rule:matches_ground_truth` caught 19/32 in `model-live`, 6 of them alone
  (a wrong cancel reason, a wrong exchange variant and the like leave the
  tool sequence intact). For 9 of those 19 mutations a run ended in an
  exception, and the rule and the other checks fail there as a side effect
  of the run stopping early. The state check `rule:db_matches_ground_truth` caught 7/32
  in `tools-live`, 5 alone; it cannot apply in either `model-live`
  configuration (served tool calls never touch the database), so its 0/32
  there is structural. `tool_sequence` caught 13 but never alone. `budget`
  (model-call limit) caught dropped parallel calls alone.
- **The drift signal is already there, and it is a trade-off.** Every
  served step whose request differs from the recording is flagged
  `approximate`; the four lenient configurations record the count without
  failing on it. `tools-live+strict` (`fail_on_approximate`) turns it into a
  check: `no_approximate_steps` caught 14/32, 7 of them alone, including
  all 8 tool bugs (the three read-only ones change only what a tool
  returns, which the next model request carries) and the stripped policy
  (every later request differs). The price is 1/5 false alarms: the
  JSON-layout refactor changes the same request bytes. The lenient
  configurations stayed at 0/5.
- **False alarms.** 0/5 benign refactors failed in the four lenient
  configurations and 1/5 in the strict one. Five refactors is a small
  control group, and four of them send byte-identical requests, so only
  `compact_tool_json` exercises the matcher.
- **One mutation survives everything.** `drops_error_flag` (not marking a
  failed tool result `is_error`) is equivalent to the original on this
  suite: no recorded run has a failing tool, and it leaves 0 approximate
  steps in every configuration.
- **Caveats.** The fake model stands in for a live one; it ignores the
  prompt, so `strips_policy` is invisible in `model-live`. Three
  `run_error` catches (`JSONDecodeError` for `truncates_tool_results` and
  `unregisters_product_tool`, `KeyError` for `mismatches_tool_use_ids`) are
  the fake model crashing on a conversation the mutant corrupted; a real
  model would misbehave differently. With `fallback`, a tool call that has
  no recorded envelope left runs for real instead of ending the run, but a
  call whose arguments merely differ is still served the recorded result
  (flagged `approximate`), so the checks judge the requested arguments, not
  a real tool's answer; the same 22 are caught.

## `replay flake` on live models (M4)

Does the retail agent reach the same outcomes run after run when the model
is live, and does a cheaper model? `replay flake` replays each scenario
repeatedly in `model-live` (tools served from the Opus recordings), counts
passes and classifies each scenario passing, flaky or failing. These are the
first repeated live runs in the project.

- **Dataset:** the six live retail recordings
  (`examples/retail_agent/cassettes/live`), suite
  `examples/retail_agent/replay.toml` (every check of the mutation study, no
  judge).
- **Agents:** the shipped agent (`claude-opus-5-5`, effort `low`) and
  `examples.retail_agent.variants:run_haiku`, the same agent with
  `claude-haiku-5-5`. Haiku cache prices in `agent_replay/pricing.py` are
  the standard multipliers of its $0.10 input price, not separately
  published figures.
- **Statistics:** `REPLAY_STATS_IMPL=baseline`, because `stats.py` is not
  written yet. The gate is therefore the fixed-k majority stand-in (k =
  `--max-runs`), and the printed intervals are Wald intervals, which collapse
  to [1.00, 1.00] at k/k. Read the pass counts; Wilson intervals come with
  the core implementation (see "To measure").
- **Date:** 2026-10-10. Spend: $0.1045 (Haiku, 120 runs), $1.0696 (Opus, 30
  runs), $0.00137 (one Haiku smoke run before them).

**Haiku 5.5, 20 runs per scenario.** Raw JSON: `eval/results/m4_flake_haiku.json`.

```
$ uv run replay flake examples/retail_agent/replay.toml --no-store --cassette-dir examples/retail_agent/cassettes/live --client anthropic --agent examples.retail_agent.variants:run_haiku --max-runs 20 --concurrency 4 --json eval/results/m4_flake_haiku.json
PASS cancel_pending_laptop (passing 20/20 (100%), interval [1.00, 1.00])
PASS modify_address_suite (passing 20/20 (100%), interval [1.00, 1.00])
PASS return_damaged_bicycle (passing 20/20 (100%), interval [1.00, 1.00])
PASS exchange_waterproof_boots (passing 20/20 (100%), interval [1.00, 1.00])
PASS lookup_latest_order_total (passing 20/20 (100%), interval [1.00, 1.00])
PASS transfer_swapped_refunds (flaky 19/20 (95%), interval [0.85, 1.00])
    error:UnrecordedCallError: failed 1/20 runs
    forbidden_tools: failed 1/20 runs
    rule:matches_ground_truth: failed 1/20 runs
    tool_sequence: failed 1/20 runs
6/6 scenario(s) passed the gate (1 flaky; 120 runs, $0.1045; gate: fixed 20 
runs, pass on a pass rate of at least 0.75 (the baseline stand-in for the SPRT);
Wald intervals): passed
```

The one failure is the last of the 20 runs of `transfer_swapped_refunds`
(index 19 in the JSON), where the user asks
to refund each order to the other order's payment method. Policy allows only
the original method or a gift card, so the right action is a transfer to a
human. In that run Haiku instead called `return_delivered_order_items` on
order #W5490111 with `paypal_9497703`. That is a policy breach, and
`forbidden_tools`, `rule:matches_ground_truth` and `tool_sequence` all
caught it. The other 19 runs of that task, and all 100 runs of the other five
tasks, passed.

**Opus 5.5, 5 runs per scenario.** Raw JSON: `eval/results/m4_flake_opus.json`.

```
$ uv run replay flake examples/retail_agent/replay.toml --no-store --cassette-dir examples/retail_agent/cassettes/live --client anthropic --max-runs 5 --concurrency 3 --json eval/results/m4_flake_opus.json
PASS cancel_pending_laptop (passing 5/5 (100%), interval [1.00, 1.00])
PASS modify_address_suite (passing 5/5 (100%), interval [1.00, 1.00])
PASS return_damaged_bicycle (passing 5/5 (100%), interval [1.00, 1.00])
PASS exchange_waterproof_boots (passing 5/5 (100%), interval [1.00, 1.00])
PASS lookup_latest_order_total (passing 5/5 (100%), interval [1.00, 1.00])
PASS transfer_swapped_refunds (passing 5/5 (100%), interval [1.00, 1.00])
6/6 scenario(s) passed the gate (0 flaky; 30 runs, $1.0696; gate: fixed 5 runs, 
pass on a pass rate of at least 0.75 (the baseline stand-in for the SPRT); Wald 
intervals): passed
```

| Agent | Runs | Passed | Flaky scenarios | Cost | Cost per run |
|---|---|---|---|---|---|
| `claude-opus-5-5` | 30 | 30 | 0/6 | $1.0696 | $0.0357 |
| `claude-haiku-5-5` | 120 | 119 | 1/6 | $0.1045 | $0.0009 |

What this does and does not show: Haiku ran these six tasks for about 1/41
of Opus's cost per run and failed one run in 120, a policy breach that a
single CI run would have caught only by luck. With 30 runs of Opus and one
failure in 120 for Haiku, the data cannot tell the two models' failure rates
apart; it does show that the swap is not free on the refund-policy task. The
gate passed Haiku's flaky scenario (19/20 is above the majority threshold)
while the report flagged it flaky; separating "passes the gate" from
"never fails" is the point of the classification.

**Provenance, and a repeat.** Both JSON reports record commit `ff07cef`,
which predates the scheduler rework in `1342707` (the window bound and the
`broken` handling); with the fixed-k baseline gate every run is counted, so
neither number depends on it. A repeat with 5 runs per scenario on the
reviewed engine (`7ac313c`, `eval/results/m4_flake_haiku_5runs.json`) passed
30/30 for $0.0379:

```
$ uv run replay flake examples/retail_agent/replay.toml --no-store --cassette-dir examples/retail_agent/cassettes/live --client anthropic --agent examples.retail_agent.variants:run_haiku --max-runs 5 --concurrency 4 --json eval/results/m4_flake_haiku_5runs.json
PASS cancel_pending_laptop (passing 5/5 (100%), interval [1.00, 1.00])
PASS modify_address_suite (passing 5/5 (100%), interval [1.00, 1.00])
PASS return_damaged_bicycle (passing 5/5 (100%), interval [1.00, 1.00])
PASS exchange_waterproof_boots (passing 5/5 (100%), interval [1.00, 1.00])
PASS lookup_latest_order_total (passing 5/5 (100%), interval [1.00, 1.00])
PASS transfer_swapped_refunds (passing 5/5 (100%), interval [1.00, 1.00])
6/6 scenario(s) passed the gate (0 flaky; 30 runs, $0.0379; gate: fixed 5 runs, 
pass on a pass rate of at least 0.75 (the baseline stand-in for the SPRT); Wald 
intervals): passed
```

Its cost per run ($0.0013) is above the 120-run figure ($0.0009): with
fewer runs per scenario, a larger share of each run's prompt is a cache
write rather than a cache read.

## `replay flake` offline (M4)

`make flake-demo` (a CI step) runs the engine with no network. The plain fake
model passes every scenario's gate. A seeded flaky fake model
(`examples/retail_agent/flaky_model.py`) repeats its first lookup with a
per-task probability (0 for three tasks, 0.05, 0.3 and 1.0 for the others),
and must fail the gate (exit 1):

```
$ uv run replay flake examples/retail_agent/replay.toml --no-store --client examples.retail_agent.flaky_model:make_client
PASS cancel_pending_laptop (passing 20/20 (100%), interval [1.00, 1.00])
PASS modify_address_suite (passing 20/20 (100%), interval [1.00, 1.00])
PASS return_damaged_bicycle (flaky 19/20 (95%), interval [0.85, 1.00])
    arg_schema:return_delivered_order_items: failed 1/20 runs
    error:UnrecordedCallError: failed 1/20 runs
    rule:matches_ground_truth: failed 1/20 runs
    tool_sequence: failed 1/20 runs
FAIL exchange_waterproof_boots (flaky 14/20 (70%), interval [0.50, 0.90])
    error:UnrecordedCallError: failed 6/20 runs
    required_tools: failed 6/20 runs
    rule:matches_ground_truth: failed 6/20 runs
    tool_sequence: failed 6/20 runs
PASS lookup_latest_order_total (passing 20/20 (100%), interval [1.00, 1.00])
FAIL transfer_swapped_refunds (failing 0/20 (0%), interval [0.00, 0.00])
    error:UnrecordedCallError: failed 20/20 runs
    required_tools: failed 20/20 runs
    rule:matches_ground_truth: failed 20/20 runs
    tool_sequence: failed 20/20 runs
4/6 scenario(s) passed the gate (2 flaky; 120 runs, $0.0000; gate: fixed 20 
runs, pass on a pass rate of at least 0.75 (the baseline stand-in for the SPRT);
Wald intervals): failed
```

Each run gets a seed derived from the scenario and run index, and outcomes are
fed to the gate in run order, so the result is the same at any
`--concurrency` (asserted in `tests/test_flake.py`).

## Dashboard demo data and `docker compose up` (M5, updated in M6; offline)

Not a quality measurement. It records where the dashboard's demo numbers come
from and that the Compose stack serves them. No API calls: the live retail runs
replay the committed `claude-opus-5-5` recordings, judged from the committed
verdict cache.

- **Dataset:** the retail example: `examples/retail_agent/replay.toml` (fake-model
  cassettes) and `replay-live.toml` (six live Opus recordings), three seeded
  regressions, and the seeded flaky model at seeds 0 to 4.
- **Method:** `examples/demo_data.py` runs `replay ci`, `replay flake` and (since
  M6) `replay calibrate` in-process, exactly as on the command line, into one
  store. The two calibration reports are on the retail golden set against its
  objective labels: the real judge from the committed verdict cache (the section
  below), and the deterministic fake judge with three repeats and every probe,
  so each part of the calibration page has data. Neither is a calibration
  against human labels, and neither can gate: the page says why.
- **Date:** 2026-10-10; re-run 2026-10-14 for M6.
- **Commands:** `make demo-data` (SQLite), then `docker compose up --build --detach
  --wait app` (PostgreSQL 16).

Raw output (`make demo-data`, last line; about 26 s, a 39 MB SQLite file):

```
demo store ready: 480 runs, 14 suites, 6 flake reports, 2 calibration reports
```

Raw output (Compose; `curl` against the running stack):

```
$ curl -sf 'http://localhost:8000/api/runs?limit=1' | jq .total
480
$ curl -sf http://localhost:8000/api/health
{"status":"ok","version":"0.1.0","backend":"postgresql","revision":"0002"}
seed-1  | demo store ready: 480 runs, 14 suites, 6 flake reports, 2 calibration reports
# second `docker compose up`:
seed-1  | the store already holds 480 runs; left as it is
```

The Playwright smoke test (`npm run e2e` in `dashboard/`) passed against both the
SQLite demo store and the Compose stack (`2 passed`) in M5, and against the
Compose stack again in M6 (`2 passed`), when the screenshots were recaptured.

The M5 version of this store held a calibration report computed by hand
whose `policy_followed` row "gated" at 0.6 on raw agreement. Since M6 a report
gates only with human labels and kappa from the `core` statistics, so that
row could no longer gate; the reports above come from `replay calibrate`
itself.

## Judge calibration pipeline on the retail golden set (M6, offline)

What `replay calibrate` measures today without a key or human labels: the
real judge's committed verdicts against the suite's deterministic verdicts.
This is the M3 smoke test re-run through the calibration pipeline, not a
calibration. The reference is the suite's rules (`--reference objective`), so
the report says it may not gate, and the statistics are the `baseline`
implementation, so the "kappa" column holds raw agreement.

- **Dataset:** `eval/golden/retail` (`make golden`): the six live retail
  recordings replayed with the real agent and the three M2 mutants, 24 runs.
  Golden items are keyed by the judge's transcript, so the two
  `wrong_write_argument` runs whose transcripts equal the real agent's are one
  item each: 21 items, 6 objective passes and 15 fails.
- **Judge:** `claude-opus-5-5`, effort `low`, rubric version 1, from the
  verdicts committed in `examples/retail_agent/judge-cache/` (M3; no new
  calls, $0). One verdict per item, so no self-consistency or probes.
- **Date:** 2026-10-14. Raw JSON: `eval/results/calibration_objective.json`
  (byte-stable: `created_at` is the newest cached verdict used).

**Command:** `make eval-judge-objective`

```
estimate: 0 judge call(s) to make, 21 cached (of 21 verdicts); estimated cost $0.0000
criterion  kappa* [95% CI]    agreement  majority    pass/pass, false pass, false fail, fail/fail  self-consistency  n   gates
_overall   0.95 [0.86, 1.00]  95%        71% (fail)  6, 1, 0, 14                                   -                 21  -
The judge's first verdict per item against objective labels by checks (1 = pass, 0 = fail): raw agreement in the kappa column (REPLAY_STATS_IMPL=baseline; set
it to core for Cohen's kappa), with a normal-approximation bootstrap CI (2000 resamples, seed 0, clamped to [-1, 1]).
no criterion gates: the calibration report compares the judge with objective labels, not human ones; label the golden set with `replay label` and recalibrate
21 item(s), 0 judge call(s), 21 cached; cost $0.0000
```

Reading it: the judge agrees with the deterministic verdict on 20 of 21
distinct runs (95%), against 71% for a judge that always answers "fail".
The one disagreement is a false pass, the `first_tool_only` run on
`cancel_pending_laptop` discussed in the M3 section. Objective labels exist
only for the whole run (`_overall`): a deterministic check does not answer
one rubric question, so per-criterion agreement needs human labels.

## LLMBar heuristic baselines (M6, offline)

The floor a real judge's LLMBar numbers must be read against. Three
deterministic "judges" pick the first output, the longer one or the shorter
one, on every LLMBar instance in both orders. No model is called.

- **Dataset:** LLMBar (Zeng et al., ICLR 2024, MIT), vendored in
  `eval/data/llmbar` at `princeton-nlp/LLMBar@900616bff90b` with sha256 sums:
  Natural (100) and Adversarial Neighbor (134), GPTInst (92), GPTOut (47) and
  Manual (46), 419 instances.
- **Method:** each instance judged in its original order and swapped (838
  verdicts per judge). Accuracy per order and averaged; "consistent-correct"
  is right in both orders; position bias is the share of instances whose
  chosen output changes with the order. Intervals come from
  `REPLAY_STATS_IMPL=baseline`, so they are Wald intervals, which collapse to
  a point at 0% and 100%; the averaged accuracy's interval treats the two
  orders of one instance as independent, so it is too narrow.
- **Date:** 2026-10-14. Raw JSON: `eval/results/llmbar_heuristic_{first,longer,shorter}.json`.

**Command:** `make eval-llmbar-heuristics`, which runs, for each judge,
`uv run python eval/llmbar.py --provider fake-<judge> --cache-dir <scratch> --out eval/results/llmbar_heuristic_<judge>.json`.
The tables below are those runs' standard output.

**Always the first output** (`fake:first`):

```
LLMBar (princeton-nlp/LLMBar@900616bff90b), judge `fake:first`, pairwise prompt v1. Intervals: 95% Wald (`REPLAY_STATS_IMPL=baseline`); the averaged accuracy's interval treats the two orders of an instance as independent trials, so it is too narrow.

| Group | n | Acc. original | Acc. swapped | Acc. average | Consistent-correct | Position bias | Flips to first | Errors |
|---|---|---|---|---|---|---|---|---|
| Natural | 100 | 42.0% [32.3%, 51.7%] | 58.0% [48.3%, 67.7%] | 50.0% [43.1%, 56.9%] | 0.0% [0.0%, 0.0%] | 100.0% [100.0%, 100.0%] | 100/100 | 0 |
| Adversarial/Neighbor | 134 | 56.0% [47.6%, 64.4%] | 44.0% [35.6%, 52.4%] | 50.0% [44.0%, 56.0%] | 0.0% [0.0%, 0.0%] | 100.0% [100.0%, 100.0%] | 134/134 | 0 |
| Adversarial/GPTInst | 92 | 48.9% [38.7%, 59.1%] | 51.1% [40.9%, 61.3%] | 50.0% [42.8%, 57.2%] | 0.0% [0.0%, 0.0%] | 100.0% [100.0%, 100.0%] | 92/92 | 0 |
| Adversarial/GPTOut | 47 | 46.8% [32.5%, 61.1%] | 53.2% [38.9%, 67.5%] | 50.0% [39.9%, 60.1%] | 0.0% [0.0%, 0.0%] | 100.0% [100.0%, 100.0%] | 47/47 | 0 |
| Adversarial/Manual | 46 | 47.8% [33.4%, 62.3%] | 52.2% [37.7%, 66.6%] | 50.0% [39.8%, 60.2%] | 0.0% [0.0%, 0.0%] | 100.0% [100.0%, 100.0%] | 46/46 | 0 |
| Adversarial (pooled) | 319 | 51.4% [45.9%, 56.9%] | 48.6% [43.1%, 54.1%] | 50.0% [46.1%, 53.9%] | 0.0% [0.0%, 0.0%] | 100.0% [100.0%, 100.0%] | 319/319 | 0 |
| Overall | 419 | 49.2% [44.4%, 54.0%] | 50.8% [46.0%, 55.6%] | 50.0% [46.6%, 53.4%] | 0.0% [0.0%, 0.0%] | 100.0% [100.0%, 100.0%] | 419/419 | 0 |
```

**The longer output** (`fake:longer`):

```
LLMBar (princeton-nlp/LLMBar@900616bff90b), judge `fake:longer`, pairwise prompt v1. Intervals: 95% Wald (`REPLAY_STATS_IMPL=baseline`); the averaged accuracy's interval treats the two orders of an instance as independent trials, so it is too narrow.

| Group | n | Acc. original | Acc. swapped | Acc. average | Consistent-correct | Position bias | Flips to first | Errors |
|---|---|---|---|---|---|---|---|---|
| Natural | 100 | 57.0% [47.3%, 66.7%] | 56.0% [46.3%, 65.7%] | 56.5% [49.6%, 63.4%] | 56.0% [46.3%, 65.7%] | 1.0% [0.0%, 3.0%] | 1/1 | 0 |
| Adversarial/Neighbor | 134 | 11.9% [6.5%, 17.4%] | 11.9% [6.5%, 17.4%] | 11.9% [8.1%, 15.8%] | 11.9% [6.5%, 17.4%] | 0.0% [0.0%, 0.0%] | 0/0 | 0 |
| Adversarial/GPTInst | 92 | 13.0% [6.2%, 19.9%] | 13.0% [6.2%, 19.9%] | 13.0% [8.2%, 17.9%] | 13.0% [6.2%, 19.9%] | 0.0% [0.0%, 0.0%] | 0/0 | 0 |
| Adversarial/GPTOut | 47 | 44.7% [30.5%, 58.9%] | 44.7% [30.5%, 58.9%] | 44.7% [34.6%, 54.7%] | 44.7% [30.5%, 58.9%] | 0.0% [0.0%, 0.0%] | 0/0 | 0 |
| Adversarial/Manual | 46 | 19.6% [8.1%, 31.0%] | 17.4% [6.4%, 28.3%] | 18.5% [10.5%, 26.4%] | 17.4% [6.4%, 28.3%] | 2.2% [0.0%, 6.4%] | 1/1 | 0 |
| Adversarial (pooled) | 319 | 18.2% [13.9%, 22.4%] | 17.9% [13.7%, 22.1%] | 18.0% [15.0%, 21.0%] | 17.9% [13.7%, 22.1%] | 0.3% [0.0%, 0.9%] | 1/1 | 0 |
| Overall | 419 | 27.4% [23.2%, 31.7%] | 27.0% [22.7%, 31.2%] | 27.2% [24.2%, 30.2%] | 27.0% [22.7%, 31.2%] | 0.5% [0.0%, 1.1%] | 2/2 | 0 |
```

**The shorter output** (`fake:shorter`):

```
LLMBar (princeton-nlp/LLMBar@900616bff90b), judge `fake:shorter`, pairwise prompt v1. Intervals: 95% Wald (`REPLAY_STATS_IMPL=baseline`); the averaged accuracy's interval treats the two orders of an instance as independent trials, so it is too narrow.

| Group | n | Acc. original | Acc. swapped | Acc. average | Consistent-correct | Position bias | Flips to first | Errors |
|---|---|---|---|---|---|---|---|---|
| Natural | 100 | 44.0% [34.3%, 53.7%] | 43.0% [33.3%, 52.7%] | 43.5% [36.6%, 50.4%] | 43.0% [33.3%, 52.7%] | 1.0% [0.0%, 3.0%] | 1/1 | 0 |
| Adversarial/Neighbor | 134 | 88.1% [82.6%, 93.5%] | 88.1% [82.6%, 93.5%] | 88.1% [84.2%, 91.9%] | 88.1% [82.6%, 93.5%] | 0.0% [0.0%, 0.0%] | 0/0 | 0 |
| Adversarial/GPTInst | 92 | 87.0% [80.1%, 93.8%] | 87.0% [80.1%, 93.8%] | 87.0% [82.1%, 91.8%] | 87.0% [80.1%, 93.8%] | 0.0% [0.0%, 0.0%] | 0/0 | 0 |
| Adversarial/GPTOut | 47 | 55.3% [41.1%, 69.5%] | 55.3% [41.1%, 69.5%] | 55.3% [45.3%, 65.4%] | 55.3% [41.1%, 69.5%] | 0.0% [0.0%, 0.0%] | 0/0 | 0 |
| Adversarial/Manual | 46 | 82.6% [71.7%, 93.6%] | 80.4% [69.0%, 91.9%] | 81.5% [73.6%, 89.5%] | 80.4% [69.0%, 91.9%] | 2.2% [0.0%, 6.4%] | 1/1 | 0 |
| Adversarial (pooled) | 319 | 82.1% [77.9%, 86.3%] | 81.8% [77.6%, 86.1%] | 82.0% [79.0%, 85.0%] | 81.8% [77.6%, 86.1%] | 0.3% [0.0%, 0.9%] | 1/1 | 0 |
| Overall | 419 | 73.0% [68.8%, 77.3%] | 72.6% [68.3%, 76.8%] | 72.8% [69.8%, 75.8%] | 72.6% [68.3%, 76.8%] | 0.5% [0.0%, 1.1%] | 2/2 | 0 |
```

Reading it: on the Adversarial subsets, picking the *shorter* output is right
82.0% of the time and picking the longer one 18.0%. Those subsets were built
so that the wrong output looks better on the surface (longer, more fluent,
more "helpful"), so a judge that rewards length or polish scores well below
chance there, and an Adversarial accuracy near 82% says nothing until it is
compared with this floor. On Natural the same heuristics are near chance
(56.5% and 43.5%). The always-first judge scores exactly 50% averaged over
both orders with a position-bias rate of 100%, the reason every instance is
judged in both orders. A length tie flips with the order, which gives the
0.5% position bias of the length judges.

## Recorder overhead and replay speed (M7, offline)

What recording costs per intercepted call, and how long a `full` replay of a
live recording takes next to the time the live run spent in its calls.

- **Datasets:** recorder overhead on D5-style synthetic calls: the fake model
  (`agent_replay/fake.py`) answering the retail agent's real first-turn request
  (policy, tool list and task: 12,543 bytes of canonical JSON), the retail
  `get_order_details` tool, `agent_replay.now()` and `agent_replay.random()`.
  Replay speed on the six live `claude-opus-5-5` recordings of
  `examples/retail_agent/replay-live.toml` (recorded 2026-10-09).
- **Method, overhead:** per boundary, 2000 interleaved pairs of the same call
  bare (not intercepted) and recorded (inside a `RecordingSession`), the order
  within a pair alternating, after 200 discarded warm-up pairs, timed with
  `time.perf_counter_ns`, in sessions of 20 calls, GC enabled. Overhead is the
  median and p95 of the 2000 paired differences (recorded − bare); pairing
  keeps machine drift out of the difference. Each recorded session is then
  replayed in `full` mode and every served call is timed ("replay served").
  The no-call session row is the fixed cost of creating, entering, exiting
  and finishing one recording session.
- **Method, replay vs live:** the recorded live duration is the sum of the
  cassette's envelope `latency_ms`. The agent makes one call at a time, so this
  is the time spent inside model and tool calls; the cassette stores no
  end-of-run time, so the agent's own code between calls is left out and the
  sum is a lower bound on the live wall-clock. The replay is
  `agent_replay.ci.run_scenario`, what `replay ci` runs per scenario (load the
  cassette and the agent, replay in `full` mode), without checks and with the
  suite's checks (judge read from the committed cache,
  `REPLAY_JUDGE_OFFLINE=1`), median of 20 runs after one warm-up. Speedup is
  the recorded live duration over that median.
- **Machine:** AMD EPYC 7763 64-Core Processor, 4 cores, Linux x86_64, CPython
  3.12.3 (the environment block of the JSON).
- **Date:** 2026-10-16. Raw JSON: `eval/results/bench_replay.json` (not
  byte-stable: timings depend on the machine, so CI runs the benchmark but does
  not diff it; `make bench-replay` writes to `.replay/`, and
  `make bench-replay-record` rewrote this committed copy).
- **Command:** `make bench-replay-record` (= `uv run python eval/bench_replay.py --out eval/results/bench_replay.json`).

```
Recorder overhead per boundary (2000 paired calls, 200 warm-up, sessions of 20; microseconds)

| boundary | request bytes | bare median | bare p95 | recorded median | recorded p95 | overhead median | overhead p95 | replay served median | replay served p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| model | 12543 | 509.03 | 541.75 | 1189.33 | 1306.92 | 683.98 | 788.43 | 654.96 | 719.88 |
| tool | 24 | 20.07 | 24.30 | 125.23 | 151.32 | 106.23 | 131.64 | 74.99 | 99.15 |
| clock | 2 | 0.56 | 0.73 | 48.19 | 67.62 | 47.69 | 66.83 | 42.09 | 60.22 |
| random | 2 | 0.14 | 0.24 | 43.69 | 63.54 | 43.55 | 63.38 | 41.48 | 60.18 |

- one recording session with no calls (create, enter, exit, finish; 100 sessions): median 14.35, p95 17.17

Full replay vs recorded live duration (20 runs per scenario after one warm-up; milliseconds)

| scenario | envelopes | recorded live | replay median | replay p95 | speedup | replay+checks median | speedup with checks |
|---|---:|---:|---:|---:|---:|---:|---:|
| cancel_pending_laptop | 11 | 8592.2 | 33.21 | 33.79 | 259x | 39.52 | 217x |
| modify_address_suite | 6 | 6564.2 | 20.18 | 22.68 | 325x | 24.87 | 264x |
| return_damaged_bicycle | 12 | 8752.5 | 33.92 | 79.40 | 258x | 40.56 | 216x |
| exchange_waterproof_boots | 14 | 10830.2 | 41.21 | 45.60 | 263x | 47.09 | 230x |
| lookup_latest_order_total | 9 | 6546.3 | 26.82 | 31.13 | 244x | 32.16 | 204x |
| transfer_swapped_refunds | 10 | 9725.6 | 32.83 | 76.02 | 296x | 37.63 | 258x |

- suite: recorded live 51011.1 ms, full replay 188.17 ms (271x), with checks 221.83 ms (230x); per scenario 244x to 325x
- environment: Python 3.12.3 (CPython), Linux x86_64, 4 CPUs, AMD EPYC 7763 64-Core Processor

wrote eval/results/bench_replay.json
```

Repeatability: a second run of the same command on the same machine class
during review, timed with `/usr/bin/time` (summary lines only):

```
$ /usr/bin/time -f "%e s wall" uv run python eval/bench_replay.py --out .replay/bench_replay.json
| model | 12543 | 508.35 | 546.25 | 1175.81 | 1289.01 | 669.93 | 762.91 | 649.53 | 720.89 |
| tool | 24 | 19.62 | 24.12 | 124.33 | 149.40 | 105.77 | 130.31 | 74.68 | 97.98 |
| clock | 2 | 0.57 | 0.74 | 46.34 | 64.63 | 45.85 | 63.60 | 40.67 | 58.38 |
| random | 2 | 0.13 | 0.23 | 42.16 | 57.31 | 42.03 | 57.15 | 39.81 | 55.64 |
- one recording session with no calls (create, enter, exit, finish; 100 sessions): median 13.88, p95 17.45
- suite: recorded live 51011.1 ms, full replay 185.41 ms (275x), with checks 217.36 ms (235x); per scenario 248x to 328x
17.53 s wall
```

Every median is within 4% of the table above on an idle machine. On a shared
machine expect worse: these are wall-clock timings on a 4-core runner, so read
them as orders of magnitude, not as a precise ratio. The recorded live
durations are read from the cassettes and do not vary.

Writing the benchmark removed one fixed cost. Every recording session read
the installed package metadata to fill its cassette header, and that lookup
costs about a millisecond:

```
$ uv run python -c 'import timeit; from importlib.metadata import version; n = 200; t = timeit.repeat(lambda: version("agent-replay"), number=n, repeat=5); print("median %.1f us per call" % (sorted(t)[2] / n * 1e6))'
median 865.4 us per call
```

The version is now looked up once per process (`functools.cache` in
`agent_replay/session.py`); the no-call session row above is 14.35 µs, so the
lookup was about 98% of a session's fixed cost.

Reading it: recording adds about 50 µs to a clock read or random draw, about
0.1 ms to a tool call and about 0.7 ms to a model call with a 12.5 KB request,
against model calls that took 1.4 to 2.0 s each in the live recordings (the
first cassette's five calls), so under 0.1% of a call. A `full` replay of a
live recording, checks included, took 1/204 to 1/264 of the time the live run
spent in its calls.

## Release readiness (M7)

Whether what a release ships works outside this repository. Nothing was
published; these are the checks the release workflow runs before it would.

**Wheel and sdist in clean venvs.** `make dist` builds the dashboard into the
package, runs `uv build`, then `scripts/check_wheel.sh`: for the wheel (with
the `server` extra) and then the sdist, a fresh `python3.12 -m venv` in a
temporary directory outside the repo, `pip install`, and checks that
`agent_replay` imports from that venv (not the checkout), the migrations and
the dashboard are in the installed package, a new store migrates to head,
`replay show` and `replay run` work on the toy cassette (only the toy agent's
three modules are copied next to it), and `replay serve` answers
`/api/health` and serves the bundled `index.html` at `/`. Then `twine check
--strict`. Date 2026-10-16; 49 s. Raw output from `scripts/check_wheel.sh`
on (absolute paths shortened):

```
scripts/check_wheel.sh --expect-dashboard dist
wheel: dist/agent_replay-1.0.0-py3-none-any.whl (336 KiB)
sdist: dist/agent_replay-1.0.0.tar.gz (320 KiB)
version: 1.0.0

== wheel: fresh venv, pip install 'agent_replay-1.0.0-py3-none-any.whl[server]'

== replay --version
agent-replay 1.0.0

== import agent_replay from the venv
agent_replay 1.0.0 from $TMP/venv/lib/python3.12/site-packages/agent_replay
migrations: 0001_initial.py, 0002_flake_and_calibration.py; dashboard bundled: True

== the run store migrates to head
run store at schema revision 0002

== replay show and replay run on the toy cassette
replay show: 14 lines
Replaying refund_eligible.jsonl (mode full, policy strict) against 
examples.toy_agent.agent:run
#  kind   name                       status  request                           
0  model  anthropic.messages.create  exact   "1 message(s)"                    
1  tool   lookup_order               exact   {"order_id":"A1001"}              
2  tool   get_refund_policy          exact   {}                                
3  model  anthropic.messages.create  exact   "3 message(s)"                    
4  tool   issue_refund               exact   {"amount":42.5,"order_id":"A1001"}
5  model  anthropic.messages.create  exact   "5 message(s)"                    
PASS tool_sequence: 3 tool call(s) match the expected sequence (exact)
passed

== replay serve
/api/health: {"status":"ok","version":"1.0.0","backend":"sqlite","revision":"0002"}
/: the dashboard's index.html
serving the sqlite run store on http://127.0.0.1:39855 (dashboard: 

== sdist: fresh venv, pip install agent_replay-1.0.0.tar.gz
agent-replay 1.0.0

check_wheel: OK (1.0.0)
uvx twine check --strict dist/*
Checking dist/agent_replay-1.0.0-py3-none-any.whl: PASSED
Checking dist/agent_replay-1.0.0.tar.gz: PASSED

real	0m49.132s
user	0m50.139s
sys	0m8.032s
```

Sizes and install times, raw (2026-10-16; `ls -l` of `make dist`'s output, the
vite lines from the same run, the M6 sdist rebuilt from `main` with `uv build
--sdist` in a worktree, and `/usr/bin/time` around a `pip install` of the
source tree into two fresh `python3.12 -m venv`s, the second with pip's cache
warm, which is what the Action's `install: action` does):

```
$ ls -l dist/
-rw-r--r-- 1 runner runner 342782 Oct 16 07:27 agent_replay-1.0.0-py3-none-any.whl
-rw-r--r-- 1 runner runner 327427 Oct 16 07:27 agent_replay-1.0.0.tar.gz
../agent_replay/server/static/index.html                   0.48 kB │ gzip:   0.29 kB
../agent_replay/server/static/assets/index-DU9qURRW.css   21.12 kB │ gzip:   5.20 kB
../agent_replay/server/static/assets/index-CM2IjIM_.js   416.24 kB │ gzip: 127.10 kB
$ ls -l /tmp/maindist/            # main (M6), before the sdist was trimmed
-rw-r--r-- 1 runner runner 1992496 Oct 16 07:45 agent_replay-0.1.0.tar.gz
$ /usr/bin/time -f "%e s" python -m pip install -q <repo>
7.75 s
9.87 s (second, pip cache warm)
```

The M6 sdist was the whole repository, untracked local files included; the
1.0.0 sdist holds `agent_replay/`, `README.md`, `CHANGELOG.md`, `LICENSE`,
`pyproject.toml`, `PKG-INFO` and `.gitignore`. The M6 wheel had no dashboard.

**The Action, run as the CI job does.** The `action` job in
`.github/workflows/ci.yml` is this repository as the Action's consumer:
`uses: ./` on the retail suite with `install: action` (a real `pip install` of
the checkout) and `dry-run: true`, then again with the `first_tool_only`
mutant, then a step asserting the outputs. Run locally with act 0.2.89 on the
`catthehacker/ubuntu:act-latest` image and a pull-request event:

```
act pull_request -e event.json -j action -W .github/workflows/ci.yml \
  -P ubuntu-latest=catthehacker/ubuntu:act-latest --artifact-server-path /tmp/act-artifacts --pull=false
```

Lines from its output after the review fixes (2026-10-16; job-name prefixes and
artifact ids stripped; act repeats each composite action's outputs once more
after its last step). The passing step uses the default artifact name:

```
| Successfully installed agent-replay-1.0.0
| 6/6 scenario(s) passed: passed
⚙  ::set-output:: exit-code=0
⚙  ::set-output:: passed=true
⚙  ::set-output:: scenarios=6
⚙  ::set-output:: failed=0
❓  ::group::PR comment for #7: dry run, not posted
| Artifact replay-report-action-examples-retail_agent-replay.toml has been successfully uploaded! Final size is 3023 bytes.
⚙  ::set-output:: exit-code=0
⚙  ::set-output:: passed=true
⚙  ::set-output:: scenarios=6
⚙  ::set-output:: failed=0
| 1/6 scenario(s) passed: failed
⚙  ::set-output:: exit-code=1
⚙  ::set-output:: passed=false
⚙  ::set-output:: scenarios=6
⚙  ::set-output:: failed=5
❓  ::group::PR comment for #7: dry run, not posted
| Artifact replay-report-mutant has been successfully uploaded! Final size is 4183 bytes.
⚙  ::set-output:: exit-code=1
⚙  ::set-output:: passed=false
⚙  ::set-output:: scenarios=6
⚙  ::set-output:: failed=5
| pass: exit 0, passed true
| mutant: exit 1, passed false, failed 5
🏁  Job succeeded
```

Posting a real comment through the GitHub API was not exercised here (no
token for a real pull request in this environment); `tests/test_action.py`
covers create, update (including a marker comment on the second page), a 403
from a fork's read-only token (a warning, not a failure) and non-PR events
against a fake HTTP layer. The first real comment will be on this
repository's own pull requests once a workflow runs without `dry-run`.

**Containers.** `docker compose up --build --detach --wait app` seeded
PostgreSQL ("480 runs, 14 suites, 6 flake reports, 2 calibration reports"),
`/api/runs?limit=1` reported a total of 480 and `/api/health` the
`postgresql` backend at revision `0002`. The `demo` stage
(`docker build --target demo`) served the same 480 runs from its baked
SQLite store on `PORT=8123`: `/api/health` returned
`{"status":"ok","version":"1.0.0","backend":"sqlite","revision":"0002"}` and
the container's health check reported `healthy`. Neither hosted route in
RELEASE.md was deployed (no account).

```
$ docker images agent-replay-demo
REPOSITORY          TAG       IMAGE ID       CREATED         SIZE
agent-replay-demo   latest    f107e5a97e0d   6 minutes ago   285MB
```

## The Action on a real failing PR (M8; rehearsed, the real PR is to measure)

Question 3: does the GitHub Action stop a bad pull request in practice? The
harness is built and rehearsed twice; the real PR needs the maintainer's own
GitHub login, which the environment this milestone was built in does not have.

**Method.** `.github/workflows/replay.yml` runs the Action (`uses: ./`,
`install: action`, dry-run off, `pull-requests: write`) on every pull request
that changes `examples/retail_agent/**`. `make regression-pr`
(`eval/regression_pr.py`) pushes a base branch `regression-demo/base-<stamp>`
at `origin/main` (ci.yml ignores PRs against `regression-demo/**`, so only the
replay check spends minutes), commits on `regression-demo/<stamp>` the one-line
change that points the suite's default agent at
`examples.retail_agent.mutants:wrong_write_argument` (every write action gets
one wrong argument: a swapped cancel reason, an emptied `address2`, a gift-card
refund), opens the PR with `gh`, waits for the run, reads the job's wall time
from `gh run view`, the failed scenarios from the run's report artifact and the
sticky comment from the issue-comments API (only `github-actions[bot]`'s
comment with the suite's marker counts), renders the comment's GitHub HTML
into `docs/screenshots/pr-comment.png`, then closes the PR and deletes both
branches. It writes `eval/results/regression_pr/{evidence,report}.json` and
`comment.md` only once everything is captured and checked, and exits 2,
leaving the PR open with the commands that clean it up, if the check did not
conclude `failure` or the report shows a suite that could not run (an exit-2
broken suite also fails the check, but it is not a caught regression). It
prints the lines below.

**Rehearsal 1: the same edit and `replay ci` command, no GitHub.**
`make regression-pr-rehearse` (a detached worktree at `HEAD`, the action's
`<python> -m agent_replay ci <suite> --json --markdown --no-store`, then
`action/replay_action.py comment --post false` for the body):

```
## ❌ Replay: 2/6 scenarios passed
| `cancel_pending_laptop` | full | ❌ fail | 6/7 | 1 approximate, 10 exact | $0.0000 |
| `modify_address_suite` | full | ❌ fail | 6/7 | 1 approximate, 7 exact | $0.0000 |
| `return_damaged_bicycle` | full | ❌ fail | 6/7 | 1 approximate, 11 exact | $0.0000 |
| `exchange_waterproof_boots` | full | ❌ fail | 6/7 | 1 approximate, 13 exact | $0.0000 |
| `lookup_latest_order_total` | full | ✅ pass | 8/8 | 7 exact | $0.0000 |
| `transfer_swapped_refunds` | full | ✅ pass | 8/8 | 10 exact | $0.0000 |
"replay_ci_wall_seconds": 2.39, "passed": false,
"failed": cancel_pending_laptop, modify_address_suite, return_damaged_bicycle,
          exchange_waterproof_boots, each on ["rule:matches_ground_truth"]
```

**Rehearsal 2: the real workflow job under act.** `act pull_request -j replay
-W .github/workflows/replay.yml -P ubuntu-latest=catthehacker/ubuntu:act-latest
--artifact-server-path /tmp/act-artifacts` (act 0.2.89) in a worktree with the
seeded suite:

```
✅  Success - Main Install agent-replay [14.124163087s]
✅  Success - Main Run replay ci [4.106743334s]          exit-code=1
🚧  ::warning::agent-replay did not comment: no github-token or GITHUB_REPOSITORY
✅  Success - Main Upload the report [540.503924ms]      Final size is 4107 bytes
❗  ::error::agent-replay: 4 of 6 scenario(s) in examples/retail_agent/replay.toml regressed
❌  Failure - Main Verdict [62.095148ms]
❌  Failure - Main ./ [32.88214098s]
🏁  Job failed
```

The job fails with exit 1 and the reason as an annotation; under act there is
no token, so the comment step warns instead of posting, which is the one path
the real PR has to show.

**What the seeded regression says about the checks.** All four failures come
from one check, τ-bench's ground-truth rule. `tool_sequence`, `arg_schema` and
`required_tools` pass, because the mutant calls the same tools in the same
order with well-typed arguments, and in `full` mode the changed write call is
served its recorded result as one `approximate` step. A suite with only
structural checks would let this PR through unless it set
`fail_on_approximate`, the same gap the mutation study measures for tool bugs.

**Observed on GitHub: the passing half.** This milestone's own pull request
([#11](https://github.com/TheAsianFish/agent-replay/pull/11)) changes
`.github/workflows/replay.yml`, so the workflow ran on it for real, with dry-run
off and the unmodified suite. It did the one thing act could not do: it posted
the sticky comment as `github-actions[bot]`
([run 39578764254](https://github.com/TheAsianFish/agent-replay/actions/runs/39578764254)):

```
<!-- agent-replay: examples/retail_agent/replay.toml -->
## ✅ Replay: 6/6 scenarios passed

Suite `examples/retail_agent/replay.toml` passed at `6bba25535a03` in 0.23s.
| `cancel_pending_laptop` | full | ✅ pass | 7/7 | 11 exact | $0.0000 |
| `modify_address_suite` | full | ✅ pass | 7/7 | 8 exact | $0.0000 |
| `return_damaged_bicycle` | full | ✅ pass | 7/7 | 12 exact | $0.0000 |
| `exchange_waterproof_boots` | full | ✅ pass | 7/7 | 14 exact | $0.0000 |
| `lookup_latest_order_total` | full | ✅ pass | 8/8 | 7 exact | $0.0000 |
| `transfer_swapped_refunds` | full | ✅ pass | 8/8 | 10 exact | $0.0000 |
```

`6bba25535a03` is the PR's merge commit (a `pull_request` run checks out
`refs/pull/11/merge`), not the branch head. The same command at the branch head
(`replay ci examples/retail_agent/replay.toml --json r.json --markdown r.md
--no-store`, at `b39cdadbdb45`) gives the same table row for row, with the same
0.23 s. That is the control for the failing PR: the unmodified suite passes on
GitHub with 0 approximate steps, so a failure on the seeded branch is the
mutant's doing. The 0.23 s is `replay ci`'s own time; the job's wall time was
not captured (no `gh` here), and `make regression-pr` records it for the failing
run.

**This environment:** `make regression-pr` stops at its preflight, before any
push:

```
regression-pr: gh is not installed or not logged in; install the GitHub CLI and run `gh auth login` as a maintainer of the repository (`gh auth status` exited 1: You are not logged into any GitHub hosts. To log in, run: gh auth login)
make: *** [Makefile:160: regression-pr] Error 2
```

**To measure** (free; one command after this change is on `main`, about two
minutes): `gh auth login`, then `make regression-pr`. Paste the printed lines
here (the PR, the failing check run, the comment permalink, the job's wall
time, the scenarios that failed), and commit `eval/results/regression_pr/` and
`docs/screenshots/pr-comment.png`.

## Dashboard walkthrough (M8, offline)

`make walkthrough` (`scripts/walkthrough.sh`) builds the dashboard, seeds
`.replay/demo.db` with `make demo-data`, serves it, and runs
`dashboard/e2e/walkthrough.spec.ts` in Chromium at 1280×800 with Playwright's
video on: the runs list, a failing `full` run's checks and trace (an
`approximate` step expanded), the aligned diff against a passing run of the
same scenario picked from the run page's "Compare with", and the flakiness
page charting a scenario that is not always passing. ffmpeg writes an H.264
mp4 (CRF 28) and a 960-pixel, 10 fps GIF. On the build machine, with an
existing demo store (`SKIP_DEMO_DATA=1`): 53.8 s wall, a 34.4 s video,
`walkthrough.mp4` 855,493 bytes and `walkthrough.gif` 3,473,123 bytes. Sizes
change from run to run (timestamps and run ids differ), so the files are not
byte-stable and CI does not re-record them.

## To measure

- **The Action on a real failing PR** (question 3): `gh auth login` as a
  maintainer, then `make regression-pr` (free, about two minutes); see the M8
  section for what it records.
- **SPRT savings vs the fixed-sample test.** `make bench-sprt` (offline, about
  1 s) once `agent_replay/stats.py` is implemented: the core rows and
  "runs vs the fixed-sample test" lines join the table above with no other
  change.
- **Wilson intervals for the live flake runs.** Re-run `replay flake` with
  `REPLAY_STATS_IMPL=core`, or recompute from the pass counts in
  `eval/results/m4_flake_*.json`.
- **D3, the full flake benchmark** (30 scenarios × 20 runs in `model-live`)
  needs a larger golden set (M6). At the measured cost per run that is about
  $21 with Opus 5.5 or $0.52 with Haiku 5.5.
- **Judge-human agreement (kappa with bootstrap CI, self-consistency, bias
  probes)** needs human labels and a key. Label the retail golden set
  (`uv run replay label eval/golden/retail --rubric examples/retail_agent/rubric.yaml --labeller NAME`,
  21 items × 3 criteria, about 20 minutes), then run
  `REPLAY_STATS_IMPL=core make eval-judge`: 84 calls to `claude-opus-5-5`
  (3 verdicts per item plus the position and verbosity probes; the first
  verdicts are already cached), estimated at $1.30 by `replay calibrate
  --dry-run` from the measured tokens of the committed verdicts (`make
  token-rates` re-derives them: 2.3 characters per input token and 289
  output tokens a call, medians over the 21 verdicts; the estimator gives
  $0.3274 for calls that really cost $0.3302) (about
  $0.015 a call: $0.0153 for a main verdict, a little more for a position-probe
  call, whose reversed rubric is a fresh prompt-cache write). A gating report
  also needs kappa from `stats.py`
  and at least 20 labelled items per criterion (`REPLAY_JUDGE_MIN_ITEMS`);
  21 items barely clear that floor and give a wide interval, so the gating
  report is meant to come from D1.
- **D1, the larger golden set.** `make record-golden` records 10
  `model-live` runs per retail task with Opus 5.5 and 10 with Haiku 5.5 (120
  runs), estimated at $2.19 from the M4 cost per run; then label it with
  `replay label eval/golden/retail-d1 ...` and calibrate it with the
  `eval-judge` command pointed at `eval/golden/retail-d1`, at about $0.077 a
  distinct item (5 verdicts at about $0.015).
- **LLMBar accuracy and position bias for the real judge.** `make
  eval-llmbar`: 838 calls to `claude-opus-5-5` at effort `low`, estimated at
  $12.73 by `make eval-llmbar-dry` (characters / 4 for input, an assumed 600
  output tokens a call; output is about 80% of the estimate). `--limit N
  --seed S` runs a stratified sample for less. Compare with the heuristic
  baselines above. Commit `eval/llmbar-cache/` with the result, as the retail
  judge cache is, so `uv run python eval/llmbar.py --offline --out
  eval/results/llmbar.json` reproduces it without a key; under the default
  `baseline` statistics the intervals are Wald, so prefer
  `REPLAY_STATS_IMPL=core make eval-llmbar` once `stats.py` exists.
- The M2 live smoke test above can be re-run with the two commands in its
  section (about $0.30 each with `claude-opus-5-5` and a cold prompt cache);
  the M3 cost-gate run with the two commands in its section (about $1.00);
  the M4 flake runs with the three commands in their section ($0.10, $1.07 and $0.04).

API spend in M3: $1.408 (cost gate $0.974; judge on recordings and mutants
$0.358; three runs of the judge's live test $0.076, two of them while the
rubric was being written, one of which failed `accurate_report` on a correct
but terse reply and led to the criterion's current wording). The committed
artefacts account for $1.329 of that (cost-gate reports $0.974, the 21 cached
verdicts $0.330, the live test above $0.025); the remaining $0.079 (two
re-judged duplicate mutant transcripts, two earlier live-test runs) comes from
the session log, not from a file in this repo.

API spend in M4: $1.213 (Opus flake runs $1.06960, Haiku flake runs
$0.10453 and the Haiku repeat $0.03786, all in their committed JSON reports;
one Haiku smoke run $0.00137, from the session log; sum $1.21336). Project
total: $3.326 (M1 $0.014, M2 $0.691, M3 $1.408, M4 $1.213).

API spend in M5: $0 (the demo data comes from offline replays and the
committed verdict cache).

API spend in M6: $0 (every judge verdict used came from the committed cache;
the new calls are under "To measure"). Project total: $3.326.

API spend in M7: $0 (packaging, the Action and the benchmark are offline).
Project total: $3.326.

API spend in M8: $0 (the rehearsals and the walkthrough are offline; the real
PR costs Actions minutes, not API calls). Project total: $3.326.
