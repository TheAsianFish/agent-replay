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
| Tool-result drift check | `tools-live+drift` 8/8 tool bugs, false alarms 0/5 (beside 8/8 and 1/5 with `fail_on_approximate`); 0 drift in 34 live tool results on the six live recordings | Tool-result drift check (M10) | `make eval-mutations`; `replay ci examples/retail_agent/replay-live.toml --mode tools-live --fail-on-tool-drift` | measured, offline |
| Identifying arguments | 2 identity mutations caught 0/2 without keys in every non-strict configuration, 2/2 with keys in `full`, `model-live`, `model-live+fallback` and `tools-live+drift`; 0 v1 cells changed by the keys; false alarms 0/5 (1/5 strict) | Identifying arguments (M11) | `make eval-mutations` | measured, offline |
| Correct-but-different paths | 5 variants in `model-live`: `tool_sequence` false alarms 4/5, ground-truth rules 0/5 under `fallback`; `strict` ends 2/5 on an unrecorded call (1/5 then fails the ground truth too) | Identifying arguments (M11) | `make eval-mutations` | measured, offline |
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
| The Action on a real PR, passing half | PR #11 (it changes `replay.yml`): 6/6 passed, sticky comment posted by `github-actions[bot]`, identical to a local run; PR #12: 6/6 again, same rows | The Action on a real failing PR (M8) | `replay.yml` on PRs #11 and #12 | measured, observed on GitHub |
| The Action on a real failing PR | PR #13 (seeded `wrong_write_argument`): check failed, 4/6 scenarios on `rule:matches_ground_truth`, sticky comment posted; job 26 s | The Action on a real failing PR (M8) | `make regression-pr` | measured, observed on GitHub |
| A pydantic-ai agent records and replays with no framework change | both suites (fake model; qwen2.5:7b via Ollama for 2 of 3 scenarios, the cancellation on the fake-model cassette) pass in `full` with 0 approximate steps, also against `run_stream`; 13 first-contact breaks listed (7 in Replay, all fixed) | pydantic-ai through Ollama (M9) | `make eval-pydantic-ai-offline` (in `make eval`) | measured, offline |
| Live flake on a local model | qwen2.5:7b, `model-live`, 10 runs each: 10/10, 2/10, 0/10; 1 approximate step in 30 runs; 1026 s on 4 vCPU (AMD EPYC 7763); $0 | pydantic-ai through Ollama (M9) | `make eval-pydantic-ai` (Ollama) | measured, local model |
| Frozen world vs the model (`model-live`, real model) | qwen2.5:7b, 20 runs per arm: `strict` 18/20 and 9/20, `fallback` 19/20 and 8/20; world-caused endings 0 and 3 of 20 under `strict` (an invented `get_booking` the recording never made), 0 by identity, 0 approximate steps in 80 runs; `cheapest_fare_in_euros` pooled with M9 47/50; 3090 s on 4 vCPU (AMD EPYC 7763); $0 | Frozen-world bias (M12) | `make eval-pydantic-ai-arms` (Ollama) | measured, local model |
| Model and prompt swaps in `model-live` (real model) | qwen2.5:7b cassettes, 20 runs each: qwen2.5:3b 0 / 20 and 0 / 20 under both `strict` and `fallback` (32 of 40 `strict` runs end by identity, all on the model's wrong argument); 7b without the prompt's guard line 2 / 20 and 2 / 20 (23 identity endings, guessed amounts from parallel tool calls) against the same-day control's 18 / 20 and 10 / 20; `temperature=0` 10 / 10 identical passing runs of a booking that passes 17 / 40 sampled; 0 approximate steps in 170 runs; 4778 s on 4 vCPU (AMD EPYC 7763); $0 | `model-live` survives the change (M15) | `make eval-pydantic-ai-swap` (Ollama) | measured, local model |
| Frozen-world divergence on a long chain (`model-live`, real model) | qwen2.5:7b, an 8-link rebooking chain (9 served tool calls on the shortest path), 20 runs per arm: 0 / 20 under `strict` and `fallback` alike; 320 of 320 chain lookups served exact; 1 world step in 321 served tool calls (0.31%, against 3 / 143 = 2.1% on M12's short runs); `strict` world endings 1 / 20 (Clopper-Pearson 0.1% to 24.9%) beside 3.47 predicted by independence; clustering not measurable (0 world steps under `fallback`); identity keys fired once in 40 runs; 39 of 40 runs answered a euro total without calling `convert_currency`; no qwen recording passed in 5 attempts; 1703 s on 4 vCPU (AMD EPYC 9V45); $0 | Divergence on a long chain (M16) | `make eval-pydantic-ai-long` (Ollama) | measured, local model |
| `synth`, the third unrecorded-call policy | offline, seventh configuration `model-live+synth`: 12/12 model mutations (as `fallback`), identity mutations 2/2 with keys by `run_error:KeyError`, refactors 0/5, and the correct `repeats_order_lookup` path fails the ground truth in 4 of 6 scenarios (as `strict`; `fallback` 0); live, qwen2.5:7b without the guard line, 20 runs per arm: `fallback` 2/20 and 3/20, `synth` 0/20 and 2/20 beside M15's `strict` 2/20 and 2/20, so M15's attribution stands; 11 of 20 `synth` fare answers state the stub's 0.00 EUR; 0 passes with a stubbed step; 3803 s on 4 vCPU (AMD EPYC 7763); $0 | `synth`, measured (M17) | `make eval-mutations`; `make eval-pydantic-ai-no-guard` (Ollama) | measured, offline and local model |
| A task where a tool legitimately fails | offline: `drops_error_flag`, which survived every configuration on the six tasks, is caught in 4 of 7 on a hand-written wrong-zip task (`model-live` under `strict`, `fallback` and `synth`; `tools-live+strict`) and survives `full`, `tools-live` and `tools-live+drift`, where the model's reply is served; unmutated passes all 7; live on qwen2.5:7b (`model-live`, `fallback`, 10 runs each): not found with no write 8/10 unmutated, 9/10 with `drops_error_flag` (9 and 10 read by hand), because Ollama's `/v1/messages` drops `is_error` before the prompt (probe 84 / 84 tokens with and without it, control 87); recorded on the first of 3 attempts; largest prompt 2,991 of 4,096 tokens; 485 s of arms on 4 vCPU (AMD EPYC 9V74); $0 | A task where a tool legitimately fails (M18) | `make eval-mutations`; `make eval-retail-error-qwen` (Ollama) | measured, offline and local model |
| The agent mutations on a real model | `lookup_latest_order_total`, qwen2.5:7b, `model-live`, `fallback`: the unmutated agent passed 2 / 10 (8 runs skipped the profile and invented an order id), so under the rule fixed before the run the eight mutations are **not resolvable on this model for this task**; beside the scripted model's on-task verdict (5 of 8 caught, `truncates_tool_results` by a JSONDecodeError crash): every qwen failure was the ground-truth rule, none a crash; `first_tool_only` and `mismatches_tool_use_ids` never acted (at most 1 `tool_use` per reply in 50 runs; a mismatched id never reaches the prompt, 88 / 88 tokens, control 92); `strips_policy` passed 5 / 5 against 2 / 10 (Fisher p = 0.007); largest prompt 3,708 tokens, 0 of 50 runs over 4,096, 0 truncated; no qwen recording passed in 3 attempts; 2,861 s on 4 vCPU (AMD EPYC 7763); $0 | The agent mutations on a real model (M19) | `make eval-retail-mutations-qwen` (Ollama) | measured, local model |
| Verdict-stability curve | 5 runs agree with the all-runs fixed-threshold verdict in ≥ 95% of draws on all 15 committed scenarios, but on only 1 of the 4 new M12 rows (90.3% to 94.3%; 0.15 to 0.35 from the 0.75 threshold); 13 of 19 rows pass 0% or 100% | Frozen-world bias (M12) | `make eval-verdict-curve` | measured, offline |
| Cassette bytes at scale | quadratic (c2 291 bytes/turn², fitted exponent 1.54 over 5 to 200 turns, 1.85 from 100 to 200); 3.65 MB at 100 turns; gzip 19.5× at 20 turns → 3.8× at 100 (envelopes outgrow its 32 KB window), xz 105×; prefix encoding under xz only 1.07× more, so `.jsonl.xz` and no schema change; `full` replay 1.7 s at 200 turns | Cassette bytes at scale (M13) | `make bench-scale` | measured, offline (sizes byte-stable) |
| Real recordings vs the fit | live Opus retail requests grow 1.4 to 2.6 KB a turn (synthetic 0.58), projecting to 8.0 to 13.9 MB of requests at 100 turns; xz 8.9× to 13.6× at 3 to 6 turns | Cassette bytes at scale (M13) | `make bench-scale` | measured, offline |
| Run store at scale | 282 KB a stored 20-turn run, 3.8 MB a 100-turn run (91% to 97% step JSON); `get_run` 3.7 MB | Cassette bytes at scale (M13) | `make bench-scale` | measured, offline (machine-dependent) |
| pydantic-ai cassettes across releases | 2.58.0 to 2.62.0: 6/6 pass, 0 approximate on each | Cassette bytes at scale (M13) | `make bench-upstream` (network) | measured, offline replay |
| Judge-human kappa, self-consistency, probes | — | To measure | `REPLAY_STATS_IMPL=core make eval-judge` (~$1.30) | **to measure** |
| D1 golden set | — | To measure | `make record-golden` (~$2.19) | **to measure** |
| Real judge on LLMBar | — | To measure | `make eval-llmbar` (~$12.73) | **to measure** |
| D3 flake benchmark | — | To measure | `make bench-flake` (not built; ~$21 Opus / ~$0.52 Haiku) | **to measure** |

API spend to date: $3.326 (M1 $0.014, M2 $0.691, M3 $1.408, M4 $1.213, M5 to
M19 $0; the live runs of M9, M12, M15, M16, M17, M18 and M19 used qwen2.5:7b and qwen2.5:3b
on a local CPU).

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

## The Action on a real failing PR (M8)

Question 3: does the GitHub Action stop a bad pull request in practice? The
harness was rehearsed twice offline, then run for real on PR #13.

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

| Scenario | Mode | Result | Checks | Steps | Cost |
|---|---|---|---|---|---|
| `cancel_pending_laptop` | full | ✅ pass | 7/7 | 11 exact | $0.0000 |
| `modify_address_suite` | full | ✅ pass | 7/7 | 8 exact | $0.0000 |
| `return_damaged_bicycle` | full | ✅ pass | 7/7 | 12 exact | $0.0000 |
| `exchange_waterproof_boots` | full | ✅ pass | 7/7 | 14 exact | $0.0000 |
| `lookup_latest_order_total` | full | ✅ pass | 8/8 | 7 exact | $0.0000 |
| `transfer_swapped_refunds` | full | ✅ pass | 8/8 | 10 exact | $0.0000 |

---
[Run log and report](https://github.com/TheAsianFish/agent-replay/actions/runs/39578764254)
```

`6bba25535a03` is the PR's merge commit (a `pull_request` run checks out
`refs/pull/11/merge`), not the branch head. The same command at the branch head
(`replay ci examples/retail_agent/replay.toml --json r.json --markdown r.md
--no-store`, at `b39cdadbdb45`) gives the same table row for row, with the same
0.23 s. That is the control for the failing PR: the unmodified suite passes on
GitHub with 0 approximate steps, so a failure on the seeded branch is the
mutant's doing. The 0.23 s is `replay ci`'s own time; the job's wall time was
not captured for those runs; for the failing PR below it is 26 s.

The branch's next pull request
([#12](https://github.com/TheAsianFish/agent-replay/pull/12)) ran the same
workflow again and got the same result: 6/6 passed, with an identical table
([run 39831038543](https://github.com/TheAsianFish/agent-replay/actions/runs/39831038543),
merge commit `b18a3cd1b446`, 0.27 s). A local run at the branch head
(`1bd4dd8d85d5`, the same command) gives the same six rows in 0.14 s. The rows
are deterministic across both runners. Only `replay ci`'s own time varies with
the machine.

**Observed on GitHub: the failing PR.** `make regression-pr` (2026-10-26,
mutant `wrong_write_argument`, base copied from `origin/main`) printed:

```
- Pull request: https://github.com/TheAsianFish/agent-replay/pull/13 (seeded mutant `wrong_write_argument`, head `90b5bc1be108`, closed after capture)
- Failing check run: https://github.com/TheAsianFish/agent-replay/actions/runs/40380755513/job/121775995565 (conclusion `failure`)
- Sticky comment: https://github.com/TheAsianFish/agent-replay/pull/13#issuecomment-6424518479
- Check wall time: 26 s for the job (28 s for the whole run, queueing included)
- Scenarios failed: 4 of 6
  - `cancel_pending_laptop`: rule:matches_ground_truth
  - `modify_address_suite`: rule:matches_ground_truth
  - `return_damaged_bicycle`: rule:matches_ground_truth
  - `exchange_waterproof_boots`: rule:matches_ground_truth
- Passed: `lookup_latest_order_total`, `transfer_swapped_refunds`
```

The real check failed on exactly the four scenarios the two rehearsals
predicted, for the same reason, and `replay ci` itself took 0.26 s of the
job's 26 s (the rest is checkout and install). `github-actions[bot]` posted
the failing comment ([permalink](https://github.com/TheAsianFish/agent-replay/pull/13#issuecomment-6424518479)),
rendered in [`docs/screenshots/pr-comment.png`](docs/screenshots/pr-comment.png);
each failed scenario names the expected and actual write call, e.g. a cancel
with reason `ordered by mistake` instead of `no longer needed`. The raw
evidence, the report artifact and the comment body are in
`eval/results/regression_pr/`. The script then closed the PR and deleted both
`regression-demo/*` branches. Cost: $0 (no model calls; the job is about half
a minute of Actions time).

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

## pydantic-ai through Ollama: first contact and a `model-live` flake on qwen2.5:7b (M9)

Does Replay work on a framework it did not write, with a live model that costs
nothing? The agent is a tool-using [pydantic-ai](https://github.com/pydantic/pydantic-ai)
travel desk (`examples/pydantic_ai_agent`: search flights, convert currency,
book, cancel; three tasks with one right answer each). Its Anthropic client is
injected through `AnthropicProvider(anthropic_client=replay.wrap(AsyncAnthropic(base_url=...)))`,
and the `base_url` is a local Ollama server, which speaks the Anthropic
Messages API. No framework code was changed.

- **Model:** `qwen2.5:7b` (digest `845dbda0ea48…`), Ollama 0.40.11, default
  sampling, CPU only: AMD EPYC 7763, 4 logical CPUs (a GitHub-hosted-size
  runner), one request at a time. pydantic-ai-slim 2.61.0, anthropic 1.12.1.
- **Checks** (`replay-qwen.toml`, the same as `replay.toml`): `in_order` tool
  sequences, the writes (`book_flight` / `cancel_booking` with the right
  arguments), the figures in the answer (`B6416` and `173.88`; `NEW001`;
  `243`), a lookup before any write, `forbidden_tools`, `max_tool_calls` 8 and
  at most 6 model calls. They judge outcomes and the one process rule, not
  agreement with one recorded path.
- **Cost:** $0 (an open model on a local CPU; reports print `unpriced`).
- **Date:** 2026-10-26.

### What broke on first contact

Every surface that failed the first time the framework or the model touched
Replay, and what was done about it. Items 1 to 7 are Replay's; 8 and 9 are the
example's; 10 to 13 are the model's behaviour, which is what the suite is
for.

| # | Surface | How it showed | Fix |
|---|---|---|---|
| 1 | `client.beta.messages.create` | The first `replay record` ended in `ReplayError: client.beta is not recorded yet`, 0 envelopes: pydantic-ai sends every request through the beta surface, which the wrapper refused | Record `beta.messages.create` (plain and `stream=True`) and `beta.messages.stream` under `anthropic.beta.messages.create`, `betas` in the fingerprint, replies decoded to `BetaMessage` and the beta stream event classes (pydantic-ai asserts `isinstance(event, BetaRawMessageStartEvent)`) |
| 2 | Offline stand-in client | The first `full` replay of a streamed run failed with `AttributeError: '_AsyncOffline' object has no attribute 'base_url'`: pydantic-ai copies `client.base_url` into every response | The stand-in exposes `base_url`, `timeout` and `max_retries` (SDK defaults; nothing is ever sent) |
| 3 | Rebuilt beta streams | A test of `beta.messages.stream()` failed with `NoHTTPResponseError`: the SDK's `BetaMessageStream` reads `response.request.headers` on every event (to look for the fine-grained tool streaming beta), and a stream rebuilt from a cassette has no HTTP response. It broke `beta.messages.stream` always, and recording `create(stream=True)` from the fake client | A rebuilt beta stream carries a stand-in response whose request headers hold the recorded `betas`; a plain stream still refuses `.response` |
| 4 | `client.with_options()` / `.copy()` | Found reading pydantic-ai's request path: a request with `max_retries` set goes through `client.with_options(...)`, which the proxy forwarded to the bare client, so the call would have gone live unrecorded | Both return a wrapped copy |
| 5 | `count_tokens`, batches | `messages.count_tokens` was forwarded to the live client inside a session (pydantic-ai calls it for `UsageLimits(count_tokens_before_request=True)`) | Inside a session they raise a `ReplayError` that names them and says token counting is not supported |
| 6 | Async client teardown | Every `model-live` run of the async agent in `replay flake` printed `RuntimeError: Event loop is closed`: the client made from `--client` for the run was garbage-collected after `asyncio.run` had closed its loop | `replay ci` / `flake` close the client they made, on the run's loop (verified: a `model-live` run on Ollama no longer prints it) |
| 7 | Cost of an unpriced model | `replay flake` printed `$0.0000` for runs of a model with no price | Reports print `unpriced` (or `$x + N unpriced calls`); a budget cost limit fails closed on unpriced calls |
| 8 | Tool errors in pydantic-ai | One recording attempt crashed the whole run: qwen2.5:7b called `get_booking` with an invented reference (`DL2207-AL-20261105`), the tool raised `ValueError`, and pydantic-ai only sends `ModelRetry` back to the model (`eval/results/m9_first_contact/recording_attempts_before_model_retry.json`) | The tools raise `ModelRetry` for bad arguments; Replay re-raises a recorded `ModelRetry` as a `ModelRetry`, so a replay gives the model the same retry prompt (tested) |
| 9 | A check that was too lenient | The first passing recording of `cheapest_fare_in_euros` stated the right price under the wrong flight: "UA512 departing at 11:40, … 173.88 EUR" (the cheapest is B6416; `recording_attempts_before_rule_fix.json`). The answer rule only looked for `173.88` | The answer must state the flight and the price; the booking answer must state the reference the tool returned |
| 10 | Parallel calls with a guessed argument | In the very first recording, with the first prompt, qwen2.5:7b called `search_flights` and `convert_currency` in the same turn, converting a guessed $350, and answered "approximately 165.00 EUR" (`eval/results/m9_first_contact/cheapest_fare_in_euros.jsonl`) | One prompt line ("give a tool only values from the message or earlier results"); no recording since has done it |
| 11 | An invented argument | The first recording of the cancellation called `cancel_booking(booking_ref, passenger)` with no lookup; pydantic-ai rejected the extra argument and the model asked the user a question (`m9_first_contact/cancel_refundable_booking.jsonl`) | None: the suite fails it |
| 12 | Cancelling without a lookup | Across the 15 recording attempts and the 10 flake runs of the cancellation, 23 of 25 cancelled without calling `get_booking` first. The booking happened to be refundable, so every outcome check passed; only `looked_up_before_writing` and the `in_order` sequence failed them | None: the suite fails it. No qwen2.5:7b recording of this task passes, so its `model-live` scenario takes its tools from the fake-model recording |
| 13 | A claimed action that never happened | The first recording of the booking asked "Shall I book this flight for Ana Lima?" and then wrote "Flight booked for Ana Lima…" with no `book_flight` call (`m9_first_contact/book_earliest_flight.jsonl`). One cancellation attempt called only `get_booking` and answered "The booking QX7Y2K has been canceled and you will receive a full refund of $243.00 USD" without calling `cancel_booking` (`recording_attempts_before_rule_fix.json`, attempt 2) | None: `writes_match` fails both; for the cancellation, `answer_states` alone would have passed it |

### The committed recordings

`make record-pydantic-ai` records each task up to five times on qwen2.5:7b and
keeps the first attempt that passes the suite; every attempt is logged in
`eval/results/m9_qwen_recording.json`:

```
$ uv run python -m examples.pydantic_ai_agent.record_all --live --attempts 5
cheapest_fare_in_euros attempt 1: passed in 55.9 s
book_earliest_flight attempt 1: passed in 67.4 s
cancel_refundable_booking attempt 1: failed ['rule:looked_up_before_writing', 'tool_sequence'] in 26.6 s
cancel_refundable_booking attempt 2: failed ['rule:looked_up_before_writing', 'tool_sequence'] in 24.6 s
cancel_refundable_booking attempt 3: failed ['rule:looked_up_before_writing', 'tool_sequence'] in 20.4 s
cancel_refundable_booking attempt 4: failed ['rule:looked_up_before_writing', 'tool_sequence'] in 17.2 s
cancel_refundable_booking attempt 5: failed ['rule:looked_up_before_writing', 'tool_sequence'] in 17.6 s
cancel_refundable_booking: no attempt passed; its cassette was left as it was
logged 7 attempt(s) to eval/results/m9_qwen_recording.json
```

The two qwen2.5:7b cassettes name the model in their header (`"model":
"qwen2.5:7b", "provider": "ollama"`, with the digest) and in every response.
Both suites replay in `full` with every step `exact` (`make eval`, so CI runs
it), and replay just as exactly against `agent.run_stream`, because `stream`
is not part of a request's identity:

```
$ make eval-pydantic-ai-offline
PASS replay.toml cheapest_fare_in_euros: 5 steps, 0 approximate
PASS replay.toml book_earliest_flight: 5 steps, 0 approximate
PASS replay.toml cancel_refundable_booking: 5 steps, 0 approximate
PASS replay-qwen.toml cheapest_fare_in_euros: 5 steps, 0 approximate
PASS replay-qwen.toml book_earliest_flight: 5 steps, 0 approximate
PASS replay-qwen.toml cancel_refundable_booking: 5 steps, 0 approximate
```

### `replay flake` in `model-live`, 10 runs per scenario

The model live (qwen2.5:7b through Ollama), the tools served from the
recordings, 10 runs per scenario, one at a time. `REPLAY_STATS_IMPL=baseline`
(`stats.py` is not written yet): the gate is a fixed 10 runs that passes at a
pass rate of at least 0.75, and the intervals are Wald intervals, so read the
pass counts. Raw report `eval/results/m9_flake_qwen.json`; summary with the
approximate steps per run, per-run times and the environment
`eval/results/m9_flake_qwen_summary.json`.

```
$ make eval-pydantic-ai
$ uv run replay flake examples/pydantic_ai_agent/replay-qwen.toml --no-store --mode model-live --client examples.pydantic_ai_agent.agent:make_client --max-runs 10 --concurrency 1 --json eval/results/m9_flake_qwen.json
PASS cheapest_fare_in_euros (passing 10/10 (100%), interval [1.00, 1.00])
FAIL book_earliest_flight (flaky 2/10 (20%), interval [0.00, 0.45])
FAIL cancel_refundable_booking (failing 0/10 (0%), interval [0.00, 0.00])
1/3 scenario(s) passed the gate (1 flaky; 30 runs, unpriced; gate: fixed 10
runs, pass on a pass rate of at least 0.75 (the baseline stand-in for the SPRT);
Wald intervals): failed
cheapest_fare_in_euros: 10/10 passed (passing), approximate steps per run [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
book_earliest_flight: 2/10 passed (flaky), approximate steps per run [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
cancel_refundable_booking: 0/10 passed (failing), approximate steps per run [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
wall time 1026.3 s on AMD EPYC 7763 64-Core Processor
```

| Scenario (qwen2.5:7b, `model-live`) | Passed | Approximate steps per run | Seconds per run (min / median / max) |
|---|---|---|---|
| `cheapest_fare_in_euros` | 10 / 10 | 0 in every run | 41.3 / 47.3 / 66.5 |
| `book_earliest_flight` | 2 / 10 | 1 in run 0, else 0 | 31.9 / 33.6 / 55.6 |
| `cancel_refundable_booking` | 0 / 10 | 0 in every run | 14.3 / 14.5 / 17.1 |
| All 30 runs | 12 / 30 | 1 step in 1 run | wall 1026 s (17.1 min) |

Beside the v1 rows (retail agent, `replay flake` on live models, M4, kept as
they were): `claude-haiku-5-5` passed 119 / 120 runs and `claude-opus-5-5`
30 / 30, every scenario passing or flaky with at most one failure. The local
model's runs are a different agent and different tasks, so this is not a
model comparison. It is the first live data that cost nothing.

Where the failures came from (per-run `failed_checks` and step counts in the
raw report; a flake report keeps no answers):

- **`book_earliest_flight`, 8 failures.** 6 runs searched once and stopped
  after one more model call without booking (`tool_sequence`, `writes_match`
  and `answer_states` failed; 2 model calls, about 1,400 tokens). That is what
  the first-contact attempt that searched without booking did, when it
  answered "The earliest flight is DL2207 … Would you…". One run converted the
  fare to euros, a call the recording never made, and ended in
  `UnrecordedCallError`. One run booked with arguments that differed from the
  recording, so the booking step was served the recorded result as
  `approximate`, and `writes_match` failed it.
- **`cancel_refundable_booking`, 10 failures.** 9 runs cancelled without a
  lookup (row 12 above). One made no tool call at all.
- **No pass needed an approximate step.** `passes_with_an_approximate_step` is
  0 for every scenario: no pass came from a recorded result served for
  different arguments, the hazard PLAN.md's round 1 found.

What this shows and does not show: on this agent qwen2.5:7b is reliable on a
two-step lookup (10/10) and unreliable as soon as it has to act (2/10 and
0/10). A single recorded run would have passed `book_earliest_flight` (both
committed recordings passed on attempt 1) and hidden an 80% failure rate.
Ten runs per scenario cannot tell 0.2 from 0.3; the counts are the result, and
Wilson intervals come with `stats.py`. The runs used default sampling on one
CPU type; another machine or Ollama version may sample differently, which is
why the digest, version and CPU are recorded.

**Provenance.** The flake report records `git_sha` 497f501, the commit before
the example itself was committed; the example code that ran is the one
committed in the next three commits (it was not changed after the run). The
teardown fix (row 6) came after this run and changes no outcome; it only
removes the traceback.

## Tool-result drift check (M10, offline)

Can `tools-live` catch a tool that returns something else without the false
alarm `fail_on_approximate` raises on a refactor that only changes request
bytes? In the M4 study three read-only tool bugs (`user_lookup_builds_id`,
`product_details_all_unavailable`, `order_details_stale_status`) were caught
only by `no_approximate_steps`, at the price of failing the benign
`compact_tool_json` refactor.

- **Check:** `fail_on_tool_drift` (`agent_replay/drift.py`). In `tools-live`,
  each live tool call is paired with an unused recorded call of the same
  tool, at the same nesting level, with the same request fingerprint
  (arguments): the first one whose result it equals, else the first one.
  Those pairs are compared as canonical JSON; a call with no
  such recording is counted, consumes nothing and is not compared. A difference fails the scenario as
  `no_tool_drift` with the first differing path. Per-tool `ignore_fields`
  skip paths that legitimately vary (none were needed here).
- **Hypothesis (PLAN.md, written before the run):** `tools-live+drift` 8/8 tool bugs, 0/5
  false alarms, 0 drift on the six live cassettes replayed `tools-live`;
  `drops_error_flag` still survives.
- **Dataset and method:** the M4 mutation study unchanged (D4: 32 mutations,
  5 benign refactors, the six-scenario fake-model suite), with a sixth
  configuration `tools-live+drift`. The five M4 columns were re-run in the
  same command and are identical to the committed M4 results (0 cells
  differ). Then the six live `claude-opus-5-5` recordings
  (`examples/retail_agent/replay-live.toml`), replayed `tools-live` with the
  flag: the model replies come from the recordings, the τ-bench tools run
  for real.
- **Commands:** `make eval-mutations` (55.6 s on 4 vCPU, AMD EPYC 7763;
  byte-stable, a second run left `eval/results/mutations.json` unchanged)
  and `REPLAY_JUDGE_OFFLINE=1 uv run replay ci examples/retail_agent/replay-live.toml --mode tools-live --fail-on-tool-drift --no-store --json report.json`
  (2.2 s). Both now also run in `make ci-demo`. $0.

Raw output of `make eval-mutations`: the tables, with only the check types
and mutations this section is about (the full matrix is in
`eval/results/mutations.json` and printed by the command):

```
### Detection rate by configuration

| configuration | all | agent | model | tool |
|---|---|---|---|---|
| full | 7/32 (22%) | 7/12 (58%) | 0/12 (0%) | 0/8 (0%) |
| tools-live | 12/32 (38%) | 7/12 (58%) | 0/12 (0%) | 5/8 (62%) |
| model-live | 22/32 (69%) | 10/12 (83%) | 12/12 (100%) | 0/8 (0%) |
| model-live+fallback | 22/32 (69%) | 10/12 (83%) | 12/12 (100%) | 0/8 (0%) |
| tools-live+strict | 19/32 (59%) | 11/12 (92%) | 0/12 (0%) | 8/8 (100%) |
| tools-live+drift | 15/32 (47%) | 7/12 (58%) | 0/12 (0%) | 8/8 (100%) |

### Mutations caught per check type (caught / all mutations, alone)

| check type | full | tools-live | model-live | model-live+fallback | tools-live+strict | tools-live+drift |
|---|---|---|---|---|---|---|
| `no_approximate_steps` | 0/32 (alone 0) | 0/32 (alone 0) | 0/32 (alone 0) | 0/32 (alone 0) | 14/32 (alone 7) | 0/32 (alone 0) |
| `no_tool_drift` | 0/32 (alone 0) | 0/32 (alone 0) | 0/32 (alone 0) | 0/32 (alone 0) | 0/32 (alone 0) | 6/32 (alone 3) |
| `rule:db_matches_ground_truth` | 0/32 (alone 0) | 7/32 (alone 5) | 0/32 (alone 0) | 0/32 (alone 0) | 7/32 (alone 2) | 7/32 (alone 2) |

### Per mutation (check types that caught it; — survived)

| mutation | target | full | tools-live | model-live | model-live+fallback | tools-live+strict | tools-live+drift |
|---|---|---|---|---|---|---|---|
| `cancel_keeps_status` | tool | — | `rule:db_matches_ground_truth` | — | — | `rule:db_matches_ground_truth` | `rule:db_matches_ground_truth` |
| `cancel_skips_gift_card_refund` | tool | — | `rule:db_matches_ground_truth` | — | — | `rule:db_matches_ground_truth` | `rule:db_matches_ground_truth` |
| `address_drops_address2` | tool | — | `rule:db_matches_ground_truth` | — | — | `no_approximate_steps`, `rule:db_matches_ground_truth` | `no_tool_drift`, `rule:db_matches_ground_truth` |
| `return_ignores_item_ids` | tool | — | `rule:db_matches_ground_truth` | — | — | `no_approximate_steps`, `rule:db_matches_ground_truth` | `no_tool_drift`, `rule:db_matches_ground_truth` |
| `exchange_flips_price_difference` | tool | — | `rule:db_matches_ground_truth` | — | — | `no_approximate_steps`, `rule:db_matches_ground_truth` | `no_tool_drift`, `rule:db_matches_ground_truth` |
| `user_lookup_builds_id` | tool | — | — | — | — | `no_approximate_steps` | `no_tool_drift` |
| `product_details_all_unavailable` | tool | — | — | — | — | `no_approximate_steps` | `no_tool_drift` |
| `order_details_stale_status` | tool | — | — | — | — | `no_approximate_steps` | `no_tool_drift` |

### Benign refactors (false alarms; approximate steps)

| refactor | full | tools-live | model-live | model-live+fallback | tools-live+strict | tools-live+drift |
|---|---|---|---|---|---|---|
| `extracted_loop` | pass; 0 approx | pass; 0 approx | pass; 0 approx | pass; 0 approx | pass; 0 approx | pass; 0 approx |
| `parallel_tools` | pass; 0 approx | pass; 0 approx | pass; 0 approx | pass; 0 approx | pass; 0 approx | pass; 0 approx |
| `compact_tool_json` | pass; 17 approx | pass; 17 approx | pass; 0 approx | pass; 0 approx | FALSE ALARM: no_approximate_steps; 17 approx | pass; 17 approx |
| `more_turns` | pass; 0 approx | pass; 0 approx | pass; 0 approx | pass; 0 approx | pass; 0 approx | pass; 0 approx |
| `async_agent` | pass; 0 approx | pass; 0 approx | pass; 0 approx | pass; 0 approx | pass; 0 approx | pass; 0 approx |

False alarms: full 0/5, tools-live 0/5, model-live 0/5, model-live+fallback 0/5, tools-live+strict 1/5, tools-live+drift 0/5.
Survived every configuration: `drops_error_flag`.
```

The live recordings, `no_tool_drift` per scenario from `report.json`
(every scenario passed, 6/6):

```
cancel_pending_laptop       6 live tool result(s) matched the recording
modify_address_suite        3 live tool result(s) matched the recording
return_damaged_bicycle      7 live tool result(s) matched the recording
exchange_waterproof_boots   8 live tool result(s) matched the recording
lookup_latest_order_total   5 live tool result(s) matched the recording
transfer_swapped_refunds    5 live tool result(s) matched the recording
```

What this shows:

- **The hypothesis held on every count.** `tools-live+drift` caught 8/8 tool
  bugs with 0/5 false alarms; `tools-live+strict` also catches 8/8 but fails
  `compact_tool_json`. The live recordings show 0 drift in 34 live tool
  results. `drops_error_flag` still survives every configuration.
- **It catches what the tool returns, not what it does.** `no_tool_drift`
  caught 6 tool bugs, 3 of them alone (the three read-only ones). The two
  cancel bugs (`cancel_keeps_status`, `cancel_skips_gift_card_refund`) return
  exactly what the real tool returns and only corrupt the database
  afterwards, so their results match the recording; the database rule catches
  them, as in M4. Drift is a check on results, and a side effect needs a
  state check.
- **It adds nothing on agent mutations, by design.** `tools-live+drift`
  detects 7/12 agent mutations, the same 7 as plain `tools-live`;
  `tools-live+strict` detects 11/12, because a changed agent changes model
  requests. Drift compares only calls made with the arguments that produced
  the recording. With `wrong_write_argument` (a separate run of the six
  fake-model scenarios with the flag), 29 live tool results were compared,
  0 drifted, and 4 write calls with other arguments were not compared; the
  ground-truth rules fail it. A changed argument is the agent's fault, and
  reporting it as tool drift would blame the wrong component. So the two flags
  are complementary, not one replacing the other. The separation is not
  complete: for a tool whose result depends on earlier calls, an agent that
  drops or reorders a write before a read with the same arguments gets a
  different read result, and that is reported as drift; so is an added
  repeated read before the write, which takes the recording of the read
  after it (unit tests pin both down: `test_a_dropped_write_before_a_read_reads_as_drift_on_a_stateful_tool`,
  `test_an_added_repeated_read_before_a_write_reads_as_drift`). A dropped
  repeated read is not drift: the remaining read pairs with the recording of
  the same result. No mutation in this study does any of these. (The
  review found the dropped-read false alarm in the first pairing, which took
  the first same-argument recording whatever its result; the mutation table
  is unchanged by the fix.)
- **Caveats.** The live tools here are deterministic over a fixed, vendored
  database, so `ignore_fields` was never exercised on real data. A tool that
  reads a clock or a remote service will drift every run until its varying
  paths are listed. The same flag on the pydantic-ai suites (the fake-model
  and the qwen2.5:7b recordings, `--mode tools-live --fail-on-tool-drift`)
  compared 2 live tool results per scenario, 0 drifted.

## Identifying arguments and correct-but-different paths (M11, offline)

When no recording matches a served tool call exactly, the session serves the
oldest unused recording of the same tool and flags the step `approximate`.
That is right for a reworded `transfer_to_human_agents` summary and wrong for
a read of another order, which gets the recorded order's details. Can a suite
name the arguments that identify a recording, so a call for an entity the
cassette never saw is `unrecorded` and the policy decides, without breaking
the reworded summary? And which check types raise a false alarm when a correct
agent takes another path?

- **Mechanism:** `[defaults.identify]` in a suite (`agent_replay/identify.py`,
  used by `Session._take` and `drift.tool_drift`). A served call only falls
  back to a recording with the same values at the named argument paths;
  with none left it is `unrecorded`. In `tools-live` with
  `fail_on_tool_drift`, a live call of a recorded tool whose identity none of
  its recordings has, used or not, is drift with reason `identity_unrecorded`
  (a tool the scenario never recorded stays counted as unrecorded). The retail suites declare
  the user, order and product ids of their nine entity tools;
  `transfer_to_human_agents` and `calculate` declare none. The pydantic-ai
  suites declare all five tools (`search_flights` by route and date,
  `convert_currency` by amount and currencies, the booking tools by
  reference, `book_flight` by flight and passenger).
- **Hypothesis (PLAN.md, written before the run):** `guesses_user_id` and
  `reads_wrong_order` are caught 0/2 in the five non-strict configurations
  without keys and 2/2 in `full`, `model-live` and `tools-live+drift` with
  them. The five refactors stay at 0/5 false alarms (1/5 strict), and the six
  live cassettes still pass. Of the five path variants in `model-live`, the
  exact `tool_sequence` fails 4/5 and the ground-truth rules 0/5. Under
  `strict` the repeated lookup and the verifying read end as `unrecorded`,
  and under `fallback` none do.
- **Dataset and method:** the M4 study (D4: six fake-model retail
  scenarios), unchanged. The v1 tables still run without the keys
  (`Overrides(identify={})`), and their 37 rows in
  `eval/results/mutations.json` are byte-identical to M10's. The study then
  re-runs all 32 mutations and 5 refactors with the keys, plus the two new
  agent mutations, in the six configurations. Both new mutations change only
  a *read's* identifying argument: `guesses_user_id` fetches the profile as
  `emma_smith` instead of `emma_smith_8564`, and `reads_wrong_order` mistypes
  the last id when it reads several orders at once, which is never the order
  later written. So the request rule `checked_before_writing` still sees the
  right ids, and only the replay's choice of recording decides. (A wrong read
  of the written order is already caught by that rule.) The five path variants
  (`eval/mutations/paths.py`) wrap the fake model, and each reaches the same
  database and answer. They run in `model-live` against the six recordings
  under `strict` and `fallback`, with and without the keys. Per scenario they
  record the failed check types and how the run ended.
- **Commands:** `make eval-mutations` (117 s on 4 vCPU, AMD EPYC 7763; a
  second run left `eval/results/mutations.json` unchanged); the live suite as
  in M10, now with its keys:
  `REPLAY_JUDGE_OFFLINE=1 uv run replay ci examples/retail_agent/replay-live.toml --mode tools-live --fail-on-tool-drift --no-store --json report.json`;
  and both pydantic-ai suites in `full` and in `tools-live --fail-on-tool-drift`
  (`uv run replay ci examples/pydantic_ai_agent/replay.toml --no-store`, and
  the same for `replay-qwen.toml`). $0.

Raw output of `make eval-mutations`, the M11 sections (the v1 tables above
it are unchanged and not repeated):

```
### Mutations only identity can see (caught by; — survived)

| mutation | identify | full | tools-live | model-live | model-live+fallback | tools-live+strict | tools-live+drift |
|---|---|---|---|---|---|---|---|
| `guesses_user_id` | without | — | — | — | — | `no_approximate_steps` | — |
| `guesses_user_id` | with | `output_schema`, `run_error:UnrecordedCallError` | — | `arg_schema`, `output_schema`, `required_tools`, `rule:matches_ground_truth`, `run_error:UnrecordedCallError`, `tool_sequence` | `arg_schema`, `output_schema`, `required_tools`, `rule:matches_ground_truth`, `run_error:JSONDecodeError`, `tool_sequence` | `no_approximate_steps` | `no_tool_drift` |
| `reads_wrong_order` | without | — | — | — | — | `no_approximate_steps` | — |
| `reads_wrong_order` | with | `run_error:UnrecordedCallError` | — | `run_error:UnrecordedCallError`, `tool_sequence` | `tool_sequence` | `no_approximate_steps` | `no_tool_drift` |

Detected (of 2) without → with the keys: full 0 → 2, tools-live 0 → 0, model-live 0 → 2, model-live+fallback 0 → 2, tools-live+strict 2 → 2, tools-live+drift 0 → 2.

### Detection rate with the keys (34 mutations)

| configuration | all | agent | model | tool |
|---|---|---|---|---|
| full | 9/34 (26%) | 9/14 (64%) | 0/12 (0%) | 0/8 (0%) |
| tools-live | 12/34 (35%) | 7/14 (50%) | 0/12 (0%) | 5/8 (62%) |
| model-live | 24/34 (71%) | 12/14 (86%) | 12/12 (100%) | 0/8 (0%) |
| model-live+fallback | 24/34 (71%) | 12/14 (86%) | 12/12 (100%) | 0/8 (0%) |
| tools-live+strict | 21/34 (62%) | 13/14 (93%) | 0/12 (0%) | 8/8 (100%) |
| tools-live+drift | 17/34 (50%) | 9/14 (64%) | 0/12 (0%) | 8/8 (100%) |

False alarms with the keys: full 0/5, tools-live 0/5, model-live 0/5, model-live+fallback 0/5, tools-live+strict 1/5, tools-live+drift 0/5.

Cells of the v1 tables the keys changed:

- none
```

```
| variant | policy | identify | affected | termination | approx | fallback | `arg_schema` | `budget` | `forbidden_tools` | `max_tool_calls` | `output_schema` | `required_tools` | `rule:checked_before_writing` | `rule:db_matches_ground_truth` | `rule:matches_ground_truth` | `tool_sequence` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `reads_only_needed_orders` | strict | on | 3/6 | 3 checks, 3 passed | 0 | 0 | · | · | · | · | · | · | · | · | · | 3 |
| `reads_only_needed_orders` | strict | off | 3/6 | 3 checks, 3 passed | 0 | 0 | · | · | · | · | · | · | · | · | · | 3 |
| `reads_only_needed_orders` | fallback | on | 3/6 | 3 checks, 3 passed | 0 | 0 | · | · | · | · | · | · | · | · | · | 3 |
| `reads_only_needed_orders` | fallback | off | 3/6 | 3 checks, 3 passed | 0 | 0 | · | · | · | · | · | · | · | · | · | 3 |
| `swaps_parallel_lookups` | strict | on | 5/6 | 1 checks, 5 passed | 0 | 0 | · | · | · | · | · | · | · | · | · | 1 |
| `swaps_parallel_lookups` | strict | off | 5/6 | 1 checks, 5 passed | 0 | 0 | · | · | · | · | · | · | · | · | · | 1 |
| `swaps_parallel_lookups` | fallback | on | 5/6 | 1 checks, 5 passed | 0 | 0 | · | · | · | · | · | · | · | · | · | 1 |
| `swaps_parallel_lookups` | fallback | off | 5/6 | 1 checks, 5 passed | 0 | 0 | · | · | · | · | · | · | · | · | · | 1 |
| `repeats_order_lookup` | strict | on | 4/6 | 2 passed, 4 unrecorded | 0 | 0 | 2 | 4 | · | 4 | · | 2 | · | · | 4 | 4 |
| `repeats_order_lookup` | strict | off | 4/6 | 2 passed, 4 unrecorded | 0 | 0 | 2 | 4 | · | 4 | · | 2 | · | · | 4 | 4 |
| `repeats_order_lookup` | fallback | on | 4/6 | 4 checks, 2 passed | 0 | 4 | · | · | · | · | · | · | · | · | · | 4 |
| `repeats_order_lookup` | fallback | off | 4/6 | 4 checks, 2 passed | 0 | 4 | · | · | · | · | · | · | · | · | · | 4 |
| `verifies_after_write` | strict | on | 4/6 | 2 passed, 4 unrecorded | 0 | 0 | · | · | · | · | · | · | · | · | · | 4 |
| `verifies_after_write` | strict | off | 4/6 | 2 passed, 4 unrecorded | 0 | 0 | · | · | · | · | · | · | · | · | · | 4 |
| `verifies_after_write` | fallback | on | 4/6 | 4 checks, 2 passed | 0 | 4 | · | · | · | · | · | · | · | · | · | 4 |
| `verifies_after_write` | fallback | off | 4/6 | 4 checks, 2 passed | 0 | 4 | · | · | · | · | · | · | · | · | · | 4 |
| `rewords_transfer_summary` | strict | on | 1/6 | 6 passed | 1 | 0 | · | · | · | · | · | · | · | · | · | · |
| `rewords_transfer_summary` | strict | off | 1/6 | 6 passed | 1 | 0 | · | · | · | · | · | · | · | · | · | · |
| `rewords_transfer_summary` | fallback | on | 1/6 | 6 passed | 1 | 0 | · | · | · | · | · | · | · | · | · | · |
| `rewords_transfer_summary` | fallback | off | 1/6 | 6 passed | 1 | 0 | · | · | · | · | · | · | · | · | · | · |

Per policy (variants out of 5):

- strict, identify on: 4 raise a false alarm; 2 end on an unrecorded call (`repeats_order_lookup`, `verifies_after_write`); failing check types: `arg_schema` 1, `budget` 1, `max_tool_calls` 1, `required_tools` 1, `rule:matches_ground_truth` 1, `tool_sequence` 4.
- strict, identify off: 4 raise a false alarm; 2 end on an unrecorded call (`repeats_order_lookup`, `verifies_after_write`); failing check types: `arg_schema` 1, `budget` 1, `max_tool_calls` 1, `required_tools` 1, `rule:matches_ground_truth` 1, `tool_sequence` 4.
- fallback, identify on: 4 raise a false alarm; 0 end on an unrecorded call (none); failing check types: `tool_sequence` 4.
- fallback, identify off: 4 raise a false alarm; 0 end on an unrecorded call (none); failing check types: `tool_sequence` 4.
```

The live recordings with their keys, `no_tool_drift` per scenario from
`report.json` (6/6 passed, every step `exact` or `live`):

```
cancel_pending_laptop       6 live tool result(s) matched the recording
modify_address_suite        3 live tool result(s) matched the recording
return_damaged_bicycle      7 live tool result(s) matched the recording
exchange_waterproof_boots   8 live tool result(s) matched the recording
lookup_latest_order_total   5 live tool result(s) matched the recording
transfer_swapped_refunds    5 live tool result(s) matched the recording
```

The pydantic-ai suites with their keys, in `full`:

```
$ uv run replay ci examples/pydantic_ai_agent/replay.toml --no-store
PASS cheapest_fare_in_euros (full; 5 exact)
PASS book_earliest_flight (full; 5 exact)
PASS cancel_refundable_booking (full; 5 exact)
3/3 scenario(s) passed: passed
$ uv run replay ci examples/pydantic_ai_agent/replay-qwen.toml --no-store
PASS cheapest_fare_in_euros (full; 5 exact)
PASS book_earliest_flight (full; 5 exact)
PASS cancel_refundable_booking (full; 5 exact)
3/3 scenario(s) passed: passed
```

In `tools-live --fail-on-tool-drift`, every scenario of both suites passed
with "2 live tool result(s) matched the recording".

What this shows:

- **Mutation half: the hypothesis held, with one addition.** Without keys
  both identity mutations survive every non-strict configuration (0/2). In
  `full` and `model-live` the misdirected read is served the right entity's
  recording, and in `tools-live` the live call has other arguments, so
  drift does not compare it. With keys they are caught 2/2 in `full`,
  `model-live` and `tools-live+drift`. They are also caught 2/2 in
  `model-live+fallback`, which the plan did not predict. There the real tool
  runs for the unrecorded identity. `reads_wrong_order` then needs one extra
  lookup, which `tool_sequence` sees. For `guesses_user_id`, the real tool
  rejects the guessed id and the fake model crashes on the error result
  (`run_error:JSONDecodeError`). The fake model never saw an error there; a
  real model would answer it, and the ground-truth rule fails either way.
  Plain `tools-live` stays at 0/2 by construction: no tool call is served
  there. `tools-live+strict` catches both with or without keys, through the
  approximate model step that follows.
- **The keys cost nothing measurable here.** No cell of the v1 tables
  changed with the keys: all 32 mutations and 5 refactors give the same
  outcome and the same approximate count. False alarms stay at 0/5 (1/5
  under `fail_on_approximate`, `compact_tool_json` as before). The six live
  recordings pass `tools-live --fail-on-tool-drift` with 0 drift in 34 live
  tool results, and every committed pydantic-ai cassette replays in `full`
  with 0 approximate steps. Detection with the keys is 9/34 in `full`, 24/34
  in `model-live` and 17/34 in `tools-live+drift`: the v1 counts (7, 22, 15
  of 32) plus the two new mutations.
- **The summary step stays approximate where it can be.** The reworded
  summary (`rewords_transfer_summary`) passes in all 4 arms with 1
  approximate step: `transfer_to_human_agents` has no key, so it is still
  served. On the six live recordings in `tools-live` the transfer runs live,
  so there is no approximate step to keep. The approximate summary in
  RESULTS.md M2 came from a live Opus run in `model-live`, which needs a key
  to repeat.
- **Path half: `tool_sequence` raises the false alarms, and `strict` adds
  its own.** Under `fallback` the exact `tool_sequence` fails 4/5 variants
  and every other check type 0/5: the ground-truth rules passed every
  correct path. Under `strict` the repeated lookup and the verifying read end
  as `unrecorded` in 4 scenarios each, as predicted, and `fallback` ends none
  that way. The predicted 0/5 for the ground-truth rules fails under
  `strict`. `repeats_order_lookup` loses its run to the unrecorded re-read
  before it writes, and the scenario then also fails
  `rule:matches_ground_truth`, `budget`, `max_tool_calls`, `arg_schema` and
  `required_tools`. That is the frozen world deciding, not the agent.
  `swaps_parallel_lookups` changes 5 scenarios but fails `tool_sequence` in 1
  (`modify_address_suite`), because the check compares tool names. Reversing
  several `get_order_details` calls keeps the name sequence. Only the
  user/order pair changes it.
- **Keys and the frozen world are separate problems.** The keys changed
  none of the 20 path-variant rows. Every `unrecorded` ending here is a
  second call to an entity whose one recording was already used, which is
  unrecorded with or without the keys. The keys decide something else: which
  recording a call for a *different* entity may take. So a team that reads
  an entity twice should replay `model-live` under `fallback`, or record the
  second read. The keys do not help, and they do not hurt.
- **Caveats.** Everything here is the fake model on fake-model recordings.
  The real-model test of the keys is M12's `strict` arm on qwen2.5:7b. M9's
  one approximate booking (a `book_flight` with other arguments) would now
  be `unrecorded` under `book_flight`'s keys, but the committed M9 report
  keeps no step arguments to confirm it offline. Identity values compare
  exactly, so a model that writes `"sfo"` for `"SFO"` reads as another
  entity.

## Frozen-world bias and the verdict-stability curve (M12)

Two questions (PLAN.md, M12). First, when a real model runs live against tools
served from a recording (`model-live`), how many of its failures come from the
frozen world and how many from the model? Second, how many runs does the
flakiness gate's verdict need before it stops changing?

### `model-live` under `strict` and `fallback` on qwen2.5:7b

- **Dataset:** the two pydantic-ai travel-desk scenarios whose cassettes
  qwen2.5:7b recorded (`cheapest_fare_in_euros`, `book_earliest_flight`,
  `examples/pydantic_ai_agent/replay-qwen.toml`), with the suite's M11
  `identify` keys on.
- **Method:** the model live (qwen2.5:7b through Ollama 0.40.11, digest
  `845dbda0…`, default sampling, no seed), the tools served from the
  recordings, 20 runs per scenario under each unrecorded-call policy, one run
  at a time. Under `strict` a call that no unused recording matches ends the
  run. Under `fallback` it runs the real tool. Every run is classified by how
  it ended (`eval/pydantic_ai.py`, `cause`): **passed**; **unrecorded** (an
  unrecorded call, by tool or by identity, ended a `strict` run); **approximate**
  (a step was served another request's recording and a check then failed);
  **model** (a check failed, or the agent raised, with every served step
  exact). A pass with an approximate step would be counted separately, and
  none occurred. The gate is the committed reports' baseline rule: pass at a
  pass rate of at least 0.75.
- **Machine:** 4 vCPU, AMD EPYC 7763, CPU only. $0.
- **Raw output:** `eval/results/m12_arms_qwen.json` (every run's cause, failed
  checks, step counts, error and seconds).

```
$ make eval-pydantic-ai-arms
uv run python -m eval.pydantic_ai --arms strict,fallback --runs 20
PASS replay.toml cheapest_fare_in_euros: 5 steps, 0 approximate
  ... (the six full-mode replays of M9, all PASS)
arm strict: 20 model-live runs of cheapest_fare_in_euros, book_earliest_flight
  cheapest_fare_in_euros: 18/20 passed (18 passed, 2 model)
  book_earliest_flight: 9/20 passed (9 passed, 3 unrecorded, 8 model)
  wall time 1562.3 s
arm fallback: 20 model-live runs of cheapest_fare_in_euros, book_earliest_flight
  cheapest_fare_in_euros: 19/20 passed (19 passed, 1 model)
  book_earliest_flight: 8/20 passed (8 passed, 12 model)
  wall time 1528.3 s
wrote eval/results/m12_arms_qwen.json
```

| qwen2.5:7b, `model-live` | Passed | passed / unrecorded / approximate / model | Gate | Seconds per run (min / median / max) | Wall |
|---|---|---|---|---|---|
| M9: `cheapest_fare_in_euros` (no keys, 10 runs) | 10 / 10 | (not classified in M9) | pass | 41.3 / 47.3 / 66.5 | 1026 s for M9's 30 runs |
| M9: `book_earliest_flight` (no keys, 10 runs) | 2 / 10 | (1 `UnrecordedCallError`, 1 approximate) | fail | 31.9 / 33.6 / 55.6 | |
| M12 `strict`: `cheapest_fare_in_euros` | 18 / 20 | 18 / 0 / 0 / 2 | pass | 40.1 / 40.4 / 54.7 | 1562 s for the arm's 40 runs |
| M12 `strict`: `book_earliest_flight` | 9 / 20 | 9 / 3 / 0 / 8 | fail | 30.3 / 36.9 / 45.8 | |
| M12 `fallback`: `cheapest_fare_in_euros` | 19 / 20 | 19 / 0 / 0 / 1 | pass | 40.0 / 40.3 / 43.0 | 1528 s for the arm's 40 runs |
| M12 `fallback`: `book_earliest_flight` | 8 / 20 | 8 / 0 / 0 / 12 | fail | 29.8 / 33.6 / 47.6 | |

The M9 rows are copied unchanged from the M9 section. The v1 rows (retail
agent: `claude-haiku-5-5` 119 / 120, `claude-opus-5-5` 30 / 30) are unchanged too.

What the 80 runs say, against the plan's hypotheses:

- **World-caused endings: 0 / 20 and 3 / 20 under `strict`. "At most 2 of 20
  per scenario" is refuted on `book_earliest_flight`, by one run.** All three
  happened after the booking: qwen searched, booked (both steps served
  `exact`), then called `get_booking` with a booking reference it invented
  (`DL2207_20261105_456789`, `DL2207-ALIMA-20261105`, `DL2207_20261105_ALIMA`).
  The recording has no `get_booking` call at all, so these are unrecorded
  *by tool*, not by identity, and the run ended before the model could answer.
  `fallback` had no run that made an unrecorded call (0 fallback steps in 40
  runs), so these 80 runs cannot say whether those three would have passed
  with the real tool, which rejects an unknown reference with a retry prompt.
- **Arms differ by one pass per scenario: 18 vs 19 and 9 vs 8. "At most 1 in
  20" holds.** The pass rate the gate sees did not depend on the policy here.
  With 20 runs per arm, a one-run difference is far inside sampling noise, so
  this is "no difference measured", not "no difference".
- **`cheapest_fare_in_euros` failed 3 times in its 40 new runs (2 `strict`, 1
  `fallback`). "At least once" holds.** Every failure is `rule:answer_states`
  on all-exact steps: the model got the right tool results and misreported the
  figure. Because the arms differ by one, the plan's pooling rule allows the
  committed 10: **47 / 50** passes (0.94).
- **The identity keys never fired, and nothing was served approximately.**
  0 unrecorded-by-identity endings and 0 approximate steps in 80 runs. M11
  worried that qwen's arguments would differ in case from the recording
  (`"sfo"` against `"SFO"`); in these runs every search, conversion and
  booking matched its recording exactly. The keys cost nothing here, and
  this sample gave them nothing to catch.
- **The model is the main source of failure.** 20 of the 23 failures on
  `book_earliest_flight` across both arms are qwen searching once and
  stopping without booking (`tool_sequence`, `writes_match` and `answer_states`
  fail; 1 exact step, 2 model calls). This is the same mode M9 found in 6 of
  its 10 runs. M9's 2 / 10 is lower than the 17 / 40 pooled here, but at a
  pass rate of 17 / 40 a run of 10 has a 13% chance of 2 or fewer passes
  (binomial, computed with `math.comb`), so the two are compatible.

### The verdict-stability curve

- **Dataset:** every committed `replay flake` report run with the baseline
  gate. `claude-haiku-5-5` (M4, 6 scenarios × 20 runs), `claude-opus-5-5` (M4,
  6 × 5), qwen2.5:7b (M9, 3 × 10) and the four M12 arm rows above (4 × 20).
  `m4_flake_haiku_5runs.json` is left out: it repeats the Haiku suite at 5 runs,
  all 30 passed, and would add six rows identical to Opus's.
- **Method:** the all-runs verdict is the baseline gate's rule, pass rate at
  least `(p_fail + p_pass) / 2` = 0.75, with no interval. For each scenario and
  each subset size n from 1 to runs − 1, 2,000 seeded subsets of n runs are
  drawn without replacement from its recorded outcomes. The curve is the share
  whose verdict agrees with the all-runs verdict. The exact share (a
  hypergeometric count over every subset) sits beside it in the raw output,
  and the sampled shares match it to within 0.02 everywhere. "Stable from" is the
  smallest n from which every larger n agrees in at least 95% of draws.
  A single run's false-pass rate (all-runs verdict fail, one run passes) is the
  observed pass rate, and its false-fail rate is the observed failure rate, by
  construction. They are reported, not estimated. Standard library only,
  `agent_replay/stats.py` untouched; byte-stable (CI re-runs it in `make
  eval`).
- **Raw output:** `eval/results/verdict_curve.json`.

```
$ make eval-verdict-curve
uv run python -m eval.verdict_curve
```

| source | scenario | passes | pass rate − 0.75 | verdict | agree at n=5 | stable from n | single-run false pass | single-run false fail |
|---|---|---|---|---|---|---|---|---|
| claude-haiku-5-5 (M4) | `cancel_pending_laptop` | 20/20 | +0.25 | pass | 100.0% | 1 | 0% | 0% |
| claude-haiku-5-5 (M4) | `modify_address_suite` | 20/20 | +0.25 | pass | 100.0% | 1 | 0% | 0% |
| claude-haiku-5-5 (M4) | `return_damaged_bicycle` | 20/20 | +0.25 | pass | 100.0% | 1 | 0% | 0% |
| claude-haiku-5-5 (M4) | `exchange_waterproof_boots` | 20/20 | +0.25 | pass | 100.0% | 1 | 0% | 0% |
| claude-haiku-5-5 (M4) | `lookup_latest_order_total` | 20/20 | +0.25 | pass | 100.0% | 1 | 0% | 0% |
| claude-haiku-5-5 (M4) | `transfer_swapped_refunds` | 19/20 | +0.20 | pass | 100.0% | 4 | 0% | 5% |
| claude-opus-5-5 (M4) | `cancel_pending_laptop` | 5/5 | +0.25 | pass | — | 1 | 0% | 0% |
| claude-opus-5-5 (M4) | `modify_address_suite` | 5/5 | +0.25 | pass | — | 1 | 0% | 0% |
| claude-opus-5-5 (M4) | `return_damaged_bicycle` | 5/5 | +0.25 | pass | — | 1 | 0% | 0% |
| claude-opus-5-5 (M4) | `exchange_waterproof_boots` | 5/5 | +0.25 | pass | — | 1 | 0% | 0% |
| claude-opus-5-5 (M4) | `lookup_latest_order_total` | 5/5 | +0.25 | pass | — | 1 | 0% | 0% |
| claude-opus-5-5 (M4) | `transfer_swapped_refunds` | 5/5 | +0.25 | pass | — | 1 | 0% | 0% |
| qwen2.5:7b (M9) | `cheapest_fare_in_euros` | 10/10 | +0.25 | pass | 100.0% | 1 | 0% | 0% |
| qwen2.5:7b (M9) | `book_earliest_flight` | 2/10 | -0.55 | fail | 100.0% | 2 | 20% | 0% |
| qwen2.5:7b (M9) | `cancel_refundable_booking` | 0/10 | -0.75 | fail | 100.0% | 1 | 0% | 0% |
| qwen2.5:7b (M12) fallback | `cheapest_fare_in_euros` | 19/20 | +0.20 | pass | 100.0% | 4 | 0% | 5% |
| qwen2.5:7b (M12) fallback | `book_earliest_flight` | 8/20 | -0.35 | fail | 94.2% | 6 | 40% | 0% |
| qwen2.5:7b (M12) strict | `cheapest_fare_in_euros` | 18/20 | +0.15 | pass | 94.3% | 8 | 0% | 10% |
| qwen2.5:7b (M12) strict | `book_earliest_flight` | 9/20 | -0.30 | fail | 90.3% | 6 | 45% | 0% |

(Opus's scenarios have 5 runs, so n stops at 4; "—" means n = 5 is not a
subset.)

**Hypothesis: "five runs agree with the all-runs verdict in at least 95% of
draws on every scenario." It holds on all 15 committed scenarios and is
refuted by three of the four new M12 rows (94.3%, 94.2%, 90.3%).** The plan
predicted this curve would be close to arithmetic, and it is: 13 of the 19
rows pass 0% or 100% of their runs, so every subset agrees. The
distance column decides the rest. The two committed mixed scenarios sit 0.20
and 0.55 from the threshold. The new rows sit 0.15 to 0.35 from it, and those
are the ones five runs cannot settle. `strict`'s `cheapest_fare_in_euros`
(18 / 20, +0.15) needs 8 runs, where a 19 / 20 scenario needs 4.

The curve is not monotone, and the reason is the threshold, not noise. The
sampled and exact shares agree. At n = 1 to 3, a 0.75 rule passes only with
no failures. At n = 4 to 7 it tolerates one, and at n = 8 to 11 it tolerates
two. So a scenario that should pass gets *less* stable from n = 1 to 3
(18 / 20: 0.90, 0.81, 0.72), jumps at n = 4 (0.97), and decays again until
n = 8. A failing scenario does the reverse. A fixed-k gate with k chosen
just after one of those steps is cheaper and more stable than one chosen
before it. This is a property of the fixed-threshold rule and does not carry
over to the SPRT, which `stats.py` will provide.

What the curve does not say: how many runs any agent needs. It says how far
these particular pass rates sit from 0.75. A scenario with a true pass rate
near the threshold needs many more runs than any of these, and the
sequential-gate simulation (`make bench-sprt`) is the tool for that question.
The single-run columns show the cost of gating on one run. A single run of
`book_earliest_flight` would have passed a scenario that fails 55% to 80% of
its runs, 20% to 45% of the time.

**Provenance.** The arms ran from the harness committed in `96f17e5` (only
its formatting changed after the run started), with the pydantic-ai example
unchanged since M11. Runs were one at a time. Apart from a few seconds of
offline tests during the `strict` arm, nothing else ran on the machine.

## Cassette bytes at scale (M13, offline)

The question (PLAN.md, M13): what does a long run cost in cassette bytes, load
and replay time and run-store bytes, and does a standard-library compressor
already recover what a shared-prefix encoding would? The answer decides the
fix, by a rule written into PLAN.md before the run: transparent compressed
cassettes with the best stdlib codec that recovers at least 10× at 100 turns,
and a new cassette schema with the prefix encoding only if it beats that codec
by more than 3×. (PLAN.md calls it "v2", but cassettes are already at schema 2,
so it would have been v3.)

- **Dataset, synthetic:** a fake-model agent (`eval/bench_scale.py`, `run`)
  that calls one tool a turn and answers on its last turn, so `n` turns are
  `n` model calls and `n - 1` tool calls, at 5, 20, 50, 100 and 200 turns. The
  system prompt is the retail agent's length (6,123 characters) and each tool
  result about 250 bytes. Both are seeded, non-repeating text: a seeded
  vocabulary of 4,096 pseudo-words drawn uniformly, about 12 bits a word. A
  repeated filler would flatter every compressor. Recorded with a fixed
  clock, timer and cassette id, so the bytes are byte-stable.
- **Dataset, real:** every committed real recording: the six live
  `claude-opus-5-5` retail cassettes (M2) and the two qwen2.5:7b pydantic-ai
  cassettes (M9).
- **Method:** per cassette, the JSONL bytes and the largest line (one
  envelope); gzip (stdlib `gzip` over zlib 1.3.2, level 9, DEFLATE with a
  32 KB window), bz2 (level 9, 900 KB blocks) and xz (stdlib `lzma` over
  liblzma 5.8.3, preset 6, an 8 MB dictionary); and a shared-prefix encoding
  (`prefix_encode`: each model request stored as the messages it appends to
  the previous model request, plus any changed field), raw and under each
  codec. The encoding is a prototype kept inside `eval/`. Every cassette is
  checked to decode back to the same JSON. The exponent is the least-squares
  slope of log bytes on log turns. The quadratic is an exact least-squares fit
  of bytes = c0 + c1·n + c2·n². Timings are medians of 5. The store
  measurement replays the 20-turn run 100 times and the 100-turn run 10 times
  in `full` into a fresh SQLite store (`Store.save_run`), then reads one run
  back (`Store.get_run`, the `/api/runs/{id}` body).
- **Command:** `make bench-scale` (about a minute). The test suite re-checks
  the raw and prefix-encoded sizes against the committed report
  (`tests/test_bench_scale.py`). Compressed sizes are recorded with the codec
  versions, because another zlib or liblzma build can differ by a few bytes.
  The benchmark's cassettes leave the header's `recorder` versions out, so
  the raw sizes do not depend on the Python or package version (with them,
  Python 3.14.4 against 3.12.16 made every cassette a byte shorter).
- **Machine:** 4 vCPU, AMD EPYC 7763, CPython 3.12.16. $0.
- **Raw output:** `eval/results/bench_scale_sizes.json` (sizes, fit,
  decision) and `eval/results/bench_scale.json` (timings, store).

### Synthetic runs: bytes and what each codec recovers

Each ratio is raw bytes over compressed bytes (compressed size in brackets).
"prefix" is the prefix encoding alone; "prefix+gzip" and "prefix+xz" put the
codec on top of it.

| turns | envelopes | raw bytes | bytes/turn | largest line | gzip | bz2 | xz | prefix | prefix+gzip | prefix+xz |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 9 | 44,325 | 8,865 | 9,447 | 6.93× (6,392) | 4.61× (9,620) | 7.68× (5,772) | 2.93× (15,111) | 7.32× | 7.77× (5,708) |
| 20 | 39 | 265,085 | 13,254 | 18,163 | 19.5× (13,594) | 14.36× (18,464) | 25.15× (10,540) | 6.07× (43,661) | 23.26× | 25.91× (10,232) |
| 50 | 99 | 1,099,202 | 21,984 | 35,611 | 11.13× (98,785) | 19.67× (55,895) | 55.58× (19,776) | 10.9× (100,817) | 49.89× | 58.37× (18,832) |
| 100 | 199 | 3,653,031 | 36,530 | 64,701 | 3.84× (951,007) | 22.23× (164,360) | 104.95× (34,808) | 18.6× (196,421) | 92.08× | 111.97× (32,624) |
| 200 | 399 | 13,133,680 | 65,668 | 123,066 | 3.28× (4,003,926) | 17.88× (734,732) | 206.02× (63,748) | 33.84× (388,150) | 175.38× | 224.15× (58,592) |

**Growth.** The fitted exponent over 5 to 200 turns is 1.539. Between
neighbouring sizes it climbs: 1.29 (5→20), 1.55 (20→50), 1.73 (50→100), 1.85
(100→200). The exact fit is bytes ≈ 451 + 7,395·n + 291.4·n². So bytes are
quadratic, but the linear term (the system prompt and tool list every request
carries) dominates until about 25 turns (7,395 / 291.4). That is why a slope
over the whole range reads 1.5 and not 2. The hypothesis said "an exponent
near 2". It holds for the n² term but not for the slope over this range. The
n² coefficient is half of what each model request adds to the next: requests
grow 581 bytes a turn (the reply and the tool result), and 581 / 2 ≈ 291.
Model envelopes are 94% of the bytes at 5 turns and 98.5% at 100.

**The gzip-window hypothesis: confirmed.** gzip recovers 19.5× at 20 turns,
where the largest envelope is 18 KB. At 50 turns, with envelopes up to 35.6 KB,
it drops to 11.1×. At 100 turns (64.7 KB) it is 3.8× and at 200 turns 3.3×.
DEFLATE can only point back 32 KB. Once one envelope is longer than that, the
previous envelope's copy of the same conversation is out of reach, and gzip
compresses each envelope as if it were new text. The control is the prefix
encoding. With the copies removed before gzip runs, gzip recovers 92× at 100
turns instead of 3.8×. So the collapse is about the distance to the repeat,
not the text. bz2's 900 KB blocks peak at 22× and fall back to 17.9× at 200
turns, when the cassette is 15 blocks long. xz's 8 MB dictionary keeps
finding the repeats: 105× at 100 turns and 206× at 200.

**The decision.** At 100 turns the best codec is xz at 104.95×, which clears
the 10× bar. The prefix encoding under xz is 32,624 bytes against 34,808 for xz
alone, 1.07× smaller. The raw prefix encoding (196,421 bytes) is 5.6× *larger*
than xz. Neither comes near 3×, so the rule chooses transparent `.jsonl.xz`
cassettes and no schema change. That is what was built: a cassette path
ending in `.xz` is read and written as xz (`Cassette.load`, `Cassette.dump`,
so `replay record --out run.jsonl.xz`, `replay run`, `replay ci`, `replay
diff` and the suite files take it as is). The file is plain xz, so the `xz`
tool converts either way. Other compression suffixes (`.gz`, `.bz2`, `.zst`, …)
are refused with the rename that fixes them, because gzip is the codec this
table rules out.

### Real recordings beside the synthetic fit

"growth/turn" is the mean number of bytes each model request adds to the one
before it (the reply and the tool results; it sets the n² term at c2 ≈
growth / 2). "requests at 100 turns" projects this recording's model requests
alone to 100 turns at that growth (100 · first + growth · 100 · 99 / 2). The
same projection from the 5-turn synthetic run gives 3.52 MB. The measured
100-turn run holds 3.60 MB of model envelopes (98.5% of 3.65 MB), so the
projection is close.

| cassette | turns | raw bytes | bytes/turn | largest line | model share | first request | growth/turn | gzip | bz2 | xz | prefix | requests at 100 turns |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| synthetic, 5 turns | 5 | 44,325 | 8,865 | 9,447 | 0.943 | 6,469 | 580 | 6.93× | 4.61× | 7.68× | 2.93× | 3.5 MB |
| `live/cancel_pending_laptop.jsonl` | 5 | 84,880 | 16,976 | 19,044 | 0.934 | 12,543 | 1,369 | 11.27× | 9.18× | 12.85× | 2.92× | 8.0 MB |
| `live/exchange_waterproof_boots.jsonl` | 6 | 126,419 | 21,070 | 27,470 | 0.907 | 12,531 | 2,562 | 11.15× | 9.73× | 13.59× | 2.82× | 13.9 MB |
| `live/lookup_latest_order_total.jsonl` | 4 | 66,814 | 16,704 | 19,700 | 0.922 | 12,405 | 1,596 | 7.94× | 6.87× | 8.85× | 2.37× | 9.1 MB |
| `live/modify_address_suite.jsonl` | 3 | 50,489 | 16,830 | 17,306 | 0.921 | 12,478 | 1,932 | 8.12× | 6.75× | 8.88× | 2.14× | 10.8 MB |
| `live/return_damaged_bicycle.jsonl` | 5 | 92,071 | 18,414 | 21,575 | 0.915 | 12,486 | 2,011 | 10.8× | 8.92× | 12.41× | 2.72× | 11.2 MB |
| `live/transfer_swapped_refunds.jsonl` | 5 | 86,020 | 17,204 | 19,206 | 0.945 | 12,539 | 1,405 | 10.15× | 8.1× | 11.54× | 2.9× | 8.2 MB |
| `qwen/book_earliest_flight.jsonl` | 3 | 13,156 | 4,385 | 4,360 | 0.859 | 2,342 | 618 | 5.45× | 4.64× | 5.54× | 1.7× | 3.3 MB |
| `qwen/cheapest_fare_in_euros.jsonl` | 3 | 12,897 | 4,299 | 4,202 | 0.866 | 2,353 | 556 | 5.46× | 4.66× | 5.56× | 1.73× | 3.0 MB |

What the real traces say about the fit:

- **The shape holds.** Model envelopes are 86% to 95% of every real cassette,
  and each model request carries the one before it plus a fixed increment, as
  in the synthetic run.
- **Real Opus runs grow faster.** The live retail requests start at 12.5 KB
  (a 6.1 KB policy plus a 6.1 KB tool list) against 6.5 KB synthetic. They
  grow 1.4 to 2.6 KB a turn against 0.58 KB, because Opus's replies and the
  retail tool results are longer. Projected to 100 turns, their model
  requests alone come to 8.0 to 13.9 MB, against the synthetic run's
  3.65 MB. The pydantic-ai agent on qwen2.5:7b, with a short prompt and short
  replies, projects to 3.0 to 3.3 MB, close to the synthetic workload. So the
  synthetic table is a lower bound for a retail-sized Opus agent, not a
  typical case.
- **Real runs compress better per turn, not per byte.** At 3 to 6 turns,
  xz recovers 8.9× to 13.6× on the live retail cassettes against 7.68× on
  the 5-turn synthetic run. But the 5-turn live cassettes are also about
  twice the 5-turn synthetic run's bytes (84,880 to 92,071 against 44,325),
  each envelope repeating a 12.5 KB prefix, and the synthetic table shows
  the xz ratio rising with length on its own. So this does not show that
  real text is more compressible byte for byte; it shows that at a given
  number of turns a retail-sized Opus run has more repeated bytes, and
  clears the 10× bar in fewer turns than the synthetic run. These short real runs have no envelope near 32 KB
  (the largest is 27 KB), so they cannot show the gzip collapse themselves.
  At their measured growth, a live retail run would pass 32 KB at about
  turn 8 to 15. (That is an extrapolation from the growth column, not a
  measurement.)
- The prefix encoding alone recovers 1.7× to 2.9× on the real cassettes,
  against 5.5× to 13.6× for xz.

### Timings (machine-dependent)

Seconds, medians of 5 (recording: one run). "load" is `Cassette.loads` on the
JSONL, or decompress then `loads`, which is what `Cassette.load` does for a
`.jsonl.xz` path.

| turns | record | load JSONL | load gzip | load bz2 | load xz | compress xz | compress bz2 | `full` replay |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.006 | 0.0002 | 0.0003 | 0.0011 | 0.0005 | 0.0033 | 0.0047 | 0.004 |
| 20 | 0.036 | 0.0013 | 0.0016 | 0.0064 | 0.0018 | 0.0144 | 0.0563 | 0.024 |
| 50 | 0.157 | 0.0062 | 0.0076 | 0.0318 | 0.0080 | 0.0648 | 0.2225 | 0.100 |
| 100 | 0.673 | 0.0325 | 0.0429 | 0.1200 | 0.0360 | 0.2423 | 0.8317 | 0.456 |
| 200 | 2.537 | 0.2903 | 0.3066 | 0.5501 | 0.2850 | 1.0493 | 3.0614 | 1.660 |

- `full` replay at 200 turns takes 1.7 s, under the 3 s the hypothesis allowed.
- Compression barely changes load time. xz adds 0.3 to 3.5 ms up to 100
  turns, and at 200 turns it measured 5 ms *faster*, which is noise. Parsing
  13 MB of JSON costs far more than decompressing 64 KB. Loading is still
  superlinear in turns, about ×9 from 100 to 200. So compression saves disk
  and git, not parse time. The hypothesis held.
  **Corrected in M14:** the parsing explanation was mostly wrong. Most of
  the superlinear load came from the garbage collector's full collections.
  The parse alone is close to linear but not quite: it rises from about 5 to
  7 ms per MB. With the collector paused, load from 100 to 400 turns is 1.22×
  linear in bytes on the first run and 1.33× on a re-run; with the collector
  on, it is 1.84× and 2.11×. The table above is kept as measured. See "Long recordings and a load linear in bytes
  (M14)" below.
- Writing xz costs 0.24 s at 100 turns and 1.05 s at 200, once per recording.
  bz2 is the slowest to write and to read, 3.1 s and 0.55 s at 200 turns, and
  recovers the least.

### The run store (SQLite, not changed by this milestone)

| load | cassette bytes | DB bytes | DB bytes per run | step JSON share | `save_run` median | `get_run` payload | `get_run` median |
|---|---|---|---|---|---|---|---|
| 100 runs × 20 turns | 265,085 | 28,209,152 | 282,092 | 0.914 | 0.006 s | 267,622 | 0.004 s |
| 10 runs × 100 turns | 3,653,031 | 38,428,672 | 3,842,867 | 0.967 | 0.042 s | 3,665,429 | 0.232 s |

The store keeps every step's request and response in full, so a stored run
costs about what its cassette costs uncompressed: 282 KB a 20-turn run, 3.8 MB
a 100-turn run, 91% to 97% of it step JSON. The dashboard's run page
downloads 3.7 MB for one 100-turn run, and reading it back took 0.23 s in the
committed run. That timing is noisy: an earlier run of the same command gave
0.067 s. Compressed cassettes do not touch this.
The store fix PLAN.md deferred (not copying an `exact` step's request, which
the cassette already holds) is now the next storage problem, with this table
as its baseline. DB file sizes vary by a few KB between runs (random run ids),
so they are not byte-checked.

### Across upstream releases of pydantic-ai

The committed pydantic-ai cassettes were recorded on pydantic-ai-slim 2.61.0.
`make bench-upstream` installs each release into a throwaway venv and replays
both suites in `full` (needs network for PyPI; raw output
`eval/results/upstream_pydantic_ai.json`):

| pydantic-ai-slim | `replay.toml` (fake model) | `replay-qwen.toml` (qwen2.5:7b) |
|---|---|---|
| 2.58.0 | 3/3 pass, 0 of 15 steps approximate | 3/3 pass, 0 of 15 steps approximate |
| 2.59.0 | 3/3 pass, 0 of 15 approximate | 3/3 pass, 0 of 15 approximate |
| 2.60.0 | 3/3 pass, 0 of 15 approximate | 3/3 pass, 0 of 15 approximate |
| 2.61.0 (recorded) | 3/3 pass, 0 of 15 approximate | 3/3 pass, 0 of 15 approximate |
| 2.62.0 | 3/3 pass, 0 of 15 approximate | 3/3 pass, 0 of 15 approximate |

Across these five releases the framework sent byte-identical requests for
this agent, so a cassette survives the upgrade, and the matrix has no
regression to catch. This is the evolution-round measurement, reproduced by
a committed command.

## Long recordings and a load linear in bytes (M14, offline)

Two questions (PLAN.md, M14). Can a long recording survive its process being
killed? And is load time bounded by parsing or by the collector? Hypotheses,
written before the run:
- appending each envelope to `<out>.partial` costs under 1% of the median
  recorded model latency;
- the finished file is byte-identical to `Cassette.dump`;
- a recording killed with SIGKILL after envelope k replays those k in `full`
  and diffs against a finished recording;
- with the collector paused inside `Cassette.loads`, load is within 1.3× of
  linear in bytes from 100 to 400 turns;
- `gc.freeze()` after load cuts `full` replay at 200 turns by at least 20%
  (reported, not gated).

- **Built:** `CassetteWriter` (`agent_replay/cassette.py`). Through
  `RecordingSession(out=...)`, `record(..., out=...)` and `replay record`, it
  writes the header and each envelope to `<out>.partial` as it is recorded,
  flushed per line. `finish()` appends the footer and renames the file. For
  `.xz` it streams the partial through one `lzma.LZMACompressor` first.
  `Cassette.load` reads a footer-less `.partial` as interrupted. That is a
  load-time state: the footer schema is unchanged, and a torn last line is
  dropped and counted. `replay ci` refuses an interrupted cassette and names
  the re-record command. `replay run` warns and replays it. `replay record`
  keeps the partial on Ctrl-C and will not overwrite it without `--force`.
  `Cassette.loads` pauses the cyclic collector only while it parses. The
  pause is shared by nested and concurrent loads, and a collector that was
  already off stays off.
- **Datasets:** the M13 synthetic runs (`eval/bench_scale.py`) plus a new
  400-turn row (799 envelopes, 49.6 MB), and the eight committed real
  recordings. For the append overhead, `make bench-replay`'s four boundaries
  (D5) and the six live `claude-opus-5-5` retail cassettes' 28 model calls.
- **Commands:** `make bench-scale` (about 3.5 minutes; sizes byte-stable and
  re-checked in CI; timings to `.replay/`), `make bench-replay` (about 1.5
  minutes), and the kill test in `tests/test_partial_cassettes.py`. A second
  `make bench-scale BENCH_SCALE_OUT=eval/results/m14_bench_scale_rerun.json`
  on the same machine type checks the load timings. $0.
- **Machine:** 4 vCPU, Intel Xeon 6973P-C, CPython 3.12.16. This is a
  different CPU from the M7 and M13 timings, so every comparison below is
  within this run, never against the older tables.
- **Raw output:** `eval/results/bench_scale_sizes.json` (the new 400-turn row
  and `xz_streaming`; its `fit` now spans 5 to 400 turns: exponent 1.607,
  bytes ≈ 2,004 + 7,325·n + 291.7·n², against M13's 1.539 over 5 to 200);
  `eval/results/m14_bench_scale.json` (memory, collector, freeze, linearity,
  real loads), `eval/results/m14_bench_scale_rerun.json` (the re-run);
  `eval/results/m14_bench_replay.json` (append overhead). The
  committed M7 `bench_replay.json` and M13 `bench_scale.json` are left as they
  were.

### A killed recording keeps what it saw

`tests/test_partial_cassettes.py::test_a_recording_killed_after_envelope_k_replays_those_k`
records the 8-turn fake-model run in a subprocess. The subprocess SIGKILLs
itself inside model call m, for m = 0, 3 and 7, so after k = 0, 6 and 14
envelopes. It runs for both writers (`.jsonl` and `.jsonl.xz`), six cases in
all, and every case passes:
- the process exits by signal 9, and no finished cassette exists;
- the `.partial` loads as interrupted, with exactly k envelopes and no torn
  bytes;
- they equal the first k envelopes of a finished recording of the same run,
  latency aside;
- a `full` replay serves all k `exact`, then fails `unrecorded` on call k + 1;
- `replay diff` against the finished recording aligns k rows as `same` and
  the rest as `added`.

`test_the_finished_file_is_byte_identical_to_dump` checks the byte-identity
hypothesis for both writers. Streamed through one compressor, the `.xz` file
is the same bytes as `lzma.compress` of the whole JSONL, so chunking does not
change the output.

**Why the partial is plain JSONL even for `.xz`.** Python's `lzma` has no
sync flush, so a compressor holds back what it has not emitted. The sizes
report feeds the 100-turn run's 201 lines to one `LZMACompressor` and counts
the envelopes that the bytes emitted so far decode to (`xz_streaming`):

| lines fed | envelopes decodable from the emitted bytes |
|---|---|
| 1 to 147 | 0 |
| 148 to 201 (before `flush()`) | 146 |

A recorder that compressed as it went and died before line 148 would have
left nothing. Even at the end, 53 envelopes are still held back until
`flush()`. The 20-turn run (265 KB) emits nothing decodable before `flush()`
at all (a unit test checks this).

### Append overhead per envelope

Microseconds. Each boundary was timed 2,000 times in sessions of 20 calls
that keep envelopes in memory, and in sessions that also append them to a
partial. Calls are paired by index, with the two session kinds alternating
which goes first.

| boundary | in memory median | appending median | append overhead median | p95 | `finish()` in memory | `finish()` appending |
|---|---|---|---|---|---|---|
| model (12.5 KB request) | 652.73 | 697.04 | 44.25 | 103.53 | 10.04 | 52.46 |
| tool | 65.66 | 73.42 | 7.78 | 29.68 | 8.40 | 37.85 |
| clock | 23.55 | 28.96 | 5.45 | 15.05 | 6.85 | 29.52 |
| random | 19.39 | 24.16 | 4.85 | 10.90 | 5.83 | 25.19 |

**Hypothesis held, by a wide margin.** The worst boundary costs 44 µs per
envelope. The median recorded latency of the 28 live Opus model calls is
1,775 ms, so the append is 0.0025% of a model call, against the 1% bound.
Most of the cost is serialising the envelope, which an in-memory recording
also pays, once, at `dump`. The append only moves that cost into the call.
That cost grows with the envelope. The model envelope measured here is a
first turn (12.5 KB), while a 100-turn envelope is 65 KB. That size is not
measured; at five times the serialising cost it would still be about 0.01%.
Finishing an appending session costs 19 to 42 µs more, depending on the
boundary (the last line and a rename).

### Load: the collector, not the parser

Each size runs in a fresh interpreter. Loads alternate paused
(`Cassette.loads`, as shipped) and unpaused (`Cassette._loads`, the parse as
before M14); medians of 5. "full" counts the gen2
collections the collector started during one unpaused load. "peak RSS" is
`VmHWM` after one load and one `full` replay; the interpreter alone holds
115 MB after imports, and "RSS / bytes" is the peak above that.

| turns | bytes | load paused | load collector on | ms/MB paused | ms/MB on | collections gen0 / gen1 / full | peak RSS | RSS / bytes |
|---|---|---|---|---|---|---|---|---|
| 5 | 44,325 | 0.0003 s | 0.0003 s | 7.10 | 7.10 | 0 / 0 / 0 | 116.6 MB | — |
| 20 | 265,085 | 0.0013 s | 0.0013 s | 5.14 | 5.14 | 2 / 0 / 0 | 118.5 MB | 13.8× |
| 50 | 1,099,202 | 0.0056 s | 0.0057 s | 5.34 | 5.44 | 13 / 1 / 0 | 127.0 MB | 11.4× |
| 100 | 3,653,031 | 0.0201 s | 0.0200 s | 5.77 | 5.74 | 44 / 3 / 0 | 155.6 MB | 11.7× |
| 200 | 13,133,680 | 0.0763 s | 0.1456 s | 6.09 | 11.62 | 134 / 12 / 1 | 267.1 MB | 12.1× |
| 400 | 49,605,749 | 0.3328 s | 0.4988 s | 7.03 | 10.54 | 318 / 28 / 2 | 704.8 MB | 12.5× |

(At 5 turns the ratio is lost in page-granularity noise and left out.)

- **Linear in bytes: held on the first run, not on the re-run.** From 100 to
  400 turns the bytes grow 13.58×. On the run in the table, paused load grows
  16.56×, 1.22× linear (bound 1.3), and collector-on load grows 24.94×, 1.84×
  linear (`load_linearity`). The re-run (`m14_bench_scale_rerun.json`, same
  CPU model) gives 0.0160 s and 0.2888 s: 18.05×, **1.33× linear, outside the
  bound**, with the collector on at 2.11×. The first run's 100-turn load was
  about 25% slower than the re-run's (0.0201 against 0.0160 s), which is
  where its lower ratio comes from. So the pause removes most of the excess
  (2.1× down to 1.3×), but the parse itself is still slightly superlinear:
  its ms per MB rises by about a third from 100 to 400 turns. The hypothesis
  is not robustly met.
- **Where the collector's cost comes from.** Young-generation collections are
  cheap: the 44 at 100 turns cost nothing measurable. A full collection walks
  every live object, and one fires only once enough survivors have piled up.
  No load of 100 turns or fewer triggers one. At 200 turns one fires and
  doubles the load (0.076 s against 0.146 s); at 400 turns two fire.
- **The busier the process, the bigger the cost.** M13 timed load in a
  process holding every other cassette. The same in-process column now
  (`runs[].load_s`, the main process with all six cassettes and the store
  runs alive, one timing each above 200 turns, not a median) gives 0.458 s
  paused against 1.183 s on at 400 turns (2.6×), and 0.371 against 0.928 s on
  the re-run. That paused load is 1.53× linear against its own 100-turn row
  (0.022 s), so in a busy process the pause helps more and leaves more.
  At 200 turns it gives 0.101 against 0.105: no full collection happened to
  fire. The round-3 pilot (1.10 against 0.27 s at 400 turns) ran in the same
  kind of process. So the collector's cost is lumpy and depends on what else
  is alive. That is why the tables disagree on the size of the gain, and why
  the fix is a pause rather than a tuned threshold.
- **Small cassettes gain nothing.** On the committed real recordings (13 KB
  to 126 KB, medians of 21), paused load is 0.08 to 0.46 ms, 3.2 to 6.4 ms
  per MB. That is 5 to 40 µs *slower* than unpaused, the cost of the lock
  and the two `gc` calls, and far below any decision.
- **Memory: about 12× the cassette's bytes.** A loaded and replayed cassette
  costs 11.4× to 13.8× its JSONL size in resident memory above the
  interpreter: 590 MB for the 50 MB cassette. Pydantic models and Python
  strings take several times their UTF-8 bytes, and replay holds the
  requests it fingerprints. So `replay ci --concurrency` over long cassettes
  is bounded by memory first. This is measured and reported, not changed.

### Replay with `gc.freeze()` after load

`gc.freeze()` moves every object alive at that moment into a generation the
collector never scans, so collections during the replay no longer walk the
cassette. Fresh process, interleaved with an ordinary replay of the same
cassette, medians of 5:

| turns | `full` replay | with `gc.freeze()` after load | gain |
|---|---|---|---|
| 50 | 0.0647 s | 0.0652 s | −0.8% |
| 100 | 0.2744 s | 0.2101 s | 23.4% |
| 200 | 0.9543 s | 0.7880 s | 17.4% |
| 400 | 4.5994 s | 3.7496 s | 18.5% |

**Hypothesis not met at 200 turns:** 17.4%, against the 20% predicted (the
round-3 pilot measured 25% in a busier process); the re-run gives 13.5%. The gain is real from 100
turns on, but it stays a measurement, not a product change. Freezing is
process-wide, and it would also keep the agent's own long-lived objects out
of every later collection. A caller that replays one long cassette per
process can use it. `full` replay itself grows faster than the bytes:
4.6 s at 400 turns is 17× the time at 100 turns, for 13.6× the bytes. Every
served request is fingerprinted, and the requests grow with the turns; that
is outside this milestone.

## Does `model-live` survive the change it exists to test? (M15)

`model-live` exists to answer one question before a merge: the model, prompt
or sampling changed, so does the agent still work? This milestone replays the
qwen2.5:7b recordings with that change applied to the live side and asks two
things. When the world was frozen by another model or prompt, how often does
the frozen world end the run rather than the model's own failure? And does
the gate still report the changed model's pass rate?

- **Dataset:** the two pydantic-ai scenarios qwen2.5:7b recorded
  (`cheapest_fare_in_euros`, `book_earliest_flight`,
  `examples/pydantic_ai_agent/replay-qwen.toml`, M11 `identify` keys on),
  the same cassettes as M12.
- **Arms** (`eval/pydantic_ai.py`, `SWAP_ARMS`), each `model-live`, one run at
  a time:
  - qwen2.5:3b (digest `357c53fb…`), recording prompt, under `strict`, then
    under `fallback`, 20 runs per scenario each.
  - qwen2.5:7b (digest `845dbda0…`, the recorder, as in M12) with the
    recording prompt minus its guard line ("Give a tool only values from the
    traveller's message or from earlier tool results: when a call needs
    another call's result, wait for that result."), `strict`, 20 runs each.
  - The control: qwen2.5:7b, recording prompt, `strict`, 20 runs each, the
    same day and machine as the other arms.
  - `book_earliest_flight` at `temperature=0`, qwen2.5:7b, `strict`, 10 runs,
    after four probe calls (below).
  Every other arm leaves sampling at the model's default, as the cassettes
  were recorded. The live model comes from `REPLAY_OLLAMA_MODEL` and the
  prompt from `run_no_guard` / `run_temperature_0` in
  `examples/pydantic_ai_agent/agent.py`. Each run row records the model,
  temperature and guard line its requests actually carried; all 170 runs
  carried what their arm asked for.
- **Causes** as in M12 (`cause`): passed; unrecorded (a call no unused
  recording matched, by tool or by `identify` key, ended a `strict` run);
  approximate; model (a check failed, or the agent raised, with every served
  step exact). Identity endings are counted separately. The gate is the
  committed reports' rule, a pass rate of at least 0.75.
- **Machine:** 4 vCPU, AMD EPYC 7763, CPU only, Ollama 0.40.11. $0.
- **Raw output:** `eval/results/m15_swap_qwen.json` (every run's cause, tool
  calls with their status, answer, step counts, error and seconds; the
  probe; the diff below as `m15_swap_qwen.diff.md`).

```
$ make eval-pydantic-ai-swap
uv run python -m eval.pydantic_ai --swap
  ... (the six full-mode replays, all PASS, 0 approximate)
arm 3b_strict: 20 model-live runs of cheapest_fare_in_euros, book_earliest_flight (qwen2.5:3b, recording, temperature None, strict)
  cheapest_fare_in_euros: 0/20 passed (12 unrecorded, 8 model); 12 by identity, 3 tool paths
  book_earliest_flight: 0/20 passed (20 unrecorded); 20 by identity, 1 tool paths
  wall time 585.0 s
arm 3b_fallback: 20 model-live runs of cheapest_fare_in_euros, book_earliest_flight (qwen2.5:3b, recording, temperature None, fallback)
  cheapest_fare_in_euros: 0/20 passed (20 model); 0 by identity, 3 tool paths
  book_earliest_flight: 0/20 passed (20 model); 0 by identity, 2 tool paths
  wall time 737.9 s
arm 7b_no_guard_strict: 20 model-live runs of cheapest_fare_in_euros, book_earliest_flight (qwen2.5:7b, no_guard, temperature None, strict)
  cheapest_fare_in_euros: 2/20 passed (2 passed, 18 unrecorded); 18 by identity, 6 tool paths
  book_earliest_flight: 2/20 passed (2 passed, 5 unrecorded, 13 model); 5 by identity, 4 tool paths
  wall time 1169.9 s
arm 7b_control_strict: 20 model-live runs of cheapest_fare_in_euros, book_earliest_flight (qwen2.5:7b, recording, temperature None, strict)
  cheapest_fare_in_euros: 18/20 passed (18 passed, 2 model); 0 by identity, 1 tool paths
  book_earliest_flight: 10/20 passed (10 passed, 3 unrecorded, 7 model); 2 by identity, 4 tool paths
  wall time 1802.4 s
arm 7b_temperature_0_strict: 10 model-live runs of book_earliest_flight (qwen2.5:7b, recording, temperature 0.0, strict)
  book_earliest_flight: 10/10 passed (10 passed); 0 by identity, 1 tool paths
  wall time 482.6 s
wrote eval/results/m15_swap_qwen.json
```

| `model-live` on the qwen2.5:7b cassettes | Passed | passed / unrecorded / approximate / model | By identity | Gate | Seconds per run (min / median / max) | Wall |
|---|---|---|---|---|---|---|
| M9: `cheapest_fare_in_euros` (7b, no keys, 10 runs) | 10 / 10 | (not classified in M9) | — | pass | 41.3 / 47.3 / 66.5 | 1026 s for M9's 30 runs |
| M9: `book_earliest_flight` (7b, no keys, 10 runs) | 2 / 10 | (1 `UnrecordedCallError`, 1 approximate) | — | fail | 31.9 / 33.6 / 55.6 | |
| M12 `strict`: `cheapest_fare_in_euros` (7b) | 18 / 20 | 18 / 0 / 0 / 2 | 0 | pass | 40.1 / 40.4 / 54.7 | 1562 s |
| M12 `strict`: `book_earliest_flight` (7b) | 9 / 20 | 9 / 3 / 0 / 8 | 0 | fail | 30.3 / 36.9 / 45.8 | |
| M12 `fallback`: `cheapest_fare_in_euros` (7b) | 19 / 20 | 19 / 0 / 0 / 1 | 0 | pass | 40.0 / 40.3 / 43.0 | 1528 s |
| M12 `fallback`: `book_earliest_flight` (7b) | 8 / 20 | 8 / 0 / 0 / 12 | 0 | fail | 29.8 / 33.6 / 47.6 | |
| M15 3b `strict`: `cheapest_fare_in_euros` | 0 / 20 | 0 / 12 / 0 / 8 | 12 | fail | 11.4 / 14.3 / 31.1 | 585 s |
| M15 3b `strict`: `book_earliest_flight` | 0 / 20 | 0 / 20 / 0 / 0 | 20 | fail | 13.3 / 13.6 / 15.8 | |
| M15 3b `fallback`: `cheapest_fare_in_euros` | 0 / 20 | 0 / 0 / 0 / 20 | 0 | fail | 11.1 / 19.3 / 20.0 | 738 s |
| M15 3b `fallback`: `book_earliest_flight` | 0 / 20 | 0 / 0 / 0 / 20 | 0 | fail | 12.2 / 19.9 / 22.7 | |
| M15 7b no guard, `strict`: `cheapest_fare_in_euros` | 2 / 20 | 2 / 18 / 0 / 0 | 18 | fail | 14.9 / 15.5 / 86.6 | 1170 s |
| M15 7b no guard, `strict`: `book_earliest_flight` | 2 / 20 | 2 / 5 / 0 / 13 | 5 | fail | 32.2 / 35.8 / 55.4 | |
| M15 7b control, `strict`: `cheapest_fare_in_euros` | 18 / 20 | 18 / 0 / 0 / 2 | 0 | pass | 44.8 / 45.8 / 82.2 | 1802 s |
| M15 7b control, `strict`: `book_earliest_flight` | 10 / 20 | 10 / 3 / 0 / 7 | 2 | fail | 33.7 / 43.7 / 49.9 | |
| M15 7b `temperature=0`, `strict`: `book_earliest_flight` (10 runs) | 10 / 10 | 10 / 0 / 0 / 0 | 0 | pass | 47.9 / 48.3 / 48.6 | 483 s |

The M9 and M12 rows are copied unchanged from their sections, and the v1
rows (retail agent: `claude-haiku-5-5` 119 / 120, `claude-opus-5-5` 30 / 30)
are unchanged too. 170 runs, 4778 s of arms on 4 vCPU, 0 approximate steps
in any of them. Apart from a few seconds of unit tests during the no-guard
arm, nothing else ran on the machine during the arms.

**Temperature probe.** Before the arms, `probe_temperature` sent one
open-ended prompt ("Invent a name for a new travel agency and give it a
one-line slogan.") to Ollama's `/v1/messages` on qwen2.5:7b, twice at
`temperature: 0` and twice at `temperature: 1`. Both temperature-0 answers
were identical ("Travel Wonders Agency: Discover the Magic of Every
Destination!"); the two at 1 differed from each other and from it
(`identical_at_0: true`, `identical_at_1: false` in the raw report). So the
endpoint takes the parameter. One plumbing detail mattered: pydantic-ai sends
`temperature` inside `extra_body`, because `anthropic` 1.x has no named
argument for it, and the SDK merges it into the JSON body. The harness reads
it from there when it checks what reached the wire.

What the 170 runs say, against the plan's hypotheses:

- **(a) qwen2.5:3b fails the gate on both scenarios (0 / 20 and 0 / 20, at
  most 14 was the bar), and `fallback` is within 2 of `strict` (0 against 0
  on each). Holds. "Its world-caused endings under `strict` exceed 7b's 3 of
  40" holds too: 32 of 40. But none of the 32 is the world's fault.** Every
  one is an identity ending on a wrong argument:
  - the booking: 20 of 20 runs book AS330 (09:30, the cheapest) instead of
    DL2207 (06:10, the earliest), one identical tool path;
  - the fare: 12 runs convert 212 (AA178, sold out) or 243 (UA512) instead of
    189 (B6416).
  The `fallback` arm is the counterfactual. With the real tool behind those
  calls, the same mistakes run to the end and fail the ground-truth rules:
  18 of 20 booking runs book AS330 and fail `writes_match`, and 13 of 20
  fare runs convert 212 and state the wrong euros. So the `strict`
  classification called 32 endings "unrecorded" that were the model's.
  That is the right verdict for the wrong reason. An identity ending means
  "the model asked for something the recording does not have", which is a
  model error whenever the recording holds the right answer. The `fallback`
  pass count, not the `strict` cause count, says whose failure it was.
  In the other 8 `strict` fare runs no tool ran at all: pydantic-ai's
  argument validation rejected the search call, and 7 runs then answered
  that origin, destination or date were missing, while 1 exceeded the
  tool's retry limit (`UnexpectedModelBehavior`). The diff below shows
  why for run 0: its first reply wrapped the arguments in
  `{"object": {...}}`. Two `fallback` fare runs converted the
  right fare (173.88 EUR) and still failed `answer_states`, because the
  answer never named the flight.
- **(b) Without the guard line, at least one run ends `unrecorded` or by
  identity. Holds, by a wide margin: 23 of 40 (18 fare, 5 booking), all by
  identity.** The first was fare run 2, `convert_currency` with
  `amount: 123.45`. These are the first identity endings ever on this prompt's
  model: M12 had 0 in 80 runs, and the M11 keys had only fired on seeded
  mutations before. The fare gate fails, 2 / 20 against the control's
  18 / 20. Fisher's exact test (one-sided, hypergeometric, `math.comb`)
  puts the chance of a split this lopsided between two equal arms at
  2.6 × 10⁻⁷ for the fare and 0.0069 for the booking (2 / 20 against
  10 / 20), so the prompt change caused both drops. (Treating the control's
  18 / 20 as the true rate, as the build first did, gives 1.6 × 10⁻¹⁶, but
  ignores that the control is itself a 20-run sample.)
  The converted amounts (350, 150, 123.45, 125, 250) appear nowhere in the
  search result (243, 189, 212). The mechanism is the one the guard was
  written for. A 10-run re-run with the harness's `parallel_tool_replies`
  count (added after the main run; below) shows all 8 of its fare identity
  endings came from a single reply that asked for `search_flights` and
  `convert_currency` together, the amount guessed before the search came
  back; both passing runs made one call per reply. The booking's no-guard
  failures are different: 13 runs search and stop (the M9/M12 mode), and 5
  book WN1849 or AS330.
- **(c) The control is inside both intervals: 18 / 20 (16 to 20) and
  10 / 20 (4 to 13). Holds.** It matches M12's `strict` arm almost exactly
  (18 and 9). It also produced the first identity endings on the shipped
  prompt: 2 booking runs chose AS330, the same mistake 3b makes every time.
  Its third unrecorded ending (run 6) was different: it searched, booked
  DL2207 correctly, then called `convert_currency` (158 USD to EUR), a tool
  the booking recording never called (by tool, not by identity), and
  `strict` stopped it before it answered. The write was already right, so
  this is the world's ending in M12's sense: a path the recording lacks, not
  a wrong argument. Whether it would have passed is unknown (it never
  answered). The swap arms' differences from the control are the change's
  effect, not a different day.
- **(d) At `temperature=0` the 10 trajectories are identical: one tool path
  (search, then DL2207), one answer word for word, all 10 pass. Holds.** That
  is the problem. One deterministic run says "pass" for a booking whose
  sampled pass rate is 17 / 40 in M12 and 10 / 20 in the control, which the
  fixed 0.75 rule fails. Greedy decoding picks one
  trajectory from the distribution, and here it happens to be a passing one.
  Ten repeats of that one trajectory measure nothing about the 23 of 40
  M12 runs (and 10 of 20 control runs) that failed.
  The agent ships at default sampling, so a temperature-0 CI run tests a
  different agent from the one users get.

**What this means for `model-live`.** It survived the change it exists to
test. With the model or prompt swapped, the frozen world served every step
the new model asked for that the recording had (0 approximate in 170 runs).
It stopped every run that asked for something else, and in this sample
those requests were wrong in every swap arm: each of the 55 identity endings
named a flight or amount the ground truth rules out. The gate's verdict was the model's in
every arm: 3b fails under both policies, the guardless prompt fails, and
the control is unchanged. Under `strict`, the cause column overstates the
world's share whenever the model guesses an argument. Read it beside a
`fallback` arm, or the pass counts, before blaming the recording. The
limit is still the one M12 found: a different *correct* path ends the same
way. The control had one (run 6, a correct booking followed by an extra
conversion), so the cause column can understate the world's share as well
as overstate it; a single `strict` arm cannot tell the two apart.

### A smaller re-run, in a later server session

Two ad hoc arms and a fresh probe on the same machine, after the
main run, to see whether the arms reproduce (raw output in `/tmp`, not
committed: too few runs to report beside the main table):

```
$ uv run python -m eval.pydantic_ai --arms strict,fallback --model qwen2.5:3b --runs 3 --arms-out /tmp/rv/3b.json
  cheapest_fare_in_euros: 0/3 passed (3 unrecorded); 3 by identity, 1 tool paths        (strict)
  book_earliest_flight: 0/3 passed (2 unrecorded, 1 model); 2 by identity, 2 tool paths (strict)
  cheapest_fare_in_euros: 0/3 passed (3 model); 0 by identity, 2 tool paths             (fallback)
  book_earliest_flight: 0/3 passed (3 model); 0 by identity, 2 tool paths               (fallback)
$ uv run python -m eval.pydantic_ai --arms strict --temperature 0 --runs 3 --arms-out /tmp/rv/t0.json
  cheapest_fare_in_euros: 3/3 passed (3 passed); 0 by identity, 1 tool paths
  book_earliest_flight: 3/3 passed (3 passed); 0 by identity, 1 tool paths
probe_temperature(): identical_at_0 true, identical_at_1 false
```

3b fails the same way, and temperature 0 again gives one tool path per
scenario. But the probe's temperature-0 answer this time was "Travel Wonders
Agency: Discover the World, Dream B…", not the main run's "…Discover the
Magic of Every Destination!". Greedy decoding was repeatable within one
server session and not across sessions, so "temperature 0 makes CI
deterministic" holds per session at best: the one trajectory it pins can
change when the server restarts.

### No guard, with replies that ask for several tools

```
$ uv run python -m eval.pydantic_ai --arms strict --prompt-variant no_guard --runs 10 --arms-out eval/results/m15_no_guard_parallel.json
arm qwen2.5:7b_no_guard_tNone_strict: 10 model-live runs of cheapest_fare_in_euros, book_earliest_flight (qwen2.5:7b, no_guard, temperature None, strict)
  cheapest_fare_in_euros: 2/10 passed (2 passed, 8 unrecorded); 8 by identity, 6 tool paths
  book_earliest_flight: 0/10 passed (2 unrecorded, 8 model); 2 by identity, 2 tool paths
  wall time 646.1 s
```

Raw output: `eval/results/m15_no_guard_parallel.json`. The 8 fare identity
endings converted 350, 450, 150, 150, 1, 1, 350 and 100, each from a reply
with two tool calls (`parallel_tool_replies: 1`); the 2 passes converted 189
with one call per reply (`0`). No booking run made a parallel reply; its 2
identity endings booked WN1849 after reading the search. This run replicates
the arm (2 / 10 and 0 / 10 against 2 / 20 and 2 / 20). Its wall time is not
comparable: the offline test suite ran beside it.

### `replay diff` of a failed qwen2.5:3b run

The harness diffs the first failed 3b `strict` run against the cassette it
replayed, with the same alignment and table `replay diff --markdown` prints
(`agent_replay/diff.py`), into `eval/results/m15_swap_qwen.diff.md`:

## Trace diff: `examples/pydantic_ai_agent/cassettes/qwen/cheapest_fare_in_euros.jsonl` → `qwen2.5:3b run 0`

distance 4.188 (0 same, 2 changed, 0 added, 3 removed); cost unpriced → unpriced; latency 55759 ms → 31117 ms.

| Op | Base | Head | Cost | Changes |
|---|---|---|---|---|
| changed | `0 model anthropic.beta.messages.create` | `0 model anthropic.beta.messages.create` | 0.562 | `request.model: "qwen2.5:7b" → "qwen2.5:3b"`<br>`response: {"content":[{"id":"call_7hce5hca","input":{"date":"2026-11-… → {"content":[{"id":"call_93sy9455","input":{"object":{"date"…` |
| removed | `1 tool search_flights` | — | 1 | — |
| changed | `2 model anthropic.beta.messages.create` | `1 model anthropic.beta.messages.create` | 0.625 | `request.messages[-1]: {"content":[{"content":[{"text":"{\"flight_id\":\"UA512\",\… → {"content":[{"content":[{"text":"[{\"type\":\"missing\",\"l…`<br>`request.model: "qwen2.5:7b" → "qwen2.5:3b"`<br>`response: {"content":[{"id":"call_qv17br9g","input":{"amount":189,"fr… → {"content":[{"text":"The provided details for the flight se…` |
| removed | `3 tool convert_currency` | — | 1 | — |
| removed | `4 model anthropic.beta.messages.create` | — | 1 | — |

Read top to bottom: 3b's first reply put the search arguments under an
`object` key, so no `search_flights` ran. Its second request carried
pydantic-ai's validation error (`"type":"missing"`) where 7b's carried the
flight list, and 3b answered in text instead of retrying. The run made
neither tool call the recording has.

## Does frozen-world divergence compound on a long chain? (M16)

M12 measured the frozen world on short runs: 3 of 143 served tool calls left
the recorded path (`m12_arms_qwen.json`), all three an invented `get_booking`
on the booking task. If that 2.1% applies independently to every call, a run
of 9 served calls leaves the path 17% of the time, and if it clusters (one
miss puts the model off course for the next) long runs are worse still. This
milestone builds a task long enough to tell and runs it on qwen2.5:7b.

- **Task:** `trace_rebooking_chain` in the pydantic-ai travel desk. The fixed
  world gains an eight-booking chain for "Lee Park" (`CHAIN` in
  `examples/pydantic_ai_agent/tools.py`); each booking names the one it was
  rebooked from (`rebooked_from`), the original has none. The traveller gives
  the newest reference, K7W2QD, and asks for the original's reference and the
  total paid across the chain in euros: $200.00 plus seven $25.00 change fees
  is $375.00, 345.00 EUR. The shortest path is 8 `get_booking` calls, each on
  a reference only the previous result holds (so parallel calls cannot
  shorten it), then one `convert_currency`: 9 served tool calls, 10 model
  turns. The chain bookings are a `RebookedBooking` subclass, so the two
  older bookings serialise exactly as before: every committed cassette still
  replays with 0 approximate steps, and both pydantic-ai suites in
  `tools-live --fail-on-tool-drift` compare 30 live tool results with 0
  drift (below).
- **Checks** (both suites): `in_order` sequence of 8 `get_booking` then
  `convert_currency`; `answer_states` for P6B2LH and 345; no writes; at most
  12 tool calls and 14 model calls. The step limits moved into each scenario,
  so the three short scenarios keep 8 and 6.
- **Recording:** up to 5 qwen2.5:7b attempts (`make record-pydantic-ai-chain`,
  log `eval/results/m16_qwen_recording.json`). None passed: every attempt
  walked all 8 links to P6B2LH, then four answered with a total of their own
  without converting (25, 500 and 260 USD, and "200 plus 100"), and one
  converted 25 USD.
  As M9 did for the cancellation, the qwen suite serves the chain from the
  fake-model recording. In `model-live` only tool steps are served, and the
  chain's tool results do not depend on who recorded them, so the world is
  the one qwen would have recorded, except the conversion: the fake
  recording holds the right amount, 375.
- **Arms** (`eval/pydantic_ai.py --long`, `LONG_ARMS`): qwen2.5:7b (digest
  `845dbda0…`, the recording prompt, default sampling) in `model-live` with
  the suite's `identify` keys, 20 runs under `strict` and 20 under
  `fallback`, one at a time. 20 runs, not 15: the slowest recording attempt
  took 59.4 s, under the 150 s cut.
- **What is counted:** each run's tool steps as a status sequence. A *world
  step* is any step not served `exact` (approximate, unrecorded, fallback).
  An *identity firing* is an unrecorded or fallback call made while unused
  recordings of the same tool remained (`identity_firings`). The survival
  curve is the product-limit share of runs with no world step through
  served-call index n (a run that finished earlier leaves the risk set),
  with the at-risk count, beside `(1 − 3/143)^n`. Clustering is the rate of
  world steps among steps that follow an earlier world step in the same run,
  divided by the arm's base rate, reported only with at least 5 world steps.
  Intervals are Clopper-Pearson (`exact_interval`, stdlib bisection on the
  binomial tail, checked against scipy in the tests).
- **Machine:** 4 vCPU, AMD EPYC 9V45 (not M15's EPYC 7763), CPU only,
  Ollama 0.40.11. $0.
- **Raw output:** `eval/results/m16_long_qwen.json` (every run's statuses,
  tool calls with inputs, answer, cause, error and seconds; per arm the
  survival curve, clustering, identity firings and world endings).

```
$ make eval-pydantic-ai-long
uv run python -m eval.pydantic_ai --long
  ... (the eight full-mode replays, all PASS, 0 approximate)
PASS replay-qwen.toml trace_rebooking_chain: 19 steps, 0 approximate
arm strict: 20 model-live runs of trace_rebooking_chain (qwen2.5:7b, recording, temperature None, strict)
  trace_rebooking_chain: 0/20 passed (1 unrecorded, 19 model); 1 by identity, 2 tool paths
  wall time 849.4 s
  strict: 1/20 runs with a world step, 1 identity firings, clustering None
arm fallback: 20 model-live runs of trace_rebooking_chain (qwen2.5:7b, recording, temperature None, fallback)
  trace_rebooking_chain: 0/20 passed (20 model); 0 by identity, 1 tool paths
  wall time 853.1 s
  fallback: 0/20 runs with a world step, 0 identity firings, clustering None
wrote eval/results/m16_long_qwen.json
```

| `model-live`, `trace_rebooking_chain` (qwen2.5:7b, 20 runs) | Passed | passed / unrecorded / approximate / model | By identity | Gate | Served tool calls per run | World steps | Seconds per run (min / median / max) | Wall |
|---|---|---|---|---|---|---|---|---|
| M16 `strict` | 0 / 20 | 0 / 1 / 0 / 19 | 1 | fail | 8 (19 runs), 9 (1 run) | 1 in 161 | 40.8 / 41.7 / 49.8 | 849 s |
| M16 `fallback` | 0 / 20 | 0 / 0 / 0 / 20 | 0 | fail | 8 (all 20) | 0 in 160 | 41.1 / 42.1 / 48.3 | 853 s |

The M9, M12 and M15 rows above are unchanged.

**Survival by served-call index** (both arms: at risk, world steps there,
product-limit survival; independence `(1 − 3/143)^n`):

| n | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| `strict` at risk / world / survival | 20 / 0 / 1.0 | 20 / 0 / 1.0 | 20 / 0 / 1.0 | 20 / 0 / 1.0 | 20 / 0 / 1.0 | 20 / 0 / 1.0 | 20 / 0 / 1.0 | 20 / 0 / 1.0 | 1 / 1 / 0.0 |
| `fallback` at risk / world / survival | 20 / 0 / 1.0 | 20 / 0 / 1.0 | 20 / 0 / 1.0 | 20 / 0 / 1.0 | 20 / 0 / 1.0 | 20 / 0 / 1.0 | 20 / 0 / 1.0 | 20 / 0 / 1.0 | — |
| independence | 0.979 | 0.959 | 0.938 | 0.919 | 0.899 | 0.881 | 0.862 | 0.844 | 0.826 |

The 0.0 at n = 9 rests on one run, the only one that reached a ninth call.

**The hypotheses:**

- **Clustering (the decision test): not measurable.** The `fallback` arm had
  0 world steps in 160 served calls, short of the 5 the test needs, so it is
  reported as not measurable, not as refuted. (`strict` had 1, and a
  `strict` run ends at its first world step, so it can never show a second.)
- **`strict` world endings: 1 of 20 (Clopper-Pearson 95%: 0.1% to 24.9%),
  beside 3.47 predicted by independence at 9 calls (95% range 1 to 7).**
  Inside the range, and, as the plan said, 20 runs cannot separate 1 from
  3.5: this count is descriptive. The one ending (run 14) read all 8 links,
  then converted 325 USD where the chain totals 375, and the identify key on
  `amount` stopped it: a model error the keys caught, not a path the
  recording lacked.
- **The identify keys fire at least 5 times in 40 runs: refuted, 1.** The
  plan expected a chain of references to be where a model mis-copies one.
  It never did: 320 of 320 `get_booking` calls (40 runs × 8 links) carried
  the exact reference the previous result named, and all 40 lookups in the
  5 recording attempts did too.
- **The survival curve against `(1 − 3/143)^n`:** every run survived its 8
  lookups (40 of 40) where independence at the short-run rate predicts
  0.844. Forty clean 8-lookup prefixes have probability 0.844^40 = 0.0011
  under that model. The per-call rate on this chain is 1 world step in 321
  served calls, 0.31% (0.01% to 1.72%), against M12's 2.1% (0.43% to 6.0%);
  for the lookups alone it is 0 of 320 (at most 1.15%). A one-sided Fisher
  test of 1 / 321 against 3 / 143 gives 0.089, so the two rates are not
  shown to differ, but the independent-per-call model with M12's rate is
  rejected for this chain.

**Whose failure, then.** Every run failed, and it was the model in 39 of
40: with all served steps exact, 39 runs (19 `strict`, 20 `fallback`) never
called `convert_currency` and answered with a euro figure of their own
making. 39 named the right original, P6B2LH. 38 of the 39 stated a USD
total: 500 (10 runs), 250 (9), 260 (4), 275 (3), 251 (2), 25 (2), 150 (2),
255 (2), and 625, 325, 125 and 175 once each, never 375. Of the 9 runs that
showed their working, 7 summed fewer payments than they had read ("25 * 6 =
150", five $25 fees plus $200 = 275, seven fees without the fare = 175), one
counted ten fees ("25 * 10 = 250"), and one wrote the right expression and
added it wrong ("25 * 7 + 200 = 325"). The euro figure beside each total implies a rate
of 0.71 to 0.90 EUR per USD, never the world's 0.92: the conversion was
invented along with the sum. The `fallback` arm confirms it: with the real tool
behind any call the recording lacked, there was no such call, and the pass
count was the same 0 / 20. The `in_order` sequence check failed 39 runs and
`answer_states` all 40.

**What this means for `model-live`.** On this chain, frozen-world
divergence does not compound: it hardly happens. The short-run 2.1% is not
a per-call constant. M12's three world steps were decisions (an extra
`get_booking` on a booking task), and a forced chain whose arguments are
printed verbatim in the previous result leaves no decision to make until the
end. The risk on a long run sits where the model computes or chooses, and
there the identify keys stop a wrong value under `strict` the way M15
showed, which `fallback` then attributes to the model. The limits are real:
one model, one task, 40 runs. A chain needing a computed or chosen argument
at every link, or a model weaker at copying, could behave differently. The
first real cassette past 6 turns did not happen either: no qwen2.5:7b
attempt passed, so `make bench-scale` has no new real row (its sizes
report is unchanged, re-checked by `tests/test_bench_scale.py`), and the
longest committed real trace is still the 6-turn retail recording.

### Drift and full replay with the chain in the world

```
$ uv run replay ci examples/pydantic_ai_agent/replay.toml --mode tools-live --fail-on-tool-drift
PASS cheapest_fare_in_euros (tools-live; 3 exact, 2 live)
PASS book_earliest_flight (tools-live; 3 exact, 2 live)
PASS cancel_refundable_booking (tools-live; 3 exact, 2 live)
PASS trace_rebooking_chain (tools-live; 10 exact, 9 live)
4/4 scenario(s) passed: passed
$ uv run replay ci examples/pydantic_ai_agent/replay-qwen.toml --mode tools-live --fail-on-tool-drift
  ... the same four rows, 4/4 passed
```

`no_tool_drift`: 2, 2, 2 and 9 live tool results matched the recording in
each suite, 0 drifted (30 compared). `make eval-pydantic-ai-offline` (in
`make eval`) replays both suites in `full`: the chain 19 steps, 0
approximate, and the other six scenarios unchanged at 5 steps, 0
approximate.

## `synth`, measured for the first time, and the no-guard arm's twins (M17)

`synth` is the third unrecorded-call policy. When a replay meets a tool call
the cassette never saw, `strict` ends the run, `fallback` runs the real tool,
and `synth` answers with the smallest value that validates against the tool's
return type (`agent_replay/synth.py`): an empty record, an empty string,
`0.0`. It shipped in v1 and no number in this file had ever used it. Two
questions. Offline: in the mutation study, does a stub keep a diverged run
alive to a true verdict, or does it manufacture a world that the checks then
blame on the agent? Live: M15's no-guard arm ended 18 of 20 fare runs by
identity under `strict`, and RESULTS.md M15 attributed them to the model's
guessed amounts. Does that attribution survive the same arm under `fallback`
(the guessed amount converted for real) and `synth` (the guessed amount
answered `0.0`)?

### Offline: a seventh mutation-study configuration

- **Configuration:** `model-live+synth` (`eval/mutation_study.py`, `CONFIGS`):
  `model-live`, the scripted fake model live, tools served, an unrecorded
  tool call answered with a stub. It is appended after the six earlier
  configurations, and its path-variant rows follow the M11 `strict` and
  `fallback` rows, so every earlier cell keeps its place.
- **Hypotheses (PLAN.md, written before the run):** the 12 model mutations
  are caught as in `model-live+fallback` within one; the two identity
  mutations are caught 2 / 2 with the keys, by a run error or a wrong answer
  from the stub rather than by ground truth; at least one path variant fails
  a ground-truth rule or ends in a run error under `synth`, where `fallback`
  failed none (`repeats_order_lookup` reads a field from a `{}` stub); the
  refactors stay 0 / 5.
- **Dataset and method:** the M4 study (D4: 32 mutations, 5 benign
  refactors, the six fake-model retail scenarios), the M11 identity
  mutations and path variants, unchanged.
- **Command:** `make eval-mutations` (195 s on 4 vCPU, AMD EPYC 7763, with
  the live arms below running on the same cores; a second run left
  `eval/results/mutations.json` unchanged). Every earlier cell is identical:
  removing the seventh configuration and the `synth` path rows from the new
  file gives back the committed M11 file key for key (`mutations`,
  `refactors`, `summary`, `identify`, `path_variants`, `configurations`,
  `definitions` all equal). $0.

Raw output of `make eval-mutations`, cut to the rows and the `model-live`
columns this section is about (`...` marks a cut; the full matrix is in
`eval/results/mutations.json` and printed by the command):

```
### Detection rate by configuration

| configuration | all | agent | model | tool |
|---|---|---|---|---|
| model-live | 22/32 (69%) | 10/12 (83%) | 12/12 (100%) | 0/8 (0%) |
| model-live+fallback | 22/32 (69%) | 10/12 (83%) | 12/12 (100%) | 0/8 (0%) |
| model-live+synth | 22/32 (69%) | 10/12 (83%) | 12/12 (100%) | 0/8 (0%) |

### Benign refactors (false alarms; approximate steps)

False alarms: full 0/5, tools-live 0/5, model-live 0/5, model-live+fallback 0/5, tools-live+strict 1/5, tools-live+drift 0/5, model-live+synth 0/5.
Survived every configuration: `drops_error_flag`.

### Mutations only identity can see (caught by; — survived)

| mutation | identify | model-live | model-live+fallback | model-live+synth |
|---|---|---|---|---|
| `guesses_user_id` | without | — | — | — |
| `guesses_user_id` | with | `arg_schema`, `output_schema`, `required_tools`, `rule:matches_ground_truth`, `run_error:UnrecordedCallError`, `tool_sequence` | `arg_schema`, `output_schema`, `required_tools`, `rule:matches_ground_truth`, `run_error:JSONDecodeError`, `tool_sequence` | `arg_schema`, `output_schema`, `required_tools`, `rule:matches_ground_truth`, `run_error:KeyError`, `tool_sequence` |
| `reads_wrong_order` | without | — | — | — |
| `reads_wrong_order` | with | `run_error:UnrecordedCallError`, `tool_sequence` | `tool_sequence` | `arg_schema`, `required_tools`, `rule:matches_ground_truth`, `run_error:KeyError`, `tool_sequence` |

Detected (of 2) without → with the keys: ... model-live 0 → 2, model-live+fallback 0 → 2, ... model-live+synth 0 → 2.

### Detection rate with the keys (34 mutations)

| model-live+synth | 24/34 (71%) | 12/14 (86%) | 12/12 (100%) | 0/8 (0%) |

False alarms with the keys: ... model-live+synth 0/5.

## Correct-but-different paths (M11), the synth rows with the keys on

| variant | policy | identify | affected | termination | approx | fallback | `arg_schema` | `budget` | `forbidden_tools` | `max_tool_calls` | `output_schema` | `required_tools` | `rule:checked_before_writing` | `rule:db_matches_ground_truth` | `rule:matches_ground_truth` | `tool_sequence` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `reads_only_needed_orders` | synth | on | 3/6 | 3 checks, 3 passed | 0 | 0 | · | · | · | · | · | · | · | · | · | 3 |
| `swaps_parallel_lookups` | synth | on | 5/6 | 1 checks, 5 passed | 0 | 0 | · | · | · | · | · | · | · | · | · | 1 |
| `repeats_order_lookup` | synth | on | 4/6 | 2 passed, 4 run_error:KeyError | 0 | 0 | 2 | · | · | · | · | 2 | · | · | 4 | 4 |
| `verifies_after_write` | synth | on | 4/6 | 4 checks, 2 passed | 0 | 0 | · | · | · | · | · | · | · | · | · | 4 |
| `rewords_transfer_summary` | synth | on | 1/6 | 6 passed | 1 | 0 | · | · | · | · | · | · | · | · | · | · |

- synth, identify on: 4 raise a false alarm; 0 end on an unrecorded call (none); failing check types: `arg_schema` 1, `required_tools` 1, `rule:matches_ground_truth` 1, `tool_sequence` 4.
- synth, identify off: 4 raise a false alarm; 0 end on an unrecorded call (none); failing check types: `arg_schema` 1, `required_tools` 1, `rule:matches_ground_truth` 1, `tool_sequence` 4.
```

(The `synth` rows with the keys off are identical to those with them on.
The `repeats_order_lookup` and `verifies_after_write` rows each served 4
stubbed steps, one per affected scenario; the other variants served none.)

What the seventh column says, against the hypotheses:

- **Model mutations: 12 / 12, the same as `model-live+fallback`. Holds.**
  All 32 mutations are detected in the same 22 cases as under `fallback`
  and `strict`, and 31 of the 32 rows name the same check types. The one
  difference is `repeats_lookups`, which re-reads the user's profile until
  it is cut off: its stubbed re-reads add `run_error:KeyError` beside the
  checks `fallback` already failed. The model mutations rarely make a call the
  recording lacks, so the policy rarely gets a say.
- **Identity mutations: 2 / 2 with the keys, by a run error. Holds, with a
  correction to "rather than by ground truth".** Both end in
  `run_error:KeyError`: the scripted model reads a field from the `{}` it
  was served for the misdirected read and crashes. The ground-truth rule
  fails too, but only because the run ended before its write. Read beside
  `fallback`, this is the stub's verdict, not the agent's. Under `fallback`
  `reads_wrong_order` reads the wrong order for real, needs one extra
  lookup, and fails only `tool_sequence`; the database ends correct. Under
  `synth` the same mutation fails `rule:matches_ground_truth`,
  `arg_schema` and `required_tools`, all after the crash.
- **Path variants: `repeats_order_lookup` fails the ground truth in 4 of 6
  scenarios under `synth`, where `fallback` failed none. Holds.** The
  variant reads an order twice; the cassette has one recording of it, so
  the second read is unrecorded and gets `{}`. The scripted model then
  indexes `order_id` and the run ends in `KeyError` before its write. That
  is a false alarm on a correct agent: 4 scenarios fail
  `rule:matches_ground_truth`, 2 also fail `arg_schema` and `required_tools`.
  `strict` failed the same 4 scenarios the same way in M11 (the run ended on
  the unrecorded re-read). So for a correct agent that reads twice, `synth`
  gave the same false verdict as `strict`, and only `fallback` passed the
  ground truth. `verifies_after_write` was also stubbed 4 times, but its
  extra read comes after the write, so only `tool_sequence` noticed (as
  under `fallback`).
- **Refactors: 0 / 5. Holds.** No refactor makes an unrecorded call, so
  `synth` never acts on one (0 stubbed steps).
- **Caveat.** The crashes are the scripted model's: it indexes the JSON it
  is served, so a `{}` stub is a `KeyError`. A real model reads `{}` as
  text and carries on with it, which is what the live arms measure.

### Live: M15's no-guard arm under `fallback` and `synth` on qwen2.5:7b

- **Dataset:** the two scenarios qwen2.5:7b recorded (`cheapest_fare_in_euros`,
  `book_earliest_flight`, `examples/pydantic_ai_agent/replay-qwen.toml`, M11
  `identify` keys on), the same cassettes as M12, M15 and the control.
- **Arms** (`eval/pydantic_ai.py`, `NO_GUARD_ARMS`): qwen2.5:7b (digest
  `845dbda0…`, the recorder) with the recording prompt minus its guard line
  (`run_no_guard`, as in M15), `model-live`, 20 runs per scenario, one at a
  time, under `fallback`, then under `synth`. The `strict` twin is M15's
  `7b_no_guard_strict`, read from the committed `m15_swap_qwen.json` into the
  new report (`m15_strict`). Every run row also carries `guard_in_system:
  [false]`, as asked.
- **Causes** as in M15, plus one: `synth`, a run that failed (a check, or
  the agent raised) after a stubbed step, which is possibly the stub's
  fault, as `approximate` is possibly the recording's. A stubbed step is a
  world step, as in M16 (any tool step not served `exact`). Each run also
  records the money figures its answer states (a number next to a currency
  word or sign, `figures`) and whether one equals a number a stubbed step
  returned (`states_stub_figure`).
- **Hypotheses (PLAN.md, written before the run):** under `fallback` the fare
  passes at most 6 / 20, so M15's attribution stands (15 / 20 or more would
  withdraw it; 7 to 14 is partial). Under `synth` the fare passes at most as
  often as under `fallback`, at least 10 of the 20 fare answers state the
  stub's figure, and no run with a stubbed step passes.
- **Machine:** 4 vCPU, AMD EPYC 7763, CPU only, Ollama 0.40.11. $0. During
  the `fallback` arm the offline gate and `make eval-mutations` ran on the
  same cores, so its wall time is not comparable with the `synth` arm's or
  M15's; the pass counts do not depend on it.
- **Raw output:** `eval/results/m17_no_guard_qwen.json` (every run's cause,
  tool calls with their status and, off the recording, the result they got;
  answer, figures, stub figures, steps, seconds; the hypotheses as computed).

```
$ make eval-pydantic-ai-no-guard
uv run python -m eval.pydantic_ai --no-guard
  ... (the eight full-mode replays, all PASS, 0 approximate)
arm fallback: 20 model-live runs of cheapest_fare_in_euros, book_earliest_flight (qwen2.5:7b, no_guard, temperature None, fallback)
  cheapest_fare_in_euros: 2/20 passed (2 passed, 18 model); 0 by identity, 9 tool paths
  book_earliest_flight: 3/20 passed (3 passed, 17 model); 0 by identity, 4 tool paths
  wall time 2069.1 s
  cheapest_fare_in_euros: 0 runs with a stubbed step, 0 stating a stub's figure, 0 passes with one
  book_earliest_flight: 0 runs with a stubbed step, 0 stating a stub's figure, 0 passes with one
arm synth: 20 model-live runs of cheapest_fare_in_euros, book_earliest_flight (qwen2.5:7b, no_guard, temperature None, synth)
  cheapest_fare_in_euros: 0/20 passed (20 synth); 0 by identity, 7 tool paths
  book_earliest_flight: 2/20 passed (2 passed, 1 synth, 17 model); 0 by identity, 3 tool paths
  wall time 1733.8 s
  cheapest_fare_in_euros: 20 runs with a stubbed step, 11 stating a stub's figure, 0 passes with one
  book_earliest_flight: 1 runs with a stubbed step, 0 stating a stub's figure, 0 passes with one
wrote eval/results/m17_no_guard_qwen.json
```

| `model-live`, qwen2.5:7b without the guard line | Passed | passed / unrecorded / approximate / synth / model | Runs with a world step | Gate | Seconds per run (min / median / max) | Wall |
|---|---|---|---|---|---|---|
| M15 `strict`: `cheapest_fare_in_euros` | 2 / 20 | 2 / 18 / 0 / — / 0 | 18 (all by identity) | fail | 14.9 / 15.5 / 86.6 | 1170 s |
| M15 `strict`: `book_earliest_flight` | 2 / 20 | 2 / 5 / 0 / — / 13 | 5 (all by identity) | fail | 32.2 / 35.8 / 55.4 | |
| M17 `fallback`: `cheapest_fare_in_euros` | 2 / 20 | 2 / 0 / 0 / 0 / 18 | 18 (a guessed conversion, run for real) | fail | 47.1 / 60.5 / 94.1 | 2069 s |
| M17 `fallback`: `book_earliest_flight` | 3 / 20 | 3 / 0 / 0 / 0 / 17 | 3 (a wrong flight, booked for real) | fail | 34.3 / 38.9 / 60.0 | |
| M17 `synth`: `cheapest_fare_in_euros` | 0 / 20 | 0 / 0 / 0 / 20 / 0 | 20 (a guessed conversion, answered `0.0`) | fail | 45.1 / 47.5 / 50.6 | 1734 s |
| M17 `synth`: `book_earliest_flight` | 2 / 20 | 2 / 0 / 0 / 1 / 17 | 1 (a wrong flight, answered with a blank booking) | fail | 33.7 / 35.6 / 56.4 | |

The M15 rows are copied unchanged from its section; the control on the
shipped prompt was 18 / 20 and 10 / 20. 80 runs, 3803 s of arms, 0
approximate steps.

What the 80 runs say, against the hypotheses:

- **The `fallback` fare passes 2 / 20 (at most 6 was the bar), so M15's
  attribution stands.** The same 18 failures that ended by identity under
  `strict` ran to the end here, and every one failed `answer_states` on
  its own. Each of the 18 sent `search_flights` and a guessed
  `convert_currency` in one reply (`parallel_tool_replies: 1`; amounts 100,
  120, 123.45, 150, 250, 350 and 450, none of them a fare in the search
  result). The real tool converted the guess, and 11 of the 18 answers
  stated that conversion as the fare. The other 7 noticed it did not fit
  the $189 they had just read and wrote their own estimate (134 to 171 EUR);
  none wrote 173.88. All 20 answers named the right flight, B6416. So the
  recording never stood between the model and a passing run: with the
  real tool behind every guess, the pass count is M15's to the run. The two
  passes made one call per reply and converted 189, served `exact`.
- **The `synth` fare passes 0 / 20, at most `fallback`'s 2. Holds, but not
  because of the stub.** In all 20 `synth` fare runs the model guessed the
  amount in a parallel reply (against 18 / 20 under `fallback` and 18 / 20
  in M15), so every run had a stubbed step and no run took the path that
  passes. Across all three policies no run with an off-recording step
  passed and no run without one failed the fare, so the policy decided no
  verdict here; 0 against 2 of 20 is sampling (one-sided Fisher, 0.24).
- **11 of the 20 `synth` fare answers state the stub's figure. Holds (at
  least 10).** They converted the guess, got `0.0`, and wrote "approximately
  0.00 EUR" (or 0 EUR, 0.0 EUR) beside the $189 they had read. The other 9
  ignored the zero and invented a euro figure (133.00 to 166.15). Neither
  kind is a figure any real tool returns, so the stub's world shows up in
  the answer about half the time and the ground-truth rule fails it either
  way.
- **0 passes with a stubbed step, on either scenario. Holds.** The one
  stubbed booking (run 12, WN1849, the latest flight) got a blank booking
  (`booking_ref: ""`, `paid_usd: 0.0`) and answered "The booking was
  successful… The fare is $99.00": no reference, and a fare from the search,
  not the stub. It failed `writes_match` (nothing was written) and
  `answer_states`, which is the verdict `fallback` gave its 3 wrong
  bookings with the write performed.
- **The cause column moves with the policy, not with the fault.** The same
  failure, a guessed amount, is `unrecorded` (by identity) under `strict`,
  `model` under `fallback` and `synth` under `synth`. Only `fallback`
  labels it as the model's, because only `fallback` lets the guess meet a
  real tool and fail on its own. The booking arm shows the model's other
  failure mode, unchanged by policy: 13 to 17 runs of 20 searched, announced
  a flight and stopped without booking (17 under `synth`, 14 under
  `fallback`, 13 in M15).

**What `synth` is for, from these numbers.** On a real model it gave the
same gate verdict as `fallback` on both scenarios (fail, fail) without
running a single real tool past the recording, and it performed no write:
the wrong booking was answered with a blank record, not made. That is its
use: a `model-live` gate for an agent whose tools cannot be called from CI
(a payment, a booking, a production database), where `fallback` is not an
option and `strict` stops every diverged run before its answer. It is not a
way to tell a correct new path from a wrong one. Offline, the correct agent
that re-reads an order failed the ground truth under `synth` exactly as under
`strict` (4 of 6 scenarios), where `fallback` passed it. And live, its cause
column blamed the stub for 20 runs that were the model's. Read a `synth`
failure as "the run left the recording"; whose fault it was takes a
`fallback` twin, as M15 found for `strict`.

## A task where a tool legitimately fails (M18)

**Question.** Since v1, `drops_error_flag` (the agent reports a failed tool as
an ordinary result: `is_error` lost in a refactor) has survived every
configuration of the mutation study, with 0 approximate steps. Is that the
checks' failure or the dataset's? Which replay modes can see a lost error flag
at all? And does a lost flag matter to a real model, which can read the error
text the agent still sends (`Error: user not found`)?

**Dataset.** A seventh retail task, hand-written for this repo and not from
τ-bench (`examples/retail_agent/tasks.py`, `HANDWRITTEN`; `TASKS` keeps the six
vendored ones): *"I'm Noah Ito, zip code 98178. Please cancel my pending order
#W6729841 because I no longer need it. Yes, I confirm."* Noah Ito exists, at
98187, so `find_user_id_by_name_zip` raises `user not found` and the agent sends
the model an `is_error: true` result. Ground truth: no write and no transfer
(`expected_actions = ()`), and an answer that says the account was not found
(the new rule `reports_account_not_found`: no write tool called, and the answer
matches a fixed not-found pattern, `rules.NOT_FOUND`, which must name the account,
the user or their identity; tightened after the live runs so that "the payment did
not match" no longer passes, with all 23 committed live answers read the same and
`mutations.json` unchanged). Its suite,
`examples/retail_agent/replay-error.toml`, is separate from `replay.toml`, so
every earlier table is unchanged. It has no `tool_sequence` check: which
lookups a model tries before giving up is not the regression. Its fake-model
cassette (`cassettes/error/`, 3 envelopes) is the first retail recording with
an `is_error: true` tool result.

### Offline: the error task in the mutation study

`make eval-mutations` gains the `error_task` section: the task unmutated and
under each of the 12 agent mutations, in all seven configurations, without the
identify keys (as the v1 tables run). Every earlier key of
`eval/results/mutations.json` is byte-identical to the committed M17 file;
only `error_task` is new. 1 min 41 s on 4 vCPU, offline.

```
$ make eval-mutations
uv run python eval/mutation_study.py --out eval/results/mutations.json
...
Detected (of 12 agent mutations): full 1, tools-live 1, model-live 2, model-live+fallback 2, tools-live+strict 4, tools-live+drift 1, model-live+synth 2. Unmutated fails in: none.
```

| variant | full | tools-live | model-live | model-live+fallback | tools-live+strict | tools-live+drift | model-live+synth |
|---|---|---|---|---|---|---|---|
| `unmutated` | pass; 0 approx | pass; 0 | pass; 0 | pass; 0 | pass; 0 | pass; 0 | pass; 0 |
| `drops_error_flag` | —; 1 approx | —; 1 | `run_error:UnrecordedCallError`, `rule:reports_account_not_found`; 0 | `run_error:JSONDecodeError`, `rule:reports_account_not_found`; 0 | `no_approximate_steps`; 1 | —; 1 | `run_error:KeyError`, `rule:reports_account_not_found`; 0 |
| `returns_empty_answer` | `rule:reports_account_not_found` | same | same | same | same | same | same |
| `unregisters_product_tool` | —; 2 approx | —; 2 | —; 0 | —; 0 | `no_approximate_steps`; 2 | —; 2 | —; 0 |
| `strips_policy` | —; 2 approx | —; 2 | —; 0 | —; 0 | `no_approximate_steps`; 2 | —; 2 | —; 0 |
| the other 8 agent mutations | — everywhere, 0 approx | | | | | | |

The full table is in the study's Markdown output and in `mutations.json`
(`error_task`).

**Hypotheses (PLAN.md, written before the run), answered:**

- **`drops_error_flag` is caught in at least 2 of the 7 configurations. Holds:
  4 of 7.** The predicted three (`model-live`, `model-live+fallback`,
  `tools-live+strict`), plus `model-live+synth`. In every `model-live`
  configuration the fake model, which branches on `is_error`, gets the error
  text as an ordinary result and uses it as a user id: it calls
  `get_user_details(user_id="Error: user not found")` and
  `get_order_details("#W6729841")`. Under `strict` the recording has neither
  call (`UnrecordedCallError`). Under `fallback` the real `get_user_details` fails
  too, the mutant strips that flag as well, and the fake model parses the
  error text as a profile (`JSONDecodeError`). Under `synth` the stubs' empty
  records lack the keys the fake model reads (`KeyError`). In `tools-live+strict` the model reply is
  served, but the request that should carry the flag differs from the
  recording, so its one approximate step fails `no_approximate_steps`.
- **It survives `full` and `tools-live+drift`, as predicted, and plain
  `tools-live` too.** The reason is the same in all three: the flag lives in
  the agent's message *to the model*, and in these modes the model's reply is
  served from the recording whatever that message says. The changed request
  is recorded as 1 approximate step, which only `tools-live+strict` turns into
  a failure. The drift check compares *tool results*, and the tool fails
  exactly as recorded (same error, same text); the flag is added afterwards,
  by the agent.
- **The unmutated task passes every configuration. Holds.**
- **So v1's 0 of 6 was the dataset's.** No recorded run had a failing tool, so
  the mutation had nothing to drop. Given one, the modes where the model runs
  catch it, and so does the strict request match.

The other agent mutations show how narrow a probe this task is. It ends after
one failed lookup (two model calls, three messages), so 8 of the 12 mutations
have nothing to act on: no order is read, nothing is written, one tool per
turn, a result under 120 characters, a conversation within the 3-turn and
4-message limits. `returns_empty_answer` is caught everywhere by the new rule.
`unregisters_product_tool` and `strips_policy` change the request bytes, so
`tools-live+strict` catches them; the fake model ignores both the tool list and
the prompt.

### Live: the same task on qwen2.5:7b through Ollama

**Recording.** `make record-retail-error-qwen` (`eval/retail_error.py
--record`): `examples/retail_agent/ollama.py` runs the retail agent unchanged
(same prompt, tools and loop) with its `Anthropic` client pointed at Ollama's
`/v1/messages`. Up to 3 attempts; **attempt 1 passed** (152.2 s) and is
committed as `examples/retail_agent/cassettes/qwen/account_not_found_wrong_zip.jsonl`.
qwen2.5:7b looked the user up by name and zip, then by an email it made up
(`noah.ito@example.com`, which also fails), and answered *"I couldn't find a
user associated with the provided email and name. Could you please provide
your email again or check your name details?"* No write. Log:
`eval/results/m18_qwen_recording.json`.

**Context window.** Ollama 0.40.11 starts qwen2.5:7b with a 4096-token context
and context shift on (`llama-server -c 4096 --context-shift`), so a long
prompt would be cut silently. The retail policy, tools and task are 2,876
tokens (the recording's first call, on a cold cache). Ollama's
`usage.input_tokens` counts only what its prompt cache did not already hold:
43 and 33 for the recording's next two calls. (Even the cold count is 3
tokens under the server's own, 2,876 against 2,879, so the bound below is
approximate; the server's counts are the check.) So each run also gets an upper
bound on its largest prompt (`prompt_bound`: the cold prefix, plus every later
call's input, plus every earlier call's output). It is 3,024 tokens for the
recording and at most 3,024 in all 20 arm runs. The server's own
`task.n_tokens`, read from its debug log, agree: the largest prompt among all
73 requests of the session was 2,991 tokens (2,879 / 2,941 / 2,991 per turn).
No prompt reached the window. The counts are in
`eval/results/m18_ollama_is_error_trace.txt`. Of the 73, the 60 at those three
sizes are the recording's 3, the arms' 55 and 2 from the stopped start
described below; the 13 small ones are the first `is_error` check and three
runs of the probe (in the stopped start, once by hand after the fix, and in
the arms).

**What Ollama does with `is_error`.** With `OLLAMA_DEBUG=2` the server logs
each Anthropic request as it arrived and as converted. A `tool_result` with
`is_error: true` arrives with the flag and is converted to a `role: tool`
message holding only the content; with and without the flag the model's
prompt is the same 201 tokens. The excerpt is in
`eval/results/m18_ollama_is_error_trace.txt`. The harness repeats this without the
log (`probe_flag`, in the raw report as `flag_probe`). It sends the same
tool result with the flag, without it, and with three words added (the
control), each after a warm-up call so that all three find the same prefix in
the prompt cache. The counts were 84 (flag), 84 (no flag) and 87 (control):
**on Ollama the flag never reaches the model**. A first start of the arms was stopped
after its probe misread. Without the warm-up, the first probe also paid for
the uncached chat-template prefix (189 against 84). The warm-up was added
and the arms restarted from scratch; nothing from the stopped start is in
the report.

**Arms.** `make eval-retail-error-qwen` (`eval/retail_error.py`): the probe,
then the task in `model-live` under `fallback` on the qwen recording with the
suite's identify keys, 10 runs with the unmutated agent and 10 with
`drops_error_flag` (`eval/mutations/agents.py:drops_error_flag_ollama`), both on
qwen2.5:7b (digest `845dbda0…`), one run at a time (`wilson` gate, so exactly 10).
4 vCPU AMD EPYC 9V74, CPU only, Ollama 0.40.11, nothing else running on the
cores. $0. Raw report: `eval/results/m18_error_qwen.json` (every run's tool calls
with their status, answer, not-found verdict, `is_error` flags its requests
carried, input and output tokens, prompt bound, seconds). The committed runs
predate keeping a failed call's error text with the call (a review fix); every
lookup was served `exact`, so each failed with the recording's `user not found`.

```
$ make eval-retail-error-qwen
uv run python -m eval.retail_error --runs 10
flag probe: the flag never reaches the model ([84, 84, 84, 87])
arm unmutated: 10 model-live runs under fallback
  account_not_found_wrong_zip: 8/10 passed, 8 replied not found with no write; 235.9 s
arm drops_error_flag: 10 model-live runs under fallback
  account_not_found_wrong_zip: 9/10 passed, 9 replied not found with no write; 248.8 s
near-equivalent: the mutant replies not found as the unmutated agent does; wrote .../eval/results/m18_error_qwen.json
	Elapsed (wall clock) time (h:mm:ss or m:ss): 8:20.80
```

| `model-live`, `fallback`, qwen2.5:7b | Passed | Not found, no write (rule) | By reading the answers | Runs whose requests carried `is_error` | Tool paths (1 lookup / 2) | World steps | Seconds per run (min / median / max) | Wall |
|---|---|---|---|---|---|---|---|---|
| unmutated | 8 / 10 | 8 / 10 | 9 / 10 | 10 / 10 (1 or 2 flags) | 3 / 7 | 0 (every step `exact`) | 15.2 / 25.3 / 29.1 | 236 s |
| `drops_error_flag` | 9 / 10 | 9 / 10 | 10 / 10 | 0 / 10 | 2 / 8 | 0 (every step `exact`) | 18.3 / 25.7 / 28.8 | 249 s |

Every failure was the not-found rule. No run in either arm wrote, transferred
or called anything but the two lookups, and every tool step was served
`exact`: the model's guessed email was always the one in the recording. Read
by hand, two of the three answers the pattern missed do say it: *"It seems
there was an issue locating your user ID"* (mutant run 4) and *"there was an
issue finding your user ID"* (unmutated run 1). The third, *"Could you please
provide me with your email address so I can locate your user ID?"* (unmutated
run 3), does not. The rule was fixed before the runs and is not loosened after
them; the "by reading" column is my reading of the 20 answers in the raw
report. Both arms fail the default `wilson` gate (lower bound under 0.8 at
8 / 10 and 9 / 10): 10 runs cannot certify an 80% pass rate, and the gate is
not the question here.

**Hypothesis (PLAN.md): the mutant replies "not found" with no write in at
least 8 of 10, within 2 of the unmutated arm. Holds: 9 against 8 by the rule,
10 against 9 by reading.** For qwen2.5:7b served by Ollama, `drops_error_flag`
is not a regression. The plan's reason was "a model that reads the error text
does not need the flag". The probe shows something stronger, and different.
qwen never saw the flag in either arm, because Ollama's Anthropic layer drops
it before the prompt is built. The unmutated runs sent it (10 of 10), the
mutant's did not (0 of 10). Every one of the arms' 55 model requests was
2,879, 2,941 or 2,991 tokens, the sizes of the recording's three turns, so
the flag added no token in either arm (the probe and the trace show that the
content is the same, not only the size). So on this stack the two
arms are two samples of the same input, and 8 against 9 is sampling. The
fake-model catches above are the fake policy's dependence on the flag, as
predicted. Whether a model that *does* see the flag misbehaves without it
needs a provider that delivers it. Anthropic's API documents `is_error` as
telling the model the tool failed, so the Claude recording is under "To
measure".

What this means for a team using Replay. A lost `is_error` is caught where the
model runs against the recording (`model-live` under any policy) or where the
request bytes are compared (`fail_on_approximate`), on a fake model that
branches on it. Whether it is a regression depends on the provider. On
Ollama's `/v1/messages` the flag is dropped on the way in, so the refactor
changes nothing the model sees. The approximate step `tools-live+strict` flags
is then a true difference in what the agent sent that makes no difference to
the model.

## The agent mutations on a real model (M19)

**Question.** The mutation study's agent mutations were only ever caught or
missed by the scripted fake model. On a model that reads the prompt and the
tool results, which of the mutations the scripted model catches are
regressions, and by which check? One task, `lookup_latest_order_total` (*"I'm
Noah Ito, zip code 98187. How much did I pay for the order I most recently
placed?"*, τ-bench's answer 829.43), read-only, so the eight agent mutations
that can act on it: `skips_order_lookup`, `first_tool_only`,
`truncates_tool_results`, `max_turns_too_low`, `returns_empty_answer`,
`mismatches_tool_use_ids`, `strips_policy`, `truncates_history`. The other four
cannot: `wrong_write_argument` and `repeats_write_calls` act on writes (the
task makes none), `drops_error_flag` on a failed tool (none fails on the right
path), `unregisters_product_tool` removes a tool the task never needs.

**The comparison column.** The scripted model's `model-live` verdict on this
task alone (`eval/retail_error.py:scripted_verdicts`, the mutation study's
configuration on `replay.toml`'s scenario; it matches each mutation's
`failed_scenarios` in `eval/results/mutations.json`) catches 5 of the 8:
`skips_order_lookup`, `truncates_tool_results` (by `run_error:JSONDecodeError`:
the fake model parses the cut JSON), `max_turns_too_low`,
`returns_empty_answer`, `truncates_history`. It survives `first_tool_only`,
`mismatches_tool_use_ids` and `strips_policy`. Not the suite-wide column's 7 of
8, which counts all six tasks.

**Suite.** `examples/retail_agent/replay-qwen.toml`: `replay.toml`'s checks on
this task without the exact `tool_sequence` (a local model reads one order a
reply where Opus read three) and without the `$829.43` output pattern
(τ-bench's own check, `rule:matches_ground_truth`, already needs 829.43 in the
answer); the write checks, the budget and the M11 `identify` keys for the
three tools its recordings hold. Both changes were made before any run.

**Detection rule (PLAN.md, fixed before the run).** The arms resolve only if
the unmutated agent passes at least 8 of 10. Then a mutation is caught at 1 of
5 passes or fewer, survives at 4 of 5 or more, and is unresolved between. A
mutation that acts only on a reply with two `tool_use` blocks
(`first_tool_only`, `mismatches_tool_use_ids`) and never saw one never acted.
Fewer than 8 unmutated passes is written as "not resolvable on qwen2.5:7b for
this task".

### Recording

`make record-retail-lookup-qwen` (`eval/retail_error.py --task
lookup_latest_order_total --record`): 3 attempts, **none passed** (log:
`eval/results/m19_qwen_recording.json`). Attempts 1 and 2 (310.7 s with the
cold load, 52.2 s) looked the user up, skipped `get_user_details`, asked for an
order id they made up (`#W0000001`, `#W0023456`), and answered that no recent
order was found. Attempt 3 (56.1 s) ended on the second model call with
Ollama's HTTP 500 *"prediction aborted, token repeat limit reached"*. So the
arms ran on the fake-model recording
(`examples/retail_agent/cassettes/lookup_latest_order_total.jsonl`, the same
world: user, profile, order #W6729841), and `make eval` has no qwen lookup
cassette to replay. `make eval-retail-qwen-offline` (new, in `make eval`)
replays every qwen retail recording that exists in `full`, and names this one
as missing:

```
$ make eval-retail-qwen-offline
uv run python -m eval.retail_error --offline
PASS examples/retail_agent/replay-error.toml account_not_found_wrong_zip: 5 steps, 0 approximate
SKIP lookup_latest_order_total: no qwen recording in examples/retail_agent/cassettes/qwen
```

### The id probe and the window

**Does a mismatched `tool_use_id` reach the model?** As M18's flag probe:
after a warm-up, the same tool result with its matching id, with a mismatched
id of the same length, and with three words added (the control). Ollama's net
counts were 88, 88 and 92; the server's own `task.n_tokens` were 257, 257 and
261. **The id never reaches the model**: Ollama's `/v1/messages` builds qwen's
prompt from the order of the messages, not from the ids, so even a reply with
two calls would give `mismatches_tool_use_ids` nothing to change in the prompt.

**The window.** The server was started with `OLLAMA_CONTEXT_LENGTH=8192`; its
log (`n_ctx_slot = 8192` at load) was kept and read run by run
(`ServerLog`; the lines it reads, with every HTTP status, are committed as
`eval/results/m19_ollama_server_log.txt`): every request's whole prompt (`task.n_tokens`) and whether the
server truncated it. Over the 50 runs (166 requests), the largest prompt was
**3,708 tokens, 0 runs over 4,096, 0 truncated requests**. The bound M18 used
when it had no server counts (`prompt_bound`) reached 4,890, so it would have
flagged a run the server shows was 3,708: the bound is loose by a reply or
two, and the server's count is the one to trust. qwen read one order (or
none) in every run, as the plan expected, so the arms ran uncut at the
default window too.

### Arms

`make eval-retail-mutations-qwen` (`eval/retail_error.py --task
lookup_latest_order_total --runs 10 --mutant-runs 5 --server-log
/tmp/ollama.log`): the probe, then `model-live` under `fallback` with the
suite's identify keys, 10 runs of the unmutated agent and 5 of each mutation
(`eval/mutations/agents.py:<mutation>_ollama`: the mutant's loop with the
model from `REPLAY_OLLAMA_MODEL`; `_CopiedLoop`, which `strips_policy` and
`truncates_history` use, used to send `claude-opus-5-5` whatever the agent's
model). qwen2.5:7b (digest `845dbda0…`), Ollama 0.40.11, 4 vCPU AMD EPYC 7763,
CPU only, one run at a time, nothing else on the cores. **2,861 s of arms
(48 min 21 s for the command); runs 35.5 s / 40.3 s / 231.6 s min / median /
max; $0.** Raw report: `eval/results/m19_retail_mutations_qwen.json` (every
run's failed checks and cause, tool calls with status and result, the most
`tool_use` blocks in one reply, answer, seconds, each request's prompt size).

```
$ make eval-retail-mutations-qwen OLLAMA_LOG=/tmp/ollama.log
uv run python -m eval.retail_error --task lookup_latest_order_total \
	--runs 10 --mutant-runs 5 --server-log /tmp/ollama.log
id probe: the id never reaches the model ([254, 88, 88, 92])
arm unmutated: 10 model-live runs under fallback
  lookup_latest_order_total: 2/10 passed; 522.5 s
arm skips_order_lookup: 5 model-live runs under fallback
  lookup_latest_order_total: 0/5 passed; 213.4 s
arm first_tool_only: 5 model-live runs under fallback
  lookup_latest_order_total: 1/5 passed; 232.5 s
arm truncates_tool_results: 5 model-live runs under fallback
  lookup_latest_order_total: 0/5 passed; 212.0 s
arm max_turns_too_low: 5 model-live runs under fallback
  lookup_latest_order_total: 0/5 passed; 209.7 s
arm returns_empty_answer: 5 model-live runs under fallback
  lookup_latest_order_total: 0/5 passed; 238.1 s
arm mismatches_tool_use_ids: 5 model-live runs under fallback
  lookup_latest_order_total: 1/5 passed; 249.9 s
arm strips_policy: 5 model-live runs under fallback
  lookup_latest_order_total: 5/5 passed; 514.1 s
arm truncates_history: 5 model-live runs under fallback
  lookup_latest_order_total: 0/5 passed; 468.4 s
not resolvable on qwen2.5:7b for this task; wrote .../eval/results/m19_retail_mutations_qwen.json
	Elapsed (wall clock) time (h:mm:ss or m:ss): 48:21.37
```

The path columns are counted from the raw report's tool calls ("read the
profile": a `get_user_details` call; "guessed an order": a `get_order_details`
call on an id the recording never fetched, run for real under `fallback` and
failing with `order not found`).

| `model-live`, `fallback`, qwen2.5:7b | Passed | Failed checks | Run errors | Read the profile | Guessed an order | Most `tool_use` in a reply | Largest prompt | Scripted model on this task | Rule on qwen (not resolvable) |
|---|---|---|---|---|---|---|---|---|---|
| unmutated | 2 / 10 | `rule:matches_ground_truth` 8 | — | 2 | 8 | 1 | 3,489 | pass | — |
| `skips_order_lookup` | 0 / 5 | ground truth 5 | — | 1 | 0 (the mutant answers every read "lookup skipped") | 1 | 3,205 | caught | ≤ 1 / 5 |
| `first_tool_only` | 1 / 5 | ground truth 4 | — | 1 | 4 | 1 (never acted) | 3,489 | survives | ≤ 1 / 5 |
| `truncates_tool_results` | 0 / 5 | ground truth 5 | **none** | 1 | 5 | 1 | 3,083 | caught, by `run_error:JSONDecodeError` | ≤ 1 / 5 |
| `max_turns_too_low` | 0 / 5 | ground truth 5 | `RuntimeError` 1 | 1 | 4 | 1 | 3,154 | caught | ≤ 1 / 5 |
| `returns_empty_answer` | 0 / 5 | ground truth 5 | — | 1 | 4 | 1 | 3,489 | caught | ≤ 1 / 5 |
| `mismatches_tool_use_ids` | 1 / 5 | ground truth 4 | — | 1 | 4 | 1 (never acted) | 3,489 | survives | ≤ 1 / 5 |
| `strips_policy` | **5 / 5** | — | — | **5** | **0** | 1 | 2,285 | survives | ≥ 4 / 5 |
| `truncates_history` | 0 / 5 | ground truth 5 | — | 3 | 3 | 1 | 3,708 | caught | ≤ 1 / 5 |

Every failure in all 50 runs was `rule:matches_ground_truth` (the answer lacks
829.43); every cause is `model` (no approximate step, no unrecorded ending, no
stub); no run wrote or transferred.

**The rule: not resolvable on qwen2.5:7b for this task.** The unmutated agent
passed 2 of 10. In the 8 failures qwen did what the three recording attempts
did: it skipped `get_user_details`, asked for an order id it made up
(`#W0000001` three times, `#W0000023`, `#W12345678`, …), got `order not found`
from the real tool (`fallback`; the identify keys kept the recording's order
from being served to the wrong id), and asked the user for the order id. With
a baseline that fails 80% of the time, a mutant failing 5 of 5 says nothing
about the mutation. `first_tool_only` and `mismatches_tool_use_ids` show it:
they could not act in any run (no reply held two `tool_use` blocks), and the
rule still reads 1 of 5 as "caught". So no per-mutation verdict is stated, and
none of the agreement with the scripted model either. The rule's readings are
in the last column for the record only.

**Per-mutation hypotheses, answered as far as the runs allow:**

- **`first_tool_only`, `mismatches_tool_use_ids`: predicted to survive because
  they cannot act. They never acted.** The most `tool_use` blocks in any reply
  of the 50 runs was 1. Their 1 / 5 passes are the baseline's 2 / 10, not the
  mutation. The probe adds that `mismatches_tool_use_ids` could not change
  qwen's prompt on Ollama even with two calls (88 / 88 tokens).
- **`strips_policy`: predicted to survive (a read-only task's ground truth
  does not need the policy). It passed 5 of 5, against the unmutated agent's
  2 of 10** (two-sided Fisher exact p = 0.007, `scipy.stats.fisher_exact([[5, 0],
  [2, 8]])`). The direction is the surprise: without τ-bench's 2,800-token
  retail policy in front of the deployment note, qwen read the profile in all
  5 runs (1,662 / 1,728 / 1,950 / 2,285 tokens a turn) and answered $829.43
  every time; with it, it read the profile in 2 of 10 and guessed an order id
  in 8. On this model and task the "regression" improves the agent. Why is not
  measured here (the shorter prompt, or something the policy says); 5 runs on
  one task is a lead, not a finding about policies.
- **`truncates_tool_results`: predicted caught by the ground truth, not by a
  crash. All 5 failures are the ground-truth rule and none is a crash**
  (the fake model's catch on this task was `run_error:JSONDecodeError`). But
  the mutation acted in one run only: the cut matters only for results over
  120 characters, and 4 of 5 runs never got one (they guessed an order, whose
  error is short). In the one that read the profile, the cut ended it inside
  the address, 249 characters before the list of orders, and qwen asked for `#W20230101123456`, an id it made up:
  the mechanism the plan predicted (a model reading a cut string instead of a
  parser failing), seen once.
- **`truncates_history`: predicted caught because the last four messages lose
  the request.** Seen in the runs that got far enough: 3 of 5 read the profile,
  and then, with the task out of the window, asked *"which order you want to
  modify"* (2 runs) or read another order and listed a product's variants
  (1 run). The other 2 failed as the baseline does.
- **`max_turns_too_low`: predicted `run_error:RuntimeError`.** One run, the one
  on the policy's path (user, profile, order), hit the 3-turn limit with
  `RuntimeError`; the other 4 guessed an order and answered within 3 turns.
- **`skips_order_lookup`, `returns_empty_answer`: caught by construction or
  close to it.** The mutant answers every `get_order_details` with "lookup skipped",
  so no run saw an order; every `returns_empty_answer` answer was empty.

**What it says.** M19's question needs a baseline the model passes, and on
`lookup_latest_order_total` qwen2.5:7b with the retail policy does not: it
fails the unmutated task by inventing order ids instead of reading the
profile. That is the result, written as the plan allowed. What the 50 runs
still show: (1) Replay classified every run the same way the scripted study
does, by the ground-truth check, with no crash catches and no world steps;
(2) two mutations never had anything to act on, and a pass-rate rule would
have reported them as caught, so recording each run's most `tool_use` blocks
was what kept the rule honest; (3) the mutation the plan expected to be
harmless, `strips_policy`, is the only arm that does better than the
unmutated agent. A resolvable version of this question needs a task qwen
passes at least 8 of 10 (M18's error task did) or a model that does.

## To measure

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
- **The M18 error task on Claude.** `uv run replay record
  examples.retail_agent.agent:run --client anthropic -t "I'm Noah Ito, zip code
  98178. Please cancel my pending order #W6729841 because I no longer need it.
  Yes, I confirm." -o examples/retail_agent/cassettes/opus/account_not_found_wrong_zip.jsonl`
  records it once with `claude-opus-5-5`, for about $0.10 (the M2 cold-cache run
  of the six tasks cost $0.29, and this one stops after the failed lookup). Then
  `uv run replay ci examples/retail_agent/replay-error.toml --cassette-dir
  examples/retail_agent/cassettes/opus` checks it. The directory is its own, so
  `make bench-scale`, which reads `cassettes/live/`, is unchanged. Unlike
  Ollama's layer (M18), Anthropic's API documents `is_error` as telling the
  model the tool failed. So this is the recording on which `drops_error_flag`
  changes what a real model sees. `eval/retail_error.py` runs its arms on
  Ollama only; a Claude arm is not built.
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

API spend in M13: $0 (fake-model runs and committed recordings). Project
total: $3.326.
