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

Exit codes 0 and 1. Raw reports: `eval/results/m3_cost_gate_*.json`. Deltas
are against the baseline, from the budget check's details:

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

## Sequential gate simulation, baseline only (M3, offline)

The CI gate's operating characteristic before the SPRT exists. `agent_replay/stats.py`
(the SPRT, Wilson and kappa) is a specification with unimplemented bodies;
`agent_replay/stats_baseline.py` is the naive reference: run `max_runs` times
and pass on a majority (pass rate ≥ (p_fail + p_pass) / 2).

- **Method:** for each true pass rate p, 2000 simulated sequences of
  Bernoulli(p) run outcomes (seed 0) through a fresh gate; p_fail = 0.6,
  p_pass = 0.9, α = β = 0.05, max_runs = 20.
- **Command:** `make bench-sprt` (= `uv run python eval/sprt_sim.py --out eval/results/sprt_sim.json`), about 1.3 s.

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

- baseline: expected runs 20.00 at p_fail, 20.00 at p_pass; false-pass rate at p_fail 0.134, false-fail rate at p_pass 0.007
```

The fixed-20 majority gate passes 13.4% of scenarios whose true pass rate is
0.6 (exact binomial: P(X ≥ 15 | n = 20, p = 0.6) ≈ 0.126), well over the 5%
the SPRT targets, and always spends 20 runs. The SPRT's numbers join this
table when `stats.py` is implemented; the same command produces them.

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

## To measure

- **Judge-human agreement (kappa with bootstrap CI)** needs the golden set
  (D1) and `replay calibrate` (M6). The judge, its rubric and the calibration
  report format exist; the report that would let a criterion gate does not.
- **SPRT savings vs fixed k** (`make bench-sprt`, offline) once
  `agent_replay/stats.py` is implemented.
- The M2 live smoke test above can be re-run with the two commands in its
  section (about $0.30 each with `claude-opus-5-5` and a cold prompt cache);
  the M3 cost-gate run with the two commands in its section (about $1.00).

API spend in M3: $1.408 (cost gate $0.974; judge on recordings and mutants
$0.358; three runs of the judge's live test $0.076, two of them while the
rubric was being written, one of which failed `accurate_report` on a correct
but terse reply and led to the criterion's current wording). The committed
artefacts account for $1.329 of that (cost-gate reports $0.974, the 21 cached
verdicts $0.330, the live test above $0.025); the remaining $0.079 (two
re-judged duplicate mutant transcripts, two earlier live-test runs) comes from
the session log, not from a file in this repo. Project total: $2.113
(M1 $0.014, M2 $0.691, M3 $1.408).
