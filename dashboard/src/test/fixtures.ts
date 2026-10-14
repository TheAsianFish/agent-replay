import type {
  CalibrationSummary,
  CalibrationView,
  FlakePoint,
  FlakeReportRecord,
  FlakeReportSummary,
  Health,
  RunDiff,
  RunPage,
  RunRecord,
  RunSummary,
  StepRecord,
  SuiteSummary,
} from "../api/types";

export const health: Health = { status: "ok", version: "0.1.0", backend: "sqlite", revision: "0002" };

export function runSummary(overrides: Partial<RunSummary> = {}): RunSummary {
  return {
    id: "aaaaaaaa11112222333344445555666",
    suite_id: "suite000111222333",
    scenario: "modify_address_suite",
    cassette_path: "examples/retail_agent/cassettes/modify_address_suite.jsonl",
    cassette_id: "cas1",
    agent: "examples.retail_agent.agent:run",
    mode: "full",
    policy: "strict",
    task: "Change my address.",
    passed: true,
    error_type: null,
    error_message: null,
    cost_usd: 0.0123,
    tokens: 18202,
    step_count: 3,
    approximate_count: 0,
    unconsumed_count: 0,
    started_at: "2026-10-10T21:48:22.395521Z",
    duration_ms: 34.791,
    ...overrides,
  };
}

export const passingRun = runSummary();
export const failingRun = runSummary({
  id: "bbbbbbbb11112222333344445555666",
  passed: false,
  approximate_count: 2,
  agent: "examples.retail_agent.mutants:skips_order_lookup",
});

export const runPage: RunPage = { total: 2, items: [passingRun, failingRun] };

function step(overrides: Partial<StepRecord>): StepRecord {
  return {
    index: 0,
    kind: "model",
    name: "anthropic.messages.create",
    status: "exact",
    fingerprint: "sha256:abc",
    request: { model: "claude-opus-5-5", messages: [{ role: "user", content: "hello" }] },
    response: { id: "msg_1", content: [{ type: "text", text: "hi" }] },
    error: null,
    input_tokens: 3134,
    output_tokens: 43,
    cache_read_input_tokens: 0,
    cache_creation_input_tokens: 0,
    cost_usd: 0.0021,
    latency_ms: 800,
    span_id: "s0",
    parent_span_id: null,
    envelope_seq: 0,
    ...overrides,
  };
}

export const runRecord: RunRecord = {
  ...failingRun,
  output: "A human agent will take it from here.",
  error_type: "UnrecordedCallError",
  error_message: "the agent made a tool call that the cassette never recorded",
  checks: [
    { position: 1, check: "required_tools", passed: true, message: "all 1 required tool(s) were called", details: {}, advisory: false },
    { position: 0, check: "tool_sequence", passed: false, message: "tool call #3 differs", details: {}, advisory: false },
    { position: 2, check: "judge:tone", passed: false, message: "tone is curt", details: {}, advisory: true },
  ],
  steps: [
    step({ index: 0 }),
    step({
      index: 1,
      kind: "tool",
      name: "find_user_id_by_email",
      status: "approximate",
      request: { email: "mia@example.com", note: "x".repeat(400) },
      response: "mia_garcia_4516",
      input_tokens: null,
      output_tokens: null,
      cost_usd: null,
      latency_ms: 200,
      span_id: "s1",
    }),
    step({ index: 2, kind: "tool", name: "nested_lookup", status: "unrecorded", parent_span_id: "s1", span_id: "s2", latency_ms: null, error: { type: "KeyError" } }),
  ],
};

export const diff: RunDiff = {
  base: passingRun,
  head: failingRun,
  diff: {
    base_label: null,
    head_label: null,
    distance: 2.5,
    counts: { same: 1, changed: 1, added: 1, removed: 1 },
    base_cost_usd: 0.01,
    head_cost_usd: 0.02,
    base_latency_ms: 1200,
    head_latency_ms: 900,
    rows: [
      {
        op: "same",
        cost: 0,
        changes: [],
        base: { index: 0, kind: "model", name: "anthropic.messages.create", status: "exact", request: {}, response: {}, error: null, latency_ms: 1, cost_usd: null },
        head: { index: 0, kind: "model", name: "anthropic.messages.create", status: "exact", request: {}, response: {}, error: null, latency_ms: 1, cost_usd: null },
      },
      {
        op: "changed",
        cost: 0.5,
        changes: [{ path: "request.email", base: "a@example.com", head: "b@example.com", base_present: true, head_present: true }],
        base: { index: 1, kind: "tool", name: "find_user_id_by_email", status: "exact", request: {}, response: {}, error: null, latency_ms: 1, cost_usd: null },
        head: { index: 1, kind: "tool", name: "find_user_id_by_email", status: "approximate", request: {}, response: {}, error: null, latency_ms: 1, cost_usd: null },
      },
      {
        op: "removed",
        cost: 1,
        changes: [],
        base: { index: 2, kind: "tool", name: "get_order_details", status: "exact", request: {}, response: {}, error: null, latency_ms: 1, cost_usd: null },
        head: null,
      },
      {
        op: "added",
        cost: 1,
        changes: [],
        base: null,
        head: { index: 2, kind: "tool", name: "transfer_to_human_agents", status: "unrecorded", request: {}, response: {}, error: null, latency_ms: null, cost_usd: null },
      },
    ],
  },
};

export const suites: SuiteSummary[] = [
  {
    id: "suite000111222333",
    started_at: "2026-10-10T21:48:20.000Z",
    finished_at: "2026-10-10T21:48:22.000Z",
    config_path: "examples/retail_agent/replay.toml",
    git_sha: "5f7113cc593bd0a12566ef9854cd7cffc1842840",
    passed: false,
    scenario_count: 6,
    failed_count: 1,
    run_count: 48,
  },
];

export const flakeSummaries: FlakeReportSummary[] = [
  {
    id: "flake1aaaabbbbcccc",
    suite_id: "suite000111222333",
    suite: "examples/retail_agent/replay.toml",
    stats_impl: "baseline",
    passed: false,
    started_at: "2026-10-10T21:48:16Z",
    duration_ms: 2400,
    git_sha: null,
    scenario_count: 2,
    failed_count: 1,
    flaky_count: 1,
    run_count: 16,
    cost_usd: 0,
  },
];

export const flakeReport: FlakeReportRecord = {
  ...(flakeSummaries[0] as FlakeReportSummary),
  settings: {
    gate: "sprt",
    max_runs: 8,
    min_pass_rate: 0.8,
    confidence: 0.95,
    p_pass: 0.9,
    p_fail: 0.6,
    alpha: 0.05,
    beta: 0.05,
    seed: 0,
    concurrency: null,
    runs_per_minute: null,
  },
  scenarios: [
    {
      name: "cancel_pending_laptop",
      cassette: "c.jsonl",
      agent: "a:run",
      mode: "model-live",
      policy: "strict",
      classification: "flaky",
      decision: "fail",
      runs: 8,
      passes: 5,
      executed: 8,
      concurrency: 1,
      pass_rate: 0.625,
      ci_low: 0.29,
      ci_high: 0.96,
      cost_usd: 0,
      error: null,
      failures: { required_tools: 3 },
      outcomes: [],
    },
    {
      name: "modify_address_suite",
      cassette: "m.jsonl",
      agent: "a:run",
      mode: "model-live",
      policy: "strict",
      classification: "passing",
      decision: "pass",
      runs: 8,
      passes: 8,
      executed: 8,
      concurrency: 1,
      pass_rate: 1,
      ci_low: 1,
      ci_high: 1,
      cost_usd: 0,
      error: null,
      failures: {},
      outcomes: [],
    },
  ],
};

export const trend: FlakePoint[] = [
  { report_id: "r1", started_at: "2026-10-09T10:00:00Z", stats_impl: "baseline", git_sha: null, name: "cancel_pending_laptop", classification: "passing", decision: "pass", runs: 8, passes: 8, pass_rate: 1, ci_low: 1, ci_high: 1 },
  { report_id: "r2", started_at: "2026-10-10T10:00:00Z", stats_impl: "baseline", git_sha: null, name: "cancel_pending_laptop", classification: "flaky", decision: "fail", runs: 8, passes: 5, pass_rate: 0.625, ci_low: 0.29, ci_high: 0.96 },
];

export const calibrations: CalibrationSummary[] = [
  {
    id: "cal1aaaabbbbcccc",
    stored_at: "2026-10-10T00:00:00Z",
    rubric: "retail_support",
    rubric_fingerprint: "sha256:3f9a",
    judge_id: "anthropic:claude-opus-5-5",
    prompt_version: "judge-v1",
    created_at: "2026-10-01T12:00:00Z",
    n_items: 120,
    reference: "human",
    repeats: 1,
    stats_impl: null,
  },
  {
    id: "cal2ddddeeeeffff",
    stored_at: "2026-10-12T00:00:00Z",
    rubric: "retail_support_objective",
    rubric_fingerprint: "sha256:3f9a",
    judge_id: "anthropic:claude-opus-5-5",
    prompt_version: "judge-v2",
    created_at: "2026-10-12T09:30:00Z",
    n_items: 24,
    reference: "objective",
    repeats: 3,
    stats_impl: "baseline",
  },
];

/**
 * A schema 1 report as the API returns it: the server fills in the schema 2 defaults, and because
 * such a report does not name its statistics it is blocked, so no criterion gates whatever its kappa.
 */
export const calibrationView: CalibrationView = {
  min_kappa: 0.6,
  min_items: 20,
  blocked:
    "the calibration report was computed with statistics that it does not name; recalibrate with REPLAY_STATS_IMPL=core",
  gates: { resolves_request: { gating: false }, tone: { gating: false } },
  record: {
    id: "cal1aaaabbbbcccc",
    stored_at: "2026-10-10T00:00:00Z",
    report: {
      schema_version: 1,
      rubric: "retail_support",
      rubric_fingerprint: "sha256:3f9a",
      judge_id: "anthropic:claude-opus-5-5",
      prompt_version: "judge-v1",
      created_at: "2026-10-01T12:00:00Z",
      n_items: 120,
      method: "Cohen's kappa, 95% bootstrap CI (2000 resamples).",
      criteria: {
        resolves_request: { kappa: 0.82, ci_low: 0.71, ci_high: 0.9, agreement: 0.92, n: 120 },
        tone: { kappa: 0.31, ci_low: 0.12, ci_high: 0.5, agreement: 0.7, n: 118 },
      },
      reference: "human",
      repeats: 1,
      probes: [],
    },
  },
};

/** A schema 2 report: 24 human-labelled retail runs, the judge run three times, all three bias probes. */
export const calibrationViewV2: CalibrationView = {
  min_kappa: 0.6,
  min_items: 20,
  blocked: null,
  gates: {
    resolves_request: { gating: true },
    tone: { gating: false },
    follows_policy: { gating: false },
    // The whole-run entry is not a rubric criterion and never gates, however good its numbers.
    _overall: { gating: false },
  },
  record: {
    id: "cal2ddddeeeeffff",
    stored_at: "2026-10-12T00:00:00Z",
    report: {
      schema_version: 2,
      rubric: "retail_support",
      rubric_fingerprint: "sha256:3f9a",
      judge_id: "anthropic:claude-opus-5-5",
      prompt_version: "judge-v2",
      created_at: "2026-10-12T09:30:00Z",
      n_items: 24,
      method: "Cohen's kappa, 95% bootstrap CI (2000 resamples); judge verdicts by majority over 3 runs.",
      reference: "human",
      labeller: "reviewer-a",
      stats_impl: "core",
      golden_set: "eval/golden/retail",
      golden_fingerprint: "sha256:9c41e07ab2d35f60",
      repeats: 3,
      judge_calls: 18,
      cached_calls: 270,
      cost_usd: 0.4312,
      criteria: {
        resolves_request: {
          kappa: 0.8222,
          ci_low: 0.63,
          ci_high: 1,
          agreement: 22 / 24,
          n: 24,
          confusion: { both_pass: 14, judge_pass_reference_fail: 1, judge_fail_reference_pass: 1, both_fail: 8 },
          majority_label: "pass",
          majority_agreement: 15 / 24,
          self_consistency: { runs: 3, n: 24, unanimous: 23 / 24, mean_pairwise_kappa: 0.91 },
        },
        tone: {
          kappa: 0.25,
          ci_low: -0.18,
          ci_high: 0.63,
          agreement: 0.75,
          n: 24,
          confusion: { both_pass: 16, judge_pass_reference_fail: 4, judge_fail_reference_pass: 2, both_fail: 2 },
          majority_label: "pass",
          majority_agreement: 0.75,
          self_consistency: { runs: 3, n: 24, unanimous: 0.75, mean_pairwise_kappa: 0.52 },
        },
        follows_policy: {
          kappa: null,
          ci_low: null,
          ci_high: null,
          agreement: 1,
          n: 24,
          confusion: { both_pass: 24, judge_pass_reference_fail: 0, judge_fail_reference_pass: 0, both_fail: 0 },
          majority_label: "pass",
          majority_agreement: 1,
          self_consistency: { runs: 3, n: 24, unanimous: 1, mean_pairwise_kappa: null },
        },
        _overall: {
          kappa: 0.6444,
          ci_low: 0.31,
          ci_high: 0.92,
          agreement: 20 / 24,
          n: 24,
          confusion: { both_pass: 13, judge_pass_reference_fail: 2, judge_fail_reference_pass: 2, both_fail: 7 },
          majority_label: "pass",
          majority_agreement: 15 / 24,
          self_consistency: { runs: 3, n: 24, unanimous: 0.875, mean_pairwise_kappa: 0.8 },
        },
      },
      probes: [
        {
          probe: "position",
          description: "Criteria listed in reverse order; a verdict that changes depends on where the criterion sits in the prompt.",
          flips: {
            resolves_request: { n: 24, flips: 1, to_pass: 0, to_fail: 1, rate: 1 / 24, ci_low: 0.0074, ci_high: 0.2024 },
            tone: { n: 24, flips: 5, to_pass: 4, to_fail: 1, rate: 5 / 24, ci_low: 0.0924, ci_high: 0.4047 },
          },
        },
        {
          probe: "verbosity",
          description: "The final answer padded with filler that adds no information; a verdict that changes rewards length.",
          flips: {
            tone: { n: 24, flips: 3, to_pass: 3, to_fail: 0, rate: 0.125, ci_low: 0.0434, ci_high: 0.3101 },
            _overall: { n: 24, flips: 2, to_pass: 2, to_fail: 0, rate: 2 / 24, ci_low: 0.0232, ci_high: 0.2584 },
          },
        },
        {
          probe: "self_preference",
          description: "Judge leniency on runs its own model produced against runs from other models.",
          groups: {
            resolves_request: {
              same_model: { n: 12, judge_pass_rate: 8 / 12, reference_pass_rate: 7 / 12, false_pass_rate: 0.2, reference_fail_n: 5 },
              other_model: { n: 12, judge_pass_rate: 7 / 12, reference_pass_rate: 8 / 12, false_pass_rate: 0, reference_fail_n: 4 },
            },
          },
        },
      ],
    },
  },
};
