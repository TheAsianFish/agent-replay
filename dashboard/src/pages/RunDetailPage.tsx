import { Fragment, useState } from "react";
import { Link, useNavigate, useParams } from "react-router";
import { useRun, useRuns } from "../api/queries";
import type { CheckRecord, RunRecord, StepRecord } from "../api/types";
import { CheckBadge, KindBadge, ResultBadge, StepStatusBadge } from "../components/Badge";
import { JsonView } from "../components/JsonView";
import { LatencyBar } from "../components/LatencyBar";
import { PageHeader } from "../components/Layout";
import { Empty, ErrorState, Loading } from "../components/States";
import { button, card, link, select, table, td, tdNum, th, thNum, thead } from "../components/table";
import { formatCost, formatInt, formatMs, formatTimestamp, shortId } from "../lib/format";
import { maxLatency, traceRows } from "../lib/trace";

export function RunDetailPage() {
  const { id = "" } = useParams();
  const run = useRun(id);
  if (run.isPending) return <Loading label="Loading run…" />;
  if (run.isError) return <ErrorState error={run.error} title={`Could not load run ${id}`} />;
  return <RunDetail run={run.data} />;
}

function Field({ label, children, mono = false }: { label: string; children: React.ReactNode; mono?: boolean }) {
  return (
    <div className="min-w-0">
      <dt className="text-xs font-semibold tracking-wide text-slate-500 uppercase">{label}</dt>
      <dd className={`mt-0.5 break-words text-slate-900 ${mono ? "font-mono text-xs" : "text-sm"}`}>{children}</dd>
    </div>
  );
}

function RunDetail({ run }: { run: RunRecord }) {
  return (
    <>
      <PageHeader title={`Run ${shortId(run.id)}`}>
        <CompareWith run={run} />
      </PageHeader>
      <section className={`${card} mb-5 p-4`} aria-label="Summary">
        <div className="mb-3 flex flex-wrap items-center gap-3">
          <ResultBadge passed={run.passed} />
          <span className="text-lg font-semibold">{run.scenario}</span>
          <span className="num text-sm text-slate-600">
            {formatTimestamp(run.started_at)} UTC · {formatInt(run.step_count)} steps · {formatInt(run.approximate_count)} approximate ·{" "}
            {formatInt(run.unconsumed_count)} unconsumed · {formatInt(run.tokens)} tokens · {formatCost(run.cost_usd, run.unpriced_calls)} · {formatMs(run.duration_ms)}
          </span>
        </div>
        <dl className="grid grid-cols-1 gap-x-6 gap-y-3 sm:grid-cols-2 lg:grid-cols-4">
          <Field label="Mode" mono>{run.mode}</Field>
          <Field label="Policy" mono>{run.policy}</Field>
          <Field label="Agent" mono>{run.agent}</Field>
          <Field label="Cassette" mono>{run.cassette_path}</Field>
          <Field label="Suite" mono>
            {run.suite_id ? (
              <Link className={link} to={`/?suite_id=${run.suite_id}`}>{shortId(run.suite_id, 12)}</Link>
            ) : "—"}
          </Field>
          <Field label="Run id" mono>{run.id}</Field>
        </dl>
        <div className="mt-3">
          <Field label="Task">{run.task}</Field>
        </div>
        {run.error_type || run.error_message ? (
          <div role="alert" className="mt-3 rounded border border-red-300 bg-red-50 px-3 py-2 text-sm text-red-900">
            <span className="font-semibold"><span aria-hidden>✗ </span>{run.error_type ?? "Error"}</span>
            {run.error_message ? <span className="font-mono text-xs">: {run.error_message}</span> : null}
          </div>
        ) : null}
      </section>

      <Checks checks={run.checks} />
      <Trace steps={run.steps} />
    </>
  );
}

function CompareWith({ run }: { run: RunRecord }) {
  const navigate = useNavigate();
  const others = useRuns({ limit: 100, offset: 0, scenario: run.scenario });
  const [choice, setChoice] = useState("");
  // Most useful first: the same recording, then the same mode, then the opposite result
  // (a passing run for a failing one).
  const rank = (r: { cassette_path: string; mode: string; passed: boolean }) =>
    (r.cassette_path === run.cassette_path ? 0 : 4) + (r.mode === run.mode ? 0 : 2) + (r.passed !== run.passed ? 0 : 1);
  const options = (others.data?.items ?? []).filter((r) => r.id !== run.id).sort((a, b) => rank(a) - rank(b));
  return (
    <form
      className="flex items-center gap-2 text-sm"
      aria-label="Compare with"
      onSubmit={(e) => {
        e.preventDefault();
        const base = choice || options[0]?.id;
        if (base) void navigate(`/diff?base=${base}&head=${run.id}`);
      }}
    >
      <label className="flex items-center gap-1.5">
        <span className="text-slate-600">Compare with</span>
        <select
          className={`${select} max-w-[28rem]`}
          value={choice}
          disabled={options.length === 0}
          onChange={(e) => { setChoice(e.target.value); }}
        >
          {options.length === 0 ? (
            <option value="">{others.isPending ? "loading…" : "no other runs of this scenario"}</option>
          ) : null}
          {options.map((r) => (
            <option key={r.id} value={r.id}>
              {shortId(r.id)} · {r.mode} · {r.passed ? "pass" : "fail"} · {formatTimestamp(r.started_at)}
            </option>
          ))}
        </select>
      </label>
      <button type="submit" className={button} disabled={options.length === 0}>
        Diff
      </button>
    </form>
  );
}

function Checks({ checks }: { checks: CheckRecord[] }) {
  const sorted = [...checks].sort((a, b) => a.position - b.position);
  return (
    <section className="mb-5" aria-labelledby="checks-title">
      <h2 id="checks-title" className="mb-2 text-base font-semibold">
        Checks <span className="num font-normal text-slate-500">({sorted.filter((c) => c.passed).length}/{sorted.length} passed)</span>
      </h2>
      {sorted.length === 0 ? (
        <Empty title="No checks ran for this run." />
      ) : (
        <ul className={`${card} divide-y divide-slate-100`}>
          {sorted.map((c) => (
            <li key={c.position} className="flex flex-wrap items-start gap-3 px-3 py-2 text-sm">
              <span className="w-28 shrink-0"><CheckBadge passed={c.passed} advisory={c.advisory} /></span>
              <span className="w-56 shrink-0 font-mono text-xs leading-5 font-semibold">{c.check}</span>
              <span className="min-w-0 flex-1 text-slate-800">{c.message}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function tokensText(step: StepRecord): string {
  if (step.input_tokens == null && step.output_tokens == null) return "—";
  const parts = [`${formatInt(step.input_tokens ?? 0)} in`, `${formatInt(step.output_tokens ?? 0)} out`];
  if (step.cache_read_input_tokens) parts.push(`${formatInt(step.cache_read_input_tokens)} cached`);
  return parts.join(" / ");
}

function Trace({ steps }: { steps: StepRecord[] }) {
  const rows = traceRows(steps);
  const max = maxLatency(steps);
  const [open, setOpen] = useState<ReadonlySet<number>>(new Set());
  const toggle = (index: number) => {
    setOpen((prev) => {
      const next = new Set(prev);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  };
  return (
    <section aria-labelledby="trace-title">
      <div className="mb-2 flex items-baseline justify-between gap-3">
        <h2 id="trace-title" className="text-base font-semibold">
          Trace <span className="num font-normal text-slate-500">({rows.length} steps)</span>
        </h2>
        {rows.length > 0 ? (
          <span className="flex gap-2">
            <button type="button" className={button} onClick={() => { setOpen(new Set(steps.map((s) => s.index))); }}>
              Expand all
            </button>
            <button type="button" className={button} onClick={() => { setOpen(new Set()); }}>
              Collapse all
            </button>
          </span>
        ) : null}
      </div>
      {rows.length === 0 ? (
        <Empty title="This run recorded no steps." />
      ) : (
        <div className={`${card} overflow-x-auto`}>
          <table className={table} aria-label="Trace steps">
            <thead className={thead}>
              <tr>
                <th className={thNum}>#</th>
                <th className={th}>Kind</th>
                <th className={th}>Name</th>
                <th className={th}>Status</th>
                <th className={th}>Latency</th>
                <th className={thNum}>Tokens</th>
                <th className={thNum}>Cost</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(({ step, depth }) => {
                const isOpen = open.has(step.index);
                return (
                  <Fragment key={step.index}>
                    <tr className={`hover:bg-slate-50 ${isOpen ? "bg-slate-50" : ""}`} data-testid="trace-step">
                      <td className={`${tdNum} text-slate-500`}>{step.index}</td>
                      <td className={td}><KindBadge kind={step.kind} /></td>
                      <td className={td}>
                        <div style={{ paddingLeft: `${depth * 1.25}rem` }} className="flex items-center gap-1">
                          {depth > 0 ? <span className="text-slate-400" aria-label={`nested level ${depth}`}>└</span> : null}
                          <button
                            type="button"
                            aria-expanded={isOpen}
                            aria-controls={`step-${step.index}`}
                            onClick={() => { toggle(step.index); }}
                            className="flex items-center gap-1 text-left font-mono text-xs text-blue-800 hover:underline"
                          >
                            <span aria-hidden className="w-3 text-slate-500">{isOpen ? "▾" : "▸"}</span>
                            {step.name}
                          </button>
                          {step.error != null ? <span className="text-xs font-semibold text-red-700">(error)</span> : null}
                        </div>
                      </td>
                      <td className={td}><StepStatusBadge status={step.status} /></td>
                      <td className={td}><LatencyBar value={step.latency_ms} max={max} /></td>
                      <td className={`${tdNum} text-xs`}>{tokensText(step)}</td>
                      <td className={tdNum}>{formatCost(step.cost_usd)}</td>
                    </tr>
                    {isOpen ? (
                      <tr id={`step-${step.index}`}>
                        <td colSpan={7} className="border-b border-slate-200 bg-slate-50 px-3 py-3">
                          <div className="mb-2 font-mono text-[11px] text-slate-500">
                            fingerprint {step.fingerprint}
                            {step.span_id ? ` · span ${step.span_id}` : ""}
                            {step.parent_span_id ? ` · parent ${step.parent_span_id}` : ""}
                          </div>
                          <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
                            <JsonView label="Request" value={step.request} />
                            <JsonView label="Response" value={step.response} />
                            {step.error != null ? <JsonView label="Error" value={step.error} /> : null}
                          </div>
                        </td>
                      </tr>
                    ) : null}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
