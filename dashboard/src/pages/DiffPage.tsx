import { useState } from "react";
import { Link, useSearchParams } from "react-router";
import { useDiff } from "../api/queries";
import type { DiffOp, DiffRow, DiffStep, RunSummary, TraceDiff } from "../api/types";
import { KindBadge, OP_STYLE, OpBadge, ResultBadge, StepStatusBadge } from "../components/Badge";
import { PageHeader } from "../components/Layout";
import { Code, Empty, ErrorState, Loading } from "../components/States";
import { button, card, link, table, td, th, thNum, thead } from "../components/table";
import { formatCost, formatFixed, formatMs, formatTimestamp, previewChange, shortId } from "../lib/format";

export const OPS: readonly DiffOp[] = ["same", "changed", "added", "removed"];

export function DiffPage() {
  const [params, setParams] = useSearchParams();
  const base = params.get("base") ?? "";
  const head = params.get("head") ?? "";
  const diff = useDiff(base, head);

  return (
    <>
      <PageHeader title="Diff">
        <DiffInputs
          key={`${base}|${head}`}
          base={base}
          head={head}
          onSubmit={(b, h) => { setParams({ base: b, head: h }); }}
        />
      </PageHeader>
      {base === "" || head === "" ? (
        <Empty title="Pick two runs to compare.">
          Enter a base and a head run id (a prefix of 6 or more characters works), or open a run and use
          <strong> Compare with</strong>. The same alignment is available on the command line as <Code>replay diff BASE HEAD</Code>.
        </Empty>
      ) : diff.isPending ? (
        <Loading label="Aligning traces…" />
      ) : diff.isError ? (
        <ErrorState error={diff.error} title="Could not diff these runs" />
      ) : (
        <DiffView base={diff.data.base} head={diff.data.head} diff={diff.data.diff} />
      )}
    </>
  );
}

function DiffInputs({ base, head, onSubmit }: { base: string; head: string; onSubmit: (base: string, head: string) => void }) {
  const [b, setB] = useState(base);
  const [h, setH] = useState(head);
  const input = "w-40 rounded border border-slate-300 bg-white px-2 py-1 font-mono text-xs focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500";
  return (
    <form
      className="flex flex-wrap items-center gap-2 text-sm"
      aria-label="Runs to compare"
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit(b.trim(), h.trim());
      }}
    >
      <label className="flex items-center gap-1.5">
        <span className="text-slate-600">Base</span>
        <input className={input} value={b} placeholder="run id" onChange={(e) => { setB(e.target.value); }} />
      </label>
      <button type="button" className={button} title="Swap base and head" aria-label="Swap base and head" onClick={() => { onSubmit(h.trim(), b.trim()); }}>
        ⇄
      </button>
      <label className="flex items-center gap-1.5">
        <span className="text-slate-600">Head</span>
        <input className={input} value={h} placeholder="run id" onChange={(e) => { setH(e.target.value); }} />
      </label>
      <button type="submit" className={button}>Compare</button>
    </form>
  );
}

function RunCard({ role, run }: { role: string; run: RunSummary }) {
  return (
    <div className={`${card} p-3`}>
      <div className="mb-1 flex items-center gap-2 text-xs font-semibold tracking-wide text-slate-500 uppercase">
        {role}
        <Link className={`${link} font-mono normal-case`} to={`/runs/${run.id}`}>{shortId(run.id)}</Link>
      </div>
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <ResultBadge passed={run.passed} />
        <span className="font-semibold">{run.scenario}</span>
        <span className="font-mono text-xs text-slate-600">{run.mode} · {run.policy}</span>
      </div>
      <div className="mt-1 font-mono text-xs text-slate-500">{run.agent} · {formatTimestamp(run.started_at)}</div>
    </div>
  );
}

function Stat({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="text-xs font-semibold tracking-wide text-slate-500 uppercase">{label}</div>
      <div className="num mt-0.5 text-sm text-slate-900">{children}</div>
    </div>
  );
}

export function DiffView({ base, head, diff }: { base: RunSummary; head: RunSummary; diff: TraceDiff }) {
  const counts = diff.counts;
  return (
    <>
      <div className="mb-4 grid grid-cols-1 gap-3 md:grid-cols-2">
        <RunCard role="Base" run={base} />
        <RunCard role="Head" run={head} />
      </div>
      <section aria-label="Diff summary" className={`${card} mb-4 flex flex-wrap gap-x-8 gap-y-3 p-3`}>
        <Stat label="Distance">{formatFixed(diff.distance, 2)}</Stat>
        <Stat label="Rows">
          <span className="flex flex-wrap gap-1.5">
            {OPS.map((op) => (
              <span key={op} data-testid={`count-${op}`} className="inline-flex items-center gap-1">
                <OpBadge op={op} />
                <span>{counts[op] ?? 0}</span>
              </span>
            ))}
          </span>
        </Stat>
        <Stat label="Cost">{formatCost(diff.base_cost_usd, diff.base_unpriced_calls)} → {formatCost(diff.head_cost_usd, diff.head_unpriced_calls)}</Stat>
        <Stat label="Latency">{formatMs(diff.base_latency_ms)} → {formatMs(diff.head_latency_ms)}</Stat>
      </section>
      {diff.rows.length === 0 ? (
        <Empty title="Both runs have no top-level steps." />
      ) : (
        <DiffTable rows={diff.rows} />
      )}
    </>
  );
}

function StepCell({ step }: { step: DiffStep | null | undefined }) {
  if (!step) return <span className="text-xs text-slate-400 italic">(none)</span>;
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <span className="num w-6 text-right text-xs text-slate-500">{step.index}</span>
      <KindBadge kind={step.kind} />
      <span className="font-mono text-xs">{step.name}</span>
      <StepStatusBadge status={step.status} />
      <span className="num text-xs text-slate-500">{formatMs(step.latency_ms)}</span>
      {step.cost_usd != null ? <span className="num text-xs text-slate-500">{formatCost(step.cost_usd)}</span> : null}
    </div>
  );
}

export function DiffTable({ rows }: { rows: DiffRow[] }) {
  return (
    <div className={`${card} overflow-x-auto`}>
      <table className={table} aria-label="Aligned steps">
        <thead className={thead}>
          <tr>
            <th className={th}>Op</th>
            <th className={`${th} w-[42%]`}>Base step</th>
            <th className={`${th} w-[42%]`}>Head step</th>
            <th className={thNum}>Row cost</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => {
            const changes = row.changes ?? [];
            return (
              <tr key={i} className={OP_STYLE[row.op].row} data-op={row.op} data-testid="diff-row">
                <td className={`${td} w-24`}><OpBadge op={row.op} /></td>
                <td className={td}><StepCell step={row.base} /></td>
                <td className={td}>
                  <StepCell step={row.head} />
                  {changes.length > 0 ? (
                    <ul className="mt-1 space-y-0.5 font-mono text-[11px] text-slate-800" aria-label="Field changes">
                      {changes.map((c) => {
                        const [before, after] = previewChange(c.base, c.head);
                        const absent = "(absent)";
                        return (
                          <li key={c.path} className="break-all" data-testid="field-change">
                            <span className="font-semibold">{c.path}</span>: <span className="text-red-800">{c.base_present ? before : absent}</span>{" "}
                            <span aria-label="becomes">→</span>{" "}
                            <span className="text-green-800">{c.head_present ? after : absent}</span>
                          </li>
                        );
                      })}
                    </ul>
                  ) : null}
                </td>
                <td className={`${td} num w-20 text-right text-xs text-slate-600`}>{formatFixed(row.cost, 2)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
