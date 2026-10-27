import { useSearchParams } from "react-router";
import { useFlakeReport, useFlakeReports, useFlakeTrend } from "../api/queries";
import type { FlakeReportRecord, FlakeSettings } from "../api/types";
import { ClassificationBadge, ResultBadge } from "../components/Badge";
import { PageHeader } from "../components/Layout";
import { Code, Empty, ErrorState, Loading } from "../components/States";
import { TrendChart } from "../components/TrendChart";
import { card, select, table, td, tdNum, th, thNum, thead } from "../components/table";
import { formatCost, formatFixed, formatInt, formatMs, formatPercent, formatTimestamp, shortId } from "../lib/format";
import { param } from "../lib/params";

/** The interval method each stats implementation uses. */
export function intervalMethod(statsImpl: string): string {
  if (statsImpl === "core") return "Wilson";
  if (statsImpl === "baseline") return "Wald";
  return statsImpl;
}

export function FlakyPage() {
  const [params, setParams] = useSearchParams();
  const reports = useFlakeReports();
  const chosen = param(params, "report") ?? reports.data?.[0]?.id;
  const report = useFlakeReport(chosen);

  if (reports.isPending) return <Loading label="Loading flake reports…" />;
  if (reports.isError) return <ErrorState error={reports.error} title="Could not load flake reports" />;
  if (reports.data.length === 0) {
    return (
      <>
        <PageHeader title="Flakiness" />
        <Empty title="No flake reports stored.">
          Run <Code>replay flake replay.toml --db $REPLAY_DB_URL</Code> to re-run each scenario and store its pass rate.
        </Empty>
      </>
    );
  }
  const set = (key: string, value: string) => {
    const next = new URLSearchParams(params);
    next.set(key, value);
    // A scenario chosen under one report may not exist in another.
    if (key === "report") next.delete("scenario");
    setParams(next);
  };
  return (
    <>
      <PageHeader title="Flakiness">
        <label className="flex items-center gap-1.5 text-sm">
          <span className="text-slate-600">Report</span>
          <select className={select} value={chosen ?? ""} onChange={(e) => { set("report", e.target.value); }}>
            {reports.data.map((r, i) => (
              <option key={r.id} value={r.id}>
                {formatTimestamp(r.started_at)} · {shortId(r.id)} · {r.passed ? "passed" : "failed"}
                {i === 0 ? " (latest)" : ""}
              </option>
            ))}
          </select>
        </label>
      </PageHeader>
      {report.isPending ? (
        <Loading label="Loading report…" />
      ) : report.isError ? (
        <ErrorState error={report.error} title="Could not load flake report" />
      ) : (
        <Report report={report.data} scenario={param(params, "scenario") ?? null} onScenario={(s) => { set("scenario", s); }} />
      )}
    </>
  );
}

function settingsText(s: FlakeSettings): [string, string][] {
  return [
    ["gate", s.gate],
    ["max runs", String(s.max_runs)],
    ["min pass rate", formatFixed(s.min_pass_rate, 2)],
    ["confidence", formatFixed(s.confidence, 2)],
    ["p_pass / p_fail", `${formatFixed(s.p_pass, 2)} / ${formatFixed(s.p_fail, 2)}`],
    ["alpha / beta", `${formatFixed(s.alpha, 2)} / ${formatFixed(s.beta, 2)}`],
    ["seed", String(s.seed)],
    ["concurrency", s.concurrency == null ? "default" : String(s.concurrency)],
  ];
}

function Report({ report, scenario, onScenario }: { report: FlakeReportRecord; scenario: string | null; onScenario: (s: string) => void }) {
  const method = intervalMethod(report.stats_impl);
  const scenarios = report.scenarios;
  const selected = scenario ?? scenarios.find((s) => s.classification !== "passing")?.name ?? scenarios[0]?.name;
  return (
    <>
      <section className={`${card} mb-4 p-4`} aria-label="Report summary">
        <div className="mb-3 flex flex-wrap items-center gap-3 text-sm">
          <ResultBadge passed={report.passed} />
          <span className="font-mono text-xs">{report.suite}</span>
          <span className="num text-slate-600">
            {formatTimestamp(report.started_at)} UTC · {formatInt(report.scenario_count)} scenarios · {formatInt(report.flaky_count)} flaky ·{" "}
            {formatInt(report.failed_count)} failed · {formatInt(report.run_count)} runs · {formatCost(report.cost_usd, report.unpriced_calls)} · {formatMs(report.duration_ms)}
          </span>
        </div>
        <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-3 lg:grid-cols-5">
          <div>
            <dt className="text-xs font-semibold tracking-wide text-slate-500 uppercase">stats_impl</dt>
            <dd className="font-mono text-xs">{report.stats_impl} <span className="font-sans text-slate-600">({method} intervals)</span></dd>
          </div>
          {settingsText(report.settings).map(([k, v]) => (
            <div key={k}>
              <dt className="text-xs font-semibold tracking-wide text-slate-500 uppercase">{k}</dt>
              <dd className="num font-mono text-xs">{v}</dd>
            </div>
          ))}
        </dl>
      </section>

      {scenarios.length === 0 ? (
        <Empty title="This report has no scenarios." />
      ) : (
        <div className={`${card} mb-5 overflow-x-auto`}>
          <table className={table}>
            <thead className={thead}>
              <tr>
                <th className={th}>Scenario</th>
                <th className={th}>Classification</th>
                <th className={th}>Decision</th>
                <th className={thNum}>Passes</th>
                <th className={thNum}>Pass rate</th>
                <th className={thNum}>{method} interval</th>
                <th className={th}>Mode</th>
                <th className={th}>Top failures</th>
              </tr>
            </thead>
            <tbody>
              {scenarios.map((s) => {
                const failures = Object.entries(s.failures).sort((a, b) => b[1] - a[1]).slice(0, 2);
                return (
                  <tr key={s.name} className={s.name === selected ? "bg-blue-50" : undefined}>
                    <td className={td}>
                      <button type="button" className="text-left font-medium text-blue-700 hover:underline" onClick={() => { onScenario(s.name); }}>
                        {s.name}
                      </button>
                      {s.error ? <div className="font-mono text-[11px] text-red-800">{s.error.type}: {s.error.message}</div> : null}
                    </td>
                    <td className={td}><ClassificationBadge value={s.classification} /></td>
                    <td className={`${td} font-mono text-xs`}>{s.decision}</td>
                    <td className={tdNum}>{s.passes}/{s.runs}</td>
                    <td className={tdNum}>{formatPercent(s.pass_rate)}</td>
                    <td className={tdNum}>[{formatFixed(s.ci_low)}, {formatFixed(s.ci_high)}]</td>
                    <td className={`${td} font-mono text-xs`}>{s.mode}</td>
                    <td className={`${td} text-xs text-slate-700`}>
                      {failures.length === 0 ? "—" : failures.map(([k, n]) => `${k} (${n})`).join(", ")}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      {selected ? <Trend scenario={selected} names={scenarios.map((s) => s.name)} onScenario={onScenario} /> : null}
    </>
  );
}

function Trend({ scenario, names, onScenario }: { scenario: string; names: string[]; onScenario: (s: string) => void }) {
  const trend = useFlakeTrend(scenario);
  return (
    <section className={`${card} p-4`} aria-labelledby="trend-title">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-3">
        <h2 id="trend-title" className="text-base font-semibold">Pass rate over reports</h2>
        <label className="flex items-center gap-1.5 text-sm">
          <span className="text-slate-600">Scenario</span>
          <select className={select} value={scenario} onChange={(e) => { onScenario(e.target.value); }}>
            {names.map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </label>
      </div>
      {trend.isPending ? (
        <Loading label="Loading trend…" />
      ) : trend.isError ? (
        <ErrorState error={trend.error} title="Could not load trend" />
      ) : trend.data.length === 0 ? (
        <Empty title={`No flake history for ${scenario}.`} />
      ) : (
        <>
          <p className="mb-2 text-xs text-slate-600">
            Line: pass rate per report. Shaded band: the report's confidence interval (method per report's <Code>stats_impl</Code>).
          </p>
          <TrendChart points={trend.data} scenario={scenario} />
          <details className="mt-2 text-sm">
            <summary className="cursor-pointer text-slate-700">Data table</summary>
            <table className={`${table} mt-2`}>
              <thead className={thead}>
                <tr>
                  <th className={th}>Report time (UTC)</th>
                  <th className={th}>Classification</th>
                  <th className={thNum}>Pass rate</th>
                  <th className={thNum}>Interval</th>
                  <th className={th}>stats_impl</th>
                </tr>
              </thead>
              <tbody>
                {trend.data.map((p) => (
                  <tr key={p.report_id}>
                    <td className={`${td} num`}>{formatTimestamp(p.started_at)}</td>
                    <td className={td}><ClassificationBadge value={p.classification} /></td>
                    <td className={tdNum}>{p.passes}/{p.runs} ({formatPercent(p.pass_rate)})</td>
                    <td className={tdNum}>[{formatFixed(p.ci_low)}, {formatFixed(p.ci_high)}]</td>
                    <td className={`${td} font-mono text-xs`}>{p.stats_impl} ({intervalMethod(p.stats_impl)})</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </details>
        </>
      )}
    </section>
  );
}
