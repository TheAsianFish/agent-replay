import { Link, useNavigate, useSearchParams } from "react-router";
import { useRuns, useScenarios, useSuite } from "../api/queries";
import { ResultBadge } from "../components/Badge";
import { PageHeader } from "../components/Layout";
import { Code, Empty, ErrorState, Loading } from "../components/States";
import { button, card, link, rowLink, select, table, td, tdNum, th, thNum, thead } from "../components/table";
import { formatCost, formatInt, formatMs, formatTimestamp, shortId } from "../lib/format";

export const PAGE_SIZE = 25;
export const MODES = ["full", "model-live", "tools-live", "record"] as const;

function parsePassed(value: string | null): boolean | undefined {
  if (value === "pass") return true;
  if (value === "fail") return false;
  return undefined;
}

export function RunsPage() {
  const [params, setParams] = useSearchParams();
  const navigate = useNavigate();
  const scenario = params.get("scenario") ?? undefined;
  const mode = params.get("mode") ?? undefined;
  const result = params.get("result") ?? "";
  const suiteId = params.get("suite_id") ?? undefined;
  const offset = Math.max(0, Number(params.get("offset") ?? "0") || 0);

  const scenarios = useScenarios();
  const suite = useSuite(suiteId);
  const runs = useRuns({
    limit: PAGE_SIZE,
    offset,
    scenario,
    mode,
    suite_id: suiteId,
    passed: parsePassed(result),
  });

  function update(key: string, value: string) {
    const next = new URLSearchParams(params);
    if (value) next.set(key, value);
    else next.delete(key);
    if (key !== "offset") next.delete("offset");
    setParams(next);
  }

  const total = runs.data?.total ?? 0;
  const items = runs.data?.items ?? [];
  const filtered = Boolean(scenario ?? mode ?? suiteId) || result !== "";

  return (
    <>
      <PageHeader title="Runs">
        <form className="flex flex-wrap items-center gap-3 text-sm" aria-label="Filters" onSubmit={(e) => { e.preventDefault(); }}>
          <label className="flex items-center gap-1.5">
            <span className="text-slate-600">Scenario</span>
            <select className={select} value={scenario ?? ""} onChange={(e) => { update("scenario", e.target.value); }}>
              <option value="">all scenarios</option>
              {(scenarios.data ?? []).map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-1.5">
            <span className="text-slate-600">Mode</span>
            <select className={select} value={mode ?? ""} onChange={(e) => { update("mode", e.target.value); }}>
              <option value="">all modes</option>
              {MODES.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-1.5">
            <span className="text-slate-600">Result</span>
            <select className={select} value={result} onChange={(e) => { update("result", e.target.value); }}>
              <option value="">any result</option>
              <option value="pass">pass</option>
              <option value="fail">fail</option>
            </select>
          </label>
          {filtered ? (
            <button type="button" className={button} onClick={() => { setParams(new URLSearchParams()); }}>
              Clear filters
            </button>
          ) : null}
        </form>
      </PageHeader>

      {suiteId ? (
        <p className="mb-3 text-sm text-slate-700">
          Showing runs of suite <Code>{shortId(suiteId, 12)}</Code>
          {suite.data ? (
            <>
              {" "}
              (<span className="font-mono">{suite.data.config_path}</span>, started {formatTimestamp(suite.data.started_at)})
            </>
          ) : null}
          . <Link className={link} to="/suites">All suites</Link>
        </p>
      ) : null}
      {scenarios.isError ? <ErrorState error={scenarios.error} title="Could not load scenarios" /> : null}

      {runs.isPending ? (
        <Loading label="Loading runs…" />
      ) : runs.isError ? (
        <ErrorState error={runs.error} title="Could not load runs" />
      ) : items.length === 0 ? (
        <Empty title={filtered ? "No runs match these filters." : "No runs stored yet."}>
          {filtered ? (
            "Clear a filter to see more runs."
          ) : (
            <>
              Record some with <Code>replay ci replay.toml --db $REPLAY_DB_URL</Code>.
            </>
          )}
        </Empty>
      ) : (
        <div className={`${card} overflow-x-auto`}>
          <table className={table}>
            <thead className={thead}>
              <tr>
                <th className={th}>Started (UTC)</th>
                <th className={th}>Scenario</th>
                <th className={th}>Mode</th>
                <th className={th}>Policy</th>
                <th className={th}>Result</th>
                <th className={thNum}>Steps</th>
                <th className={thNum} title="steps served on an approximate match">Approx.</th>
                <th className={thNum}>Cost</th>
                <th className={thNum}>Duration</th>
                <th className={th}>Run</th>
              </tr>
            </thead>
            <tbody>
              {items.map((run) => (
                <tr
                  key={run.id}
                  className={rowLink}
                  onClick={() => { void navigate(`/runs/${run.id}`); }}
                >
                  <td className={`${td} num whitespace-nowrap`}>{formatTimestamp(run.started_at)}</td>
                  <td className={td}>
                    <Link className={`${link} font-medium`} to={`/runs/${run.id}`} onClick={(e) => { e.stopPropagation(); }}>
                      {run.scenario}
                    </Link>
                  </td>
                  <td className={`${td} font-mono text-xs`}>{run.mode}</td>
                  <td className={`${td} font-mono text-xs`}>{run.policy}</td>
                  <td className={td}>
                    <ResultBadge passed={run.passed} />
                  </td>
                  <td className={tdNum}>{formatInt(run.step_count)}</td>
                  <td className={`${tdNum} ${run.approximate_count > 0 ? "font-semibold text-amber-800" : "text-slate-500"}`}>
                    {formatInt(run.approximate_count)}
                  </td>
                  <td className={tdNum}>{formatCost(run.cost_usd)}</td>
                  <td className={tdNum}>{formatMs(run.duration_ms)}</td>
                  <td className={`${td} font-mono text-xs text-slate-500`}>{shortId(run.id)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <Pagination offset={offset} total={total} count={items.length} onOffset={(o) => { update("offset", o === 0 ? "" : String(o)); }} />
        </div>
      )}
    </>
  );
}

function Pagination({ offset, total, count, onOffset }: { offset: number; total: number; count: number; onOffset: (offset: number) => void }) {
  return (
    <nav aria-label="Pagination" className="flex items-center justify-between gap-3 px-3 py-2 text-sm text-slate-700">
      <span className="num">
        {formatInt(offset + 1)}–{formatInt(offset + count)} of {formatInt(total)}
      </span>
      <span className="flex gap-2">
        <button type="button" className={button} disabled={offset === 0} onClick={() => { onOffset(Math.max(0, offset - PAGE_SIZE)); }}>
          ← Newer
        </button>
        <button type="button" className={button} disabled={offset + count >= total} onClick={() => { onOffset(offset + PAGE_SIZE); }}>
          Older →
        </button>
      </span>
    </nav>
  );
}
