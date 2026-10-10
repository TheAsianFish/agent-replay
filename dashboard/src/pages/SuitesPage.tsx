import { Link, useNavigate } from "react-router";
import { useSuites } from "../api/queries";
import { ResultBadge } from "../components/Badge";
import { PageHeader } from "../components/Layout";
import { Code, Empty, ErrorState, Loading } from "../components/States";
import { card, link, rowLink, table, td, tdNum, th, thNum, thead } from "../components/table";
import { formatInt, formatMs, formatTimestamp, shortId } from "../lib/format";

export function SuitesPage() {
  const suites = useSuites();
  const navigate = useNavigate();
  return (
    <>
      <PageHeader title="Suites" />
      <p className="mb-3 text-sm text-slate-600">
        Each <Code>replay ci</Code> or <Code>replay flake</Code> invocation stores a suite run. Open one to see its runs.
      </p>
      {suites.isPending ? (
        <Loading label="Loading suites…" />
      ) : suites.isError ? (
        <ErrorState error={suites.error} title="Could not load suites" />
      ) : suites.data.length === 0 ? (
        <Empty title="No suite runs stored yet.">
          Run <Code>replay ci replay.toml --db $REPLAY_DB_URL</Code> to store one.
        </Empty>
      ) : (
        <div className={`${card} overflow-x-auto`}>
          <table className={table}>
            <thead className={thead}>
              <tr>
                <th className={th}>Started (UTC)</th>
                <th className={th}>Suite</th>
                <th className={th}>Config</th>
                <th className={th}>Git sha</th>
                <th className={th}>Result</th>
                <th className={thNum}>Scenarios</th>
                <th className={thNum}>Failed</th>
                <th className={thNum}>Runs</th>
                <th className={thNum}>Wall time</th>
              </tr>
            </thead>
            <tbody>
              {suites.data.map((s) => {
                const to = `/?suite_id=${s.id}`;
                const wall = s.finished_at ? new Date(s.finished_at).getTime() - new Date(s.started_at).getTime() : null;
                return (
                  <tr key={s.id} className={rowLink} onClick={() => { void navigate(to); }}>
                    <td className={`${td} num whitespace-nowrap`}>{formatTimestamp(s.started_at)}</td>
                    <td className={td}>
                      <Link className={`${link} font-mono text-xs`} to={to} onClick={(e) => { e.stopPropagation(); }}>
                        {shortId(s.id, 12)}
                      </Link>
                    </td>
                    <td className={`${td} font-mono text-xs`}>{s.config_path}</td>
                    <td className={`${td} font-mono text-xs text-slate-600`}>{s.git_sha ? shortId(s.git_sha, 10) : "—"}</td>
                    <td className={td}><ResultBadge passed={s.passed} /></td>
                    <td className={tdNum}>{formatInt(s.scenario_count)}</td>
                    <td className={`${tdNum} ${s.failed_count > 0 ? "font-semibold text-red-800" : ""}`}>{formatInt(s.failed_count)}</td>
                    <td className={tdNum}>{formatInt(s.run_count)}</td>
                    <td className={tdNum}>{formatMs(wall)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
