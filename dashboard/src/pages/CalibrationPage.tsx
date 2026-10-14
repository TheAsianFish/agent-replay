import { Link, useNavigate, useParams } from "react-router";
import { useCalibration, useCalibrations } from "../api/queries";
import type { CalibrationView } from "../api/types";
import { GateBadge } from "../components/Badge";
import { KappaAxis, KappaWhisker, kappaDomain } from "../components/KappaChart";
import { PageHeader } from "../components/Layout";
import { Code, Empty, ErrorState, Loading } from "../components/States";
import { card, link, rowLink, table, td, tdNum, th, thNum, thead } from "../components/table";
import { formatFixed, formatInt, formatPercent, formatTimestamp, shortId } from "../lib/format";

export function CalibrationEmpty() {
  return (
    <Empty title="No calibration reports stored.">
      Run <Code>replay db add-calibration REPORT.json</Code> (or <Code>replay calibrate</Code> from M6).
    </Empty>
  );
}

export function CalibrationListPage() {
  const list = useCalibrations();
  const navigate = useNavigate();
  return (
    <>
      <PageHeader title="Judge calibration" />
      <p className="mb-3 max-w-3xl text-sm text-slate-600">
        A judge criterion fails a build only when its calibration report shows the judge agrees with human labels: the lower bound of its
        kappa interval must reach <Code>REPLAY_JUDGE_MIN_KAPPA</Code>. Everything else is advisory.
      </p>
      {list.isPending ? (
        <Loading label="Loading calibration reports…" />
      ) : list.isError ? (
        <ErrorState error={list.error} title="Could not load calibration reports" />
      ) : list.data.length === 0 ? (
        <CalibrationEmpty />
      ) : (
        <div className={`${card} overflow-x-auto`}>
          <table className={table}>
            <thead className={thead}>
              <tr>
                <th className={th}>Rubric</th>
                <th className={th}>Judge</th>
                <th className={th}>Prompt</th>
                <th className={th}>Created (UTC)</th>
                <th className={thNum}>Items</th>
                <th className={th}>Report</th>
              </tr>
            </thead>
            <tbody>
              {list.data.map((c) => (
                <tr key={c.id} className={rowLink} onClick={() => { void navigate(`/calibration/${c.id}`); }}>
                  <td className={td}>
                    <Link className={`${link} font-medium`} to={`/calibration/${c.id}`} onClick={(e) => { e.stopPropagation(); }}>
                      {c.rubric}
                    </Link>
                    <div className="font-mono text-[11px] text-slate-500">{c.rubric_fingerprint}</div>
                  </td>
                  <td className={`${td} font-mono text-xs`}>{c.judge_id}</td>
                  <td className={`${td} font-mono text-xs`}>{c.prompt_version}</td>
                  <td className={`${td} num whitespace-nowrap`}>{formatTimestamp(c.created_at)}</td>
                  <td className={tdNum}>{formatInt(c.n_items)}</td>
                  <td className={`${td} font-mono text-xs text-slate-500`}>{shortId(c.id)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}

export function CalibrationDetailPage() {
  const { id = "" } = useParams();
  const view = useCalibration(id);
  if (view.isPending) return <Loading label="Loading calibration report…" />;
  if (view.isError) return <ErrorState error={view.error} title={`Could not load calibration report ${id}`} />;
  return <CalibrationDetail view={view.data} />;
}

function CalibrationDetail({ view }: { view: CalibrationView }) {
  const { report } = view.record;
  const entries = Object.entries(report.criteria);
  const domain = kappaDomain(entries.map(([, c]) => c), view.min_kappa);
  const gating = entries.filter(([name]) => view.gates[name]?.gating).length;
  return (
    <>
      <PageHeader title={`Calibration: ${report.rubric}`}>
        <Link className={`${link} text-sm`} to="/calibration">← All reports</Link>
      </PageHeader>
      <section className={`${card} mb-4 p-4`} aria-label="Method">
        <h2 className="text-xs font-semibold tracking-wide text-slate-500 uppercase">Method</h2>
        <p className="mt-1 text-base text-slate-900" data-testid="method">{report.method}</p>
        <dl className="mt-3 grid grid-cols-2 gap-x-6 gap-y-2 text-sm md:grid-cols-3 lg:grid-cols-6">
          {[
            ["Judge", report.judge_id],
            ["Prompt version", report.prompt_version],
            ["Rubric fingerprint", report.rubric_fingerprint],
            ["Items", formatInt(report.n_items)],
            ["Created (UTC)", formatTimestamp(report.created_at)],
            ["min_kappa", formatFixed(view.min_kappa)],
          ].map(([k, v]) => (
            <div key={k} className="min-w-0">
              <dt className="text-xs font-semibold tracking-wide text-slate-500 uppercase">{k}</dt>
              <dd className="font-mono text-xs break-words">{v}</dd>
            </div>
          ))}
        </dl>
      </section>
      <h2 className="mb-2 text-base font-semibold">
        Criteria <span className="num font-normal text-slate-500">({gating} of {entries.length} gating)</span>
      </h2>
      {entries.length === 0 ? (
        <Empty title="This report has no criteria." />
      ) : (
        <div className={`${card} overflow-x-auto`}>
          <table className={table}>
            <thead className={thead}>
              <tr>
                <th className={th}>Criterion</th>
                <th className={th}>Gate</th>
                <th className={thNum}>Kappa</th>
                <th className={thNum}>Interval</th>
                <th className={`${th} py-0`}>
                  <KappaAxis domain={domain} minKappa={view.min_kappa} />
                </th>
                <th className={thNum}>Agreement</th>
                <th className={thNum}>n</th>
              </tr>
            </thead>
            <tbody>
              {entries.map(([name, c]) => (
                <tr key={name}>
                  <td className={`${td} font-mono text-xs font-semibold`}>{name}</td>
                  <td className={td}><GateBadge gating={view.gates[name]?.gating} /></td>
                  <td className={tdNum}>{formatFixed(c.kappa)}</td>
                  <td className={tdNum}>[{formatFixed(c.ci_low)}, {formatFixed(c.ci_high)}]</td>
                  <td className={`${td} py-0`}>
                    <KappaWhisker name={name} criterion={c} domain={domain} minKappa={view.min_kappa} />
                  </td>
                  <td className={tdNum}>{formatPercent(c.agreement)}</td>
                  <td className={tdNum}>{formatInt(c.n)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="border-t border-slate-100 px-3 py-2 text-xs text-slate-600">
            Dot: kappa. Whisker: interval. Dashed line (▲): min_kappa {formatFixed(view.min_kappa)}. A criterion gates when its interval's
            lower bound reaches min_kappa.
          </p>
        </div>
      )}
    </>
  );
}
