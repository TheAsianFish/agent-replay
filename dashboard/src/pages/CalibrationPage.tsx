import { Link, useNavigate, useParams } from "react-router";
import { useCalibration, useCalibrations } from "../api/queries";
import type { ReactNode } from "react";
import type {
  CalibrationReport,
  CalibrationView,
  Confusion,
  CriterionCalibration,
  FlipStats,
  GroupStats,
  ProbeResult,
  SelfConsistency,
} from "../api/types";
import { Badge, GateBadge } from "../components/Badge";
import { KappaAxis, KappaWhisker, kappaDomain } from "../components/KappaChart";
import { PageHeader } from "../components/Layout";
import { Code, Empty, ErrorState, Loading } from "../components/States";
import { card, link, rowLink, table, td, tdNum, th, thNum, thead } from "../components/table";
import { formatCost, formatFixed, formatInt, formatPercent, formatTimestamp, shortId } from "../lib/format";

export function CalibrationEmpty() {
  return (
    <Empty title="No calibration reports stored.">
      Run <Code>replay calibrate GOLDEN_DIR --rubric RUBRIC --store</Code>, or store an existing report with{" "}
      <Code>replay db add-calibration REPORT.json</Code>.
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
                <th className={th}>Reference</th>
                <th className={thNum}>Repeats</th>
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
                  <td className={`${td} text-xs`}>
                    <ReferenceCell reference={c.reference} statsImpl={c.stats_impl} />
                  </td>
                  <td className={tdNum}>{formatInt(c.repeats)}</td>
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

/** Human references in plain text; the reports that never gate (objective labels, baseline statistics) muted. */
function ReferenceCell({ reference, statsImpl }: { reference: string; statsImpl: string | null | undefined }) {
  return (
    <>
      {reference === "human" ? (
        <span>human</span>
      ) : (
        <span className="text-slate-500 italic" title="Compared with the suite's deterministic checks; never gates">
          {reference}
        </span>
      )}
      {statsImpl === "baseline" ? (
        <span className="block text-slate-500 italic" title="REPLAY_STATS_IMPL=baseline: raw agreement, not kappa; never gates">
          baseline stats
        </span>
      ) : null}
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

/** The pseudo-criterion for the whole run (every criterion passes). */
const OVERALL = "_overall";

function criterionName(id: string): string {
  return id === OVERALL ? "whole run" : id;
}

const UNDEFINED_KAPPA =
  "Kappa is undefined: the labels leave no room for chance disagreement (e.g. every reference label is the same). Compare raw agreement with the majority baseline instead.";

function CalibrationDetail({ view }: { view: CalibrationView }) {
  const { report } = view.record;
  const entries = Object.entries(report.criteria);
  const domain = kappaDomain(entries.map(([, c]) => c), view.min_kappa);
  const gating = entries.filter(([name]) => view.gates[name]?.gating).length;
  const baseline = report.stats_impl === "baseline";
  const hasConfusion = entries.some(([, c]) => c.confusion);
  const hasSelf = entries.some(([, c]) => c.self_consistency);
  return (
    <>
      <PageHeader title={`Calibration: ${report.rubric}`}>
        <Link className={`${link} text-sm`} to="/calibration">← All reports</Link>
      </PageHeader>
      {view.blocked ? (
        <div role="note" data-testid="blocked" className="mb-4 rounded border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900">
          <p className="font-semibold">
            <span aria-hidden>! </span>No criterion of this report can gate: its judge verdicts stay advisory whatever the numbers.
          </p>
          <p className="mt-1">Why: {view.blocked}.</p>
        </div>
      ) : null}
      <section className={`${card} mb-4 p-4`} aria-label="Method">
        <div className="flex flex-wrap items-center gap-2">
          <h2 className="text-xs font-semibold tracking-wide text-slate-500 uppercase">Method</h2>
          <ReferenceBadge reference={report.reference} />
          <StatsBadge statsImpl={report.stats_impl} />
          {report.schema_version === 1 ? <Badge tone="slate">schema 1</Badge> : null}
        </div>
        <p className="mt-1 text-base text-slate-900" data-testid="method">{report.method}</p>
        <dl className="mt-3 grid grid-cols-2 gap-x-6 gap-y-2 text-sm md:grid-cols-3 lg:grid-cols-6">
          {methodFacts(report, view.min_kappa, view.min_items).map(([k, v]) => (
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
        <div className={`${card} mb-5 overflow-x-auto`}>
          <table className={table} aria-label="Criteria">
            <thead className={thead}>
              <tr>
                <th className={th}>Criterion</th>
                <th className={th}>Gate</th>
                <th className={thNum}>{baseline ? "Kappa*" : "Kappa"}</th>
                <th className={thNum}>Interval</th>
                <th className={`${th} py-0`}>
                  <KappaAxis domain={domain} minKappa={view.min_kappa} />
                </th>
                <th className={thNum}>Agreement</th>
                <th className={thNum}>n</th>
                {hasConfusion ? <th className={th}>Judge × reference</th> : null}
                {hasSelf ? <th className={th}>Self-consistency</th> : null}
              </tr>
            </thead>
            <tbody>
              {entries.map(([name, c]) => (
                <tr key={name}>
                  <td className={`${td} font-mono text-xs font-semibold`}>{criterionName(name)}</td>
                  <td className={td}><GateBadge gating={view.gates[name]?.gating} blocked={Boolean(view.blocked)} /></td>
                  <td className={tdNum}><KappaValue criterion={c} /></td>
                  <td className={tdNum}>{c.ci_low === null || c.ci_high === null ? "—" : `[${formatFixed(c.ci_low)}, ${formatFixed(c.ci_high)}]`}</td>
                  <td className={`${td} py-0`}>
                    <KappaWhisker name={criterionName(name)} criterion={c} domain={domain} minKappa={view.min_kappa} />
                  </td>
                  <td className={tdNum}><AgreementValue criterion={c} /></td>
                  <td className={tdNum}>{formatInt(c.n)}</td>
                  {hasConfusion ? <td className={td}>{c.confusion ? <ConfusionMatrix name={criterionName(name)} confusion={c.confusion} /> : "—"}</td> : null}
                  {hasSelf ? <td className={td}>{c.self_consistency ? <SelfConsistencyValue value={c.self_consistency} /> : "—"}</td> : null}
                </tr>
              ))}
            </tbody>
          </table>
          <p className="border-t border-slate-100 px-3 py-2 text-xs text-slate-600">
            Dot: kappa. Whisker: interval. Dashed line (▲): min_kappa {formatFixed(view.min_kappa)}. A criterion gates when its interval's
            lower bound reaches min_kappa over at least {formatInt(view.min_items)} labelled items.
            {hasConfusion ? " Matrix: judge verdicts (rows) against reference labels (columns); red marks false passes, the regressions a gate would let through." : ""}
            {baseline ? " *Computed with REPLAY_STATS_IMPL=baseline: the kappa columns hold raw agreement, not kappa." : ""}
          </p>
        </div>
      )}
      {report.probes.length > 0 ? <Probes probes={report.probes} /> : null}
    </>
  );
}

function methodFacts(report: CalibrationReport, minKappa: number, minItems: number): [string, ReactNode][] {
  const facts: [string, ReactNode | null][] = [
    ["Judge", report.judge_id],
    ["Prompt version", report.prompt_version],
    ["Rubric fingerprint", report.rubric_fingerprint],
    ["Items", formatInt(report.n_items)],
    ["Created (UTC)", formatTimestamp(report.created_at)],
    ["min_kappa", formatFixed(minKappa)],
    ["min_items", formatInt(minItems)],
    ["Labeller", report.labeller ?? null],
    ["Repeats", report.schema_version >= 2 ? formatInt(report.repeats) : null],
    [
      "Golden set",
      report.golden_set || report.golden_fingerprint ? (
        <>
          {report.golden_set ?? "—"}
          {report.golden_fingerprint ? (
            <span className="block text-[11px] text-slate-500" title={report.golden_fingerprint}>
              {shortFingerprint(report.golden_fingerprint)}
            </span>
          ) : null}
        </>
      ) : null,
    ],
    [
      "Judge calls",
      report.judge_calls === null || report.judge_calls === undefined
        ? null
        : `${formatInt(report.judge_calls)} live · ${formatInt(report.cached_calls)} cached`,
    ],
    ["Cost", report.cost_usd === null || report.cost_usd === undefined ? null : formatCost(report.cost_usd)],
  ];
  return facts.filter((f): f is [string, ReactNode] => f[1] !== null);
}

/** ``sha256:0123456789ab…`` → ``sha256:01234567``. */
function shortFingerprint(fingerprint: string): string {
  const colon = fingerprint.indexOf(":");
  return colon < 0 ? shortId(fingerprint, 12) : `${fingerprint.slice(0, colon + 1)}${shortId(fingerprint.slice(colon + 1))}`;
}

function ReferenceBadge({ reference }: { reference: CalibrationReport["reference"] }) {
  return reference === "human" ? (
    <Badge tone="blue" title="Judge verdicts compared with human labels">human labels</Badge>
  ) : (
    <Badge tone="amber" icon="!" title="Judge verdicts compared with the suite's deterministic checks; such a report never gates">
      objective labels
    </Badge>
  );
}

function StatsBadge({ statsImpl }: { statsImpl: string | null | undefined }) {
  if (!statsImpl) return null;
  return statsImpl === "baseline" ? (
    <Badge tone="amber" icon="!" title="REPLAY_STATS_IMPL=baseline: such a report never gates">
      baseline stats: raw agreement, not kappa
    </Badge>
  ) : (
    <Badge tone="slate" title={`REPLAY_STATS_IMPL=${statsImpl}`}>
      {statsImpl} stats
    </Badge>
  );
}

function KappaValue({ criterion }: { criterion: CriterionCalibration }) {
  if (criterion.kappa !== null) return <>{formatFixed(criterion.kappa)}</>;
  return (
    <span title={UNDEFINED_KAPPA}>
      undefined
      <span className="block text-[11px] whitespace-normal text-slate-500">labels too uniform</span>
    </span>
  );
}

function AgreementValue({ criterion }: { criterion: CriterionCalibration }) {
  const { agreement, majority_agreement: majority, majority_label: label } = criterion;
  if (majority === null || majority === undefined) return <>{formatPercent(agreement)}</>;
  const beats = agreement > majority;
  return (
    <span title={`A judge that always answers "${label ?? "pass"}" agrees ${formatPercent(majority)} of the time; raw agreement means little unless it beats that.`}>
      <span className={beats ? undefined : "font-semibold text-amber-800"} data-testid="agreement">{formatPercent(agreement)}</span>{" "}
      <span className="text-slate-500">vs {formatPercent(majority)} majority</span>
    </span>
  );
}

function ConfusionMatrix({ name, confusion }: { name: string; confusion: Confusion }) {
  const cell = "border border-slate-200 px-1.5 text-right num";
  const head = "px-1.5 text-left font-normal text-slate-500";
  const falsePass = confusion.judge_pass_reference_fail;
  return (
    <table className="text-[11px] leading-4" aria-label={`${name}: judge verdicts against reference labels`}>
      <thead>
        <tr>
          <td />
          <th scope="col" className={head}>ref pass</th>
          <th scope="col" className={head}>ref fail</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th scope="row" className={head}>judge pass</th>
          <td className={cell}>{formatInt(confusion.both_pass)}</td>
          <td
            className={`${cell} ${falsePass > 0 ? "bg-red-50 font-semibold text-red-800" : ""}`}
            title="False passes: the judge passed what the reference failed"
            data-testid="false-pass"
          >
            {formatInt(falsePass)}
          </td>
        </tr>
        <tr>
          <th scope="row" className={head}>judge fail</th>
          <td className={cell} title="False failures: the judge failed what the reference passed" data-testid="false-fail">
            {formatInt(confusion.judge_fail_reference_pass)}
          </td>
          <td className={cell}>{formatInt(confusion.both_fail)}</td>
        </tr>
      </tbody>
    </table>
  );
}

function SelfConsistencyValue({ value }: { value: SelfConsistency }) {
  const kappa = value.mean_pairwise_kappa;
  return (
    <span className="text-xs whitespace-nowrap" title={`Over ${value.runs} runs of the judge on the same ${value.n} items`}>
      <span className="num">{formatPercent(value.unanimous)}</span> unanimous
      <span className="block text-slate-500">
        pairwise κ <span className="num">{kappa === null || kappa === undefined ? "undefined" : formatFixed(kappa)}</span> · {value.runs} runs
      </span>
    </span>
  );
}

const PROBE_TITLE: Record<ProbeResult["probe"], string> = {
  position: "Position",
  verbosity: "Verbosity",
  self_preference: "Self-preference",
};

const GROUP_NAME: Record<string, string> = { same_model: "same model as judge", other_model: "other models" };

function Probes({ probes }: { probes: readonly ProbeResult[] }) {
  return (
    <section aria-labelledby="probes-title">
      <h2 id="probes-title" className="mb-2 text-base font-semibold">Bias probes</h2>
      <div className="space-y-4">
        {probes.map((p) => (
          <section key={p.probe} className={`${card} overflow-x-auto`} aria-labelledby={`probe-${p.probe}`}>
            <div className="px-3 py-2">
              <h3 id={`probe-${p.probe}`} className="text-sm font-semibold">{PROBE_TITLE[p.probe]}</h3>
              <p className="text-sm text-slate-600">{p.description}</p>
              {p.skipped ? <p className="mt-1 text-xs text-slate-500 italic">Skipped: {p.skipped}</p> : null}
            </div>
            {p.flips ? <FlipTable title={PROBE_TITLE[p.probe]} flips={p.flips} /> : null}
            {p.groups ? <GroupTable groups={p.groups} /> : null}
          </section>
        ))}
      </div>
    </section>
  );
}

function rateInterval(low: number | null | undefined, high: number | null | undefined): string {
  return low === null || low === undefined || high === null || high === undefined ? "—" : `[${formatPercent(low)}, ${formatPercent(high)}]`;
}

function FlipTable({ title, flips }: { title: string; flips: Record<string, FlipStats> }) {
  return (
    <table className={table} aria-label={`${title} flips per criterion`}>
      <thead className={thead}>
        <tr>
          <th className={th}>Criterion</th>
          <th className={thNum}>Flips</th>
          <th className={thNum}>Rate</th>
          <th className={thNum}>Interval</th>
          <th className={thNum} title="Fail became pass under the perturbation">To pass</th>
          <th className={thNum} title="Pass became fail under the perturbation">To fail</th>
        </tr>
      </thead>
      <tbody>
        {Object.entries(flips).map(([name, f]) => (
          <tr key={name}>
            <td className={`${td} font-mono text-xs font-semibold`}>{criterionName(name)}</td>
            <td className={tdNum}>{formatInt(f.flips)} / {formatInt(f.n)}</td>
            <td className={tdNum}>{formatPercent(f.rate)}</td>
            <td className={tdNum}>{rateInterval(f.ci_low, f.ci_high)}</td>
            <td className={tdNum}>{formatInt(f.to_pass)}</td>
            <td className={tdNum}>{formatInt(f.to_fail)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function GroupTable({ groups }: { groups: Record<string, Record<string, GroupStats>> }) {
  return (
    <table className={table} aria-label="Self-preference groups per criterion">
      <thead className={thead}>
        <tr>
          <th className={th}>Criterion</th>
          <th className={th}>Group</th>
          <th className={thNum}>n</th>
          <th className={thNum}>Judge pass rate</th>
          <th className={thNum}>Reference pass rate</th>
          <th className={thNum} title="Of the items the reference failed, the fraction the judge passed">False pass rate</th>
        </tr>
      </thead>
      <tbody>
        {Object.entries(groups).flatMap(([name, byGroup]) =>
          Object.entries(byGroup).map(([group, g], i, all) => (
            <tr key={`${name}/${group}`}>
              {i === 0 ? (
                <td className={`${td} font-mono text-xs font-semibold`} rowSpan={all.length}>{criterionName(name)}</td>
              ) : null}
              <td className={td}>{GROUP_NAME[group] ?? group}</td>
              <td className={tdNum}>{formatInt(g.n)}</td>
              <td className={tdNum}>{formatPercent(g.judge_pass_rate)}</td>
              <td className={tdNum}>{formatPercent(g.reference_pass_rate)}</td>
              <td className={tdNum}>
                {formatPercent(g.false_pass_rate)} <span className="text-slate-500">of {formatInt(g.reference_fail_n)}</span>
              </td>
            </tr>
          )),
        )}
      </tbody>
    </table>
  );
}
