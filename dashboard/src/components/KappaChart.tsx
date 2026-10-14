import type { CriterionCalibration } from "../api/types";
import { formatFixed } from "../lib/format";
import { clamp, linearScale, ticks } from "../lib/scale";

/** 0..1 when every interval is non-negative, else the full −1..1 range of kappa. Undefined kappas are left out. */
export function kappaDomain(criteria: readonly CriterionCalibration[], minKappa: number): [number, number] {
  const low = Math.min(minKappa, ...criteria.flatMap((c) => [c.ci_low ?? minKappa, c.kappa ?? minKappa]));
  return low < 0 ? [-1, 1] : [0, 1];
}

const WIDTH = 320;
const HEIGHT = 30;
const PAD = 10;

/** Screen-reader text for one criterion's whisker. */
export function kappaLabel(name: string, criterion: CriterionCalibration, minKappa: number): string {
  if (criterion.kappa === null) return `${name}: kappa undefined, threshold ${formatFixed(minKappa)}`;
  const interval =
    criterion.ci_low === null || criterion.ci_high === null
      ? "no interval"
      : `interval ${formatFixed(criterion.ci_low)} to ${formatFixed(criterion.ci_high)}`;
  return `${name}: kappa ${formatFixed(criterion.kappa)}, ${interval}, threshold ${formatFixed(minKappa)}`;
}

/** One criterion: kappa point, CI whisker, and the min_kappa threshold as a vertical line.
 * An undefined kappa draws only the axis and the threshold; a missing interval draws no whisker. */
export function KappaWhisker({
  criterion,
  domain,
  minKappa,
  name,
}: {
  criterion: CriterionCalibration;
  domain: [number, number];
  minKappa: number;
  name: string;
}) {
  const x = linearScale(domain, [PAD, WIDTH - PAD]);
  const at = (v: number) => x(clamp(v, domain[0], domain[1]));
  const mid = HEIGHT / 2;
  const { kappa, ci_low: low, ci_high: high } = criterion;
  const label = kappaLabel(name, criterion, minKappa);
  return (
    <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} width={WIDTH} height={HEIGHT} role="img" aria-label={label}>
      <title>{label}</title>
      <line x1={x(domain[0])} x2={x(domain[1])} y1={mid} y2={mid} stroke="#e2e8f0" strokeWidth={1} />
      {domain[0] < 0 ? <line x1={x(0)} x2={x(0)} y1={6} y2={HEIGHT - 6} stroke="#cbd5e1" strokeWidth={1} /> : null}
      <line x1={at(minKappa)} x2={at(minKappa)} y1={2} y2={HEIGHT - 2} stroke="#b45309" strokeWidth={1.5} strokeDasharray="3 2" data-testid="min-kappa-line" />
      {kappa !== null && low !== null && high !== null ? (
        <g data-testid="kappa-interval">
          <line x1={at(low)} x2={at(high)} y1={mid} y2={mid} stroke="#1d4ed8" strokeWidth={2} />
          <line x1={at(low)} x2={at(low)} y1={mid - 5} y2={mid + 5} stroke="#1d4ed8" strokeWidth={2} />
          <line x1={at(high)} x2={at(high)} y1={mid - 5} y2={mid + 5} stroke="#1d4ed8" strokeWidth={2} />
        </g>
      ) : null}
      {kappa !== null ? <circle cx={at(kappa)} cy={mid} r={5} fill="#1d4ed8" stroke="#fff" strokeWidth={2} /> : null}
    </svg>
  );
}

/** Tick labels for the whisker column header. */
export function KappaAxis({ domain, minKappa }: { domain: [number, number]; minKappa: number }) {
  const x = linearScale(domain, [PAD, WIDTH - PAD]);
  const values = ticks(domain[0], domain[1], domain[0] < 0 ? 4 : 5);
  return (
    <svg viewBox={`0 0 ${WIDTH} 30`} width={WIDTH} height={30} aria-hidden>
      {values.map((v) => (
        <g key={v}>
          <line x1={x(v)} x2={x(v)} y1={20} y2={26} stroke="#94a3b8" />
          <text x={x(v)} y={16} textAnchor="middle" className="fill-slate-600 text-[10px] normal-case">
            {formatFixed(v, 1)}
          </text>
        </g>
      ))}
      <text x={x(clamp(minKappa, domain[0], domain[1]))} y={29} textAnchor="middle" className="fill-amber-800 text-[9px]">
        ▲
      </text>
    </svg>
  );
}
