import { useState } from "react";
import type { FlakePoint } from "../api/types";
import { formatPercent, formatTimestamp, shortId } from "../lib/format";
import { bandPath, linePath, linearScale, ticks } from "../lib/scale";

const WIDTH = 760;
const HEIGHT = 260;
const M = { top: 14, right: 44, bottom: 36, left: 60 };

export interface TrendGeometry {
  x: (t: number) => number;
  y: (v: number) => number;
  points: { point: FlakePoint; t: number; cx: number; cy: number | null }[];
}

/** Positions for each point: x by report time (evenly centred when all times are equal), y = pass rate on 0..1. */
export function trendGeometry(points: readonly FlakePoint[]): TrendGeometry {
  const times = points.map((p) => new Date(p.started_at).getTime());
  const t0 = Math.min(...times);
  const t1 = Math.max(...times);
  const x = linearScale([t0, t1], [M.left, WIDTH - M.right]);
  const y = linearScale([0, 1], [HEIGHT - M.bottom, M.top]);
  return {
    x,
    y,
    points: points.map((point, i) => {
      const t = times[i] ?? 0;
      return { point, t, cx: x(t), cy: point.pass_rate == null ? null : y(point.pass_rate) };
    }),
  };
}

/** Pass rate over flake reports for one scenario, with the confidence interval as a shaded band. */
export function TrendChart({ points, scenario }: { points: readonly FlakePoint[]; scenario: string }) {
  const [hover, setHover] = useState<number | null>(null);
  if (points.length === 0) return null;
  const geo = trendGeometry(points);
  const rated = geo.points.filter((p) => p.cy !== null);
  const upper = rated.map((p): [number, number] => [p.cx, geo.y(p.point.ci_high)]);
  const lower = rated.map((p): [number, number] => [p.cx, geo.y(p.point.ci_low)]);
  const line = rated.map((p): [number, number] => [p.cx, p.cy ?? 0]);
  const xTicks = xLabels(geo.points);
  const span = Math.max(...geo.points.map((p) => p.t)) - Math.min(...geo.points.map((p) => p.t));
  const label = (iso: string) => formatTimestamp(iso).slice(5, span < 3_600_000 ? 19 : 16);
  const active = hover === null ? null : geo.points[hover];

  return (
    <div className="relative">
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="h-auto w-full max-w-[760px]"
        role="img"
        aria-label={`Pass rate of ${scenario} over ${points.length} flake report(s), with confidence interval band`}
        onMouseLeave={() => { setHover(null); }}
      >
        {ticks(0, 1, 4).map((v) => (
          <g key={v}>
            <line x1={M.left} x2={WIDTH - M.right} y1={geo.y(v)} y2={geo.y(v)} stroke="#e2e8f0" strokeWidth={1} />
            <text x={M.left - 8} y={geo.y(v)} dy="0.32em" textAnchor="end" className="fill-slate-600 text-[11px]">
              {formatPercent(v, 0)}
            </text>
          </g>
        ))}
        <text x={12} y={(HEIGHT - M.bottom + M.top) / 2} transform={`rotate(-90 12 ${(HEIGHT - M.bottom + M.top) / 2})`} textAnchor="middle" className="fill-slate-600 text-[11px]">
          pass rate
        </text>
        {xTicks.map((p, i) => (
          <text
            key={p.point.report_id}
            x={p.cx}
            y={HEIGHT - M.bottom + 18}
            textAnchor={xTicks.length > 1 && i === 0 ? "start" : xTicks.length > 1 && i === xTicks.length - 1 ? "end" : "middle"}
            className="fill-slate-600 text-[11px]"
          >
            {label(p.point.started_at)}
          </text>
        ))}
        <text x={(M.left + WIDTH - M.right) / 2} y={HEIGHT - 4} textAnchor="middle" className="fill-slate-500 text-[11px]">
          report time (UTC)
        </text>
        {rated.length > 1 ? <path d={bandPath(upper, lower)} fill="#2563eb" fillOpacity={0.14} data-testid="ci-band" /> : null}
        {rated.length === 1 && rated[0] ? (
          <line
            data-testid="ci-band"
            x1={rated[0].cx}
            x2={rated[0].cx}
            y1={geo.y(rated[0].point.ci_high)}
            y2={geo.y(rated[0].point.ci_low)}
            stroke="#2563eb"
            strokeOpacity={0.35}
            strokeWidth={10}
            strokeLinecap="round"
          />
        ) : null}
        {rated.length > 1 ? <path d={linePath(line)} fill="none" stroke="#1d4ed8" strokeWidth={2} /> : null}
        {active ? <line x1={active.cx} x2={active.cx} y1={M.top} y2={HEIGHT - M.bottom} stroke="#94a3b8" strokeDasharray="3 3" /> : null}
        {geo.points.map((p, i) =>
          p.cy === null ? null : (
            <circle
              key={p.point.report_id}
              cx={p.cx}
              cy={p.cy}
              r={hover === i ? 6 : 4.5}
              fill="#1d4ed8"
              stroke="#fff"
              strokeWidth={2}
            />
          ),
        )}
        {geo.points.map((p, i) => (
          <rect
            key={`hit-${p.point.report_id}`}
            x={p.cx - 14}
            y={M.top}
            width={28}
            height={HEIGHT - M.top - M.bottom}
            fill="transparent"
            onMouseEnter={() => { setHover(i); }}
          >
            <title>{tooltip(p.point)}</title>
          </rect>
        ))}
      </svg>
      {active ? (
        <div
          role="tooltip"
          className="pointer-events-none absolute top-2 right-2 rounded border border-slate-200 bg-white px-2 py-1 text-xs text-slate-800 shadow"
        >
          {tooltip(active.point)}
        </div>
      ) : null}
    </div>
  );
}

const LABEL_GAP = 120;

/** The points that get a time label: the first, then any at least ``LABEL_GAP`` px right of the last labelled one. */
export function xLabels<P extends { cx: number }>(points: readonly P[]): P[] {
  const chosen: P[] = [];
  for (const p of points) {
    const last = chosen[chosen.length - 1];
    if (last === undefined || p.cx - last.cx >= LABEL_GAP) chosen.push(p);
  }
  return chosen;
}

function tooltip(p: FlakePoint): string {
  return `${formatTimestamp(p.started_at)} · ${p.classification} · ${p.passes}/${p.runs} = ${formatPercent(p.pass_rate)} [${formatPercent(p.ci_low)}, ${formatPercent(p.ci_high)}] · report ${shortId(p.report_id)}`;
}
