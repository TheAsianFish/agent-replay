import { formatMs } from "../lib/format";
import { barPercent } from "../lib/scale";

/** A horizontal bar of ``value`` relative to ``max``, with the value as text. */
export function LatencyBar({ value, max }: { value: number | null | undefined; max: number }) {
  const pct = barPercent(value, max);
  return (
    <div className="flex items-center gap-2" title={`${formatMs(value)} (${pct.toFixed(0)}% of the slowest step)`}>
      <div className="h-2 w-24 shrink-0 rounded-sm bg-slate-100" aria-hidden>
        <div className="h-2 rounded-sm bg-blue-600" style={{ width: `${pct}%` }} />
      </div>
      <span className="num w-16 text-right text-xs text-slate-700">{formatMs(value)}</span>
    </div>
  );
}
