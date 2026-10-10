/** Small helpers for hand-written SVG charts. */

export type Scale = (value: number) => number;

/** A linear map from ``[d0, d1]`` to ``[r0, r1]``; a degenerate domain maps to the range midpoint. */
export function linearScale(domain: readonly [number, number], range: readonly [number, number]): Scale {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  if (d1 === d0) {
    const mid = (r0 + r1) / 2;
    return () => mid;
  }
  const k = (r1 - r0) / (d1 - d0);
  return (value) => r0 + (value - d0) * k;
}

export function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value));
}

/** ``count + 1`` evenly spaced ticks from ``low`` to ``high`` inclusive, rounded to avoid float noise. */
export function ticks(low: number, high: number, count: number): number[] {
  if (count <= 0 || high === low) return [low];
  const step = (high - low) / count;
  return Array.from({ length: count + 1 }, (_, i) => Math.round((low + i * step) * 1e9) / 1e9);
}

/** Width of a bar as a percentage of the largest value; 0 when the maximum is not positive. */
export function barPercent(value: number | null | undefined, max: number): number {
  if (value === null || value === undefined || !(max > 0) || !(value > 0)) return 0;
  return clamp((value / max) * 100, 0, 100);
}

/** SVG path ``M x y L x y ...`` through the points. */
export function linePath(points: readonly (readonly [number, number])[]): string {
  return points
    .map(([x, y], i) => `${i === 0 ? "M" : "L"}${x.toFixed(2)} ${y.toFixed(2)}`)
    .join(" ");
}

/** Closed SVG path for a band between ``upper`` and ``lower`` sharing x coordinates. */
export function bandPath(
  upper: readonly (readonly [number, number])[],
  lower: readonly (readonly [number, number])[],
): string {
  if (upper.length === 0) return "";
  const back = [...lower].reverse();
  return `${linePath([...upper, ...back])} Z`;
}
