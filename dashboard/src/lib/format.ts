/** Number, time and text formatting shared by every page. */

const DASH = "—";

/** ``$0.0123``: four decimals below a dollar, two above; ``—`` for null. */
export function formatCost(usd: number | null | undefined): string {
  if (usd === null || usd === undefined || !Number.isFinite(usd)) return DASH;
  if (usd === 0) return "$0.00";
  const sign = usd < 0 ? "-" : "";
  const abs = Math.abs(usd);
  // Pick the unit after rounding, so 0.99996 is "$1.00" rather than "$1.0000".
  if (abs >= 0.995) return `${sign}$${abs.toFixed(2)}`;
  if (abs < 0.00005) return `${sign}<$0.0001`;
  return `${sign}$${abs.toFixed(4)}`;
}

/** Milliseconds: ``0.42 ms``, ``12.3 ms``, ``845 ms``, ``1.93 s``, ``2m 05s``. */
export function formatMs(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) return DASH;
  const abs = Math.abs(ms);
  // Thresholds sit where the shorter format would round up to the next unit,
  // so 999.6 ms is "1.00 s" and 59 999.6 ms is "1m 00s", never "1000 ms" or "60.00 s".
  if (abs < 0.995) return `${ms.toFixed(2)} ms`;
  if (abs < 99.95) return `${ms.toFixed(1)} ms`;
  if (abs < 999.5) return `${ms.toFixed(0)} ms`;
  if (abs < 59_995) return `${(ms / 1000).toFixed(2)} s`;
  const totalSeconds = Math.round(abs / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${ms < 0 ? "-" : ""}${minutes}m ${String(seconds).padStart(2, "0")}s`;
}

/** A 0..1 fraction as a percentage: ``87.5%`` (``digits`` decimals, trailing zeros dropped). */
export function formatPercent(fraction: number | null | undefined, digits = 1): string {
  if (fraction === null || fraction === undefined || !Number.isFinite(fraction)) return DASH;
  const value = (fraction * 100).toFixed(digits);
  return `${digits > 0 ? value.replace(/\.?0+$/, "") : value}%`;
}

/** Integers with thousands separators: ``18,202``. */
export function formatInt(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH;
  return Math.round(value).toLocaleString("en-US");
}

/** A signed decimal with fixed digits: ``0.82``, ``-0.05``. */
export function formatFixed(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH;
  return value.toFixed(digits);
}

/** ISO timestamp as ``2026-10-10 21:48:22`` in UTC (stable across machines). */
export function formatTimestamp(iso: string | null | undefined): string {
  if (!iso) return DASH;
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toISOString().replace("T", " ").slice(0, 19);
}

/** The first ``n`` characters of an id (default 8, like ``git log --oneline``). */
export function shortId(id: string | null | undefined, n = 8): string {
  if (!id) return DASH;
  return id.slice(0, n);
}

/** Cut ``text`` to ``limit`` characters with an ellipsis. */
export function truncate(text: string, limit: number): string {
  return text.length > limit ? `${text.slice(0, Math.max(0, limit - 1))}…` : text;
}

/** A compact one-line preview of any JSON value. */
export function previewJson(value: unknown, limit = 60): string {
  return truncate(jsonText(value), limit);
}

function jsonText(value: unknown): string {
  // JSON.stringify returns undefined for undefined, functions and symbols.
  const text = JSON.stringify(value) as string | undefined;
  return text ?? "undefined";
}

/**
 * Previews of two JSON values that keep their first difference in view: when
 * the values share a prefix longer than the window, both previews start a few
 * characters before the point where they diverge, marked with a leading ellipsis.
 */
export function previewChange(base: unknown, head: unknown, limit = 60): [string, string] {
  const a = jsonText(base);
  const b = jsonText(head);
  if (a === b) return [truncate(a, limit), truncate(b, limit)];
  let common = 0;
  while (common < a.length && common < b.length && a[common] === b[common]) common += 1;
  const lead = 12;
  if (common + lead <= limit) return [truncate(a, limit), truncate(b, limit)];
  const start = common - lead;
  const window = (text: string) => `…${truncate(text.slice(start), Math.max(1, limit - 1))}`;
  return [window(a), window(b)];
}
