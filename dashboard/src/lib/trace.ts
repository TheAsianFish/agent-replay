import type { StepRecord } from "../api/types";

export interface TraceRow {
  step: StepRecord;
  depth: number;
}

/**
 * Steps in display order: top-level steps by index, each followed by the steps
 * nested under it (``parent_span_id``), recursively. A step whose parent is not
 * in the run is shown at the top level.
 */
export function traceRows(steps: readonly StepRecord[]): TraceRow[] {
  const bySpan = new Map<string, StepRecord>();
  for (const step of steps) if (step.span_id) bySpan.set(step.span_id, step);
  const children = new Map<string, StepRecord[]>();
  const roots: StepRecord[] = [];
  for (const step of steps) {
    const parent = step.parent_span_id;
    if (parent && bySpan.has(parent) && parent !== step.span_id) {
      const list = children.get(parent) ?? [];
      list.push(step);
      children.set(parent, list);
    } else {
      roots.push(step);
    }
  }
  const byIndex = (a: StepRecord, b: StepRecord) => a.index - b.index;
  const out: TraceRow[] = [];
  const seen = new Set<StepRecord>();
  const visit = (step: StepRecord, depth: number) => {
    if (seen.has(step)) return;
    seen.add(step);
    out.push({ step, depth });
    const kids = step.span_id ? children.get(step.span_id) : undefined;
    if (kids) for (const kid of [...kids].sort(byIndex)) visit(kid, depth + 1);
  };
  for (const root of [...roots].sort(byIndex)) visit(root, 0);
  // Steps whose parents form a cycle reach no root; show them rather than lose them.
  for (const step of [...steps].sort(byIndex)) if (!seen.has(step)) visit(step, 0);
  return out;
}

/** The slowest step's latency, or 0 when none is recorded. */
export function maxLatency(steps: readonly { latency_ms?: number | null }[]): number {
  return steps.reduce((max, s) => Math.max(max, s.latency_ms ?? 0), 0);
}
