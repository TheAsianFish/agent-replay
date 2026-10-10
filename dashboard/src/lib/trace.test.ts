import { describe, expect, it } from "vitest";
import { runRecord } from "../test/fixtures";
import type { StepRecord } from "../api/types";
import { maxLatency, traceRows } from "./trace";

describe("traceRows", () => {
  it("nests steps under their parent span", () => {
    const rows = traceRows(runRecord.steps);
    expect(rows.map((r) => [r.step.index, r.depth])).toEqual([
      [0, 0],
      [1, 0],
      [2, 1],
    ]);
  });
  it("puts children right after their parent even when indexes interleave", () => {
    const base = runRecord.steps[0] as StepRecord;
    const steps: StepRecord[] = [
      { ...base, index: 0, span_id: "a", parent_span_id: null },
      { ...base, index: 1, span_id: "b", parent_span_id: null },
      { ...base, index: 2, span_id: "c", parent_span_id: "a" },
      { ...base, index: 3, span_id: "d", parent_span_id: "missing" },
    ];
    expect(traceRows(steps).map((r) => [r.step.index, r.depth])).toEqual([
      [0, 0],
      [2, 1],
      [1, 0],
      [3, 0],
    ]);
  });
  it("finds the slowest step", () => {
    expect(maxLatency(runRecord.steps)).toBe(800);
    expect(maxLatency([])).toBe(0);
  });
});
