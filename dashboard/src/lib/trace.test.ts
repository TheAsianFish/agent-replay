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
  it("sorts children by index when they arrive out of order", () => {
    const base = runRecord.steps[0] as StepRecord;
    const steps: StepRecord[] = [
      { ...base, index: 0, span_id: "a", parent_span_id: null },
      { ...base, index: 3, span_id: "d", parent_span_id: "a" },
      { ...base, index: 1, span_id: "b", parent_span_id: "a" },
      { ...base, index: 2, span_id: "c", parent_span_id: "a" },
    ];
    expect(traceRows(steps).map((r) => [r.step.index, r.depth])).toEqual([
      [0, 0],
      [1, 1],
      [2, 1],
      [3, 1],
    ]);
  });
  it("sorts top-level steps by index when they arrive shuffled", () => {
    const base = runRecord.steps[0] as StepRecord;
    const steps: StepRecord[] = [2, 0, 3, 1].map((index) => ({ ...base, index, span_id: `s${String(index)}`, parent_span_id: null }));
    expect(traceRows(steps).map((r) => [r.step.index, r.depth])).toEqual([
      [0, 0],
      [1, 0],
      [2, 0],
      [3, 0],
    ]);
  });
  it("keeps steps whose parents form a cycle", () => {
    const base = runRecord.steps[0] as StepRecord;
    const steps: StepRecord[] = [
      { ...base, index: 0, span_id: "root", parent_span_id: null },
      { ...base, index: 2, span_id: "y", parent_span_id: "x" },
      { ...base, index: 1, span_id: "x", parent_span_id: "y" },
    ];
    const rows = traceRows(steps);
    expect(rows).toHaveLength(3);
    expect(rows.map((r) => r.step.index).sort()).toEqual([0, 1, 2]);
    expect(rows[0]).toEqual({ step: steps[0], depth: 0 });
    // The first cycle member by index starts a tree at depth 0; its partner nests under it.
    expect(rows.map((r) => [r.step.index, r.depth])).toEqual([
      [0, 0],
      [1, 0],
      [2, 1],
    ]);
  });
  it("finds the slowest step", () => {
    expect(maxLatency(runRecord.steps)).toBe(800);
    expect(maxLatency([])).toBe(0);
  });
});
