import { describe, expect, it } from "vitest";
import { formatCost, formatFixed, formatInt, formatMs, formatPercent, formatTimestamp, previewChange, previewJson, shortId, truncate } from "./format";

describe("formatCost", () => {
  it("uses four decimals below a dollar", () => {
    expect(formatCost(0.0123)).toBe("$0.0123");
    expect(formatCost(0.29)).toBe("$0.2900");
  });
  it("uses two decimals from a dollar up", () => {
    expect(formatCost(1.5)).toBe("$1.50");
    expect(formatCost(12.345)).toBe("$12.35");
  });
  it("handles zero, tiny, negative and missing values", () => {
    expect(formatCost(0)).toBe("$0.00");
    expect(formatCost(0.00001)).toBe("<$0.0001");
    expect(formatCost(-0.05)).toBe("-$0.0500");
    expect(formatCost(null)).toBe("—");
    expect(formatCost(undefined)).toBe("—");
    expect(formatCost(Number.NaN)).toBe("—");
  });
  it("picks the unit after rounding", () => {
    expect(formatCost(0.99996)).toBe("$1.00");
    expect(formatCost(0.00004)).toBe("<$0.0001");
    expect(formatCost(0.00005)).toBe("$0.0001");
  });
});

describe("formatMs", () => {
  it("scales units with magnitude", () => {
    expect(formatMs(0.42)).toBe("0.42 ms");
    expect(formatMs(34.791)).toBe("34.8 ms");
    expect(formatMs(845.2)).toBe("845 ms");
    expect(formatMs(1925.378)).toBe("1.93 s");
    expect(formatMs(125_000)).toBe("2m 05s");
  });
  it("never shows a unit's rounding overflow", () => {
    expect(formatMs(99.96)).toBe("100 ms");
    expect(formatMs(999.6)).toBe("1.00 s");
    expect(formatMs(59_999.6)).toBe("1m 00s");
    expect(formatMs(119_600)).toBe("2m 00s");
    expect(formatMs(125_600)).toBe("2m 06s");
  });
  it("shows a dash for missing values", () => {
    expect(formatMs(null)).toBe("—");
  });
});

describe("formatPercent", () => {
  it("drops trailing zeros", () => {
    expect(formatPercent(1)).toBe("100%");
    expect(formatPercent(0.625)).toBe("62.5%");
    expect(formatPercent(0.92)).toBe("92%");
    expect(formatPercent(0.12345, 2)).toBe("12.35%");
  });
  it("supports zero digits and missing values", () => {
    expect(formatPercent(0.5, 0)).toBe("50%");
    expect(formatPercent(null)).toBe("—");
  });
});

describe("other formatters", () => {
  it("formats integers with separators", () => {
    expect(formatInt(18202)).toBe("18,202");
    expect(formatInt(undefined)).toBe("—");
  });
  it("formats fixed decimals", () => {
    expect(formatFixed(0.8234)).toBe("0.82");
    expect(formatFixed(-0.05)).toBe("-0.05");
  });
  it("formats timestamps in UTC", () => {
    expect(formatTimestamp("2026-10-10T21:48:22.395521Z")).toBe("2026-10-10 21:48:22");
    expect(formatTimestamp("not a date")).toBe("not a date");
    expect(formatTimestamp(null)).toBe("—");
  });
  it("shortens ids and truncates text", () => {
    expect(shortId("a80b4af3c9eb4d38")).toBe("a80b4af3");
    expect(shortId("a80b4af3c9eb4d38", 12)).toBe("a80b4af3c9eb");
    expect(truncate("abcdef", 4)).toBe("abc…");
    expect(truncate("abc", 4)).toBe("abc");
  });
  it("previews JSON values on one line", () => {
    expect(previewJson("hi")).toBe('"hi"');
    expect(previewJson({ a: 1 })).toBe('{"a":1}');
    expect(previewJson(undefined)).toBe("undefined");
    expect(previewJson("x".repeat(100), 10)).toHaveLength(10);
  });
});

describe("previewChange", () => {
  const prefix = "the quick brown fox jumps over the lazy dog and keeps running until ";

  it("shows where two values with a long shared prefix diverge", () => {
    const [before, after] = previewChange({ note: `${prefix}noon` }, { note: `${prefix}dusk` });
    expect(before).toBe('…nning until noon"}');
    expect(after).toBe('…nning until dusk"}');
    // previewJson would cut both to the same shared prefix.
    expect(previewJson({ note: `${prefix}noon` })).toBe(previewJson({ note: `${prefix}dusk` }));
  });
  it("truncates like previewJson when values differ early", () => {
    expect(previewChange("a@example.com", "b@example.com")).toEqual(['"a@example.com"', '"b@example.com"']);
    const [before, after] = previewChange("x".repeat(100), "y".repeat(100));
    expect(before).toBe(`"${"x".repeat(58)}…`);
    expect(after).toBe(`"${"y".repeat(58)}…`);
    expect([before, after]).toEqual([previewJson("x".repeat(100)), previewJson("y".repeat(100))]);
  });
  it("gives identical previews for identical values", () => {
    expect(previewChange({ a: 1 }, { a: 1 })).toEqual(['{"a":1}', '{"a":1}']);
    const [before, after] = previewChange({ a: prefix }, { a: prefix });
    expect(before).toBe(after);
  });
});
