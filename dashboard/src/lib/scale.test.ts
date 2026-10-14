import { describe, expect, it } from "vitest";
import { bandPath, barPercent, clamp, linePath, linearScale, ticks } from "./scale";

describe("linearScale", () => {
  it("maps the domain onto the range, including inverted ranges", () => {
    const y = linearScale([0, 1], [200, 0]);
    expect(y(0)).toBe(200);
    expect(y(1)).toBe(0);
    expect(y(0.25)).toBe(150);
  });
  it("maps a degenerate domain to the range midpoint", () => {
    const x = linearScale([5, 5], [0, 100]);
    expect(x(5)).toBe(50);
  });
});

describe("helpers", () => {
  it("clamps", () => {
    expect(clamp(-2, -1, 1)).toBe(-1);
    expect(clamp(0.3, -1, 1)).toBe(0.3);
    expect(clamp(4, -1, 1)).toBe(1);
  });
  it("builds evenly spaced ticks without float noise", () => {
    expect(ticks(0, 1, 4)).toEqual([0, 0.25, 0.5, 0.75, 1]);
    expect(ticks(-1, 1, 4)).toEqual([-1, -0.5, 0, 0.5, 1]);
    expect(ticks(0, 1, 5)).toEqual([0, 0.2, 0.4, 0.6, 0.8, 1]);
    expect(ticks(3, 3, 4)).toEqual([3]);
  });
  it("computes bar widths relative to the maximum", () => {
    expect(barPercent(50, 200)).toBe(25);
    expect(barPercent(300, 200)).toBe(100);
    expect(barPercent(null, 200)).toBe(0);
    expect(barPercent(10, 0)).toBe(0);
  });
  it("builds line and band paths", () => {
    expect(linePath([[0, 1], [2, 3]])).toBe("M0.00 1.00 L2.00 3.00");
    expect(bandPath([[0, 0], [1, 0]], [[0, 1], [1, 1]])).toBe("M0.00 0.00 L1.00 0.00 L1.00 1.00 L0.00 1.00 Z");
    expect(bandPath([], [])).toBe("");
  });
});
