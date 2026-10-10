import { describe, expect, it } from "vitest";
import { trend } from "../test/fixtures";
import { trendGeometry } from "./TrendChart";

describe("trendGeometry", () => {
  it("places points by time on x and pass rate on y", () => {
    const geo = trendGeometry(trend);
    const [first, second] = geo.points;
    expect(first?.cx).toBeLessThan(second?.cx ?? 0);
    expect(geo.y(1)).toBeLessThan(geo.y(0)); // SVG y grows downwards
    expect(first?.cy).toBe(geo.y(1));
    expect(second?.cy).toBe(geo.y(0.625));
  });

  it("centres a single report and skips points without a pass rate", () => {
    const one = trend[0];
    if (!one) throw new Error("fixture missing");
    const geo = trendGeometry([one, { ...one, report_id: "r9", pass_rate: null }]);
    expect(geo.points[0]?.cx).toBe(geo.points[1]?.cx);
    expect(geo.points[1]?.cy).toBeNull();
  });
});
