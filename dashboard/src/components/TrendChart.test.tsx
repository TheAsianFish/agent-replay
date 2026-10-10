import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { FlakePoint } from "../api/types";
import { trend } from "../test/fixtures";
import { TrendChart, trendGeometry, xLabels } from "./TrendChart";

const first = trend[0] as FlakePoint;

function at(report_id: string, started_at: string, overrides: Partial<FlakePoint> = {}): FlakePoint {
  return { ...first, report_id, started_at, ...overrides };
}

/** The ``x y`` pairs of an SVG path made of ``M``/``L`` commands. */
function pathPoints(d: string): [number, number][] {
  return [...d.matchAll(/[ML](-?[\d.]+) (-?[\d.]+)/g)].map((m) => [Number(m[1]), Number(m[2])]);
}

function timeLabels(container: HTMLElement): string[] {
  return [...container.querySelectorAll("text")].map((t) => t.textContent).filter((t) => /^\d\d-\d\d /.test(t));
}

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

describe("TrendChart", () => {
  it("draws the band with ci_high on the upper edge and ci_low on the lower edge", () => {
    const points = [
      at("r1", "2026-10-09T10:00:00Z", { pass_rate: 0.5, ci_low: 0.2, ci_high: 0.7 }),
      at("r2", "2026-10-10T10:00:00Z", { pass_rate: 0.6, ci_low: 0.3, ci_high: 0.9 }),
    ];
    const { container } = render(<TrendChart points={points} scenario="s" />);
    const d = container.querySelector("[data-testid=ci-band]")?.getAttribute("d") ?? "";
    const geo = trendGeometry(points);
    const [x0, x1] = geo.points.map((p) => Number(p.cx.toFixed(2)));
    const y = (v: number) => Number(geo.y(v).toFixed(2));
    // Upper edge left to right, then the lower edge back.
    expect(pathPoints(d)).toEqual([
      [x0, y(0.7)],
      [x1, y(0.9)],
      [x1, y(0.3)],
      [x0, y(0.2)],
    ]);
    expect(d.endsWith("Z")).toBe(true);
  });

  it("draws a whisker for a single rated point", () => {
    const point = at("r1", "2026-10-09T10:00:00Z", { pass_rate: 0.5, ci_low: 0.2, ci_high: 0.7 });
    const { container } = render(<TrendChart points={[point]} scenario="s" />);
    const whisker = container.querySelector("[data-testid=ci-band]");
    expect(whisker?.tagName).toBe("line");
    const geo = trendGeometry([point]);
    expect(Number(whisker?.getAttribute("y1"))).toBe(geo.y(0.7));
    expect(Number(whisker?.getAttribute("y2"))).toBe(geo.y(0.2));
  });

  it("labels times with seconds when every report is within an hour", () => {
    const points = [
      at("r1", "2026-10-09T10:00:00Z"),
      at("r2", "2026-10-09T10:00:01Z"),
      at("r3", "2026-10-09T10:30:00Z"),
      at("r4", "2026-10-09T10:59:59Z"),
    ];
    const { container } = render(<TrendChart points={points} scenario="s" />);
    // r2 sits a fraction of a pixel right of r1, so it gets no label.
    expect(timeLabels(container)).toEqual(["10-09 10:00:00", "10-09 10:30:00", "10-09 10:59:59"]);
  });

  it("labels times to the minute over longer spans", () => {
    const { container } = render(<TrendChart points={trend} scenario="s" />);
    expect(timeLabels(container)).toEqual(["10-09 10:00", "10-10 10:00"]);
  });
});

describe("xLabels", () => {
  it("labels the first point and then only points at least 120 px past the last label", () => {
    const points = [60, 100, 179, 181, 300, 301, 500].map((cx) => ({ cx }));
    expect(xLabels(points).map((p) => p.cx)).toEqual([60, 181, 301, 500]);
  });

  it("always labels a lone point and nothing for no points", () => {
    expect(xLabels([{ cx: 388 }])).toEqual([{ cx: 388 }]);
    expect(xLabels([])).toEqual([]);
  });
});
