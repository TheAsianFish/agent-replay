import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { kappaDomain } from "../components/KappaChart";
import type { CalibrationView } from "../api/types";
import { calibrationView, calibrationViewV2, calibrations } from "../test/fixtures";
import { mockApi, renderAt } from "../test/render";

describe("Calibration pages", () => {
  it("lists reports and opens one", async () => {
    mockApi({ "/api/calibration": { body: calibrations }, "/api/calibration/*": { body: calibrationView } });
    const { router } = renderAt("/calibration");
    const link = await screen.findByRole("link", { name: "retail_support" });
    const row = link.closest("tr") as HTMLElement;
    expect(within(row).getByText("anthropic:claude-opus-5-5")).toBeInTheDocument();
    expect(within(row).getByText("120")).toBeInTheDocument();
    await userEvent.click(link);
    expect(router.state.location.pathname).toBe("/calibration/cal1aaaabbbbcccc");
    expect(await screen.findByTestId("method")).toHaveTextContent("Cohen's kappa, 95% bootstrap CI");
  });

  it("lists the reference and repeats, muting reports that never gate", async () => {
    mockApi({ "/api/calibration": { body: calibrations } });
    renderAt("/calibration");
    await screen.findByRole("link", { name: "retail_support" });
    const row = (name: string) => screen.getByRole("link", { name }).closest("tr") as HTMLElement;
    const cells = (name: string) => [...row(name).querySelectorAll("td")].map((c) => c.textContent);
    expect(screen.getByRole("columnheader", { name: "Reference" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Repeats" })).toBeInTheDocument();
    expect(cells("retail_support").slice(5, 7)).toEqual(["human", "1"]);
    expect(within(row("retail_support")).queryByText("baseline stats")).toBeNull();
    expect(cells("retail_support_objective").slice(5, 7)).toEqual(["objectivebaseline stats", "3"]);
    expect(within(row("retail_support_objective")).getByText("objective")).toHaveClass("text-slate-500");
    expect(within(row("retail_support_objective")).getByText("baseline stats")).toHaveClass("text-slate-500");
  });

  it("shows each criterion with a gate badge, whisker and threshold", async () => {
    mockApi({ "/api/calibration/*": { body: calibrationViewV2 } });
    renderAt("/calibration/cal2ddddeeeeffff");
    await screen.findByTestId("method");
    expect(screen.getByText("(1 of 4 gating)")).toBeInTheDocument();
    const criteria = within(screen.getByRole("table", { name: "Criteria" }));
    const row = (name: string) => criteria.getByText(name, { selector: "td" }).closest("tr") as HTMLElement;
    expect(within(row("resolves_request")).getByText("gating")).toBeInTheDocument();
    expect(within(row("resolves_request")).getByText("91.7%")).toBeInTheDocument();
    expect(within(row("tone")).getByText("advisory")).toBeInTheDocument();
    expect(within(row("tone")).getByText("[-0.18, 0.63]")).toBeInTheDocument();
    const whisker = within(row("tone")).getByRole("img", { name: /tone: kappa 0.25/ });
    expect(whisker.querySelector("[data-testid=min-kappa-line]")).not.toBeNull();
  });

  it("takes each gate from the server rather than recomputing it from the interval", async () => {
    const criteria = calibrationView.record.report.criteria;
    const view: CalibrationView = {
      ...calibrationView,
      // The server says the reverse of what ci_low >= min_kappa would give for every criterion.
      gates: { resolves_request: { gating: false }, tone: { gating: true }, extra: { gating: false } },
      record: {
        ...calibrationView.record,
        report: {
          ...calibrationView.record.report,
          criteria: { ...criteria, extra: { kappa: 0.95, ci_low: 0.9, ci_high: 0.99, agreement: 0.97, n: 120 } },
        },
      },
    };
    mockApi({ "/api/calibration/*": { body: view } });
    renderAt("/calibration/cal1aaaabbbbcccc");
    await screen.findByTestId("method");
    const row = (name: string) => screen.getByText(name, { selector: "td" }).closest("tr") as HTMLElement;
    expect(within(row("extra")).getByText("[0.90, 0.99]")).toBeInTheDocument();
    expect(within(row("extra")).getByText("advisory")).toBeInTheDocument();
    expect(within(row("resolves_request")).getByText("advisory")).toBeInTheDocument();
    expect(within(row("tone")).getByText("gating")).toBeInTheDocument();
    expect(screen.getByText("(1 of 3 gating)")).toBeInTheDocument();
  });

  it("explains how to add a report when there are none", async () => {
    mockApi({ "/api/calibration": { body: [] } });
    renderAt("/calibration");
    expect(await screen.findByText("No calibration reports stored.")).toBeInTheDocument();
    expect(screen.getByText("replay calibrate GOLDEN_DIR --rubric RUBRIC --store")).toBeInTheDocument();
    expect(screen.getByText("replay db add-calibration REPORT.json")).toBeInTheDocument();
  });

  it("shows the API's detail for an unknown report", async () => {
    mockApi({ "/api/calibration/*": { status: 404, body: { detail: "no calibration report with id 'nope'" } } });
    renderAt("/calibration/nope");
    expect(await screen.findByRole("alert")).toHaveTextContent("no calibration report with id 'nope'");
  });

  it("picks a 0..1 axis unless an interval goes negative", () => {
    const c = { kappa: 0.5, ci_low: 0.2, ci_high: 0.7, agreement: 0.8, n: 10 };
    expect(kappaDomain([c], 0.6)).toEqual([0, 1]);
    expect(kappaDomain([c, { ...c, ci_low: -0.2 }], 0.6)).toEqual([-1, 1]);
    expect(kappaDomain([c, { ...c, kappa: null, ci_low: null, ci_high: null }], 0.6)).toEqual([0, 1]);
  });

  it("renders a schema 1 report as blocked, without the schema 2 sections", async () => {
    mockApi({ "/api/calibration/*": { body: calibrationView } });
    renderAt("/calibration/cal1aaaabbbbcccc");
    await screen.findByTestId("method");
    expect(screen.getByText("schema 1")).toBeInTheDocument();
    expect(screen.getByText("human labels")).toBeInTheDocument();
    // A schema 1 report does not say which statistics made it, so the server blocks it.
    expect(screen.getByTestId("blocked")).toHaveTextContent("statistics that it does not name");
    expect(screen.getByText("(0 of 2 gating)")).toBeInTheDocument();
    expect(screen.queryByText("Repeats")).toBeNull();
    expect(screen.queryByText("Judge × reference")).toBeNull();
    expect(screen.queryByText("Self-consistency")).toBeNull();
    expect(screen.queryByText(/majority/)).toBeNull();
    expect(screen.queryByRole("heading", { name: "Bias probes" })).toBeNull();
  });
});

describe("Calibration page, schema 2", () => {
  const row = (name: string) =>
    within(screen.getByRole("table", { name: "Criteria" })).getByText(name, { selector: "td" }).closest("tr") as HTMLElement;
  const open = async (view: CalibrationView = calibrationViewV2) => {
    mockApi({ "/api/calibration/*": { body: view } });
    renderAt("/calibration/cal2ddddeeeeffff");
    await screen.findByTestId("method");
  };

  it("shows the method facts and reference and statistics badges", async () => {
    await open();
    const method = screen.getByRole("region", { name: "Method" });
    expect(within(method).getByText("human labels")).toBeInTheDocument();
    expect(within(method).getByText("core stats")).toBeInTheDocument();
    expect(within(method).getByText("reviewer-a")).toBeInTheDocument();
    expect(within(method).getByText("Repeats").nextElementSibling).toHaveTextContent("3");
    expect(within(method).getByText("Golden set").nextElementSibling).toHaveTextContent("eval/golden/retailsha256:9c41e07a");
    expect(within(method).getByText("18 live · 270 cached")).toBeInTheDocument();
    expect(within(method).getByText("$0.4312")).toBeInTheDocument();
    expect(screen.queryByTestId("blocked")).toBeNull();
    expect(screen.getByText("(1 of 4 gating)")).toBeInTheDocument();
  });

  it("shows an undefined kappa without a whisker", async () => {
    await open();
    const r = row("follows_policy");
    expect(within(r).getByTitle(/no room for chance disagreement/)).toHaveTextContent("undefined");
    const whisker = within(r).getByRole("img", { name: "follows_policy: kappa undefined, threshold 0.60" });
    expect(whisker.querySelector("[data-testid=kappa-interval]")).toBeNull();
    expect(whisker.querySelector("circle")).toBeNull();
    expect(whisker.querySelector("[data-testid=min-kappa-line]")).not.toBeNull();
    expect(within(row("tone")).getByRole("img", { name: /tone: kappa 0.25/ }).querySelector("[data-testid=kappa-interval]")).not.toBeNull();
  });

  it("shows agreement next to the majority baseline", async () => {
    await open();
    expect(row("resolves_request")).toHaveTextContent("91.7% vs 62.5% majority");
    // 75% does not beat a judge that always says "pass".
    expect(within(row("tone")).getByTestId("agreement")).toHaveClass("text-amber-800");
    expect(within(row("resolves_request")).getByTestId("agreement")).not.toHaveClass("text-amber-800");
  });

  it("shows the whole-run criterion by name and never as gating", async () => {
    await open();
    // kappa 1.0 on 24 items would gate a rubric criterion; the whole-run entry is not one.
    expect(within(row("whole run")).getByText("advisory")).toBeInTheDocument();
    expect(screen.queryByText("_overall")).toBeNull();
  });

  it("shows a confusion matrix per criterion and highlights false passes", async () => {
    await open();
    const matrix = within(row("tone")).getByRole("table", { name: "tone: judge verdicts against reference labels" });
    expect(within(matrix).getAllByRole("columnheader").map((h) => h.textContent)).toEqual(["ref pass", "ref fail"]);
    expect(within(matrix).getAllByRole("rowheader").map((h) => h.textContent)).toEqual(["judge pass", "judge fail"]);
    const falsePass = within(matrix).getByTestId("false-pass");
    expect(falsePass).toHaveTextContent("4");
    expect(falsePass).toHaveClass("bg-red-50");
    expect(within(matrix).getByTestId("false-fail")).toHaveTextContent("2");
    expect(within(row("follows_policy")).getByTestId("false-pass")).not.toHaveClass("bg-red-50");
  });

  it("shows self-consistency over repeated judge runs", async () => {
    await open();
    expect(row("resolves_request")).toHaveTextContent("95.8% unanimouspairwise κ 0.91 · 3 runs");
    expect(row("follows_policy")).toHaveTextContent("100% unanimouspairwise κ undefined · 3 runs");
  });

  it("shows flip probes per criterion", async () => {
    await open();
    expect(screen.getByRole("heading", { name: "Bias probes" })).toBeInTheDocument();
    const position = screen.getByRole("region", { name: "Position" });
    expect(within(position).getByText(/depends on where the criterion sits/)).toBeInTheDocument();
    const tone = within(within(position).getByRole("table", { name: "Position flips per criterion" })).getByText("tone").closest("tr") as HTMLElement;
    expect(within(tone).getByText("5 / 24")).toBeInTheDocument();
    expect(within(tone).getByText("20.8%")).toBeInTheDocument();
    expect(within(tone).getByText("[9.2%, 40.5%]")).toBeInTheDocument();
    expect(tone.querySelectorAll("td")[4]).toHaveTextContent("4");
    expect(tone.querySelectorAll("td")[5]).toHaveTextContent("1");
    const verbosity = screen.getByRole("region", { name: "Verbosity" });
    expect(within(verbosity).getByText("whole run")).toBeInTheDocument();
    expect(within(verbosity).getByText("3 / 24")).toBeInTheDocument();
  });

  it("shows the self-preference groups", async () => {
    await open();
    const probe = screen.getByRole("region", { name: "Self-preference" });
    const table = within(probe).getByRole("table", { name: "Self-preference groups per criterion" });
    const same = within(table).getByText("same model as judge").closest("tr") as HTMLElement;
    expect(within(same).getByText("resolves_request")).toBeInTheDocument();
    expect([...same.querySelectorAll("td")].map((c) => c.textContent)).toEqual([
      "resolves_request",
      "same model as judge",
      "12",
      "66.7%",
      "58.3%",
      "20% of 5",
    ]);
    const other = within(table).getByText("other models").closest("tr") as HTMLElement;
    expect(other).toHaveTextContent("0% of 4");
  });

  it("shows a skipped probe as a note", async () => {
    const report = calibrationViewV2.record.report;
    await open({
      ...calibrationViewV2,
      record: {
        ...calibrationViewV2.record,
        report: {
          ...report,
          probes: [
            ...report.probes.slice(0, 2),
            {
              probe: "self_preference",
              description: "Judge leniency on runs its own model produced against runs from other models.",
              skipped: "every golden run was produced by the judge's model",
            },
          ],
        },
      },
    });
    const probe = screen.getByRole("region", { name: "Self-preference" });
    expect(within(probe).getByText("Skipped: every golden run was produced by the judge's model")).toBeInTheDocument();
    expect(within(probe).queryByRole("table")).toBeNull();
  });

  it("explains why a blocked report never gates", async () => {
    const blocked =
      "the calibration report compares the judge with objective labels, not human ones; label the golden set with `replay label` and recalibrate";
    await open({
      ...calibrationViewV2,
      blocked,
      gates: Object.fromEntries(Object.keys(calibrationViewV2.gates).map((k) => [k, { gating: false }])),
      record: {
        ...calibrationViewV2.record,
        report: { ...calibrationViewV2.record.report, reference: "objective", labeller: null },
      },
    });
    const banner = screen.getByTestId("blocked");
    expect(banner).toHaveAttribute("role", "note");
    expect(banner).toHaveTextContent("No criterion of this report can gate");
    expect(banner).toHaveTextContent(blocked);
    expect(screen.getByText("objective labels")).toBeInTheDocument();
    expect(screen.getByText("(0 of 4 gating)")).toBeInTheDocument();
    expect(within(row("resolves_request")).getByText("advisory")).toBeInTheDocument();
  });

  it("marks baseline statistics as raw agreement", async () => {
    await open({
      ...calibrationViewV2,
      blocked: "the calibration report was computed with REPLAY_STATS_IMPL=baseline (raw agreement, not kappa); recalibrate with REPLAY_STATS_IMPL=core",
      record: { ...calibrationViewV2.record, report: { ...calibrationViewV2.record.report, stats_impl: "baseline" } },
    });
    expect(screen.getByText("baseline stats: raw agreement, not kappa")).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Kappa*" })).toBeInTheDocument();
    expect(screen.getByText(/the kappa columns hold raw agreement, not kappa/)).toBeInTheDocument();
  });
});
