import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { kappaDomain } from "../components/KappaChart";
import { calibrationView, calibrations } from "../test/fixtures";
import { mockApi, renderAt } from "../test/render";

describe("Calibration pages", () => {
  it("lists reports and opens one", async () => {
    mockApi({ "/api/calibration": { body: calibrations }, "/api/calibration/*": { body: calibrationView } });
    const { router } = renderAt("/calibration");
    const link = await screen.findByRole("link", { name: "retail_support" });
    expect(screen.getByText("anthropic:claude-opus-5-5")).toBeInTheDocument();
    expect(screen.getByText("120")).toBeInTheDocument();
    await userEvent.click(link);
    expect(router.state.location.pathname).toBe("/calibration/cal1aaaabbbbcccc");
    expect(await screen.findByTestId("method")).toHaveTextContent("Cohen's kappa, 95% bootstrap CI");
  });

  it("shows each criterion with a gate badge, whisker and threshold", async () => {
    mockApi({ "/api/calibration/*": { body: calibrationView } });
    renderAt("/calibration/cal1aaaabbbbcccc");
    await screen.findByTestId("method");
    expect(screen.getByText("(1 of 2 gating)")).toBeInTheDocument();
    const row = (name: string) => screen.getByText(name, { selector: "td" }).closest("tr") as HTMLElement;
    expect(within(row("resolves_request")).getByText("gating")).toBeInTheDocument();
    expect(within(row("resolves_request")).getByText("92%")).toBeInTheDocument();
    expect(within(row("tone")).getByText("advisory")).toBeInTheDocument();
    expect(within(row("tone")).getByText("[0.12, 0.50]")).toBeInTheDocument();
    const whisker = within(row("tone")).getByRole("img", { name: /tone: kappa 0.31/ });
    expect(whisker.querySelector("[data-testid=min-kappa-line]")).not.toBeNull();
  });

  it("explains how to add a report when there are none", async () => {
    mockApi({ "/api/calibration": { body: [] } });
    renderAt("/calibration");
    expect(await screen.findByText("No calibration reports stored.")).toBeInTheDocument();
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
  });
});
