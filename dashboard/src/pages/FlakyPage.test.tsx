import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { flakeReport, flakeSummaries, trend } from "../test/fixtures";
import { mockApi, renderAt } from "../test/render";
import { intervalMethod } from "./FlakyPage";

function api() {
  return mockApi({
    "/api/flake": { body: flakeSummaries },
    "/api/flake/trend": (url) => ({ body: trend.filter((p) => p.name === url.searchParams.get("scenario")) }),
    "/api/flake/*": { body: flakeReport },
  });
}

describe("FlakyPage", () => {
  it("names the interval method from stats_impl", () => {
    expect(intervalMethod("core")).toBe("Wilson");
    expect(intervalMethod("baseline")).toBe("Wald");
    expect(intervalMethod("other")).toBe("other");
  });

  it("shows the latest report's settings and scenarios with classification badges", async () => {
    api();
    renderAt("/flaky");
    expect(await screen.findByText("Wald interval")).toBeInTheDocument();
    expect(screen.getByText("(Wald intervals)")).toBeInTheDocument();
    expect(screen.getByText("sprt")).toBeInTheDocument();
    const rows = screen.getAllByRole("row").filter((r) => within(r).queryByText(/cancel_pending_laptop|modify_address_suite/, { selector: "button" }));
    expect(rows).toHaveLength(2);
    expect(within(rows[0] as HTMLElement).getByText("flaky")).toBeInTheDocument();
    expect(within(rows[0] as HTMLElement).getByText("62.5%")).toBeInTheDocument();
    expect(within(rows[0] as HTMLElement).getByText("[0.29, 0.96]")).toBeInTheDocument();
    expect(within(rows[0] as HTMLElement).getByText("required_tools (3)")).toBeInTheDocument();
    expect(within(rows[1] as HTMLElement).getByText("passing")).toBeInTheDocument();
  });

  it("charts the trend of the first unstable scenario with a band", async () => {
    api();
    renderAt("/flaky");
    const chart = await screen.findByRole("img", { name: /Pass rate of cancel_pending_laptop over 2 flake report/ });
    expect(chart.querySelector("[data-testid=ci-band]")).not.toBeNull();
    expect(chart.querySelectorAll("circle")).toHaveLength(2);
  });

  it("switches scenario", async () => {
    const { calls } = api();
    renderAt("/flaky");
    await screen.findByRole("img", { name: /cancel_pending_laptop/ });
    await userEvent.selectOptions(screen.getByRole("combobox", { name: "Scenario" }), "modify_address_suite");
    expect(await screen.findByText("No flake history for modify_address_suite.")).toBeInTheDocument();
    expect(calls.some((u) => u.searchParams.get("scenario") === "modify_address_suite")).toBe(true);
  });

  it("shows an empty state", async () => {
    mockApi({ "/api/flake": { body: [] } });
    renderAt("/flaky");
    expect(await screen.findByText("No flake reports stored.")).toBeInTheDocument();
  });

  it("shows an error from the report endpoint", async () => {
    mockApi({ "/api/flake": { body: flakeSummaries }, "/api/flake/*": { status: 404, body: { detail: "no flake report with id 'x'" } } });
    renderAt("/flaky");
    expect(await screen.findByRole("alert")).toHaveTextContent("no flake report with id 'x'");
  });
});
