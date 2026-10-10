import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { runPage } from "../test/fixtures";
import { mockApi, renderAt } from "../test/render";

const scenarios = { body: ["cancel_pending_laptop", "modify_address_suite"] };

describe("RunsPage", () => {
  it("lists runs with formatted values and the store header", async () => {
    mockApi({ "/api/runs": { body: runPage }, "/api/scenarios": scenarios });
    renderAt("/");
    const rows = await screen.findAllByRole("row");
    expect(rows).toHaveLength(3); // header + 2 runs
    expect(screen.getAllByText("$0.0123", { selector: "td" })).toHaveLength(2);
    expect(screen.getAllByText("34.8 ms", { selector: "td" })).toHaveLength(2);
    expect(within(rows[1] as HTMLElement).getByText("pass")).toBeInTheDocument();
    expect(within(rows[2] as HTMLElement).getByText("fail")).toBeInTheDocument();
    expect(await screen.findByTestId("health")).toHaveTextContent("sqlite");
    expect(screen.getByTestId("health")).toHaveTextContent("0002");
    expect(screen.getByText("1–2 of 2")).toBeInTheDocument();
  });

  it("sends filters to the API", async () => {
    const { calls } = mockApi({ "/api/runs": { body: runPage }, "/api/scenarios": scenarios });
    renderAt("/");
    await screen.findAllByRole("row");
    await userEvent.selectOptions(screen.getByRole("combobox", { name: "Result" }), "fail");
    await userEvent.selectOptions(screen.getByRole("combobox", { name: "Mode" }), "model-live");
    await userEvent.selectOptions(await screen.findByRole("combobox", { name: "Scenario" }), "cancel_pending_laptop");
    await waitFor(() => {
      const last = calls.filter((u) => u.pathname === "/api/runs").at(-1);
      expect(last?.searchParams.get("passed")).toBe("false");
      expect(last?.searchParams.get("mode")).toBe("model-live");
      expect(last?.searchParams.get("scenario")).toBe("cancel_pending_laptop");
    });
  });

  it("pages with offset", async () => {
    const { calls } = mockApi({
      "/api/runs": (url) => ({ body: { total: 60, items: url.searchParams.get("offset") === "25" ? [runPage.items[1]] : runPage.items } }),
      "/api/scenarios": scenarios,
    });
    renderAt("/");
    await screen.findByText("1–2 of 60");
    await userEvent.click(screen.getByRole("button", { name: "Older →" }));
    await screen.findByText("26–26 of 60");
    expect(calls.some((u) => u.searchParams.get("offset") === "25")).toBe(true);
  });

  it("filters by suite from the URL", async () => {
    const { calls } = mockApi({ "/api/runs": { body: runPage }, "/api/scenarios": scenarios, "/api/suites/*": { status: 404, body: { detail: "gone" } } });
    renderAt("/?suite_id=suite000111");
    await screen.findAllByRole("row");
    expect(screen.getByText(/Showing runs of suite/)).toBeInTheDocument();
    expect(calls.find((u) => u.pathname === "/api/runs")?.searchParams.get("suite_id")).toBe("suite000111");
  });

  it("shows an empty state", async () => {
    mockApi({ "/api/runs": { body: { total: 0, items: [] } }, "/api/scenarios": { body: [] } });
    renderAt("/");
    expect(await screen.findByText("No runs stored yet.")).toBeInTheDocument();
  });

  it("shows the API's detail on error", async () => {
    mockApi({ "/api/runs": { status: 500, body: { detail: "database is locked; retry" } }, "/api/scenarios": scenarios });
    renderAt("/");
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("database is locked; retry");
    expect(alert).toHaveTextContent("HTTP 500");
  });

  it("navigates to the run on row click", async () => {
    mockApi({ "/api/runs": { body: runPage }, "/api/scenarios": scenarios, "/api/runs/*": { status: 404, body: { detail: "x" } } });
    const { router } = renderAt("/");
    const rows = await screen.findAllByRole("row");
    await userEvent.click(within(rows[2] as HTMLElement).getByText("full"));
    expect(router.state.location.pathname).toBe(`/runs/${runPage.items[1]?.id ?? ""}`);
  });
});
