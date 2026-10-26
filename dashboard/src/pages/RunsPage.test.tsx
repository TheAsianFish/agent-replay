import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { passingRun, runPage, runSummary } from "../test/fixtures";
import { type Canned, mockApi, renderAt } from "../test/render";

const scenarios = { body: ["cancel_pending_laptop", "modify_address_suite"] };

function runsCalls(calls: URL[]): URL[] {
  return calls.filter((u) => u.pathname === "/api/runs");
}

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

  it("shows unpriced runs as unpriced, not as free", async () => {
    const unpriced = runSummary({ id: "cccccccc11112222333344445555666", cost_usd: 0, unpriced_calls: 4 });
    mockApi({ "/api/runs": { body: { total: 2, items: [passingRun, unpriced] } }, "/api/scenarios": scenarios });
    renderAt("/");
    expect(await screen.findByText("unpriced", { selector: "td" })).toBeInTheDocument();
    expect(screen.getByText("$0.0123", { selector: "td" })).toBeInTheDocument();
    expect(screen.queryByText("$0.00", { selector: "td" })).not.toBeInTheDocument();
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

  it("goes back to the first page when a filter changes", async () => {
    const { calls } = mockApi({ "/api/runs": { body: { total: 60, items: runPage.items } }, "/api/scenarios": scenarios });
    const { router } = renderAt("/?offset=25");
    await screen.findByText("26–27 of 60");
    expect(runsCalls(calls).at(-1)?.searchParams.get("offset")).toBe("25");
    await userEvent.selectOptions(screen.getByRole("combobox", { name: "Mode" }), "full");
    await waitFor(() => {
      const last = runsCalls(calls).at(-1);
      expect(last?.searchParams.get("mode")).toBe("full");
      expect(last?.searchParams.get("offset")).toBe("0");
    });
    expect(router.state.location.search).toBe("?mode=full");
  });

  it("disables Older when the page reaches the total", async () => {
    mockApi({ "/api/runs": { body: runPage }, "/api/scenarios": scenarios });
    renderAt("/");
    await screen.findByText("1–2 of 2");
    expect(screen.getByRole("button", { name: "Older →" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "← Newer" })).toBeDisabled();
  });

  it("asks for the previous page of 25 when going newer", async () => {
    const { calls } = mockApi({ "/api/runs": { body: { total: 60, items: runPage.items } }, "/api/scenarios": scenarios });
    renderAt("/?offset=50");
    await screen.findByText("51–52 of 60");
    await userEvent.click(screen.getByRole("button", { name: "← Newer" }));
    await screen.findByText("26–27 of 60");
    expect(runsCalls(calls).at(-1)?.searchParams.get("offset")).toBe("25");
  });

  it("offers the first page when the URL offset is past the end", async () => {
    const { calls } = mockApi({
      "/api/runs": (url) => ({ body: url.searchParams.get("offset") === "0" ? runPage : { total: 2, items: [] } }),
      "/api/scenarios": scenarios,
    });
    const { router } = renderAt("/?offset=50");
    expect(await screen.findByText("No runs at offset 50; the store has 2.")).toBeInTheDocument();
    expect(screen.queryByText("No runs stored yet.")).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "← Back to the first page" }));
    await screen.findByText("1–2 of 2");
    expect(runsCalls(calls).at(-1)?.searchParams.get("offset")).toBe("0");
    expect(router.state.location.search).toBe("");
  });

  it("says when filters match nothing and clears them", async () => {
    mockApi({ "/api/runs": { body: { total: 0, items: [] } }, "/api/scenarios": scenarios });
    const { router } = renderAt("/?mode=full&result=fail");
    expect(await screen.findByText("No runs match these filters.")).toBeInTheDocument();
    expect(screen.getByText("Clear a filter to see more runs.")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Clear filters" }));
    expect(router.state.location.search).toBe("");
    expect(await screen.findByText("No runs stored yet.")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Clear filters" })).not.toBeInTheDocument();
  });

  it("does not send empty URL params to the API", async () => {
    const { calls } = mockApi({ "/api/runs": { body: runPage }, "/api/scenarios": scenarios });
    renderAt("/?scenario=&mode=full&suite_id=");
    await screen.findAllByRole("row");
    const last = runsCalls(calls).at(-1);
    expect(last?.searchParams.has("scenario")).toBe(false);
    expect(last?.searchParams.has("suite_id")).toBe(false);
    expect(last?.searchParams.get("mode")).toBe("full");
    expect(screen.queryByText(/Showing runs of suite/)).not.toBeInTheDocument();
  });

  it("marks the table busy and disables paging while the next page loads", async () => {
    const pending: ((canned: Canned) => void)[] = [];
    mockApi({
      "/api/runs": (url) =>
        url.searchParams.get("offset") === "25"
          ? new Promise<Canned>((resolve) => { pending.push(resolve); })
          : { body: { total: 60, items: runPage.items } },
      "/api/scenarios": scenarios,
    });
    renderAt("/");
    await screen.findByText("1–2 of 60");
    const container = screen.getByRole("table").parentElement as HTMLElement;
    expect(container).toHaveAttribute("aria-busy", "false");
    await userEvent.click(screen.getByRole("button", { name: "Older →" }));
    await waitFor(() => { expect(pending).toHaveLength(1); });
    expect(container).toHaveAttribute("aria-busy", "true");
    expect(screen.getByRole("button", { name: "Older →" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "← Newer" })).toBeDisabled();
    for (const resolve of pending) resolve({ body: { total: 60, items: [runPage.items[1]] } });
    await screen.findByText("26–26 of 60");
    expect(container).toHaveAttribute("aria-busy", "false");
    expect(screen.getByRole("button", { name: "← Newer" })).toBeEnabled();
  });
});
