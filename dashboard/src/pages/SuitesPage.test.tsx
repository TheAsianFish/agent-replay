import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { suites } from "../test/fixtures";
import { mockApi, renderAt } from "../test/render";

describe("SuitesPage", () => {
  it("lists suite runs and links to their runs", async () => {
    mockApi({ "/api/suites": { body: suites }, "/api/runs": { body: { total: 0, items: [] } }, "/api/scenarios": { body: [] }, "/api/suites/*": { body: suites[0] } });
    const { router } = renderAt("/suites");
    const rows = await screen.findAllByRole("row");
    const row = rows[1] as HTMLElement;
    expect(within(row).getByText("examples/retail_agent/replay.toml")).toBeInTheDocument();
    expect(within(row).getByText("5f7113cc59")).toBeInTheDocument();
    expect(within(row).getByText("fail")).toBeInTheDocument();
    expect(within(row).getByText("48")).toBeInTheDocument();
    expect(within(row).getByText("2.00 s")).toBeInTheDocument();
    await userEvent.click(within(row).getByText("suite0001112"));
    expect(router.state.location.search).toBe("?suite_id=suite000111222333");
  });

  it("shows an empty state", async () => {
    mockApi({ "/api/suites": { body: [] } });
    renderAt("/suites");
    expect(await screen.findByText("No suite runs stored yet.")).toBeInTheDocument();
  });

  it("shows an error", async () => {
    mockApi({ "/api/suites": { status: 503, body: { detail: "store unavailable: set REPLAY_DB_URL" } } });
    renderAt("/suites");
    expect(await screen.findByRole("alert")).toHaveTextContent("set REPLAY_DB_URL");
  });
});
