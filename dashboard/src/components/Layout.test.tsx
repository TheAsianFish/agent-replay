import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { mockApi, renderAt } from "../test/render";

describe("Layout", () => {
  it("has the section navigation and marks the active one", async () => {
    mockApi({ "/api/suites": { body: [] } });
    renderAt("/suites");
    const nav = screen.getByRole("navigation", { name: "Sections" });
    for (const name of ["Runs", "Diff", "Suites", "Flakiness", "Calibration"]) {
      expect(nav).toHaveTextContent(name);
    }
    expect(screen.getByRole("link", { name: "Suites" })).toHaveAttribute("aria-current", "page");
    expect(await screen.findByTestId("health")).toHaveTextContent("store sqlite");
  });

  it("reports an unreachable API in the header", async () => {
    mockApi({ "/api/health": { status: 502, body: {} }, "/api/suites": { body: [] } });
    renderAt("/suites");
    expect(await screen.findByText("API unreachable")).toBeInTheDocument();
  });

  it("shows a not-found page for unknown routes", () => {
    mockApi({});
    renderAt("/nowhere");
    expect(screen.getByRole("heading", { name: "Page not found" })).toBeInTheDocument();
  });
});
