import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { passingRun, runPage, runRecord, runSummary } from "../test/fixtures";
import { mockApi, renderAt } from "../test/render";

function api() {
  return mockApi({
    [`/api/runs/${runRecord.id}`]: { body: runRecord },
    "/api/runs": { body: runPage },
  });
}

describe("RunDetailPage", () => {
  it("shows the summary, error and checks in position order", async () => {
    api();
    renderAt(`/runs/${runRecord.id}`);
    expect(await screen.findByRole("heading", { name: "Run bbbbbbbb" })).toBeInTheDocument();
    expect(screen.getByText("UnrecordedCallError")).toBeInTheDocument();
    expect(screen.getByText("examples.retail_agent.mutants:skips_order_lookup")).toBeInTheDocument();
    const checks = screen.getByRole("heading", { name: /Checks/ }).parentElement as HTMLElement;
    const items = within(checks).getAllByRole("listitem");
    expect(items.map((li) => li.textContent)).toEqual([
      expect.stringContaining("tool_sequence"),
      expect.stringContaining("required_tools"),
      expect.stringContaining("judge:tone"),
    ]);
    expect(within(items[0] as HTMLElement).getByText("failed")).toBeInTheDocument();
    expect(within(items[2] as HTMLElement).getByText("advisory fail")).toBeInTheDocument();
  });

  it("renders one row per step with status badges and nesting", async () => {
    api();
    renderAt(`/runs/${runRecord.id}`);
    const steps = await screen.findAllByTestId("trace-step");
    expect(steps).toHaveLength(3);
    expect(within(steps[1] as HTMLElement).getByText("approximate")).toBeInTheDocument();
    expect(within(steps[2] as HTMLElement).getByText("unrecorded")).toBeInTheDocument();
    expect(within(steps[2] as HTMLElement).getByLabelText("nested level 1")).toBeInTheDocument();
    expect(within(steps[0] as HTMLElement).getByText("3,134 in / 43 out")).toBeInTheDocument();
  });

  it("expands a step to show request and response JSON, truncating long strings", async () => {
    api();
    renderAt(`/runs/${runRecord.id}`);
    const steps = await screen.findAllByTestId("trace-step");
    const toggle = within(steps[1] as HTMLElement).getByRole("button", { name: /find_user_id_by_email/ });
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    await userEvent.click(toggle);
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByRole("heading", { name: "Request" })).toBeInTheDocument();
    expect(screen.getByText('"mia_garcia_4516"')).toBeInTheDocument();
    const more = screen.getByRole("button", { name: /show more \(400 chars\)/ });
    await userEvent.click(more);
    expect(screen.getByRole("button", { name: "show less" })).toBeInTheDocument();
    await userEvent.click(toggle);
    expect(screen.queryByRole("heading", { name: "Request" })).not.toBeInTheDocument();
  });

  it("navigates to the diff page from Compare with", async () => {
    api();
    const { router } = renderAt(`/runs/${runRecord.id}`);
    const form = await screen.findByRole("form", { name: "Compare with" });
    await within(form).findByRole("option", { name: /aaaaaaaa/ });
    await userEvent.click(within(form).getByRole("button", { name: "Diff" }));
    expect(router.state.location.pathname).toBe("/diff");
    expect(router.state.location.search).toBe(`?base=${passingRun.id}&head=${runRecord.id}`);
  });

  it("offers runs of the same recording first", async () => {
    const otherRecording = runSummary({ id: "c".repeat(32), passed: true, cassette_path: "elsewhere.jsonl" });
    const sameRecording = runSummary({ id: "d".repeat(32), passed: false, cassette_path: runRecord.cassette_path });
    mockApi({
      [`/api/runs/${runRecord.id}`]: { body: runRecord },
      "/api/runs": { body: { total: 2, items: [otherRecording, sameRecording] } },
    });
    renderAt(`/runs/${runRecord.id}`);
    const form = await screen.findByRole("form", { name: "Compare with" });
    await within(form).findByRole("option", { name: /dddddddd/ });
    const options = within(form).getAllByRole("option").filter((o) => o.getAttribute("value"));
    expect(options[0]).toHaveTextContent("dddddddd");
  });

  it("shows the API's 404 detail", async () => {
    mockApi({ "/api/runs/*": { status: 404, body: { detail: "no run with id 'zzz' in the store. List ids with `replay runs`." } } });
    renderAt("/runs/zzz");
    expect(await screen.findByRole("alert")).toHaveTextContent("List ids with `replay runs`");
  });
});
