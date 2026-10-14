import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import type { DiffRow, RunDiff } from "../api/types";
import { diff } from "../test/fixtures";
import { mockApi, renderAt } from "../test/render";

describe("DiffPage", () => {
  it("asks for two runs when none are given", async () => {
    mockApi({});
    renderAt("/diff");
    expect(await screen.findByText("Pick two runs to compare.")).toBeInTheDocument();
  });

  it("renders aligned rows with op labels, counts and field changes", async () => {
    const { calls } = mockApi({ "/api/diff": { body: diff } });
    renderAt("/diff?base=aaaaaa&head=bbbbbb");
    const rows = await screen.findAllByTestId("diff-row");
    expect(rows.map((r) => r.getAttribute("data-op"))).toEqual(["same", "changed", "removed", "added"]);
    expect(within(rows[0] as HTMLElement).getByText("same")).toBeInTheDocument();
    expect(rows[1]).toHaveClass("bg-amber-50");
    expect(rows[2]).toHaveClass("bg-red-50");
    expect(rows[3]).toHaveClass("bg-green-50");
    expect(within(rows[2] as HTMLElement).getByText("(none)")).toBeInTheDocument();
    const changes = within(rows[1] as HTMLElement).getByRole("list", { name: "Field changes" });
    expect(changes).toHaveTextContent('request.email: "a@example.com" → "b@example.com"');
    expect(screen.getByTestId("count-changed")).toHaveTextContent("1");
    expect(screen.getByText("2.50")).toBeInTheDocument();
    expect(screen.getByText("$0.0100 → $0.0200")).toBeInTheDocument();
    expect(screen.getByText("1.20 s → 900 ms")).toBeInTheDocument();
    const call = calls.find((u) => u.pathname === "/api/diff");
    expect(call?.searchParams.get("base")).toBe("aaaaaa");
    expect(call?.searchParams.get("head")).toBe("bbbbbb");
  });

  it("keeps the difference in view when two values share a long prefix", async () => {
    const changed = diff.diff.rows[1] as DiffRow;
    const prefix = "Please change the shipping address on order #W2378156 to 445 Maple Drive, Suite ";
    const body: RunDiff = {
      ...diff,
      diff: {
        ...diff.diff,
        rows: [{ ...changed, changes: [{ path: "request.note", base: `${prefix}100`, head: `${prefix}200`, base_present: true, head_present: true }] }],
      },
    };
    mockApi({ "/api/diff": { body } });
    renderAt("/diff?base=aaaaaa&head=bbbbbb");
    const item = await screen.findByTestId("field-change");
    const before = item.querySelector(".text-red-800")?.textContent ?? "";
    const after = item.querySelector(".text-green-800")?.textContent ?? "";
    expect(before).not.toBe(after);
    expect(before).toBe('…rive, Suite 100"');
    expect(after).toBe('…rive, Suite 200"');
    expect(item).toHaveTextContent('request.note: …rive, Suite 100" → …rive, Suite 200"');
  });

  it("edits and swaps the run ids", async () => {
    mockApi({ "/api/diff": { body: diff } });
    const { router } = renderAt("/diff?base=aaaaaa&head=bbbbbb");
    await screen.findAllByTestId("diff-row");
    await userEvent.click(screen.getByRole("button", { name: "Swap base and head" }));
    expect(router.state.location.search).toBe("?base=bbbbbb&head=aaaaaa");
    const base = screen.getByRole("textbox", { name: "Base" });
    await userEvent.clear(base);
    await userEvent.type(base, "cccccc");
    await userEvent.click(screen.getByRole("button", { name: "Compare" }));
    expect(router.state.location.search).toBe("?base=cccccc&head=aaaaaa");
  });

  it("shows the API's detail when a run is missing", async () => {
    mockApi({ "/api/diff": { status: 404, body: { detail: "no run with id 'nope' in the store. List ids with `replay runs`." } } });
    renderAt("/diff?base=nope00&head=bbbbbb");
    expect(await screen.findByRole("alert")).toHaveTextContent("no run with id 'nope'");
  });
});
