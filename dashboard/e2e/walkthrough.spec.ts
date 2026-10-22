import { expect, test } from "@playwright/test";
import { mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";

const OUT = fileURLToPath(new URL("../../docs/walkthrough/", import.meta.url));

interface RunItem {
  id: string;
  scenario: string;
  mode: string;
  passed: boolean;
  cassette_path: string;
}

/** A human-ish pause, so each page is readable in the recording. */
const beat = (ms: number) => new Promise<void>((resolve) => { setTimeout(resolve, ms); });

test("walk through the dashboard", async ({ page }) => {
  test.setTimeout(120_000);
  mkdirSync(OUT, { recursive: true });

  const body = (await (await page.request.get("/api/runs?limit=500")).json()) as { items: RunItem[] };
  const failing = body.items.find((r) => !r.passed && r.mode === "full") ?? body.items.find((r) => !r.passed) ?? body.items[0];
  if (!failing) throw new Error("the store has no runs; run `make demo-data` first");
  const passing =
    body.items.find((r) => r.passed && r.cassette_path === failing.cassette_path && r.mode === failing.mode) ??
    body.items.find((r) => r.passed && r.scenario === failing.scenario && r.mode === failing.mode) ??
    body.items.find((r) => r.passed && r.scenario === failing.scenario);

  // 1. The runs list.
  await page.goto("/");
  await expect(page.locator("tbody tr").first()).toBeVisible();
  await page.waitForLoadState("networkidle");
  await beat(2000);
  const rows = page.locator("tbody tr");
  const count = Math.min(await rows.count(), 5);
  for (let i = 0; i < count; i++) {
    await rows.nth(i).hover();
    await beat(350);
  }
  await page.mouse.wheel(0, 400);
  await beat(1200);
  await page.mouse.wheel(0, -400);
  await beat(800);

  // 2. A failing run: click through from the list.
  const link = page.locator(`a[href="/runs/${failing.id}"]`).first();
  if ((await link.count()) > 0) {
    await link.scrollIntoViewIfNeeded();
    await link.hover();
    await beat(600);
    await link.click();
  } else {
    await page.goto(`/runs/${failing.id}`);
  }
  await expect(page.getByTestId("trace-step").first()).toBeVisible();
  await beat(2000);
  const approximate = page.getByTestId("trace-step").filter({ hasText: "approximate" }).first();
  const target = (await approximate.count()) > 0 ? approximate : page.getByTestId("trace-step").first();
  await target.scrollIntoViewIfNeeded();
  await beat(800);
  await target.getByRole("button").first().click();
  await beat(4500);

  // 3. The aligned diff against a passing run of the same scenario.
  await page.evaluate(() => { window.scrollTo(0, 0); });
  await beat(600);
  const compare = page.getByRole("form", { name: "Compare with" });
  if (passing && (await compare.count()) > 0) {
    await compare.getByRole("combobox").selectOption(passing.id);
    await beat(800);
    await compare.getByRole("button", { name: "Diff" }).click();
  } else {
    await page.getByRole("link", { name: "Diff", exact: true }).click();
  }
  await expect(page.getByTestId("diff-row").first()).toBeVisible();
  await page.waitForLoadState("networkidle");
  await beat(2000);
  for (let i = 0; i < 4; i++) {
    await page.mouse.wheel(0, 300);
    await beat(1300);
  }
  await page.evaluate(() => { window.scrollTo(0, 0); });
  await beat(800);

  // 4. Flakiness, charting a scenario that was not always passing.
  const trend = (await (await page.request.get("/api/flake/trend")).json()) as { name: string; classification: string }[];
  const unstable = trend.find((p) => p.classification !== "passing");
  await page.getByRole("link", { name: "Flakiness", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Flakiness" })).toBeVisible();
  await page.waitForLoadState("networkidle");
  await beat(2000);
  const row = page.locator("tbody tr").filter({ hasText: /flaky|failing/ }).first();
  const name = (await row.count()) > 0 ? (await row.locator("td").first().innerText()).trim() : unstable?.name;
  if (name) {
    await row.hover().catch(() => undefined);
    await beat(800);
    await page.getByRole("combobox").last().selectOption(name);
    await page.waitForLoadState("networkidle");
  }
  await page.mouse.wheel(0, 500);
  await beat(4500);
  await page.mouse.wheel(0, -500);
  await beat(1500);

  const video = page.video();
  await page.close();
  await video?.saveAs(`${OUT}walkthrough.webm`);
});
