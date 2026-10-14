import { expect, test, type Page } from "@playwright/test";
import { mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";

const OUT = fileURLToPath(new URL("../../docs/screenshots/", import.meta.url));
mkdirSync(OUT, { recursive: true });

async function shoot(page: Page, name: string) {
  await page.waitForLoadState("networkidle");
  await page.screenshot({ path: `${OUT}${name}.png` });
}

interface RunItem {
  id: string;
  scenario: string;
  mode: string;
  passed: boolean;
  cassette_path: string;
}

/** A failing run and a passing run of the same scenario, for the detail and diff shots. */
async function pickRuns(page: Page): Promise<{ failing: RunItem; passing: RunItem | undefined }> {
  const response = await page.request.get("/api/runs?limit=500");
  const body = (await response.json()) as { items: RunItem[] };
  const failing = body.items.find((r) => !r.passed && r.mode === "full") ?? body.items.find((r) => !r.passed) ?? body.items[0];
  if (!failing) throw new Error("the store has no runs; seed it with `replay ci replay.toml --db $REPLAY_DB_URL` first");
  const passing =
    body.items.find((r) => r.passed && r.cassette_path === failing.cassette_path && r.mode === failing.mode) ??
    body.items.find((r) => r.passed && r.scenario === failing.scenario && r.mode === failing.mode) ??
    body.items.find((r) => r.passed && r.scenario === failing.scenario);
  return { failing, passing };
}

test("capture dashboard screenshots", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("tbody tr").first()).toBeVisible();
  await shoot(page, "runs");

  const { failing, passing } = await pickRuns(page);
  await page.goto(`/runs/${failing.id}`);
  await expect(page.getByTestId("trace-step").first()).toBeVisible();
  const approximate = page.getByTestId("trace-step").filter({ hasText: "approximate" }).first();
  const target = (await approximate.count()) > 0 ? approximate : page.getByTestId("trace-step").first();
  await target.getByRole("button").first().click();
  await page.evaluate(() => { window.scrollTo(0, 0); });
  await shoot(page, "run-detail");

  if (passing) {
    await page.goto(`/diff?base=${passing.id}&head=${failing.id}`);
    await expect(page.getByTestId("diff-row").first()).toBeVisible();
  } else {
    await page.goto("/diff");
  }
  await shoot(page, "diff");

  // Chart a scenario that was not always passing, when there is one.
  const trend = (await (await page.request.get("/api/flake/trend")).json()) as { name: string; classification: string }[];
  const unstable = trend.find((p) => p.classification !== "passing");
  await page.goto(unstable ? `/flaky?scenario=${unstable.name}` : "/flaky");
  await expect(page.getByRole("heading", { name: "Flakiness" })).toBeVisible();
  await shoot(page, "flakiness");

  const calibrations = (await (await page.request.get("/api/calibration")).json()) as { id: string }[];
  await page.goto(calibrations[0] ? `/calibration/${calibrations[0].id}` : "/calibration");
  await shoot(page, "calibration");

  await page.goto("/suites");
  await expect(page.getByRole("heading", { name: "Suites" })).toBeVisible();
  await shoot(page, "suites");
});
