import { expect, test, type Page } from "@playwright/test";

function collectConsoleErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  page.on("pageerror", (error) => {
    errors.push(error.message);
  });
  return errors;
}

test("runs, run detail, diff, flakiness and calibration render without console errors", async ({ page }) => {
  const errors = collectConsoleErrors(page);

  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Runs" })).toBeVisible();
  await expect(page.getByTestId("health")).toContainText("store");
  const firstRun = page.locator("tbody tr").first();
  await expect(firstRun).toBeVisible();

  await firstRun.getByRole("link").first().click();
  await expect(page).toHaveURL(/\/runs\/[0-9a-f]+/);
  await expect(page.getByRole("heading", { name: /^Run / })).toBeVisible();
  const steps = page.getByTestId("trace-step");
  await expect(steps.first()).toBeVisible();
  expect(await steps.count()).toBeGreaterThan(0);

  // Expand the first step to see its request and response.
  await steps.first().getByRole("button").first().click();
  await expect(page.getByRole("heading", { name: "Request" })).toBeVisible();

  const compare = page.getByRole("form", { name: "Compare with" });
  await expect(compare.getByRole("button", { name: "Diff" })).toBeEnabled();
  await compare.getByRole("button", { name: "Diff" }).click();
  await expect(page).toHaveURL(/\/diff\?base=.+&head=.+/);
  await expect(page.getByTestId("diff-row").first()).toBeVisible();

  await page.getByRole("link", { name: "Flakiness" }).click();
  await expect(page.getByRole("heading", { name: "Flakiness" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Pass rate over reports" })).toBeVisible();

  await page.getByRole("link", { name: "Calibration" }).click();
  await expect(page.getByRole("heading", { name: "Judge calibration" })).toBeVisible();

  await page.getByRole("link", { name: "Suites" }).click();
  await expect(page.getByRole("heading", { name: "Suites" })).toBeVisible();

  expect(errors).toEqual([]);
});

test("an unknown run shows the API's message", async ({ page }) => {
  await page.goto("/runs/zzzzzzzz");
  await expect(page.getByRole("alert")).toContainText("replay runs");
});
