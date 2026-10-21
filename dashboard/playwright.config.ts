import { defineConfig, devices } from "@playwright/test";

const executablePath = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE;

/**
 * End-to-end tests against a running server that serves the built dashboard
 * (`npm run build`, then `replay serve --db ...`). Set DASHBOARD_URL to point
 * elsewhere, and PLAYWRIGHT_CHROMIUM_EXECUTABLE to use a system Chromium.
 */
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: 0,
  reporter: [["list"]],
  timeout: 30_000,
  use: {
    baseURL: process.env.DASHBOARD_URL ?? "http://127.0.0.1:8000",
    trace: "retain-on-failure",
    ...devices["Desktop Chrome"],
    launchOptions: executablePath ? { executablePath } : {},
  },
  projects: [
    { name: "smoke", testMatch: /smoke\.spec\.ts/ },
    {
      name: "screenshots",
      testMatch: /screenshots\.spec\.ts/,
      use: { viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 },
    },
    {
      name: "walkthrough",
      testMatch: /walkthrough\.spec\.ts/,
      use: {
        viewport: { width: 1280, height: 800 },
        deviceScaleFactor: 1,
        video: { mode: "on", size: { width: 1280, height: 800 } },
      },
    },
  ],
});
