// Regenerate src/api/schema.d.ts into a temp file and fail if the committed copy differs.
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const dir = mkdtempSync(join(tmpdir(), "replay-schema-"));
const out = join(dir, "schema.d.ts");
try {
  execFileSync(
    process.execPath,
    ["node_modules/openapi-typescript/bin/cli.js", "openapi.json", "-o", out],
    { stdio: ["ignore", "ignore", "inherit"] },
  );
  const fresh = readFileSync(out, "utf8");
  const committed = readFileSync("src/api/schema.d.ts", "utf8");
  if (fresh !== committed) {
    console.error(
      "src/api/schema.d.ts is out of date with openapi.json. Run `npm run gen:types` and commit the result.",
    );
    process.exit(1);
  }
  console.log("src/api/schema.d.ts matches openapi.json");
} finally {
  rmSync(dir, { recursive: true, force: true });
}
