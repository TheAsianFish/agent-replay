import { describe, expect, it } from "vitest";
import { param } from "./params";

describe("param", () => {
  it("returns a set value", () => {
    expect(param(new URLSearchParams("mode=full"), "mode")).toBe("full");
  });
  it("treats an absent or empty value as undefined", () => {
    expect(param(new URLSearchParams(""), "mode")).toBeUndefined();
    expect(param(new URLSearchParams("mode="), "mode")).toBeUndefined();
    expect(param(new URLSearchParams("mode"), "mode")).toBeUndefined();
  });
});
