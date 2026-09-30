// Modified by Desplega Labs, 2026: expects this workspace's VERSION.
import { describe, expect, it } from "vitest";
import { VERSION } from "../src/index.js";
describe("scaffold", () => {
  it("exposes version", () => { expect(VERSION).toBe("0.1.1"); });
});
