import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["packages/*/test/**/*.test.ts"],
    // The bpe-complexity parity case runs 20k strings; 5s is too tight on shared CI runners.
    testTimeout: 30_000,
  },
});
