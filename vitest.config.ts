import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["packages/*/test/**/*.test.ts"],
    // The bpe-complexity parity case runs 20k strings; 5s is too tight on shared CI runners.
    testTimeout: 30_000,
    // Type tests (`*.test-d.ts`) run under tsc as part of `bun run test`.
    typecheck: {
      enabled: true,
      include: ["packages/*/test/**/*.test-d.ts"],
      tsconfig: "./tsconfig.test.json",
    },
  },
});
