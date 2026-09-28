import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

const src = (p: string) => fileURLToPath(new URL(`./packages/laya/src/${p}`, import.meta.url));

export default defineConfig({
  resolve: {
    // Workspace packages import the lib by name; test against its source so `vitest run`
    // does not depend on a prior `tsc -b` having emitted packages/laya/dist.
    alias: [
      { find: /^@desplega\/laya\/raw$/, replacement: src("raw.ts") },
      { find: /^@desplega\/laya$/, replacement: src("index.ts") },
    ],
  },
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
