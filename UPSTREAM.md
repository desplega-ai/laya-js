# Upstream

| | |
|---|---|
| Source | https://github.com/NandhaKishorM/laya |
| Commit | `9d955671415fc19f069b9cc998928075c1f255ec` (tag `v0.3.21`) |
| Paths | `laya-ts/src/` → `packages/laya/src/`, `laya-ts/tests/` → `packages/laya/test/` |
| Vendored on | 2026-09-28 |
| License | Apache-2.0 (`LICENSE`, verbatim from the upstream root); attribution in `NOTICE` |

The vendor commit is byte-identical to upstream. `vendor.json` holds the SHA-256 of every vendored file at the pinned commit, and `bun run check-vendor` fails when a vendored file differs without a `// Modified by Desplega Labs, 2026: <summary>.` header or is missing from the tables below.

To bump the pin, re-vendor at the new commit, then replay the changes below.

## Modified files

| File | Change |
|---|---|
| `packages/laya/src/providers.ts` | Removed the browser path: `createWebProvider`, `loadWebBundle`, `WebBundle`, `baseUrlFor`, `fetchArrayBuffer` (onnxruntime-web). Edge targets are out of scope. |
| `packages/laya/src/agent.ts` | `Agent.load` always takes the Node path; the `window`-detected browser branch is gone. |
| `packages/laya/src/index.ts` | Dropped the `createWebProvider`, `loadWebBundle` and `WebBundle` exports. |
| `packages/laya/test/revision-pinning.test.ts` | Removed the one web-only test, "loadWebBundle reports the x-repo-commit header". |

## Removed files

| File | Reason |
|---|---|
| `packages/laya/test/package-e2e.mjs` | npm-pack test for upstream's `laya-ts` package name and its optional onnxruntime-node/web deps. This package is private and depends on onnxruntime-node directly. |
