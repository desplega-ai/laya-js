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
| `packages/laya/src/providers.ts` | Removed the browser path: `createWebProvider`, `loadWebBundle`, `WebBundle`, `baseUrlFor`, `fetchArrayBuffer` (onnxruntime-web). Edge targets are out of scope. Added `LayaLoadError`: an HF 401/403 on any bundle file throws `HF auth failed for <repo>` instead of "Incompatible model" or a silent skip. Added optional `SessionProvider.release()`, which frees the ONNX sessions (used by `dispose()`). |
| `packages/laya/src/agent.ts` | `Agent.load` always takes the Node path; the `window`-detected browser branch is gone. |
| `packages/laya/src/index.ts` | Dropped the `createWebProvider`, `loadWebBundle` and `WebBundle` exports. Exports `LayaLoadError` and the pinned artifact map (`artifacts.ts`). The vendored export list moved verbatim to `raw.ts` (`@desplega/laya/raw`); `index.ts` exports only the typed API from `typed.ts`, the answer types, `VERSION`, `SchemaError` and `LayaLoadError`; everything else is reached through `@desplega/laya/raw`. |
| `packages/laya/test/revision-pinning.test.ts` | Removed the one web-only test, "loadWebBundle reports the x-repo-commit header". |

## Removed files

| File | Reason |
|---|---|
| `packages/laya/test/package-e2e.mjs` | npm-pack test for upstream's `laya-ts` package name and its optional onnxruntime-node/web deps. This package is private and depends on onnxruntime-node directly. |

## Vendored tools

| File | Source | Change |
|---|---|---|
| `tools/export/export_split.py` | `laya-ts/scripts/export_onnx.py` | Added `--revision` to pin the HF checkpoint commit. |
