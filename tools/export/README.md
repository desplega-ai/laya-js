# tools/export

Split-graph fp32 ONNX export of the three laya checkpoints into the private artifact store `desplega/laya-onnx`.
INT8 was deferred on 2026-09-28: the split INT8 encoder missed the approved parity bar. The quantize step stays on the `phase-2-export-wip` branch for a later spike.

| Step | Script |
|---|---|
| Export `encoder.onnx` + `head.onnx` from the pinned HF revision and verify torch vs ONNX to 1e-4 | `export_split.py` |
| Inline external data, `onnx.checker`, write `manifest.json` | `finalize_bundle.py` |
| Upload to `<checkpoint>/fp32/` in the store, check remote SHA-256, write `packages/laya/src/artifacts.ts` | `upload_bundle.py` |

The `export` workflow (`workflow_dispatch`) runs all three per checkpoint. With `upload`, it pins the new store revision in a PR.

Locally:

```sh
uv sync --frozen
uv run python export_split.py --repo convaiinnovations/laya-multilingual \
  --revision e4e9ddf21a7b1903b7acffd8814ad4307bf63a67 --verify-len 1024 --out-dir ../../bundles/multilingual/fp32
uv run python finalize_bundle.py --bundle-dir ../../bundles/multilingual/fp32 --checkpoint multilingual \
  --repo convaiinnovations/laya-multilingual --revision e4e9ddf21a7b1903b7acffd8814ad4307bf63a67
HF_TOKEN=... uv run python upload_bundle.py --bundles ../../bundles --checkpoints multilingual
```

The token is read from `HF_TOKEN` (or `HF_ACCESS_TOKEN`) only, never from a flag.
