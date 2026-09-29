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

## Export for your own use

The source checkpoints are public and Apache-2.0, so the export and finalize steps need no token; only `upload_bundle.py` touches the private store. Needs Python and [uv](https://docs.astral.sh/uv/); torch runs on the CPU. Run from `tools/export`, once per checkpoint you want:

| Checkpoint | `--repo` | `--revision` | `--verify-len` |
|---|---|---|---|
| `english` | `convaiinnovations/laya` | `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` | `512` |
| `multilingual` | `convaiinnovations/laya-multilingual` | `e4e9ddf21a7b1903b7acffd8814ad4307bf63a67` | `1024` |
| `typed-decisions` | `convaiinnovations/laya-typed-decisions` | `1a793eb568e6718f15941d08f85432581df534e3` | `1024` |

```sh
uv sync --frozen
CKPT=multilingual REPO=convaiinnovations/laya-multilingual REV=e4e9ddf21a7b1903b7acffd8814ad4307bf63a67 LEN=1024
uv run python export_split.py --repo $REPO --revision $REV --verify-len $LEN --out-dir ../../bundles/$CKPT/fp32
uv run python finalize_bundle.py --bundle-dir ../../bundles/$CKPT/fp32 --checkpoint $CKPT --repo $REPO --revision $REV
```

Then set `LAYA_MODEL_DIR=<repo>/bundles` for the library, the examples and laya-server, or bake the directory into the image ([docker.md](../../docs/deploy/docker.md#without-access-to-the-artifact-store)). A local `modelDir` is loaded without checking the SHA-256 pins in `packages/laya/src/artifacts.ts`, which describe Desplega's uploads; leave `verify` off for your own export.
