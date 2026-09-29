# laya-js

A typesafe TypeScript runtime for [laya](https://github.com/NandhaKishorM/laya) decision models, in two packages:

- `@desplega/laya`: load a laya checkpoint as ONNX and ask it typed questions (`choice`, `noul`, `score`) from Node or Bun. Answers are probabilities, not generated text.
- `@desplega/laya-server`: a Hono HTTP server with the same `POST /v1/systemone` API as upstream's `laya-serve`, plus a Dockerfile and Kubernetes manifests.

## Status

Under construction. Neither package is published to npm yet (`@desplega/laya` and `@desplega/laya-server` both return 404 on the registry), so you install from a clone. No container image is published either: you build it from the Dockerfile.

## Quick start

Node 22+ or Bun 1.4+.

```sh
git clone https://github.com/desplega-ai/laya-js && cd laya-js
bun install
bun run build
```

Then get a checkpoint (next section), point `LAYA_MODEL_DIR` at it and run an example:

```sh
LAYA_MODEL_DIR=$PWD/bundles LAYA_THREADS=4 bun examples/01-support-triage.ts
```

## Checkpoints

The runtime needs fp32 ONNX bundles (`encoder.onnx`, `head.onnx`, `tokenizer.json`, `rl_agent_config.json`) in `<dir>/<checkpoint>/fp32/`. They are exported from upstream's public, Apache-2.0 Hugging Face checkpoints: [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) (`english`), [`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual) and [`convaiinnovations/laya-typed-decisions`](https://huggingface.co/convaiinnovations/laya-typed-decisions).

- **Export them yourself (no token).** [tools/export](tools/export/README.md#export-for-your-own-use) turns the public checkpoints into bundles with Python and `uv`. This is the path for anyone outside Desplega.
- **Download the prebuilt bundles.** Desplega's exports live in the private Hugging Face repo `desplega/laya-onnx`, pinned by revision and SHA-256 in `packages/laya/src/artifacts.ts`. `fetch-models` and the default Docker build download from it and need a read token with access to that repo; without one they fail with an auth error (HTTP 401).

## Docs

- [docs/usage.md](docs/usage.md): install, loading a checkpoint, the main calls, typed results, library vs server, with real outputs and latencies.
- [examples/](examples/README.md): support triage (`predictBatch`), structured decisions with zod (`decideBatch`), a mixed-language router, a long contract (`predictLong`), a laya-server HTTP client and a latency probe.
- [docs/deploy](docs/deploy/README.md): building the image, Docker, Kubernetes and 17 hosting targets, with a feasibility matrix.
- [tools/export](tools/export/README.md): exporting checkpoints to ONNX.

## License and upstream

Apache License 2.0, see [LICENSE](LICENSE). The runtime is vendored from upstream `laya-ts/` at a pinned commit; [NOTICE](NOTICE) carries the attribution and [UPSTREAM.md](UPSTREAM.md) lists the pinned commit and every file changed since.
