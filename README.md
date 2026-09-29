# laya-js

A typesafe TypeScript runtime for [laya](https://github.com/NandhaKishorM/laya) decision models, in two packages:

- `@desplega.ai/laya`: load a laya checkpoint as ONNX and ask it typed questions (`choice`, `noul`, `score`) from Node or Bun. Answers are probabilities, not generated text.
- `@desplega.ai/laya-server`: a Hono HTTP server with the same `POST /v1/systemone` API as upstream's `laya-serve`, plus a Dockerfile and Kubernetes manifests.

## Status

Under construction. Both packages are on npm at 0.1.0: `npm install @desplega.ai/laya` for the library, `@desplega.ai/laya-server` for the HTTP server. No container image is published: you build it from the Dockerfile.

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

- **Export them yourself (no token).** [tools/export](tools/export/README.md#export-for-your-own-use) turns the public checkpoints into bundles with Python and `uv`. Use it to build from the source checkpoints yourself instead of the prebuilt bundles.
- **Download the prebuilt bundles.** Desplega's exports live in the public Hugging Face repo [`desplega/laya-onnx`](https://huggingface.co/desplega/laya-onnx), pinned by revision and SHA-256 in `packages/laya/src/artifacts.ts`. `fetch-models`, `createAgent` and the default Docker build download from it with no token. `HF_TOKEN` is optional: set it to lift Hugging Face's anonymous rate limits, or to read a private mirror. A rejected token fails with an auth error (HTTP 401), even on a public repo.

## Docs

- [docs/usage.md](docs/usage.md): install, loading a checkpoint, the main calls, typed results, library vs server, with real outputs and latencies.
- [examples/](examples/README.md): support triage (`predictBatch`), structured decisions with zod (`decideBatch`), a mixed-language router, a long contract (`predictLong`), a laya-server HTTP client and a latency probe.
- [docs/deploy](docs/deploy/README.md): building the image, Docker, Kubernetes and 17 hosting targets, with a feasibility matrix.
- [tools/export](tools/export/README.md): exporting checkpoints to ONNX.

## Releasing

Both packages release together, at the same version. Bumping the version on `main` releases them: [`auto-release.yml`](.github/workflows/auto-release.yml) tags the commit and [`publish.yml`](.github/workflows/publish.yml) publishes `@desplega.ai/laya`, then `@desplega.ai/laya-server`, to npm with provenance.

1. Set the same `version` in `packages/laya/package.json` and `packages/laya-server/package.json`, and in the two matching `"version"` lines of `bun.lock`. Merge that to `main`.
2. `auto-release` runs on the push. If the two versions differ it fails and tags nothing. If the tag `v<version>` does not exist yet, it creates that tag on the pushed commit and dispatches `publish` on it. If the tag exists, it does nothing, so a push that leaves the version alone releases nothing.
3. Watch the `publish` run in Actions. It typechecks, tests, packs both tarballs, checks them and publishes.

Optional dry runs, both in Actions, Run workflow, with `dry_run` on by default: `publish` builds, packs and checks both tarballs and runs `npm publish --dry-run`; `auto-release` checks the versions and reports whether it would tag, and creates nothing. `publish` skips a package whose version is already on npm, so a rerun resumes where a failed run stopped. If the tag exists but nothing was published (the dispatch failed), run `publish` by hand from that tag with `dry_run` unticked. Pushing `git tag v<version> && git push origin v<version>` yourself still publishes: the tag must equal the version, or the job fails before publishing.

The workflow reads the Actions secret `NPM_TOKEN`, the `desplega-ai` organization secret (an npm token with read and write on the `@desplega.ai` scope). It must be shared with this repository, and because this repository is public, "Private repositories" visibility does not cover it. If the secret is empty, a real run fails at its first step with a message saying so. `bun run pack:check` runs the same tarball checks locally and in CI. Publish only through the workflow: `npm pack` and `npm publish` from a package directory leave laya-server depending on `workspace:*`, which npm cannot install; the workflow packs with `bun pm pack`, which writes the real version.

## License and upstream

Apache License 2.0, see [LICENSE](LICENSE). The runtime is vendored from upstream `laya-ts/` at a pinned commit; [NOTICE](NOTICE) carries the attribution and [UPSTREAM.md](UPSTREAM.md) lists the pinned commit and every file changed since.
