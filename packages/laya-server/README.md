# @desplega.ai/laya-server

A Hono HTTP server for [`@desplega.ai/laya`](https://www.npmjs.com/package/@desplega.ai/laya) with the same `POST /v1/systemone` contract as upstream's `laya-serve`.

```sh
npm install @desplega.ai/laya-server
LAYA_MODEL_DIR=./bundles LAYA_MODELS=multilingual node node_modules/@desplega.ai/laya-server/dist/main.js
```

Node 22+. It serves from a CPU `onnxruntime-node` session and needs about 3 GB of RAM for one fp32 checkpoint. Checkpoints come from `LAYA_MODEL_DIR` (export them with [tools/export](https://github.com/desplega-ai/laya-js/blob/main/tools/export/README.md#export-for-your-own-use)); a listed checkpoint that is missing there is downloaded from the public Hugging Face repo `desplega/laya-onnx` and verified against pinned SHA-256s. `HF_TOKEN` is optional (rate limits, or a private mirror).

Most deployments use the container image instead of this package: see the [Dockerfile and deployment guide](https://github.com/desplega-ai/laya-js/blob/main/docs/deploy/README.md). Environment variables and the request format are documented there and in [docs/usage.md](https://github.com/desplega-ai/laya-js/blob/main/docs/usage.md). Apache-2.0; see `NOTICE`.
