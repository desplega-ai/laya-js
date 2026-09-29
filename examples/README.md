# Examples

Runnable scenarios for `@desplega.ai/laya` and `@desplega.ai/laya-server`. Setup and real outputs are in [docs/usage.md](../docs/usage.md).

```sh
bun install && bun run build
node packages/laya-server/dist/fetch-models.js --dest ./bundles english multilingual   # needs HF_TOKEN; or export them, see tools/export/README.md
LAYA_MODEL_DIR=$PWD/bundles LAYA_THREADS=4 bun examples/01-support-triage.ts
```

| File | Shows |
| --- | --- |
| `01-support-triage.ts` | `predictBatch`, typed choice / noul / score answers, accuracy against labels |
| `02-structured-decisions.ts` | `decideBatch` with a zod schema, `SchemaError` |
| `03-multilingual-router.ts` | `createRouter`, `routeBatch`, per-language routing |
| `04-long-contract.ts` | `predict` vs `predictLong` on a document past the window |
| `05-server-client.ts` | laya-server over HTTP with plain `fetch` |
| `06-latency.ts` | per-call latency for capacity planning |
