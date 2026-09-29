# laya-js

`@desplega/laya`: a typesafe TypeScript runtime for [laya](https://github.com/NandhaKishorM/laya), plus `@desplega/laya-server`, a Hono HTTP server with Docker and k8s manifests.

Vendored from upstream `laya-ts/` at a pinned commit under the Apache License 2.0. See `NOTICE` and `UPSTREAM.md` once the vendor commit lands.

Using it: [docs/usage.md](docs/usage.md) covers install, loading a checkpoint, the main calls, typed results and library vs server, with real outputs and latencies. Runnable scenarios are in [examples/](examples): support triage (`predictBatch`), structured decisions with zod (`decideBatch`), a mixed-language router, a long contract (`predictLong`) and a laya-server HTTP client.

Deploying the server: [docs/deploy](docs/deploy/README.md) covers Docker, Kubernetes and 17 hosting targets, with a feasibility matrix.

Status: under construction. Packages are private and not published to npm.
