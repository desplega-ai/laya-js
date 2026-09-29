# Deploying laya-server

`@desplega.ai/laya-server` is a Hono server on Node 22 that serves `POST /v1/systemone` from a CPU `onnxruntime-node` session over a 1.3 GB fp32 ONNX bundle. It deploys anywhere that runs a 1.6 GB Linux container with 3 GB of RAM and waits a few seconds for `/health`. It does not fit serverless function bundles or edge isolates.

Every page below deploys the image built in [docker.md](docker.md#build). Provider limits were checked against the linked provider docs on **2026-09-29**. "Tested" means we ran it; everything else is a config written from the docs and has never been deployed.

## Feasibility matrix

| Provider | Target | Verdict | Binding constraint | Tested |
| --- | --- | --- | --- | --- |
| [Docker, Compose](docker.md) | container | works | none | tested (CI `docker run` smoke); Compose untested |
| [Kubernetes](kubernetes.md) | container | works | memory request: 2560Mi for one checkpoint | untested (manifests validated with kubeconform) |
| [Hetzner Cloud VM](hetzner.md) | VM | works | VM RAM: pick 8 GB or more | untested |
| [AWS ECS Fargate](aws.md#ecs-fargate) | container | works | none; 1 vCPU / 4 GB task or larger | untested |
| [AWS Lambda](aws.md#lambda-container-image) | serverless function | works with caveats | cold start: 10 s init limit, 1.6 GB image; 6 MB payload | untested |
| [AWS App Runner](aws.md#app-runner) | container | not viable | closed to new customers | n/a |
| [Google Cloud Run](cloud-run.md) | container | works | cold start: 4 min startup limit, 1.6 GB pull on scale-from-zero | untested |
| [Azure Container Apps](azure-container-apps.md) | container | works with caveats | memory: Consumption caps at 8 GiB (4 GiB in consumption-only envs); linux/amd64 only | untested |
| [Fly.io](fly.md) | container (Machine) | works with caveats | model read: rootfs capped at 8 MiB/s | untested |
| [Railway](railway.md) | container | works with caveats | private images need the Pro plan; health checked only at deploy | untested |
| [Render](render.md) | container | works | linux/amd64 only; 2c-4g or larger | untested |
| [DigitalOcean App Platform](digitalocean.md) | container | works with caveats | image size: 1.6 GB is over the 1 GiB recommendation | untested |
| [Koyeb](koyeb.md) | container | works | instance `large` (4 GB) or larger | untested |
| [Heroku](heroku.md) | container | works with caveats | memory: Performance-M (2.5 GB) is too tight, Performance-L (14 GB) is the smallest fit | untested |
| [Cloudflare Containers](cloudflare.md#containers) | container | works with caveats | Worker and Durable Object glue code; `standard-1` (4 GiB) or larger | untested |
| [Cloudflare Workers](cloudflare.md#workers) | edge | not viable | 128 MB memory, 64 MiB script, no native addons | n/a |
| [Vercel](vercel.md) | serverless function | works with caveats | beta only (container images or 5 GB functions); 4 GB memory cap; scale-in after 5 min idle | untested |
| [Netlify Functions](#netlify-functions) | serverless function | not viable | 250 MB bundle, 60 s sync limit, no containers | n/a |
| [Deno Deploy](#deno-deploy) | edge | not viable | 1 GB deployment cap, 512 MB memory | n/a |

Counts: 7 work, 8 work with caveats, 4 are not viable. Tested: Docker only (and the bare Node process measured below).

Not covered: Northflank, Porter, Coolify, Dokku and other container platforms. They run the same image; give it 1 vCPU, 4 GB and a `/health` readiness check.

## Measured footprint

| What | Value | Source |
| --- | --- | --- |
| Image, `linux/amd64`, uncompressed | 1.601 GB | CI `image.yml` (`docker image inspect`), PR #8 |
| `multilingual` fp32 bundle | 1,324,747,558 bytes (encoder 1.23 GB, head 60 MB, tokenizer 34 MB) | `packages/laya/src/artifacts.ts` |
| `english` or `typed-decisions` fp32 bundle | 1,692,191,043 bytes each | same |
| `onnxruntime-node` 1.30.0 native binaries | 45 MB (`linux/x64`), 25 MB (`linux/arm64`) | `node_modules`, measured 2026-09-29 |
| Peak RSS while loading `multilingual` | 2,280 MiB | local run, below |
| RSS idle after load | 1,856 to 1,878 MiB | local run |
| RSS after 300 requests (100 sequential, 8x25 concurrent) | 1,761 to 1,767 MiB | local run |
| Process start to `/health` 200 | 6.7 to 7.4 s, model file already in the page cache | local run |
| Sequential `POST /v1/systemone` latency, 2 questions | p50 324 to 373 ms, p95 497 to 507 ms | local run, noisy shared host |

The local run was `node packages/laya-server/dist/main.js` with the `multilingual` bundle on local disk, `LAYA_THREADS` 1 and 4, on a shared 16-core Xeon W-2145 host (2026-09-29). It measures memory reliably. Latency on a shared host does not transfer: for throughput and cost per request, see the Phase 10 perf report when it lands.

What that means for sizing:

- **Memory:** 3 GiB for one checkpoint (the k8s base limit), 8 GiB for all three. Plans with 2 GB are too small: the load peaks at 2.2 GiB.
- **CPU:** set `LAYA_THREADS` to the vCPUs you pay for. onnxruntime oversubscription is a large regression.
- **Cold start:** image pull (1.6 GB) plus reading 1.3 GB of weights plus session creation. Seven seconds is the floor with a warm page cache; a scale-from-zero start on a platform is slower. Keep at least one instance running for anything user-facing.
- **Architecture:** `linux/amd64` by default. `linux/arm64` works (build with `--platform linux/arm64`). glibc only: `onnxruntime-node` has no musl build, so no Alpine base.

## Configuration

The server reads `LAYA_*` variables, not `PORT`. On platforms that inject `PORT`, either route to 8000 or set `LAYA_PORT` to the same value.

| Variable | Default in image | Notes |
| --- | --- | --- |
| `LAYA_API_KEY` | unset | Bearer token for `POST /v1/systemone`. Unset means no auth. Store it as a secret. |
| `LAYA_PORT` | `8000` | |
| `LAYA_THREADS` | onnxruntime default | Match your vCPUs. |
| `LAYA_MODELS` | `multilingual` | Comma list of checkpoints to serve. Anything not baked is fetched into `LAYA_CACHE_DIR` at startup. |
| `HF_TOKEN` | unset | Optional Hugging Face read token. `desplega/laya-onnx` is public, so it is only for rate limits or a private mirror. Secret. |
| `LAYA_CACHE_DIR` | `/cache` | Must be writable. Mount a volume here to keep fetched checkpoints across restarts. |
| `LAYA_MAX_LOADED` | unset | Cap on resident checkpoints. |
| `LAYA_MAX_CONCURRENT` | `16` | Requests beyond this get 503 with `Retry-After`. |

The image runs as uid 1000 with a read-only-friendly layout: only `/cache` and `/tmp` need to be writable. `GET /health` returns 503 until the model is loaded and 200 after; use it as the startup and readiness check. On SIGTERM the server drains in-flight requests for up to 25 s.

## Not viable

### Netlify Functions

Functions have a 250 MB unzipped bundle ([Netlify docs](https://docs.netlify.com/build/frameworks/framework-setup-guides/nextjs/legacy-runtime/troubleshooting/)), 1024 to 4096 MB of memory (4096 only on credit-based Pro and Enterprise) and a 60 s synchronous limit that is not configurable ([functions configuration](https://docs.netlify.com/build/functions/configuration/)). Netlify runs no containers. The 1.3 GB model does not fit the bundle, and fetching it on each cold start would not finish inside 60 s. Checked 2026-09-29.

### Deno Deploy

The runtime can load Node native addons ([runtime](https://docs.deno.com/deploy/reference/runtime/)), but a deployment "should not exceed 1 gigabyte" and applications have "a maximum memory allocation of 512MB" ([pricing and limits](https://docs.deno.com/deploy/pricing_and_limits/)). The model alone is 1.32 GB and needs 2.2 GiB resident. The [changelog](https://docs.deno.com/deploy/changelog/) mentions memory configurable up to 4096 MB on some plans, which conflicts with the limits page; even then the bundle cap rules it out. There is no container option. Checked 2026-09-29.

### Cloudflare Workers and AWS App Runner

See [cloudflare.md](cloudflare.md#workers) and [aws.md](aws.md#app-runner).
