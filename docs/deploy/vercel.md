# Vercel

**Status: untested.** **Verdict: works with caveats, beta only.** Standard Vercel Functions cannot hold the model: the bundle limit is 250 MB uncompressed. Two beta features can: Container Images and 5 GB function bundles. Both cap memory at 4 GB on Pro (2 GB on Hobby, which is too small), and both scale in after 5 minutes idle, so the next request pays a full model load. Limits checked 2026-09-29.

| Limit | Value | Source |
| --- | --- | --- |
| Function memory | Hobby 2 GB; Pro and Enterprise 4 GB | [functions limitations](https://vercel.com/docs/functions/limitations) |
| Function bundle | 250 MB uncompressed | same |
| Large function bundles | up to 5 GB (beta) | [changelog, 2026-06-29](https://vercel.com/changelog/vercel-functions-can-now-be-up-to-5-gb-in-package-size) |
| Container Images | beta on all plans; OCI image from `Dockerfile.vercel`; default port 80, override with `PORT`; scale-in after 5 min idle in production | [container images](https://vercel.com/docs/functions/container-images) |
| Duration | Pro 300 s default, 800 s max | [functions limitations](https://vercel.com/docs/functions/limitations) |

## Container Images (beta)

Vercel builds `Dockerfile.vercel` during its build step and pushes it to the Vercel Container Registry. It cannot see a BuildKit secret, so base it on an image you already built and pushed to a registry the Vercel build can pull:

```dockerfile
# Dockerfile.vercel
FROM ghcr.io/<org>/laya-server:<tag>
ENV PORT=8000
```

Project settings:

- Functions: Function CPU `Performance` (2 vCPU, 4 GB).
- Environment variables: `LAYA_API_KEY` (sensitive), `LAYA_THREADS=2`, `PORT=8000`.

```sh
vercel link
vercel env add LAYA_API_KEY production
vercel deploy --prod
```

Unverified, and the reason this page is a sketch: whether the Vercel build can pull a private base image (make it public, or check the Vercel Container Registry docs for pushing a prebuilt image), whether the 1.6 GB image deploys within the build limits, and whether requests that arrive during the model load get the server's 503 or wait. Vercel exposes no health check path.

Only `multilingual` fits: three checkpoints need about 6 GiB.
