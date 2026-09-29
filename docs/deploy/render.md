# Render

**Status: untested.** **Verdict: works.** Use plan `2c-4g` or larger; `1c-2g` is below the 2.2 GiB load peak. Images must be `linux/amd64`. Limits checked 2026-09-29.

| Limit | Value | Source |
| --- | --- | --- |
| Image | `linux/amd64`; compressed size up to 10 GB | [deploying an image](https://render.com/docs/deploying-an-image) |
| Plans | `1c-2g`, `2c-4g`, `2c-8g`, `2c-16g`, `4c-8g`, and up | [compute plans](https://render.com/docs/compute-plans) |
| Health check | a deploy is cancelled if not healthy within 15 min; an instance restarts after 60 s of consecutive failures | [health checks](https://render.com/docs/health-checks) |
| Port | `PORT` defaults to 10000 | [web services](https://render.com/docs/web-services) |

## Deploy

Push the image to a registry Render can pull, add registry credentials in the dashboard if it is private, then commit a Blueprint:

```yaml
# render.yaml
services:
  - type: web
    name: laya-server
    runtime: image
    image:
      url: ghcr.io/<org>/laya-server:<tag>
      # creds: { fromRegistryCreds: { name: ghcr } }   # private image
    plan: 2c-4g
    numInstances: 1
    healthCheckPath: /health
    envVars:
      - key: PORT
        value: "8000"
      - key: LAYA_THREADS
        value: "2"
      - key: LAYA_API_KEY
        generateValue: true
```

Render routes to `PORT`; the server listens on `LAYA_PORT` (8000 in the image), so `PORT=8000` keeps them aligned. `generateValue` creates a random key; read it back from the dashboard.

For all three checkpoints: `plan: 2c-16g` (or `4c-16g`), `LAYA_MODELS=multilingual,english,typed-decisions`, `HF_TOKEN` with `sync: false`, and a persistent disk at `/cache` (paid, one instance per disk) so restarts skip the 3.4 GB download.
