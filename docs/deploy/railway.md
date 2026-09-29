# Railway

**Status: untested.** **Verdict: works with caveats.** Deploying a private image needs the Pro plan, and Railway's health check runs only at deploy time, not continuously. Limits checked 2026-09-29.

| Limit | Value | Source |
| --- | --- | --- |
| Memory, vCPU per service | Hobby 48 GB / 48 vCPU; Pro 1 TB / 1,000 vCPU | [plans](https://docs.railway.com/reference/pricing/plans) |
| Image size | Trial and Free 4 GB, Hobby 100 GB, Pro unlimited | same |
| Private registry images | Pro plan | [services](https://docs.railway.com/guides/services) |
| Health check | deploy-time only; 300 s default timeout, raised with `RAILWAY_HEALTHCHECK_TIMEOUT_SEC` | [healthchecks](https://docs.railway.com/reference/healthchecks) |

Railway services scale memory up to the plan limit, so there is no instance size to pick. Budget about 2.2 GiB per replica for one checkpoint.

## Deploy

Push the image to a registry Railway can pull (GHCR, Docker Hub, ECR, and so on), then:

```sh
railway init --name laya-server
railway add --service laya-server --image ghcr.io/<org>/laya-server:<tag> \
  --variables "LAYA_THREADS=2" --variables "PORT=8000"
railway variables --service laya-server --set "LAYA_API_KEY=$(openssl rand -hex 16)"
railway domain --service laya-server --port 8000
```

In the service settings (Deploy), set:

- Healthcheck path: `/health`
- Healthcheck timeout: 300 s (or the variable `RAILWAY_HEALTHCHECK_TIMEOUT_SEC=300`)
- Registry credentials, if the image is private (Pro)

Railway routes public traffic to `PORT`; the server listens on `LAYA_PORT` (8000 in the image), so keep `PORT=8000`.

To keep fetched checkpoints across deploys (`LAYA_MODELS` with more than `multilingual`), attach a volume at `/cache` and add `HF_TOKEN`. A service with a volume cannot run overlapping deploys, so expect a short gap on redeploy.
