# Azure Container Apps

**Status: untested.** **Verdict: works with caveats.** One checkpoint fits the Consumption profile at 2 vCPU / 4 GiB. All three need 8 GiB: the Consumption ceiling in a workload-profiles environment, so a Dedicated profile is the safer fit. Images must be `linux/amd64`. Limits checked 2026-09-29.

| Limit | Value | Source |
| --- | --- | --- |
| Consumption profile | 0.25 to 4 vCPU, 0.5 to 8 GiB per replica; consumption-only environments max 2 vCPU / 4 GiB | [workload profiles](https://learn.microsoft.com/en-us/azure/container-apps/workload-profiles-overview) |
| Image | `linux/amd64` only; up to 8 GB per replica on Consumption | [containers](https://learn.microsoft.com/en-us/azure/container-apps/containers) |
| Request timeout | 240 s | [ingress](https://learn.microsoft.com/en-us/azure/container-apps/ingress-overview) |
| Probes | startup, liveness, readiness; maximum values not published | [health probes](https://learn.microsoft.com/en-us/azure/container-apps/health-probes) |

## Deploy

```sh
az acr create -g <rg> -n <acr> --sku Basic
az acr login -n <acr>
docker tag laya-server:local <acr>.azurecr.io/laya-server:<tag>
docker push <acr>.azurecr.io/laya-server:<tag>

az containerapp env create -g <rg> -n laya-env -l <location>
az containerapp create -g <rg> -n laya-server --environment laya-env \
  --image <acr>.azurecr.io/laya-server:<tag> --registry-server <acr>.azurecr.io \
  --target-port 8000 --ingress external \
  --cpu 2 --memory 4Gi --min-replicas 1 --max-replicas 4 \
  --secrets api-key="$(openssl rand -hex 16)" \
  --env-vars LAYA_THREADS=2 LAYA_API_KEY=secretref:api-key
```

Add the `/health` startup and readiness probes with a YAML update (`az containerapp show -o yaml` first, then merge this into `properties.template.containers[0]`):

```yaml
probes:
  - type: Startup
    httpGet: { path: /health, port: 8000 }
    periodSeconds: 10
    timeoutSeconds: 5
    failureThreshold: 30
  - type: Readiness
    httpGet: { path: /health, port: 8000 }
    periodSeconds: 10
    failureThreshold: 3
```

```sh
az containerapp update -g <rg> -n laya-server --yaml app.yaml
```

For all three checkpoints: create the environment with a Dedicated profile (for example `D4`, 4 vCPU / 16 GiB), then `--workload-profile-name`, `--cpu 4 --memory 8Gi`, `LAYA_MODELS=multilingual,english,typed-decisions` and an `HF_TOKEN` secret. Each replica above 1 vCPU gets 8 GiB of ephemeral storage ([storage mounts](https://learn.microsoft.com/en-us/azure/container-apps/storage-mounts)); mount an Azure Files share at `/cache` to keep fetched checkpoints across restarts.
