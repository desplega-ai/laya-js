# Google Cloud Run

**Status: untested.** **Verdict: works.** The binding constraint is cold start: the 4 minute startup limit covers the image pull and model load on every scale-from-zero. Limits checked 2026-09-29.

| Limit | Value | Source |
| --- | --- | --- |
| Image size | "no direct limit" | [Cloud Run quotas](https://docs.cloud.google.com/run/quotas) |
| Memory, CPU | up to 32 GiB, 8 vCPU per instance; startup timeout 4 minutes | same |
| Memory needs CPU | more than 4 GiB needs at least 2 vCPU; more than 8 GiB needs 4 | [memory limits](https://docs.cloud.google.com/run/docs/configuring/services/memory-limits) |
| In-memory filesystem | files written to disk count against instance memory | same |
| Startup probe | `initialDelaySeconds` and `periodSeconds` 0 to 600 s | [health checks](https://docs.cloud.google.com/run/docs/configuring/healthchecks) |

Because writes land in RAM, keep the model baked in the image. Fetching extra checkpoints into `/cache` at startup costs about 1.7 GB of memory each on top of the loaded model; mount a Cloud Storage volume at `/cache` instead if you serve more than `multilingual`.

## Deploy

```sh
gcloud artifacts repositories create laya --repository-format=docker --location=<region>
docker tag laya-server:local <region>-docker.pkg.dev/<project>/laya/laya-server:<tag>
docker push <region>-docker.pkg.dev/<project>/laya/laya-server:<tag>
printf %s "$(openssl rand -hex 16)" | gcloud secrets create laya-api-key --data-file=-
```

```yaml
# service.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: laya-server
spec:
  template:
    metadata:
      annotations:
        run.googleapis.com/execution-environment: gen2
        run.googleapis.com/startup-cpu-boost: "true"
        autoscaling.knative.dev/minScale: "1"      # 0 lets it scale to zero; every cold start reloads the model
    spec:
      containerConcurrency: 16                     # matches LAYA_MAX_CONCURRENT
      timeoutSeconds: 60
      containers:
        - image: <region>-docker.pkg.dev/<project>/laya/laya-server:<tag>
          ports:
            - containerPort: 8000
          env:
            - name: LAYA_THREADS
              value: "2"
            - name: LAYA_API_KEY
              valueFrom:
                secretKeyRef: { name: laya-api-key, key: latest }
          resources:
            limits: { cpu: "2", memory: 4Gi }
          startupProbe:
            httpGet: { path: /health, port: 8000 }
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 24                   # 240 s, the platform startup limit
```

```sh
gcloud run services replace service.yaml --region <region>
gcloud run services add-iam-policy-binding laya-server --region <region> \
  --member=allUsers --role=roles/run.invoker   # only if LAYA_API_KEY is your auth
```

The runtime service account needs `roles/secretmanager.secretAccessor` on `laya-api-key`. Cloud Run sets `PORT` to the `containerPort`; the server ignores `PORT` and listens on `LAYA_PORT` (8000), so keep the two equal.

For all three checkpoints: `cpu: "4"`, `memory: 16Gi`, `LAYA_MODELS=multilingual,english,typed-decisions`, an `HF_TOKEN` secret, and a Cloud Storage volume at `/cache` so the 3.4 GB download does not sit in RAM.
