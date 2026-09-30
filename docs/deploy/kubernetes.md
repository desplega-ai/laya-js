# Kubernetes (kustomize)

**Status: untested.** CI renders `deploy/k8s/base` and both overlays with `kustomize build` and validates them with `kubeconform -strict` (`.github/workflows/image.yml`, job `manifests`). No rollout to a real or `kind` cluster has run. This covers any conformant cluster: EKS, GKE, AKS, DOKS, Hetzner k3s and the like.

## What ships

| Path | What it deploys |
| --- | --- |
| `deploy/k8s/base` | 2 replicas of the image with `multilingual` fp32 baked in. Requests 1 CPU / 2560Mi, limit 3Gi. Startup probe allows 5 min for the model load. Service (ClusterIP :8000), HPA (70% CPU, 2 to 6), PDB (`minAvailable: 1`). Non-root, read-only root filesystem. |
| `deploy/k8s/overlays/all-checkpoints` | Base plus an initContainer that fetches `english` and `typed-decisions` (about 3.4 GB) into an `emptyDir`. Requests 6Gi, limit 8Gi. |
| `deploy/k8s/overlays/pvc` | `all-checkpoints` with a 10Gi `ReadWriteOnce` PVC as the cache, so restarts skip the download. RWO only works while replicas share a node; use a `ReadWriteMany` class to spread them. |

## Deploy

1. Build and push the image ([docker.md](docker.md#build)).
2. Create the secrets:

   ```sh
   kubectl create namespace laya
   kubectl -n laya create secret generic laya-api --from-literal=api-key="$(openssl rand -hex 16)"
   # optional, all-checkpoints and pvc only: a Hugging Face read token (desplega/laya-onnx is public; the token only lifts rate limits)
   kubectl -n laya create secret generic laya-hf --from-file=token=./hf-read-token.txt
   # private registry only
   kubectl -n laya create secret docker-registry ghcr --docker-server=ghcr.io \
     --docker-username=<user> --docker-password=<read:packages token>
   ```

3. Pin your image in an overlay of your own, so the repo's manifests stay untouched:

   ```yaml
   # my-overlay/kustomization.yaml
   apiVersion: kustomize.config.k8s.io/v1beta1
   kind: Kustomization
   namespace: laya
   resources:
     - github.com/desplega-ai/laya-js//deploy/k8s/base?ref=main   # or a local path
   images:
     - name: ghcr.io/desplega-ai/laya-server
       newName: ghcr.io/<org>/laya-server
       newTag: <tag>
   patches:
     - target: { kind: Deployment, name: laya-server }
       patch: |-
         - op: add
           path: /spec/template/spec/imagePullSecrets
           value: [{ name: ghcr }]
   ```

   Remote bases need read access to the repo; point `resources` at a local checkout otherwise.

4. Apply and wait:

   ```sh
   kubectl apply -k my-overlay
   kubectl -n laya rollout status deploy/laya-server --timeout=10m
   kubectl -n laya port-forward svc/laya-server 8000:8000
   ```

## Sizing notes

- Keep `LAYA_THREADS` equal to the CPU request. The base sets both to 1.
- One fp32 `multilingual` process peaks at about 2.2 GiB RSS while loading and settles near 1.8 GiB ([README](README.md#measured-footprint)). Under the Phase 10 load (long states, 10 questions, 16 concurrent requests) the container peaked at 2.5 GiB (cgroup `memory.peak`, 4 vCPU) and finished with no OOM kill under a hard 3 GiB limit. The 3Gi limit holds because `LAYA_RUN_MB` (default 256) bounds the working memory of each ONNX run, one run at a time per process; one row cannot be split, so a call with a `maxLen` above the default 512 that produces rows of about 2,000 tokens or more can still pass it (a lone 2,048-token row takes the encoder to 2.7 GiB by itself); raise the limit if you serve such rows.
- Nodes pull a 1.6 GB image. Pre-pull it or keep `minReplicas` at 2 or more so a scale-up is not stuck behind a cold pull.
- Expose it with the Ingress or Gateway your cluster already runs. Nothing in `base` is public.
