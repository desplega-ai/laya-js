# Docker and Docker Compose

**Status: tested.** CI (`.github/workflows/image.yml`) builds this image on every PR that touches the server, checks it is under 2.0 GB and runs as a non-root user, starts it with `docker run`, waits for `/health` 200 and sends `POST /v1/systemone` (`bun run smoke:image`). The resource and read-only flags below and the Compose file are untested: CI runs the same image with only `-p` and `-e LAYA_API_KEY`.

Every other page in this guide deploys the image you build here.

## Build

The build context is the repo root. The `models` stage downloads `multilingual/fp32` from the private artifact store, so it needs a Hugging Face read token with access to `desplega/laya-onnx`. It is passed as a BuildKit secret and never lands in a layer.

```sh
export HF_TOKEN=hf_...   # read token scoped to desplega/laya-onnx
DOCKER_BUILDKIT=1 docker build -f packages/laya-server/Dockerfile \
  --secret id=hf_token,env=HF_TOKEN -t laya-server:local .
```

For arm64 hosts (Graviton, Ampere, Hetzner CAX), build for that platform. The Dockerfile keeps only the `linux/<arch>` onnxruntime binaries:

```sh
docker buildx build --platform linux/arm64 -f packages/laya-server/Dockerfile \
  --secret id=hf_token,env=HF_TOKEN -t laya-server:local-arm64 --load .
```

Push it to the registry your platform pulls from:

```sh
docker tag laya-server:local ghcr.io/<org>/laya-server:<tag>
docker push ghcr.io/<org>/laya-server:<tag>
```

`ghcr.io/desplega-ai/laya-server` is the name the k8s manifests use, but publishing is manual for now: there is no public image to pull.

## Run

```sh
docker run -d --name laya -p 8000:8000 \
  -e LAYA_API_KEY \
  -e LAYA_THREADS=2 --cpus 2 --memory 3g \
  --read-only --tmpfs /tmp -v laya-cache:/cache \
  laya-server:local

curl -s localhost:8000/health        # 503 while loading, 200 when ready
curl -s localhost:8000/v1/systemone \
  -H "authorization: Bearer $LAYA_API_KEY" -H 'content-type: application/json' \
  -d '{"state":{"message":"The server is down, please help today."},
       "questions":{"urgent":{"type":"noul","instructions":"Does `message` need a reply today?"}}}'
```

`-e LAYA_API_KEY` without a value passes the variable from your shell, so the key never appears in `ps` or shell history.

## Compose

```yaml
# compose.yaml
services:
  laya:
    image: laya-server:local          # or ghcr.io/<org>/laya-server:<tag>
    ports: ["8000:8000"]
    environment:
      LAYA_API_KEY: ${LAYA_API_KEY:?set LAYA_API_KEY}
      LAYA_THREADS: "2"               # keep equal to the CPUs you give it
      # Serve more checkpoints: the extras are fetched into /cache on first start.
      # LAYA_MODELS: multilingual,english
      # HF_TOKEN: ${HF_TOKEN}
    volumes:
      - laya-cache:/cache
    read_only: true
    tmpfs: [/tmp]
    cpus: 2
    mem_limit: 3g                     # one fp32 checkpoint; 8g for all three
    stop_grace_period: 30s            # the server drains in-flight requests for up to 25 s
    restart: unless-stopped
volumes:
  laya-cache:
```

```sh
LAYA_API_KEY=$(openssl rand -hex 16) docker compose up -d
```

The image's own `HEALTHCHECK` polls `/health` with a 300 s start period, so `docker compose ps` shows `healthy` once the model is loaded.
