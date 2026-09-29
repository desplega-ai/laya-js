# DigitalOcean App Platform

**Status: untested.** **Verdict: works with caveats.** The binding constraint is image size: DigitalOcean recommends images under 1 GiB and warns that images over 2 GiB "are likely to experience build and deployment issues". Ours is 1.6 GB uncompressed. Limits checked 2026-09-29.

| Limit | Value | Source |
| --- | --- | --- |
| Image size | no hard limit; under 1 GiB recommended, over 2 GiB likely to fail | [limits](https://docs.digitalocean.com/products/app-platform/details/limits/) |
| Local filesystem | 4 GiB; the container is replaced if it fills | same |
| Instance sizes | up to `apps-d-8vcpu-32gb` | [pricing](https://docs.digitalocean.com/products/app-platform/details/pricing/) |
| Health check | `initial_delay_seconds` 0 to 3600, `failure_threshold` 1 to 50 | [app spec](https://docs.digitalocean.com/products/app-platform/reference/app-spec/) |

The 4 GiB local disk rules out fetching `english` and `typed-decisions` (3.4 GB) at startup. Serve `multilingual` only, or run the other checkpoints on a Droplet or DOKS ([kubernetes.md](kubernetes.md)).

## Deploy

```sh
doctl registry create <registry>
doctl registry login
docker tag laya-server:local registry.digitalocean.com/<registry>/laya-server:<tag>
docker push registry.digitalocean.com/<registry>/laya-server:<tag>
```

```yaml
# app.yaml
name: laya-server
services:
  - name: laya-server
    image:
      registry_type: DOCR
      repository: laya-server
      tag: <tag>
    http_port: 8000
    instance_size_slug: apps-s-2vcpu-4gb
    instance_count: 1
    envs:
      - key: LAYA_THREADS
        value: "2"
      - key: LAYA_API_KEY
        type: SECRET
        value: <openssl rand -hex 16>
    health_check:
      http_path: /health
      initial_delay_seconds: 10
      period_seconds: 10
      timeout_seconds: 5
      failure_threshold: 30
```

```sh
doctl apps create --spec app.yaml
```

GHCR and Docker Hub images work too (`registry_type: GHCR` or `DOCKER_HUB`, with `registry_credentials` for private ones).
