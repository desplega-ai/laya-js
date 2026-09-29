# Fly.io

**Status: untested.** **Verdict: works with caveats.** The binding constraint is disk read speed: a Machine's root filesystem is capped at 8 MiB/s, so reading the 1.3 GB baked model could take minutes on first load. Limits checked 2026-09-29.

| Limit | Value | Source |
| --- | --- | --- |
| Image | 8 GB rootfs limit | [troubleshooting](https://docs.fly.io/getting-started/troubleshooting/) |
| Rootfs throughput | "maximum of 2000 IOPs and 8MiB/s bandwidth" regardless of Machine type | [volumes overview](https://docs.fly.io/volumes/overview/) |
| Volume throughput | performance-1x 48 MiB/s, performance-2x 64 MiB/s | same |
| Memory | up to 2 GB per shared CPU, 8 GB per performance CPU | [machine sizing](https://docs.fly.io/machines/guides-examples/machine-sizing/) |

At 8 MiB/s, 1.3 GB is about 160 s. We have not measured whether the page cache or Fly's image caching hides that. The config below gives the health check a 300 s grace period, and the variant after it moves the model onto a volume.

## Deploy

```sh
fly apps create laya-server
fly secrets set LAYA_API_KEY="$(openssl rand -hex 16)"
fly auth docker
docker tag laya-server:local registry.fly.io/laya-server:<tag>
docker push registry.fly.io/laya-server:<tag>
fly deploy --image registry.fly.io/laya-server:<tag>
```

```toml
# fly.toml
app = "laya-server"
primary_region = "<region>"

[env]
  LAYA_THREADS = "1"

[http_service]
  internal_port = 8000
  force_https = true
  auto_stop_machines = "off"     # "stop" saves money; every start reloads the model
  min_machines_running = 1
  [http_service.concurrency]
    type = "requests"
    soft_limit = 12
    hard_limit = 16              # LAYA_MAX_CONCURRENT

  [[http_service.checks]]
    grace_period = "300s"
    interval = "15s"
    timeout = "5s"
    method = "GET"
    path = "/health"

[[vm]]
  size = "performance-1x"
  memory = "4gb"

[[restart]]
  policy = "always"
```

Use performance CPUs: shared CPUs are throttled under sustained inference.

## Model on a volume

To escape the rootfs cap, fetch the checkpoint into a volume once and load it from there. Pointing `LAYA_MODEL_DIR` at an empty path makes the server fetch every listed checkpoint into `LAYA_CACHE_DIR`:

```sh
fly volumes create laya_cache --size 10 --region <region>
fly secrets set HF_TOKEN=<optional: Hugging Face read token, only for rate limits>
```

```toml
[env]
  LAYA_MODEL_DIR = "/nonexistent"
  LAYA_CACHE_DIR = "/data"

[mounts]
  source = "laya_cache"
  destination = "/data"
```

A volume attaches to one Machine: create one per Machine you run.
