# Hetzner Cloud VM

**Status: untested on Hetzner.** The same server process (`node packages/laya-server/dist/main.js`) was run and measured on a local Linux x64 host ([README](README.md#measured-footprint)); the Docker image is smoke-tested in CI. **Verdict: works.** The only limit is the VM's RAM. Checked 2026-09-29.

| Server type | Arch | vCPU | RAM | Fits |
| --- | --- | --- | --- | --- |
| CX23 | x86 | 2 | 4 GB | one checkpoint, no headroom for anything else |
| CX33 | x86 | 4 | 8 GB | one checkpoint comfortably; three is tight |
| CX43 | x86 | 8 | 16 GB | all three |
| CAX21 | arm64 | 4 | 8 GB | one checkpoint; needs the `linux/arm64` image |

Specs and current prices: [Hetzner Cloud](https://www.hetzner.com/cloud/) and the [2026 price adjustment](https://docs.hetzner.com/general/infrastructure-and-availability/price-adjustment/). Pick CX33 for one checkpoint.

## Deploy with cloud-init

This installs Docker, logs in to your registry, and runs the image behind Caddy for TLS. Replace the placeholders, save it as `cloud-init.yaml`, and create the server:

```yaml
#cloud-config
package_update: true
packages: [docker.io]
write_files:
  - path: /etc/laya.env
    permissions: "0600"
    content: |
      LAYA_API_KEY=<openssl rand -hex 16>
      LAYA_THREADS=4
  - path: /etc/caddy/Caddyfile
    content: |
      <your.domain> {
        reverse_proxy laya:8000
      }
runcmd:
  - echo '<registry read token>' | docker login ghcr.io -u <user> --password-stdin
  - docker network create laya
  - docker volume create laya-cache
  - >-
    docker run -d --name laya --network laya --restart unless-stopped
    --env-file /etc/laya.env --memory 6g --read-only --tmpfs /tmp
    -v laya-cache:/cache ghcr.io/<org>/laya-server:<tag>
  - >-
    docker run -d --name caddy --network laya --restart unless-stopped
    -p 80:80 -p 443:443 -v /etc/caddy/Caddyfile:/etc/caddy/Caddyfile:ro
    -v caddy-data:/data caddy:2
```

```sh
hcloud ssh-key create --name laya --public-key-from-file ~/.ssh/id_ed25519.pub
hcloud server create --name laya --type cx33 --image ubuntu-24.04 --location fsn1 \
  --ssh-key laya --user-data-from-file cloud-init.yaml
hcloud firewall create --name laya
hcloud firewall add-rule laya --direction in --protocol tcp --port 22 --source-ips <your ip>/32
hcloud firewall add-rule laya --direction in --protocol tcp --port 80 --source-ips 0.0.0.0/0 --source-ips ::/0
hcloud firewall add-rule laya --direction in --protocol tcp --port 443 --source-ips 0.0.0.0/0 --source-ips ::/0
hcloud firewall apply-to-resource laya --type server --server laya
```

Point `<your.domain>` at the server's IP before Caddy starts, or it cannot get a certificate. The token in `runcmd` stays in the server's cloud-init logs: use a read-only registry token, or skip the login and `docker load` an image you copy over with `scp`.

## Without Docker

The server is a plain Node process. On an Ubuntu VM with Node 22 and Bun 1.4:

```sh
git clone https://github.com/desplega-ai/laya-js && cd laya-js
ONNXRUNTIME_NODE_INSTALL=skip bun install --frozen-lockfile   # skip the CUDA download; CPU binaries ship in the package
bun run --filter @desplega/laya-server build
HF_TOKEN=<read token> node packages/laya-server/dist/fetch-models.js --dest /var/lib/laya/models multilingual
LAYA_MODEL_DIR=/var/lib/laya/models LAYA_CACHE_DIR=/var/lib/laya/cache \
  LAYA_API_KEY=<key> LAYA_THREADS=4 node packages/laya-server/dist/main.js
```

This is the path measured locally. Wrap it in a systemd unit with `Restart=always` and `EnvironmentFile=` for production.
