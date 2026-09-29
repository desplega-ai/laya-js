# Koyeb

**Status: untested.** **Verdict: works.** Use instance `large` (4 vCPU / 4 GB) or larger; `medium` (2 GB) is below the 2.2 GiB load peak. Limits checked 2026-09-29.

| Limit | Value | Source |
| --- | --- | --- |
| Instances | `large` 4 vCPU / 4 GB / 40 GB SSD; `xlarge` 8 / 8 GB; `2xlarge` 16 / 16 GB; `eco-large` 2 / 4 GB | [instances](https://www.koyeb.com/docs/reference/instances) |
| Image | uncompressed size under 5 GB plus the instance's local SSD | [storage](https://www.koyeb.com/docs/reference/storage) |
| Health checks | TCP or HTTP; grace period 5 to 900 s | [health checks](https://www.koyeb.com/docs/run-and-scale/health-checks) |

## Deploy

```sh
koyeb secrets create laya-api-key --value "$(openssl rand -hex 16)"
koyeb app init laya-server \
  --docker ghcr.io/<org>/laya-server:<tag> \
  --instance-type large \
  --ports 8000:http --routes /:8000 \
  --env LAYA_THREADS=4 \
  --env 'LAYA_API_KEY={{ secret.laya-api-key }}' \
  --checks 8000:http:/health --checks-grace-period 8000=300 \
  --min-scale 1
```

The flags come from the [CLI reference](https://www.koyeb.com/docs/build-and-deploy/cli/reference); check `koyeb app init --help` against your CLI version, since secret-reference syntax has changed between releases. A private image also needs a registry secret passed with `--docker-private-registry-secret`.

For all three checkpoints use `xlarge` (8 GB) or `2xlarge`, with `LAYA_MODELS=multilingual,english,typed-decisions` and `HF_TOKEN` as a secret.
