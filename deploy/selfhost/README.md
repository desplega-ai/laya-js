# Self-hosted auto-deploy (pull-based)

Reference copy of what runs on the swarm host (`laya.agent-swarm.dev`): a systemd timer polls the
newest stable `v*` tag of this repo every 5 minutes and, on a new one, builds it, swaps the `laya`
container, checks it, and rolls back if the check fails. No host credential is stored in GitHub.

| File | Installed as |
| --- | --- |
| `deploy.sh` | `/opt/laya/deploy.sh` |
| `laya-deploy.service` | `/etc/systemd/system/laya-deploy.service` (oneshot) |
| `laya-deploy.timer` | `/etc/systemd/system/laya-deploy.timer` (first poll 1 min after enable, then 5 min after each run) |
| `install-autodeploy.sh` | one-shot installer, embeds the three files above: `ssh <host> 'bash -s' < install-autodeploy.sh` |

`install-autodeploy.sh` is generated from the other three files. After editing any of them, replace
the matching heredoc body in the installer with the new file content.

## Contract with `/opt/laya/run.sh`

`run.sh` must (re)create the container `laya` from the image `laya-server:current` with the
production flags (`--memory`, `LAYA_MAX_LOADED`, `LAYA_THREADS`, the env file, the cache volume).
`deploy.sh` never builds the `docker run` line itself. The installer refuses to run when `run.sh`
does not mention `laya-server:current`.

## One deploy

1. Newest stable `vX.Y.Z` tag from the GitHub API. Same as the running image's
   `org.opencontainers.image.version` label, or older: stop.
2. `@desplega.ai/laya-server@X.Y.Z` must exist on npm, so a tag whose publish failed never deploys.
3. Build `laya-server:vX.Y.Z` from the tag tarball. The live container is untouched. Three failed
   builds mark the tag failed.
4. Tag the running image `laya-server:prev`, point `laya-server:current` at the new image, run `run.sh`.
5. Within 15 min `/health` must be `ok` with `multilingual`, `english` and `typed-decisions` loaded.
   Then one classify call per model and a 20-request load per model run inside the container (its own
   `LAYA_API_KEY`), and the peak RSS (`VmHWM`) and cgroup peak are logged.
6. Any failure in 4 or 5: `current` goes back to the previous image, `run.sh` runs again, `/health` is
   re-checked, and the tag is marked failed in `/opt/laya/state/failed-<tag>` (no retry loop).

The swap is remove-and-recreate, so `/health` is down for the model load (about 15 s with a warm
`/cache` volume, longer on a first fetch). Caddy and other containers are never touched.

## Operate

```sh
systemctl list-timers laya-deploy.timer        # next poll
journalctl -u laya-deploy.service -n 100       # what the last runs did
/opt/laya/deploy.sh --tag v0.1.1 --force       # deploy or re-deploy a tag by hand (also downgrades)
rm /opt/laya/state/failed-v0.1.2               # let a failed tag retry
systemctl disable --now laya-deploy.timer      # stop auto-deploys
docker tag laya-server:prev laya-server:current && /opt/laya/run.sh   # manual rollback
```

Overrides (environment of the service): `LAYA_EXPECT_MODELS`, `LAYA_HEALTH_TIMEOUT`,
`LAYA_LOAD_PER_MODEL`, `LAYA_RUN_SCRIPT`, `LAYA_DIR`, `LAYA_IMAGE`, `LAYA_CONTAINER`.
