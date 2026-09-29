# Heroku

**Status: untested.** **Verdict: works with caveats.** The binding constraint is memory: Performance-M has 2.5 GB, which leaves almost nothing over the 2.2 GiB (2.4 GB) load peak, so Performance-L (14 GB) is the smallest safe dyno. Limits checked 2026-09-29.

| Limit | Value | Source |
| --- | --- | --- |
| Dynos | Standard-2X 1 GB; Performance-M 2.5 GB; Performance-L 14 GB | [dyno types](https://devcenter.heroku.com/articles/dyno-types) |
| Image size | Docker images "are not subject to size restrictions", but are subject to the boot time limit | [container registry and runtime](https://devcenter.heroku.com/articles/container-registry-and-runtime) |
| Port | the web process must listen on `$PORT`; `EXPOSE` and `HEALTHCHECK` are ignored | same |
| Boot | the process must bind `$PORT` within 60 s (R10) | [limits](https://devcenter.heroku.com/articles/limits) |
| Router | 30 s to the first response byte | same |

R10 is not a problem: the server binds its port before loading the model and answers 503 until ready. Requests take well under a second, so the 30 s router window does not bind either.

## Deploy

Heroku sets `PORT` at runtime, so wrap the image to map it onto `LAYA_PORT`:

```dockerfile
# Dockerfile.heroku
FROM laya-server:local
CMD ["sh", "-c", "LAYA_PORT=$PORT exec node dist/main.js"]
```

```sh
heroku create laya-server --stack container
heroku ps:type web=performance-l -a laya-server
heroku config:set -a laya-server LAYA_API_KEY="$(openssl rand -hex 16)" LAYA_THREADS=2
docker build -f Dockerfile.heroku -t registry.heroku.com/laya-server/web .
heroku container:login
docker push registry.heroku.com/laya-server/web
heroku container:release web -a laya-server
```

Performance-L has room for all three checkpoints: add `LAYA_MODELS=multilingual,english,typed-decisions`, `HF_TOKEN`, and `LAYA_CACHE_DIR=/tmp/laya-cache` (the dyno filesystem is ephemeral, so every restart re-downloads 3.4 GB).
