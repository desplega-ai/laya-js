# Cloudflare

Limits checked 2026-09-29.

## Containers

**Status: untested.** **Verdict: works with caveats.** Containers sit behind a Worker and a Durable Object, so you write a few lines of Worker code. Use `standard-2` (1 vCPU / 6 GiB) or larger: `standard-1` has 4 GiB but only half a vCPU. Needs the Workers Paid plan.

| Limit | Value | Source |
| --- | --- | --- |
| Instance types | `standard-1` 1/2 vCPU, 4 GiB, 8 GB disk; `standard-2` 1 vCPU, 6 GiB, 12 GB; `standard-3` 2 vCPU, 8 GiB, 16 GB; `standard-4` 4 vCPU, 12 GiB, 20 GB | [limits](https://developers.cloudflare.com/containers/platform-details/limits/) |
| Image size | same as the instance's disk | same |
| Plan | Workers Paid | [pricing](https://developers.cloudflare.com/containers/pricing/) |

Wrangler builds the image from a Dockerfile with your local Docker. Point it at the image you already built, so the HF-token build step stays out of Wrangler:

```dockerfile
# Dockerfile (next to wrangler.jsonc)
FROM laya-server:local
```

```jsonc
// wrangler.jsonc
{
  "name": "laya-server",
  "main": "src/index.ts",
  "compatibility_date": "2026-09-29",
  "containers": [
    { "class_name": "Laya", "image": "./Dockerfile", "instance_type": "standard-2", "max_instances": 3 }
  ],
  "durable_objects": { "bindings": [{ "name": "LAYA", "class_name": "Laya" }] },
  "exports": { "Laya": { "type": "durable-object", "storage": "sqlite" } }
}
```

```ts
// src/index.ts
import { Container, getContainer } from "@cloudflare/containers";

interface Env {
  LAYA: DurableObjectNamespace<Laya>;
  LAYA_API_KEY: string;
}

export class Laya extends Container<Env> {
  defaultPort = 8000;
  sleepAfter = "30m"; // each wake-up reloads the model
  pingEndpoint = "localhost/health";

  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    this.envVars = { LAYA_THREADS: "1", LAYA_API_KEY: env.LAYA_API_KEY };
  }
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    // One named instance. Shard by a key (or use a load balancer helper) to spread load.
    return getContainer(env.LAYA, "main").fetch(request);
  },
};
```

```sh
npm i @cloudflare/containers
npx wrangler secret put LAYA_API_KEY
npx wrangler deploy
```

Unverified: whether the ping waits for `/health` to return 200 or accepts the 503 the server sends while loading. If it accepts 503, the first requests after a wake-up get 503 with `Retry-After: 5` until the model is loaded, which is safe for clients that retry.

For all three checkpoints use `standard-4` (12 GiB, 20 GB disk) with `LAYA_MODELS=multilingual,english,typed-decisions` and `HF_TOKEN` passed through `envVars` from a Worker secret.

## Workers

**Verdict: not viable.** A Worker has 128 MB of memory and a 64 MiB script size limit on every plan ([Workers limits](https://developers.cloudflare.com/workers/platform/limits/)). The model is 1.3 GB and needs 2.2 GiB resident. Workers cannot load native Node addons like `onnxruntime-node`, and `@desplega/laya` has no WASM backend: the vendored browser path (`onnxruntime-web`) was removed (`packages/laya/src/providers.ts`). Use Containers instead.
