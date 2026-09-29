// Container smoke for the laya-server image (plan, Phase 9). Run: bun run smoke:image
//
// Expects an image already built from packages/laya-server/Dockerfile. Checks, in order:
//   1. the image is under LAYA_SMOKE_MAX_BYTES (default 2.0 GB) and runs as a non-root user;
//   2. `docker run` with the baked multilingual fp32 bundle: /health 200, then one
//      POST /v1/systemone (with bearer auth on), and /health never echoes the key;
//   3. with LAYA_SMOKE_EXTRA=<checkpoint> (HF_TOKEN optional): LAYA_MODELS=multilingual,<checkpoint>
//      and a cache volume fetches the extra checkpoint, and /health lists both.
// Secrets reach `docker run` as bare `-e NAME` pass-throughs, never as values on the command line.
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { randomBytes } from "node:crypto";
import { checkServer, waitForHealth } from "../packages/laya-server/test/smoke.ts";

const image = process.env.LAYA_SMOKE_IMAGE ?? "laya-server:smoke";
const maxBytes = Number(process.env.LAYA_SMOKE_MAX_BYTES ?? 2.0e9);
const extra = process.env.LAYA_SMOKE_EXTRA?.trim();

function docker(args: string[], env: Record<string, string> = {}): string {
  const r = spawnSync("docker", args, { env: { ...process.env, ...env }, encoding: "utf8" });
  if (r.error) throw new Error(`cannot run docker: ${r.error.message}`);
  if (r.status !== 0) throw new Error(`docker ${args[0]} failed (${r.status}): ${r.stderr.trim()}`);
  return r.stdout.trim();
}

async function withContainer(
  name: string,
  runArgs: string[],
  env: Record<string, string>,
  body: (base: string, id: string) => Promise<void>,
) {
  const id = docker(["run", "-d", "--name", name, "-p", "127.0.0.1::8000", ...runArgs, image], env);
  try {
    const port = docker(["port", id, "8000/tcp"]).split("\n")[0].split(":").pop();
    const base = `http://127.0.0.1:${port}`;
    const alive = () => docker(["inspect", "-f", "{{.State.Running}}", id]) === "true";
    await waitForHealth(base, 10 * 60_000, alive);
    await body(base, id);
  } catch (e) {
    const logs = spawnSync("docker", ["logs", "--tail", "200", id], { encoding: "utf8" });
    console.error(`--- ${name} logs ---\n${logs.stdout}${logs.stderr}`);
    throw e;
  } finally {
    spawnSync("docker", ["rm", "-f", id]);
  }
}

async function main() {
  const size = Number(docker(["image", "inspect", "-f", "{{.Size}}", image]));
  console.log(`smoke:image: ${image} is ${(size / 1e9).toFixed(3)} GB`);
  assert.ok(size < maxBytes, `image is ${size} bytes, over the ${maxBytes} byte budget`);
  const user = docker(["image", "inspect", "-f", "{{.Config.User}}", image]);
  assert.ok(user && !/^(0|root)(:|$)/.test(user), `image runs as ${JSON.stringify(user || "root")}`);

  const apiKey = randomBytes(16).toString("hex");
  await withContainer(
    `laya-smoke-${process.pid}`,
    ["-e", "LAYA_API_KEY"],
    { LAYA_API_KEY: apiKey },
    async (base, id) => {
      assert.equal(docker(["exec", id, "id", "-u"]), "1000", "container is not running as uid 1000");
      await checkServer(base, { apiKey, secrets: [apiKey] });
    },
  );

  if (extra) {
    const token = process.env.HF_TOKEN; // optional: the artifact store is public
    const volume = `laya-smoke-cache-${process.pid}`;
    try {
      await withContainer(
        `laya-smoke-extra-${process.pid}`,
        ["-e", "HF_TOKEN", "-e", `LAYA_MODELS=multilingual,${extra}`, "-v", `${volume}:/cache`],
        {},
        async (base) => {
          const text = await (await fetch(`${base}/health`)).text();
          if (token) assert.ok(!text.includes(token), "/health echoes HF_TOKEN");
          const health = JSON.parse(text);
          assert.ok(health.loaded.includes("multilingual") && health.loaded.includes(extra), `loaded: ${text}`);
          console.log(`smoke:image: cache volume fetched ${extra}; /health lists ${health.loaded.join(",")}`);
        },
      );
    } finally {
      spawnSync("docker", ["volume", "rm", "-f", volume]);
    }
  }
  console.log("smoke:image: ok");
}

main().catch((e: unknown) => {
  console.error(`smoke:image: FAILED: ${e instanceof Error ? e.message : String(e)}`);
  process.exit(1);
});
