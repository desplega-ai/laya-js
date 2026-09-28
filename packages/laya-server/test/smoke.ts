// Real-bundle smoke test for laya-server (plan Phase 8, "Automated QA").
//
//   bun run --filter @desplega/laya-server build && bun packages/laya-server/test/smoke.ts
//
// Starts `node packages/laya-server/dist/main.js` with LAYA_MODELS=multilingual (fetching the
// fp32 bundle from the private artifact store, so HF_TOKEN must be set), waits for /health 200,
// sends one POST /v1/systemone, checks auth and that no secret leaks into /health, then sends
// SIGTERM and expects a clean exit. With LAYA_SMOKE_URL set it only runs the HTTP checks against
// an already-running server (used by the image smoke).
import assert from "node:assert/strict";
import { type ChildProcess, spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const SMOKE_QUESTIONS = {
  department: {
    type: "choice",
    instructions: "Which team should handle `message`?",
    criteria: ["billing", "technical", "other"],
  },
  urgent: { type: "noul", instructions: "Does `message` need a reply today?" },
};
export const SMOKE_ENGLISH = "The server is down and none of our customers can log in. Please help us today.";
export const SMOKE_STATE = {
  message: "Mi factura se cobró dos veces este mes, por favor devuelvan uno de los cargos.",
};

async function sleep(ms: number) {
  await new Promise((r) => setTimeout(r, ms));
}

/** Poll /health until 200, failing early if `alive()` reports the server died. */
export async function waitForHealth(base: string, timeoutMs: number, alive: () => boolean = () => true) {
  const deadline = Date.now() + timeoutMs;
  let last = "no response";
  while (Date.now() < deadline) {
    if (!alive()) throw new Error(`server exited before /health turned 200 (last: ${last})`);
    try {
      const res = await fetch(`${base}/health`);
      last = `HTTP ${res.status}`;
      if (res.status === 200) return (await res.json()) as Record<string, unknown>;
      if (res.status !== 503) throw new Error(`/health answered ${res.status}: ${await res.text()}`);
    } catch (e) {
      if (e instanceof Error && e.message.startsWith("/health answered")) throw e;
    }
    await sleep(2000);
  }
  throw new Error(`/health not 200 after ${timeoutMs / 1000}s (last: ${last})`);
}

/** The HTTP checks, against a server that is already up. `secrets` must not appear in /health. */
export async function checkServer(base: string, opts: { apiKey?: string; secrets?: string[] } = {}) {
  const healthRes = await fetch(`${base}/health`);
  assert.equal(healthRes.status, 200);
  const healthText = await healthRes.text();
  for (const s of opts.secrets ?? []) assert.ok(!healthText.includes(s), "/health echoes a secret");
  const health = JSON.parse(healthText);
  assert.ok(health.loaded.includes("multilingual"), `multilingual not loaded: ${healthText}`);
  assert.equal(health.precision, "fp32");
  assert.equal(health.device, "cpu");

  const body = JSON.stringify({ state: SMOKE_STATE, questions: SMOKE_QUESTIONS, model: "multilingual" });
  const headers: Record<string, string> = { "content-type": "application/json" };
  if (opts.apiKey) {
    const denied = await fetch(`${base}/v1/systemone`, { method: "POST", headers, body });
    assert.equal(denied.status, 401, "request without the bearer token was not refused");
    headers.authorization = `Bearer ${opts.apiKey}`;
  }
  const t0 = performance.now();
  const res = await fetch(`${base}/v1/systemone`, { method: "POST", headers, body });
  const text = await res.text();
  assert.equal(res.status, 200, `POST /v1/systemone -> ${res.status}: ${text}`);
  const r = JSON.parse(text);
  assert.equal(r.routing?.model, "multilingual");
  const dept = r.answers?.department;
  assert.ok(SMOKE_QUESTIONS.department.criteria.includes(dept?.choice), `choice outside the labels: ${text}`);
  const total = Object.values(dept.probabilities as Record<string, number>).reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(total - 1) < 1e-3, `probabilities sum to ${total}`);
  const noul = r.answers?.urgent?.noul;
  assert.ok(typeof noul === "number" && noul >= 0 && noul <= 1, `noul answer missing: ${text}`);
  assert.ok(r.usage?.input_tokens > 0, `usage missing: ${text}`);
  assert.ok(res.headers.get("x-inference-time-ms"), "X-Inference-Time-Ms header missing");
  // No `model`, English text: routing picks `english`, which is not in LAYA_MODELS, so the
  // server must answer with its default checkpoint and say so, never load or fetch english.
  const en = await fetch(`${base}/v1/systemone`, {
    method: "POST",
    headers,
    body: JSON.stringify({ state: SMOKE_ENGLISH, questions: SMOKE_QUESTIONS }),
  });
  const enText = await en.text();
  assert.equal(en.status, 200, `POST /v1/systemone (no model, English) -> ${en.status}: ${enText}`);
  const enRouting = JSON.parse(enText).routing;
  assert.equal(enRouting?.model, "multilingual", `English text not served by the default: ${enText}`);
  assert.match(enRouting.reason, /"english" is not in LAYA_MODELS, served by "multilingual"$/, enText);
  const after = JSON.parse(await (await fetch(`${base}/health`)).text());
  assert.deepEqual(after.loaded, ["multilingual"], "a request loaded a checkpoint outside LAYA_MODELS");
  console.log(`smoke: English without model -> ${enRouting.reason}`);
  console.log(
    `smoke: department=${dept.choice} (${JSON.stringify(dept.probabilities)}), urgent=${r.answers.urgent.noul}, ` +
      `${(performance.now() - t0).toFixed(0)} ms`,
  );
}

async function stop(child: ChildProcess): Promise<number | null> {
  if (child.exitCode !== null) return child.exitCode;
  const exited = new Promise<number | null>((r) => child.once("exit", (code) => r(code)));
  child.kill("SIGTERM");
  const timer = setTimeout(() => child.kill("SIGKILL"), 30_000);
  const code = await exited;
  clearTimeout(timer);
  return code;
}

async function main() {
  const external = process.env.LAYA_SMOKE_URL;
  if (external) {
    await waitForHealth(external, 10 * 60_000);
    await checkServer(external, { apiKey: process.env.LAYA_SMOKE_API_KEY });
    console.log("smoke: ok");
    return;
  }
  if (!process.env.HF_TOKEN) {
    throw new Error(
      "HF_TOKEN is not set: the smoke fetches the multilingual fp32 bundle from the private artifact store",
    );
  }
  const here = dirname(fileURLToPath(import.meta.url));
  const entry = resolve(here, "../dist/main.js");
  const port = String(18000 + Math.floor(Math.random() * 1000));
  const apiKey = randomBytes(16).toString("hex");
  const cacheDir = process.env.LAYA_CACHE_DIR ?? (await mkdtemp(join(tmpdir(), "laya-smoke-")));
  const child = spawn(process.env.LAYA_SMOKE_NODE ?? "node", [entry], {
    env: {
      ...process.env,
      LAYA_HOST: "127.0.0.1",
      LAYA_PORT: port,
      LAYA_MODELS: "multilingual",
      LAYA_MODEL_DIR: join(cacheDir, "no-baked-models"),
      LAYA_CACHE_DIR: cacheDir,
      LAYA_API_KEY: apiKey,
    },
    stdio: ["ignore", "pipe", "pipe"],
  });
  let logs = "";
  const capture = (b: Buffer) => {
    const s = b.toString();
    logs += s;
    process.stdout.write(s);
  };
  child.stdout?.on("data", capture);
  child.stderr?.on("data", capture);
  const base = `http://127.0.0.1:${port}`;
  try {
    await waitForHealth(base, 15 * 60_000, () => child.exitCode === null);
    await checkServer(base, { apiKey, secrets: [apiKey, process.env.HF_TOKEN] });
  } catch (e) {
    await stop(child);
    throw e;
  }
  const code = await stop(child);
  for (const s of [apiKey, process.env.HF_TOKEN]) assert.ok(!logs.includes(s), "server log contains a secret");
  assert.equal(code, 0, `server exited ${code} on SIGTERM`);
  console.log("smoke: ok");
}

if (resolve(process.argv[1] ?? "") === fileURLToPath(import.meta.url)) {
  main().catch((e: unknown) => {
    console.error(`smoke: FAILED: ${e instanceof Error ? e.message : String(e)}`);
    process.exit(1);
  });
}
