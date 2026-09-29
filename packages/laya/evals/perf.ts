// Phase 10 perf driver (plan, Phase 10). Runs on the reference host; results land in
// evals/results/perf/<host>/<target>-<checkpoint>-<precision>.json.
//
//   bun run eval:perf --make-workload
//   bun run eval:perf --target node|bun   --checkpoint c --bundle-dir <dir>
//   bun run eval:perf --target python     --checkpoint c --runtime torch|onnx-fp32|onnx-int8
//   bun run eval:perf --target docker     --checkpoint c [--image laya-server:eval]
//   bun run eval:perf --target k8s        --checkpoint c --overlay base|all-checkpoints --url http://<node>:30080
//
// In-process targets spawn perf-worker.mjs (node, bun) or evals/python/perf.py and add cold and
// warm model-cache starts (page cache dropped before each cold start; needs root). HTTP targets
// time POST /v1/systemone: latency at 1, 5 and 10 questions, throughput at concurrency 1, 4, 16.
import { execFileSync, spawn } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { parseArgs } from "node:util";
import { hostInfo } from "./host.js";

const here = dirname(new URL(import.meta.url).pathname);
const root = resolve(here, "../../..");
const { values } = parseArgs({
  options: {
    target: { type: "string" },
    checkpoint: { type: "string", default: "multilingual" },
    precision: { type: "string", default: "fp32" },
    runtime: { type: "string", default: "torch" },
    "bundle-dir": { type: "string" },
    image: { type: "string", default: "laya-server:eval" },
    overlay: { type: "string", default: "base" },
    url: { type: "string" },
    "api-key": { type: "string", default: process.env.LAYA_API_KEY ?? "" },
    host: { type: "string", default: "hetzner-ccx23" },
    "start-trials": { type: "string", default: "3" },
    "make-workload": { type: "boolean", default: false },
  },
});
const workloadPath = resolve(root, "evals/out/perf/workload.json");
const trials = Number(values["start-trials"]);

// ---- workload: upstream's bench_latency.py questions and state, plus 96 sampled suite states ----
if (values["make-workload"]) {
  const suites = ["ag_news", "emotion", "support_triage", "email_spam", "phishing", "rag_relevance", "model_routing"];
  const pool = suites.flatMap((s) =>
    readFileSync(resolve(root, "evals/datasets/build", `${s}.jsonl`), "utf8")
      .split("\n")
      .filter((l) => l.trim())
      .map((l) => (JSON.parse(l) as { state: unknown }).state),
  );
  let seed = 13;
  const rnd = () => {
    seed = (seed * 1103515245 + 12345) % 2147483648;
    return seed / 2147483648;
  };
  const states = Array.from({ length: 96 }, () => pool[Math.floor(rnd() * pool.length)]);
  const text = "Hi, my Stripe payouts have failed for 3 days and I am losing sales. Please help ASAP. ".repeat(6);
  const workload = {
    note: "bench_latency.py STATE_EN / Q_NOUL / Q_CHOICE; states sampled (seed 13) from the Phase 4 suites",
    control_state: { ticket: { subject: "Payout failing", messages: [{ from: "customer", text }] } },
    q_noul: { type: "noul", instructions: "Does `ticket.messages[0].text` express urgency?" },
    q_choice: {
      type: "choice",
      instructions: "Which team should handle this?",
      criteria: { billing: "payments", technical: "bugs and integrations", sales: "pricing" },
    },
    states,
  };
  mkdirSync(dirname(workloadPath), { recursive: true });
  writeFileSync(workloadPath, JSON.stringify(workload));
  console.log(`wrote ${workloadPath} (${states.length} states from ${pool.length})`);
  process.exit(0);
}

const workload = JSON.parse(readFileSync(workloadPath, "utf8"));
const qs = (n: number) =>
  Object.fromEntries(Array.from({ length: n }, (_, i) => [`q${i}`, i % 2 ? workload.q_noul : workload.q_choice]));
const pct = (xs: number[], p: number) => {
  const s = [...xs].sort((a, b) => a - b);
  const k = (s.length - 1) * p;
  const lo = Math.floor(k);
  return s[lo] + (s[Math.ceil(k)] - s[lo]) * (k - lo);
};
const r2 = (x: number) => Math.round(x * 100) / 100;
const summary = (ts: number[]) => ({
  p50_ms: r2(pct(ts, 0.5)),
  p95_ms: r2(pct(ts, 0.95)),
  p99_ms: r2(pct(ts, 0.99)),
  mean_ms: r2(ts.reduce((a, b) => a + b, 0) / ts.length),
  n: ts.length,
});
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function dropCaches() {
  execFileSync("sh", ["-c", "sync && echo 3 > /proc/sys/vm/drop_caches"]);
}

/** Spawn a worker; resolve with its JSON line and the wall time from spawn to its READY line. */
function runWorker(
  cmd: string,
  args: string[],
  cwd = root,
): Promise<{ out: Record<string, unknown>; readyMs: number }> {
  return new Promise((res, rej) => {
    const t0 = Date.now();
    const p = spawn(cmd, args, { cwd, stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    let readyMs = Number.NaN;
    p.stdout.on("data", (d) => {
      stdout += d;
    });
    p.stderr.on("data", (d) => {
      stderr += d;
      const m = /READY (\d+)/.exec(stderr);
      if (m && Number.isNaN(readyMs)) readyMs = Number(m[1]) - t0;
    });
    p.on("close", (code) => {
      const line = stdout.trim().split("\n").pop() ?? "";
      if (code !== 0) return rej(new Error(`${cmd} exited ${code}: ${stderr.slice(-2000)}`));
      res({ out: JSON.parse(line), readyMs });
    });
  });
}

function workerCmd(): [string, string[]] {
  const t = values.target;
  if (t === "node" || t === "bun") {
    const bundle = values["bundle-dir"];
    if (!bundle) throw new Error("--bundle-dir is required for in-process targets");
    return [
      t,
      [
        resolve(here, "perf-worker.mjs"),
        "--checkpoint",
        values.checkpoint,
        "--bundle-dir",
        bundle,
        "--workload",
        workloadPath,
      ],
    ];
  }
  return [
    "uv",
    [
      "run",
      "--project",
      resolve(root, "evals"),
      "python",
      resolve(root, "evals/python/perf.py"),
      "--checkpoint",
      values.checkpoint,
      "--runtime",
      values.runtime,
      "--workload",
      workloadPath,
    ],
  ];
}

async function starts(fn: () => Promise<Record<string, number>>) {
  if (trials < 1) return null;
  const cold: Record<string, number>[] = [];
  const warm: Record<string, number>[] = [];
  for (let i = 0; i < trials; i++) {
    dropCaches();
    cold.push(await fn());
    warm.push(await fn());
  }
  const agg = (xs: Record<string, number>[]) =>
    Object.fromEntries(
      Object.keys(xs[0]).map((k) => [k, { ...summary(xs.map((x) => x[k])), samples: xs.map((x) => x[k]) }]),
    );
  return { cold: agg(cold), warm: agg(warm) };
}

async function inProcess() {
  const [cmd, args] = workerCmd();
  const start = await starts(async () => {
    const { out, readyMs } = await runWorker(cmd, [...args, "--mode", "start"]);
    return { to_first_answer_ms: readyMs, load_ms: out.load_ms as number, peak_rss_mb: out.peak_rss_mb as number };
  });
  const { out } = await runWorker(cmd, args);
  return { ...out, start };
}

// ---- HTTP targets ----------------------------------------------------------------------------
async function post(url: string, body: unknown): Promise<number> {
  const headers: Record<string, string> = { "content-type": "application/json" };
  if (values["api-key"]) headers.authorization = `Bearer ${values["api-key"]}`;
  const t = performance.now();
  const r = await fetch(`${url}/v1/systemone`, { method: "POST", headers, body: JSON.stringify(body) });
  await r.arrayBuffer();
  if (r.status !== 200) throw new Error(`POST ${url}/v1/systemone: ${r.status}`);
  return performance.now() - t;
}

async function waitHealthy(url: string, timeoutMs: number): Promise<number> {
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    try {
      const r = await fetch(`${url}/health`);
      await r.arrayBuffer();
      if (r.status === 200) return Date.now() - t0;
    } catch {}
    await sleep(100);
  }
  throw new Error(`${url}/health not 200 after ${timeoutMs} ms`);
}

async function httpLoad(url: string, model: string | null) {
  const body = (state: unknown, n: number) => ({ state, questions: qs(n), ...(model ? { model } : {}) });
  const control: Record<string, unknown> = {};
  const latency: Record<string, unknown> = {};
  for (const n of [1, 5, 10]) {
    for (let i = 0; i < 2; i++) await post(url, body(workload.control_state, n));
    const c: number[] = [];
    for (let i = 0; i < 10; i++) c.push(await post(url, body(workload.control_state, n)));
    control[`${n}_questions`] = summary(c);
    const ts: number[] = [];
    for (let i = 0; i < 22; i++) {
      const d = await post(url, body(workload.states[i % workload.states.length], n));
      if (i >= 2) ts.push(d);
    }
    latency[`${n}_questions`] = summary(ts);
  }
  const throughput: Record<string, unknown> = {};
  for (const conc of [1, 4, 16]) {
    const total = Math.max(48, conc * 6);
    let next = 0;
    const lat: number[] = [];
    const t0 = performance.now();
    await Promise.all(
      Array.from({ length: conc }, async () => {
        while (next < total) {
          const i = next++;
          lat.push(await post(url, body(workload.states[i % workload.states.length], 2)));
        }
      }),
    );
    const s = (performance.now() - t0) / 1000;
    throughput[`concurrency_${conc}`] = {
      requests: total,
      classifications: total * 2,
      seconds: r2(s),
      per_s: r2((total * 2) / s),
      ...summary(lat),
    };
  }
  return { control, latency, throughput };
}

function sh(cmd: string): string {
  return execFileSync("sh", ["-c", cmd], { encoding: "utf8" }).trim();
}

async function docker() {
  const port = 18000;
  const name = "laya-perf";
  const url = `http://127.0.0.1:${port}`;
  const run = () =>
    sh(`docker run -d --name ${name} -p ${port}:8000 -e LAYA_MODELS=${values.checkpoint} ${values.image}`);
  const rm = () => sh(`docker rm -f ${name} >/dev/null 2>&1 || true`);
  rm();
  const start = await starts(async () => {
    const t0 = Date.now();
    run();
    const health = await waitHealthy(url, 300_000);
    await post(url, { state: workload.states[0], questions: qs(1) });
    const first = Date.now() - t0;
    rm();
    return { to_health_ms: health, to_first_answer_ms: first };
  });
  run();
  await waitHealthy(url, 300_000);
  const load = await httpLoad(url, null);
  const peak = Number(sh(`docker exec ${name} cat /sys/fs/cgroup/memory.peak`));
  const image = sh(`docker image inspect ${values.image} --format '{{.Id}} {{.Size}}'`).split(" ");
  rm();
  return {
    ...load,
    start,
    memory_peak_mb: Math.round(peak / 2 ** 20),
    image: { id: image[0], bytes: Number(image[1]) },
  };
}

async function k8s() {
  const url = values.url;
  if (!url) throw new Error("--url http://<kind node ip>:30080 is required");
  const ns = "default";
  const sel = "app.kubernetes.io/name=laya-server";
  const k = (a: string) => sh(`kubectl -n ${ns} ${a}`);
  const pods = () => k(`get pods -l ${sel} -o jsonpath='{.items[*].metadata.name}'`).split(" ").filter(Boolean);
  const replicas = Number(k("get deployment/laya-server -o jsonpath='{.spec.replicas}'"));
  // Scale to zero and back, so every pod of a trial starts on the same (cold or warm) page cache.
  const scaleDown = () => {
    k("scale deployment/laya-server --replicas=0");
    k(`wait --for=delete pod -l ${sel} --timeout=300s`);
  };
  scaleDown();
  const start = await starts(async () => {
    const t0 = Date.now();
    k(`scale deployment/laya-server --replicas=${replicas}`);
    k("rollout status deployment/laya-server --timeout=900s");
    const ready = Date.now() - t0;
    await post(url, { state: workload.states[0], questions: qs(1), model: values.checkpoint });
    const first = Date.now() - t0;
    scaleDown();
    return { scale_up_to_ready_ms: ready, to_first_answer_ms: first };
  });
  k(`scale deployment/laya-server --replicas=${replicas}`);
  k("rollout status deployment/laya-server --timeout=900s");
  const model = values.overlay === "base" ? null : values.checkpoint;
  const load = await httpLoad(url, model);
  const names = pods();
  const peaks = names.map((p) => Number(k(`exec ${p} -c laya-server -- cat /sys/fs/cgroup/memory.peak`)));
  return {
    ...load,
    start,
    overlay: values.overlay,
    replicas: names.length,
    memory_peak_mb: Math.round(Math.max(...peaks) / 2 ** 20),
    memory_peak_mb_by_pod: peaks.map((p) => Math.round(p / 2 ** 20)),
  };
}

const target = values.target;
if (!target) throw new Error("--target is required");
const result = target === "docker" ? await docker() : target === "k8s" ? await k8s() : await inProcess();
const precision = target === "python" && values.runtime === "onnx-int8" ? "int8" : values.precision;
const label =
  target === "python"
    ? `python-${values.runtime.replace("-fp32", "").replace("-int8", "")}`
    : target === "k8s"
      ? `k8s-${values.overlay}`
      : target;
const outDir = resolve(root, "evals/results/perf", values.host);
mkdirSync(outDir, { recursive: true });
const file = resolve(outDir, `${label}-${values.checkpoint}-${precision}.json`);
const versions = (cmd: string) => {
  try {
    return sh(cmd);
  } catch {
    return null;
  }
};
writeFileSync(
  file,
  `${JSON.stringify(
    {
      meta: {
        target: label,
        checkpoint: values.checkpoint,
        precision,
        host: { label: values.host, ...hostInfo() },
        versions: {
          node: versions("node --version"),
          bun: versions("bun --version"),
          python: versions(`uv run --project ${resolve(root, "evals")} python --version`),
          docker: target === "docker" ? versions("docker --version") : undefined,
          kind: target === "k8s" ? versions("kind --version") : undefined,
        },
        date: new Date().toISOString(),
      },
      ...result,
    },
    null,
    2,
  )}\n`,
);
console.log(`wrote ${file}`);
process.exit(0);
