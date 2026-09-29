// Phase 10 cost per 1M classifications (plan, Phase 10 "Cost"), from the perf results.
//
//   bun run eval:cost [--host hetzner-ccx23]
//
// A classification is one question answered for one state. Prices come from evals/prices.json;
// the run fails if any price there is older than 30 days.
// - in-process (node, bun, python-*): host €/h ÷ (best batched classifications/s × 3600) × 1e6;
// - docker: the same with the best HTTP throughput, p95 at that load beside it;
// - k8s: the pod's share of the node, max(cpu request / vCPU, memory request / GiB), because
//   Hetzner publishes one price per server and no per-resource split, ÷ the per-pod throughput
//   at 70% CPU (the HPA target), taken as 0.7 × the measured per-pod saturated throughput;
// - LLM reference (computed only, no API calls): Claude Haiku 4.5 list price × the suites' mean
//   input and output tokens per classification. Input tokens are estimated as UTF-8 bytes / 4 of
//   the state and questions as JSON plus 150 tokens of instructions; output is 10 tokens.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { parseArgs } from "node:util";

const here = dirname(new URL(import.meta.url).pathname);
const root = resolve(here, "../../..");
const { values } = parseArgs({ options: { host: { type: "string", default: "hetzner-ccx23" } } });

type Dated = { date: string };
const prices = JSON.parse(readFileSync(resolve(root, "evals/prices.json"), "utf8")) as {
  hosts: Record<string, Dated & { eur_per_hour: number; vcpu: number; memory_gib: number }>;
  llm: Record<string, Dated & { usd_per_mtok_input: number; usd_per_mtok_output: number }>;
  fx: Dated & { usd_per_eur: number };
};
const dated: [string, Dated][] = [...Object.entries(prices.hosts), ...Object.entries(prices.llm), ["fx", prices.fx]];
const stale = dated.filter(([, p]) => (Date.now() - Date.parse(p.date)) / 86_400_000 > 30);
if (stale.length) {
  console.error(`stale prices (> 30 days): ${stale.map(([k, p]) => `${k} ${p.date}`).join(", ")}`);
  process.exit(1);
}
const host = prices.hosts[values.host];
if (!host) {
  console.error(`no price for host ${values.host} in evals/prices.json`);
  process.exit(2);
}

// Shipped requests (deploy/k8s), per overlay.
const K8S_REQUESTS: Record<string, { cpu: number; memory_gib: number }> = {
  base: { cpu: 1, memory_gib: 2560 / 1024 },
  "all-checkpoints": { cpu: 1, memory_gib: 6 },
};
const HPA_TARGET = 0.7;

type Tp = { per_s: number; p95_ms?: number };
const perM = (eurPerHour: number, perS: number) => (eurPerHour / (perS * 3600)) * 1e6;
const r4 = (x: number) => Math.round(x * 1e4) / 1e4;
const best = (tp: Record<string, Tp>): [string, Tp] =>
  Object.entries(tp).reduce((b, e) => (e[1].per_s > b[1].per_s ? e : b));

const dir = resolve(root, "evals/results/perf", values.host);
const rows: Record<string, unknown>[] = [];
for (const f of readdirSync(dir)
  .filter((f) => f.endsWith(".json"))
  .sort()) {
  const r = JSON.parse(readFileSync(resolve(dir, f), "utf8"));
  const { target, checkpoint, precision } = r.meta as { target: string; checkpoint: string; precision: string };
  const [at, tp] = best(r.throughput as Record<string, Tp>);
  if (target.startsWith("k8s-")) {
    const overlay = target.slice(4);
    const req = K8S_REQUESTS[overlay];
    const share = Math.max(req.cpu / host.vcpu, req.memory_gib / host.memory_gib);
    const perPod = (tp.per_s / r.replicas) * HPA_TARGET;
    rows.push({
      target,
      checkpoint,
      precision,
      eur_per_1m: r4(perM(host.eur_per_hour * share, perPod)),
      node_share: r4(share),
      per_pod_per_s_at_70pct: r4(perPod),
      measured_at: at,
      p95_ms_at_load: tp.p95_ms ?? null,
    });
  } else {
    rows.push({
      target,
      checkpoint,
      precision,
      eur_per_1m: r4(perM(host.eur_per_hour, tp.per_s)),
      per_s: tp.per_s,
      measured_at: at,
      p95_ms_at_load: tp.p95_ms ?? null,
    });
  }
}

// LLM reference over the Phase 4 suites.
const manifest = JSON.parse(readFileSync(resolve(root, "evals/manifest.json"), "utf8")) as {
  suites: Record<string, unknown>;
};
let inTok = 0;
let n = 0;
for (const s of Object.keys(manifest.suites)) {
  const path = resolve(root, "evals/datasets/build", `${s}.jsonl`);
  for (const line of readFileSync(path, "utf8").split("\n")) {
    if (!line.trim()) continue;
    const row = JSON.parse(line) as { state: unknown; questions: Record<string, unknown> };
    const k = Object.keys(row.questions).length;
    inTok += Buffer.byteLength(JSON.stringify(row.state) + JSON.stringify(row.questions), "utf8") / 4 + 150;
    n += k;
  }
}
const haiku = prices.llm["claude-haiku-4-5"];
const meanIn = inTok / n;
const meanOut = 10;
const usdPer1m = meanIn * haiku.usd_per_mtok_input + meanOut * haiku.usd_per_mtok_output;
rows.push({
  target: "llm-claude-haiku-4-5",
  checkpoint: null,
  precision: null,
  eur_per_1m: r4(usdPer1m / prices.fx.usd_per_eur),
  usd_per_1m: r4(usdPer1m),
  mean_input_tokens: Math.round(meanIn),
  mean_output_tokens: meanOut,
  note: "computed only: list price x estimated tokens, no API calls, no accuracy claim",
});

const out = resolve(root, "evals/results/cost", `${values.host}.json`);
mkdirSync(dirname(out), { recursive: true });
writeFileSync(
  out,
  `${JSON.stringify({ host: values.host, price: host, prices_checked: new Date().toISOString().slice(0, 10), rows }, null, 2)}\n`,
);
for (const r of rows)
  console.log(`  ${String(r.target).padEnd(22)} ${String(r.checkpoint ?? "-").padEnd(16)} €${r.eur_per_1m} / 1M`);
console.log(`wrote ${out}`);
