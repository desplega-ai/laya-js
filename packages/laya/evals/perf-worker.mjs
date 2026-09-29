// Phase 10 in-process worker: one checkpoint, one runtime (run under `node` or `bun`), against
// the built package (`bun run build` first). Plain JS so Node 22 runs it without a TS loader.
//
//   node packages/laya/evals/perf-worker.mjs --checkpoint multilingual --bundle-dir <dir>
//        --workload evals/out/perf/workload.json --mode full|start [--threads n]
//
// Protocol (upstream research/scripts/bench_latency.py): questions alternate a 3-option choice and
// a noul; 1, 5 and 10 questions per call; 2 warm-up calls, then timed calls.
// - control: upstream's STATE_EN, 10 timed calls (the ±25% check against BENCHMARKS.md);
// - latency: 20 timed calls, each on the next state sampled from the Phase 4 suites;
// - throughput: predictBatch over 96 sampled states x 2 questions at batchSize 1, 8, 32;
// - head share: time spent in the head graph over the whole call, at 1 question.
// `--mode start` loads, answers once and exits: the cold-start probe. Prints one JSON line.
import { readFileSync } from "node:fs";
import { parseArgs } from "node:util";
import { createAgent } from "@desplega/laya";

const { values } = parseArgs({
  options: {
    checkpoint: { type: "string" },
    "bundle-dir": { type: "string" },
    workload: { type: "string" },
    mode: { type: "string", default: "full" },
    threads: { type: "string" },
  },
});
const w = JSON.parse(readFileSync(values.workload, "utf8"));
const qs = (n) => Object.fromEntries(Array.from({ length: n }, (_, i) => [`q${i}`, i % 2 ? w.q_noul : w.q_choice]));
const pct = (xs, p) => {
  const s = [...xs].sort((a, b) => a - b);
  // numpy.percentile's default (linear interpolation), as bench_latency.py uses.
  const k = (s.length - 1) * p;
  const lo = Math.floor(k);
  const hi = Math.ceil(k);
  return s[lo] + (s[hi] - s[lo]) * (k - lo);
};
const summary = (ts) => ({
  p50_ms: round(pct(ts, 0.5)),
  p95_ms: round(pct(ts, 0.95)),
  p99_ms: round(pct(ts, 0.99)),
  mean_ms: round(ts.reduce((a, b) => a + b, 0) / ts.length),
  n: ts.length,
});
const round = (x) => Math.round(x * 100) / 100;

const tLoad = performance.now();
const agent = await createAgent({
  checkpoint: values.checkpoint,
  modelDir: values["bundle-dir"],
  verify: false,
  numThreads: values.threads ? Number(values.threads) : undefined,
});
const loadMs = performance.now() - tLoad;
const raw = agent.raw;
await raw.systemOne(w.states[0], qs(1));
const firstMs = performance.now() - tLoad;
console.error(`READY ${Date.now()}`);
if (values.mode === "start") {
  console.log(JSON.stringify({ load_ms: round(loadMs), first_answer_ms: round(firstMs), peak_rss_mb: rssMb() }));
  process.exit(0);
}

function rssMb() {
  // resourceUsage().maxRSS is in KiB on Linux (Node and Bun).
  return Math.round(process.resourceUsage().maxRSS / 1024);
}

async function timed(fn, warmup, reps) {
  for (let i = 0; i < warmup; i++) await fn(i);
  const ts = [];
  for (let i = 0; i < reps; i++) {
    const t = performance.now();
    await fn(i);
    ts.push(performance.now() - t);
  }
  return ts;
}

const out = { load_ms: round(loadMs), first_answer_ms: round(firstMs), control: {}, latency: {}, throughput: {} };
for (const n of [1, 5, 10]) {
  out.control[`${n}_questions`] = summary(await timed(() => raw.systemOne(w.control_state, qs(n)), 2, 10));
}

// Head share: wrap the provider's two graph calls while the 1-question latency runs.
const provider = raw.provider;
let headMs = 0;
let encMs = 0;
const runHead = provider.runHead.bind(provider);
const runEncoder = provider.runEncoder.bind(provider);
provider.runHead = async (h, b) => {
  const t = performance.now();
  try {
    return await runHead(h, b);
  } finally {
    headMs += performance.now() - t;
  }
};
provider.runEncoder = async (b) => {
  const t = performance.now();
  try {
    return await runEncoder(b);
  } finally {
    encMs += performance.now() - t;
  }
};
for (const n of [1, 5, 10]) {
  let k = 0;
  const call = () => raw.systemOne(w.states[k++ % w.states.length], qs(n));
  await timed(call, 0, 2);
  headMs = 0;
  encMs = 0;
  const ts = await timed(call, 0, 20);
  const total = ts.reduce((a, b) => a + b, 0);
  out.latency[`${n}_questions`] = {
    ...summary(ts),
    head_share: round((headMs * 100) / total) / 100,
    encoder_share: round((encMs * 100) / total) / 100,
  };
}
provider.runHead = runHead;
provider.runEncoder = runEncoder;

const states = w.states.slice(0, 96);
for (const bs of [1, 8, 32]) {
  await raw.predictBatch(states.slice(0, bs), qs(2), { batchSize: bs, sortByLength: true });
  const t = performance.now();
  await raw.predictBatch(states, qs(2), { batchSize: bs, sortByLength: true });
  const s = (performance.now() - t) / 1000;
  out.throughput[`batch_${bs}`] = {
    classifications: states.length * 2,
    seconds: round(s),
    per_s: round((states.length * 2) / s),
  };
}
out.peak_rss_mb = rssMb();
console.log(JSON.stringify(out));
process.exit(0);
