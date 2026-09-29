// Phase 4 TS runner: score every suite with @desplega.ai/laya and write the same schema as
// evals/python/reference.py.
//
//   bun run eval:accuracy --checkpoint multilingual --bundle-dir bundles/multilingual/fp32
//                         [--precision fp32] [--suites a,b] [--limit n] [--threads 4] [--tag t]
//
// Rows sharing a question set go through predictBatch (batch 8, sortByLength), the call
// reference.py makes on the Python side. Writes evals/results/<checkpoint>/ts-<precision>.json
// and evals/out/<checkpoint>/ts-<precision>.rows.jsonl (per-row answers, not committed).
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { cpus } from "node:os";
import { dirname, resolve } from "node:path";
import { parseArgs } from "node:util";
import { CHECKPOINTS, type CheckpointName, createAgent } from "../src/index.js";
import { ortVersion } from "./host.js";
import { type EvalAnswer, normalizeAnswer, type Row, recordsFrom, sliced } from "./metrics.js";

const here = dirname(new URL(import.meta.url).pathname);
const root = resolve(here, "../../..");
const { values } = parseArgs({
  options: {
    checkpoint: { type: "string" },
    precision: { type: "string", default: "fp32" },
    "bundle-dir": { type: "string" },
    suites: { type: "string", default: "" },
    limit: { type: "string", default: "0" },
    "batch-size": { type: "string", default: "8" },
    threads: { type: "string" },
    tag: { type: "string", default: "" },
    // Keep the previous run's rows for suites whose SHA-256 is unchanged; score the rest.
    reuse: { type: "boolean", default: false },
  },
});
const ckpt = values.checkpoint as CheckpointName;
if (!ckpt || !(ckpt in CHECKPOINTS)) {
  console.error(`--checkpoint must be one of: ${Object.keys(CHECKPOINTS).join(", ")}`);
  process.exit(2);
}
if (values.precision !== "fp32") {
  console.error(`--precision ${values.precision}: only fp32 is built (INT8 deferred, plan 2026-09-28)`);
  process.exit(2);
}
const limit = Number(values.limit);
const batchSize = Number(values["batch-size"]);
const threads = values.threads ? Number(values.threads) : undefined;

type Manifest = { upstream_sha: string; suites: Record<string, { checkpoints: string[]; sha256: string }> };
const manifest = JSON.parse(readFileSync(resolve(root, "evals/manifest.json"), "utf8")) as Manifest;
const only = values.suites ? values.suites.split(",") : [];
const suites = Object.keys(manifest.suites).filter(
  (s) => manifest.suites[s].checkpoints.includes(ckpt) && (!only.length || only.includes(s)),
);

const t0 = performance.now();
const agent = await createAgent({
  checkpoint: ckpt,
  modelDir: values["bundle-dir"],
  numThreads: threads,
  // A re-exported local bundle has its own digests (dynamo export is not byte-reproducible);
  // the artifact store path verifies against artifacts.ts as usual.
  verify: values["bundle-dir"] ? false : undefined,
});
const loadS = (performance.now() - t0) / 1000;
type Answers = Record<string, EvalAnswer> | null;
const raw = agent.raw as unknown as {
  predictBatch(s: unknown[], q: unknown, o: object): Promise<{ answers: Record<string, Record<string, unknown>> }[]>;
  systemOne(s: unknown, q: unknown): Promise<{ answers: Record<string, Record<string, unknown>> }>;
};
const norm = (a: Record<string, Record<string, unknown>>): Record<string, EvalAnswer> =>
  Object.fromEntries(Object.entries(a).map(([q, v]) => [q, normalizeAnswer(v)]));

async function runGroup(rows: Row[], questions: unknown): Promise<[Answers, string | null][]> {
  try {
    const res = await raw.predictBatch(
      rows.map((r) => r.state),
      questions,
      { batchSize, sortByLength: true },
    );
    return res.map((r) => [norm(r.answers), null]);
  } catch {
    const out: [Answers, string | null][] = [];
    for (const r of rows) {
      try {
        out.push([norm((await raw.systemOne(r.state, questions)).answers), null]);
      } catch (e) {
        out.push([null, (e instanceof Error ? e.message : String(e)).slice(0, 300)]);
      }
    }
    return out;
  }
}

const name = `ts-${values.precision}${values.tag ? `-${values.tag}` : ""}`;
const outDir = resolve(root, "evals/out", ckpt);
const resDir = resolve(root, "evals/results", ckpt);
mkdirSync(outDir, { recursive: true });
mkdirSync(resDir, { recursive: true });
const rowsPath = resolve(outDir, `${name}.rows.jsonl`);
const resPath = resolve(resDir, `${name}.json`);
type RowOut = { id: string; suite: string; answers: Answers; error: string | null };
const oldRows = new Map<string, RowOut[]>();
let oldMeta: { suite_sha256: Record<string, string>; seconds_by_suite: Record<string, number> } | null = null;
if (values.reuse && existsSync(rowsPath) && existsSync(resPath)) {
  oldMeta = JSON.parse(readFileSync(resPath, "utf8")).meta;
  for (const l of readFileSync(rowsPath, "utf8").split("\n")) {
    if (!l.trim()) continue;
    const r = JSON.parse(l) as RowOut;
    oldRows.set(r.suite, [...(oldRows.get(r.suite) ?? []), r]);
  }
}
writeFileSync(rowsPath, "");
const records = [];
const timing: Record<string, number> = {};
const reused: string[] = [];
for (const suite of suites) {
  let rows = readFileSync(resolve(root, "evals/datasets/build", `${suite}.jsonl`), "utf8")
    .split("\n")
    .filter((l) => l.trim())
    .map((l) => JSON.parse(l) as Row);
  if (limit) rows = rows.slice(0, limit);
  const prev = oldRows.get(suite);
  if (prev && oldMeta?.suite_sha256[suite] === manifest.suites[suite].sha256) {
    appendFileSync(rowsPath, prev.map((r) => `${JSON.stringify(r)}\n`).join(""));
    timing[suite] = oldMeta.seconds_by_suite[suite];
    reused.push(suite);
    records.push(...recordsFrom(rows, Object.fromEntries(prev.map((r) => [r.id, r.answers]))));
    console.log(`   ${suite.padEnd(22)} reused`);
    continue;
  }
  const groups = new Map<string, Row[]>();
  for (const r of rows) {
    const key = JSON.stringify(r.questions);
    groups.set(key, [...(groups.get(key) ?? []), r]);
  }
  const ts = performance.now();
  const outputs: Record<string, Answers> = {};
  for (const grp of groups.values()) {
    const res = await runGroup(grp, grp[0].questions);
    // Synchronous appends: a buffered stream never drains while ORT keeps the loop busy.
    appendFileSync(
      rowsPath,
      grp
        .map((r, i) => {
          const [ans, err] = res[i];
          outputs[r.id] = ans;
          return `${JSON.stringify({ id: r.id, suite, answers: ans, error: err })}\n`;
        })
        .join(""),
    );
  }
  timing[suite] = Math.round((performance.now() - ts) / 100) / 10;
  const recs = recordsFrom(rows, outputs);
  records.push(...recs);
  const acc = sliced(recs).overall.accuracy;
  console.log(
    `   ${suite.padEnd(22)} ${String(rows.length).padStart(4)} rows  acc ${acc?.toFixed(4)}  ${timing[suite]}s`,
  );
}

const result = {
  meta: {
    runtime: `ts-${typeof Bun === "undefined" ? "node" : "bun"}`,
    checkpoint: ckpt,
    precision: values.precision,
    source: { repo: CHECKPOINTS[ckpt].source.repo, revision: CHECKPOINTS[ckpt].source.revision },
    bundle_dir: values["bundle-dir"] ?? null,
    upstream_sha: manifest.upstream_sha,
    batch_size: batchSize,
    limit,
    suites,
    suite_sha256: Object.fromEntries(suites.map((s) => [s, manifest.suites[s].sha256])),
    host: {
      cpu: cpus()[0]?.model ?? "unknown",
      cores: cpus().length,
      node: process.versions.node,
      bun: process.versions.bun ?? null,
      onnxruntime: ortVersion(),
      threads: threads ?? null,
    },
    load_s: Math.round(loadS * 10) / 10,
    seconds_by_suite: timing,
    reused_suites: reused,
  },
  metrics: sliced(records),
};
writeFileSync(resPath, `${JSON.stringify(result, null, 2)}\n`);
console.log(`wrote ${resolve(resDir, `${name}.json`)}`);
process.exit(0);
