// Python-vs-TS parity gate for one checkpoint. Loads the fp32 bundle, runs every fixture in
// tools/parity/fixtures/cases.jsonl through Agent.systemOne, every batch.jsonl case through
// Agent.predictBatch and every long.jsonl case through Agent.predictLong, compares to the committed
// Python goldens (<checkpoint>.json, .batch.json, .long.json) and exits non-zero on any failure.
//
//   bun run parity -- --checkpoint multilingual [--precision fp32] [--bundle-dir <dir>]
//                     [--perturb 0.01] [--report <path.json>]
//
// Without --bundle-dir the bundle comes from the private artifact store (needs HF_TOKEN).
// --perturb adds the value to the first logit of every head call: the negative control.
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { parseArgs } from "node:util";
import { ARTIFACT_REPO, ARTIFACT_REVISION, ARTIFACTS, type CheckpointName } from "../../src/artifacts.js";
import { Agent } from "../../src/raw.js";
import { compareCheckpoint, formatReport, type Golden, type Observed } from "./compare.js";

const here = dirname(new URL(import.meta.url).pathname);
const root = resolve(here, "../../../..");

const { values } = parseArgs({
  options: {
    checkpoint: { type: "string" },
    precision: { type: "string", default: "fp32" },
    "bundle-dir": { type: "string" },
    perturb: { type: "string" },
    report: { type: "string" },
  },
});

const ckpt = values.checkpoint as CheckpointName | undefined;
if (!ckpt || !(ckpt in ARTIFACTS)) {
  console.error(`--checkpoint must be one of: ${Object.keys(ARTIFACTS).join(", ")}`);
  process.exit(2);
}
if (values.precision !== "fp32") {
  console.error(`--precision ${values.precision}: only fp32 is built (INT8 deferred, plan 2026-09-28)`);
  process.exit(2);
}
const perturb = values.perturb === undefined ? 0 : Number(values.perturb);
if (!Number.isFinite(perturb)) {
  console.error(`--perturb must be a number, got ${values.perturb}`);
  process.exit(2);
}

type Opts = {
  lang?: string;
  batch_size?: number | null;
  sort_by_length?: boolean;
  window?: number | null;
  stride?: number | null;
};
const readJsonl = <T>(name: string): T[] =>
  readFileSync(resolve(root, "tools/parity/fixtures", name), "utf8")
    .split("\n")
    .filter((l) => l.trim())
    .map((l) => JSON.parse(l) as T);
const readGolden = (suffix: string): Golden =>
  JSON.parse(readFileSync(resolve(here, "golden", `${ckpt}${suffix}.json`), "utf8")) as Golden;

const single = readGolden("");
// Batch results are keyed `<case id>#<index>`, as make_goldens.py --kind batch writes them.
const golden: Golden = {
  meta: single.meta,
  cases: { ...single.cases, ...readGolden(".batch").cases, ...readGolden(".long").cases },
};
const cases = readJsonl<{ id: string; state: unknown; questions: Record<string, never>; opts: Opts }>("cases.jsonl");
const longCases = readJsonl<{ id: string; state: unknown; questions: Record<string, never>; opts: Opts }>("long.jsonl");
const batchCases = readJsonl<{ id: string; states: unknown[]; questions: Record<string, never>; opts: Opts }>(
  "batch.jsonl",
);

const t0 = performance.now();
const bundleDir = values["bundle-dir"];
const artifact = ARTIFACTS[ckpt].fp32;
if (!bundleDir && !process.env.HF_TOKEN) {
  console.error("HF_TOKEN is empty; the artifact store is private (or pass --bundle-dir)");
  process.exit(2);
}
const agent = bundleDir
  ? await Agent.load(bundleDir, { localDir: bundleDir, expectedSha256: artifact.sha256 })
  : await Agent.load(ARTIFACT_REPO, {
      subfolder: artifact.subfolder,
      revision: ARTIFACT_REVISION,
      expectedSha256: artifact.sha256,
    });
const loadMs = performance.now() - t0;

if (perturb !== 0) {
  const provider = (
    agent as unknown as { provider: { runHead: (h: unknown, b: unknown) => Promise<{ logits: number[][] }> } }
  ).provider;
  const runHead = provider.runHead.bind(provider);
  provider.runHead = async (h, b) => {
    const out = await runHead(h, b);
    out.logits[0][0] += perturb;
    return out;
  };
}

const observed: Record<string, Observed> = {};
for (const c of cases) {
  try {
    observed[c.id] = (await agent.systemOne(c.state, c.questions, { lang: c.opts?.lang ?? null })) as Observed;
  } catch (e) {
    observed[c.id] = { error: e instanceof Error ? e.message : String(e) };
  }
}
for (const c of batchCases) {
  try {
    const rs = await agent.predictBatch(c.states, c.questions, {
      lang: c.opts?.lang ?? null,
      batchSize: c.opts?.batch_size ?? null,
      sortByLength: c.opts?.sort_by_length ?? false,
    });
    rs.forEach((r, i) => {
      observed[`${c.id}#${i}`] = r as Observed;
    });
  } catch (e) {
    observed[c.id] = { error: e instanceof Error ? e.message : String(e) };
  }
}
for (const c of longCases) {
  try {
    observed[c.id] = (await agent.predictLong(c.state, c.questions, {
      lang: c.opts?.lang ?? null,
      window: c.opts?.window ?? null,
      stride: c.opts?.stride ?? null,
      batchSize: c.opts?.batch_size ?? null,
    })) as Observed;
  } catch (e) {
    observed[c.id] = { error: e instanceof Error ? e.message : String(e) };
  }
}
const runMs = performance.now() - t0 - loadMs;

const report = compareCheckpoint(golden, observed);
console.log(formatReport(report));
console.log(
  `  load ${(loadMs / 1000).toFixed(1)}s, run ${(runMs / 1000).toFixed(1)}s${perturb ? `, perturb ${perturb}` : ""}`,
);
if (values.report) {
  mkdirSync(dirname(resolve(values.report)), { recursive: true });
  writeFileSync(resolve(values.report), `${JSON.stringify({ ...report, perturb, golden: golden.meta }, null, 2)}\n`);
}
process.exit(report.pass ? 0 : 1);
