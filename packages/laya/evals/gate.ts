// Phase 4 gate: TS fp32 vs Python fp32, per checkpoint (plan, Phase 4 "Gates"; INT8 deferred).
//
//   bun run eval:gate [--checkpoints a,b] [--shuffle-labels <suite>] [--out <path.json>]
//
// - every metric of every slice (overall, suite, tag, language) equal within 0.001;
// - per row, the Phase 3 fp32 parity bar: max |Δp| ≤ 1e-4 on every probability, `noul` and
//   `act_probability`, top-1 agreement 100% (a Python top-1/top-2 margin ≤ 1e-4 is a tie);
// - the same rows error on both sides.
// --shuffle-labels permutes that suite's expected labels (seeded) before the TS metrics are
// recomputed from the per-row answers: the negative control, which must fail.
// `--perf` runs the Phase 10 gates instead (see perf-gate.ts).
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { parseArgs } from "node:util";
import { type EvalAnswer, type Metrics, type Row, recordsFrom, type Sliced, sliced } from "./metrics.js";

export const METRIC_TOL = 0.001;
export const FP32_MAX_DP = 1e-4;
const FLOAT_SLACK = 1e-9;

const here = dirname(new URL(import.meta.url).pathname);
const root = resolve(here, "../../..");

type RowOut = { id: string; suite: string; answers: Record<string, EvalAnswer> | null; error: string | null };
const readJsonl = <T>(p: string): T[] =>
  readFileSync(p, "utf8")
    .split("\n")
    .filter((l) => l.trim())
    .map((l) => JSON.parse(l) as T);

const probsOf = (a: EvalAnswer): Record<string, number> =>
  a.type === "noul" ? { true: a.noul ?? 0 } : (a.probabilities ?? {});
const top = (p: Record<string, number>): string => Object.entries(p).reduce((b, e) => (e[1] > b[1] ? e : b))[0];
const topOf = (a: EvalAnswer): string =>
  a.type === "noul" ? ((a.noul ?? 0) >= 0.5 ? "true" : "false") : top(a.probabilities ?? {});
const marginOf = (a: EvalAnswer): number => {
  if (a.type === "noul") return Math.abs(2 * (a.noul ?? 0) - 1);
  const v = Object.values(a.probabilities ?? {}).sort((x, y) => y - x);
  return v[0] - (v[1] ?? 0);
};

export interface Agreement {
  rows: number;
  answers: number;
  maxDp: number;
  meanDp: number;
  top1: number;
  errorsPy: number;
  errorsTs: number;
  failures: string[];
}

export function agreement(py: RowOut[], ts: RowOut[]): Agreement {
  const tsById = new Map(ts.map((r) => [r.id, r]));
  const failures: string[] = [];
  let answers = 0;
  let agree = 0;
  let maxDp = 0;
  let sumDp = 0;
  let nDp = 0;
  for (const p of py) {
    const t = tsById.get(p.id);
    if (!t) {
      failures.push(`${p.id}: missing on the TS side`);
      continue;
    }
    if (!p.answers || !t.answers) {
      if (!p.answers !== !t.answers) failures.push(`${p.id}: errors on one side only`);
      continue;
    }
    for (const [qid, pa] of Object.entries(p.answers)) {
      const ta = t.answers[qid];
      if (!ta) {
        failures.push(`${p.id}/${qid}: no TS answer`);
        continue;
      }
      answers++;
      const pp = probsOf(pa);
      const tp = probsOf(ta);
      const diffs = Object.keys(pp).map((k) => Math.abs(pp[k] - (tp[k] ?? Number.POSITIVE_INFINITY)));
      if (isNum(pa.act_probability) && isNum(ta.act_probability))
        diffs.push(Math.abs(pa.act_probability - ta.act_probability));
      const d = Math.max(...diffs);
      for (const x of diffs) {
        sumDp += x;
        nDp++;
      }
      maxDp = Math.max(maxDp, d);
      if (d > FP32_MAX_DP + FLOAT_SLACK) failures.push(`${p.id}/${qid}: |Δp| ${d.toFixed(6)}`);
      const pt = topOf(pa);
      const tt = topOf(ta);
      const m = marginOf(pa);
      const full = pa.type === "noul" ? { true: pa.noul ?? 0, false: 1 - (pa.noul ?? 0) } : (pa.probabilities ?? {});
      const tied = m <= FP32_MAX_DP + FLOAT_SLACK && (full[pt] ?? 0) - (full[tt] ?? 0) <= m + FLOAT_SLACK;
      if (pt === tt || tied) agree++;
      else failures.push(`${p.id}/${qid}: top-1 ${tt} vs Python ${pt} (margin ${m.toFixed(4)})`);
    }
  }
  return {
    rows: py.length,
    answers,
    maxDp,
    meanDp: nDp ? sumDp / nDp : 0,
    top1: answers ? agree / answers : 1,
    errorsPy: py.filter((r) => !r.answers).length,
    errorsTs: ts.filter((r) => !r.answers).length,
    failures,
  };
}

function isNum(x: unknown): x is number {
  return typeof x === "number";
}

/** Every metric present on both sides, in every slice, within `tol`. Returns the misses. */
export function metricDiffs(py: Sliced, ts: Sliced, tol = METRIC_TOL): string[] {
  const out: string[] = [];
  const cmp = (where: string, a: Metrics | undefined, b: Metrics | undefined) => {
    if (!a || !b) {
      out.push(`${where}: slice missing on one side`);
      return;
    }
    for (const k of Object.keys(a)) {
      if (!(k in b)) out.push(`${where}.${k}: missing on the TS side`);
      else if (Math.abs(a[k] - b[k]) > tol + FLOAT_SLACK)
        out.push(`${where}.${k}: Python ${a[k].toFixed(4)} vs TS ${b[k].toFixed(4)}`);
    }
  };
  cmp("overall", py.overall, ts.overall);
  for (const key of ["by_suite", "by_tag", "by_language"] as const)
    for (const s of new Set([...Object.keys(py[key]), ...Object.keys(ts[key])]))
      cmp(`${key}.${s}`, py[key][s], ts[key][s]);
  return out;
}

/** Seeded Fisher-Yates over one suite's expected values (mulberry32, seed 13). */
function shuffleLabels(rows: Row[], suite: string): Row[] {
  let s = 13;
  const rnd = () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const idx = rows.map((r, i) => (r.suite === suite ? i : -1)).filter((i) => i >= 0);
  const exp = idx.map((i) => rows[i].expected);
  for (let i = exp.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [exp[i], exp[j]] = [exp[j], exp[i]];
  }
  const out = rows.slice();
  idx.forEach((i, k) => {
    out[i] = { ...rows[i], expected: exp[k] };
  });
  return out;
}

async function main() {
  const { values } = parseArgs({
    options: {
      checkpoints: { type: "string", default: "english,multilingual,typed-decisions" },
      "shuffle-labels": { type: "string" },
      out: { type: "string" },
      perf: { type: "boolean", default: false },
      host: { type: "string" },
    },
  });
  if (values.perf) {
    const { perfGate } = await import("./perf-gate.js");
    process.exit((await perfGate()) ? 0 : 1);
  }
  const summary: Record<string, unknown> = {};
  let pass = true;
  for (const ckpt of values.checkpoints.split(",")) {
    const dir = resolve(root, "evals/results", ckpt);
    const pyRes = JSON.parse(readFileSync(resolve(dir, "python-fp32.json"), "utf8")) as { metrics: Sliced };
    let tsMetrics = (JSON.parse(readFileSync(resolve(dir, "ts-fp32.json"), "utf8")) as { metrics: Sliced }).metrics;
    const pyRows = readJsonl<RowOut>(resolve(root, "evals/out", ckpt, "python-fp32.rows.jsonl"));
    const tsRows = readJsonl<RowOut>(resolve(root, "evals/out", ckpt, "ts-fp32.rows.jsonl"));
    const shuffled = values["shuffle-labels"];
    if (shuffled) {
      const suites = [...new Set(tsRows.map((r) => r.suite))];
      let rows = suites.flatMap((s) => readJsonl<Row>(resolve(root, "evals/datasets/build", `${s}.jsonl`)));
      rows = shuffleLabels(rows, shuffled);
      tsMetrics = sliced(recordsFrom(rows, Object.fromEntries(tsRows.map((r) => [r.id, r.answers]))));
    }
    const diffs = metricDiffs(pyRes.metrics, tsMetrics);
    const agr = agreement(pyRows, tsRows);
    const ok = !diffs.length && !agr.failures.length;
    pass &&= ok;
    console.log(`\n${ckpt}: ${ok ? "PASS" : "FAIL"}`);
    console.log("  suite                      n   py acc   ts acc   py ECE   ts ECE   |Δacc|");
    for (const [s, m] of Object.entries(pyRes.metrics.by_suite)) {
      const t = tsMetrics.by_suite[s] ?? {};
      const f = (x: number | undefined) => (x === undefined ? "    -  " : x.toFixed(4).padStart(7));
      console.log(
        `  ${s.padEnd(22)} ${String(m.n).padStart(5)}  ${f(m.accuracy)}  ${f(t.accuracy)}  ${f(m.ece)}  ${f(t.ece)}  ${f(Math.abs((m.accuracy ?? 0) - (t.accuracy ?? 0)))}`,
      );
    }
    console.log(
      `  rows ${agr.rows}, answers ${agr.answers}, max |Δp| ${agr.maxDp.toFixed(6)}, mean |Δp| ${agr.meanDp.toExponential(2)}, top-1 ${(agr.top1 * 100).toFixed(2)}%, errors py ${agr.errorsPy} / ts ${agr.errorsTs}`,
    );
    for (const d of [...diffs, ...agr.failures].slice(0, 10)) console.log(`  - ${d}`);
    if (diffs.length + agr.failures.length > 10) console.log(`  ... ${diffs.length + agr.failures.length - 10} more`);
    summary[ckpt] = {
      pass: ok,
      metric_misses: diffs.length,
      agreement: { ...agr, failures: agr.failures.length, first_failures: agr.failures.slice(0, 20) },
      metric_first_misses: diffs.slice(0, 20),
    };
  }
  if (values.out)
    writeFileSync(
      resolve(values.out),
      `${JSON.stringify({ shuffled: values["shuffle-labels"] ?? null, pass, checkpoints: summary }, null, 2)}\n`,
    );
  console.log(`\n${pass ? "PASS" : "FAIL"}`);
  process.exit(pass ? 0 : 1);
}

if (import.meta.main ?? process.argv[1] === new URL(import.meta.url).pathname) await main();
