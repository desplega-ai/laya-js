// Phase 10 gates (plan, Phase 10 "Gates"), over evals/results/perf/<host>/. Run via
// `bun run eval:gate --perf [--host hetzner-ccx23]`.
//
// - memory: k8s pod memory high-water ≤ 80% of the shipped limit (base 3Gi, all-checkpoints 8Gi);
// - cold start: cold-cache start to /health 200 (docker) or to Ready (k8s), p95 < 50% of the
//   startupProbe budget (base 30 × 10 s, all-checkpoints 60 × 10 s);
// - port overhead: TS fp32 p50 ≤ 1.25 × Python ONNX fp32 p50 (upstream ONNXAgent, single-graph
//   fp32: the closest Python path, same engine), per checkpoint, at 1, 5 and 10 questions, for
//   Node and Bun. INT8 is deferred, so the plan's INT8 form of this gate is N/A.
// Report only: the Python torch fp32 multilingual 1-question p50 against upstream's 193 ms
// (BENCHMARKS.md, m7a.xlarge) ±25%, the head share of p50, everything else.
import { existsSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { parseArgs } from "node:util";

const here = dirname(new URL(import.meta.url).pathname);
const root = resolve(here, "../../..");
const GIB = 1024;
const LIMIT_MB: Record<string, number> = { base: 3 * GIB, "all-checkpoints": 8 * GIB };
const PROBE_BUDGET_MS: Record<string, number> = { base: 300_000, "all-checkpoints": 600_000 };
const OVERHEAD = 1.25;
const UPSTREAM_CONTROL_MS = 193;

type Summary = { p50_ms: number; p95_ms: number };
type Result = {
  meta: { target: string; checkpoint: string; precision: string };
  latency: Record<string, Summary>;
  control: Record<string, Summary>;
  start?: { cold: Record<string, Summary> };
  memory_peak_mb?: number;
};

export async function perfGate(): Promise<boolean> {
  const { values } = parseArgs({
    options: { host: { type: "string", default: "hetzner-ccx23" }, perf: { type: "boolean" } },
    strict: false,
  });
  const host = String(values.host);
  const dir = resolve(root, "evals/results/perf", host);
  const res = new Map<string, Result>();
  for (const f of readdirSync(dir).filter((f) => f.endsWith(".json")))
    res.set(f.replace(/\.json$/, ""), JSON.parse(readFileSync(resolve(dir, f), "utf8")));
  const checks: { gate: string; cell: string; value: number; limit: number; pass: boolean }[] = [];
  const add = (gate: string, cell: string, value: number, limit: number) =>
    checks.push({
      gate,
      cell,
      value: Math.round(value * 100) / 100,
      limit: Math.round(limit * 100) / 100,
      pass: value <= limit,
    });

  for (const overlay of ["base", "all-checkpoints"]) {
    const cells = [...res.entries()].filter(([, r]) => r.meta.target === `k8s-${overlay}`);
    for (const [cell, r] of cells) {
      if (r.memory_peak_mb !== undefined)
        add("memory ≤ 80% of limit (MiB)", cell, r.memory_peak_mb, 0.8 * LIMIT_MB[overlay]);
      const cold = r.start?.cold.scale_up_to_ready_ms;
      if (cold)
        add("cold start p95 < 50% of startupProbe (ms)", cell, cold.p95_ms, 0.5 * PROBE_BUDGET_MS[overlay] - 1e-9);
    }
  }
  for (const [cell, r] of res) {
    if (r.meta.target !== "docker") continue;
    const cold = r.start?.cold.to_health_ms;
    if (cold) add("cold start p95 < 50% of startupProbe (ms)", cell, cold.p95_ms, 0.5 * PROBE_BUDGET_MS.base - 1e-9);
  }
  for (const ckpt of ["english", "multilingual", "typed-decisions"]) {
    const py = res.get(`python-onnx-${ckpt}-fp32`);
    for (const rt of ["node", "bun"]) {
      const ts = res.get(`${rt}-${ckpt}-fp32`);
      if (!py || !ts) {
        checks.push({
          gate: "port overhead",
          cell: `${rt}-${ckpt}-fp32`,
          value: Number.NaN,
          limit: OVERHEAD,
          pass: false,
        });
        continue;
      }
      for (const n of ["1_questions", "5_questions", "10_questions"]) {
        add(
          `port overhead ≤ ${OVERHEAD}× python-onnx p50`,
          `${rt}-${ckpt}-fp32 ${n}`,
          ts.latency[n].p50_ms / py.latency[n].p50_ms,
          OVERHEAD,
        );
      }
    }
  }
  const control = res.get("python-torch-multilingual-fp32")?.control["1_questions"]?.p50_ms ?? null;
  const report = {
    host,
    pass: checks.every((c) => c.pass),
    checks,
    control: {
      python_torch_multilingual_1q_p50_ms: control,
      upstream_m7a_xlarge_ms: UPSTREAM_CONTROL_MS,
      ratio: control ? Math.round((control / UPSTREAM_CONTROL_MS) * 1000) / 1000 : null,
      within_25pct: control ? Math.abs(control / UPSTREAM_CONTROL_MS - 1) <= 0.25 : null,
    },
  };
  for (const c of checks)
    console.log(
      `  ${c.pass ? "PASS" : "FAIL"}  ${c.gate.padEnd(44)} ${c.cell.padEnd(40)} ${c.value} (limit ${c.limit})`,
    );
  console.log(
    `  control: python torch multilingual 1q p50 ${control} ms vs upstream ${UPSTREAM_CONTROL_MS} ms (${report.control.ratio}x)`,
  );
  const out = resolve(root, "evals/results", `perf-gate-${host}.json`);
  if (existsSync(dirname(out))) writeFileSync(out, `${JSON.stringify(report, null, 2)}\n`);
  console.log(report.pass ? "PASS" : "FAIL");
  return report.pass;
}
