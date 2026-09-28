// Probability drift of candidate bundles vs an fp32 bundle, through the TS product path.
// bun run --filter @desplega/laya build && node tools/export/diagnostics/int8_drift.mjs <ref: fp32 dir or py_reference.py JSON> <candidate dir>...
import { readFileSync } from "node:fs";
import { Agent, emailQuestions, guardQuestions, triageQuestions } from "../../../packages/laya/dist/index.js";

const STATES = JSON.parse(readFileSync(new URL("./states.json", import.meta.url), "utf8"));
const SETS = { triage: triageQuestions(), guard: guardQuestions(), email: emailQuestions() };

function probs(ans) {
  if (ans.type === "noul") return { true: ans.noul, false: 1 - ans.noul };
  return ans.probabilities ?? {};
}
function top1(p) {
  const e = Object.entries(p).sort((a, b) => b[1] - a[1]);
  return { label: e[0]?.[0], margin: (e[0]?.[1] ?? 0) - (e[1]?.[1] ?? 0) };
}
async function runAll(dir) {
  const agent = await Agent.load(dir, { localDir: dir });
  const out = [];
  for (const [si, s] of STATES.entries())
    for (const [qs, q] of Object.entries(SETS)) {
      const r = await agent.systemOne(s, q);
      for (const [k, a] of Object.entries(r.answers))
        out.push({ id: `${si}/${qs}/${k}`, p: probs(a), act: a.action?.act_probability ?? null });
    }
  return out;
}
const [ref, ...cands] = process.argv.slice(2);
const R = ref.endsWith(".json") ? JSON.parse(readFileSync(ref, "utf8")) : await runAll(ref);
for (const c of cands) {
  const C = await runAll(c);
  let max = 0,
    sum = 0,
    n = 0,
    agree = 0,
    agreeM = 0,
    nM = 0,
    worst = null;
  for (let i = 0; i < R.length; i++) {
    for (const [lab, v] of Object.entries(R[i].p)) {
      const d = Math.abs(v - (C[i].p[lab] ?? NaN));
      sum += d;
      n++;
      if (d > max) {
        max = d;
        worst = R[i].id;
      }
    }
    const a = top1(R[i].p),
      b = top1(C[i].p);
    if (a.label === b.label) agree++;
    if (a.margin >= 0.05) {
      nM++;
      if (a.label === b.label) agreeM++;
    }
  }
  console.log(
    JSON.stringify({
      cand: c.split("/").slice(-2).join("/"),
      answers: R.length,
      maxDp: +max.toFixed(4),
      meanDp: +(sum / n).toFixed(5),
      top1: +(agree / R.length).toFixed(4),
      top1Margin05: +(agreeM / nM).toFixed(4),
      worst,
    }),
  );
}
