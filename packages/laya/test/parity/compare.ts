// Python-vs-TS probability parity: compares one checkpoint's TS answers to the committed
// Python goldens (tools/parity/make_goldens.py) and returns a report.
//
// Gate (plan, Phase 3, approved 2026-09-28), fp32 vs Python torch fp32:
//   - max |Δp| ≤ 1e-4 on every probability, on `noul` (P(true)) and on `act_probability`;
//   - top-1 agreement 100%;
//   - the same cases error, and every other case has the same input token count.
// Both runtimes round probabilities to 4 decimals, so 1e-4 is one unit in the last place.
// FLOAT_SLACK only absorbs float noise in the subtraction (0.1235 - 0.1234 > 1e-4 in binary).
// A golden whose top-1/top-2 margin is within 1e-4 is a tie at that precision, so any tied
// label counts as agreement there.

export const FP32_MAX_DP = 1e-4;
const FLOAT_SLACK = 1e-9;

export interface GoldenAnswer {
  type: "choice" | "score" | "noul";
  choice?: string;
  score?: number;
  noul?: number;
  probabilities?: Record<string, number>;
  confidence: number;
  answer_confidence: number;
  act_probability: number;
  margin: number;
  /** predictLong: the window that decided the answer. */
  window?: { index: number; token_start: number; token_end: number; count: number };
}

export type GoldenCase =
  | { error: string }
  | { input_tokens: number; answers: Record<string, GoldenAnswer>; windows?: number | null };

export interface Golden {
  meta: Record<string, unknown> & { checkpoint: string; precision: string };
  cases: Record<string, GoldenCase>;
}

/** What the TS side produced for one case: a SystemOneResult, or the error it threw. */
export type Observed =
  | { error: string }
  | { usage: { input_tokens: number; windows?: number }; answers: Record<string, unknown> };

export interface Failure {
  id: string;
  reason: string;
}

export interface CaseDiff {
  id: string;
  maxDp: number;
  where: string;
}

export interface ParityReport {
  checkpoint: string;
  precision: string;
  cases: number;
  answers: number;
  probabilities: number;
  maxDp: number;
  meanDp: number;
  top1Agreement: number;
  /** Informational: not gated (entropy-based confidence amplifies sub-1e-4 input noise). */
  maxDConfidence: number;
  maxDScore: number;
  worst: CaseDiff[];
  failures: Failure[];
  pass: boolean;
}

function isError(x: GoldenCase | Observed): x is { error: string } {
  return typeof (x as { error?: unknown }).error === "string";
}

function probs(ans: Record<string, unknown>): Record<string, number> {
  if (ans.type === "noul") {
    const t = ans.noul as number;
    return { true: t, false: Math.round((1 - t) * 1e4) / 1e4 };
  }
  return (ans.probabilities as Record<string, number>) ?? {};
}

function top1(p: Record<string, number>): string | undefined {
  let best: string | undefined;
  for (const [k, v] of Object.entries(p)) if (best === undefined || v > p[best]) best = k;
  return best;
}

export function compareCheckpoint(golden: Golden, observed: Record<string, Observed>): ParityReport {
  const failures: Failure[] = [];
  const perCase: CaseDiff[] = [];
  let answers = 0;
  let nP = 0;
  let sumDp = 0;
  let maxDp = 0;
  let agree = 0;
  let maxDConfidence = 0;
  let maxDScore = 0;

  for (const [id, g] of Object.entries(golden.cases)) {
    const o = observed[id];
    if (o === undefined) {
      failures.push({ id, reason: "case not run" });
      continue;
    }
    if (isError(g) || isError(o)) {
      if (!(isError(g) && isError(o))) {
        failures.push({
          id,
          reason: isError(g)
            ? `Python raised (${g.error}), TS did not`
            : `TS raised (${(o as { error: string }).error})`,
        });
      }
      continue;
    }
    if (o.usage.input_tokens !== g.input_tokens) {
      failures.push({ id, reason: `input_tokens ${o.usage.input_tokens} vs Python ${g.input_tokens}` });
    }
    if (g.windows !== undefined && (o.usage.windows ?? null) !== g.windows) {
      failures.push({ id, reason: `usage.windows ${o.usage.windows} vs Python ${g.windows}` });
    }
    let caseMax = 0;
    let caseWhere = "";
    const note = (d: number, where: string) => {
      if (d > caseMax) {
        caseMax = d;
        caseWhere = where;
      }
      if (d > FP32_MAX_DP + FLOAT_SLACK) failures.push({ id, reason: `|Δ| ${d.toFixed(6)} at ${where}` });
    };
    const gIds = Object.keys(g.answers).sort();
    const oIds = Object.keys(o.answers).sort();
    if (gIds.join("\u0000") !== oIds.join("\u0000")) {
      failures.push({ id, reason: `answer ids [${oIds}] vs Python [${gIds}]` });
      continue;
    }
    for (const qid of gIds) {
      const ga = g.answers[qid];
      const oa = o.answers[qid] as Record<string, unknown>;
      answers++;
      const ow = oa.window as GoldenAnswer["window"];
      const sameWindow = (a: GoldenAnswer["window"], b: NonNullable<GoldenAnswer["window"]>) =>
        a?.index === b.index && a.token_start === b.token_start && a.token_end === b.token_end && a.count === b.count;
      if (ga.window && !sameWindow(ow, ga.window)) {
        failures.push({ id, reason: `${qid}: window ${JSON.stringify(ow)} vs Python ${JSON.stringify(ga.window)}` });
      }
      if (oa.type !== ga.type) {
        failures.push({ id, reason: `${qid}: type ${String(oa.type)} vs Python ${ga.type}` });
        continue;
      }
      const gp = probs(ga as unknown as Record<string, unknown>);
      const op = probs(oa);
      for (const [label, gv] of Object.entries(gp)) {
        const ov = op[label];
        if (typeof ov !== "number") {
          failures.push({ id, reason: `${qid}: missing probability ${label}` });
          continue;
        }
        const d = Math.abs(ov - gv);
        nP++;
        sumDp += d;
        if (d > maxDp) maxDp = d;
        note(d, `${qid}.p[${label}]`);
      }
      const act = (oa.action as { act_probability?: number } | undefined)?.act_probability;
      if (typeof act !== "number") failures.push({ id, reason: `${qid}: missing act_probability` });
      else {
        const d = Math.abs(act - ga.act_probability);
        if (d > maxDp) maxDp = d;
        note(d, `${qid}.act_probability`);
      }
      const oTop = top1(op);
      const gTop = top1(gp);
      const tied =
        oTop !== undefined &&
        ga.margin <= FP32_MAX_DP + FLOAT_SLACK &&
        gp[gTop as string] - gp[oTop] <= ga.margin + FLOAT_SLACK;
      if (oTop === gTop || tied) agree++;
      else failures.push({ id, reason: `${qid}: top-1 ${oTop} vs Python ${gTop} (margin ${ga.margin})` });
      maxDConfidence = Math.max(
        maxDConfidence,
        Math.abs((oa.confidence as number) - ga.confidence),
        Math.abs((oa.answer_confidence as number) - ga.answer_confidence),
      );
      if (ga.type === "score") maxDScore = Math.max(maxDScore, Math.abs((oa.score as number) - (ga.score as number)));
    }
    perCase.push({ id, maxDp: caseMax, where: caseWhere });
  }

  perCase.sort((a, b) => b.maxDp - a.maxDp);
  return {
    checkpoint: golden.meta.checkpoint,
    precision: golden.meta.precision,
    cases: Object.keys(golden.cases).length,
    answers,
    probabilities: nP,
    maxDp,
    meanDp: nP ? sumDp / nP : 0,
    top1Agreement: answers ? agree / answers : 0,
    maxDConfidence,
    maxDScore,
    worst: perCase.slice(0, 5),
    failures,
    pass: failures.length === 0,
  };
}

export function formatReport(r: ParityReport): string {
  const lines = [
    `parity ${r.checkpoint} ${r.precision}: ${r.pass ? "PASS" : "FAIL"}`,
    `  cases ${r.cases}, answers ${r.answers}, probabilities ${r.probabilities}`,
    `  max |Δp| ${r.maxDp.toFixed(6)} (gate ${FP32_MAX_DP}), mean |Δp| ${r.meanDp.toExponential(2)}`,
    `  top-1 agreement ${(r.top1Agreement * 100).toFixed(2)}% (gate 100%)`,
    `  max |Δ confidence| ${r.maxDConfidence.toFixed(6)}, max |Δ score| ${r.maxDScore.toFixed(6)} (not gated)`,
    "  worst cases:",
    ...r.worst.map((w) => `    ${w.id}: ${w.maxDp.toFixed(6)} at ${w.where}`),
  ];
  if (r.failures.length) {
    lines.push(`  failures (${r.failures.length}, first 20):`);
    for (const f of r.failures.slice(0, 20)) lines.push(`    ${f.id}: ${f.reason}`);
  }
  return lines.join("\n");
}
