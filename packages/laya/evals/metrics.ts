// Phase 4 metrics, TS side: a port of evals/python/metrics.py, which uses upstream's
// `laya.evals` (evaluators, `ece` over `laya.common.ece_score`) plus bench_local.py's macro-F1
// and Brier. `test/evals-metrics.test.ts` holds both to 1e-9 on evals/fixtures/metrics-20.json.

export interface EvalAnswer {
  type: "choice" | "noul" | "score";
  choice?: string;
  noul?: number;
  score?: number;
  probabilities?: Record<string, number>;
  answer_confidence?: number | null;
  act_probability?: number | null;
}

export interface EvalRecord {
  qid: string;
  type: string;
  expected: unknown;
  answer: EvalAnswer | null;
  suite: string;
  tags: string[];
  language: string | null;
}

export type Metrics = Record<string, number>;

export interface Sliced {
  overall: Metrics;
  by_suite: Record<string, Metrics>;
  by_tag: Record<string, Metrics>;
  by_language: Record<string, Metrics>;
}

/** The fields every metric and agreement check reads, from a systemOne/predictBatch answer. */
export function normalizeAnswer(ans: Record<string, unknown>): EvalAnswer {
  const type = ans.type as EvalAnswer["type"];
  const out: EvalAnswer = {
    type,
    answer_confidence: (ans.answer_confidence as number | undefined) ?? null,
    act_probability: ((ans.action as { act_probability?: number } | undefined)?.act_probability as number) ?? null,
  };
  if (type === "noul") out.noul = ans.noul as number;
  else {
    out.probabilities = ans.probabilities as Record<string, number>;
    if (type === "choice") out.choice = ans.choice as string;
    else out.score = ans.score as number;
  }
  return out;
}

const mean = (xs: number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length;
const isNum = (x: unknown): x is number => typeof x === "number";

// laya.evals evaluators, same names and None rules.
const EVALUATORS: [string, (a: EvalAnswer, e: unknown) => number | null][] = [
  ["choice_accuracy", (a, e) => (a.type !== "choice" || typeof e !== "string" ? null : a.choice === e ? 1 : 0)],
  [
    "noul_accuracy",
    (a, e) => (a.type !== "noul" || typeof e !== "boolean" ? null : (a.noul ?? 0) >= 0.5 === e ? 1 : 0),
  ],
  ["score_mae", (a, e) => (a.type !== "score" || !isNum(e) ? null : Math.abs((a.score ?? 0) - e))],
  ["mean_confidence", (a) => answerConfidence(a)],
];

function answerConfidence(a: EvalAnswer): number | null {
  if (isNum(a.answer_confidence)) return a.answer_confidence;
  if (a.type === "noul") {
    const p = a.noul ?? 0;
    return Math.max(p, 1 - p);
  }
  const ps = Object.values(a.probabilities ?? {});
  return ps.length ? Math.max(...ps) : null;
}

/** laya.common.ece_score: 15 bins over np.linspace(0, 1, 16), first bin closed on the left. */
export function ece(conf: number[], correct: boolean[], bins = 15): number {
  if (!conf.length) return Number.NaN;
  // np.linspace computes arange(n) * step, then pins the last edge to stop.
  const step = 1 / bins;
  const edges = Array.from({ length: bins + 1 }, (_, i) => (i === bins ? 1 : i * step));
  let e = 0;
  for (let i = 0; i < bins; i++) {
    const lo = edges[i];
    const hi = edges[i + 1];
    const sel: number[] = [];
    conf.forEach((c, j) => {
      if ((i === 0 ? c >= lo : c > lo) && c <= hi) sel.push(j);
    });
    if (sel.length) {
      const confMean = mean(sel.map((j) => conf[j]));
      const corrMean = mean(sel.map((j) => (correct[j] ? 1 : 0)));
      e += (sel.length / conf.length) * Math.abs(confMean - corrMean);
    }
  }
  return e;
}

const probs = (a: EvalAnswer): Record<string, number> =>
  a.type === "noul" ? { false: 1 - (a.noul ?? 0), true: a.noul ?? 0 } : (a.probabilities ?? {});
const label = (a: EvalAnswer): string =>
  a.type === "noul" ? ((a.noul ?? 0) >= 0.5 ? "true" : "false") : String(a.choice);
const gold = (r: EvalRecord): string => (r.type === "noul" ? (r.expected ? "true" : "false") : String(r.expected));
const classified = (r: EvalRecord): boolean =>
  r.type === "choice" ? typeof r.expected === "string" : r.type === "noul" ? typeof r.expected === "boolean" : false;

/** bench_local.py macro_f1: classes are the union of gold and predicted labels. */
export function macroF1(pairs: [string, string][]): number {
  const classes = [...new Set(pairs.flat())].sort();
  return mean(
    classes.map((c) => {
      let tp = 0;
      let fp = 0;
      let fn = 0;
      for (const [g, p] of pairs) {
        if (p === c && g === c) tp++;
        else if (p === c) fp++;
        else if (g === c) fn++;
      }
      return (2 * tp) / Math.max(1, 2 * tp + fp + fn);
    }),
  );
}

export function brier(r: EvalRecord): number {
  const g = gold(r);
  return Object.entries(probs(r.answer as EvalAnswer)).reduce((s, [k, p]) => s + (p - (k === g ? 1 : 0)) ** 2, 0);
}

export function metrics(records: EvalRecord[]): Metrics {
  const out: Metrics = { n: records.length };
  const answered = records.filter((r) => r.answer !== null && r.answer !== undefined);
  out.errors = records.length - answered.length;
  for (const [name, score] of EVALUATORS) {
    const vals = answered.map((r) => score(r.answer as EvalAnswer, r.expected)).filter(isNum);
    if (vals.length) out[name] = mean(vals);
  }
  const cls = answered.filter(classified);
  if (cls.length) {
    const pairs = cls.map((r): [string, string] => [
      `${r.qid}=${gold(r)}`,
      `${r.qid}=${label(r.answer as EvalAnswer)}`,
    ]);
    const corr = pairs.map(([g, p]) => g === p);
    out.accuracy = mean(corr.map((c) => (c ? 1 : 0)));
    out.macro_f1 = macroF1(pairs);
    out.brier = mean(cls.map(brier));
    const conf = cls.map((r, i) => [r.answer?.answer_confidence, corr[i]] as const).filter(([c]) => isNum(c));
    if (conf.length) {
      const v = ece(
        conf.map(([c]) => c as number),
        conf.map(([, k]) => k),
      );
      if (!Number.isNaN(v)) out.ece = v;
    }
  }
  return out;
}

export function sliced(records: EvalRecord[]): Sliced {
  const group = (key: "suite" | "tags" | "language"): Record<string, Metrics> => {
    const g = new Map<string, EvalRecord[]>();
    for (const r of records) {
      const vals = key === "tags" ? r.tags : r[key] == null ? [] : [r[key] as string];
      for (const v of vals) g.set(String(v), [...(g.get(String(v)) ?? []), r]);
    }
    return Object.fromEntries([...g.keys()].sort().map((k) => [k, metrics(g.get(k) ?? [])]));
  };
  return { overall: metrics(records), by_suite: group("suite"), by_tag: group("tags"), by_language: group("language") };
}

export interface Row {
  id: string;
  suite: string;
  state: unknown;
  questions: Record<string, { type: string }>;
  expected: Record<string, unknown>;
  tags: string[];
  language: string | null;
}

/** Join built rows with runtime outputs (`{id: {qid: answer} | null}`) into records. */
export function recordsFrom(rows: Row[], outputs: Record<string, Record<string, EvalAnswer> | null>): EvalRecord[] {
  const recs: EvalRecord[] = [];
  for (const row of rows) {
    if (!(row.id in outputs)) continue;
    const answers = outputs[row.id] ?? {};
    for (const [qid, qd] of Object.entries(row.questions)) {
      if (!(qid in row.expected)) continue;
      recs.push({
        qid,
        type: qd.type,
        expected: row.expected[qid],
        answer: answers[qid] ?? null,
        suite: row.suite,
        tags: row.tags,
        language: row.language,
      });
    }
  }
  return recs;
}
