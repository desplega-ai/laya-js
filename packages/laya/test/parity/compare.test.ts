import { describe, expect, it } from "vitest";
import { compareCheckpoint, type Golden, type Observed } from "./compare.js";

function golden(): Golden {
  return {
    meta: { checkpoint: "multilingual", precision: "fp32" },
    cases: {
      a: {
        input_tokens: 40,
        answers: {
          dept: {
            type: "choice",
            choice: "billing",
            probabilities: { billing: 0.7, technical: 0.2, other: 0.1 },
            confidence: 0.5,
            answer_confidence: 0.7,
            act_probability: 0.9,
            margin: 0.5,
          },
          churn: {
            type: "noul",
            noul: 0.1235,
            confidence: 0.8765,
            answer_confidence: 0.8765,
            act_probability: 0.4,
            margin: 0.753,
          },
        },
      },
      bad: { error: "ValueError: options exceed head_max_len" },
    },
  };
}

function observed(): Record<string, Observed> {
  return {
    a: {
      usage: { input_tokens: 40 },
      answers: {
        dept: {
          type: "choice",
          choice: "billing",
          probabilities: { billing: 0.7, technical: 0.2, other: 0.1 },
          confidence: 0.5,
          answer_confidence: 0.7,
          action: { act_probability: 0.9 },
        },
        churn: {
          type: "noul",
          noul: 0.1235,
          confidence: 0.8765,
          answer_confidence: 0.8765,
          action: { act_probability: 0.4 },
        },
      },
    },
    bad: { error: "options exceed head_max_len=256" },
  };
}

type Ans = Record<string, Record<string, unknown>>;
const answersOf = (o: Record<string, Observed>) => (o.a as { answers: Ans }).answers;

describe("parity compare", () => {
  it("passes identical answers and matching errors", () => {
    const r = compareCheckpoint(golden(), observed());
    expect(r.failures).toEqual([]);
    expect(r.pass).toBe(true);
    expect(r.answers).toBe(2);
    expect(r.top1Agreement).toBe(1);
  });

  it("accepts one unit in the 4th decimal, the rounding both runtimes apply", () => {
    const o = observed();
    answersOf(o).churn.noul = 0.1234;
    expect(compareCheckpoint(golden(), o).pass).toBe(true);
  });

  it("fails a probability drift above 1e-4", () => {
    const o = observed();
    (answersOf(o).dept.probabilities as Record<string, number>).technical = 0.2002;
    const r = compareCheckpoint(golden(), o);
    expect(r.pass).toBe(false);
    expect(r.failures[0].reason).toContain("dept.p[technical]");
  });

  it("fails an act_probability drift above 1e-4", () => {
    const o = observed();
    answersOf(o).churn.action = { act_probability: 0.4003 };
    expect(compareCheckpoint(golden(), o).pass).toBe(false);
  });

  it("fails a top-1 flip", () => {
    const o = observed();
    answersOf(o).dept.probabilities = { billing: 0.2, technical: 0.7, other: 0.1 };
    const r = compareCheckpoint(golden(), o);
    expect(r.failures.some((f) => f.reason.includes("top-1 technical vs Python billing"))).toBe(true);
  });

  it("treats a golden tie at 4-dp precision as agreement", () => {
    const g = golden();
    const ga = (g.cases.a as { answers: Record<string, { probabilities?: Record<string, number>; margin: number }> })
      .answers;
    ga.dept.probabilities = { billing: 0.4, technical: 0.4, other: 0.2 };
    ga.dept.margin = 0;
    const o = observed();
    answersOf(o).dept.probabilities = { billing: 0.4, technical: 0.4001, other: 0.2 };
    expect(compareCheckpoint(g, o).pass).toBe(true);
  });

  it("fails when only one side raises, or token counts differ", () => {
    const o = observed();
    o.bad = observed().a;
    (o.a as { usage: { input_tokens: number } }).usage.input_tokens = 41;
    const reasons = compareCheckpoint(golden(), o).failures.map((f) => f.reason);
    expect(reasons.some((r) => r.startsWith("Python raised"))).toBe(true);
    expect(reasons).toContain("input_tokens 41 vs Python 40");
  });

  it("fails a case the TS side never ran", () => {
    const o = observed();
    delete o.bad;
    expect(compareCheckpoint(golden(), o).failures).toEqual([{ id: "bad", reason: "case not run" }]);
  });

  it("gates predictLong's deciding window and window count", () => {
    const g = golden();
    const ga = g.cases.a as { windows?: number; answers: Record<string, { window?: unknown }> };
    ga.windows = 3;
    ga.answers.dept.window = { count: 3, index: 1, token_end: 700, token_start: 350 };
    const o = observed();
    const oa = o.a as { usage: { windows?: number }; answers: Record<string, Record<string, unknown>> };
    oa.usage.windows = 3;
    oa.answers.dept.window = { index: 1, token_start: 350, token_end: 700, count: 3 };
    expect(compareCheckpoint(g, o).pass).toBe(true);

    oa.usage.windows = 2;
    oa.answers.dept.window = { index: 2, token_start: 700, token_end: 900, count: 3 };
    const reasons = compareCheckpoint(g, o).failures.map((f) => f.reason);
    expect(reasons).toContain("usage.windows 2 vs Python 3");
    expect(reasons.some((r) => r.startsWith("dept: window"))).toBe(true);
  });
});
