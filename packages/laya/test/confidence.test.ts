// Port of the min_confidence cases in upstream tests/test_confidence.py, plus decide/Router abstention.
import { describe, expect, it } from "vitest";
import { Agent } from "../src/agent.js";
import { checkMinConfidence, flagLowConfidence } from "../src/confidence.js";
import { Router } from "../src/router.js";

describe("checkMinConfidence", () => {
  it.each([0.0, 0.5, 1.0, 0, 1, 0.85])("accepts %s", (v) => {
    expect(checkMinConfidence(v)).toBe(v);
  });

  it.each([
    true,
    false,
    -0.01,
    1.01,
    -1.0,
    2.0,
    Number.NaN,
    Number.POSITIVE_INFINITY,
    Number.NEGATIVE_INFINITY,
    "0.5",
    null,
    undefined,
    [0.5],
  ])("rejects %s", (v) => {
    expect(() => checkMinConfidence(v)).toThrow(/minConfidence must be a number in \[0.0, 1.0\]/);
  });
});

describe("flagLowConfidence", () => {
  const sample = () => [
    {
      answers: {
        q_high: { type: "choice", choice: "a", answer_confidence: 0.92, confidence: 0.8 },
        q_low: { type: "score", score: 1, answer_confidence: 0.45, confidence: 0.4 },
        q_fallback: { type: "choice", choice: "b", confidence: 0.3 },
        q_exact: { type: "choice", choice: "c", answer_confidence: 0.7 },
      } as Record<string, Record<string, unknown>>,
    },
  ];

  it("is a no-op at 0", () => {
    const res = sample();
    flagLowConfidence(res, 0);
    expect(Object.values(res[0].answers).some((a) => "low_confidence" in a)).toBe(false);
  });

  it("flags below the threshold (falling back to confidence) and keeps the raw answer", () => {
    const res = sample();
    flagLowConfidence(res, 0.7);
    const a = res[0].answers;
    expect(a.q_low.low_confidence).toBe(true);
    expect(a.q_fallback.low_confidence).toBe(true);
    expect("low_confidence" in a.q_high).toBe(false);
    expect("low_confidence" in a.q_exact).toBe(false);
    expect(a.q_low.answer_confidence).toBe(0.45);
    expect(a.q_high.choice).toBe("a");
  });

  it("skips results and answers that are not objects", () => {
    const res: unknown[] = [null, "x", { answers: null }, { answers: { q: null, r: 3, s: { confidence: "0.1" } } }];
    expect(() => flagLowConfidence(res, 0.9)).not.toThrow();
    expect(JSON.stringify(res)).not.toContain("low_confidence");
  });
});

function fakeAgent() {
  const provider = {
    async runEncoder() {
      return { lastHidden: [] };
    },
    async runHead(_h: unknown, batch: { inputIds: number[][] }) {
      // Two rows: a confident choice (high) and a coin-flip noul (low).
      return {
        logits: batch.inputIds.map((_, r) => (r % 2 === 0 ? [4, 0, 0] : [0, 0])),
        act: batch.inputIds.map(() => [1, 0]),
      };
    },
  };
  return new Agent({ provider } as never);
}

const QS = {
  sure: { type: "choice", instructions: "?", criteria: ["a", "b", "c"] },
  coin: { type: "noul", instructions: "?" },
};

describe("abstention through decide and Router", () => {
  it("decide projects an abstained field to null and keeps the raw answer in the details", async () => {
    const agent = fakeAgent();
    const schema = {
      type: "object",
      properties: { sure: { type: "string", enum: ["a", "b", "c"] }, coin: { type: "boolean" } },
    };
    const plain = await agent.decide("x", schema);
    expect(plain).toEqual({ sure: "a", coin: true });
    const gated = await agent.decide("x", schema, { minConfidence: 0.6, returnDetails: true });
    expect(gated.values).toEqual({ sure: "a", coin: null });
    expect(await agent.decide("x", schema, { minConfidence: 0 })).toEqual(plain);
    expect((gated.answers.coin as Record<string, unknown>).low_confidence).toBe(true);
    expect((gated.answers.coin as Record<string, unknown>).noul).toBe(0.5);
    await expect(agent.decide("x", schema, { minConfidence: 2 })).rejects.toThrow(/minConfidence/);
  });

  it("Router.predict flags before its end hooks", async () => {
    const agent = fakeAgent();
    const seen: unknown[] = [];
    const router = new Router({ loader: () => agent });
    const r = await router.predict("hello there, a plain English sentence", QS as never, {
      minConfidence: 0.6,
      onPredictEnd: (ctx) => void seen.push(JSON.parse(JSON.stringify(ctx.results))),
    });
    expect((r.answers.coin as { low_confidence?: true }).low_confidence).toBe(true);
    expect("low_confidence" in r.answers.sure).toBe(false);
    expect(seen).toEqual([[JSON.parse(JSON.stringify(r))]]);
  });
});
