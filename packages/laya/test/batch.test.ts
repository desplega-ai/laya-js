// Port of upstream tests/test_batch.py (plus the plan's Phase 6 criteria) on a fake provider.
import { describe, expect, it } from "vitest";
import { Agent, defaultTokenizer, type QuestionDef, type SystemOneResult } from "../src/agent.js";
import type { PredictContext } from "../src/hooks.js";
import type { Batch } from "../src/providers.js";

const QUESTIONS: Record<string, QuestionDef> = {
  dept: { type: "choice", instructions: "Which team?", criteria: { billing: "money", support: "help", sales: "buy" } },
  level: { type: "score", instructions: "How urgent?", criteria: ["low", "mid", "high"] },
  urgent: { type: "noul", instructions: "Urgent?" },
};

/** Row logits derived from the row's own tokens, so a row decoded for the wrong state changes the answer. */
function makeAgent(opts: Record<string, unknown> = {}) {
  const calls: { rows: number; width: number; lengths: number[] }[] = [];
  const encoded: string[] = [];
  const provider = {
    async runEncoder(batch: Batch) {
      calls.push({
        rows: batch.inputIds.length,
        width: batch.inputIds[0]?.length ?? 0,
        lengths: batch.attentionMask.map((m) => m.reduce((a, b) => a + b, 0)),
      });
      return { lastHidden: [] as number[][][] };
    },
    async runHead(_h: unknown, batch: Batch) {
      const logits = batch.inputIds.map((ids, r) => {
        const s = ids.reduce((a, b, i) => a + b * (i + 1), 0) + batch.attentionMask[r].reduce((a, b) => a + b, 0);
        return batch.markerMask[r].map((_, k) => ((s * (k + 3)) % 17) / 5 - 1.5);
      });
      const act = batch.attentionMask.map((m) => [(m.reduce((a, b) => a + b, 0) % 5) / 3, 0.5]);
      return { logits, act };
    },
  };
  const base = defaultTokenizer();
  const tok = {
    ...base,
    encode(text: string): number[] {
      if (text.startsWith("state")) encoded.push(text);
      return base.encode(text);
    },
  };
  const agent = new Agent({ provider, tok, ...opts } as never);
  return { agent, calls, encoded };
}

function state(i: number, words: number): string {
  return `state${i} ${Array.from({ length: words }, (_, w) => `w${(i * 7 + w) % 13}x${w % 3}`).join(" ")}`;
}

const LENGTHS = [40, 5, 25, 8, 60, 3, 17, 33, 12, 50, 2, 29];
const STATES = LENGTHS.map((n, i) => state(i, n));

function expectClose(got: unknown, want: unknown, path = "$"): void {
  if (typeof want === "number") {
    expect(typeof got, path).toBe("number");
    expect(Math.abs((got as number) - want), path).toBeLessThanOrEqual(1e-6);
    return;
  }
  if (want && typeof want === "object") {
    expect(got && typeof got === "object", path).toBe(true);
    expect(Object.keys(got as object).sort(), path).toEqual(Object.keys(want).sort());
    for (const k of Object.keys(want)) {
      expectClose((got as Record<string, unknown>)[k], (want as Record<string, unknown>)[k], `${path}.${k}`);
    }
    return;
  }
  expect(got, path).toEqual(want);
}

async function sequential(states: string[]): Promise<SystemOneResult[]> {
  const { agent } = makeAgent();
  const out: SystemOneResult[] = [];
  for (const st of states) out.push(await agent.predict(st, QUESTIONS));
  return out;
}

describe("predictBatch equals predict", () => {
  it("matches a sequential predict loop within 1e-6", async () => {
    const want = await sequential(STATES);
    const { agent, calls } = makeAgent();
    const got = await agent.predictBatch(STATES, QUESTIONS);
    expectClose(got, want);
    expect(calls).toHaveLength(1);
    expect(calls[0].rows).toBe(STATES.length * 3);
  });

  it("gives identical results for batchSize 1, 3 and > n", async () => {
    const want = await sequential(STATES);
    for (const batchSize of [1, 3, STATES.length + 5]) {
      const { agent } = makeAgent();
      expectClose(await agent.predictBatch(STATES, QUESTIONS, { batchSize }), want);
    }
  });

  it("runs the encoder ceil(n / batchSize) times", async () => {
    for (const batchSize of [1, 2, 3, 5, 12, 100]) {
      const { agent, calls } = makeAgent();
      await agent.predictBatch(STATES, QUESTIONS, { batchSize });
      expect(calls.length, `batchSize=${batchSize}`).toBe(
        Math.ceil(STATES.length / Math.min(batchSize, STATES.length)),
      );
    }
  });

  it("chunks 5 states as 2+2+1 at batchSize 2", async () => {
    const { agent, calls } = makeAgent();
    await agent.predictBatch(STATES.slice(0, 5), QUESTIONS, { batchSize: 2 });
    expect(calls.map((c) => c.rows)).toEqual([6, 6, 3]);
  });

  it("counts only each state's own tokens, not padding", async () => {
    const want = await sequential(STATES);
    const { agent } = makeAgent();
    const got = await agent.predictBatch(STATES, QUESTIONS);
    expect(got.map((r) => r.usage)).toEqual(want.map((r) => r.usage));
  });

  it("returns [] for no states and empty answers for no questions, without encoding", async () => {
    const { agent, calls, encoded } = makeAgent();
    expect(await agent.predictBatch([], QUESTIONS, { sortByLength: true })).toEqual([]);
    const empty = await agent.predictBatch(STATES.slice(0, 2), {});
    expect(empty).toEqual([
      { model: "laya-rl-agent", answers: {}, usage: { input_tokens: 0, output_tokens: 0 } },
      { model: "laya-rl-agent", answers: {}, usage: { input_tokens: 0, output_tokens: 0 } },
    ]);
    expect(calls).toEqual([]);
    expect(encoded).toEqual([]);
  });

  it("rejects a bare string, an object and a null state", async () => {
    const { agent } = makeAgent();
    await expect(agent.predictBatch("just a string" as never, QUESTIONS)).rejects.toThrow(TypeError);
    await expect(agent.predictBatch({ body: "x" } as never, QUESTIONS)).rejects.toThrow(TypeError);
    await expect(agent.predictBatch(["a", null], QUESTIONS)).rejects.toThrow(/must not be null/);
    await expect(agent.predictBatch(["a"], ["q"] as never)).rejects.toThrow(/questions must be an object/);
  });
});

describe("sortByLength", () => {
  it("keeps input order for shuffled lengths and pads less", async () => {
    const many = Array.from({ length: 37 }, (_, i) => state(i, [40, 5, 25, 8][i % 4]));
    const plain = makeAgent();
    const grouped = makeAgent();
    const want = await plain.agent.predictBatch(many, QUESTIONS, { batchSize: 2 });
    const got = await grouped.agent.predictBatch(many, QUESTIONS, { batchSize: 2, sortByLength: true });
    expectClose(got, want);
    expectClose(got, await sequential(many));
    const slots = (c: typeof plain.calls) => c.reduce((a, x) => a + x.rows * x.width, 0);
    expect(slots(grouped.calls)).toBeLessThan(slots(plain.calls));
    expect(grouped.calls.length).toBe(plain.calls.length);
    expect(grouped.calls.at(-1)?.rows).toBe(3);
    // Each state is encoded once, in input order.
    expect(grouped.encoded).toEqual(many);
  });

  it("buffers at most eight batches before the first forward pass", async () => {
    const many = Array.from({ length: 37 }, (_, i) => state(i, [40, 5, 25, 8][i % 4]));
    const { agent, encoded } = makeAgent();
    let atFirst = -1;
    const provider = agent.provider;
    const run = provider.runEncoder.bind(provider);
    provider.runEncoder = async (b) => {
      if (atFirst < 0) atFirst = encoded.length;
      return run(b);
    };
    await agent.predictBatch(many, QUESTIONS, { batchSize: 2, sortByLength: true });
    expect(atFirst).toBe(16);
  });

  it("has no effect without a batchSize strictly between 1 and n", async () => {
    for (const batchSize of [null, 0, -1, 1, 100]) {
      const plain = makeAgent();
      const grouped = makeAgent();
      const want = await plain.agent.predictBatch(STATES, QUESTIONS, { batchSize });
      const got = await grouped.agent.predictBatch(STATES, QUESTIONS, { batchSize, sortByLength: true });
      expect(got).toEqual(want);
      expect(grouped.calls).toEqual(plain.calls);
    }
  });

  it("keeps row order on equal-length ties", async () => {
    const equal = Array.from({ length: 20 }, (_, i) => `state${i} a b c d e f g h`);
    const plain = makeAgent();
    const grouped = makeAgent();
    await plain.agent.predictBatch(equal, QUESTIONS, { batchSize: 3 });
    await grouped.agent.predictBatch(equal, QUESTIONS, { batchSize: 3, sortByLength: true });
    expect(grouped.calls).toEqual(plain.calls);
    expect(grouped.encoded).toEqual(plain.encoded);
  });

  it("applies hook rewrites and budgets before sorting", async () => {
    const rewritten = await makeAgent().agent.predictBatch(["unused"], QUESTIONS, {
      batchSize: 2,
      sortByLength: true,
      onPredictStart: (ctx) => {
        ctx.states = STATES;
        ctx.maxLen = 40;
      },
    });
    const reference = await makeAgent().agent.predictBatch(STATES, QUESTIONS, { batchSize: 2, maxLen: 40 });
    expectClose(rewritten, reference);
  });
});

describe("predictBatch hooks", () => {
  it("fires start and end once per call, with every state and the restored results", async () => {
    const events: string[] = [];
    let endResults: unknown = null;
    let endUsage: unknown = null;
    const { agent } = makeAgent();
    const got = await agent.predictBatch(STATES, QUESTIONS, {
      batchSize: 2,
      sortByLength: true,
      hooks: {
        onPredictStart(ctx: PredictContext) {
          events.push(`start:${ctx.states.length}`);
        },
        onPredictEnd(ctx: PredictContext) {
          events.push("end");
          endResults = ctx.results;
          endUsage = ctx.usage;
        },
      },
    });
    expect(events).toEqual([`start:${STATES.length}`, "end"]);
    expect(endResults).toBe(got);
    expect(endUsage).toEqual({
      input_tokens: got.reduce((a, r) => a + r.usage.input_tokens, 0),
      output_tokens: 0,
    });
  });

  it("a cache skip does not encode", async () => {
    const { agent, encoded, calls } = makeAgent();
    const cached = [{ model: "cache", answers: {}, usage: { input_tokens: 0, output_tokens: 0 } }];
    const got = await agent.predictBatch(STATES, QUESTIONS, { onPredictStart: (ctx) => ctx.skip(cached) });
    expect(got).toBe(cached);
    expect(encoded).toEqual([]);
    expect(calls).toEqual([]);
  });

  it("a forward error propagates and reaches onError once", async () => {
    const { agent } = makeAgent();
    agent.provider.runEncoder = async () => {
      throw new Error("forward failed");
    };
    const errors: unknown[] = [];
    await expect(
      agent.predictBatch(STATES, QUESTIONS, {
        batchSize: 2,
        sortByLength: true,
        hooks: { onError: (ctx: PredictContext) => void errors.push(ctx.error) },
      }),
    ).rejects.toThrow("forward failed");
    expect(errors.map(String)).toEqual(["Error: forward failed"]);
  });

  it("systemOne and predict run through predictBatch with one state", async () => {
    const { agent } = makeAgent();
    const seen: unknown[][] = [];
    agent.addHook({ onPredictStart: (ctx: PredictContext) => void seen.push(ctx.states) });
    const one = await agent.systemOne(STATES[0], QUESTIONS);
    expect(seen).toEqual([[STATES[0]]]);
    expect(one).toEqual((await makeAgent().agent.predictBatch([STATES[0]], QUESTIONS))[0]);
  });
});

describe("minConfidence", () => {
  it("flags answers below the threshold before the end hooks, raw answers intact", async () => {
    const { agent } = makeAgent();
    const plain = await makeAgent().agent.predictBatch(STATES.slice(0, 4), QUESTIONS);
    const seen: boolean[][] = [];
    const threshold = 0.6;
    const got = await agent.predictBatch(STATES.slice(0, 4), QUESTIONS, {
      minConfidence: threshold,
      onPredictEnd: (ctx) => {
        for (const r of ctx.results as unknown as SystemOneResult[]) {
          seen.push(Object.values(r.answers).map((a) => "low_confidence" in a));
        }
      },
    });
    const flags = got.map((r) => Object.values(r.answers).map((a) => "low_confidence" in a));
    const want = plain.map((r) => Object.values(r.answers).map((a) => a.answer_confidence < threshold));
    expect(flags).toEqual(want);
    expect(seen).toEqual(flags);
    expect(flags.flat()).toContain(true);
    expect(flags.flat()).toContain(false);
    for (const [i, r] of got.entries()) {
      for (const [qid, a] of Object.entries(r.answers)) {
        const { low_confidence: _flag, ...rest } = a as typeof a & { low_confidence?: true };
        expect(rest).toEqual(plain[i].answers[qid]);
      }
    }
  });

  it("is off by default and at 0", async () => {
    for (const minConfidence of [undefined, null, 0]) {
      const got = await makeAgent().agent.predictBatch(STATES.slice(0, 3), QUESTIONS, { minConfidence });
      expect(JSON.stringify(got)).not.toContain("low_confidence");
    }
  });

  it("works on predict and systemOne", async () => {
    const { agent } = makeAgent();
    for (const run of [agent.predict.bind(agent), agent.systemOne.bind(agent)]) {
      const r = await run(STATES[0], QUESTIONS, { minConfidence: 1 });
      expect(Object.values(r.answers).every((a) => (a as { low_confidence?: true }).low_confidence === true)).toBe(
        true,
      );
    }
  });

  it("rejects out-of-range and non-number thresholds before any hook runs", async () => {
    const { agent, calls } = makeAgent();
    let started = false;
    for (const bad of [1.5, -0.1, Number.NaN, true, "0.5"]) {
      await expect(
        agent.predictBatch(STATES, QUESTIONS, {
          minConfidence: bad as never,
          onPredictStart: () => {
            started = true;
          },
        }),
      ).rejects.toThrow(/minConfidence must be a number in \[0.0, 1.0\]/);
    }
    expect(started).toBe(false);
    expect(calls).toEqual([]);
  });
});
