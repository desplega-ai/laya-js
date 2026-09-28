// Modified by Desplega Labs, 2026: per-call maxLen/headMaxLen cases (call opts, ctx.maxLen, budget cap).
import { describe, expect, it } from "vitest";
import { Agent, defaultTokenizer } from "../src/agent.js";
import { MAX_TOKEN_BUDGET, checkTokenBudget } from "../src/budget.js";
import { maxOf } from "../src/common.js";
import { Router } from "../src/router.js";

describe("maxOf", () => {
  it("returns max without spread overflow", () => {
    expect(maxOf([1, 5, 3], 1)).toBe(5);
    expect(maxOf([], 1)).toBe(1);
  });
  it("handles 300k rows without RangeError", () => {
    const lens = new Array(300_000).fill(1);
    lens[123456] = 77;
    expect(maxOf(lens, 1)).toBe(77);
  });
});

function recordingAgent(cfg: Record<string, unknown> = { max_len: 512, head_max_len: 192 }) {
  const widths: number[] = [];
  const provider = {
    async runEncoder(b: { attentionMask: number[][] }) {
      widths.push(Math.max(...b.attentionMask.map((m) => m.reduce((a, v) => a + v, 0))));
      return { lastHidden: [] };
    },
    async runHead(_h: unknown, b: { inputIds: number[][] }) {
      return { logits: b.inputIds.map(() => Array(8).fill(0.3)), act: b.inputIds.map(() => [1, 0]) };
    },
  };
  return { agent: new Agent({ provider, tok: defaultTokenizer(), cfg } as never), widths };
}

const Q = { q: { type: "noul", instructions: "Is it urgent?" } } as never;
const LONG = Array.from({ length: 400 }, (_, i) => `word${i}`).join(" ");

describe("per-call token budget", () => {
  it("a per-call maxLen truncates differently from the config", async () => {
    const { agent, widths } = recordingAgent();
    await agent.predict(LONG, Q);
    await agent.predict(LONG, Q, { maxLen: 64 });
    await agent.predictBatch([LONG, LONG], Q, { maxLen: 100 });
    expect(widths.slice(1)).toEqual([64, 100]);
    expect(widths[0]).toBeGreaterThan(400);
    expect(agent.maxLen).toBe(512);
  });

  it("call opts outrank the config and ctx.maxLen outranks call opts", async () => {
    const { agent, widths } = recordingAgent();
    await agent.predict(LONG, Q, {
      maxLen: 300,
      onPredictStart: (ctx) => {
        expect(ctx.maxLen).toBe(300);
        ctx.maxLen = 48;
      },
    });
    expect(widths).toEqual([48]);
  });

  it("ctx.maxLen set by an installed start hook takes effect", async () => {
    const { agent, widths } = recordingAgent();
    agent.addHook({ onPredictStart: (ctx: { maxLen: number | null }) => void (ctx.maxLen = 80) });
    await agent.predict(LONG, Q);
    expect(widths).toEqual([80]);
  });

  it("headMaxLen reaches the option budget", async () => {
    const { agent, widths } = recordingAgent();
    const long = (i: number) => `option ${i} ` + "with a long description ".repeat(6);
    const many = { q: { type: "choice", instructions: "Pick", criteria: Object.fromEntries(Array.from({ length: 6 }, (_, i) => [`o${i}`, long(i)])) } } as never;
    await agent.predict("short", many);
    await agent.predict("short", many, { headMaxLen: 48 });
    expect(widths[1]).toBeLessThan(widths[0]);
  });

  it("above-budget and malformed values throw", async () => {
    const { agent, widths } = recordingAgent();
    for (const [key, value, re] of [
      ["maxLen", MAX_TOKEN_BUDGET + 1, /maxLen exceeds the token budget \(8193 > 8192\)/],
      ["headMaxLen", 9000, /headMaxLen exceeds the token budget/],
      ["maxLen", 0, /maxLen must be a positive integer/],
      ["maxLen", -5, /maxLen must be a positive integer/],
      ["maxLen", 1.5, /maxLen must be an integer/],
      ["headMaxLen", "12", /headMaxLen must be an integer/],
      ["maxLen", true, /maxLen must be an integer/],
    ] as const) {
      await expect(agent.predict(LONG, Q, { [key]: value } as never)).rejects.toThrow(re);
      await expect(agent.predictBatch([LONG], Q, { [key]: value } as never)).rejects.toThrow(re);
    }
    await expect(agent.predict(LONG, Q, { onPredictStart: (ctx) => void (ctx.maxLen = 10_000) })).rejects.toThrow(
      /maxLen exceeds the token budget/,
    );
    expect(widths).toEqual([]);
    expect(checkTokenBudget("maxLen", MAX_TOKEN_BUDGET)).toBe(8192);
    expect(checkTokenBudget("maxLen", null)).toBe(null);
  });

  it("Router.predict forwards the per-call budget and a start hook may replace it", async () => {
    const { agent, widths } = recordingAgent();
    const router = new Router({ loader: () => agent });
    await router.predict(LONG, Q, { maxLen: 72 });
    await router.predict(LONG, Q, { maxLen: 72, onPredictStart: (ctx) => void (ctx.maxLen = 56) });
    expect(widths).toEqual([72, 56]);
    await expect(router.predict(LONG, Q, { maxLen: 9999 })).rejects.toThrow(/token budget/);
  });
});
