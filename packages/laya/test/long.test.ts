// Port of upstream tests/test_onnx_long.py (plus the predict_long hook outcomes) on a fake provider:
// a character tokenizer and a head whose logits derive from each row's own tokens, so a window
// decoded for the wrong span changes the answer.
import { afterEach, describe, expect, it, vi } from "vitest";
import { Agent, type LongResult, type QuestionDef, type SystemOneResult } from "../src/agent.js";
import type { PredictContext } from "../src/hooks.js";
import type { Batch } from "../src/providers.js";
import { decodeWithData, parseTokenizerJson } from "../src/tokenizer.js";

const tok = {
  clsId: 1,
  sepId: 2,
  maskId: 3,
  padId: 0,
  maskToken: "[M]",
  encode: (text: string) => Array.from(text, (c) => 10 + ((c.codePointAt(0) ?? 0) % 40)),
  // Length-preserving: every window comes back as distinct text of its slice's length.
  decode: (ids: number[]) => ids.map((i) => String.fromCharCode(65 + (i % 26))).join(""),
};

function makeAgent() {
  const calls: number[] = [];
  const provider = {
    async runEncoder(batch: Batch) {
      calls.push(batch.inputIds.length);
      return { lastHidden: [] as number[][][] };
    },
    async runHead(_h: unknown, batch: Batch) {
      const logits = batch.inputIds.map((ids, r) => {
        const weight = 1 + (ids.reduce((a, b) => a + b, 0) % 4);
        return batch.markerMask[r].map((_, k) => (k === 0 ? weight : 1 / (k + 1)));
      });
      return { logits, act: batch.inputIds.map(() => [0.25, 0.75]) };
    },
  };
  const agent = new Agent({ provider, tok, cfg: { max_len: 64, head_max_len: 32 } } as never);
  return { agent, calls };
}

const QUESTIONS: Record<string, QuestionDef> = {
  dept: { type: "choice", instructions: "Which team?", criteria: { billing: "money", support: "help", sales: "buy" } },
  urgent: { type: "noul", instructions: "Urgent?" },
};
// 200 tokens over a 64-token budget; an 11-character cycle so the windows differ.
const LONG_STATE = Array.from({ length: 200 }, (_, k) => String.fromCharCode(65 + (k % 11))).join("");

afterEach(() => {
  vi.restoreAllMocks();
});

async function capture(agent: Agent, run: () => Promise<LongResult>) {
  const windows: unknown[] = [];
  const real = agent.predictBatch.bind(agent);
  vi.spyOn(agent, "predictBatch").mockImplementation(async (sts, q, o) => {
    windows.push(...sts);
    return real(sts, q, o);
  });
  const result = await run();
  return { result, windows };
}

describe("predictLong: a state that fits one window", () => {
  it("equals systemOne, with usage.windows = 1", async () => {
    const { agent, calls } = makeAgent();
    const short = await agent.predictLong("aa", QUESTIONS);
    expect(calls).toEqual([2]);
    const one = await makeAgent().agent.systemOne("aa", QUESTIONS);
    expect(short).toEqual({ ...one, usage: { ...one.usage, windows: 1 } });
    expect(Object.values(short.answers).some((a) => "window" in a)).toBe(false);
  });

  it("forwards lang to systemOne", async () => {
    const agent = new Agent({
      provider: makeAgent().agent.provider,
      tok,
      cfg: { max_len: 64, head_max_len: 32 },
      lang_temperatures: { de: { temperature: [3, 3, 3] } },
    } as never);
    const got = await agent.predictLong("aa", QUESTIONS, { lang: "de" });
    const want = await agent.systemOne("aa", QUESTIONS, { lang: "de" });
    expect(got).toEqual({ ...want, usage: { ...want.usage, windows: 1 } });
    expect(got).not.toEqual({ ...(await agent.predictLong("aa", QUESTIONS)) });
  });

  it("reports usage.windows = 0 when a start hook answered", async () => {
    const { agent, calls } = makeAgent();
    const cached = { model: "cache", answers: {}, usage: { input_tokens: 0, output_tokens: 0 } };
    const got = await agent.predictLong("aa", QUESTIONS, { onPredictStart: (ctx) => ctx.skip([cached]) });
    expect(got).toEqual({ ...cached, usage: { input_tokens: 0, output_tokens: 0, windows: 0 } });
    expect(cached.usage).toEqual({ input_tokens: 0, output_tokens: 0 });
    expect(calls).toEqual([]);
  });
});

describe("predictLong: windows and aggregation", () => {
  it("windows at max(64, maxLen - headMaxLen - 8) tokens with stride window / 2", async () => {
    const { agent } = makeAgent();
    const { result, windows } = await capture(agent, () => agent.predictLong(LONG_STATE, QUESTIONS));
    // 200 tokens, budget 64, step 32: starts 0, 32, ..., 160, where 160 + 64 covers the end.
    expect(windows).toHaveLength(6);
    expect(windows.map((w) => (w as string).length)).toEqual([64, 64, 64, 64, 64, 40]);
    const ids = tok.encode(LONG_STATE);
    expect(windows).toEqual([0, 32, 64, 96, 128, 160].map((s) => tok.decode(ids.slice(s, s + 64))));
    expect(result.usage.windows).toBe(6);
  });

  it("aggregates noul by max P(true) and choice by the most confident window, naming it", async () => {
    const { agent } = makeAgent();
    const { result, windows } = await capture(agent, () => agent.predictLong(LONG_STATE, QUESTIONS));
    vi.restoreAllMocks();
    const perWindow = await makeAgent().agent.predictBatch(windows, QUESTIONS);
    expect(result.model).toBe("laya-rl-agent");
    expect(Object.keys(result).sort()).toEqual(["answers", "model", "usage"]);
    for (const [qid, key] of [
      ["dept", "answer_confidence"],
      ["urgent", "noul"],
    ] as const) {
      const ans = result.answers[qid] as Record<string, unknown> & { window: { index: number } };
      const j = ans.window.index;
      const { window, ...rest } = ans;
      expect(rest).toEqual(perWindow[j].answers[qid]);
      const values = perWindow.map((pw) => (pw.answers[qid] as unknown as Record<string, number>)[key]);
      expect(ans[key]).toBe(Math.max(...values));
      // The first window among equals decides, as Python's max() does.
      expect(j).toBe(values.indexOf(Math.max(...values)));
      expect(window).toEqual({ index: j, token_start: j * 32, token_end: Math.min(j * 32 + 64, 200), count: 6 });
    }
    expect(new Set(perWindow.map((pw) => pw.answers.dept.answer_confidence)).size).toBeGreaterThan(1);
    expect(result.usage.input_tokens).toBe(perWindow.reduce((a, r) => a + r.usage.input_tokens, 0));
    expect(result.usage.output_tokens).toBe(0);
  });

  it("scores every window in one shared pass, and batchSize chunks without changing the answer", async () => {
    const shared = makeAgent();
    const result = await shared.agent.predictLong(LONG_STATE, QUESTIONS);
    expect(shared.calls).toEqual([6 * 2]);
    const chunked = makeAgent();
    expect(await chunked.agent.predictLong(LONG_STATE, QUESTIONS, { batchSize: 4 })).toEqual(result);
    expect(chunked.calls).toEqual([8, 4]);
  });

  it("honours an explicit window and stride", async () => {
    const r = await makeAgent().agent.predictLong(LONG_STATE, QUESTIONS, { window: 96, stride: 96 });
    expect(r.usage.windows).toBe(3);
    expect(r.answers.urgent.window?.count).toBe(3);
    expect([0, 96, 192]).toContain(r.answers.urgent.window?.token_start);
  });

  it("refuses any aggregate but auto", async () => {
    await expect(makeAgent().agent.predictLong(LONG_STATE, QUESTIONS, { aggregate: "mean" as never })).rejects.toThrow(
      /only aggregate='auto'/,
    );
  });

  it("returns empty answers with a window count for no questions", async () => {
    const r = await makeAgent().agent.predictLong(LONG_STATE, {});
    expect(r.answers).toEqual({});
    expect(r.usage.windows).toBe(6);
  });

  it("needs a tokenizer that can decode", async () => {
    const { decode: _decode, ...noDecode } = tok;
    const agent = new Agent({
      provider: makeAgent().agent.provider,
      tok: noDecode,
      cfg: { max_len: 64, head_max_len: 32 },
    } as never);
    await expect(agent.predictLong(LONG_STATE, QUESTIONS)).rejects.toThrow(/cannot decode/);
    await expect(agent.predictLong("short", QUESTIONS)).resolves.toBeTruthy();
  });
});

describe("predictLong: hook outcomes", () => {
  it("fires start and end once around the scan, with the window texts as ctx.states", async () => {
    const seen: string[] = [];
    const { agent } = makeAgent();
    await agent.predictLong(LONG_STATE, QUESTIONS, {
      hooks: {
        onPredictStart: (ctx: PredictContext) => void seen.push(`start:${ctx.states.length}`),
        onPredictEnd: (ctx: PredictContext) => void seen.push(`end:${ctx.results?.length}`),
      },
    });
    expect(seen).toEqual(["start:6", "end:6"]);
  });

  it("a skip answers the document unattributed with usage.windows = 0", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const { agent, calls } = makeAgent();
    const cached = {
      model: "cache",
      answers: { urgent: { type: "noul", noul: 0.9 } },
      usage: { input_tokens: 5, output_tokens: 0 },
    };
    const r = await agent.predictLong(LONG_STATE, QUESTIONS, { onPredictStart: (ctx) => ctx.skip([cached]) });
    expect(r).toEqual({ ...cached, usage: { input_tokens: 5, output_tokens: 0, windows: 0 } });
    expect(calls).toEqual([]);
    expect(warn).toHaveBeenCalledOnce();
    await expect(
      agent.predictLong(LONG_STATE, QUESTIONS, { onPredictStart: (ctx) => ctx.skip([cached, cached]) }),
    ).rejects.toThrow(/answered this state with 2 results/);
  });

  it("a rewritten scan is aggregated without answer.window", async () => {
    const { agent } = makeAgent();
    const r = await agent.predictLong(LONG_STATE, QUESTIONS, {
      onPredictStart: (ctx) => {
        ctx.states = ctx.states.slice(0, 2);
      },
    });
    expect(r.usage.windows).toBe(2);
    expect(Object.values(r.answers).some((a) => "window" in a)).toBe(false);
    expect(Object.keys(r.answers)).toEqual(["dept", "urgent"]);
  });

  it("an in-place mutation of ctx.states counts as a rewrite, not a split mismatch", async () => {
    const { agent } = makeAgent();
    const r = await agent.predictLong(LONG_STATE, QUESTIONS, {
      onPredictStart: (ctx) => {
        ctx.states.push("extra window");
      },
    });
    expect(r.usage.windows).toBe(7);
    expect(Object.values(r.answers).some((a) => "window" in a)).toBe(false);
  });

  it("a hook that leaves no states returns no answers and usage.windows = 0", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    const r = await makeAgent().agent.predictLong(LONG_STATE, QUESTIONS, {
      onPredictStart: (ctx) => {
        ctx.states = [];
      },
    });
    expect(r).toEqual({
      model: "laya-rl-agent",
      answers: {},
      usage: { input_tokens: 0, output_tokens: 0, windows: 0 },
    });
  });

  it("installed agent hooks and the caller's end hook still see the scan", async () => {
    const { agent } = makeAgent();
    const installed: number[] = [];
    agent.addHook({ onPredictStart: (ctx: PredictContext) => void installed.push(ctx.states.length) });
    let endResults: SystemOneResult[] | null = null;
    await agent.predictLong(LONG_STATE, QUESTIONS, {
      onPredictEnd: (ctx) => {
        endResults = ctx.results as unknown as SystemOneResult[];
      },
    });
    expect(installed).toEqual([6]);
    expect(endResults).toHaveLength(6);
  });
});

describe("decodeWithData", () => {
  it("decodes ByteLevel ids back to text, including a split multi-byte character", () => {
    const data = parseTokenizerJson({
      model: { vocab: { h: 0, i: 1, Ġ: 2, Ã: 3, "©": 4, "[MASK]": 5 }, merges: [] },
      pre_tokenizer: { type: "ByteLevel" },
      decoder: { type: "ByteLevel" },
      added_tokens: [{ id: 5, content: "[MASK]" }],
    });
    expect(data).not.toBeNull();
    const d = data as NonNullable<typeof data>;
    expect(decodeWithData(d, [0, 1, 2, 3, 4, 5])).toBe("hi é[MASK]");
    expect(decodeWithData(d, [0, 3])).toBe("h�");
    expect(decodeWithData(d, [0, 99, 1])).toBe("hi");
  });

  it("decodes a Metaspace/ByteFallback sequence as the multilingual checkpoint's decoder does", () => {
    const d = parseTokenizerJson({
      model: { vocab: { "▁hi": 0, "▁there": 1, "<0xC3>": 2, "<0xA9>": 3, "!": 4 }, merges: [], byte_fallback: true },
      pre_tokenizer: { type: "Metaspace", replacement: "▁", prepend_scheme: "always", split: true },
      decoder: {
        type: "Sequence",
        decoders: [
          { type: "Replace", pattern: { String: "▁" }, content: " " },
          { type: "ByteFallback" },
          { type: "Fuse" },
        ],
      },
    }) as NonNullable<ReturnType<typeof parseTokenizerJson>>;
    expect(decodeWithData(d, [0, 1, 2, 3, 4])).toBe(" hi thereé!");
    expect(decodeWithData(d, [0, 2, 4])).toBe(" hi�!");
  });

  it("returns null for a decoder it does not support", () => {
    const d = parseTokenizerJson({ model: { vocab: { a: 0 }, merges: [] }, decoder: { type: "WordPiece" } });
    expect(decodeWithData(d as NonNullable<typeof d>, [0])).toBeNull();
  });
});
