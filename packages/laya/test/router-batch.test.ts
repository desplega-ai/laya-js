// Port of upstream tests/test_router_batch.py, plus Router.predictLong, decideBatch and loadedRevisions.
import { afterEach, describe, expect, it } from "vitest";
import { Agent, type PredictBatchOptions, type QuestionDef } from "../src/agent.js";
import { clearDefaultHooks, type PredictContext, setDefaultHooks } from "../src/hooks.js";
import type { Batch } from "../src/providers.js";
import { Router, type RouterRequest } from "../src/router.js";

const Q: Record<string, QuestionDef> = { intent: { type: "noul", instructions: "Relevant?" } };

function request(state: unknown, overrides: Partial<RouterRequest> = {}): RouterRequest {
  return { state, questions: Q, ...overrides };
}

type Call = { checkpoint: string; states: unknown[]; questions: unknown; opts: PredictBatchOptions };

/** A Router whose loader builds recording fake agents; `built` counts loads. */
function fakeRouter(opts: Record<string, unknown> = {}, agentExtra: Record<string, unknown> = {}) {
  const built: string[] = [];
  const calls: Call[] = [];
  const loader = (name: string) => {
    built.push(name);
    return {
      checkpoint: name,
      revision: `rev-${name}`,
      ...agentExtra,
      async predictBatch(states: unknown[], questions: unknown, o: PredictBatchOptions = {}) {
        calls.push({ checkpoint: name, states: [...states], questions, opts: { ...o } });
        if (states.includes("raise")) throw new Error("inference failed");
        return states.map((state) => ({
          model: "laya-rl-agent",
          answers: {
            seen: state,
            intent: { type: "noul", noul: 0.5, answer_confidence: state === "unsure" ? 0.4 : 0.9 },
          },
          usage: { input_tokens: 3, output_tokens: 0 },
        }));
      },
      async systemOne(state: unknown, questions: unknown, o: PredictBatchOptions = {}) {
        return (await this.predictBatch([state], questions, o))[0];
      },
      async predictLong(state: unknown, _q: unknown, o: Record<string, unknown> = {}) {
        calls.push({ checkpoint: name, states: [state], questions: "long", opts: { ...o } });
        return {
          model: "laya-rl-agent",
          answers: { seen: state },
          usage: { input_tokens: 9, output_tokens: 0, windows: 4 },
        };
      },
    };
  };
  const router = new Router({ loader, ...opts } as never);
  return { router, built, calls };
}

const TYPED = Object.fromEntries(
  ["action", "needs_review", "outcome", "risk", "urgency"].map((k) => [k, { type: "noul", instructions: "?" }]),
) as Record<string, QuestionDef>;

afterEach(() => clearDefaultHooks());

describe("Router.routeBatch / predictBatch", () => {
  it.each([
    [1, ["typed-decisions"]],
    [2, ["multilingual", "typed-decisions"]],
    [3, ["english", "multilingual", "typed-decisions"]],
  ])("routes mixed states per state, keeps order and respects maxLoaded=%i", async (capacity, loaded) => {
    const { router, built, calls } = fakeRouter({ maxLoaded: capacity, autoTaskDetection: true });
    const items: RouterRequest[] = [
      request("English text one"),
      request("مرحبا"),
      request("English text two"),
      { state: "decision", questions: TYPED },
      request("forced", { model: "ml" }),
      request("forced english", { lang: "en" }),
      request("explicit task", { task: "typed_decisions" }),
    ];
    const decisions = router.routeBatch(items);
    expect(router.loaded).toEqual([]);
    const results = await router.predictBatch(items);
    expect(results.map((r) => (r.answers as Record<string, unknown>).seen)).toEqual(items.map((i) => i.state));
    expect(results.map((r) => r.routing)).toEqual(decisions);
    expect(built).toEqual(["english", "multilingual", "typed-decisions"]);
    expect(router.loaded).toEqual(loaded);
    expect(calls.map((c) => [c.checkpoint, c.states])).toEqual([
      ["english", ["English text one", "English text two", "forced english"]],
      ["multilingual", ["مرحبا", "forced"]],
      ["typed-decisions", ["decision"]],
      ["typed-decisions", ["explicit task"]],
    ]);
    expect(await router.predictBatch([])).toEqual([]);
    expect(router.routeBatch([])).toEqual([]);
  });

  it.each([
    [null, TypeError, "requests must be an array"],
    [{}, TypeError, "requests must be an array"],
    ["text", TypeError, "requests must be an array"],
    [[null], TypeError, "request 0"],
    [[{ questions: Q }], Error, "request 0 is missing required key 'state'"],
    [[{ state: "x" }], Error, "request 0 is missing required key 'questions'"],
    [[request("x"), { state: "y", questions: null }], TypeError, "request 1 'questions'"],
    [[request("x"), request("y", { model: "invalid" })], Error, "unknown model"],
    [[request("x"), request("y", { maxLen: 9000 })], RangeError, "maxLen exceeds the token budget"],
  ])("rejects an invalid batch before loading (%#)", async (items, error, fragment) => {
    const { router, built } = fakeRouter();
    const run = router.predictBatch(items as never);
    await expect(run).rejects.toThrow(error as ErrorConstructor);
    await expect(router.predictBatch(items as never)).rejects.toThrow(fragment as string);
    expect(built).toEqual([]);
    expect(router.loaded).toEqual([]);
  });

  it("propagates an inference error, ends every started request, and keeps the cache consistent", async () => {
    const { router, built, calls } = fakeRouter({ maxLoaded: 1 });
    const events: string[] = [];
    router.addHook({
      onPredictStart: (ctx: PredictContext) => void events.push(`start:${ctx.states[0]}`),
      onError: (ctx: PredictContext) => void events.push(`error:${ctx.states[0]}`),
      onPredictEnd: (ctx: PredictContext) => void events.push(`end:${ctx.states[0]}`),
    });
    await expect(
      router.predictBatch([request("first"), request("raise", { lang: "ar" }), request("unreached", { lang: "ar" })]),
    ).rejects.toThrow("inference failed");
    expect(built).toEqual(["english", "multilingual"]);
    expect(calls.map((c) => [c.checkpoint, c.states])).toEqual([
      ["english", ["first"]],
      ["multilingual", ["raise", "unreached"]],
    ]);
    expect(events).toEqual([
      "start:first",
      "end:first",
      "start:raise",
      "start:unreached",
      "error:unreached",
      "end:unreached",
      "error:raise",
      "end:raise",
    ]);
    expect(router.loaded).toEqual(["multilingual"]);
    expect([...router._agents.keys()]).toEqual(router.loaded);
    const after = await router.predict("after failure", Q, { lang: "ar" });
    expect((after.answers as Record<string, unknown>).seen).toBe("after failure");
  });

  it("reuses a warm checkpoint across batches", async () => {
    const { router, built } = fakeRouter();
    await router.predictBatch([request("one"), request("two")]);
    await router.predictBatch([request("three")]);
    expect(built).toEqual(["english"]);
  });

  it("shares one agent batch for equal questions and splits on different or reordered ones", async () => {
    const { router, calls } = fakeRouter();
    const other = { intent: { type: "noul", instructions: "Other?" } } as Record<string, QuestionDef>;
    const ab = { c: { type: "choice", instructions: "?", criteria: ["a", "b"] } } as Record<string, QuestionDef>;
    const ba = { c: { type: "choice", instructions: "?", criteria: ["b", "a"] } } as Record<string, QuestionDef>;
    await router.predictBatch([
      request("one"),
      { state: "two", questions: other },
      request("three"),
      { state: "four", questions: ab },
      { state: "five", questions: ba },
    ]);
    expect(calls.map((c) => c.states)).toEqual([["one", "three"], ["two"], ["four"], ["five"]]);
  });

  it("forwards batchSize and sortByLength to every agent call", async () => {
    const { router, calls } = fakeRouter();
    await router.predictBatch([request("a"), { state: "b", questions: TYPED }], { batchSize: 7, sortByLength: true });
    expect(calls.map((c) => [c.opts.batchSize, c.opts.sortByLength])).toEqual([
      [7, true],
      [7, true],
    ]);
    const plain = fakeRouter();
    await plain.router.predictBatch([request("a")]);
    expect("sortByLength" in plain.calls[0].opts).toBe(false);
  });

  it("passes lang only to an agent with per-language temperatures, and splits groups on it", async () => {
    const withLang = fakeRouter({}, { langTemperatures: { de: {} } });
    await withLang.router.predictBatch([
      request("Guten Tag, ich habe eine Frage zu meiner Rechnung und bitte um Hilfe."),
      request("x", { model: "ml", lang: "fr" }),
      request("y", { model: "ml", lang: "fr" }),
    ]);
    expect(withLang.calls.map((c) => [c.states.length, c.opts.lang ?? null])).toEqual([
      [1, "de"],
      [2, "fr"],
    ]);
    const without = fakeRouter();
    await without.router.predictBatch([
      request("x", { model: "ml", lang: "de" }),
      request("y", { model: "ml", lang: "fr" }),
    ]);
    expect(without.calls.map((c) => [c.states, "lang" in c.opts])).toEqual([[["x", "y"], false]]);
  });

  it("predict and predictBatch send the same lang and budget", async () => {
    const { router, calls } = fakeRouter({}, { langTemperatures: { de: {} } });
    const req = request("Guten Tag, ich habe eine Frage zu meiner Rechnung und bitte um Hilfe.", {
      maxLen: 128,
      headMaxLen: 48,
    });
    await router.predict(req.state, Q, { maxLen: 128, headMaxLen: 48 });
    await router.predictBatch([req]);
    expect(calls[0].opts.lang).toBe("de");
    expect(calls[1].opts.lang).toBe(calls[0].opts.lang);
    expect([calls[1].opts.maxLen, calls[1].opts.headMaxLen]).toEqual([128, 48]);
    expect([calls[0].opts.maxLen, calls[0].opts.headMaxLen]).toEqual([128, 48]);
  });

  it("per-request budgets reach the forward pass, split groups, and a start hook outranks them", async () => {
    const { router, calls } = fakeRouter();
    router.addHook({
      onPredictStart: (ctx: PredictContext) => {
        if (ctx.states[0] === "hooked") ctx.maxLen = 40;
      },
    });
    const decisions = router.routeBatch([request("a", { maxLen: 100 }), request("a")]);
    await router.predictBatch([request("a", { maxLen: 100 }), request("b"), request("hooked", { maxLen: 100 })]);
    expect(calls.map((c) => [c.states, c.opts.maxLen ?? null, "headMaxLen" in c.opts])).toEqual([
      [["a"], 100, false],
      [["b"], null, false],
      [["hooked"], 40, false],
    ]);
    expect(decisions[0]).toEqual(decisions[1]);
  });

  it("runs Router hooks per request and skips a cached request", async () => {
    const { router, calls } = fakeRouter();
    const cached = { model: "cache", answers: {}, usage: { input_tokens: 0, output_tokens: 0 } };
    const ends: unknown[] = [];
    router.addHook({
      onPredictStart: (ctx: PredictContext) => {
        if (ctx.states[0] === "cached") ctx.skip([cached]);
      },
      onPredictEnd: (ctx: PredictContext) => void ends.push(ctx.states[0]),
    });
    const results = await router.predictBatch([request("one"), request("cached"), request("two")]);
    expect(calls.map((c) => c.states)).toEqual([["one", "two"]]);
    expect(results[1]).toBe(cached);
    expect((results[1] as unknown as { routing: { model: string } }).routing.model).toBe("english");
    expect(ends).toEqual(["two", "cached", "one"]);
  });

  it("runs process-wide default hooks once per request, not again inside the agent", async () => {
    const seen: string[] = [];
    setDefaultHooks({ onPredictStart: (ctx: PredictContext) => void seen.push(ctx.router ? "router" : "agent") });
    const provider = {
      async runEncoder() {
        return { lastHidden: [] };
      },
      async runHead(_h: unknown, b: Batch) {
        return { logits: b.inputIds.map(() => [0, 1]), act: b.inputIds.map(() => [1, 0]) };
      },
    };
    const agent = new Agent({ provider } as never);
    const router = new Router({ loader: () => agent });
    await router.predictBatch([request("one"), request("two")]);
    expect(seen).toEqual(["router", "router"]);
  });

  it("flags minConfidence before the end hooks", async () => {
    const { router } = fakeRouter();
    const seen: unknown[] = [];
    router.addHook({
      onPredictEnd: (ctx: PredictContext) =>
        void seen.push(
          (ctx.results?.[0]?.answers as Record<string, Record<string, unknown>> | undefined)?.intent.low_confidence,
        ),
    });
    const results = await router.predictBatch([request("sure"), request("unsure")], { minConfidence: 0.5 });
    expect(results.map((r) => (r.answers.intent as { low_confidence?: true }).low_confidence)).toEqual([
      undefined,
      true,
    ]);
    expect(seen).toEqual([true, undefined]);
    await expect(router.predictBatch([request("x")], { minConfidence: 2 })).rejects.toThrow(/minConfidence/);
  });
});

describe("Router.predictLong", () => {
  it("routes, then scans with the routed agent's predictLong and keeps the routing key", async () => {
    const { router, calls } = fakeRouter();
    const r = await router.predictLong("مرحبا بكم في المتجر", Q, { window: 96, stride: 48, batchSize: 4 });
    expect(r.routing.model).toBe("multilingual");
    expect(r.usage.windows).toBe(4);
    expect(calls).toEqual([
      {
        checkpoint: "multilingual",
        states: ["مرحبا بكم في المتجر"],
        questions: "long",
        opts: { window: 96, stride: 48, aggregate: undefined, batchSize: 4, lang: null },
      },
    ]);
  });

  it("forwards an explicit lang and the router hooks see the scan", async () => {
    const { router, calls } = fakeRouter();
    const events: string[] = [];
    router.addHook({
      onPredictStart: () => void events.push("start"),
      onPredictEnd: (ctx: PredictContext) => void events.push(`end:${ctx.results?.length}`),
    });
    await router.predictLong("hallo", Q, { lang: "de" });
    expect(calls[0].opts.lang).toBe("de");
    expect(events).toEqual(["start", "end:1"]);
  });

  it("a caller's start hook that answers wins over the scan", async () => {
    const { router, calls } = fakeRouter();
    const cached = { model: "cache", answers: {}, usage: { input_tokens: 0, output_tokens: 0 } };
    const r = await router.predictLong("hello", Q, { onPredictStart: (ctx) => ctx.skip([cached]) });
    expect(r).toBe(cached);
    expect(calls).toEqual([]);
  });

  it("raises TypeError for an agent without predictLong", async () => {
    const router = new Router({ loader: () => ({ systemOne: async () => ({}) }) });
    await expect(router.predictLong("hello", Q)).rejects.toThrow(/has no predictLong/);
  });
});

describe("decideBatch and loadedRevisions", () => {
  const schema = { type: "object", properties: { intent: { type: "boolean" } } };

  it("Router.decideBatch routes each state and projects in input order", async () => {
    const { router, calls } = fakeRouter();
    const out = await router.decideBatch(["English here", "مرحبا"], schema);
    expect(out).toEqual([{ intent: true }, { intent: true }]);
    expect(calls.map((c) => c.checkpoint)).toEqual(["english", "multilingual"]);
    const gated = await router.decideBatch(["sure", "unsure"], schema, { minConfidence: 0.5 });
    expect(gated).toEqual([{ intent: true }, { intent: null }]);
  });

  it("Agent.decideBatch shares one predictBatch and returns details on request", async () => {
    const seen: number[] = [];
    const provider = {
      async runEncoder(b: Batch) {
        seen.push(b.inputIds.length);
        return { lastHidden: [] };
      },
      async runHead(_h: unknown, b: Batch) {
        return { logits: b.inputIds.map((_, r) => (r === 0 ? [0, 3] : [3, 0])), act: b.inputIds.map(() => [1, 0]) };
      },
    };
    const agent = new Agent({ provider } as never);
    const out = await agent.decideBatch(["a", "b", "c"], schema, { batchSize: 2 });
    expect(out).toEqual([{ intent: true }, { intent: false }, { intent: true }]);
    expect(seen).toEqual([2, 1]);
    const details = await agent.decideBatch(["a"], schema, { returnDetails: true });
    expect(details[0].values).toEqual({ intent: true });
    expect(details[0].probabilities.intent).toEqual({ false: 0.0474, true: 0.9526 });
    await expect(agent.decideBatch("a" as never, schema)).rejects.toThrow(/states must be an array/);
    await expect(agent.decideBatch(["a"])).rejects.toThrow(/exactly one of schema= or questions=/);
  });

  it("loadedRevisions maps each resident checkpoint to its revision", async () => {
    const { router } = fakeRouter();
    expect(router.loadedRevisions).toEqual({});
    await router.predictBatch([request("English"), request("مرحبا")]);
    expect(router.loadedRevisions).toEqual({ english: "rev-english", multilingual: "rev-multilingual" });
    router.attach("typed", { revision: undefined });
    expect(router.loadedRevisions["typed-decisions"]).toBeNull();
  });
});
