import { Router } from "@desplega/laya/raw";
import { describe, expect, it } from "vitest";
import { createApp } from "../src/app.js";
import { agentRegistry, fallbackReason, routeWithin, toServerRouter } from "../src/routing.js";

const ENGLISH = "The server is down and none of our customers can log in. Please help us today.";
const QUESTIONS = { urgent: { type: "noul", instructions: "Does `state` need a reply today?" } };
const REPOS = {
  english: "desplega/laya-onnx/english/fp32",
  multilingual: "desplega/laya-onnx/multilingual/fp32",
  "typed-decisions": "desplega/laya-onnx/typed-decisions/fp32",
};

function build(allowed: string[] | null) {
  const loads: string[] = [];
  const router = new Router({
    autoTaskDetection: true,
    hooks: allowed ? [routeWithin(allowed, allowed[0], REPOS)] : [],
    loader: (name) => {
      loads.push(name);
      return {
        systemOne: async () => ({
          model: name,
          answers: { urgent: { type: "noul", noul: 0.9, confidence: 0.9, answer_confidence: 0.9 } },
          usage: { input_tokens: 5, output_tokens: 1 },
        }),
      };
    },
  });
  const app = createApp(
    toServerRouter(router, () => null),
    { allowedModels: allowed ?? undefined, logger: { error: () => {} } },
  );
  const post = (body: unknown) =>
    app.request("/v1/systemone", { method: "POST", body: JSON.stringify(body) }).then(async (r) => ({
      status: r.status,
      body: await r.json(),
    }));
  return { post, loads };
}

describe("routing outside LAYA_MODELS", () => {
  it("control: without the policy, English text with no model routes to english", async () => {
    const { post, loads } = build(null);
    const r = await post({ state: ENGLISH, questions: QUESTIONS });
    expect(r.status).toBe(200);
    expect(r.body.routing.model).toBe("english");
    expect(loads).toEqual(["english"]);
  });

  it("falls back to the default checkpoint and says so in routing.reason", async () => {
    const { post, loads } = build(["multilingual"]);
    const control = (await build(null).post({ state: ENGLISH, questions: QUESTIONS })).body.routing;
    const r = await post({ state: ENGLISH, questions: QUESTIONS });
    expect(r.status).toBe(200);
    expect(r.body.model).toBe("multilingual");
    expect(r.body.routing).toEqual({
      ...control,
      model: "multilingual",
      repo: REPOS.multilingual,
      reason: fallbackReason(control.reason, "english", "multilingual"),
    });
    expect(r.body.routing.reason).toMatch(/; "english" is not in LAYA_MODELS, served by "multilingual"$/);
    // english was never loaded.
    expect(loads).toEqual(["multilingual"]);
  });

  it("falls back from an auto-task typed-decisions route too", async () => {
    const { post, loads } = build(["multilingual"]);
    const questions = Object.fromEntries(
      ["action", "category", "churn_risk", "needs_human", "urgency"].map((q) => [q, QUESTIONS.urgent]),
    );
    const r = await post({ state: ENGLISH, questions });
    expect(r.body.routing.model).toBe("multilingual");
    expect(r.body.routing.workflow).toBe("customer_service");
    expect(r.body.routing.reason).toMatch(/"typed-decisions" is not in LAYA_MODELS, served by "multilingual"$/);
    expect(loads).toEqual(["multilingual"]);
  });

  it("leaves in-set decisions untouched, and the default is the first of LAYA_MODELS", async () => {
    const { post, loads } = build(["typed-decisions", "english"]);
    const inSet = await post({ state: ENGLISH, questions: QUESTIONS });
    expect(inSet.body.routing.model).toBe("english");
    expect(inSet.body.routing.reason).not.toMatch(/LAYA_MODELS/);
    const outOfSet = await post({
      state: "Das ist ein ganz normaler deutscher Satz über Rechnungen.",
      questions: QUESTIONS,
    });
    expect(outOfSet.body.routing.model).toBe("typed-decisions");
    expect(outOfSet.body.routing.reason).toMatch(/"multilingual" is not in LAYA_MODELS, served by "typed-decisions"$/);
    expect(loads).toEqual(["english", "typed-decisions"]);
  });

  it("an explicit out-of-set model is a 400, never a fallback", async () => {
    const { post, loads } = build(["multilingual"]);
    const r = await post({ state: ENGLISH, questions: QUESTIONS, model: "english" });
    expect(r.status).toBe(400);
    expect(loads).toEqual([]);
  });
});

describe("eviction releases ONNX sessions", () => {
  function fakeAgent(name: string, released: string[]) {
    return {
      systemOne: async () => ({ model: name, answers: {}, usage: { input_tokens: 1, output_tokens: 0 } }),
      dispose: async () => {
        released.push(name);
      },
    };
  }

  it("control: the lib Router alone evicts without releasing", async () => {
    const released: string[] = [];
    const router = new Router({ maxLoaded: 1, loader: (name) => fakeAgent(name, released) });
    await router.load("english");
    await router.load("multilingual");
    expect(router.loaded).toEqual(["multilingual"]);
    expect(released).toEqual([]);
  });

  it("the registry hook disposes each evicted agent once, and disposeAll the rest", async () => {
    const released: string[] = [];
    const agents = agentRegistry<ReturnType<typeof fakeAgent>>();
    const router = new Router({
      maxLoaded: 1,
      hooks: [agents.hook],
      loader: (name) => {
        const a = fakeAgent(name, released);
        agents.track(name, a);
        return a;
      },
    });
    await router.load("english");
    await router.load("multilingual");
    await router.load("multilingual");
    expect(released).toEqual(["english"]);
    expect(agents.get("english")).toBeUndefined();
    await router.load("english");
    expect(released).toEqual(["english", "multilingual"]);
    router.unload();
    await agents.disposeAll();
    expect(released).toEqual(["english", "multilingual", "english"]);
  });

  it("forwards per-call token budgets to the lib Router", async () => {
    const seen: unknown[] = [];
    const router = new Router({
      loader: (name) => ({
        systemOne: async (_s: unknown, _q: unknown, o: unknown) => {
          seen.push(o);
          return { model: name, answers: {}, usage: { input_tokens: 1, output_tokens: 0 } };
        },
      }),
    });
    await toServerRouter(router, () => null).predict("hi", {}, { model: "english", maxLen: 256, headMaxLen: 64 });
    expect(seen[0]).toMatchObject({ maxLen: 256, headMaxLen: 64 });
  });
});
