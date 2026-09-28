import { describe, expect, it, vi } from "vitest";
import {
  type AppOptions,
  createApp,
  MAX_BODY_BYTES,
  MAX_CHOICE_OPTIONS,
  MAX_QUESTIONS,
  MAX_SCORE_LEVELS,
  MAX_STATE_CHARS,
  type PredictCallOptions,
  resolveModel,
  type ServerRouter,
} from "../src/app.js";

const QUESTIONS = {
  department: { type: "choice", instructions: "Which team?", criteria: ["billing", "technical"] },
};

const RESULT = {
  model: "multilingual",
  answers: { department: { choice: "billing", confidence: 0.9 } },
  usage: { input_tokens: 12, output_tokens: 1 },
  routing: { model: "multilingual", reason: "explicit model", detection: null, workflow: null },
};

interface FakeRouter extends ServerRouter {
  calls: Array<{ state: unknown; questions: unknown; opts: PredictCallOptions }>;
}

function fakeRouter(predict?: (state: unknown) => Promise<unknown>): FakeRouter {
  const calls: FakeRouter["calls"] = [];
  return {
    calls,
    loaded: ["multilingual"],
    revisions: { multilingual: "e1d00cd746843d098f4c1aa3179964243cecfec2" },
    async predict(state, questions, opts) {
      calls.push({ state, questions, opts });
      return predict ? predict(state) : RESULT;
    },
  };
}

function post(app: ReturnType<typeof createApp>, body: unknown, headers: Record<string, string> = {}) {
  return app.request("/v1/systemone", {
    method: "POST",
    headers: { "content-type": "application/json", ...headers },
    body: typeof body === "string" ? body : JSON.stringify(body),
  });
}

function setup(opts: AppOptions = {}, predict?: (state: unknown) => Promise<unknown>) {
  const router = fakeRouter(predict);
  const logger = { error: vi.fn() };
  const app = createApp(router, { logger, ...opts });
  return { app, router, logger };
}

describe("GET /health", () => {
  it("reports loaded checkpoints, revisions, device and precision", async () => {
    const { app } = setup();
    const res = await app.request("/health");
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({
      status: "ok",
      loaded: ["multilingual"],
      revisions: { multilingual: "e1d00cd746843d098f4c1aa3179964243cecfec2" },
      device: "cpu",
      device_is_preference: false,
      checkpoint_devices: { multilingual: "cpu" },
      cpu_fallbacks: { multilingual: { count: 0, last_reason: null } },
      precision: "fp32",
    });
  });

  it("answers 503 until preload completes", async () => {
    let ready = false;
    const { app } = setup({ isReady: () => ready });
    const before = await app.request("/health");
    expect(before.status).toBe(503);
    expect((await before.json()).status).toBe("loading");
    ready = true;
    expect((await app.request("/health")).status).toBe(200);
  });

  it("needs no auth and never echoes the API key", async () => {
    const { app } = setup({ apiKey: "sekret-key-123" });
    const res = await app.request("/health");
    expect(res.status).toBe(200);
    expect(await res.text()).not.toContain("sekret");
  });
});

describe("auth", () => {
  it("is off when no key is configured", async () => {
    const { app } = setup();
    expect((await post(app, { state: "hi", questions: QUESTIONS })).status).toBe(200);
  });

  it("requires the exact bearer token when a key is configured", async () => {
    const { app, router } = setup({ apiKey: "k3y" });
    const body = { state: "hi", questions: QUESTIONS };
    for (const auth of [undefined, "Bearer wrong", "k3y", "bearer k3y", "Bearer s\xe9cret"]) {
      const res = await post(app, body, auth === undefined ? {} : { authorization: auth });
      expect(res.status, String(auth)).toBe(401);
      expect(await res.json()).toEqual({ detail: "invalid or missing bearer token" });
    }
    expect(router.calls).toHaveLength(0);
    expect((await post(app, body, { authorization: "Bearer k3y" })).status).toBe(200);
  });
});

describe("POST /v1/systemone", () => {
  it("returns the router result verbatim with timing headers", async () => {
    const { app, router } = setup();
    const res = await post(app, { state: "My invoice was charged twice", questions: QUESTIONS });
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual(RESULT);
    expect(res.headers.get("server-timing")).toMatch(/^inference;dur=\d+\.\d{2}$/);
    expect(res.headers.get("x-inference-time-ms")).toMatch(/^\d+\.\d{2}$/);
    expect(router.calls).toEqual([
      { state: "My invoice was charged twice", questions: QUESTIONS, opts: { model: null } },
    ]);
  });

  it("passes non-string states through", async () => {
    const { app, router } = setup();
    const state = { from: "a@b.c", body: "refund please" };
    expect((await post(app, { state, questions: QUESTIONS })).status).toBe(200);
    expect(router.calls[0].state).toEqual(state);
  });

  it("answers 503 before the models are ready", async () => {
    const { app, router } = setup({ isReady: () => false });
    const res = await post(app, { state: "hi", questions: QUESTIONS });
    expect(res.status).toBe(503);
    expect(router.calls).toHaveLength(0);
  });

  describe("400", () => {
    const cases: Array<[string, unknown, string]> = [
      ["malformed JSON", "{not json", "request body must be valid JSON"],
      ["an empty body", "", "request body must be valid JSON"],
      ["a non-object body", [1, 2], "request body must be an object with a 'questions' field"],
      ["missing questions", { state: "hi" }, "request body must be an object with a 'questions' field"],
      ["a null state", { state: null, questions: QUESTIONS }, "'state' is required"],
      ["a missing state", { questions: QUESTIONS }, "'state' is required"],
      ["questions as a list", { state: "hi", questions: [QUESTIONS] }, "'questions' must be an object"],
      ["null questions", { state: "hi", questions: null }, "'questions' must be an object"],
    ];
    it.each(cases)("on %s", async (_name, body, detail) => {
      const { app, router } = setup();
      const res = await post(app, body);
      expect(res.status).toBe(400);
      expect(await res.json()).toEqual({ detail });
      expect(router.calls).toHaveLength(0);
    });

    it("on a body that is not valid UTF-8", async () => {
      const { app } = setup();
      const res = await app.request("/v1/systemone", { method: "POST", body: new Uint8Array([0x7b, 0xff, 0x7d]) });
      expect(res.status).toBe(400);
    });
  });

  describe("413", () => {
    it("on a declared Content-Length over the cap, before reading", async () => {
      const { app } = setup();
      const res = await app.request("/v1/systemone", {
        method: "POST",
        headers: { "content-length": String(MAX_BODY_BYTES + 1) },
        body: "{}",
      });
      expect(res.status).toBe(413);
      expect(await res.json()).toEqual({ detail: "request body too large" });
    });

    it("on a streamed body over the cap with no Content-Length", async () => {
      const { app, router } = setup();
      const chunk = new Uint8Array(256 * 1024).fill(0x20);
      let sent = 0;
      const body = new ReadableStream<Uint8Array>({
        pull(ctrl) {
          if (sent > MAX_BODY_BYTES + chunk.length) return ctrl.close();
          sent += chunk.length;
          ctrl.enqueue(chunk);
        },
      });
      const res = await app.request("/v1/systemone", { method: "POST", body, duplex: "half" } as RequestInit);
      expect(res.status).toBe(413);
      expect(router.calls).toHaveLength(0);
    });

    it("on too many questions", async () => {
      const { app } = setup();
      const questions = Object.fromEntries(
        Array.from({ length: MAX_QUESTIONS + 1 }, (_, i) => [`q${i}`, { type: "noul", instructions: "?" }]),
      );
      const res = await post(app, { state: "hi", questions });
      expect(res.status).toBe(413);
      expect((await res.json()).detail).toBe(`too many questions (${MAX_QUESTIONS + 1} > ${MAX_QUESTIONS})`);
    });

    it("on too many choice options or score levels", async () => {
      const { app } = setup();
      const choice = {
        c: { type: "choice", instructions: "?", criteria: Array.from({ length: 101 }, (_, i) => `o${i}`) },
      };
      const r1 = await post(app, { state: "hi", questions: choice });
      expect(r1.status).toBe(413);
      expect((await r1.json()).detail).toBe(`too many choice options for 'c' (101 > ${MAX_CHOICE_OPTIONS})`);
      const score = {
        s: { type: "score", instructions: "?", criteria: Array.from({ length: 33 }, (_, i) => `l${i}`) },
      };
      const r2 = await post(app, { state: "hi", questions: score });
      expect(r2.status).toBe(413);
      expect((await r2.json()).detail).toBe(`too many score levels for 's' (33 > ${MAX_SCORE_LEVELS})`);
    });

    it("on too many options across questions", async () => {
      const { app } = setup();
      const questions = Object.fromEntries(
        Array.from({ length: 6 }, (_, q) => [
          `q${q}`,
          {
            type: "choice",
            instructions: "?",
            criteria: Object.fromEntries(Array.from({ length: 90 }, (_, i) => [`o${i}`, "x"])),
          },
        ]),
      );
      const res = await post(app, { state: "hi", questions });
      expect(res.status).toBe(413);
      expect((await res.json()).detail).toBe("too many answer options across questions (540 > 512)");
    });

    it("on a state over the character cap", async () => {
      const { app } = setup();
      const res = await post(app, { state: "a".repeat(MAX_STATE_CHARS + 1), questions: QUESTIONS });
      expect(res.status).toBe(413);
      expect((await res.json()).detail).toBe(`state too large (${MAX_STATE_CHARS + 1} > ${MAX_STATE_CHARS} chars)`);
      // Counted in characters (code points), as Python len(), not UTF-16 units.
      const ok = await post(app, { state: "😀".repeat(MAX_STATE_CHARS), questions: QUESTIONS });
      expect(ok.status).toBe(200);
    });
  });

  describe("token budget", () => {
    const cases: Array<[string, unknown, string]> = [
      ["max_len", 8193, "max_len exceeds server limit (8193 > 8192)"],
      ["max_len", 0, "max_len must be a positive integer"],
      ["max_len", -3, "max_len must be a positive integer"],
      ["max_len", 1.5, "max_len must be an integer"],
      ["max_len", "512", "max_len must be an integer"],
      ["max_len", true, "max_len must be an integer"],
      ["head_max_len", 9000, "head_max_len exceeds server limit (9000 > 8192)"],
    ];
    it.each(cases)("422 on %s=%j", async (key, value, detail) => {
      const { app, router } = setup();
      const res = await post(app, { state: "hi", questions: QUESTIONS, [key]: value });
      expect(res.status).toBe(422);
      expect(await res.json()).toEqual({ detail });
      expect(router.calls).toHaveLength(0);
    });

    it("honours LAYA_MAX_TOKEN_BUDGET", async () => {
      const { app } = setup({ maxTokenBudget: 256 });
      const res = await post(app, { state: "hi", questions: QUESTIONS, max_len: 300 });
      expect(await res.json()).toEqual({ detail: "max_len exceeds server limit (300 > 256)" });
    });

    it("forwards valid budgets and ignores null", async () => {
      const { app, router } = setup();
      await post(app, { state: "hi", questions: QUESTIONS, max_len: 1024, head_max_len: 128 });
      await post(app, { state: "hi", questions: QUESTIONS, max_len: null });
      expect(router.calls.map((c) => c.opts)).toEqual([
        { model: null, maxLen: 1024, headMaxLen: 128 },
        { model: null },
      ]);
    });
  });

  describe("model", () => {
    const cases: Array<[unknown, string | null]> = [
      [undefined, null],
      [null, null],
      ["", null],
      ["multilingual", "multilingual"],
      ["english", "english"],
      ["typed-decisions", "typed-decisions"],
      ["ml", "multilingual"],
      ["multi", "multilingual"],
      ["laya-multilingual", "multilingual"],
      ["en", "english"],
      ["LAYA", "english"],
      ["default", "english"],
      [" Typed_Decisions ", "typed-decisions"],
      ["decisions", "typed-decisions"],
      ["convaiinnovations/laya-multilingual", "multilingual"],
      ["ConvAIInnovations/Laya-Typed-Decisions", "typed-decisions"],
      // The root bundle id means "let the router choose", not English.
      ["convaiinnovations/laya", null],
      // A Jev model id is expected to miss and auto-route.
      ["jev-1", null],
      [42, null],
    ];
    it.each(cases)("%j resolves to %j", async (model, want) => {
      expect(resolveModel(model)).toBe(want);
      const { app, router } = setup();
      const res = await post(app, { state: "hi", questions: QUESTIONS, model });
      expect(res.status).toBe(200);
      expect(router.calls[0].opts.model).toBe(want);
    });
  });

  describe("errors from inference", () => {
    it("422 with the message on an invalid question, before inference", async () => {
      const { app, router } = setup();
      const res = await post(app, { state: "hi", questions: { q: { type: "bogus", instructions: "?" } } });
      expect(res.status).toBe(422);
      expect((await res.json()).detail).toMatch(/^question "q": /);
      expect(router.calls).toHaveLength(0);
    });

    it("422 on a question error raised during inference", async () => {
      const { app } = setup({}, async () => {
        throw new Error('question "department" options exceed head_max_len=192');
      });
      const res = await post(app, { state: "hi", questions: QUESTIONS });
      expect(res.status).toBe(422);
      expect(await res.json()).toEqual({ detail: 'question "department" options exceed head_max_len=192' });
    });

    it("500 with a fixed detail on anything else, cause logged but not returned", async () => {
      const { app, logger } = setup({}, async () => {
        throw new Error("ENOENT /models/multilingual/fp32/encoder.onnx");
      });
      const res = await post(app, { state: "hi", questions: QUESTIONS, model: "ml" });
      expect(res.status).toBe(500);
      const text = await res.text();
      expect(JSON.parse(text)).toEqual({ detail: "inference failed" });
      expect(text).not.toContain("models");
      expect(logger.error).toHaveBeenCalledWith("inference failed for model=multilingual", expect.any(Error));
    });
  });

  describe("concurrency", () => {
    it("refuses requests past LAYA_MAX_CONCURRENT with 503 and Retry-After", async () => {
      let release!: () => void;
      const blocked = new Promise<void>((r) => {
        release = r;
      });
      const { app } = setup({ maxConcurrent: 1 }, async () => {
        await blocked;
        return RESULT;
      });
      const first = post(app, { state: "one", questions: QUESTIONS });
      await vi.waitFor(() => new Promise((r) => setTimeout(r, 5)));
      const second = await post(app, { state: "two", questions: QUESTIONS });
      expect(second.status).toBe(503);
      expect(second.headers.get("retry-after")).toBe("1");
      expect(await second.json()).toEqual({ detail: "server busy, try again later" });
      release();
      expect((await first).status).toBe(200);
      // The slot is free again.
      expect((await post(app, { state: "three", questions: QUESTIONS })).status).toBe(200);
    });

    it("runs one inference at a time", async () => {
      let active = 0;
      let peak = 0;
      const { app } = setup({ maxConcurrent: 8 }, async () => {
        active++;
        peak = Math.max(peak, active);
        await new Promise((r) => setTimeout(r, 10));
        active--;
        return RESULT;
      });
      const all = await Promise.all(
        Array.from({ length: 5 }, (_, i) => post(app, { state: `s${i}`, questions: QUESTIONS })),
      );
      expect(all.map((r) => r.status)).toEqual([200, 200, 200, 200, 200]);
      expect(peak).toBe(1);
    });

    it("releases the slot after an error", async () => {
      const { app } = setup({ maxConcurrent: 1 });
      for (let i = 0; i < 3; i++) expect((await post(app, "{bad")).status).toBe(400);
      expect((await post(app, { state: "ok", questions: QUESTIONS })).status).toBe(200);
    });
  });
});

describe("routing", () => {
  it("404 on an unknown path and 405 on a wrong method, FastAPI-shaped", async () => {
    const { app } = setup();
    const nf = await app.request("/nope");
    expect(nf.status).toBe(404);
    expect(await nf.json()).toEqual({ detail: "Not Found" });
    const na = await app.request("/v1/systemone");
    expect(na.status).toBe(405);
    expect(await na.json()).toEqual({ detail: "Method Not Allowed" });
  });
});
