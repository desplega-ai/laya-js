// HTTP surface of laya-serve (upstream `laya/serve.py` at the pin), on Hono.
//
// Contract kept from serve.py: routes, request body fields, response bodies (`{"detail": ...}`
// for errors, the router result verbatim on success), status codes, the order checks run in,
// and the Server-Timing / X-Inference-Time-Ms headers. Additions: `/health` answers 503 until
// preload completes (the Python server does not listen before that), and reports `precision`.
import { timingSafeEqual } from "node:crypto";
import { checkQuestion, type Precision } from "@desplega/laya";
import { Hono } from "hono";
import type { ContentfulStatusCode } from "hono/utils/http-status";

// Guardrails for unauthenticated remote input, same values as serve.py.
export const MAX_QUESTIONS = 64;
export const MAX_STATE_CHARS = 50000;
export const MAX_BODY_BYTES = 2 * 1024 * 1024;
export const MAX_CHOICE_OPTIONS = 100;
export const MAX_SCORE_LEVELS = 32;
export const MAX_TOTAL_OPTIONS = 512;

const KNOWN_MODELS = new Set(["english", "multilingual", "typed-decisions"]);
// Public HF ids a client may name. The root bundle `convaiinnovations/laya` is deliberately
// absent: it means "let the router choose", not "pin English" (serve.py `_PUBLISHED_MODEL_IDS`).
const PUBLISHED_MODEL_IDS: Record<string, string> = {
  "convaiinnovations/laya-multilingual": "multilingual",
  "convaiinnovations/laya-typed-decisions": "typed-decisions",
};
// Router aliases (upstream `router.py` `_ALIASES`).
const ALIASES: Record<string, string> = {
  en: "english",
  laya: "english",
  default: "english",
  multi: "multilingual",
  ml: "multilingual",
  "laya-multilingual": "multilingual",
  typed: "typed-decisions",
  typed_decisions: "typed-decisions",
  "laya-typed-decisions": "typed-decisions",
  decisions: "typed-decisions",
};

/** Map a client's `model` onto a checkpoint, or null to auto-route (serve.py `_resolve_model`). */
export function resolveModel(model: unknown): string | null {
  if (model === null || model === undefined || model === false || model === "" || model === 0) return null;
  const s = String(model).trim().toLowerCase();
  const published = PUBLISHED_MODEL_IDS[s];
  if (published) return published;
  // A Jev client's `model` (e.g. "jev-1") is expected to miss: that means auto-select.
  const key = ALIASES[s] ?? s;
  return KNOWN_MODELS.has(key) ? key : null;
}

export interface PredictCallOptions {
  model: string | null;
  maxLen?: number;
  headMaxLen?: number;
}

/** What the app needs from a router. `main.ts` adapts the lib's Router to it; tests pass a fake. */
export interface ServerRouter {
  /** Resident checkpoints, least recently used first. */
  readonly loaded: string[];
  /** Artifact-store commit each resident checkpoint was loaded from; null when unknown. */
  readonly revisions: Record<string, string | null>;
  predict(state: unknown, questions: Record<string, unknown>, opts: PredictCallOptions): Promise<unknown>;
}

export interface Logger {
  error(msg: string, err?: unknown): void;
}

export interface AppOptions {
  /** Require `Authorization: Bearer <apiKey>` on /v1/systemone when set. */
  apiKey?: string | null;
  maxConcurrent?: number;
  maxTokenBudget?: number;
  precision?: Precision;
  /** False until preload completes; /health and /v1/systemone answer 503 meanwhile. */
  isReady?: () => boolean;
  logger?: Logger;
}

class HttpError extends Error {
  constructor(
    readonly status: ContentfulStatusCode,
    readonly detail: string,
    readonly headers: Record<string, string> = {},
  ) {
    super(detail);
  }
}

function codePoints(s: string): number {
  let n = 0;
  for (const _ of s) n++;
  return n;
}

function isObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

/** Reject absent or oversized inference requests before tokenization (serve.py `_check_request_limits`). */
export function checkRequestLimits(state: unknown, questions: unknown): asserts questions is Record<string, unknown> {
  // A null state would be answered as a decision about the text "null".
  if (state === null || state === undefined) throw new HttpError(400, "'state' is required");
  if (!isObject(questions)) throw new HttpError(400, "'questions' must be an object");
  const ids = Object.keys(questions);
  if (ids.length > MAX_QUESTIONS) {
    throw new HttpError(413, `too many questions (${ids.length} > ${MAX_QUESTIONS})`);
  }
  let total = 0;
  for (const qid of ids) {
    const q = questions[qid];
    if (!isObject(q)) continue;
    const crit = q.criteria;
    if (q.type === "choice" && (isObject(crit) || Array.isArray(crit))) {
      const count = Array.isArray(crit) ? crit.length : Object.keys(crit).length;
      total += count;
      if (count > MAX_CHOICE_OPTIONS) {
        throw new HttpError(413, `too many choice options for '${qid}' (${count} > ${MAX_CHOICE_OPTIONS})`);
      }
    } else if (q.type === "score" && Array.isArray(crit)) {
      total += crit.length;
      if (crit.length > MAX_SCORE_LEVELS) {
        throw new HttpError(413, `too many score levels for '${qid}' (${crit.length} > ${MAX_SCORE_LEVELS})`);
      }
    }
  }
  if (total > MAX_TOTAL_OPTIONS) {
    throw new HttpError(413, `too many answer options across questions (${total} > ${MAX_TOTAL_OPTIONS})`);
  }
  let len: number;
  try {
    len = codePoints(typeof state === "string" ? state : (JSON.stringify(state) ?? String(state)));
  } catch {
    len = MAX_STATE_CHARS + 1;
  }
  if (len > MAX_STATE_CHARS) throw new HttpError(413, `state too large (${len} > ${MAX_STATE_CHARS} chars)`);
}

/** Validate an optional max_len / head_max_len override (serve.py `_validate_budget_param`, 422). */
export function validateBudgetParam(body: Record<string, unknown>, key: string, cap: number): number | undefined {
  if (!(key in body)) return undefined;
  const v = body[key];
  if (v === null) return undefined;
  if (typeof v !== "number" || !Number.isInteger(v)) throw new HttpError(422, `${key} must be an integer`);
  if (v <= 0) throw new HttpError(422, `${key} must be a positive integer`);
  if (v > cap) throw new HttpError(422, `${key} exceeds server limit (${v} > ${cap})`);
  return v;
}

async function readBodyCapped(req: Request): Promise<Uint8Array> {
  // Content-Length is a client claim and absent under chunked framing, so it only allows an
  // early reject; the streaming cap below is what enforces the limit.
  const declared = req.headers.get("content-length");
  if (declared && /^\d+$/.test(declared.trim()) && Number(declared) > MAX_BODY_BYTES) {
    throw new HttpError(413, "request body too large");
  }
  if (!req.body) return new Uint8Array();
  const reader = req.body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (!value?.length) continue;
    total += value.length;
    if (total > MAX_BODY_BYTES) {
      await reader.cancel().catch(() => {});
      throw new HttpError(413, "request body too large");
    }
    chunks.push(value);
  }
  const out = new Uint8Array(total);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }
  return out;
}

function authMatches(supplied: string | undefined, expected: Buffer): boolean {
  const got = Buffer.from(supplied ?? "", "utf8");
  // timingSafeEqual needs equal lengths; compare against itself on a length mismatch so the
  // time spent does not depend on where the strings differ.
  if (got.length !== expected.length) {
    timingSafeEqual(expected, expected);
    return false;
  }
  return timingSafeEqual(got, expected);
}

/** One-at-a-time inference, as serve.py's single-worker pool plus its asyncio.Lock. */
class Gate {
  private tail: Promise<void> = Promise.resolve();
  run<T>(fn: () => Promise<T>): Promise<T> {
    const result = this.tail.then(fn);
    this.tail = result.then(
      () => {},
      () => {},
    );
    return result;
  }
}

/**
 * Question-validation errors name the question and what to fix, so serve.py returns their
 * text as 422 (Python `ValueError`). The TS lib throws plain `Error`s, so this recognises them
 * by the `question "<id>"` prefix every such message in `agent.ts` carries.
 */
function isQuestionError(e: unknown): e is Error {
  return e instanceof Error && /^question "/.test(e.message);
}

export function createApp(router: ServerRouter, opts: AppOptions = {}): Hono {
  const maxConcurrent = opts.maxConcurrent ?? 16;
  const maxTokenBudget = opts.maxTokenBudget ?? 8192;
  const precision = opts.precision ?? "fp32";
  const isReady = opts.isReady ?? (() => true);
  const logger = opts.logger ?? { error: (m, e) => console.error(m, e) };
  const expectedAuth = opts.apiKey ? Buffer.from(`Bearer ${opts.apiKey}`, "utf8") : null;
  const gate = new Gate();
  let inFlight = 0;

  const app = new Hono();

  app.onError((err, c) => {
    if (err instanceof HttpError) return c.json({ detail: err.detail }, err.status, err.headers);
    logger.error("unhandled error", err);
    return c.json({ detail: "Internal Server Error" }, 500);
  });
  app.notFound((c) => c.json({ detail: "Not Found" }, 404));

  app.get("/health", (c) => {
    const loaded = [...router.loaded];
    const revisions: Record<string, string | null> = {};
    const checkpointDevices: Record<string, string> = {};
    const cpuFallbacks: Record<string, { count: number; last_reason: string | null }> = {};
    for (const name of loaded) {
      revisions[name] = router.revisions[name] ?? null;
      // The TS runtime computes on CPU only, so every resident checkpoint is on "cpu" and
      // there is never a GPU -> CPU fallback to count. Keys kept for serve.py parity.
      checkpointDevices[name] = "cpu";
      cpuFallbacks[name] = { count: 0, last_reason: null };
    }
    const ready = isReady();
    return c.json(
      {
        status: ready ? "ok" : "loading",
        loaded,
        revisions,
        device: "cpu",
        device_is_preference: loaded.length === 0,
        checkpoint_devices: checkpointDevices,
        cpu_fallbacks: cpuFallbacks,
        precision,
      },
      ready ? 200 : 503,
    );
  });

  app.post("/v1/systemone", async (c) => {
    if (expectedAuth && !authMatches(c.req.header("authorization"), expectedAuth)) {
      throw new HttpError(401, "invalid or missing bearer token");
    }
    if (!isReady()) throw new HttpError(503, "model loading, try again later", { "Retry-After": "5" });
    // Non-blocking admission: excess load is refused rather than queued, so the bodies
    // buffered at once stay bounded (serve.py #330).
    if (inFlight >= maxConcurrent) throw new HttpError(503, "server busy, try again later", { "Retry-After": "1" });
    inFlight++;
    try {
      const raw = await readBodyCapped(c.req.raw);
      let body: unknown;
      try {
        body = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(raw));
      } catch {
        throw new HttpError(400, "request body must be valid JSON");
      }
      if (!isObject(body) || !("questions" in body)) {
        throw new HttpError(400, "request body must be an object with a 'questions' field");
      }
      const state = body.state;
      const questions = body.questions;
      checkRequestLimits(state, questions);
      const model = resolveModel(body.model);
      const maxLen = validateBudgetParam(body, "max_len", maxTokenBudget);
      const headMaxLen = validateBudgetParam(body, "head_max_len", maxTokenBudget);
      const callOpts: PredictCallOptions = { model };
      if (maxLen !== undefined) callOpts.maxLen = maxLen;
      if (headMaxLen !== undefined) callOpts.headMaxLen = headMaxLen;

      return await gate.run(async () => {
        try {
          for (const [qid, q] of Object.entries(questions)) checkQuestion(qid, q);
          const t0 = performance.now();
          const result = await router.predict(state, questions, callOpts);
          const ms = (performance.now() - t0).toFixed(2);
          return c.json(result as Record<string, unknown>, 200, {
            "Server-Timing": `inference;dur=${ms}`,
            "X-Inference-Time-Ms": ms,
          });
        } catch (e) {
          if (e instanceof HttpError) throw e;
          if (isQuestionError(e)) throw new HttpError(422, e.message);
          // Never leak paths, weights or memory state to the client; the log gets the cause.
          // `model` is already reduced to a checkpoint name or null, so it is safe to log.
          logger.error(`inference failed for model=${model}`, e);
          throw new HttpError(500, "inference failed");
        }
      });
    } finally {
      inFlight--;
    }
  });

  // FastAPI answers a known path with the wrong method as 405.
  const methodNotAllowed = () => {
    throw new HttpError(405, "Method Not Allowed");
  };
  app.all("/health", methodNotAllowed);
  app.all("/v1/systemone", methodNotAllowed);

  return app;
}
