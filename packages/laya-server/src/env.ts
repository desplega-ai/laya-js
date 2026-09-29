// Server configuration from environment variables. Names and defaults mirror upstream
// `laya/serve.py`; the additions (LAYA_PRECISION, LAYA_MODEL_DIR, LAYA_CACHE_DIR,
// LAYA_ONNX_REPO, LAYA_ONNX_REVISION, HF_TOKEN, HF_ENDPOINT) come from the plan (Phase 8).
//
// Deliberate differences from serve.py:
// - LAYA_MODELS is the set of checkpoints the server serves, not only what it preloads; the
//   first entry is the default that out-of-set routing falls back to. See routing.ts.
// - An invalid value fails fast instead of silently
// falling back to the default. A typo in a deployment file should crash-loop visibly rather
// than run a server with a different concurrency cap or token budget than the operator wrote.
import { homedir } from "node:os";
import { join } from "node:path";
import type { CheckpointName, Precision } from "@desplega.ai/laya";
import { ARTIFACT_REPO, ARTIFACT_REVISION, MAX_TOKEN_BUDGET, normaliseName } from "@desplega.ai/laya/raw";

export const DEFAULT_PORT = 8000;
export const DEFAULT_MAX_CONCURRENT = 16;
export const DEFAULT_MAX_TOKEN_BUDGET = 8192;
export const DEFAULT_MODELS: CheckpointName[] = ["multilingual"];
export const DEFAULT_MODEL_DIR = "/models";
export const DEFAULT_HF_ENDPOINT = "https://huggingface.co";
export const LOG_LEVELS = ["critical", "error", "warning", "info", "debug", "trace"] as const;
export type LogLevel = (typeof LOG_LEVELS)[number];

export interface ServerEnv {
  host: string;
  port: number;
  /** LAYA_DEVICE as given. The TS runtime is CPU-only, so this is a preference that is never met by anything else. */
  device: string | null;
  preload: boolean;
  /** The only checkpoints the server loads (preloaded, fetched into the cache when not baked). */
  models: CheckpointName[];
  /** Serves requests whose routing picks a checkpoint outside `models`: the first of LAYA_MODELS. */
  defaultModel: CheckpointName;
  threads: number | null;
  autoTask: boolean;
  maxLoaded: number | null;
  /** Secret. Never log it. */
  apiKey: string | null;
  logLevel: LogLevel;
  maxConcurrent: number;
  maxTokenBudget: number;
  precision: Precision;
  modelDir: string;
  cacheDir: string;
  onnxRepo: string;
  onnxRevision: string;
  /** Secret. Never log it. */
  hfToken: string | null;
  hfEndpoint: string;
}

export class EnvError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "EnvError";
  }
}

type Env = Record<string, string | undefined>;

function str(env: Env, name: string): string | null {
  const v = env[name];
  if (v === undefined) return null;
  const t = v.trim();
  return t === "" ? null : t;
}

function int(env: Env, name: string, check: (n: number) => boolean, rule: string): number | null {
  const raw = str(env, name);
  if (raw === null) return null;
  // Python int() semantics: optional sign, digits, surrounding whitespace already stripped.
  if (!/^[+-]?\d+$/.test(raw)) throw new EnvError(`invalid ${name} ${JSON.stringify(raw)}: must be ${rule}`);
  const n = Number(raw);
  if (!Number.isSafeInteger(n) || !check(n))
    throw new EnvError(`invalid ${name} ${JSON.stringify(raw)}: must be ${rule}`);
  return n;
}

const positive = (n: number) => n > 0;

function bool(env: Env, name: string, dflt: boolean): boolean {
  const raw = str(env, name);
  if (raw === null) return dflt;
  const v = raw.toLowerCase();
  // serve.py `_env_bool` treats these as true; everything else it reads as false. We also
  // accept the obvious false spellings and reject anything else rather than guess.
  if (["1", "true", "yes", "on"].includes(v)) return true;
  if (["0", "false", "no", "off"].includes(v)) return false;
  throw new EnvError(`invalid ${name} ${JSON.stringify(raw)}: must be one of 1/true/yes/on or 0/false/no/off`);
}

function models(env: Env): CheckpointName[] {
  const raw = str(env, "LAYA_MODELS");
  if (raw === null) return [...DEFAULT_MODELS];
  const names = raw
    .split(",")
    .map((m) => m.trim())
    .filter(Boolean);
  if (names.length === 0) return [...DEFAULT_MODELS];
  const out: CheckpointName[] = [];
  for (const n of names) {
    let key: CheckpointName;
    try {
      key = normaliseName(n);
    } catch (e) {
      throw new EnvError(`invalid LAYA_MODELS entry ${JSON.stringify(n)}: ${(e as Error).message}`);
    }
    if (!out.includes(key)) out.push(key);
  }
  return out;
}

function defaultCacheDir(env: Env): string {
  const xdg = str(env, "XDG_CACHE_HOME");
  return join(xdg ?? join(homedir(), ".cache"), "laya-server");
}

/** Parse and validate the server environment. Throws `EnvError` on the first bad value. */
export function loadEnv(env: Env = process.env): ServerEnv {
  const port = int(env, "LAYA_PORT", (n) => n >= 1 && n <= 65535, "an integer 1-65535") ?? DEFAULT_PORT;
  const precision = str(env, "LAYA_PRECISION") ?? "fp32";
  if (precision !== "fp32") {
    throw new EnvError(
      `invalid LAYA_PRECISION ${JSON.stringify(precision)}: only "fp32" is available (INT8 is deferred)`,
    );
  }
  const logLevel = (str(env, "LAYA_LOG_LEVEL") ?? "info").toLowerCase();
  if (!(LOG_LEVELS as readonly string[]).includes(logLevel)) {
    throw new EnvError(`invalid LAYA_LOG_LEVEL ${JSON.stringify(logLevel)}: must be one of ${LOG_LEVELS.join(", ")}`);
  }
  const hfEndpoint = (str(env, "HF_ENDPOINT") ?? DEFAULT_HF_ENDPOINT).replace(/\/+$/, "");
  if (!/^https?:\/\//.test(hfEndpoint)) {
    throw new EnvError(`invalid HF_ENDPOINT ${JSON.stringify(hfEndpoint)}: must be an http(s) URL`);
  }
  const served = models(env);
  return {
    host: str(env, "LAYA_HOST") ?? "0.0.0.0",
    port,
    device: str(env, "LAYA_DEVICE"),
    preload: bool(env, "LAYA_PRELOAD", true),
    models: served,
    defaultModel: served[0],
    threads: int(env, "LAYA_THREADS", positive, "a positive integer"),
    autoTask: bool(env, "LAYA_AUTO_TASK", false),
    maxLoaded: int(env, "LAYA_MAX_LOADED", positive, "a positive integer"),
    // Not trimmed: the key is compared byte for byte, as serve.py does.
    apiKey: env.LAYA_API_KEY ? env.LAYA_API_KEY : null,
    logLevel: logLevel as LogLevel,
    maxConcurrent: int(env, "LAYA_MAX_CONCURRENT", positive, "a positive integer") ?? DEFAULT_MAX_CONCURRENT,
    // The lib refuses any per-call budget above its MAX_TOKEN_BUDGET, so a higher server cap would
    // only turn valid-looking requests into 500s.
    maxTokenBudget:
      int(env, "LAYA_MAX_TOKEN_BUDGET", (n) => n > 0 && n <= MAX_TOKEN_BUDGET, `an integer 1-${MAX_TOKEN_BUDGET}`) ??
      DEFAULT_MAX_TOKEN_BUDGET,
    precision,
    modelDir: str(env, "LAYA_MODEL_DIR") ?? DEFAULT_MODEL_DIR,
    cacheDir: str(env, "LAYA_CACHE_DIR") ?? defaultCacheDir(env),
    onnxRepo: str(env, "LAYA_ONNX_REPO") ?? ARTIFACT_REPO,
    onnxRevision: str(env, "LAYA_ONNX_REVISION") ?? ARTIFACT_REVISION,
    hfToken: str(env, "HF_TOKEN"),
    hfEndpoint,
  };
}

/** The config with every secret replaced by a presence flag: the only form that may be logged. */
export function describeEnv(e: ServerEnv): Record<string, unknown> {
  const { apiKey, hfToken, ...rest } = e;
  return { ...rest, apiKey: apiKey ? "(set)" : "(unset)", hfToken: hfToken ? "(set)" : "(unset)" };
}
