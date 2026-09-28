// Typed public interface over the vendored runtime (plan, Phase 5). Pure types plus thin
// factories: every number still comes from the vendored Agent/Router, so the Phase 3 parity
// gate covers this layer unchanged.
import type { StandardSchemaV1 } from "@standard-schema/spec";
import {
  Agent,
  type ChoiceAnswer,
  checkQuestion,
  type NoulAnswer,
  type PredictBatchOptions,
  type PredictLongOptions,
  type PredictOptions,
  type QuestionDef,
  type ScoreAnswer,
  type SystemOneResult,
  type SystemUsage,
  type WindowInfo,
} from "./agent.js";
import { ARTIFACT_REPO, ARTIFACT_REVISION, ARTIFACTS, type CheckpointName, type Precision } from "./artifacts.js";
import type { SessionProvider } from "./providers.js";
import {
  type ModelName,
  type ModelSpec,
  type RouteDecision,
  type RouteOptions,
  Router,
  type RouterBatchOptions,
  type RouterRequest,
} from "./router.js";
import { SchemaError } from "./structured.js";

// ---- Question definitions ------------------------------------------------------------------

/** A choice question: `criteria` is a list of labels or a label -> description map. */
export interface ChoiceQuestion<L extends string = string> {
  type: "choice";
  instructions: string;
  criteria: readonly L[] | { readonly [K in L]: string | null };
}

/** A score question: `criteria` lists the level descriptions, index 0 first. */
export interface ScoreQuestion {
  type: "score";
  instructions: string;
  criteria: readonly string[];
}

/** A yes/no question with optional `true`/`false` descriptions and display labels. */
export interface NoulQuestion {
  type: "noul";
  instructions: string;
  criteria?: { readonly true?: string; readonly false?: string } | null;
  labels?: { readonly false: string; readonly true: string } | null;
}

export type Question = ChoiceQuestion | ScoreQuestion | NoulQuestion;
export type QuestionMap = Record<string, Question>;

/** Set by `minConfidence` (Phase 6) on answers below the threshold; the raw answer is unchanged. */
export interface AbstentionFlag {
  low_confidence?: true;
}

export type LabelsOf<C> = C extends readonly (infer L)[] ? L & string : C extends object ? keyof C & string : never;

export type AnswerOf<D> = D extends { type: "choice"; criteria: infer C }
  ? Omit<ChoiceAnswer, "choice" | "probabilities"> & {
      choice: LabelsOf<C>;
      probabilities: Record<LabelsOf<C>, number>;
    }
  : D extends { type: "score" }
    ? ScoreAnswer
    : D extends { type: "noul" }
      ? NoulAnswer
      : never;

export type Answers<Q extends QuestionMap> = { [K in keyof Q]: AnswerOf<Q[K]> & AbstentionFlag };

export type PredictResult<Q extends QuestionMap> = Omit<SystemOneResult, "answers"> & { answers: Answers<Q> };

export type RoutedPredictResult<Q extends QuestionMap> = PredictResult<Q> & { routing: RouteDecision };

/** `predictLong` answers: the deciding window's answer, with `window` naming it when the scan was not rewritten. */
export type LongAnswers<Q extends QuestionMap> = {
  [K in keyof Q]: AnswerOf<Q[K]> & AbstentionFlag & { window?: WindowInfo };
};

export type LongPredictResult<Q extends QuestionMap> = {
  model: string;
  answers: LongAnswers<Q>;
  /** `windows`: windows the model scored (1 for a state that fit one, 0 when a hook answered). */
  usage: SystemUsage & { windows: number };
};

export type RoutedLongPredictResult<Q extends QuestionMap> = LongPredictResult<Q> & { routing: RouteDecision };

/** A typed `Router.predictBatch` request: its answers narrow to its own `questions`. */
export type TypedRouterRequest<Q extends QuestionMap = QuestionMap> = Omit<RouterRequest, "questions"> & {
  questions: Q;
};

/** One routed result per request, in input order, each narrowed to that request's questions. */
export type RoutedBatchResults<R extends readonly TypedRouterRequest[]> = {
  -readonly [K in keyof R]: RoutedPredictResult<R[K] extends { questions: infer Q extends QuestionMap } ? Q : never>;
};

/**
 * Validate a question map once and keep its literal types, so answers narrow to the labels.
 * Throws the same errors as the runtime's `checkQuestion`.
 */
export function defineQuestions<const Q extends QuestionMap>(questions: Q): Q {
  for (const [qid, qdef] of Object.entries(questions)) checkQuestion(qid, qdef);
  return questions;
}

// ---- Typed decide (Standard Schema) --------------------------------------------------------

type JsonSchemaCapable = {
  "~standard": { jsonSchema?: { input(opts: { target: string }): Record<string, unknown> } };
  toJSONSchema?: () => Record<string, unknown>;
};

/** JSON schema of a Standard Schema: its `~standard.jsonSchema` converter, else `toJSONSchema()`. */
export function jsonSchemaOf(schema: StandardSchemaV1): Record<string, unknown> {
  const s = schema as unknown as JsonSchemaCapable;
  const converter = s["~standard"]?.jsonSchema;
  if (converter && typeof converter.input === "function") return converter.input({ target: "draft-2020-12" });
  if (typeof s.toJSONSchema === "function") return s.toJSONSchema();
  const vendor = (schema as { "~standard"?: { vendor?: string } })["~standard"]?.vendor ?? "unknown";
  throw new SchemaError(
    `schema from ${JSON.stringify(vendor)} has no JSON Schema export; pass a schema that implements ` +
      "Standard JSON Schema (`~standard.jsonSchema`) or has `toJSONSchema()`",
  );
}

/** Run `~standard.validate` on a decided value, so the static type is honest at runtime. */
export async function validateDecision<S extends StandardSchemaV1>(
  schema: S,
  value: unknown,
): Promise<StandardSchemaV1.InferOutput<S>> {
  const result = await schema["~standard"].validate(value);
  if (result.issues) {
    const detail = result.issues
      .map(
        (i) =>
          `${(i.path ?? []).map((p) => String(typeof p === "object" ? p.key : p)).join(".") || "(root)"}: ${i.message}`,
      )
      .join("; ");
    throw new SchemaError(`decided value fails the schema: ${detail}`);
  }
  return result.value as StandardSchemaV1.InferOutput<S>;
}

/**
 * Options of the typed `decide`. `minConfidence` is left out: an abstained field decides to
 * `null`, which a non-nullable schema output cannot hold (use `raw.decide` for abstention).
 */
export type DecideCallOptions = Omit<PredictOptions, "minConfidence">;

type Decider = { decide(state: unknown, schema?: unknown, opts?: PredictOptions): Promise<Record<string, unknown>> };

async function typedDecide<S extends StandardSchemaV1>(
  runner: Decider,
  state: unknown,
  schema: S,
  opts?: DecideCallOptions,
): Promise<StandardSchemaV1.InferOutput<S>> {
  return validateDecision(schema, await runner.decide(state, jsonSchemaOf(schema), opts));
}

type BatchDecider = {
  decideBatch(states: unknown[], schema?: unknown, opts?: object): Promise<Record<string, unknown>[]>;
};

async function typedDecideBatch<S extends StandardSchemaV1>(
  runner: BatchDecider,
  states: readonly unknown[],
  schema: S,
  opts?: object,
): Promise<StandardSchemaV1.InferOutput<S>[]> {
  const values = await runner.decideBatch(states as unknown[], jsonSchemaOf(schema), opts);
  return Promise.all(
    values.map((v, i) =>
      validateDecision(schema, v).catch((e: unknown) => {
        throw e instanceof SchemaError ? new SchemaError(`state ${i}: ${e.message}`) : e;
      }),
    ),
  );
}

// ---- Factories -----------------------------------------------------------------------------

export type { CheckpointName, Precision };

export interface CheckpointInfo {
  repo: string;
  subfolder: string;
  revision: string;
  source: { repo: string; revision: string };
  sha256: Record<string, string>;
}

/** Where each checkpoint's fp32 bundle lives in the private artifact store, pinned by SHA-256. */
export const CHECKPOINTS: Record<CheckpointName, CheckpointInfo> = Object.fromEntries(
  (Object.keys(ARTIFACTS) as CheckpointName[]).map((name) => {
    const a = ARTIFACTS[name].fp32;
    return [
      name,
      { repo: ARTIFACT_REPO, subfolder: a.subfolder, revision: ARTIFACT_REVISION, source: a.source, sha256: a.sha256 },
    ];
  }),
) as Record<CheckpointName, CheckpointInfo>;

export interface CreateAgentOptions {
  checkpoint: CheckpointName;
  /** Only fp32 is built; INT8 is deferred (plan, 2026-09-28). */
  precision?: Precision;
  /** Load a local bundle dir (`encoder.onnx`, `head.onnx`, `tokenizer.json`, `rl_agent_config.json`). */
  modelDir?: string;
  /** Artifact-store revision. The pinned SHA-256 map is only enforced at the pinned revision. */
  revision?: string;
  /** Read token for the private artifact store; defaults to `HF_TOKEN`. */
  token?: string;
  numThreads?: number;
  /** Verify the bundle's SHA-256 against `artifacts.ts` (default: true at the pinned revision). */
  verify?: boolean;
}

export interface LayaAgent {
  readonly checkpoint: CheckpointName;
  readonly precision: Precision;
  /** Artifact-store commit the bundle came from; null for a local `modelDir`. */
  readonly revision: string | null;
  /** The untyped vendored Agent, as an escape hatch. */
  readonly raw: Agent;
  /** Answer `questions` for one state. `maxLen`/`headMaxLen` set the budget; `minConfidence` flags abstentions. */
  predict<const Q extends QuestionMap>(state: unknown, questions: Q, opts?: PredictOptions): Promise<PredictResult<Q>>;
  /** Answer the same `questions` for many states in shared forward passes; results keep input order. */
  predictBatch<const Q extends QuestionMap>(
    states: readonly unknown[],
    questions: Q,
    opts?: PredictBatchOptions,
  ): Promise<PredictResult<Q>[]>;
  /**
   * Answer a state longer than the context window: score overlapping windows and aggregate per
   * question (noul: max P(true); choice/score: the most confident window).
   */
  predictLong<const Q extends QuestionMap>(
    state: unknown,
    questions: Q,
    opts?: PredictLongOptions,
  ): Promise<LongPredictResult<Q>>;
  decide<S extends StandardSchemaV1>(
    state: unknown,
    schema: S,
    opts?: DecideCallOptions,
  ): Promise<StandardSchemaV1.InferOutput<S>>;
  /** `decide` over many states in shared forward passes; one validated output per state, in input order. */
  decideBatch<S extends StandardSchemaV1>(
    states: readonly unknown[],
    schema: S,
    opts?: Omit<PredictBatchOptions, "minConfidence">,
  ): Promise<StandardSchemaV1.InferOutput<S>[]>;
  /** Release the ONNX sessions. The agent is unusable afterwards. */
  dispose(): Promise<void>;
}

function checkPrecision(precision: string | undefined): Precision {
  const p = precision ?? "fp32";
  if (p !== "fp32") throw new Error(`precision ${JSON.stringify(p)}: only "fp32" is available (INT8 is deferred)`);
  return p;
}

function checkCheckpoint(name: string): CheckpointName {
  if (!(name in ARTIFACTS)) {
    throw new Error(
      `unknown checkpoint ${JSON.stringify(name)}; choose one of ${JSON.stringify(Object.keys(ARTIFACTS))}`,
    );
  }
  return name as CheckpointName;
}

async function loadCheckpoint(
  checkpoint: CheckpointName,
  opts: { modelDir?: string; revision?: string; token?: string; numThreads?: number; verify?: boolean },
): Promise<Agent> {
  const info = CHECKPOINTS[checkpoint];
  if (opts.modelDir) {
    return Agent.load(opts.modelDir, {
      localDir: opts.modelDir,
      numThreads: opts.numThreads,
      expectedSha256: opts.verify ? info.sha256 : undefined,
    });
  }
  const revision = opts.revision ?? info.revision;
  const verify = opts.verify ?? revision === info.revision;
  return Agent.load(info.repo, {
    subfolder: info.subfolder,
    revision,
    token: opts.token,
    numThreads: opts.numThreads,
    expectedSha256: verify ? info.sha256 : undefined,
  });
}

async function releaseAgent(agent: unknown): Promise<void> {
  const provider = (agent as { provider?: SessionProvider } | null)?.provider;
  await provider?.release?.();
}

function wrapAgent(raw: Agent, checkpoint: CheckpointName, precision: Precision): LayaAgent {
  return {
    checkpoint,
    precision,
    revision: raw.revision,
    raw,
    predict: (state, questions, opts) =>
      raw.predict(state, questions as unknown as Record<string, QuestionDef>, opts) as Promise<never>,
    predictBatch: (states, questions, opts) =>
      raw.predictBatch(
        states as unknown[],
        questions as unknown as Record<string, QuestionDef>,
        opts,
      ) as Promise<never>,
    predictLong: (state, questions, opts) =>
      raw.predictLong(state, questions as unknown as Record<string, QuestionDef>, opts) as Promise<never>,
    decide: (state, schema, opts) => typedDecide(raw, state, schema, opts),
    decideBatch: (states, schema, opts) => typedDecideBatch(raw, states, schema, opts),
    dispose: () => releaseAgent(raw),
  };
}

/** Load one checkpoint's fp32 bundle (from the artifact store, or `modelDir`) as a typed agent. */
export async function createAgent(opts: CreateAgentOptions): Promise<LayaAgent> {
  const checkpoint = checkCheckpoint(opts.checkpoint);
  const precision = checkPrecision(opts.precision);
  return wrapAgent(await loadCheckpoint(checkpoint, opts), checkpoint, precision);
}

export interface CreateRouterOptions {
  /** Checkpoints the router may load (default: all three). */
  checkpoints?: CheckpointName[];
  /** LRU cap on loaded checkpoints (default 2, as Python `LAYA_MAX_LOADED`). */
  maxLoaded?: number;
  precision?: Precision;
  /** Route to `typed-decisions` when the question ids match one of its workflows. */
  autoTaskDetection?: boolean;
  /** Local bundle dirs per checkpoint; others come from the artifact store. */
  modelDirs?: Partial<Record<CheckpointName, string>>;
  /** Default checkpoint when routing has no signal. */
  default?: CheckpointName;
  token?: string;
  numThreads?: number;
  /** Load these checkpoints before returning. */
  preload?: CheckpointName[];
}

export interface LayaRouter {
  readonly precision: Precision;
  /** Checkpoints currently held in memory, least recently used first. */
  readonly loaded: string[];
  /** Commit SHA each loaded checkpoint came from (null for a local `modelDir`). */
  readonly loadedRevisions: Readonly<Record<string, string | null>>;
  /** The untyped vendored Router, as an escape hatch. */
  readonly raw: Router;
  predict<const Q extends QuestionMap>(
    state: unknown,
    questions: Q,
    opts?: PredictOptions & { model?: string | null },
  ): Promise<RoutedPredictResult<Q>>;
  decide<S extends StandardSchemaV1>(
    state: unknown,
    schema: S,
    opts?: DecideCallOptions & { model?: string | null },
  ): Promise<StandardSchemaV1.InferOutput<S>>;
  /** Route requests without loading anything; decisions keep input order. */
  routeBatch(requests: readonly TypedRouterRequest[]): RouteDecision[];
  /**
   * Route requests, group them by checkpoint (one load each) and question schema, and run each
   * group as one batched pass. Results keep input order and narrow to each request's questions.
   */
  predictBatch<const R extends readonly TypedRouterRequest[]>(
    requests: R,
    opts?: RouterBatchOptions,
  ): Promise<RoutedBatchResults<R>>;
  /** Route, then scan every window of the state with the routed agent's `predictLong`. */
  predictLong<const Q extends QuestionMap>(
    state: unknown,
    questions: Q,
    opts?: PredictLongOptions & Omit<RouteOptions, "hooks" | "hooksRaise">,
  ): Promise<RoutedLongPredictResult<Q>>;
  /** `decide` over many states; each state routes on its own. */
  decideBatch<S extends StandardSchemaV1>(
    states: readonly unknown[],
    schema: S,
    opts?: Omit<RouterBatchOptions, "minConfidence">,
  ): Promise<StandardSchemaV1.InferOutput<S>[]>;
  /** Release every loaded checkpoint's ONNX sessions. */
  dispose(): Promise<void>;
}

/** A typed Router over the fp32 artifact store: routes each state to a checkpoint, loads lazily (LRU). */
export async function createRouter(opts: CreateRouterOptions = {}): Promise<LayaRouter> {
  const precision = checkPrecision(opts.precision);
  const allowed = (opts.checkpoints ?? (Object.keys(ARTIFACTS) as CheckpointName[])).map(checkCheckpoint);
  const models: Record<string, ModelSpec> = Object.fromEntries(
    allowed.map((name) => [name, { repo: CHECKPOINTS[name].repo, subfolder: CHECKPOINTS[name].subfolder }]),
  );
  const raw = new Router({
    models,
    default: opts.default ?? allowed[0],
    maxLoaded: opts.maxLoaded,
    autoTaskDetection: opts.autoTaskDetection,
    loader: (name: ModelName) => {
      if (!allowed.includes(name as CheckpointName)) {
        throw new Error(
          `routed to ${JSON.stringify(name)}, which is not in this router's checkpoints ${JSON.stringify(allowed)}`,
        );
      }
      return loadCheckpoint(name as CheckpointName, {
        modelDir: opts.modelDirs?.[name],
        token: opts.token,
        numThreads: opts.numThreads,
      });
    },
  });
  if (opts.preload?.length) await raw.preload(opts.preload.map(checkCheckpoint));
  const agents = (raw as unknown as { _agents: Map<string, unknown> })._agents;
  return {
    precision,
    get loaded() {
      return raw.loaded;
    },
    get loadedRevisions() {
      return raw.loadedRevisions;
    },
    raw,
    predict: (state, questions, o) =>
      raw.predict(state, questions as unknown as Record<string, QuestionDef>, o ?? {}) as Promise<never>,
    decide: (state, schema, o) => typedDecide(raw as unknown as Decider, state, schema, o),
    routeBatch: (requests) => raw.routeBatch(requests as unknown as RouterRequest[]),
    predictBatch: (requests, o) => raw.predictBatch(requests as unknown as RouterRequest[], o) as Promise<never>,
    predictLong: (state, questions, o) =>
      raw.predictLong(state, questions as unknown as Record<string, QuestionDef>, o) as Promise<never>,
    decideBatch: (states, schema, o) => typedDecideBatch(raw as unknown as BatchDecider, states, schema, o),
    dispose: async () => {
      const held = [...agents.values()];
      raw.unload();
      await Promise.all(held.map(releaseAgent));
    },
  };
}
