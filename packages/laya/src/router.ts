// Modified by Desplega Labs, 2026: predict forwards per-call maxLen/headMaxLen and flags minConfidence; added routeBatch, predictBatch, predictLong, decideBatch and loadedRevisions (router.py).
import { analyse, type AnalyseResult } from "./lang.js";
import type {
  LongResult,
  PredictBatchOptions,
  PredictLongOptions,
  PredictOptions,
  QuestionDef,
  SystemOneResult,
} from "./agent.js";
import { checkTokenBudget } from "./budget.js";
import { checkMinConfidence, flagLowConfidence } from "./confidence.js";
import { decide, decideBatch, type DecideOptions, type DecisionResult } from "./structured.js";
import {
  HookRegistry,
  PredictContext,
  aggregateUsage,
  composeHooks,
  dispatch,
  markDefaultsRan,
  dispatchAsync,
  normaliseHooks,
  type Hook,
  type HookArg,
  type PredictHook,
} from "./hooks.js";

export const BUNDLE_REPO = "convaiinnovations/laya";

export interface ModelSpec {
  repo: string;
  subfolder: string | null;
}

export const DEFAULT_MODELS: Record<string, ModelSpec> = {
  english: { repo: BUNDLE_REPO, subfolder: null },
  multilingual: { repo: BUNDLE_REPO, subfolder: "multilingual" },
  "typed-decisions": { repo: BUNDLE_REPO, subfolder: "typed-decisions" },
};

export const STANDALONE_MODELS: Record<string, string> = {
  english: "convaiinnovations/laya",
  multilingual: "convaiinnovations/laya-multilingual",
  "typed-decisions": "convaiinnovations/laya-typed-decisions",
};

export type ModelName = "english" | "multilingual" | "typed-decisions";

const ALIASES: Record<string, ModelName> = {
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

export function normaliseName(name: string): ModelName {
  const raw = String(name).trim().toLowerCase();
  const key = ALIASES[raw] ?? raw;
  if (!(key in DEFAULT_MODELS)) {
    throw new Error(
      `unknown model ${JSON.stringify(name)}; choose one of ${JSON.stringify(
        Object.keys(DEFAULT_MODELS).sort(),
      )} (or an alias: ${JSON.stringify(Object.keys(ALIASES).sort())})`,
    );
  }
  return key as ModelName;
}

const TYPED_DECISION_WORKFLOWS: Record<string, Set<string>> = {
  agent_trace_observability: new Set(["action", "needs_review", "outcome", "risk", "urgency"]),
  customer_service: new Set(["action", "category", "churn_risk", "needs_human", "urgency"]),
  invoice_processing: new Set(["discrepancy_severity", "disposition", "duplicate", "matches_order", "urgency"]),
  security_incidents: new Set(["credential_compromise", "disposition", "severity", "true_positive", "urgency"]),
};

export function matchTypedDecisionsWorkflow(
  questions: Record<string, unknown> | null | undefined,
): string | null {
  const ids = new Set(Object.keys(questions ?? {}));
  for (const [wf, sig] of Object.entries(TYPED_DECISION_WORKFLOWS)) {
    if (sig.size === ids.size && [...sig].every((id) => ids.has(id))) return wf;
  }
  return null;
}

const ENGLISH_SUBTAGS = new Set(["en", "eng", "english"]);

// Valid `$LANG` values that name no language, so they answer nothing about the state: `C`,
// `POSIX` and `C.UTF-8` (the official Python image's default), plus the ISO 639-2 special codes
// `und` (undetermined), `zxx` (no linguistic content) and `mul` (multiple). They abstain like a
// blank code instead of forcing the multilingual checkpoint on English text (Python parity).
const LANGUAGE_AGNOSTIC_CODES = new Set(["c", "posix", "und", "zxx", "mul"]);

export function englishFromCode(value: unknown): boolean | null {
  if (value === null || value === undefined) return null;
  let code = String(value).trim().toLowerCase();
  if (!code) return null;
  code = code.split(".", 1)[0]; // en_US.UTF-8 -> en_US
  const primary = code.replace(/_/g, "-").split("-", 1)[0]; // en_US -> en
  if (!primary || LANGUAGE_AGNOSTIC_CODES.has(primary)) return null;
  return ENGLISH_SUBTAGS.has(primary);
}

/** Parity alias for the Python `_english_from_code` name. */
export const _englishFromCode = englishFromCode;

export interface RouteDecision {
  model: ModelName;
  repo: string;
  reason: string;
  detection: AnalyseResult | null;
  workflow: string | null;
}

export type RoutedResult = SystemOneResult & { routing: RouteDecision };
export type RoutedLongResult = LongResult & { routing: RouteDecision };

/** One request of `Router.routeBatch` / `Router.predictBatch`: a state, its questions, and optional overrides. */
export interface RouterRequest {
  state: unknown;
  questions: Record<string, QuestionDef>;
  model?: string | null;
  task?: string | null;
  lang?: string | null;
  langGuess?: LangGuess;
  lang_guess?: LangGuess;
  /** Per-request token budget; requests with different budgets run in separate forward passes. */
  maxLen?: number | null;
  headMaxLen?: number | null;
}

/** Options of `Router.predictBatch`. */
export interface RouterBatchOptions {
  /** States per forward pass within each checkpoint/question group. */
  batchSize?: number | null;
  minConfidence?: number | null;
  /** Forwarded to every `Agent.predictBatch` call. */
  sortByLength?: boolean;
}

type BatchAgent = {
  predictBatch(states: unknown[], questions: Record<string, QuestionDef>, opts?: PredictBatchOptions): Promise<SystemOneResult[]>;
  langTemperatures?: Record<string, unknown>;
};

/** A language forwarded to the agent: an explicit `lang`, else the non-English language routing detected. */
function forwardedLang(lang: string | null | undefined, decision: unknown): string | null {
  if (lang !== null && lang !== undefined) return lang;
  const detected = (decision as { detection?: { language?: string | null } | null } | null)?.detection?.language;
  return detected && detected !== "en" ? detected : null;
}

function typeName(v: unknown): string {
  if (v === null) return "null";
  if (Array.isArray(v)) return "Array";
  return typeof v;
}

/**
 * Last start hook of `Router.predictLong` (router.py `_ScanLong`): scan the routed state with the
 * agent's `predictLong` instead of letting `predict` score one window. Appended after every other
 * start hook, so one that answered or rewrote the state wins.
 */
function scanLongHook(opts: PredictLongOptions): Hook {
  return {
    async onPredictStart(ctx) {
      if (ctx.results !== null) return;
      const agent = ctx.agent as { predictLong?: (...a: unknown[]) => Promise<unknown> } | null;
      if (typeof agent?.predictLong !== "function") {
        throw new TypeError(
          `${(agent as object | null)?.constructor?.name ?? "agent"} has no predictLong, so a state longer than its window cannot be scanned`,
        );
      }
      const result = await agent.predictLong(ctx.states[0], ctx.questions, {
        window: opts.window,
        stride: opts.stride,
        aggregate: opts.aggregate,
        batchSize: opts.batchSize,
        lang: forwardedLang(opts.lang, ctx.decision),
      });
      ctx.results = [result as Record<string, unknown>];
    },
  };
}

export type LangGuess = string | null | undefined | ((state: unknown) => unknown);

export type AgentLoader = (name: ModelName, spec: ModelSpec) => unknown | Promise<unknown>;

export interface RouterOptions {
  models?: Record<string, string | ModelSpec | [string, string | null]>;
  device?: string | null;
  token?: string | null;
  maxLoaded?: number;
  max_loaded?: number;
  default?: string;
  autoTaskDetection?: boolean;
  auto_task_detection?: boolean;
  standaloneRepos?: boolean;
  standalone_repos?: boolean;
  preload?: boolean | string[];
  langGuess?: LangGuess;
  lang_guess?: LangGuess;
  loader?: AgentLoader;
  /** Optional hub revision (commit SHA/branch/tag) applied to every checkpoint load. */
  revision?: string | null;
  /** Per-model revision overrides, keyed by model name or alias. */
  revisions?: Record<string, string | null>;
  hooks?: HookArg;
  onPredictStart?: PredictHook;
  onPredictEnd?: PredictHook;
  hooksRaise?: boolean;
}

export interface RouteOptions {
  model?: string | null;
  task?: string | null;
  lang?: string | null;
  langGuess?: LangGuess;
  lang_guess?: LangGuess;
  hooks?: HookArg;
  hooksRaise?: boolean;
}

function toSpec(spec: string | ModelSpec | [string, string | null]): ModelSpec {
  if (typeof spec === "string") return { repo: spec, subfolder: null };
  if (Array.isArray(spec)) {
    const [repo, sub] = [...spec, null].slice(0, 2) as [string, string | null];
    return { repo, subfolder: sub ?? null };
  }
  return { repo: spec.repo, subfolder: spec.subfolder ?? null };
}

function repoStr(spec: ModelSpec): string {
  return spec.subfolder ? `${spec.repo}/${spec.subfolder}` : spec.repo;
}

export class Router extends HookRegistry {
  hooksRaise: boolean;
  models: Record<string, ModelSpec>;
  device: string | null;
  token: string | null | undefined;
  revision: string | null;
  revisions: Partial<Record<ModelName, string | null>>;
  maxLoaded: number;
  default: ModelName;
  autoTaskDetection: boolean;
  langGuess: LangGuess;
  loader: AgentLoader | null;
  _agents: Map<string, unknown> = new Map();
  _order: string[] = []; // least-recently-used first
  private readonly _loading = new Map<string, Promise<unknown>>();

  constructor(opts: RouterOptions = {}) {
    super();
    // Hooks are opt-in; an unset hook list is a no-op. Router-level onPredictStart /
    // onPredictEnd hooks wrap the whole route+infer call and see ctx.decision; see hooks.ts.
    this.hooks = normaliseHooks(opts.hooks, opts.onPredictStart, opts.onPredictEnd);
    this.hooksRaise = opts.hooksRaise ?? true;
    const base: Record<string, string | ModelSpec> = opts.standaloneRepos ?? opts.standalone_repos
      ? { ...STANDALONE_MODELS }
      : Object.fromEntries(Object.entries(DEFAULT_MODELS).map(([k, v]) => [k, { ...v }]));
    this.models = Object.fromEntries(Object.entries(base).map(([k, v]) => [k, toSpec(v)]));
    if (opts.models) {
      for (const [k, v] of Object.entries(opts.models)) {
        this.models[normaliseName(k)] = toSpec(v);
      }
    }
    this.device = opts.device ?? null;
    this.token = opts.token ?? (typeof process !== "undefined" ? process.env?.["HF_TOKEN"] : undefined);
    // Optional hub revision applied to every checkpoint load. Per-model overrides support
    // standalone repositories whose reviewed commits differ.
    this.revision = opts.revision ?? null;
    this.revisions = Object.fromEntries(
      Object.entries(opts.revisions ?? {}).map(([name, value]) => [normaliseName(name), value]),
    ) as Partial<Record<ModelName, string | null>>;
    this.maxLoaded = Math.max(1, Math.trunc(Number(opts.maxLoaded ?? opts.max_loaded ?? 2)));
    this.default = normaliseName(opts.default ?? "english");
    this.autoTaskDetection = Boolean(opts.autoTaskDetection ?? opts.auto_task_detection ?? false);
    this.langGuess = opts.langGuess ?? opts.lang_guess ?? null;
    this.loader = opts.loader ?? null;
    if (opts.preload === true) {
      void this.preload();
    } else if (Array.isArray(opts.preload)) {
      void this.preload(opts.preload);
    }
  }

  async load(name: string): Promise<unknown> {
    const key = normaliseName(name);
    if (this._agents.has(key)) {
      this._touch(key);
      return this._agents.get(key);
    }
    const loading = this._loading.get(key);
    if (loading) return loading;
    // Start in a microtask so even a synchronous loader sees its in-flight entry.
    const pending = Promise.resolve().then(async () => {
      let agent: unknown;
      if (this.loader) {
        agent = await this.loader(key, this.models[key]);
      } else {
        const { Agent } = await import("./agent.js");
        const spec = this.models[key];
        const revision = Object.prototype.hasOwnProperty.call(this.revisions, key)
          ? this.revisions[key]
          : this.revision;
        const opts: Record<string, unknown> = {
          subfolder: spec.subfolder,
          device: this.device ?? undefined,
          token: this.token ?? undefined,
        };
        if (revision) opts.revision = revision;
        agent = await (Agent as unknown as {
          load(repo: string, opts?: Record<string, unknown>): Promise<unknown>;
        }).load(spec.repo, opts);
      }
      this._agents.set(key, agent);
      this._order.push(key);
      const evicted = this._evict();
      // Lifecycle hooks fire after the maps settle, so a hook can safely call the Router.
      for (const victim of evicted) {
        await dispatchAsync(
          composeHooks(this.hooks),
          "onEvict",
          new PredictContext({ states: [], questions: {}, model: victim, router: this }),
          { raiseErrors: this.hooksRaise },
        );
      }
      await dispatchAsync(
        composeHooks(this.hooks),
        "onLoad",
        new PredictContext({ states: [], questions: {}, model: key, agent, router: this }),
        { raiseErrors: this.hooksRaise },
      );
      return agent;
    });
    this._loading.set(key, pending);
    try {
      return await pending;
    } finally {
      this._loading.delete(key);
    }
  }

  _touch(key: string): void {
    const i = this._order.indexOf(key);
    if (i !== -1) this._order.splice(i, 1);
    this._order.push(key);
  }

  /** Drop least-recently-used agents until `maxLoaded` holds. Returns evicted names. */
  _evict(): string[] {
    const evicted: string[] = [];
    while (this._order.length > this.maxLoaded) {
      const victim = this._order.shift()!;
      if (this._agents.delete(victim)) evicted.push(victim);
    }
    // Keep the two views consistent.
    if (this._order.length < this._agents.size) {
      for (const k of [...this._agents.keys()]) {
        if (!this._order.includes(k)) {
          this._agents.delete(k);
          evicted.push(k);
        }
      }
    }
    return evicted;
  }

  attach(name: string, agent: unknown): unknown {
    const key = normaliseName(name);
    this._agents.set(key, agent);
    this._touch(key);
    this.maxLoaded = Math.max(this.maxLoaded, this._agents.size);
    return agent;
  }

  async preload(names?: string[]): Promise<this> {
    const keys = (names ?? Object.keys(this.models)).map((n) => normaliseName(n));
    this.maxLoaded = Math.max(this.maxLoaded, new Set([...keys, ...this._agents.keys()]).size);
    for (const n of keys) {
      if (!this._agents.has(n)) await this.load(n);
    }
    return this;
  }

  unload(name?: string | null): void {
    if (name === null || name === undefined) {
      this._agents.clear();
      this._order = [];
    } else {
      const key = normaliseName(name);
      this._agents.delete(key);
      const i = this._order.indexOf(key);
      if (i !== -1) this._order.splice(i, 1);
    }
  }

  get loaded(): string[] {
    return [...this._order];
  }

  /** Commit SHA each resident agent was loaded from (null for local paths). */
  get loadedRevisions(): Record<string, string | null> {
    return Object.fromEntries(
      [...this._agents].map(([name, agent]) => [name, (agent as { revision?: string | null } | null)?.revision ?? null]),
    );
  }

  _resolveHint(hint: LangGuess, state: unknown): boolean | null {
    if (hint === null || hint === undefined) return null;
    const value = typeof hint === "function" ? (hint as (s: unknown) => unknown)(state) : hint;
    return englishFromCode(value);
  }

  /**
   * Decide which checkpoint to use, then let `onRoute` hooks observe or replace the decision.
   *
   * `ctx.decision` is the RouteDecision; a hook may replace it (for example to pin a
   * checkpoint) and the replacement is what gets returned and used. `opts.hooks` are
   * per-call hooks, appended after any installed on the Router.
   */
  route(
    state: unknown,
    questions: Record<string, unknown> | null = null,
    opts: RouteOptions = {},
  ): RouteDecision {
    const decision = this._route(state, questions, opts);
    const raiseErrors = opts.hooksRaise ?? this.hooksRaise;
    const active = composeHooks(this.hooks, opts.hooks);
    const ctx = new PredictContext({
      states: [state],
      questions: (questions ?? {}) as Record<string, unknown>,
      decision: decision as unknown as Record<string, unknown>,
      router: this,
    });
    dispatch(active, "onRoute", ctx, { raiseErrors });
    return ctx.decision as unknown as RouteDecision;
  }

  /** Decide which checkpoint to use, without loading, running, or hooking anything. */
  _route(
    state: unknown,
    questions: Record<string, unknown> | null = null,
    opts: RouteOptions = {},
  ): RouteDecision {
    const { model = null, task = null, lang = null } = opts;
    const langGuessOpt = opts.langGuess ?? opts.lang_guess ?? null;

    if (model !== null && model !== undefined) {
      const key = normaliseName(model);
      return {
        model: key,
        repo: repoStr(this.models[key]),
        reason: `explicit model=${JSON.stringify(model)}`,
        detection: null,
        workflow: null,
      };
    }

    if (task !== null && task !== undefined) {
      const key = normaliseName(task);
      return {
        model: key,
        repo: repoStr(this.models[key]),
        reason: `explicit task=${JSON.stringify(task)}`,
        detection: null,
        workflow: null,
      };
    }

    const workflow = matchTypedDecisionsWorkflow(questions ?? {});
    if (workflow && this.autoTaskDetection) {
      return {
        model: "typed-decisions",
        repo: repoStr(this.models["typed-decisions"]),
        reason: `question ids match the ${JSON.stringify(workflow)} typed-decisions workflow`,
        detection: null,
        workflow,
      };
    }

    // An explicit `lang` is decisive only when the code names a language. Blank or whitespace
    // resolves to no usable hint, so it falls through to langGuess/detection exactly as an
    // abstaining hint does (Python parity); real English/non-English codes still route now.
    const resolvedLang = englishFromCode(lang);
    if (resolvedLang !== null) {
      const key: ModelName = resolvedLang ? "english" : "multilingual";
      return {
        model: key,
        repo: repoStr(this.models[key]),
        reason: `explicit lang=${JSON.stringify(lang)}`,
        detection: null,
        workflow,
      };
    }

    const hints: Array<[string, LangGuess]> = [
      ["lang_guess", langGuessOpt],
      ["Router(lang_guess=...)", this.langGuess],
    ];
    for (const [source, hint] of hints) {
      const resolved = this._resolveHint(hint, state);
      if (resolved !== null && resolved !== undefined) {
        const key: ModelName = resolved ? "english" : "multilingual";
        return {
          model: key,
          repo: repoStr(this.models[key]),
          reason: `${source}: the caller identified this as ${resolved ? "English" : "non-English"} text`,
          detection: null,
          workflow,
        };
      }
    }

    const det = analyse(state);
    let key: ModelName;
    let reason: string;
    if (det.script === "unknown") {
      key = this.default;
      reason = `no letters detected in state; using default (${key})`;
    } else if (det.script !== "latin") {
      key = "multilingual";
      reason =
        `non-Latin script (${det.script}, ${Math.round(100 * det.nonLatinFraction)}% of letters); ` +
        "the English checkpoint cannot read it";
    } else if (!det.isEnglish) {
      key = "multilingual";
      if (det.mixedSegment) {
        reason =
          `Latin script, mostly English, but a line or field reads as ${JSON.stringify(det.language)} ` +
          `(${JSON.stringify(det.mixedSegment.slice(0, 60))}); the English checkpoint cannot read it`;
      } else if (det.language) {
        reason = `Latin script but language looks like ${JSON.stringify(det.language)}, not English`;
      } else {
        reason =
          `Latin script, language not identified but ${Math.round(100 * det.diacriticRate)}% ` +
          "non-English letters; not safe for the English checkpoint";
      }
    } else if (det.languageUndecided) {
      key = this.default;
      reason = `Latin script, language not identified and no non-English letters; using default (${key})`;
    } else {
      key = "english";
      reason = "English Latin text";
    }
    return { model: key, repo: repoStr(this.models[key]), reason, detection: det, workflow };
  }

  /**
   * Route, then answer every question in one forward pass on the chosen checkpoint.
   *
   * The result is the usual systemOne payload plus a `routing` key recording the decision.
   * Router-level `onPredictStart` / `onPredictEnd` hooks wrap the whole route+infer call and
   * see `ctx.decision`; see hooks.ts.
   */
  async predict(
    state: unknown,
    questions: Record<string, QuestionDef>,
    opts: RouteOptions & PredictOptions = {},
  ): Promise<RoutedResult> {
    const mc = opts.minConfidence !== null && opts.minConfidence !== undefined
      ? checkMinConfidence(opts.minConfidence)
      : null;
    checkTokenBudget("maxLen", opts.maxLen);
    checkTokenBudget("headMaxLen", opts.headMaxLen);
    const active = composeHooks(this.hooks, opts.hooks, opts.onPredictStart, opts.onPredictEnd);
    const raiseErrors = opts.hooksRaise ?? this.hooksRaise;

    // Per-call hooks apply to the whole call, including onRoute inside route().
    const decision = this.route(state, questions, opts);
    const agent = (await this.load(decision.model)) as {
      systemOne(
        s: unknown,
        q: Record<string, QuestionDef>,
        opts?: PredictOptions,
      ): Promise<SystemOneResult>;
    };
    const ctx = new PredictContext({
      states: [state],
      questions: questions as Record<string, unknown>,
      decision: { ...decision } as unknown as Record<string, unknown>,
      model: decision.model,
      agent,
      router: this,
      maxLen: opts.maxLen ?? null,
      headMaxLen: opts.headMaxLen ?? null,
    });
    try {
      await dispatchAsync(active, "onPredictStart", ctx, { raiseErrors });
      if (ctx.results === null) {
        // Python parity (router.py predict): the request's language also shapes the answer
        // distribution through the agent's lang_temperatures. An explicit lang wins;
        // otherwise forward the language the router detected for the routing decision.
        // TS analyse() names English "en" where Python's analyse returns None (it only
        // ever names non-English), so a detected "en" forwards as null — in Python only an
        // explicit lang="en" can select an "en" override.
        const detected = decision.detection?.language;
        const effectiveLang = opts.lang ?? (detected && detected !== "en" ? detected : null);
        // Budget overrides only when set, so an Agent-like object that does not take them still works.
        const agentOpts: PredictOptions = { lang: effectiveLang };
        if (ctx.maxLen !== null) agentOpts.maxLen = ctx.maxLen;
        if (ctx.headMaxLen !== null) agentOpts.headMaxLen = ctx.headMaxLen;
        markDefaultsRan(agentOpts);
        const result = (await agent.systemOne(
          ctx.states[0],
          ctx.questions as Record<string, QuestionDef>,
          agentOpts,
        )) as RoutedResult;
        result["routing"] = { ...decision };
        ctx.results = [result as unknown as Record<string, unknown>];
      } else {
        // A cache hit short-circuits inference, but predict still promises a `routing` key.
        // Add it without overwriting a routing the cached payload already has.
        for (const result of ctx.results) {
          if (result && typeof result === "object" && !("routing" in result)) {
            (result as unknown as RoutedResult).routing = { ...decision };
          }
        }
      }
    } catch (err) {
      ctx.error = err;
      try {
        await dispatchAsync(active, "onError", ctx, { raiseErrors });
      } catch {
        // A failing onError hook must not hide the failure that triggered it.
      }
      throw err;
    } finally {
      ctx.markElapsed();
      if (ctx.results !== null) {
        ctx.usage = aggregateUsage(ctx.results);
        if (mc !== null) flagLowConfidence(ctx.results, mc);
      }
      try {
        await dispatchAsync(active, "onPredictEnd", ctx, { raiseErrors });
      } catch (hookErr) {
        // End hooks run on the failure path too; do not let one mask the real error.
        if (ctx.error === null) throw hookErr;
      }
    }
    return (ctx.results as unknown as RoutedResult[])[0];
  }

  /**
   * Answer `state` against a JSON schema (or explicit `opts.questions`) and return typed
   * values — see `structured.ts`. Routing options (`model`, `task`, ...) are forwarded to
   * `predict`.
   */
  async decide(
    state: unknown,
    schema: unknown,
    opts: DecideOptions & RouteOptions & PredictOptions & { returnDetails: true },
  ): Promise<DecisionResult>;
  async decide(
    state: unknown,
    schema?: unknown,
    opts?: DecideOptions & RouteOptions & PredictOptions,
  ): Promise<Record<string, unknown>>;
  async decide(
    state: unknown,
    schema?: unknown,
    opts: DecideOptions & RouteOptions & PredictOptions = {},
  ): Promise<Record<string, unknown> | DecisionResult> {
    return decide(this, state, schema, opts);
  }

  async systemOne(
    state: unknown,
    questions: Record<string, QuestionDef>,
    opts: RouteOptions & PredictOptions = {},
  ): Promise<RoutedResult> {
    return this.predict(state, questions, opts);
  }

  /**
   * Route, then scan every window of the state with the routed agent's `predictLong`, instead of
   * scoring only its first window. Routing, router-level hooks and the `routing` key are
   * `predict`'s; a caller's start hook that answers or rewrites the state wins over the scan.
   */
  async predictLong(
    state: unknown,
    questions: Record<string, QuestionDef>,
    opts: RouteOptions & PredictLongOptions = {},
  ): Promise<RoutedLongResult> {
    const perCall = normaliseHooks(opts.hooks, opts.onPredictStart, opts.onPredictEnd);
    perCall.push(scanLongHook(opts));
    return (await this.predict(state, questions, {
      model: opts.model,
      task: opts.task,
      lang: opts.lang,
      langGuess: opts.langGuess ?? opts.lang_guess,
      hooks: perCall,
      hooksRaise: opts.hooksRaise,
    })) as unknown as RoutedLongResult;
  }

  /** The batched `decide`: one `{ state, questions }` request per state through `predictBatch`. */
  async decideBatch(
    states: unknown[],
    schema: unknown,
    opts: DecideOptions & RouterBatchOptions & { returnDetails: true },
  ): Promise<DecisionResult[]>;
  async decideBatch(
    states: unknown[],
    schema?: unknown,
    opts?: DecideOptions & RouterBatchOptions,
  ): Promise<Record<string, unknown>[]>;
  async decideBatch(
    states: unknown[],
    schema?: unknown,
    opts: DecideOptions & RouterBatchOptions = {},
  ): Promise<Array<Record<string, unknown> | DecisionResult>> {
    return decideBatch(this as never, states, schema, opts);
  }

  /**
   * Route a heterogeneous request batch without loading any checkpoint. Each request carries
   * `state` and `questions` plus `route`'s optional overrides; decisions keep input order.
   */
  routeBatch(requests: RouterRequest[]): RouteDecision[] {
    if (!Array.isArray(requests)) throw new TypeError("requests must be an array of request objects");
    return requests.map((request, i) => {
      if (typeof request !== "object" || request === null || Array.isArray(request)) {
        throw new TypeError(`request ${i} must be an object, got ${typeName(request)}`);
      }
      if (!("state" in request)) throw new Error(`request ${i} is missing required key 'state'`);
      if (!("questions" in request)) throw new Error(`request ${i} is missing required key 'questions'`);
      const q: unknown = request.questions;
      if (typeof q !== "object" || q === null || Array.isArray(q)) {
        throw new TypeError(`request ${i} 'questions' must be an object, got ${typeName(q)}`);
      }
      return this.route(request.state, request.questions, {
        model: request.model,
        task: request.task,
        lang: request.lang,
        langGuess: request.langGuess ?? request.lang_guess,
      });
    });
  }

  /**
   * Route and run a heterogeneous request batch with minimal model churn (router.py
   * `predict_batch`). Requests are routed, grouped by checkpoint (one load per checkpoint), then
   * split again by question schema, token budget and language so each group is one
   * `Agent.predictBatch`. Results keep input order.
   *
   * Router-level predict hooks run per request, as `predict` runs them: each request gets its
   * own context, so a start hook can rewrite or skip it before it joins a shared pass. A
   * checkpoint group's requests end in reverse of the order they started; if the group fails,
   * each started request gets `onError` and then `onPredictEnd` before the error propagates.
   */
  async predictBatch(requests: RouterRequest[], opts: RouterBatchOptions = {}): Promise<RoutedResult[]> {
    const mc = opts.minConfidence !== null && opts.minConfidence !== undefined
      ? checkMinConfidence(opts.minConfidence)
      : null;
    const decisions = this.routeBatch(requests);
    if (decisions.length === 0) return [];
    for (const r of requests) {
      checkTokenBudget("maxLen", r.maxLen);
      checkTokenBudget("headMaxLen", r.headMaxLen);
    }
    // Insertion order keeps loads deterministic: at most one load per routed checkpoint.
    const groups = new Map<ModelName, number[]>();
    decisions.forEach((decision, i) => {
      // An onRoute hook may name the checkpoint by an alias; it must still share its group.
      const key = normaliseName((decision as { model: string }).model);
      groups.set(key, [...(groups.get(key) ?? []), i]);
    });
    const results: RoutedResult[] = new Array(requests.length);
    let answered = 0;
    const active = composeHooks(this.hooks);
    const raiseErrors = this.hooksRaise;

    for (const [modelName, indices] of groups) {
      const agent = (await this.load(modelName)) as BatchAgent;
      const started: PredictContext[] = [];
      try {
        for (const i of indices) {
          const ctx = new PredictContext({
            states: [requests[i].state],
            questions: requests[i].questions as Record<string, unknown>,
            decision: { ...decisions[i] } as unknown as Record<string, unknown>,
            model: modelName,
            agent,
            router: this,
            maxLen: requests[i].maxLen ?? null,
            headMaxLen: requests[i].headMaxLen ?? null,
          });
          started.push(ctx);
          await dispatchAsync(active, "onPredictStart", ctx, { raiseErrors });
        }

        // One Agent.predictBatch carries one question schema, budget and language, so split
        // again on what the start hooks left (order-sensitive schema key: options are positional).
        const questionGroups: {
          questions: Record<string, QuestionDef>;
          key: string;
          lang: string | null;
          overrides: PredictBatchOptions;
          items: [number, PredictContext][];
        }[] = [];
        indices.forEach((i, j) => {
          const ctx = started[j];
          if (ctx.results !== null) {
            for (const result of ctx.results) {
              if (result && typeof result === "object" && !("routing" in result)) {
                (result as unknown as RoutedResult).routing = { ...decisions[i] };
              }
            }
            return;
          }
          const overrides: PredictBatchOptions = {};
          if (ctx.maxLen !== null) overrides.maxLen = ctx.maxLen;
          if (ctx.headMaxLen !== null) overrides.headMaxLen = ctx.headMaxLen;
          // The language only splits a group for an agent that has per-language temperatures.
          const lang = agent.langTemperatures && Object.keys(agent.langTemperatures).length > 0
            ? forwardedLang(requests[i].lang, decisions[i])
            : null;
          const key = JSON.stringify([ctx.questions, overrides.maxLen ?? null, overrides.headMaxLen ?? null, lang]);
          const group = questionGroups.find((g) => g.key === key);
          if (group) group.items.push([i, ctx]);
          else {
            questionGroups.push({
              questions: ctx.questions as Record<string, QuestionDef>,
              key,
              lang,
              overrides,
              items: [[i, ctx]],
            });
          }
        });

        for (const group of questionGroups) {
          const batchOpts: PredictBatchOptions = { ...group.overrides, batchSize: opts.batchSize };
          if (group.lang !== null) batchOpts.lang = group.lang;
          if (opts.sortByLength) batchOpts.sortByLength = true;
          markDefaultsRan(batchOpts);
          const batchResults = await agent.predictBatch(
            group.items.map(([, ctx]) => ctx.states[0]),
            group.questions,
            batchOpts,
          );
          if (!Array.isArray(batchResults) || batchResults.length !== group.items.length) {
            throw new Error(
              `internal error: Agent.predictBatch returned ${batchResults?.length} results for ${group.items.length} states`,
            );
          }
          group.items.forEach(([i, ctx], k) => {
            const result = batchResults[k] as RoutedResult;
            result.routing = { ...decisions[i] };
            ctx.results = [result as unknown as Record<string, unknown>];
          });
        }
        // Summed here, not while ending, so a malformed usage block fails the group like any error.
        const usages = started.map((ctx) => aggregateUsage(ctx.results ?? []));
        started.forEach((ctx, k) => {
          ctx.usage = usages[k];
        });
      } catch (err) {
        for (const ctx of started) if (mc !== null && ctx.results) flagLowConfidence(ctx.results, mc);
        await this._endContexts(active, started, raiseErrors, { error: err });
        throw err;
      }
      for (const ctx of started) if (mc !== null && ctx.results) flagLowConfidence(ctx.results, mc);
      await this._endContexts(active, started, raiseErrors, null);
      indices.forEach((i, j) => {
        results[i] = (started[j].results as unknown as RoutedResult[])[0];
      });
      answered += indices.length;
    }
    if (answered !== requests.length) throw new Error("internal error: batch execution did not produce every result");
    return results;
  }

  /**
   * End each request of a batch the way `predict` ends one (router.py `_end_contexts`):
   * `elapsedMs` is set on all first; they end in reverse start order; a failed context gets
   * `onError` before `onPredictEnd`; every context ends even if another's hooks raise, and the
   * first such failure on a successful context is thrown afterwards.
   */
  private async _endContexts(
    active: Hook[],
    contexts: PredictContext[],
    raiseErrors: boolean,
    failure: { error: unknown } | null,
  ): Promise<void> {
    for (const ctx of contexts) {
      ctx.markElapsed();
      if (failure) ctx.error = failure.error;
    }
    let first: { error: unknown } | null = null;
    for (const ctx of [...contexts].reverse()) {
      const failed = ctx.error !== null;
      for (const event of failed ? (["onError", "onPredictEnd"] as const) : (["onPredictEnd"] as const)) {
        try {
          await dispatchAsync(active, event, ctx, { raiseErrors });
        } catch (hookErr) {
          if (!failed && first === null) first = { error: hookErr };
        }
      }
    }
    if (first !== null) throw first.error;
  }
}
