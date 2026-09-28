// Modified by Desplega Labs, 2026: Agent.load always takes the Node path (browser branch removed); predictBatch, predictLong, decideBatch, per-call maxLen/headMaxLen and minConfidence (agent.py).
import {
  TEMP_MAX,
  TEMP_MIN,
  buildQuestionPrefix,
  clampTemperature,
  collateItems,
  confidenceFromProbs,
  answerConfidence,
  renderCriterion,
  renderOptions,
  sequenceWithState,
  serializeState,
  softmax,
  tempBucket,
} from "./common.js";
import type { Batch, SessionProvider } from "./providers.js";
import { decodeWithData, encodeWithData, parseTokenizerJson, type TokenizerLike } from "./tokenizer.js";
import { decide, decideBatch, type DecideOptions, type DecisionResult } from "./structured.js";
import {
  HookRegistry,
  PredictContext,
  aggregateUsage,
  composeHooks,
  defaultsAlreadyRan,
  dispatchAsync,
  normaliseHooks,
  type Hook,
  type HookArg,
  type PredictHook,
} from "./hooks.js";
import { checkTokenBudget } from "./budget.js";
import { checkMinConfidence, flagLowConfidence } from "./confidence.js";

export const QTYPES: Record<string, number> = { choice: 0, score: 1, noul: 2 };

export interface QuestionDef {
  type: string;
  instructions?: unknown;
  criteria?: unknown;
  [k: string]: unknown;
}

export interface ActionInfo {
  act_probability: number;
}

export interface ChoiceAnswer {
  type: "choice";
  choice: string;
  probabilities: Record<string, number>;
  confidence: number;
  answer_confidence: number;
  action: ActionInfo;
}

export interface ScoreAnswer {
  type: "score";
  score: number;
  legend: Record<string, string>;
  probabilities: Record<string, number>;
  confidence: number;
  answer_confidence: number;
  action: ActionInfo;
}

export interface NoulAnswer {
  type: "noul";
  noul: number;
  confidence: number;
  answer_confidence: number;
  action: ActionInfo;
}

export type SystemAnswer = ChoiceAnswer | ScoreAnswer | NoulAnswer;

export interface SystemUsage {
  input_tokens: number;
  output_tokens: number;
}

export interface SystemOneResult {
  model: string;
  answers: Record<string, SystemAnswer>;
  usage: SystemUsage;
}

export interface AgentCfg {
  max_len?: number;
  head_max_len?: number;
  temperature?: unknown;
  temperature_by_options?: Record<string, unknown>;
  [k: string]: unknown;
}

export interface AgentOptions {
  provider: SessionProvider;
  tok?: TokenizerLike;
  cfg?: AgentCfg;
  /** Commit SHA the artifacts were loaded from (pinned/requested or `x-repo-commit`); null for local dirs. */
  revision?: string | null;
  max_len?: number;
  head_max_len?: number;
  temperature?: unknown;
  temperature_by_options?: Record<string, unknown>;
  /**
   * Per-language temperature overrides, keyed by language code; keys are normalised to
   * their base subtag (`de-AT` -> `de`), matching Python `Agent(lang_temperatures=...)`.
   * Each entry may carry a `temperature` list of 3 floats (default: the base raw
   * temperature) and/or a `temperature_by_options` map (default: none). A matching
   * override replaces the scale wholesale — see the note at the decode site.
   */
  lang_temperatures?: Record<
    string,
    { temperature?: unknown; temperature_by_options?: Record<string, unknown> } | null
  >;
  hooks?: HookArg;
  onPredictStart?: PredictHook;
  onPredictEnd?: PredictHook;
  hooksRaise?: boolean;
}

/** Per-call options shared by Agent.systemOne/predict and Router.predict. */
export interface PredictOptions {
  /**
   * Language of the request (e.g. "de"); when the Agent has a matching
   * `lang_temperatures` override it selects that language's temperature, exactly like
   * Python `system_one(..., lang=...)`. Routing alone never sets this.
   */
  lang?: string | null;
  hooks?: HookArg;
  onPredictStart?: PredictHook;
  onPredictEnd?: PredictHook;
  hooksRaise?: boolean;
  /**
   * Per-call token budget (Python `max_len`), overriding the checkpoint config; a start hook may
   * set `ctx.maxLen` instead. A positive integer up to `MAX_TOKEN_BUDGET`.
   */
  maxLen?: number | null;
  /** Per-call option-head budget (Python `head_max_len`); a start hook may set `ctx.headMaxLen`. */
  headMaxLen?: number | null;
  /**
   * Opt-in abstention (#361): an answer whose `answer_confidence` is below this threshold comes
   * back with `low_confidence: true`; the raw answer is unchanged. A number in [0, 1].
   */
  minConfidence?: number | null;
}

/** Options of `Agent.predictBatch`: `PredictOptions` plus batching. */
export interface PredictBatchOptions extends PredictOptions {
  /** States per forward pass; unset or non-positive sends all states in one pass. */
  batchSize?: number | null;
  /**
   * Group similarly sized encoded states within windows of eight batches to reduce padding.
   * Needs a `batchSize` above 1 and below the number of states; results keep input order.
   */
  sortByLength?: boolean;
}

/** Options of `Agent.predictLong`. The window, not `maxLen`, sizes what the model reads. */
export interface PredictLongOptions extends Omit<PredictOptions, "maxLen" | "headMaxLen" | "minConfidence"> {
  /** State tokens per window; default `max(64, maxLen - headMaxLen - 8)` from the config. */
  window?: number | null;
  /** Token step between windows; default `floor(window / 2)`. */
  stride?: number | null;
  /** Only "auto" (noul: max P(true); choice/score: the most confident window). */
  aggregate?: "auto";
  /** Windows per forward pass. */
  batchSize?: number | null;
}

/** The deciding window of a `predictLong` answer, as token offsets into the tokenized state. */
export interface WindowInfo {
  index: number;
  token_start: number;
  token_end: number;
  count: number;
}

/** `predictLong` result: `systemOne`'s shape, `answer.window` on scanned answers and `usage.windows`. */
export interface LongResult {
  model: string;
  answers: Record<string, SystemAnswer & { window?: WindowInfo }>;
  usage: SystemUsage & { windows: number };
}

type EncodedItem = { ids: number[]; markers: number[]; qtype: number };
type Internal = ReturnType<typeof toInternal>;

/**
 * Recorder for `predictLong` (agent.py `_start_evidence`): what the start-hook chain left for
 * inference. `answered` is whether a hook skipped inference; `states` snapshots the states that
 * reached it, or stays null when no start chain ran.
 */
function startEvidence(): { probe: Hook; evidence: { answered: boolean; states: unknown[] | null } } {
  const evidence: { answered: boolean; states: unknown[] | null } = { answered: false, states: null };
  const probe: Hook = {
    onPredictStart(ctx) {
      evidence.answered = ctx.results !== null;
      evidence.states = Array.isArray(ctx.states) ? [...ctx.states] : [ctx.states];
    },
  };
  return { probe, evidence };
}

/** The caller's hook options with `probe` after their own start hooks (agent.py `_with_start_probe`). */
function withStartProbe(opts: PredictLongOptions, probe: Hook): PredictOptions {
  return {
    lang: opts.lang,
    hooksRaise: opts.hooksRaise,
    hooks: [...normaliseHooks(opts.hooks, opts.onPredictStart), probe],
    onPredictEnd: opts.onPredictEnd,
  };
}

function typeName(v: unknown): string {
  if (v === null) return "null";
  if (Array.isArray(v)) return "Array";
  return typeof v;
}

function qidStr(qid: string): string {
  return JSON.stringify(qid);
}

export function checkQuestion(qid: string, qdef: unknown): void {
  if (typeof qdef !== "object" || qdef === null || Array.isArray(qdef)) {
    const got = Array.isArray(qdef) ? "list" : qdef === null ? "NoneType" : typeof qdef;
    throw new Error(`question ${qidStr(qid)}: definition must be a dict, got ${got}`);
  }
  const q = qdef as Record<string, unknown>;
  const t = q["type"];
  if (t !== "choice" && t !== "score" && t !== "noul") {
    throw new Error(
      `question ${qidStr(qid)}: unknown type ${JSON.stringify(t)}; use one of ${JSON.stringify(Object.keys(QTYPES).sort())}`,
    );
  }
  if (!("instructions" in q)) {
    throw new Error(`question ${qidStr(qid)}: no 'instructions'; add the text the model should answer`);
  }
  const crit = q["criteria"];
  if (t === "choice") {
    if (typeof crit !== "object" || crit === null) {
      throw new Error(
        `question ${qidStr(qid)}: a choice question takes 'criteria' as a dict of label -> description, or a list of labels`,
      );
    }
    if (Object.keys(crit as object).length === 0) {
      throw new Error(`question ${qidStr(qid)}: a choice question needs at least one criterion`);
    }
    if (Array.isArray(crit)) {
      crit.forEach((label: unknown, i) => {
        if (typeof label !== "object" || label === null) return;
        throw new Error(
          `question ${qidStr(qid)}: choice label ${i} is a ${Array.isArray(label) ? "list" : "dict"}; a label is ` +
            `rendered as option text and used as the answer key, so it must be a scalar (a string, number or ` +
            `null), got ${JSON.stringify(label)}`,
        );
      });
    }
  } else if (t === "score") {
    if (!Array.isArray(crit)) {
      throw new Error(
        `question ${qidStr(qid)}: a score question takes 'criteria' as a list of level descriptions, index 0 first`,
      );
    }
    if (crit.length === 0) {
      throw new Error(`question ${qidStr(qid)}: a score question needs at least one level`);
    }
    const nullAt = crit.findIndex((c) => c === null || c === undefined);
    if (nullAt >= 0) {
      throw new Error(
        `question ${qidStr(qid)}: score level ${nullAt} is null; give every level a description, index 0 first`,
      );
    }
  } else if (crit !== undefined && crit !== null && (typeof crit !== "object" || Array.isArray(crit))) {
    throw new Error(
      `question ${qidStr(qid)}: a noul question takes 'criteria' as a dict with optional 'true'/'false' descriptions, or omits it`,
    );
  } else if (crit && typeof crit === "object" && !Array.isArray(crit)) {
    const invalid = Object.keys(crit as Record<string, unknown>).filter((key) => key.toLowerCase() !== "true" && key.toLowerCase() !== "false");
    if (invalid.length > 0) {
      throw new Error(
        `question ${qidStr(qid)}: noul criteria may contain only 'true' and 'false'; got ${JSON.stringify(invalid)}`,
      );
    }
  }
  if ("labels" in q && t !== "noul") {
    throw new Error(`question ${qidStr(qid)}: 'labels' is only supported for noul questions`);
  }
  if (t === "noul" && q["labels"] !== null && q["labels"] !== undefined) {
    const labels = q["labels"];
    if (typeof labels !== "object" || labels === null || Array.isArray(labels)) {
      throw new Error(`question ${qidStr(qid)}: noul labels must be an object with 'false' and 'true'`);
    }
    const entries = Object.entries(labels as Record<string, unknown>);
    const keys = entries.map(([key]) => key).sort();
    if (keys.length !== 2 || keys[0] !== "false" || keys[1] !== "true" ||
        entries.some(([, value]) => typeof value !== "string" || value.trim() === "")) {
      throw new Error(`question ${qidStr(qid)}: noul labels must map exactly 'false' and 'true' to distinct non-empty strings`);
    }
    const falseLabel = String((labels as Record<string, unknown>)["false"]).trim();
    const trueLabel = String((labels as Record<string, unknown>)["true"]).trim();
    if (falseLabel === trueLabel) {
      throw new Error(`question ${qidStr(qid)}: noul labels must be distinct`);
    }
  }
}

export function toInternal(qdef: QuestionDef): { t: "choice" | "score" | "noul"; ins: string; crit: unknown; labels?: { false: string; true: string } } {
  const t = qdef["type"] as "choice" | "score" | "noul";
  let crit: unknown = qdef["criteria"];
  if (t === "choice" && Array.isArray(crit)) {
    crit = Object.fromEntries(crit.map((c: unknown) => [c as string, null]));
  } else if (t === "noul" && crit !== null && crit !== undefined && typeof crit === "object" && !Array.isArray(crit)) {
    crit = Object.fromEntries(Object.entries(crit as Record<string, unknown>).map(([k, v]) => [String(k).toLowerCase(), v]));
  }
  let ins: unknown = qdef["instructions"];
  if (typeof ins !== "string") ins = serializeState(ins);
  const out: { t: "choice" | "score" | "noul"; ins: string; crit: unknown; labels?: { false: string; true: string } } = {
    t, ins: ins as string, crit,
  };
  if (t === "noul" && qdef["labels"] && typeof qdef["labels"] === "object" && !Array.isArray(qdef["labels"])) {
    const labels = qdef["labels"] as Record<string, unknown>;
    out.labels = { false: String(labels["false"]).trim(), true: String(labels["true"]).trim() };
  }
  return out;
}

export function defaultTokenizer(): TokenizerLike {
  return {
    clsId: 101,
    sepId: 102,
    maskId: 103,
    padId: 0,
    maskToken: "[MASK]",
    encode(text: string): number[] {
      return text
        .split(/\s+/)
        .filter(Boolean)
        .map((w, i) => 1000 + ((w.length * 31 + i * 7) % 20000));
    },
  };
}

function tokenizerFromHF(tokenizerJson: unknown): TokenizerLike | null {
  const data = parseTokenizerJson(tokenizerJson);
  if (!data) return null;
  const tok: TokenizerLike = {
    clsId: data.ids.cls,
    sepId: data.ids.sep,
    maskId: data.ids.mask,
    padId: data.ids.pad,
    maskToken: data.maskToken,
    encode: (text: string) => encodeWithData(data, text),
  };
  if (decodeWithData(data, []) !== null) tok.decode = (ids: number[]) => decodeWithData(data, ids) ?? "";
  return tok;
}

const r4 = (v: number): number => Math.round(v * 1e4) / 1e4;

export class Agent extends HookRegistry {
  hooksRaise: boolean;
  cfg: AgentCfg;
  provider: SessionProvider;
  revision: string | null;
  tok: TokenizerLike;
  maxLen: number;
  headMaxLen: number;
  temperatureRaw: unknown;
  temperatureByOptionsRaw: Record<string, unknown>;
  temperature: number[];
  temperatureByOptions: Record<string, number>;
  langTemperatures: Record<
    string,
    { temperature: number[]; temperatureByOptions: Record<string, number> }
  >;

  constructor(opts: AgentOptions) {
    super();
    if (!opts || !opts.provider) throw new Error("Agent needs a provider");
    this.provider = opts.provider;
    this.revision = opts.revision ?? null;
    // Hooks are opt-in; an unset hook list is a no-op. See hooks.ts.
    this.hooks = normaliseHooks(opts.hooks, opts.onPredictStart, opts.onPredictEnd);
    this.hooksRaise = opts.hooksRaise ?? true;
    const cfg = { ...(opts.cfg ?? {}) } as AgentCfg;
    if (opts.max_len !== undefined) cfg.max_len = opts.max_len;
    if (opts.head_max_len !== undefined) cfg.head_max_len = opts.head_max_len;
    if (opts.temperature !== undefined) cfg.temperature = opts.temperature;
    if (opts.temperature_by_options !== undefined) cfg.temperature_by_options = opts.temperature_by_options;
    this.cfg = cfg;
    this.maxLen = Number(cfg.max_len ?? 512);
    this.headMaxLen = Number(cfg.head_max_len ?? 192);
    this.tok = opts.tok ?? defaultTokenizer();
    const raw = (cfg.temperature ?? [1.0, 1.0, 1.0]) as unknown;
    this.temperatureRaw = raw;
    const rawList = Array.isArray(raw) ? raw : [raw, raw, raw];
    this.temperature = [0, 1, 2].map((i) => clampTemperature(rawList[i] ?? 1.0));
    this.temperatureByOptionsRaw = (cfg.temperature_by_options ?? {}) as Record<string, unknown>;
    this.temperatureByOptions = Object.fromEntries(
      Object.entries(this.temperatureByOptionsRaw).map(([k, v]) => [k, clampTemperature(v)]),
    );
    const entries: Array<[string, unknown, number]> = [
      ...Object.entries(this.temperatureByOptionsRaw).map(
        ([k, v]) => [k, v, this.temperatureByOptions[k]] as [string, unknown, number],
      ),
      ...[0, 1, 2].map(
        (i) => [`temperature[${i}]`, rawList[i] ?? 1.0, this.temperature[i]] as [string, unknown, number],
      ),
    ];
    const rejected: string[] = [];
    for (const [name, rawV, applied] of entries) {
      if (Number(rawV) === applied) continue;
      rejected.push(`${name}=${JSON.stringify(rawV) ?? String(rawV)} -> ${applied}`);
    }
    if (rejected.length > 0) {
      console.warn(
        `laya: this checkpoint ships invalid temperatures or values outside [${TEMP_MIN}, ${TEMP_MAX}]; ` +
          `using ${rejected.join(", ")}. Treat confidence from the affected entries as uncalibrated.`,
      );
    }
    // Mirrors Agent.__init__ (agent.py): keys normalise to the base subtag, an omitted
    // temperature defaults to the base raw temperature, and every value is clamped.
    this.langTemperatures = {};
    for (const [l, lc] of Object.entries(opts.lang_temperatures ?? {})) {
      const normL = l.split("-")[0].toLowerCase();
      const tRaw = lc?.temperature ?? rawList;
      if (!Array.isArray(tRaw) || tRaw.length !== 3) {
        throw new Error(
          `Language override ${JSON.stringify(l)} temperature must be a list of 3 floats`,
        );
      }
      const tboRaw = (lc?.temperature_by_options ?? {}) as Record<string, unknown>;
      this.langTemperatures[normL] = {
        temperature: [0, 1, 2].map((i) => clampTemperature(tRaw[i])),
        temperatureByOptions: Object.fromEntries(
          Object.entries(tboRaw).map(([k, v]) => [k, clampTemperature(v)]),
        ),
      };
    }
  }

  /**
   * Evaluate typed questions across one state in a single forward pass.
   *
   * `hooks` / `onPredictStart` / `onPredictEnd` observe or shape the prediction, appended
   * after any hooks installed on the Agent; a start hook may rewrite the state/questions or
   * call `ctx.skip(...)` to short-circuit inference, an end hook may rewrite the results.
   * See hooks.ts. `hooksRaise` overrides the Agent's setting for this call.
   */
  async systemOne(
    state: unknown,
    questions: Record<string, QuestionDef>,
    opts: PredictOptions = {},
  ): Promise<SystemOneResult> {
    // Python parity (agent.py system_one): defined through predictBatch so the two cannot drift.
    return (await this.predictBatch([state], questions, opts))[0];
  }

  /**
   * Evaluate the same questions over many states, packing their question rows into shared
   * encoder and head passes of `batchSize` states (default: all of them in one pass).
   *
   * Hooks fire once per call: `ctx.states` is the whole list. Results are aligned with
   * `states` by index, each identical in shape to `systemOne`'s output.
   */
  async predictBatch(
    states: unknown[],
    questions: Record<string, QuestionDef>,
    opts: PredictBatchOptions = {},
  ): Promise<SystemOneResult[]> {
    const mc = opts.minConfidence !== null && opts.minConfidence !== undefined
      ? checkMinConfidence(opts.minConfidence)
      : null;
    checkTokenBudget("maxLen", opts.maxLen);
    checkTokenBudget("headMaxLen", opts.headMaxLen);
    const active = defaultsAlreadyRan(opts)
      ? [...this.hooks, ...normaliseHooks(opts.hooks, opts.onPredictStart, opts.onPredictEnd)]
      : composeHooks(this.hooks, opts.hooks, opts.onPredictStart, opts.onPredictEnd);
    const raiseErrors = opts.hooksRaise ?? this.hooksRaise;
    const ctx = new PredictContext({
      states,
      questions: questions as Record<string, unknown>,
      agent: this,
      maxLen: opts.maxLen ?? null,
      headMaxLen: opts.headMaxLen ?? null,
    });
    try {
      await dispatchAsync(active, "onPredictStart", ctx, { raiseErrors });
      if (ctx.results === null) {
        // Only the values that survive the start hooks are validated (agent.py predict_batch).
        const sts: unknown = ctx.states;
        const qs: unknown = ctx.questions;
        if (!Array.isArray(sts)) {
          throw new TypeError("predictBatch expects a list of states; pass a single state to predict()/systemOne().");
        }
        if (typeof qs !== "object" || qs === null || Array.isArray(qs)) {
          throw new TypeError(`questions must be an object of question id -> definition, got ${typeName(qs)}`);
        }
        if (sts.some((s) => s === null || s === undefined)) {
          throw new TypeError("state must not be null or undefined; pass a string, object, or array");
        }
        const out = await this._inferBatch([...sts], qs as Record<string, QuestionDef>, {
          maxLen: checkTokenBudget("maxLen", ctx.maxLen) ?? this.maxLen,
          headMaxLen: checkTokenBudget("headMaxLen", ctx.headMaxLen) ?? this.headMaxLen,
          lang: opts.lang ?? null,
          batchSize: opts.batchSize ?? null,
          sortByLength: opts.sortByLength ?? false,
        });
        ctx.results = out as unknown as Record<string, unknown>[];
        ctx.model ??= out[0]?.model ?? null;
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
        // Flagged before the end hooks, so an audit hook sees exactly what the caller gets.
        if (mc !== null) flagLowConfidence(ctx.results, mc);
      }
      try {
        await dispatchAsync(active, "onPredictEnd", ctx, { raiseErrors });
      } catch (hookErr) {
        // End hooks run on the failure path too; do not let one mask the real error.
        if (ctx.error === null) throw hookErr;
      }
    }
    return ctx.results as unknown as SystemOneResult[];
  }

  /** Validate once, then encode, collate, run and decode in chunks of `batchSize` states. */
  private async _inferBatch(
    states: unknown[],
    questions: Record<string, QuestionDef>,
    o: { maxLen: number; headMaxLen: number; lang: string | null; batchSize: number | null; sortByLength: boolean },
  ): Promise<SystemOneResult[]> {
    if (states.length === 0) return [];
    const ids = Object.keys(questions);
    if (ids.length === 0) {
      return states.map(() => ({ model: "laya-rl-agent", answers: {}, usage: { input_tokens: 0, output_tokens: 0 } }));
    }
    for (const qid of ids) checkQuestion(qid, questions[qid]);
    const internals = ids.map((qid) => toInternal(questions[qid]));
    const chunk = o.batchSize && o.batchSize > 0 ? o.batchSize : states.length;
    const reorder = o.sortByLength && 1 < chunk && chunk < states.length;
    // agent.py:1025 — bound the tokenized lookahead to eight batches, and sort by each state's
    // longest post-truncation row so a state's questions stay together.
    const window = reorder ? chunk * 8 : chunk;
    const results: SystemOneResult[] = [];
    for (let start = 0; start < states.length; start += window) {
      const part = states.slice(start, start + window);
      const encoded = part.map((st) => this._encodeState(st, ids, internals, o.maxLen, o.headMaxLen));
      const order = encoded.map((_, i) => i);
      if (reorder) {
        const longest = encoded.map((items) => Math.max(...items.map((it) => it.ids.length)));
        order.sort((a, b) => longest[a] - longest[b]);
      }
      const windowResults: SystemOneResult[] = new Array(encoded.length);
      for (let offset = 0; offset < order.length; offset += chunk) {
        const indices = order.slice(offset, offset + chunk);
        const perState = indices.map((i) => encoded[i]);
        const collated = collateItems(perState, this.tok.padId);
        if (!collated) throw new Error("no items to collate");
        const batch: Batch = collated;
        const { lastHidden } = await this.provider.runEncoder(batch);
        const { logits, act } = await this.provider.runHead(lastHidden, batch);
        const rows = perState.flat();
        if (!Array.isArray(logits) || !Array.isArray(act) || logits.length < rows.length || act.length < rows.length) {
          throw new Error("model provider returned fewer output rows than input items");
        }
        for (let r = 0; r < rows.length; r++) {
          const k = rows[r].markers.length;
          if (!Array.isArray(logits[r]) || logits[r].length < k || !Array.isArray(act[r]) || act[r].length === 0 ||
              logits[r].some((v) => !Number.isFinite(v)) || act[r].some((v) => !Number.isFinite(v))) {
            throw new Error(`model provider returned invalid output for question ${ids[r % ids.length]}`);
          }
        }
        let row = 0;
        indices.forEach((index, j) => {
          const items = perState[j];
          let nTokens = 0;
          for (let r = row; r < row + items.length; r++) {
            for (const m of batch.attentionMask[r]) nTokens += m;
          }
          windowResults[index] = {
            model: "laya-rl-agent",
            answers: this._decodeAnswers(logits, act, items, ids, internals, row, o.lang),
            usage: { input_tokens: nTokens, output_tokens: 0 },
          };
          row += items.length;
        });
      }
      results.push(...windowResults);
    }
    return results;
  }

  /** One state's question rows; the state is tokenized once and shared by every question. */
  private _encodeState(
    state: unknown,
    ids: string[],
    internals: Internal[],
    maxLen: number,
    headMaxLen: number,
  ): EncodedItem[] {
    const stAll = this.tok.encode(serializeState(state).split(this.tok.maskToken).join(" "));
    return ids.map((qid, r) => {
      const q = internals[r];
      const prefix = buildQuestionPrefix(this.tok, q, maxLen, headMaxLen);
      const { ids: seq, markers } = sequenceWithState(prefix, stAll, this.tok.sepId, maxLen, Array.isArray(state));
      if (markers.length !== renderOptions(q).length) {
        throw new Error(`question ${qidStr(qid)} options exceed head_max_len=${headMaxLen}`);
      }
      return { ids: seq, markers, qtype: QTYPES[q.t] };
    });
  }

  /** Decode the `ids.length` head rows starting at `offset` into one state's answers. */
  private _decodeAnswers(
    logits: number[][],
    act: number[][],
    items: EncodedItem[],
    ids: string[],
    internals: Internal[],
    offset: number,
    lang: string | null,
  ): Record<string, SystemAnswer> {
    const answers: Record<string, SystemAnswer> = {};
    for (let qi = 0; qi < ids.length; qi++) {
      const r = offset + qi;
      const qid = ids[qi];
      const q = internals[qi];
      const k = items[qi].markers.length;
      const qt = QTYPES[q.t];
      const bucket = tempBucket(qt, k);
      // Python parity (_decode_answers): a matching lang override replaces the scale
      // wholesale — its own temperature_by_options first, then its 3-slot temperature —
      // so an override without buckets intentionally ignores the base per-bucket entries.
      const langCfg = lang ? this.langTemperatures[lang.split("-")[0].toLowerCase()] : undefined;
      const scale = langCfg
        ? (langCfg.temperatureByOptions[bucket] ?? langCfg.temperature[qt] ?? 1.0)
        : (this.temperatureByOptions[bucket] ?? this.temperature[qt] ?? 1.0);
      const z = (logits[r] as number[]).slice(0, k).map((v) => v / scale);
      const p = softmax(z);
      const actRow = (act[r] as number[]) ?? [1, 0];
      const actP = softmax(actRow.slice(0, Math.max(2, actRow.length)));
      const ext = { act_probability: r4(actP[0]) };
      // Same quantity on every question type (max(p)), so callers can gate across types on
      // one number; `confidence` stays as-is for existing callers (entropy for choice/score).
      const ansConf = r4(answerConfidence(p));
      if (q.t === "choice") {
        const keys = Object.keys(q.crit as Record<string, unknown>);
        let best = 0;
        for (let i = 1; i < p.length; i++) if (p[i] > p[best]) best = i;
        answers[qid] = {
          type: "choice",
          choice: keys[best],
          probabilities: Object.fromEntries(keys.map((kk, i) => [kk, r4(p[i] ?? 0)])),
          confidence: r4(confidenceFromProbs(p)),
          answer_confidence: ansConf,
          action: ext,
        };
      } else if (q.t === "score") {
        const exp = p.reduce((a, v, i) => a + i * v, 0);
        answers[qid] = {
          type: "score",
          score: r4(exp),
          // `renderCriterion`, not the raw `c`: a legend maps an index to the TEXT of a level, and
          // the keys are already strings. A numeric scale written directly used to come back with
          // the caller's own JSON types (`{"0": 1}`), so a client that reads a level as a string
          // had to handle a number, a boolean and null as well -- and `{"0": null}` is not even
          // parseable by a Jev client (#302). Same change as the Python `agent.py` and
          // `onnx_agent.py` legends, so the two runtimes return the same value types.
          legend: Object.fromEntries(
            (q.crit as unknown[]).map((c, i) => [String(i), renderCriterion(c)]),
          ),
          probabilities: Object.fromEntries(p.map((v, i) => [String(i), r4(v)])),
          confidence: r4(confidenceFromProbs(p)),
          answer_confidence: ansConf,
          action: ext,
        };
      } else {
        const pt = p[1] ?? 0;
        answers[qid] = {
          type: "noul",
          noul: r4(pt),
          confidence: r4(Math.max(pt, 1 - pt)),
          // over two options max(p_true, 1 - p_true) is max(p): identical to confidence here
          answer_confidence: ansConf,
          action: ext,
        };
      }
    }
    return answers;
  }

  async predict(
    state: unknown,
    questions: Record<string, QuestionDef>,
    opts: PredictOptions = {},
  ): Promise<SystemOneResult> {
    return this.systemOne(state, questions, opts);
  }

  /**
   * Evaluate questions over a state longer than the context window: tokenize it once, split it
   * into overlapping windows, score them all through `predictBatch`, and aggregate per question
   * (noul: max P(true); choice/score: the most confident window). Port of agent.py `predict_long`.
   *
   * The returned probability is the deciding window's, not a calibrated document-level number;
   * `answer.window` names that window. A state that fits one window goes to `systemOne`.
   * Hooks wrap the one `predictBatch` over the windows (`ctx.states` holds the window texts):
   * `ctx.skip([result])` answers the document with `usage.windows === 0`, and a rewritten scan is
   * aggregated without `answer.window`.
   */
  async predictLong(
    state: unknown,
    questions: Record<string, QuestionDef>,
    opts: PredictLongOptions = {},
  ): Promise<LongResult> {
    if ((opts.aggregate ?? "auto") !== "auto") throw new Error("predictLong: only aggregate='auto' is supported");
    const budget = opts.window && opts.window > 0 ? opts.window : Math.max(64, this.maxLen - this.headMaxLen - 8);
    const stateIds = this.tok.encode(serializeState(state).split(this.tok.maskToken).join(" "));

    if (stateIds.length <= budget) {
      const { probe, evidence } = startEvidence();
      const single = { ...(await this.systemOne(state, questions, withStartProbe(opts, probe))) };
      single.usage = { ...(single.usage ?? {}), windows: evidence.answered ? 0 : 1 } as LongResult["usage"];
      return single as LongResult;
    }

    const decode = this.tok.decode?.bind(this.tok);
    if (!decode) throw new Error("predictLong: this tokenizer cannot decode ids, so a long state cannot be windowed");
    const step = opts.stride && opts.stride > 0 ? opts.stride : Math.max(1, Math.floor(budget / 2));
    const windows: string[] = [];
    const starts: number[] = [];
    for (let i = 0; i < stateIds.length; i += step) {
      windows.push(decode(stateIds.slice(i, i + budget)));
      starts.push(i);
      if (i + budget >= stateIds.length) break;
    }

    const { probe, evidence } = startEvidence();
    // A copy, so a start hook that mutates `ctx.states` in place cannot shift `starts`.
    const results = await this.predictBatch([...windows], questions, {
      ...withStartProbe(opts, probe),
      batchSize: opts.batchSize,
    });

    if (evidence.answered) {
      if (results.length !== 1) {
        throw new Error(
          `predictLong: a start hook answered this state with ${results.length} results; ctx.skip()` +
            " takes one result for the document, not one per window",
        );
      }
      console.warn(
        "laya: predictLong: a hook answered the state before it was scanned, so no window decided the result and none is reported",
      );
      const document = { ...results[0] } as LongResult;
      document.usage = { ...(document.usage ?? {}), windows: 0 };
      return document;
    }

    const rewritten =
      evidence.states !== null &&
      (evidence.states.length !== windows.length || evidence.states.some((s, i) => s !== windows[i]));
    if (results.length !== windows.length && !rewritten) {
      throw new Error(
        `predictLong: the state was split into ${windows.length} windows and the call returned ${results.length} results`,
      );
    }
    if (results.length === 0) {
      console.warn(
        "laya: predictLong: a start hook left no states to score, so the call aggregated nothing and returns no answers",
      );
      return { model: "laya-rl-agent", answers: {}, usage: { ...aggregateUsage([]), windows: 0 } } as LongResult;
    }

    const answers: LongResult["answers"] = {};
    for (const qid of Object.keys(questions)) {
      const per = results.map((r) => r.answers[qid] as SystemAnswer & { window?: WindowInfo });
      // noul: the strongest window decides; choice/score: the most confident window, so a
      // localized signal is not out-voted by the neutral majority (agent.py:1252-1262).
      const key = toInternal(questions[qid]).t === "noul" ? "noul" : "answer_confidence";
      let best = 0;
      for (let j = 1; j < per.length; j++) {
        if (Number(per[j][key as keyof SystemAnswer]) > Number(per[best][key as keyof SystemAnswer])) best = j;
      }
      const ans = per[best];
      if (!rewritten) {
        ans.window = {
          index: best,
          token_start: starts[best],
          token_end: Math.min(starts[best] + budget, stateIds.length),
          count: results.length,
        };
      }
      answers[qid] = ans;
    }
    // Sum numeric usage fields across windows and carry the rest, then record the window count.
    const usage: Record<string, unknown> = {};
    for (const r of results) {
      for (const [k, v] of Object.entries(r.usage ?? {})) {
        usage[k] = typeof v === "number" ? ((usage[k] as number | undefined) ?? 0) + v : v;
      }
    }
    usage.output_tokens = 0;
    usage.windows = results.length;
    return { model: "laya-rl-agent", answers, usage: usage as unknown as LongResult["usage"] };
  }

  /**
   * Answer `state` against a JSON schema (or explicit `opts.questions`) and return typed
   * values — see `structured.ts`. Pass exactly one of `schema` or `opts.questions`; other
   * options are forwarded to `predict`.
   */
  async decide(
    state: unknown,
    schema: unknown,
    opts: DecideOptions & PredictOptions & { returnDetails: true },
  ): Promise<DecisionResult>;
  async decide(
    state: unknown,
    schema?: unknown,
    opts?: DecideOptions & PredictOptions,
  ): Promise<Record<string, unknown>>;
  async decide(
    state: unknown,
    schema?: unknown,
    opts: DecideOptions & PredictOptions = {},
  ): Promise<Record<string, unknown> | DecisionResult> {
    return decide(this, state, schema, opts);
  }

  /**
   * The batched `decide`: plan the schema once, answer every state through `predictBatch`
   * (shared forward passes, input order), then project each result as `decide` does. Other
   * options (`batchSize`, `lang`, hooks, ...) are forwarded to `predictBatch`.
   */
  async decideBatch(
    states: unknown[],
    schema: unknown,
    opts: DecideOptions & PredictBatchOptions & { returnDetails: true },
  ): Promise<DecisionResult[]>;
  async decideBatch(
    states: unknown[],
    schema?: unknown,
    opts?: DecideOptions & PredictBatchOptions,
  ): Promise<Record<string, unknown>[]>;
  async decideBatch(
    states: unknown[],
    schema?: unknown,
    opts: DecideOptions & PredictBatchOptions = {},
  ): Promise<Array<Record<string, unknown> | DecisionResult>> {
    return decideBatch(this, states, schema, opts);
  }

  static async load(
    modelDirOrRepo: string,
    opts?: {
      device?: string;
      subfolder?: string | null;
      localDir?: string;
      token?: string | null;
      numThreads?: number;
      /** Per-language temperature overrides; see AgentOptions.lang_temperatures. */
      lang_temperatures?: AgentOptions["lang_temperatures"];
      /** Optional commit SHA/branch/tag to fetch; omitted uses the Hub default and existing cache. */
      revision?: string | null;
      /**
       * Opt-in `{artifact name: SHA-256 hexdigest}` check before any artifact is parsed
       * or executed. A missing artifact or digest mismatch throws and loading is refused.
       */
      expectedSha256?: Record<string, string>;
    },
  ): Promise<Agent> {
    const sub = opts?.subfolder ?? null;
    let cfg: AgentCfg = {};
    let tokenizerJson: unknown | null = null;
    let dir = opts?.localDir ?? modelDirOrRepo;
    let revision: string | null = null;
    let provider: SessionProvider;
    {
      const { loadNodeBundle, createNodeProvider } = await import("./providers.js");
      const bundle = await loadNodeBundle(modelDirOrRepo, {
        subfolder: sub,
        localDir: opts?.localDir,
        token: opts?.token,
        revision: opts?.revision,
        expectedSha256: opts?.expectedSha256,
      });
      cfg = bundle.cfg;
      tokenizerJson = bundle.tokenizerJson;
      dir = bundle.dir;
      revision = bundle.revision;
      provider = await createNodeProvider(dir, {
        device: opts?.device, numThreads: opts?.numThreads, expectedSha256: opts?.expectedSha256,
      });
    }
    if (!tokenizerJson) {
      throw new Error(
        `Incompatible model: tokenizer.json is missing or invalid in ${JSON.stringify(dir)}`,
      );
    }
    let tok: TokenizerLike;
    try {
      const parsed = tokenizerFromHF(tokenizerJson);
      if (!parsed) throw new Error("unsupported tokenizer.json format");
      tok = parsed;
    } catch (error) {
      throw new Error(
        `Incompatible model: tokenizer.json is missing or invalid in ${JSON.stringify(dir)}: ${String(error)}`,
      );
    }
    return new Agent({ provider, tok, cfg, revision, lang_temperatures: opts?.lang_temperatures });
  }
}
