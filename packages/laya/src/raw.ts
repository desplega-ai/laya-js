// The untyped vendored surface of laya-ts, importable as `@desplega.ai/laya/raw` (escape hatch).
// Moved here from index.ts by Desplega Labs, 2026; see UPSTREAM.md.
export const VERSION = "0.1.1";
export type {
  ActionInfo,
  AgentCfg,
  AgentOptions,
  ChoiceAnswer,
  LongResult,
  NoulAnswer,
  PredictBatchOptions,
  PredictLongOptions,
  PredictOptions,
  QuestionDef,
  ScoreAnswer,
  SystemAnswer,
  SystemOneResult,
  SystemUsage,
  WindowInfo,
} from "./agent.js";
export { Agent, checkQuestion, defaultTokenizer, toInternal } from "./agent.js";
export type { BundleArtifact, CheckpointName, Precision } from "./artifacts.js";
export { ARTIFACT_REPO, ARTIFACT_REVISION, ARTIFACTS } from "./artifacts.js";
export { checkTokenBudget, MAX_TOKEN_BUDGET } from "./budget.js";
export type { CollatedBatch, CollateItem, InternalQ, QType, QuestionPrefix } from "./common.js";
export {
  answerConfidence,
  buildQuestionPrefix,
  buildSequence,
  clampTemperature,
  collateItems,
  confidenceFromProbs,
  renderOptions,
  sequenceWithState,
  serializeState,
  softmax,
  TEMP_MAX,
  TEMP_MIN,
  tempBucket,
} from "./common.js";
export { checkMinConfidence, flagLowConfidence } from "./confidence.js";
export { cleanEmailBody, emailState } from "./email.js";
export type { Hook, HookArg, HookEvent, PredictHook, PredictHookArg } from "./hooks.js";
export {
  addDefaultHook,
  aggregateUsage,
  BaseHook,
  clearDefaultHooks,
  composeHooks,
  defaultHooks,
  dispatch,
  HOOK_EVENTS,
  HookRegistry,
  normaliseHooks,
  PredictContext,
  setDefaultHooks,
} from "./hooks.js";
export type { AnalyseResult, LatinProfile } from "./lang.js";
export { analyse, detectScript, guessLatinLanguage, isEnglish } from "./lang.js";
export { emailQuestions, guardQuestions, moderationQuestions, routerQuestions, triageQuestions } from "./presets.js";
export type { Batch, NodeBundle, ProviderOptions, SessionProvider } from "./providers.js";
export {
  createNodeProvider,
  feed,
  feedHead,
  LayaLoadError,
  loadNodeBundle,
  PINNED_REVISIONS,
  resolveRevision,
} from "./providers.js";
export type {
  AgentLoader,
  LangGuess,
  ModelName,
  ModelSpec,
  RouteDecision,
  RoutedLongResult,
  RoutedResult,
  RouteOptions,
  RouterBatchOptions,
  RouterOptions,
  RouterRequest,
} from "./router.js";
export { DEFAULT_MODELS, normaliseName, Router } from "./router.js";
export type { EmbedFn, ShortlistMeta } from "./shortlist.js";
export { DEFAULT_SHORTLIST_K, embedFnFromAgent, predictShortlist, shortlistChoice } from "./shortlist.js";
export type { BatchDecideRunner, DecideOptions, DecideRunner, DecisionResult, PlannedField } from "./structured.js";
export {
  answersToJson,
  decide,
  decideBatch,
  MAX_OPTIONS,
  MAX_PROPERTIES,
  MAX_SCORE_LEVELS,
  planFromJsonSchema,
  questionsFromJsonSchema,
  SchemaError,
} from "./structured.js";
export type { AddedToken, PreTokenizerKind, TokenizerData, TokenizerIds, TokenizerLike } from "./tokenizer.js";
export {
  bpeEncode,
  CHECKPOINT_IDS,
  decodeWithData,
  encodeWithData,
  loadTokenizerJson,
  METASPACE_REPLACEMENT,
  metaspaceEncode,
  parseTokenizerJson,
  SPECIAL_ALIASES,
} from "./tokenizer.js";
