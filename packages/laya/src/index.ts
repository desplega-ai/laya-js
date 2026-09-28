// Modified by Desplega Labs, 2026: dropped the createWebProvider, loadWebBundle and WebBundle exports; moved the vendored exports to raw.ts (`@desplega/laya/raw`) and export only the typed API (typed.ts) here.
export type {
  ActionInfo,
  ChoiceAnswer,
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
export type { CheckpointName, Precision } from "./artifacts.js";
export { LayaLoadError } from "./providers.js";
export { VERSION } from "./raw.js";
export type { Hook, HookArg, PredictContext, PredictHook } from "./hooks.js";
export type { AnalyseResult } from "./lang.js";
export type { LangGuess, ModelName, RouteDecision, RouteOptions, RouterBatchOptions, RouterRequest } from "./router.js";
export { SchemaError } from "./structured.js";
export { CHECKPOINTS, createAgent, createRouter, defineQuestions, jsonSchemaOf, validateDecision } from "./typed.js";
export type {
  AbstentionFlag,
  AnswerOf,
  Answers,
  CheckpointInfo,
  ChoiceQuestion,
  CreateAgentOptions,
  CreateRouterOptions,
  DecideCallOptions,
  LabelsOf,
  LayaAgent,
  LayaRouter,
  LongAnswers,
  LongPredictResult,
  NoulQuestion,
  PredictResult,
  Question,
  QuestionMap,
  RoutedBatchResults,
  RoutedLongPredictResult,
  RoutedPredictResult,
  ScoreQuestion,
  TypedRouterRequest,
} from "./typed.js";
