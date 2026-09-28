// Server-side routing policy over the lib Router.
//
// Deliberate deviation from serve.py: there LAYA_MODELS only lists what to preload, and the
// router may still lazy-load any of the three checkpoints on demand. Here LAYA_MODELS is the
// set of checkpoints the server will ever load. A deployment sized for one fp32 checkpoint
// (the default image bakes only `multilingual`) must not start fetching and loading a second
// 1.6 GB model because a request happened to be in English.
//   - No explicit `model`, and routing picks a checkpoint outside the set: the request is
//     served by the default checkpoint (the first of LAYA_MODELS; serve.py has no
//     default-model variable), and `routing.reason` says so.
//   - An explicit `model` outside the set: 400 (app.ts), never a fallback, because the caller
//     asked for that checkpoint by name.
import type { Hook, PredictContext, Router } from "@desplega/laya";
import type { PredictCallOptions, ServerRouter } from "./app.js";

/** Suffix appended to `routing.reason` when the decision was redirected to the default checkpoint. */
export function fallbackReason(reason: string, routed: string, served: string): string {
  return `${reason}; ${JSON.stringify(routed)} is not in LAYA_MODELS, served by ${JSON.stringify(served)}`;
}

/**
 * An `onRoute` hook that redirects any decision outside `allowed` to `fallback`. `repos` maps
 * a checkpoint to the `repo` string the RouteDecision should carry.
 */
export function routeWithin(allowed: readonly string[], fallback: string, repos: Record<string, string>): Hook {
  if (!allowed.includes(fallback)) throw new Error(`fallback ${JSON.stringify(fallback)} is not in ${allowed}`);
  return {
    onRoute(ctx: PredictContext) {
      const d = ctx.decision;
      if (!d || allowed.includes(String(d.model))) return;
      ctx.decision = {
        ...d,
        model: fallback,
        repo: repos[fallback] ?? d.repo,
        reason: fallbackReason(String(d.reason), String(d.model), fallback),
      };
    },
  };
}

/** Adapt the lib Router to the app's `ServerRouter`. `revisionOf` names the bundle commit of a resident checkpoint. */
export function toServerRouter(router: Router, revisionOf: (name: string) => string | null): ServerRouter {
  return {
    get loaded() {
      return router.loaded;
    },
    get revisions() {
      return Object.fromEntries(router.loaded.map((n) => [n, revisionOf(n)]));
    },
    predict: (state: unknown, questions: Record<string, unknown>, o: PredictCallOptions) =>
      router.predict(
        state,
        questions as Parameters<Router["predict"]>[1],
        {
          model: o.model,
          // Per-call token budgets; honoured once the lib's predict reads them (plan, Phase 6).
          ...(o.maxLen !== undefined ? { maxLen: o.maxLen } : {}),
          ...(o.headMaxLen !== undefined ? { headMaxLen: o.headMaxLen } : {}),
        } as Parameters<Router["predict"]>[2],
      ),
  };
}
