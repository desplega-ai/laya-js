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
import type { Hook, PredictContext } from "@desplega/laya";
import type { Router } from "@desplega/laya/raw";
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
      router.predict(state, questions as Parameters<Router["predict"]>[1], {
        model: o.model,
        maxLen: o.maxLen,
        headMaxLen: o.headMaxLen,
      }),
  };
}

/** Anything holding native memory that must be freed explicitly (a `LayaAgent`). */
export interface Disposable {
  dispose(): Promise<void>;
}

/**
 * Owns the loaded agents so their ONNX sessions are released when the lib Router drops them.
 * The Router's LRU eviction (`maxLoaded`) only removes its map entry and fires `onEvict`; the
 * native sessions (1-2 GB per fp32 bundle) stay allocated until something calls release.
 * Register every loaded agent with `track`, install `hook` on the Router, and call `disposeAll`
 * on shutdown (`Router.unload()` fires no `onEvict`).
 */
export function agentRegistry<A extends Disposable>() {
  const agents = new Map<string, A>();
  return {
    track(name: string, agent: A): void {
      agents.set(name, agent);
    },
    get(name: string): A | undefined {
      return agents.get(name);
    },
    hook: {
      // Inference runs one call at a time (app.ts Gate) and eviction happens inside the load
      // of the next call's checkpoint, so no in-flight call still holds the evicted agent.
      async onEvict(ctx: PredictContext) {
        const name = String(ctx.model);
        const agent = agents.get(name);
        agents.delete(name);
        await agent?.dispose();
      },
    } satisfies Hook,
    async disposeAll(): Promise<void> {
      const held = [...agents.values()];
      agents.clear();
      await Promise.all(held.map((a) => a.dispose()));
    },
  };
}
