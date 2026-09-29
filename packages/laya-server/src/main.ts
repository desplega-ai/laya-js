// laya-server entry point: read the environment, start listening (health answers 503 while
// loading), fetch any listed checkpoint that is not baked, preload, then serve. SIGTERM and
// SIGINT drain in-flight requests before exiting.
import { createAgent, type LayaAgent } from "@desplega/laya";
import { Router } from "@desplega/laya/raw";
import { serve } from "@hono/node-server";
import { createApp, type ServerRouter } from "./app.js";
import { createBundleStore } from "./bundles.js";
import { describeEnv, EnvError, type LogLevel, loadEnv, type ServerEnv } from "./env.js";
import { agentRegistry, routeWithin, toServerRouter } from "./routing.js";

const RANK: Record<LogLevel, number> = { trace: 0, debug: 1, info: 2, warning: 3, error: 4, critical: 5 };

function createLogger(level: LogLevel) {
  const emit = (lvl: LogLevel, msg: string, err?: unknown) => {
    if (RANK[lvl] < RANK[level]) return;
    const line = `${new Date().toISOString()} ${lvl.toUpperCase()} laya-server: ${msg}`;
    const out = RANK[lvl] >= RANK.warning ? console.error : console.log;
    if (err === undefined) out(line);
    else out(line, err instanceof Error ? (err.stack ?? err.message) : err);
  };
  return {
    debug: (m: string) => emit("debug", m),
    info: (m: string) => emit("info", m),
    warn: (m: string) => emit("warning", m),
    error: (m: string, e?: unknown) => emit("error", m, e),
  };
}

type Log = ReturnType<typeof createLogger>;

/** The lib Router with a loader that resolves bundles through the local store, plus the revisions it saw. */
function buildRouter(env: ServerEnv, log: Log): { router: Router; server: ServerRouter; dispose(): Promise<void> } {
  const store = createBundleStore({
    modelDir: env.modelDir,
    cacheDir: env.cacheDir,
    precision: env.precision,
    repo: env.onnxRepo,
    revision: env.onnxRevision,
    token: env.hfToken,
    endpoint: env.hfEndpoint,
    log: (m) => log.info(m),
  });
  const agents = agentRegistry<LayaAgent>();
  const revisions = new Map<string, string | null>();
  const names = ["english", "multilingual", "typed-decisions"] as const;
  const repos = Object.fromEntries(names.map((n) => [n, `${env.onnxRepo}/${n}/${env.precision}`]));
  const router = new Router({
    models: Object.fromEntries(names.map((n) => [n, { repo: env.onnxRepo, subfolder: `${n}/${env.precision}` }])),
    maxLoaded: env.maxLoaded ?? undefined,
    autoTaskDetection: env.autoTask,
    // Tokens never go through the Router: bundles are resolved to local dirs by the store.
    token: null,
    loader: async (name) => {
      // routeWithin and the app's 400 keep routing inside LAYA_MODELS; this is the backstop.
      if (!env.models.includes(name)) throw new Error(`refusing to load ${name}: not in LAYA_MODELS`);
      const bundle = await store.resolve(name);
      log.info(`loading ${name}/${env.precision} from ${bundle.dir} (${bundle.source})`);
      const agent = await createAgent({
        checkpoint: name,
        precision: env.precision,
        modelDir: bundle.dir,
        numThreads: env.threads ?? undefined,
        // Baked bundles were verified at image build, cached ones when fetched.
        verify: false,
      });
      agents.track(name, agent);
      revisions.set(name, bundle.revision);
      return agent.raw;
    },
    hooks: [
      routeWithin(env.models, env.defaultModel, repos),
      // Frees the evicted checkpoint's ONNX sessions; the lib Router only drops its reference.
      agents.hook,
      {
        onEvict: (ctx) => {
          revisions.delete(String(ctx.model));
          log.info(`evicted ${ctx.model}; ONNX sessions released`);
        },
      },
    ],
  });
  const server = toServerRouter(router, (n) => revisions.get(n) ?? null);
  return {
    router,
    server,
    dispose: async () => {
      router.unload();
      await agents.disposeAll();
    },
  };
}

async function main(): Promise<void> {
  let env: ServerEnv;
  try {
    env = loadEnv();
  } catch (e) {
    console.error(`laya-server: ${e instanceof EnvError ? e.message : String(e)}`);
    process.exit(1);
  }
  const log = createLogger(env.logLevel);
  log.info(`config ${JSON.stringify(describeEnv(env))}`);
  if (env.device && env.device.toLowerCase() !== "cpu") {
    log.warn(`LAYA_DEVICE=${JSON.stringify(env.device)} ignored: the TS runtime computes on CPU only`);
  }

  const built = buildRouter(env, log);
  let ready = !env.preload;
  const app = createApp(built.server, {
    apiKey: env.apiKey,
    maxConcurrent: env.maxConcurrent,
    maxTokenBudget: env.maxTokenBudget,
    precision: env.precision,
    isReady: () => ready,
    allowedModels: env.models,
    logger: { error: (m, e) => log.error(m, e) },
  });
  const server = serve({ fetch: app.fetch, hostname: env.host, port: env.port }, (info) => {
    log.info(`listening on ${info.address}:${info.port}`);
  });

  let stopping = false;
  const shutdown = (signal: string) => {
    if (stopping) return;
    stopping = true;
    log.info(`${signal}: draining`);
    const force = setTimeout(() => {
      log.warn("drain timed out; exiting");
      process.exit(1);
    }, 25_000);
    force.unref();
    server.close(() => {
      built.dispose().finally(() => {
        log.info("stopped");
        process.exit(0);
      });
    });
    // Idle keep-alive sockets would otherwise hold close() open until they time out.
    (server as { closeIdleConnections?: () => void }).closeIdleConnections?.();
  };
  process.on("SIGTERM", () => shutdown("SIGTERM"));
  process.on("SIGINT", () => shutdown("SIGINT"));

  if (env.preload) {
    const t0 = performance.now();
    try {
      await built.router.preload(env.models);
    } catch (e) {
      log.error(`preload of ${env.models.join(",")} failed`, e);
      process.exit(1);
    }
    ready = true;
    log.info(`ready: ${built.router.loaded.join(",")} in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
  }
}

await main();
