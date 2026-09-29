// Fetch checkpoints' fp32 bundles from the artifact store into a directory and verify each
// file's SHA-256 against the pinned map. Used by the Docker `models` stage (bakes
// `multilingual` into /models) and by the k8s initContainer (fills the /cache volume).
//
//   node dist/fetch-models.js [--dest DIR] [--precision fp32] [--no-rehash] [checkpoint ...]
//
// Checkpoints default to LAYA_MODELS; --dest defaults to LAYA_CACHE_DIR. HF_TOKEN is read
// from the environment only, never from argv, and is never printed.
import { realpathSync } from "node:fs";
import { fileURLToPath } from "node:url";
import type { CheckpointName } from "@desplega.ai/laya";
import { normaliseName } from "@desplega.ai/laya/raw";
import { bundleDir, bundleSpec, fetchBundle, isVerified } from "./bundles.js";
import { EnvError, loadEnv } from "./env.js";

export interface FetchModelsArgs {
  dest: string;
  checkpoints: CheckpointName[];
  rehash: boolean;
}

const USAGE = "usage: fetch-models [--dest DIR] [--precision fp32] [--no-rehash] [checkpoint ...]";

export function parseFetchArgs(argv: string[], env: Record<string, string | undefined>): FetchModelsArgs {
  const cfg = loadEnv(env);
  let dest = cfg.cacheDir;
  let rehash = true;
  const names: string[] = [];
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--dest") {
      const v = argv[++i];
      if (!v) throw new EnvError(`--dest needs a directory\n${USAGE}`);
      dest = v;
    } else if (a === "--precision") {
      const v = argv[++i];
      if (v !== "fp32") throw new EnvError(`--precision ${JSON.stringify(v)}: only "fp32" is available`);
    } else if (a === "--no-rehash") {
      rehash = false;
    } else if (a === "-h" || a === "--help") {
      throw new EnvError(USAGE);
    } else if (a.startsWith("-")) {
      throw new EnvError(`unknown option ${a}\n${USAGE}`);
    } else {
      names.push(...a.split(",").filter(Boolean));
    }
  }
  const checkpoints: CheckpointName[] = [];
  for (const n of names.length ? names : cfg.models) {
    let key: CheckpointName;
    try {
      key = normaliseName(n);
    } catch (e) {
      throw new EnvError((e as Error).message);
    }
    if (!checkpoints.includes(key)) checkpoints.push(key);
  }
  return { dest, checkpoints, rehash };
}

export async function fetchModels(
  argv: string[],
  env: Record<string, string | undefined> = process.env,
  log: (msg: string) => void = (m) => console.log(m),
): Promise<void> {
  const args = parseFetchArgs(argv, env);
  const cfg = loadEnv(env);
  for (const checkpoint of args.checkpoints) {
    const spec = bundleSpec(checkpoint, cfg.precision, { repo: cfg.onnxRepo, revision: cfg.onnxRevision });
    const dir = bundleDir(args.dest, checkpoint, cfg.precision);
    if (await isVerified(dir, spec, { rehash: args.rehash })) {
      log(`${checkpoint}/${cfg.precision}: ${dir} already verified`);
      continue;
    }
    await fetchBundle(spec, dir, { token: cfg.hfToken, endpoint: cfg.hfEndpoint, log });
    log(`${checkpoint}/${cfg.precision}: ${dir} verified against the pinned SHA-256 map`);
  }
}

function invokedDirectly(): boolean {
  const argv1 = process.argv[1];
  if (!argv1) return false;
  try {
    return realpathSync(argv1) === realpathSync(fileURLToPath(import.meta.url));
  } catch {
    return false;
  }
}

if (invokedDirectly()) {
  fetchModels(process.argv.slice(2)).catch((e: unknown) => {
    console.error(`fetch-models: ${e instanceof Error ? e.message : String(e)}`);
    process.exit(1);
  });
}
