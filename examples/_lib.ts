// Shared helpers for the examples: load a checkpoint from a local bundle root when
// LAYA_MODEL_DIR is set (layout `<root>/<checkpoint>/fp32`), else from the artifact store
// (public; HF_TOKEN is optional). Not part of the library API.
import { join } from "node:path";
import { type CheckpointName, type CreateAgentOptions, createAgent, createRouter } from "@desplega.ai/laya";

const root = process.env.LAYA_MODEL_DIR;
const threads = process.env.LAYA_THREADS ? Number(process.env.LAYA_THREADS) : undefined;

export function agentOptions(checkpoint: CheckpointName): CreateAgentOptions {
  return { checkpoint, numThreads: threads, ...(root ? { modelDir: join(root, checkpoint, "fp32") } : {}) };
}

export async function loadAgent(checkpoint: CheckpointName) {
  return createAgent(agentOptions(checkpoint));
}

export async function loadRouter(checkpoints: CheckpointName[]) {
  const modelDirs = root ? Object.fromEntries(checkpoints.map((c) => [c, join(root, c, "fp32")])) : undefined;
  return createRouter({ checkpoints, modelDirs, numThreads: threads, maxLoaded: checkpoints.length });
}

/** Run `fn` and return its result with the wall time in milliseconds. */
export async function timed<T>(fn: () => Promise<T>): Promise<{ value: T; ms: number }> {
  const t0 = performance.now();
  const value = await fn();
  return { value, ms: performance.now() - t0 };
}

export const pct = (p: number) => `${(p * 100).toFixed(0)}%`;
export const ms = (n: number) => `${n.toFixed(0)} ms`;
