// Modified by Desplega Labs, 2026: removed the browser path (createWebProvider, loadWebBundle, fetchArrayBuffer, onnxruntime-web); HF 401/403 throws LayaLoadError; SessionProvider.release frees the ONNX sessions; numThreads/LAYA_THREADS set the sessions' intraOpNumThreads (ort.env.numThreads is a web-only knob). Bounded working memory: the encoder and head graphs run in row chunks sized by `runMemoryMb` / `LAYA_RUN_MB`, one graph run at a time per provider, and each chunk is trimmed to its longest real row (the graphs materialise [rows, heads, len, len] attention, so peak RSS grew with batch x len^2).
import { maxOf } from "./common.js";

/** ONNX session shim: Node (onnxruntime-node).
 * Lazy imports only — unit tests with a fake provider never touch onnxruntime. */

/** Opt-in reviewed commit SHAs of the published checkpoints (mirror of laya/revisions.py).
 * They are not applied implicitly, so existing Hub/offline caches keep working. */
export const PINNED_REVISIONS: Record<string, string> = {
  "convaiinnovations/laya": "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
  "convaiinnovations/laya-multilingual": "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67",
  "convaiinnovations/laya-typed-decisions": "1a793eb568e6718f15941d08f85432581df534e3",
};

/** Return an explicit revision unchanged; otherwise preserve the Hub default and cache. */
export function resolveRevision(_repoOrId: string, revision?: string | null): string | null {
  return revision || null;
}

/** SHA-256 hex via Web Crypto (browsers and modern Node expose globalThis.crypto). */
async function sha256Hex(data: ArrayBuffer | Uint8Array): Promise<string> {
  const subtle = (globalThis as { crypto?: { subtle?: any } }).crypto?.subtle;
  if (!subtle) {
    throw new Error("laya-ts: SHA-256 verification requires Web Crypto (globalThis.crypto.subtle)");
  }
  const digest = await subtle.digest("SHA-256", data);
  return Array.from(new Uint8Array(digest as ArrayBuffer), (b) => b.toString(16).padStart(2, "0")).join("");
}

/** Reject absolute or escaping digest paths before they ever reach the filesystem. */
function normaliseDigestPath(rel: string): string {
  const raw = String(rel).replace(/\\/g, "/");
  if (/^(?:[A-Za-z]:|\/)/.test(raw)) {
    throw new Error(`laya-ts: unsafe absolute path in expectedSha256: ${JSON.stringify(rel)}`);
  }
  const norm = raw;
  if (!norm || norm === ".." || norm.startsWith("../") || norm.includes("/../")) {
    throw new Error(`laya-ts: unsafe path in expectedSha256: ${JSON.stringify(rel)}`);
  }
  return norm;
}

/** Verify `data` against expectedSha256[rel]; artifacts not listed are left unchecked. */
async function expectDigest(
  rel: string,
  data: ArrayBuffer | Uint8Array,
  expected: Record<string, string>,
): Promise<void> {
  const want = expected[rel] ?? expected[normaliseDigestPath(rel)];
  if (want === undefined) return;
  const got = await sha256Hex(data);
  if (got.toLowerCase() !== String(want).trim().toLowerCase()) {
    throw new Error(
      `laya-ts: SHA-256 mismatch for ${rel}: expected ${want}, got ${got}. Refusing to load the artifact.`,
    );
  }
}

export interface Batch {
  inputIds: number[][];
  attentionMask: number[][];
  markerPos: number[][];
  markerMask: boolean[][];
  qtype: number[];
}

export interface SessionProvider {
  runEncoder(batch: Batch): Promise<{ lastHidden: number[][][] }>;
  runHead(hidden: number[][][] | unknown, batch: Batch): Promise<{ logits: number[][]; act: number[][] }>;
  /** Free the underlying sessions; the provider is unusable afterwards. */
  release?(): Promise<void>;
}

function toNested(data: ArrayLike<number | bigint | boolean>, dims: number[]): any {
  // Single-pass copy + precomputed steps; no per-node slice/reduce.
  let total = 1;
  for (const d of dims) total *= d;
  const flat = new Array(total);
  for (let i = 0; i < total; i++) {
    const v = (data as any)[i];
    flat[i] = typeof v === "bigint" ? Number(v) : v;
  }
  if (dims.length === 0) return flat[0];
  const steps: number[] = new Array(dims.length);
  for (let d = 0; d < dims.length; d++) {
    let s = 1;
    for (let k = d + 1; k < dims.length; k++) s *= dims[k];
    steps[d] = s;
  }
  const rec = (d: number, off: number): any => {
    if (d === dims.length - 1) return flat.slice(off, off + dims[d]);
    const out: any[] = new Array(dims[d]);
    for (let i = 0; i < dims[d]; i++) out[i] = rec(d + 1, off + i * steps[d]);
    return out;
  };
  return rec(0, 0);
}

function i64(ort: any, arr: number[] | number[][], dims: number[]): any {
  // Rank <= 2 by construction; direct loop avoids flat(Infinity) intermediates.
  const out = new BigInt64Array(dims.reduce((a, b) => a * b, 1));
  let p = 0;
  if (Array.isArray((arr as any)[0])) {
    for (const row of arr as number[][]) for (const v of row) out[p++] = BigInt(Math.trunc(v));
  } else {
    for (const v of arr as number[]) out[p++] = BigInt(Math.trunc(v));
  }
  return new ort.Tensor("int64", out, dims);
}

/** Encoder feeds: input_ids + attention_mask (int64). */
export function feed(ort: any, b: Batch): Record<string, any> {
  const n = b.inputIds.length;
  let L = 1;
  for (const r of b.inputIds) if (r.length > L) L = r.length;
  return {
    input_ids: i64(ort, b.inputIds, [n, L]),
    attention_mask: i64(ort, b.attentionMask, [n, L]),
  };
}

/** Head feeds: encoder hidden + marker_pos/mask + qtype. */
export function feedHead(ort: any, hidden: number[][][] | any, b: Batch, seqLen?: number): Record<string, any> {
  const n = b.markerPos.length;
  let k = 1;
  for (const r of b.markerPos) if (r.length > k) k = r.length;
  // Direct flatten into typed arrays; no flat(Infinity)+map intermediates.
  const nH = (hidden as any).length ?? n;
  // `seqLen` lets a caller pass rows trimmed to their own length; missing positions read as zeros.
  const S = seqLen ?? (hidden as any)[0]?.length ?? 1;
  const Hd = (hidden as any)[0]?.[0]?.length ?? 1;
  const flatH = new Float32Array(nH * S * Hd);
  let p = 0;
  for (let i = 0; i < nH; i++) {
    const bi = (hidden as any)[i] ?? [];
    for (let j = 0; j < S; j++) {
      const hj = bi[j] ?? [];
      for (let h = 0; h < Hd; h++) flatH[p++] = Number(hj[h] ?? 0);
    }
  }
  const H = new ort.Tensor("float32", flatH, [nH, S, Hd]);
  // Pad/trim mask rows to S so the mask always matches hidden_states even
  // if a caller passes unpadded rows.
  const maskRows = b.attentionMask.map((r) => {
    const row = r.slice(0, S);
    while (row.length < S) row.push(0);
    return row;
  });
  return {
    hidden_states: H,
    marker_pos: i64(ort, b.markerPos, [n, k]),
    marker_mask: new ort.Tensor("bool", (() => {
      const out = new Uint8Array(n * k);
      let q = 0;
      for (const row of b.markerMask as unknown as boolean[][])
        for (let j = 0; j < k; j++) out[q++] = row[j] ? 1 : 0;
      return out;
    })(), [n, k]),
    qtype: i64(ort, b.qtype.map((v) => [v]), [n, 1]),
    // Padding mask for the head transformer (py DecisionModel.forward).
    // Without it, batch mates of unequal length corrupt each other's markers.
    attention_mask: i64(ort, maskRows, [n, S]),
  };
}

function pickOutput(out: Record<string, any>, names: string[]): any {
  for (const n of names) if (out[n] !== undefined) return out[n];
  const vals = Object.values(out);
  return vals[0];
}

export interface ProviderOptions {
  device?: string;
  numThreads?: number;
  /** Working-memory budget of one graph run, in MiB (`LAYA_RUN_MB` when unset). Rows beyond it go
   * through the graph in several runs. Default 256 on the CPU, unbounded on other devices. */
  runMemoryMb?: number;
  /** Opt-in {artifact name: SHA-256 hexdigest} check for fetched ONNX files (web). */
  expectedSha256?: Record<string, string>;
}

/** Session options for the requested thread count. onnxruntime-node ignores `ort.env.numThreads`
 * (an onnxruntime-web setting), so the count has to go into each session's `intraOpNumThreads`. */
function threadOptions(numThreads?: number): { intraOpNumThreads?: number } {
  const raw =
    numThreads ??
    (typeof process !== "undefined" ? Number((process as any).env?.["LAYA_THREADS"]) : NaN);
  return Number.isFinite(raw) && (raw as number) > 0 ? { intraOpNumThreads: Math.trunc(raw as number) } : {};
}

/** Default working-memory budget of one graph run on the CPU, in MiB. */
export const DEFAULT_RUN_MB = 256;

/** Working set of one row per token^2, in bytes. Both graphs build [rows, 12 heads, len, len]
 * attention and keep about ten such tensors live: 372 B (head) and 412 B (encoder) per token^2 at
 * one 1024-token row, onnxruntime-node 1.30, arena on. Rounded up. */
const RUN_BYTES_PER_TOKEN_SQ = 400;

function runBudgetBytes(runMemoryMb: number | undefined, cpu: boolean): number {
  const raw = runMemoryMb ?? (typeof process !== "undefined" ? Number((process as any).env?.["LAYA_RUN_MB"]) : NaN);
  if (Number.isFinite(raw) && (raw as number) > 0) return (raw as number) * 2 ** 20;
  return cpu ? DEFAULT_RUN_MB * 2 ** 20 : Number.POSITIVE_INFINITY;
}

/** Tokens a row really uses: its attention mask up to the last 1 (rows are right-padded). */
export function usedLengths(attentionMask: number[][]): number[] {
  return attentionMask.map((row) => {
    let n = row.length;
    while (n > 1 && !row[n - 1]) n--;
    return Math.max(1, n);
  });
}

/** Split rows, in order, into runs whose padded working set fits `budgetBytes`. A run always holds
 * at least one row, so a single row longer than the budget still runs. */
export function planRuns(lengths: number[], budgetBytes: number): number[][] {
  const runs: number[][] = [];
  let run: number[] = [];
  let longest = 0;
  for (let i = 0; i < lengths.length; i++) {
    const len = Math.max(longest, lengths[i]);
    if (run.length > 0 && (run.length + 1) * len * len * RUN_BYTES_PER_TOKEN_SQ > budgetBytes) {
      runs.push(run);
      run = [];
      longest = 0;
    }
    run.push(i);
    longest = Math.max(longest, lengths[i]);
  }
  if (run.length > 0) runs.push(run);
  return runs;
}

/** Run one task at a time, in call order. Concurrent session runs each allocate their own working
 * set, so a busy server would multiply the per-run budget by its concurrency. */
function serialQueue(): <T>(fn: () => Promise<T>) => Promise<T> {
  let tail: Promise<unknown> = Promise.resolve();
  return (fn) => {
    const result = tail.then(fn);
    tail = result.catch(() => undefined);
    return result;
  };
}

function isOomError(e: unknown): boolean {
  const m = String((e as any)?.message ?? e).toLowerCase();
  return m.includes("memory") || m.includes("cuda") || m.includes("out of memory") || m.includes("oom");
}
/** A model bundle could not be loaded (for example, the Hub rejected the token). */
export class LayaLoadError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "LayaLoadError";
  }
}

export interface NodeBundle {
  dir: string;
  cfg: any;
  tokenizerJson: unknown | null;
  /** Commit SHA the artifacts came from (pinned/requested, or the hub's `x-repo-commit`); null for local dirs. */
  revision: string | null;
}

export async function loadNodeBundle(
  modelDirOrRepo: string,
  opts?: {
    subfolder?: string | null;
    localDir?: string;
    token?: string | null;
    revision?: string | null;
    expectedSha256?: Record<string, string>;
  },
): Promise<NodeBundle> {
  const fs: typeof import("node:fs/promises") = await import("node:fs/promises");
  const path: typeof import("node:path") = await import("node:path");
  const os: typeof import("node:os") = await import("node:os");
  const sub = opts?.subfolder ?? null;
  let dir = opts?.localDir ?? modelDirOrRepo;
  let resolvedRevision: string | null = null;
  try {
    const st = await fs.stat(sub ? path.join(dir, sub) : dir);
    if (st.isDirectory()) dir = sub ? path.join(dir, sub) : dir;
    else dir = path.dirname(dir);
  } catch {
    // An explicit revision joins the cache key so differently-pinned artifacts never collide;
    // otherwise the existing Hub-default cache is reused.
    const revision = resolveRevision(modelDirOrRepo, opts?.revision);
    resolvedRevision = revision;
    const cache = path.join(
      os.homedir(),
      ".cache",
      "laya-ts",
      "hf",
      modelDirOrRepo.replace(/\//g, "__"),
      sub ?? "root",
      ...(revision && revision !== "main" ? [revision] : []),
    );
    await fs.mkdir(cache, { recursive: true });
    const token =
      opts?.token ?? (typeof process !== "undefined" ? (process as any).env?.["HF_TOKEN"] : undefined);
    for (const f of ["rl_agent_config.json", "tokenizer.json", "tokenizer/tokenizer.json", "encoder.onnx", "head.onnx"]) {
      try {
        await fs.stat(path.join(cache, f));
      } catch {
        const url = `https://huggingface.co/${modelDirOrRepo}/resolve/${revision ?? "main"}/${sub ? sub + "/" : ""}${f}`;
        const res = await fetch(url, token ? { headers: { Authorization: `Bearer ${token}` } } : undefined);
        const commit = res.headers?.get?.("x-repo-commit");
        if (commit) resolvedRevision = commit;
        if (!res.ok) {
          // A bad or missing token must never read as a wrong model or a skipped file.
          if (res.status === 401 || res.status === 403) {
            throw new LayaLoadError(
              `HF auth failed for ${modelDirOrRepo} (HTTP ${res.status} on ${f}); check HF_TOKEN or opts.token, and that the repo exists.`,
            );
          }
          if (f === "rl_agent_config.json") {
            throw new Error(
              `Incompatible model: ${JSON.stringify(modelDirOrRepo)} does not contain 'rl_agent_config.json'.`,
            );
          }
          continue;
        }
        const target = path.join(cache, f);
        await fs.mkdir(path.dirname(target), { recursive: true });
        // Write-then-rename so an interrupted download never leaves a truncated
        // artifact that later loads treat as complete.
        const tmp = `${target}.tmp-${typeof process !== "undefined" ? process.pid : 0}`;
        await fs.writeFile(tmp, new Uint8Array(await res.arrayBuffer()));
        await fs.rename(tmp, target);
      }
    }
    dir = cache;
  }
  // Opt-in integrity check over the resolved directory (covers local dirs, warm cache,
  // and fresh downloads alike) before any artifact is parsed or executed.
  if (opts?.expectedSha256) {
    const { createHash } = await import("node:crypto");
    for (const [rel, want] of Object.entries(opts.expectedSha256)) {
      const norm = normaliseDigestPath(rel);
      let buf: Uint8Array;
      try {
        buf = new Uint8Array(await fs.readFile(path.join(dir, norm)));
      } catch {
        throw new Error(`laya-ts: cannot verify ${JSON.stringify(rel)}: not found under ${dir}`);
      }
      const got = createHash("sha256").update(buf).digest("hex");
      if (got.toLowerCase() !== String(want).trim().toLowerCase()) {
        throw new Error(
          `laya-ts: SHA-256 mismatch for ${rel}: expected ${want}, got ${got}. Refusing to load the artifact.`,
        );
      }
    }
  }
  let cfg: any = {};
  try {
    cfg = JSON.parse(await fs.readFile(path.join(dir, "rl_agent_config.json"), "utf8"));
  } catch {
    throw new Error(
      `Incompatible model: ${JSON.stringify(modelDirOrRepo)} does not contain 'rl_agent_config.json'.`,
    );
  }
  let tokenizerJson: unknown | null = null;
  for (const candidate of ["tokenizer.json", "tokenizer/tokenizer.json"]) {
    try {
      tokenizerJson = JSON.parse(await fs.readFile(path.join(dir, candidate), "utf8"));
      break;
    } catch {
      // Try the next supported Hugging Face layout.
    }
  }
  return { dir, cfg, tokenizerJson, revision: resolvedRevision };
}

export async function createNodeProvider(
  modelDir: string,
  opts?: ProviderOptions,
): Promise<SessionProvider> {
  const spec = "onnxruntime-" + "node";
  const ort: any = await import(/* @vite-ignore */ spec);
  const threads = threadOptions(opts?.numThreads);
  const fs: typeof import("node:fs/promises") = await import("node:fs/promises");
  const path: typeof import("node:path") = await import("node:path");
  for (const f of ["encoder.onnx", "head.onnx"]) {
    const p = path.join(modelDir, f);
    try {
      await fs.stat(p);
    } catch {
      throw new Error(`Incompatible model: '${f}' not found in ${JSON.stringify(modelDir)} (expected ${p}).`);
    }
    if (opts?.expectedSha256) await expectDigest(f, await fs.readFile(p), opts.expectedSha256);
  }
  const dev = String(opts?.device ?? "cpu").toLowerCase();
  const want = dev === "cuda" ? "cuda" : dev === "dml" ? "dml" : "cpu";
  const make = async (ep: string) => {
    const e = await ort.InferenceSession.create(`${modelDir}/encoder.onnx`, {
      executionProviders: [ep],
      ...threads,
    });
    const h = await ort.InferenceSession.create(`${modelDir}/head.onnx`, {
      executionProviders: ["cpu"],
      ...threads,
    });
    return { e, h };
  };
  let enc: any;
  let head: any;
  let activeEP = want;
  try {
    ({ e: enc, h: head } = await make(want));
  } catch (e) {
    if (want !== "cpu") {
      console.warn(`Warning: ${want.toUpperCase()} requested but not available. Falling back to CPU.`);
      ({ e: enc, h: head } = await make("cpu"));
      activeEP = "cpu";
    } else {
      throw e;
    }
  }
  let cpuEnc: any = null;
  let cpuHead: any = null;
  const ensureCpu = async () => {
    if (!cpuEnc) {
      cpuEnc = await ort.InferenceSession.create(`${modelDir}/encoder.onnx`, {
        executionProviders: ["cpu"],
        ...threads,
      });
      cpuHead = await ort.InferenceSession.create(`${modelDir}/head.onnx`, {
        executionProviders: ["cpu"],
        ...threads,
      });
    }
    return { cpuEnc, cpuHead };
  };
  const runWithCpuFallback = async <T>(fn: (e: any, h: any) => Promise<T>): Promise<T> => {
    try {
      return await fn(enc, head);
    } catch (e) {
      if (activeEP !== "cpu" && isOomError(e)) {
        console.warn("Warning: GPU memory exceeded during inference. Falling back to CPU...");
        const { cpuEnc: ce, cpuHead: ch } = await ensureCpu();
        enc = ce;
        head = ch;
        activeEP = "cpu";
        return await fn(enc, head);
      }
      if (isOomError(e)) {
        throw new Error(`${(e as Error).message} (GPU out of memory; try device: "cpu")`);
      }
      throw e;
    }
  };
  const serial = serialQueue();
  const budget = () => runBudgetBytes(opts?.runMemoryMb, activeEP === "cpu");
  return {
    runEncoder: async (b) =>
      runWithCpuFallback(async (e) => {
        const lens = usedLengths(b.attentionMask);
        const lastHidden: number[][][] = new Array(b.inputIds.length);
        for (const rows of planRuns(lens, budget())) {
          const len = maxOf(rows.map((i) => lens[i]), 1);
          const chunk = {
            inputIds: rows.map((i) => b.inputIds[i].slice(0, len)),
            attentionMask: rows.map((i) => b.attentionMask[i].slice(0, len)),
          } as Batch;
          const out = await serial<Record<string, any>>(() => e.run(feed(ort, chunk)));
          const t = pickOutput(out, ["last_hidden_state", "lastHidden", "hidden_states"]);
          const nested: number[][][] = toNested(t.data, t.dims);
          // Keep each row's real tokens only: the head reads its own row length from the mask.
          rows.forEach((row, j) => {
            lastHidden[row] = nested[j].slice(0, lens[row]);
          });
        }
        return { lastHidden };
      }),
    runHead: async (h, b) =>
      runWithCpuFallback(async (_e, hd) => {
        const hidden = h as number[][][];
        const lens = usedLengths(b.attentionMask);
        const logits: number[][] = new Array(b.markerPos.length);
        const act: number[][] = new Array(b.markerPos.length);
        for (const rows of planRuns(lens, budget())) {
          const len = maxOf(rows.map((i) => lens[i]), 1);
          const chunk: Batch = {
            inputIds: [],
            attentionMask: rows.map((i) => b.attentionMask[i]),
            markerPos: rows.map((i) => b.markerPos[i]),
            markerMask: rows.map((i) => b.markerMask[i]),
            qtype: rows.map((i) => b.qtype[i]),
          };
          const out = await serial<Record<string, any>>(() => hd.run(feedHead(ort, rows.map((i) => hidden[i]), chunk, len)));
          const vals = Object.values(out) as any[];
          const lt = pickOutput(out, ["logits"]);
          const at = pickOutput(out, ["act_logits", "act"]) ?? vals[1] ?? vals[0];
          const lNested: number[][] = toNested(lt.data, lt.dims);
          const aNested: number[][] = toNested(at.data, at.dims);
          rows.forEach((row, j) => {
            logits[row] = lNested[j];
            act[row] = aNested[j];
          });
        }
        return { logits, act };
      }),
    release: async () => {
      const sessions = [...new Set([enc, head, cpuEnc, cpuHead].filter(Boolean))];
      enc = head = cpuEnc = cpuHead = null;
      await Promise.all(sessions.map((s) => s.release()));
    },
  };
}
