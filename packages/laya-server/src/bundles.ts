// Local fp32 bundle store: find a checkpoint's bundle in the baked model dir or the download
// cache, or fetch it from the artifact store and verify every file against the pinned SHA-256
// map in `@desplega/laya` (artifacts.ts). Used by main.ts (checkpoints not baked into the
// image) and by fetch-models.ts (the Docker models stage and the k8s initContainer).
import { createHash, randomBytes } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import { mkdir, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import type { ReadableStream as NodeWebReadableStream } from "node:stream/web";
import { ARTIFACTS, type CheckpointName, type Precision } from "@desplega/laya";

/** Written next to the files once every one of them verified. */
export const MANIFEST = "laya-bundle.json";

export interface BundleSpec {
  checkpoint: string;
  precision: string;
  repo: string;
  revision: string;
  /** Path inside the artifact repo, `<checkpoint>/<precision>`. */
  subfolder: string;
  sha256: Record<string, string>;
  bytes: Record<string, number>;
}

export interface BundleManifest {
  checkpoint: string;
  precision: string;
  repo: string;
  revision: string;
  subfolder: string;
  sha256: Record<string, string>;
}

export function bundleSpec(
  checkpoint: CheckpointName,
  precision: Precision,
  source: { repo: string; revision: string },
): BundleSpec {
  const a = ARTIFACTS[checkpoint][precision];
  return { checkpoint, precision, ...source, subfolder: a.subfolder, sha256: a.sha256, bytes: a.bytes };
}

/** `<root>/<checkpoint>/<precision>`, the layout the image and the k8s volumes use. */
export function bundleDir(root: string, checkpoint: string, precision: string): string {
  return join(root, checkpoint, precision);
}

export async function sha256File(path: string): Promise<string> {
  const h = createHash("sha256");
  await pipeline(createReadStream(path), h);
  return h.digest("hex");
}

async function exists(path: string): Promise<boolean> {
  try {
    await stat(path);
    return true;
  } catch {
    return false;
  }
}

export async function readManifest(dir: string): Promise<BundleManifest | null> {
  try {
    return JSON.parse(await readFile(join(dir, MANIFEST), "utf8")) as BundleManifest;
  } catch {
    return null;
  }
}

function sameDigests(a: Record<string, string>, b: Record<string, string>): boolean {
  const ka = Object.keys(a).sort();
  const kb = Object.keys(b).sort();
  return ka.length === kb.length && ka.every((k, i) => k === kb[i] && a[k].toLowerCase() === b[k].toLowerCase());
}

/** True when every bundle file is present in `dir` (no manifest needed: a hand-placed bundle counts). */
export async function hasBundleFiles(dir: string, spec: Pick<BundleSpec, "sha256">): Promise<boolean> {
  for (const f of Object.keys(spec.sha256)) if (!(await exists(join(dir, f)))) return false;
  return true;
}

/**
 * Whether `dir` holds a verified copy of `spec`: a manifest with the same digests, and every
 * file at its pinned size. With `rehash`, every file's SHA-256 is recomputed too.
 */
export async function isVerified(dir: string, spec: BundleSpec, opts: { rehash?: boolean } = {}): Promise<boolean> {
  const m = await readManifest(dir);
  if (!m || !sameDigests(m.sha256, spec.sha256)) return false;
  for (const [f, want] of Object.entries(spec.sha256)) {
    let size: number;
    try {
      size = (await stat(join(dir, f))).size;
    } catch {
      return false;
    }
    if (spec.bytes[f] !== undefined && size !== spec.bytes[f]) return false;
    if (opts.rehash && (await sha256File(join(dir, f))) !== want.toLowerCase()) return false;
  }
  return true;
}

export class BundleFetchError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "BundleFetchError";
  }
}

export interface FetchOptions {
  /** Read token for the private artifact store. Sent as a bearer header, never logged. */
  token?: string | null;
  /** Hub base URL (default https://huggingface.co). */
  endpoint?: string;
  fetch?: typeof fetch;
  log?: (msg: string) => void;
}

/**
 * Download every file of `spec` into `dir`, hashing while streaming, and keep a file only when
 * its SHA-256 matches. Files land via write-then-rename, and the manifest is written last, so an
 * interrupted fetch never leaves a bundle that looks complete.
 */
export async function fetchBundle(spec: BundleSpec, dir: string, opts: FetchOptions = {}): Promise<void> {
  const endpoint = (opts.endpoint ?? "https://huggingface.co").replace(/\/+$/, "");
  const doFetch = opts.fetch ?? fetch;
  const log = opts.log ?? (() => {});
  await mkdir(dir, { recursive: true });
  await rm(join(dir, MANIFEST), { force: true });
  for (const [file, want] of Object.entries(spec.sha256)) {
    const target = join(dir, file);
    const size = spec.bytes[file];
    if (
      (await exists(target)) &&
      (size === undefined || (await stat(target)).size === size) &&
      (await sha256File(target)) === want.toLowerCase()
    ) {
      log(`${spec.checkpoint}/${spec.precision}: ${file} already present and verified`);
      continue;
    }
    const url = `${endpoint}/${spec.repo}/resolve/${spec.revision}/${spec.subfolder}/${file}`;
    log(`${spec.checkpoint}/${spec.precision}: fetching ${file}${size ? ` (${(size / 1e6).toFixed(1)} MB)` : ""}`);
    const res = await doFetch(url, opts.token ? { headers: { Authorization: `Bearer ${opts.token}` } } : undefined);
    if (!res.ok || !res.body) {
      await res.body?.cancel().catch(() => {});
      if (res.status === 401 || res.status === 403) {
        throw new BundleFetchError(
          `artifact store auth failed for ${spec.repo} (HTTP ${res.status} on ${spec.subfolder}/${file}); ` +
            "check that HF_TOKEN is set and can read the repo",
        );
      }
      throw new BundleFetchError(
        `fetching ${spec.subfolder}/${file} from ${spec.repo}@${spec.revision}: HTTP ${res.status}`,
      );
    }
    // Unique per writer: replicas sharing a cache volume often run with the same PID.
    const tmp = `${target}.part-${process.pid}-${randomBytes(4).toString("hex")}`;
    const h = createHash("sha256");
    try {
      const body = Readable.fromWeb(res.body as unknown as NodeWebReadableStream<Uint8Array>);
      await pipeline(
        body,
        async function* (src: AsyncIterable<Buffer>) {
          for await (const chunk of src) {
            h.update(chunk);
            yield chunk;
          }
        },
        createWriteStream(tmp),
      );
      const got = h.digest("hex");
      if (got !== want.toLowerCase()) {
        throw new BundleFetchError(
          `SHA-256 mismatch for ${spec.subfolder}/${file}: expected ${want}, got ${got}; refusing to keep it`,
        );
      }
      await rename(tmp, target);
    } finally {
      await rm(tmp, { force: true });
    }
  }
  const manifest: BundleManifest = {
    checkpoint: spec.checkpoint,
    precision: spec.precision,
    repo: spec.repo,
    revision: spec.revision,
    subfolder: spec.subfolder,
    sha256: spec.sha256,
  };
  await writeFile(join(dir, MANIFEST), `${JSON.stringify(manifest, null, 2)}\n`);
}

export interface ResolvedBundle {
  dir: string;
  /** Artifact-store commit, from the bundle's manifest; null for a bundle placed by hand. */
  revision: string | null;
  source: "model-dir" | "cache" | "fetched";
}

export interface StoreOptions extends FetchOptions {
  modelDir: string;
  cacheDir: string;
  precision: Precision;
  repo: string;
  revision: string;
}

/**
 * Resolve a checkpoint to a local bundle dir: the baked `modelDir` first, then a verified copy
 * in `cacheDir`, else fetch it into `cacheDir`. Concurrent calls for one checkpoint share a fetch.
 */
export function createBundleStore(opts: StoreOptions) {
  const pending = new Map<string, Promise<ResolvedBundle>>();

  async function resolveOnce(checkpoint: CheckpointName): Promise<ResolvedBundle> {
    const spec = bundleSpec(checkpoint, opts.precision, { repo: opts.repo, revision: opts.revision });
    const baked = bundleDir(opts.modelDir, checkpoint, opts.precision);
    if (await hasBundleFiles(baked, spec)) {
      return { dir: baked, revision: (await readManifest(baked))?.revision ?? null, source: "model-dir" };
    }
    const cached = bundleDir(opts.cacheDir, checkpoint, opts.precision);
    if (await isVerified(cached, spec)) {
      return { dir: cached, revision: (await readManifest(cached))?.revision ?? null, source: "cache" };
    }
    await fetchBundle(spec, cached, opts);
    return { dir: cached, revision: spec.revision, source: "fetched" };
  }

  return {
    resolve(checkpoint: CheckpointName): Promise<ResolvedBundle> {
      let p = pending.get(checkpoint);
      if (!p) {
        p = resolveOnce(checkpoint).finally(() => pending.delete(checkpoint));
        pending.set(checkpoint, p);
      }
      return p;
    },
  };
}
