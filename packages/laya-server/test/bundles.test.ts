import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readdir, readFile, rm, writeFile } from "node:fs/promises";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";
import {
  type BundleSpec,
  bundleDir,
  createBundleStore,
  fetchBundle,
  isVerified,
  MANIFEST,
  readManifest,
} from "../src/bundles.js";
import { parseFetchArgs } from "../src/fetch-models.js";

const FILES: Record<string, Buffer> = {
  "rl_agent_config.json": Buffer.from('{"max_len": 512}'),
  "tokenizer.json": Buffer.from('{"model": {}}'),
  "encoder.onnx": Buffer.alloc(300_000, 7),
  "head.onnx": Buffer.alloc(1234, 3),
};
const sha = (b: Buffer) => createHash("sha256").update(b).digest("hex");
const TOKEN = "hf_test_token_value";

function spec(overrides: Partial<BundleSpec> = {}): BundleSpec {
  return {
    checkpoint: "multilingual",
    precision: "fp32",
    repo: "desplega/laya-onnx",
    revision: "rev1",
    subfolder: "multilingual/fp32",
    sha256: Object.fromEntries(Object.entries(FILES).map(([f, b]) => [f, sha(b)])),
    bytes: Object.fromEntries(Object.entries(FILES).map(([f, b]) => [f, b.length])),
    ...overrides,
  };
}

let server: Server;
let endpoint: string;
let requests: Array<{ url: string; auth: string | undefined }>;
let tamper: string | null = null;

beforeAll(async () => {
  server = createServer((req, res) => {
    requests.push({ url: req.url ?? "", auth: req.headers.authorization });
    if (req.headers.authorization !== `Bearer ${TOKEN}`) {
      res.writeHead(401).end();
      return;
    }
    const m = /^\/desplega\/laya-onnx\/resolve\/rev1\/multilingual\/fp32\/(.+)$/.exec(req.url ?? "");
    const body = m ? FILES[m[1]] : undefined;
    if (!body) {
      res.writeHead(404).end();
      return;
    }
    res.writeHead(200, { "content-length": body.length });
    res.end(m?.[1] === tamper ? Buffer.alloc(body.length, 0) : body);
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  endpoint = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});
afterAll(() => new Promise<void>((r) => server.close(() => r())));

let dir: string;
beforeEach(async () => {
  requests = [];
  tamper = null;
  dir = await mkdtemp(join(tmpdir(), "laya-bundles-"));
  return () => rm(dir, { recursive: true, force: true });
});

describe("fetchBundle", () => {
  it("downloads, verifies and writes the manifest last", async () => {
    const target = join(dir, "multilingual", "fp32");
    await fetchBundle(spec(), target, { token: TOKEN, endpoint });
    for (const [f, b] of Object.entries(FILES)) expect(await readFile(join(target, f))).toEqual(b);
    expect(await readManifest(target)).toMatchObject({ repo: "desplega/laya-onnx", revision: "rev1" });
    expect(await isVerified(target, spec(), { rehash: true })).toBe(true);
    expect(requests.map((r) => r.url)).toContain("/desplega/laya-onnx/resolve/rev1/multilingual/fp32/encoder.onnx");
    // Only the finished files and the manifest remain.
    expect((await readdir(target)).sort()).toEqual([...Object.keys(FILES), MANIFEST].sort());
  });

  it("refuses a file whose SHA-256 does not match and leaves no partial file", async () => {
    tamper = "encoder.onnx";
    const target = join(dir, "b");
    await expect(fetchBundle(spec(), target, { token: TOKEN, endpoint })).rejects.toThrow(
      /SHA-256 mismatch for multilingual\/fp32\/encoder\.onnx/,
    );
    const left = await readdir(target);
    expect(left).not.toContain("encoder.onnx");
    expect(left).not.toContain(MANIFEST);
    expect(left.filter((f) => f.includes(".part-"))).toEqual([]);
  });

  it("names the auth problem without ever printing the token", async () => {
    const err = await fetchBundle(spec(), join(dir, "c"), { token: "hf_wrong_secret", endpoint }).catch((e) => e);
    expect(err.message).toMatch(/auth failed.*HTTP 401.*HF_TOKEN/);
    expect(err.message).not.toContain("hf_wrong_secret");
  });

  it("reuses verified files from an interrupted fetch", async () => {
    const target = join(dir, "d");
    await mkdir(target, { recursive: true });
    await writeFile(join(target, "encoder.onnx"), FILES["encoder.onnx"]);
    await fetchBundle(spec(), target, { token: TOKEN, endpoint });
    expect(requests.map((r) => r.url).some((u) => u.endsWith("encoder.onnx"))).toBe(false);
    expect(await isVerified(target, spec())).toBe(true);
  });
});

describe("isVerified", () => {
  it("needs a matching manifest, the pinned sizes, and with rehash the digests", async () => {
    const target = join(dir, "e");
    expect(await isVerified(target, spec())).toBe(false);
    await fetchBundle(spec(), target, { token: TOKEN, endpoint });
    const other = spec({ sha256: { ...spec().sha256, "head.onnx": "0".repeat(64) } });
    expect(await isVerified(target, other)).toBe(false);
    // Same size, different bytes: only a rehash catches it.
    await writeFile(join(target, "head.onnx"), Buffer.alloc(FILES["head.onnx"].length, 9));
    expect(await isVerified(target, spec())).toBe(true);
    expect(await isVerified(target, spec(), { rehash: true })).toBe(false);
    await writeFile(join(target, "head.onnx"), Buffer.alloc(5));
    expect(await isVerified(target, spec())).toBe(false);
  });
});

describe("createBundleStore", () => {
  it("prefers a bundle baked into the model dir, reporting its manifest revision", async () => {
    const modelDir = join(dir, "models");
    const baked = bundleDir(modelDir, "multilingual", "fp32");
    await mkdir(baked, { recursive: true });
    for (const f of ["rl_agent_config.json", "tokenizer.json", "encoder.onnx", "head.onnx"]) {
      await writeFile(join(baked, f), "x");
    }
    await writeFile(join(baked, MANIFEST), JSON.stringify({ revision: "baked-rev" }));
    const store = createBundleStore({
      modelDir,
      cacheDir: join(dir, "cache"),
      precision: "fp32",
      repo: "desplega/laya-onnx",
      revision: "rev1",
      endpoint,
    });
    expect(await store.resolve("multilingual")).toEqual({ dir: baked, revision: "baked-rev", source: "model-dir" });
    expect(requests).toEqual([]);
  });
});

describe("parseFetchArgs", () => {
  it("defaults to LAYA_MODELS and LAYA_CACHE_DIR", () => {
    expect(parseFetchArgs([], { LAYA_CACHE_DIR: "/cache", LAYA_MODELS: "english,ml" })).toEqual({
      dest: "/cache",
      checkpoints: ["english", "multilingual"],
      rehash: true,
    });
  });

  it("takes --dest, --no-rehash and positional or comma-separated checkpoints", () => {
    expect(parseFetchArgs(["--dest", "/models", "--no-rehash", "english,typed", "ml"], {})).toEqual({
      dest: "/models",
      checkpoints: ["english", "typed-decisions", "multilingual"],
      rehash: false,
    });
  });

  it("rejects unknown checkpoints, options and precisions", () => {
    expect(() => parseFetchArgs(["gpt"], {})).toThrow(/unknown model/);
    expect(() => parseFetchArgs(["--token", "x"], {})).toThrow(/unknown option --token/);
    expect(() => parseFetchArgs(["--precision", "int8"], {})).toThrow(/fp32/);
  });
});
