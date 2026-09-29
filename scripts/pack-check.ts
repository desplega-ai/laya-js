// Packs both published packages the way the publish workflow does (`bun pm pack`, which rewrites
// `workspace:*` to the real version) and fails if a tarball would not install cleanly from npm.
// Publishes nothing. Needs a prior `bun run build`.
//
//   bun scripts/pack-check.ts                        # pack to .pack/ and check
//   bun scripts/pack-check.ts --expect-tag v0.1.0    # also require both versions to equal the tag
//   bun scripts/pack-check.ts --dest <dir>           # keep the tarballs somewhere else
import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";

type Manifest = {
  name: string;
  version: string;
  private?: boolean;
  license?: string;
  main?: string;
  types?: string;
  exports?: unknown;
  repository?: { url?: string; directory?: string };
  publishConfig?: { access?: string };
  dependencies?: Record<string, string>;
  optionalDependencies?: Record<string, string>;
  peerDependencies?: Record<string, string>;
};

// Publish order: laya-server depends on laya, so laya must be on the registry first.
const PACKAGES = [
  { dir: "packages/laya", name: "@desplega.ai/laya" },
  { dir: "packages/laya-server", name: "@desplega.ai/laya-server" },
];
const REPOSITORY_URL = "git+https://github.com/desplega-ai/laya-js.git";
const SEMVER = /^\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?$/;

const root = resolve(import.meta.dirname, "..");
const args = process.argv.slice(2);
const option = (flag: string) => {
  const i = args.indexOf(flag);
  return i === -1 ? undefined : args[i + 1];
};
const dest = resolve(root, option("--dest") ?? ".pack");
const expectTag = option("--expect-tag");

const run = (cmd: string, cmdArgs: string[], cwd: string) => {
  const r = spawnSync(cmd, cmdArgs, { cwd, encoding: "utf8", maxBuffer: 64 * 1024 * 1024 });
  if (r.status !== 0) throw new Error(`${cmd} ${cmdArgs.join(" ")} (in ${cwd}) exited ${r.status}\n${r.stderr}`);
  return r.stdout;
};

// Flatten `exports` (string | nested conditions) into the file paths it points at.
const exportTargets = (value: unknown): string[] =>
  typeof value === "string"
    ? [value]
    : value && typeof value === "object"
      ? Object.values(value).flatMap(exportTargets)
      : [];

const errors: string[] = [];
const fail = (pkg: string, message: string) => errors.push(`${pkg}: ${message}`);

mkdirSync(dest, { recursive: true });
const packed: { name: string; version: string; manifest: Manifest; files: string[]; tarball: string }[] = [];

for (const { dir, name } of PACKAGES) {
  const cwd = resolve(root, dir);
  if (!existsSync(resolve(cwd, "dist"))) {
    fail(name, `${dir}/dist is missing; run \`bun run build\` first`);
    continue;
  }
  // `bun pm pack` prints the tarball path as the last non-empty line of stdout.
  const out = run("bun", ["pm", "pack", "--destination", dest], cwd);
  const tarball = out
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l.endsWith(".tgz"))
    .pop();
  if (!tarball) {
    fail(name, `could not find the tarball path in \`bun pm pack\` output:\n${out}`);
    continue;
  }
  const files = run("tar", ["-tzf", tarball], cwd)
    .split("\n")
    .filter(Boolean)
    .map((f) => f.replace(/^package\//, ""));
  const manifest = JSON.parse(run("tar", ["-xzOf", tarball, "package/package.json"], cwd)) as Manifest;
  packed.push({ name, version: manifest.version, manifest, files, tarball });
}

for (const { name, manifest, files } of packed) {
  if (manifest.name !== name) fail(name, `packed name is ${manifest.name}`);
  if (manifest.private) fail(name, "packed manifest is still private");
  if (!SEMVER.test(manifest.version)) fail(name, `version "${manifest.version}" is not semver`);
  if (manifest.license !== "Apache-2.0") fail(name, `license is ${manifest.license}`);
  if (manifest.publishConfig?.access !== "public") fail(name, "publishConfig.access is not public");
  if (manifest.repository?.url !== REPOSITORY_URL)
    fail(name, `repository.url is ${manifest.repository?.url}; provenance needs ${REPOSITORY_URL}`);

  // A dependency that npm cannot resolve makes the package uninstallable.
  for (const field of ["dependencies", "optionalDependencies", "peerDependencies"] as const) {
    for (const [dep, range] of Object.entries(manifest[field] ?? {})) {
      if (/^(workspace|file|link|portal):/.test(range)) fail(name, `${field}.${dep} is "${range}" in the tarball`);
    }
  }

  for (const required of ["LICENSE", "NOTICE", "package.json"]) {
    if (!files.includes(required)) fail(name, `tarball has no ${required}`);
  }
  const entries = [manifest.main, manifest.types, ...exportTargets(manifest.exports)].filter(
    (p): p is string => typeof p === "string",
  );
  for (const entry of entries) {
    const path = entry.replace(/^\.\//, "");
    if (!files.includes(path)) fail(name, `entry point ${entry} is not in the tarball`);
  }
  const stray = files.filter(
    (f) => f !== "package.json" && !/^(dist\/|LICENSE$|NOTICE$)/.test(f) && !/^README/i.test(f),
  );
  if (stray.length) fail(name, `unexpected files in the tarball: ${stray.join(", ")}`);
  const junk = files.filter((f) => f.endsWith(".tsbuildinfo"));
  if (junk.length) fail(name, `build cache in the tarball: ${junk.join(", ")}`);
}

const [runtime, server] = packed;
if (runtime && server) {
  if (runtime.version !== server.version) {
    fail("versions", `laya is ${runtime.version} but laya-server is ${server.version}; they release in lockstep`);
  }
  const range = server.manifest.dependencies?.[runtime.name];
  if (range !== runtime.version) {
    fail(server.name, `depends on ${runtime.name}@${range}, expected exactly ${runtime.version}`);
  }
}
if (expectTag) {
  for (const { name, version } of packed) {
    if (`v${version}` !== expectTag) fail(name, `version ${version} does not match tag ${expectTag}`);
  }
}

for (const { name, version, manifest, files, tarball } of packed) {
  const deps = Object.entries(manifest.dependencies ?? {})
    .map(([d, r]) => `${d}@${r}`)
    .join(", ");
  console.log(`${name}@${version}: ${files.length} files, ${tarball}\n  dependencies: ${deps}`);
}
if (errors.length) {
  console.error(`pack-check: ${errors.length} problem(s)\n${errors.map((e) => `  - ${e}`).join("\n")}`);
  process.exit(1);
}
console.log(`pack-check: ok (${packed.map((p) => `${p.name}@${p.version}`).join(", ")})`);
