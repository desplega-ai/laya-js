// Fails if a vendored file differs from the pinned upstream without the
// "Modified by Desplega Labs" header, or if UPSTREAM.md does not list it.
// Upstream hashes live in vendor.json, so the check needs no network.
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

type Vendor = {
  upstream: { repo: string; sha: string };
  files: Record<string, { from: string; sha256: string }>;
  removed: Record<string, string>;
};

const root = resolve(import.meta.dirname, "..");
const vendor = JSON.parse(readFileSync(resolve(root, "vendor.json"), "utf8")) as Vendor;
const upstreamMd = readFileSync(resolve(root, "UPSTREAM.md"), "utf8");
const HEADER = /Modified by Desplega Labs, 2026: \S/;

const errors: string[] = [];
let unchanged = 0;
let modified = 0;

for (const [path, { from, sha256 }] of Object.entries(vendor.files)) {
  const abs = resolve(root, path);
  if (!existsSync(abs)) {
    if (!(path in vendor.removed)) errors.push(`${path}: missing, and not listed under "removed" in vendor.json`);
    else if (!upstreamMd.includes(`\`${path}\``)) errors.push(`${path}: removed, but UPSTREAM.md does not list it`);
    continue;
  }
  const bytes = readFileSync(abs);
  if (createHash("sha256").update(bytes).digest("hex") === sha256) {
    unchanged++;
    continue;
  }
  modified++;
  const head = bytes.toString("utf8").split("\n").slice(0, 5).join("\n");
  if (!HEADER.test(head))
    errors.push(
      `${path}: differs from upstream ${from} but has no "Modified by Desplega Labs, 2026: <summary>" header`,
    );
  if (!upstreamMd.includes(`\`${path}\``)) errors.push(`${path}: modified, but UPSTREAM.md does not list it`);
}

for (const path of Object.keys(vendor.removed)) {
  if (!(path in vendor.files)) errors.push(`${path}: listed as removed but is not a vendored file`);
}

if (errors.length) {
  console.error(`check-vendor: ${errors.length} problem(s)\n${errors.map((e) => `  - ${e}`).join("\n")}`);
  process.exit(1);
}
console.log(
  `check-vendor: ok (upstream ${vendor.upstream.sha.slice(0, 8)}; ${unchanged} unchanged, ${modified} modified, ${Object.keys(vendor.removed).length} removed)`,
);
