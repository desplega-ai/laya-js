// Host and runtime facts every eval result records.
import { createRequire } from "node:module";
import { cpus } from "node:os";
import { dirname, resolve } from "node:path";

const lib = resolve(dirname(new URL(import.meta.url).pathname), "..");

export function ortVersion(): string {
  try {
    return (createRequire(resolve(lib, "package.json"))("onnxruntime-node/package.json") as { version: string })
      .version;
  } catch {
    return "unknown";
  }
}

export function hostInfo(): { cpu: string; cores: number; node: string; bun: string | null; onnxruntime: string } {
  return {
    cpu: cpus()[0]?.model ?? "unknown",
    cores: cpus().length,
    node: process.versions.node,
    bun: process.versions.bun ?? null,
    onnxruntime: ortVersion(),
  };
}
