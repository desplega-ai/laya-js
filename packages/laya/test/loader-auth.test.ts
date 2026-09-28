import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { LayaLoadError, loadNodeBundle } from "../src/index.js";

class StubResponse {
  constructor(
    public body: string | null,
    public status = 200,
  ) {}
  get ok() {
    return this.status >= 200 && this.status < 300;
  }
  headers = { get: () => null };
  async arrayBuffer() {
    return new TextEncoder().encode(this.body ?? "").buffer as ArrayBuffer;
  }
}

const REPO = "desplega/laya-onnx-auth-test";
const cacheRoot = path.join(os.homedir(), ".cache", "laya-ts", "hf", REPO.replace(/\//g, "__"));

function stubHub(statusFor: (file: string) => number) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      const file = url.split("/resolve/main/multilingual/fp32/")[1] ?? "";
      const status = statusFor(file);
      return new StubResponse(status === 200 ? '{"act_costs":{}}' : null, status);
    }),
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
  fs.rmSync(cacheRoot, { recursive: true, force: true });
});

describe("loadNodeBundle HF auth errors", () => {
  it.each([401, 403])("maps %i on rl_agent_config.json to LayaLoadError, not 'Incompatible model'", async (status) => {
    stubHub(() => status);
    const err = await loadNodeBundle(REPO, { subfolder: "multilingual/fp32", token: "bad" }).catch((e) => e);
    expect(err).toBeInstanceOf(LayaLoadError);
    expect(err.message).toContain(`HF auth failed for ${REPO}`);
    expect(err.message).not.toContain("Incompatible model");
    expect(err.message).not.toContain("bad");
  });

  it("maps a 401 on a later file to LayaLoadError instead of skipping it", async () => {
    stubHub((file) => (file === "encoder.onnx" ? 401 : file === "rl_agent_config.json" ? 200 : 404));
    const err = await loadNodeBundle(REPO, { subfolder: "multilingual/fp32" }).catch((e) => e);
    expect(err).toBeInstanceOf(LayaLoadError);
    expect(err.message).toContain("encoder.onnx");
  });

  it("keeps 'Incompatible model' for a 404 on rl_agent_config.json", async () => {
    stubHub(() => 404);
    const err = await loadNodeBundle(REPO, { subfolder: "multilingual/fp32" }).catch((e) => e);
    expect(err).not.toBeInstanceOf(LayaLoadError);
    expect(err.message).toContain("Incompatible model");
  });
});
