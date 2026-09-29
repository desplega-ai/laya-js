// Loads the exported fp32 bundles from the public artifact store and runs one
// systemOne call on each. Opt-in, since each bundle is 1.3 to 1.7 GB:
//   LAYA_ARTIFACTS=multilingual bun run test -- artifacts
// LAYA_ARTIFACTS takes a comma list of checkpoints, or "all".
import { describe, expect, it } from "vitest";
import { ARTIFACT_REPO, ARTIFACT_REVISION, ARTIFACTS, type CheckpointName } from "../src/artifacts.js";
import { Agent, triageQuestions } from "../src/raw.js";

const ALL = Object.keys(ARTIFACTS) as CheckpointName[];
const wanted = (process.env.LAYA_ARTIFACTS ?? "")
  .split(",")
  .map((s) => s.trim())
  .filter(Boolean);
const selected = wanted.includes("all") ? ALL : ALL.filter((c) => wanted.includes(c));

describe("artifact manifest", () => {
  it("pins every checkpoint at fp32 with a digest for every file", () => {
    expect(ARTIFACT_REVISION).toMatch(/^[0-9a-f]{40}$/);
    for (const ckpt of ALL) {
      const a = ARTIFACTS[ckpt].fp32;
      expect(a.subfolder).toBe(`${ckpt}/fp32`);
      expect(Object.keys(a.sha256).sort()).toEqual([
        "encoder.onnx",
        "head.onnx",
        "rl_agent_config.json",
        "tokenizer.json",
      ]);
      for (const digest of Object.values(a.sha256)) expect(digest).toMatch(/^[0-9a-f]{64}$/);
    }
  });
});

describe.skipIf(selected.length === 0)("artifact bundles (real weights)", () => {
  it.each(selected)(
    "%s fp32 loads with expectedSha256 and answers one systemOne call",
    async (ckpt) => {
      const a = ARTIFACTS[ckpt].fp32;
      const agent = await Agent.load(ARTIFACT_REPO, {
        subfolder: a.subfolder,
        revision: ARTIFACT_REVISION,
        expectedSha256: a.sha256,
      });
      const r = await agent.systemOne("My invoice was charged twice this month, please refund.", triageQuestions());
      const answers = Object.values(r.answers);
      expect(answers.length).toBeGreaterThan(0);
      for (const ans of answers) {
        const probs = Object.values((ans as { probabilities?: Record<string, number> }).probabilities ?? {});
        for (const p of probs) expect(p).toBeGreaterThanOrEqual(0);
      }
    },
    600_000,
  );
});
