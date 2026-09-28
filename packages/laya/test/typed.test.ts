import { describe, expect, it } from "vitest";
import { z } from "zod";
import {
  CHECKPOINTS,
  checkQuestion,
  createAgent,
  createRouter,
  defineQuestions,
  jsonSchemaOf,
  SchemaError,
  validateDecision,
} from "../src/index.js";

describe("defineQuestions", () => {
  it("returns the map unchanged", () => {
    const q = { a: { type: "choice", instructions: "?", criteria: ["x", "y"] } } as const;
    expect(defineQuestions(q)).toBe(q);
  });

  it.each([
    ["no instructions", { type: "choice", criteria: ["x"] }],
    ["empty choice", { type: "choice", instructions: "?", criteria: [] }],
    ["score dict", { type: "score", instructions: "?", criteria: { a: "b" } }],
    ["labels on choice", { type: "choice", instructions: "?", criteria: ["x"], labels: { false: "n", true: "y" } }],
    ["unknown type", { type: "rank", instructions: "?" }],
  ])("rejects %s with the checkQuestion error", (_name, def) => {
    let expected = "";
    try {
      checkQuestion("q", def);
    } catch (e) {
      expected = (e as Error).message;
    }
    expect(expected).not.toBe("");
    expect(() => defineQuestions({ q: def } as never)).toThrow(expected);
  });
});

describe("Standard Schema decide helpers", () => {
  const schema = z.object({ urgent: z.boolean(), tier: z.enum(["a", "b"]) });

  it("exports a JSON schema from a Zod schema", () => {
    const js = jsonSchemaOf(schema);
    expect(js.type).toBe("object");
    expect(Object.keys(js.properties as object).sort()).toEqual(["tier", "urgent"]);
  });

  it("throws SchemaError for a schema without a JSON Schema export", () => {
    const bare = { "~standard": { version: 1, vendor: "bare", validate: (v: unknown) => ({ value: v }) } } as const;
    expect(() => jsonSchemaOf(bare)).toThrow(SchemaError);
    expect(() => jsonSchemaOf(bare)).toThrow(/"bare" has no JSON Schema export/);
  });

  it("returns the validated value", async () => {
    await expect(validateDecision(schema, { urgent: true, tier: "a" })).resolves.toEqual({ urgent: true, tier: "a" });
  });

  it("rejects a decided value that fails the schema", async () => {
    await expect(validateDecision(schema, { urgent: "yes", tier: "c" })).rejects.toThrow(SchemaError);
    await expect(validateDecision(schema, { urgent: true, tier: "c" })).rejects.toThrow(/tier:/);
  });
});

describe("factories", () => {
  it("maps every checkpoint to the pinned artifact store", () => {
    expect(Object.keys(CHECKPOINTS).sort()).toEqual(["english", "multilingual", "typed-decisions"]);
    for (const info of Object.values(CHECKPOINTS)) {
      expect(info.repo).toBe("desplega/laya-onnx");
      expect(info.revision).toMatch(/^[0-9a-f]{40}$/);
      expect(Object.keys(info.sha256).sort()).toEqual([
        "encoder.onnx",
        "head.onnx",
        "rl_agent_config.json",
        "tokenizer.json",
      ]);
    }
  });

  it("rejects precisions other than fp32 and unknown checkpoints before loading", async () => {
    await expect(createAgent({ checkpoint: "multilingual", precision: "int8" as never })).rejects.toThrow(
      /only "fp32"/,
    );
    await expect(createAgent({ checkpoint: "french" as never })).rejects.toThrow(/unknown checkpoint/);
    await expect(createRouter({ checkpoints: ["french" as never] })).rejects.toThrow(/unknown checkpoint/);
  });
});
