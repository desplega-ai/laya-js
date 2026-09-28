import { describe, expectTypeOf, it } from "vitest";
import { z } from "zod";
import { defineQuestions, type LayaAgent, type LayaRouter, type PredictResult } from "../../src/index.js";

declare const agent: LayaAgent;
declare const router: LayaRouter;

const questions = defineQuestions({
  department: { type: "choice", instructions: "Which team?", criteria: ["billing", "technical", "other"] },
  churn: { type: "noul", instructions: "Might the customer leave?" },
});

describe("predictBatch", () => {
  it("returns one narrowed result per state", async () => {
    const rs = await agent.predictBatch(["a", "b"], questions, { batchSize: 8, sortByLength: true });
    expectTypeOf(rs).toEqualTypeOf<PredictResult<typeof questions>[]>();
    expectTypeOf(rs[0].answers.department.choice).toEqualTypeOf<"billing" | "technical" | "other">();
    expectTypeOf(rs[0].answers.churn.low_confidence).toEqualTypeOf<true | undefined>();
    // @ts-expect-error "sales" is not a label of `department`
    void (rs[0].answers.department.choice === "sales");
  });

  it("accepts a readonly state list", async () => {
    const states = ["a", { body: "b" }] as const;
    const rs = await agent.predictBatch(states, questions);
    expectTypeOf(rs[1].answers.churn.noul).toEqualTypeOf<number>();
  });

  it("rejects options it does not take", async () => {
    // @ts-expect-error batchSize is a number
    await agent.predictBatch(["a"], questions, { batchSize: "8" });
    // @ts-expect-error sortByLength is a boolean
    await agent.predictBatch(["a"], questions, { sortByLength: 1 });
  });
});

describe("per-call budget and abstention", () => {
  it("predict takes maxLen, headMaxLen and minConfidence", async () => {
    const r = await agent.predict("x", questions, { maxLen: 256, headMaxLen: 64, minConfidence: 0.7 });
    expectTypeOf(r.answers.department.low_confidence).toEqualTypeOf<true | undefined>();
    const routed = await router.predict("x", questions, { maxLen: 256, minConfidence: 0.7 });
    expectTypeOf(routed.answers.churn.low_confidence).toEqualTypeOf<true | undefined>();
    // @ts-expect-error maxLen is a number
    await agent.predict("x", questions, { maxLen: "256" });
  });

  it("typed decide does not take minConfidence, since an abstained field would be null", async () => {
    const schema = z.object({ urgent: z.boolean() });
    expectTypeOf(await agent.decide("x", schema, { maxLen: 128 })).toEqualTypeOf<{ urgent: boolean }>();
    // @ts-expect-error minConfidence is not a typed-decide option
    await agent.decide("x", schema, { minConfidence: 0.5 });
  });
});
