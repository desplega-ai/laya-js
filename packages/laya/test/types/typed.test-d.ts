import { describe, expectTypeOf, it } from "vitest";
import { z } from "zod";
import { defineQuestions, type LayaAgent, type LayaRouter } from "../../src/index.js";

declare const agent: LayaAgent;
declare const router: LayaRouter;

const questions = defineQuestions({
  department: {
    type: "choice",
    instructions: "Which team owns `message`?",
    criteria: ["billing", "technical", "other"],
  },
  tier: {
    type: "choice",
    instructions: "Which plan is the customer on?",
    criteria: { free: "no paid plan", pro: "the pro plan" },
  },
  urgency: { type: "score", instructions: "How urgent is it?", criteria: ["not urgent", "soon", "now"] },
  churn: { type: "noul", instructions: "Might the customer leave?" },
});

describe("typed answers", () => {
  it("narrows choice answers to the label union", async () => {
    const r = await agent.predict("my invoice is wrong", questions);
    expectTypeOf(r.answers.department.choice).toEqualTypeOf<"billing" | "technical" | "other">();
    expectTypeOf(r.answers.department.probabilities).toEqualTypeOf<Record<"billing" | "technical" | "other", number>>();
    expectTypeOf(r.answers.tier.choice).toEqualTypeOf<"free" | "pro">();
    expectTypeOf(r.answers.urgency.score).toEqualTypeOf<number>();
    expectTypeOf(r.answers.churn.noul).toEqualTypeOf<number>();
    expectTypeOf(r.answers.churn.low_confidence).toEqualTypeOf<true | undefined>();
    // @ts-expect-error "sales" is not a label of `department`
    void (r.answers.department.choice === "sales");
    // @ts-expect-error `nope` is not a question id
    void r.answers.nope;
  });

  it("narrows inline question maps without defineQuestions", async () => {
    const r = await agent.predict("x", { q: { type: "choice", instructions: "?", criteria: ["a", "b"] } });
    expectTypeOf(r.answers.q.choice).toEqualTypeOf<"a" | "b">();
  });

  it("keeps routing on router results", async () => {
    const r = await router.predict("x", questions);
    expectTypeOf(r.answers.department.choice).toEqualTypeOf<"billing" | "technical" | "other">();
    expectTypeOf(r.routing.model).toBeString();
  });
});

describe("typed decide", () => {
  it("infers the Standard Schema output", async () => {
    const out = await agent.decide("x", z.object({ urgent: z.boolean(), tier: z.enum(["a", "b"]) }));
    expectTypeOf(out).toEqualTypeOf<{ urgent: boolean; tier: "a" | "b" }>();
    const routed = await router.decide("x", z.object({ urgent: z.boolean() }));
    expectTypeOf(routed).toEqualTypeOf<{ urgent: boolean }>();
  });
});
