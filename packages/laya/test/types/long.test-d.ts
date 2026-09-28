import { describe, expectTypeOf, it } from "vitest";
import { z } from "zod";
import {
  defineQuestions,
  type LayaAgent,
  type LayaRouter,
  type RouteDecision,
  type RoutedPredictResult,
  type TypedRouterRequest,
  type WindowInfo,
} from "../../src/index.js";

declare const agent: LayaAgent;
declare const router: LayaRouter;

const questions = defineQuestions({
  department: { type: "choice", instructions: "Which team?", criteria: ["billing", "technical", "other"] },
  churn: { type: "noul", instructions: "Might the customer leave?" },
});
const other = defineQuestions({ tier: { type: "choice", instructions: "Plan?", criteria: { free: "", pro: "" } } });

describe("decideBatch", () => {
  const schema = z.object({ urgent: z.boolean(), tier: z.enum(["a", "b"]) });

  it("infers an array of the schema output", async () => {
    const out = await agent.decideBatch(["x", "y"], schema, { batchSize: 4 });
    expectTypeOf(out).toEqualTypeOf<{ urgent: boolean; tier: "a" | "b" }[]>();
    const routed = await router.decideBatch(["x"], schema, { sortByLength: true });
    expectTypeOf(routed).toEqualTypeOf<{ urgent: boolean; tier: "a" | "b" }[]>();
  });

  it("does not take minConfidence", async () => {
    // @ts-expect-error an abstained field would be null, which the schema output cannot hold
    await agent.decideBatch(["x"], schema, { minConfidence: 0.5 });
  });
});

describe("predictLong", () => {
  it("narrows answers and adds the window and window count", async () => {
    const r = await agent.predictLong("a long document", questions, { window: 128, stride: 64, batchSize: 8 });
    expectTypeOf(r.answers.department.choice).toEqualTypeOf<"billing" | "technical" | "other">();
    expectTypeOf(r.answers.churn.window).toEqualTypeOf<WindowInfo | undefined>();
    expectTypeOf(r.usage.windows).toEqualTypeOf<number>();
    // @ts-expect-error predictLong sizes windows itself; maxLen is not an option
    await agent.predictLong("x", questions, { maxLen: 128 });
    // @ts-expect-error only "auto" aggregation exists
    await agent.predictLong("x", questions, { aggregate: "mean" });
  });

  it("keeps routing on router results", async () => {
    const r = await router.predictLong("a long document", questions, { lang: "de", model: "multilingual" });
    expectTypeOf(r.routing).toEqualTypeOf<RouteDecision>();
    expectTypeOf(r.answers.department.window).toEqualTypeOf<WindowInfo | undefined>();
  });
});

describe("Router batch", () => {
  it("narrows each result to its own request's questions", async () => {
    const [a, b] = await router.predictBatch([
      { state: "x", questions },
      { state: "y", questions: other, lang: "de", maxLen: 256 },
    ]);
    expectTypeOf(a.answers.department.choice).toEqualTypeOf<"billing" | "technical" | "other">();
    expectTypeOf(b.answers.tier.choice).toEqualTypeOf<"free" | "pro">();
    // @ts-expect-error `tier` is not a question of the first request
    void a.answers.tier;
  });

  it("maps a homogeneous array to an array", async () => {
    const reqs: TypedRouterRequest<typeof questions>[] = [{ state: "x", questions }];
    const rs = await router.predictBatch(reqs, { batchSize: 4, minConfidence: 0.6 });
    expectTypeOf(rs).toEqualTypeOf<RoutedPredictResult<typeof questions>[]>();
  });

  it("routes without loading and reports revisions", () => {
    expectTypeOf(router.routeBatch([{ state: "x", questions }])).toEqualTypeOf<RouteDecision[]>();
    expectTypeOf(router.loadedRevisions).toEqualTypeOf<Readonly<Record<string, string | null>>>();
  });
});
