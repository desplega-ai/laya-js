// Bun smoke test: agent-swarm runs on Bun, so the lib must import and
// onnxruntime-node must create a session under Bun. Run: bun run test:bun-smoke
import assert from "node:assert/strict";
import { resolve } from "node:path";

if (typeof Bun === "undefined") throw new Error("bun-smoke must run under Bun");

const laya = await import("@desplega.ai/laya");
assert.equal(typeof laya.createAgent, "function", "createAgent export missing");
const raw = await import("@desplega.ai/laya/raw");
assert.equal(typeof raw.Agent, "function", "Agent export missing");
assert.equal(typeof raw.loadNodeBundle, "function", "loadNodeBundle export missing");

const ort = await import("onnxruntime-node");
const session = await ort.InferenceSession.create(resolve(import.meta.dir, "fixtures/tiny.onnx"));
const out = await session.run({ x: new ort.Tensor("float32", Float32Array.from([1, 2]), [1, 2]) });
assert.deepEqual(Array.from(out.y.data as Float32Array), [2, 3]);
await session.release();

// With LAYA_SMOKE_CHECKPOINT (and HF_TOKEN for the private store), also run the typed API on a
// real fp32 bundle: createAgent -> predict -> dispose.
const checkpoint = process.env.LAYA_SMOKE_CHECKPOINT;
if (checkpoint) {
  const agent = await laya.createAgent({ checkpoint: checkpoint as laya.CheckpointName });
  const r = await agent.predict("My invoice was charged twice this month, please refund one.", {
    department: {
      type: "choice",
      instructions: "Which team should handle `message`?",
      criteria: ["billing", "technical", "other"],
    },
  });
  assert.ok(["billing", "technical", "other"].includes(r.answers.department.choice), "choice outside the labels");
  const total = Object.values(r.answers.department.probabilities).reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(total - 1) < 1e-3, `probabilities sum to ${total}`);
  await agent.dispose();
  console.log(`bun-smoke: ${checkpoint} fp32 predict -> ${r.answers.department.choice}, disposed`);
}

console.log(`bun-smoke: ok (bun ${Bun.version}, laya VERSION ${laya.VERSION}, onnxruntime-node session ran)`);
