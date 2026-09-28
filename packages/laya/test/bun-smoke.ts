// Bun smoke test: agent-swarm runs on Bun, so the lib must import and
// onnxruntime-node must create a session under Bun. Run: bun run test:bun-smoke
import assert from "node:assert/strict";
import { resolve } from "node:path";

if (typeof Bun === "undefined") throw new Error("bun-smoke must run under Bun");

const laya = await import("@desplega/laya");
assert.equal(typeof laya.Agent, "function", "Agent export missing");
assert.equal(typeof laya.loadNodeBundle, "function", "loadNodeBundle export missing");

const ort = await import("onnxruntime-node");
const session = await ort.InferenceSession.create(resolve(import.meta.dir, "fixtures/tiny.onnx"));
const out = await session.run({ x: new ort.Tensor("float32", Float32Array.from([1, 2]), [1, 2]) });
assert.deepEqual(Array.from(out.y.data as Float32Array), [2, 3]);
await session.release();

console.log(`bun-smoke: ok (bun ${Bun.version}, laya VERSION ${laya.VERSION}, onnxruntime-node session ran)`);
