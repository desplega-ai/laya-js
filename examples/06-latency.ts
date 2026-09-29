// Latency check for capacity planning: single `predict` calls on a support ticket, 3 questions.
// Numbers depend on the CPU, on LAYA_THREADS and on what else the host runs; measure on yours.
//   LAYA_MODEL_DIR=/path/to/bundles LAYA_THREADS=4 bun examples/06-latency.ts [checkpoint ...]

import { cpus } from "node:os";
import { type CheckpointName, defineQuestions } from "@desplega/laya";
import { loadAgent, ms, timed } from "./_lib.ts";

const questions = defineQuestions({
  team: {
    type: "choice",
    instructions: "Which team handles `message`?",
    criteria: ["billing", "technical", "sales", "account"],
  },
  urgent: { type: "noul", instructions: "Does `message` say the customer is blocked or has a deadline?" },
  frustration: {
    type: "score",
    instructions: "How frustrated is the customer?",
    criteria: ["calm", "annoyed", "angry", "furious"],
  },
});
const message =
  "Our webhook endpoint has returned 500 for every event since 09:00 UTC. Production orders are not syncing and we ship in two hours.";

const checkpoints = (
  process.argv.slice(2).length ? process.argv.slice(2) : ["english", "multilingual"]
) as CheckpointName[];
const runs = 30;
console.log(
  `cpu: ${cpus()[0].model.trim()} x${cpus().length}, LAYA_THREADS=${process.env.LAYA_THREADS ?? "unset"}, runs=${runs}`,
);

for (const checkpoint of checkpoints) {
  const load = await timed(() => loadAgent(checkpoint));
  const agent = load.value;
  for (let i = 0; i < 3; i++) await agent.predict(message, questions); // warm-up
  const samples: number[] = [];
  for (let i = 0; i < runs; i++) samples.push((await timed(() => agent.predict(message, questions))).ms);
  samples.sort((a, b) => a - b);
  const q = (p: number) => samples[Math.min(runs - 1, Math.floor(p * runs))];
  console.log(
    `${checkpoint.padEnd(16)} load ${ms(load.ms)}  predict p50 ${ms(q(0.5))}  p95 ${ms(q(0.95))}  min ${ms(samples[0])}`,
  );
  await agent.dispose();
}
