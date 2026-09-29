// Support inbox triage: classify a batch of real-looking tickets in one shared forward pass.
//   LAYA_MODEL_DIR=/path/to/bundles LAYA_THREADS=4 bun examples/01-support-triage.ts
// LAYA_CHECKPOINT=multilingual runs the same tickets on the smaller checkpoint.
import { type CheckpointName, defineQuestions } from "@desplega/laya";
import { loadAgent, ms, pct, timed } from "./_lib.ts";

// defineQuestions keeps the literal labels, so `answers.team.choice` is a union, not a string.
const questions = defineQuestions({
  team: {
    type: "choice",
    instructions: "Which team should handle the ticket in `message`?",
    criteria: {
      billing: "charges, invoices, refunds, payment methods",
      technical: "bugs, outages, errors, integrations, API problems",
      sales: "pricing, quotes, plan upgrades, new purchases",
      account: "login, password, user access, account deletion",
    },
  },
  urgent: { type: "noul", instructions: "Does `message` say the customer is blocked or has a deadline?" },
  frustration: {
    type: "score",
    instructions: "How frustrated does the customer sound in `message`?",
    criteria: ["calm", "mildly annoyed", "clearly angry", "furious or threatening to leave"],
  },
});

const tickets: { message: string; team: "billing" | "technical" | "sales" | "account"; urgent: boolean }[] = [
  {
    message: "I was charged twice for the Pro plan on March 3rd. Please refund the duplicate.",
    team: "billing",
    urgent: false,
  },
  {
    message:
      "Our webhook endpoint has returned 500 for every event since 09:00 UTC. Production orders are not syncing and we ship in two hours.",
    team: "technical",
    urgent: true,
  },
  {
    message:
      "How much would 40 seats on the Business plan cost with annual billing? We would like a quote before Friday.",
    team: "sales",
    urgent: false,
  },
  {
    message: "I can't log in anymore, the reset email never arrives and I have a client demo in 30 minutes.",
    team: "account",
    urgent: true,
  },
  {
    message:
      "The API returns 429 even though we are under the documented rate limit. Started after yesterday's deploy.",
    team: "technical",
    urgent: false,
  },
  {
    message: "Please update the card on file to the one ending 4417, the old one expires this month.",
    team: "billing",
    urgent: false,
  },
  {
    message:
      "This is the third time your export job has failed. I am done with this product, cancel everything and refund me.",
    team: "billing",
    urgent: true,
  },
  {
    message: "Can you delete my account and all associated data? I no longer use the service.",
    team: "account",
    urgent: false,
  },
  {
    message: "Do you offer a discount for non-profits? We are a 12-person charity evaluating your tool.",
    team: "sales",
    urgent: false,
  },
  {
    message: "The dashboard shows a blank page after the latest update, tried Chrome and Firefox.",
    team: "technical",
    urgent: false,
  },
];

const agent = await loadAgent((process.env.LAYA_CHECKPOINT ?? "english") as CheckpointName);
console.log(`loaded ${agent.checkpoint} (${agent.precision})\n`);

const batch = await timed(() =>
  agent.predictBatch(
    tickets.map((t) => t.message),
    questions,
  ),
);

let teamHits = 0;
let urgentHits = 0;
batch.value.forEach((r, i) => {
  const t = tickets[i];
  const a = r.answers;
  const team = a.team.choice; // typed: "billing" | "technical" | "sales" | "account"
  const urgent = a.urgent.noul >= 0.5;
  teamHits += team === t.team ? 1 : 0;
  urgentHits += urgent === t.urgent ? 1 : 0;
  console.log(
    `${team === t.team ? "ok  " : "MISS"} team=${team.padEnd(9)} (${pct(a.team.probabilities[team])}) ` +
      `urgent=${pct(a.urgent.noul).padStart(4)} frustration=${a.frustration.score} | ${t.message.slice(0, 60)}`,
  );
});
console.log(`\nteam accuracy ${teamHits}/${tickets.length}, urgent accuracy ${urgentHits}/${tickets.length}`);
console.log(`batch of ${tickets.length}: ${ms(batch.ms)} (${ms(batch.ms / tickets.length)} per ticket)`);

const seq = await timed(async () => {
  for (const t of tickets) await agent.predict(t.message, questions);
});
console.log(`same ${tickets.length} one by one: ${ms(seq.ms)} (${ms(seq.ms / tickets.length)} per ticket)`);
await agent.dispose();
