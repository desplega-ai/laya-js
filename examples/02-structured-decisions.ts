// Structured decisions: describe the outcome you want as a zod schema, get a validated, typed
// object back. Enums become choice questions, booleans yes/no, bounded integers a score scale.
//   LAYA_MODEL_DIR=/path/to/bundles bun examples/02-structured-decisions.ts
import { SchemaError } from "@desplega/laya";
import { z } from "zod";
import { loadAgent, ms, timed } from "./_lib.ts";

// `.describe()` text becomes the question the model reads, so write it like an instruction.
const Decision = z.object({
  department: z
    .enum(["billing", "technical", "sales", "account"])
    .describe(
      "Which team should handle the ticket: billing (charges, invoices), technical (bugs, outages, API errors), sales (pricing, quotes), account (login, deletion)?",
    ),
  priority: z
    .number()
    .int()
    .min(0)
    .max(2)
    .describe(
      "How urgent is the ticket? 0 = routine, 1 = needs an answer today, 2 = customer is blocked or losing money now",
    ),
  needs_human: z.boolean().describe("Does the customer threaten to leave or need a manager, not a template answer?"),
  refund_requested: z.boolean().describe("Does the customer explicitly ask for money back?"),
});
type Decision = z.infer<typeof Decision>;

// The state can be plain text or a structured object; laya renders both.
const tickets: { subject: string; body: string; plan: string; expect: Decision }[] = [
  {
    subject: "Double charge",
    body: "Two identical charges of $49 appeared on my statement on 3 March. Please reverse one.",
    plan: "pro",
    expect: { department: "billing", priority: 1, needs_human: false, refund_requested: true },
  },
  {
    subject: "Production down",
    body: "All API calls fail with 503 since 09:00 UTC. Our checkout depends on it and we lose orders every minute.",
    plan: "enterprise",
    expect: { department: "technical", priority: 2, needs_human: false, refund_requested: false },
  },
  {
    subject: "Team pricing",
    body: "We are 25 people evaluating a switch. Can you send a quote for annual billing?",
    plan: "trial",
    expect: { department: "sales", priority: 0, needs_human: false, refund_requested: false },
  },
  {
    subject: "Close my account",
    body: "Please delete my account and all my data, I stopped using the product.",
    plan: "free",
    expect: { department: "account", priority: 0, needs_human: false, refund_requested: false },
  },
  {
    subject: "Invoice copy",
    body: "Could you resend the invoice for February? Our accountant lost it.",
    plan: "pro",
    expect: { department: "billing", priority: 0, needs_human: false, refund_requested: false },
  },
  {
    subject: "Threatening to leave",
    body: "The export has failed for the fourth week in a row. Refund this month or I will cancel and move to a competitor.",
    plan: "pro",
    expect: { department: "technical", priority: 1, needs_human: true, refund_requested: true },
  },
];

const agent = await loadAgent("english");

const batch = await timed(() => agent.decideBatch(tickets, Decision));
const decisions: Decision[] = batch.value; // already validated by zod, so the type is honest
let fields = 0;
let right = 0;
tickets.forEach((t, i) => {
  const d = decisions[i];
  const wrong = (Object.keys(t.expect) as (keyof Decision)[]).filter((k) => d[k] !== t.expect[k]);
  fields += 4;
  right += 4 - wrong.length;
  console.log(
    `${t.subject.padEnd(22)} ${JSON.stringify(d)}${wrong.length ? `   <- differs on ${wrong.join(", ")}` : ""}`,
  );
});
console.log(`\nfield accuracy ${right}/${fields}`);
console.log(`decideBatch of ${tickets.length}: ${ms(batch.ms)}`);

// Application code branches on typed fields instead of parsing text.
for (const [i, d] of decisions.entries()) {
  if (d.needs_human || d.priority === 2)
    console.log(`escalate: ${tickets[i].subject} -> ${d.department}, priority ${d.priority}`);
}

// A schema laya cannot answer from a fixed option set is refused with the path, before any inference.
try {
  await agent.decide(tickets[0], z.object({ summary: z.string() }));
} catch (e) {
  if (!(e instanceof SchemaError)) throw e;
  console.log(`\nfree text is rejected: ${e.message}`);
}
await agent.dispose();
