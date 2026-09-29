// Mixed-language inbox: one router, two checkpoints. English goes to `english`, everything else
// to `multilingual`; the router loads each checkpoint once and batches per checkpoint.
//   LAYA_MODEL_DIR=/path/to/bundles bun examples/03-multilingual-router.ts
import { defineQuestions } from "@desplega/laya";
import { loadRouter, ms, pct, timed } from "./_lib.ts";

const questions = defineQuestions({
  intent: {
    type: "choice",
    instructions: "What does the customer want in `message`?",
    criteria: {
      refund: "money returned or a duplicate charge reversed",
      technical_help: "a bug, outage or integration problem",
      cancellation: "wants to cancel the subscription",
      information: "asks for pricing, features or how-to",
    },
  },
  urgent: { type: "noul", instructions: "Does `message` communicate time pressure or a deadline?" },
});

const inbox: {
  lang: string;
  message: string;
  intent: "refund" | "technical_help" | "cancellation" | "information";
  urgent: boolean;
}[] = [
  {
    lang: "en",
    message: "You charged my card twice for the same order, please send the money back.",
    intent: "refund",
    urgent: false,
  },
  {
    lang: "es",
    message: "Me habéis cobrado dos veces el mismo pedido, necesito que me devolváis el dinero.",
    intent: "refund",
    urgent: false,
  },
  {
    lang: "de",
    message: "Die App stürzt seit dem Update bei jedem Start ab, ich kann nicht mehr arbeiten.",
    intent: "technical_help",
    urgent: false,
  },
  {
    lang: "fr",
    message: "Je voudrais résilier mon abonnement à la fin du mois, merci de confirmer.",
    intent: "cancellation",
    urgent: false,
  },
  {
    lang: "pt",
    message: "Quanto custa o plano anual para uma equipe de dez pessoas?",
    intent: "information",
    urgent: false,
  },
  { lang: "hi", message: "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।", intent: "refund", urgent: false },
  {
    lang: "ja",
    message: "ログインするとエラーが出て、ダッシュボードが開けません。急ぎで対応をお願いします。",
    intent: "technical_help",
    urgent: true,
  },
  { lang: "en", message: "Does the Team plan include SSO and audit logs?", intent: "information", urgent: false },
  { lang: "it", message: "Vorrei disdire l'abbonamento, non lo uso più.", intent: "cancellation", urgent: false },
  {
    lang: "en",
    message: "The CSV export has been stuck at 0% for an hour and the board meeting is at 3pm.",
    intent: "technical_help",
    urgent: true,
  },
];

const router = await loadRouter(["english", "multilingual"]);
const requests = inbox.map((m) => ({ state: { message: m.message }, questions }));

// Routing is free: no model is loaded until predictBatch runs.
const plan = router.routeBatch(requests);
console.log("routing plan (no model loaded yet):", router.loaded.length === 0);
for (const [i, d] of plan.entries()) console.log(`  ${inbox[i].lang} -> ${d.model.padEnd(12)} ${d.reason}`);

const run = await timed(() => router.predictBatch(requests));
console.log(
  `\nfirst predictBatch, includes loading both checkpoints: ${ms(run.ms)}; loaded: ${router.loaded.join(", ")}`,
);

let hits = 0;
let urgentHits = 0;
run.value.forEach((r, i) => {
  const intent = r.answers.intent.choice;
  hits += intent === inbox[i].intent ? 1 : 0;
  urgentHits += r.answers.urgent.noul >= 0.5 === inbox[i].urgent ? 1 : 0;
  console.log(
    `${intent === inbox[i].intent ? "ok  " : "MISS"} [${inbox[i].lang}] ${r.routing.model.padEnd(12)} ` +
      `intent=${intent.padEnd(14)} (${pct(r.answers.intent.probabilities[intent])}) urgent=${pct(r.answers.urgent.noul)}`,
  );
});
console.log(`\nintent accuracy ${hits}/${inbox.length}, urgent accuracy ${urgentHits}/${inbox.length}`);

const warm = await timed(() => router.predictBatch(requests));
console.log(`second predictBatch, both checkpoints warm: ${ms(warm.ms)} (${ms(warm.ms / inbox.length)} per message)`);
await router.dispose();
