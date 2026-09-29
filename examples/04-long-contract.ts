// Long document: a 1,000-word master services agreement is longer than the model's window.
// `predict` reads the start of it; `predictLong` scans overlapping windows and aggregates.
//   LAYA_MODEL_DIR=/path/to/bundles bun examples/04-long-contract.ts
import { readFileSync } from "node:fs";
import { defineQuestions } from "@desplega.ai/laya";
import { loadAgent, ms, pct, timed } from "./_lib.ts";

const contract = readFileSync(new URL("./data/msa.txt", import.meta.url), "utf8");

const questions = defineQuestions({
  auto_renews: {
    type: "noul",
    instructions: "Does the contract renew automatically unless a party gives notice of non-renewal?",
  },
  breach_notification: {
    type: "noul",
    instructions:
      "Does the provider have to notify the customer of a personal data breach within a fixed number of hours?",
  },
  governing_law: {
    type: "choice",
    instructions: "Which law governs the contract?",
    criteria: {
      england_and_wales: "the laws of England and Wales",
      new_york: "the laws of the State of New York",
      california: "the laws of the State of California",
      other: "any other law or not stated",
    },
  },
});

// Ground truth, read off the text: section 11, section 7, section 16.
const truth = { auto_renews: true, breach_notification: true, governing_law: "new_york" } as const;

const agent = await loadAgent("multilingual");
const show = (
  label: string,
  a: {
    auto_renews: { noul: number };
    breach_notification: { noul: number };
    governing_law: { choice: string; probabilities: Record<string, number> };
  },
) => {
  const law = a.governing_law.choice;
  console.log(
    `${label.padEnd(12)} auto_renews=${pct(a.auto_renews.noul).padStart(4)}  breach_notification=${pct(a.breach_notification.noul).padStart(4)}  ` +
      `governing_law=${law} (${pct(a.governing_law.probabilities[law])})`,
  );
};

console.log(`contract: ${contract.split(/\s+/).length} words`);
console.log(`truth:        auto_renews=true  breach_notification=true  governing_law=${truth.governing_law}\n`);

const short = await timed(() => agent.predict(contract, questions));
show("predict", short.value.answers);
console.log(
  `             usage.input_tokens (all question rows): ${short.value.usage.input_tokens}, ${ms(short.ms)}\n`,
);

const long = await timed(() => agent.predictLong(contract, questions));
show("predictLong", long.value.answers);
console.log(`             windows scored: ${long.value.usage.windows}, ${ms(long.ms)}`);
for (const [qid, a] of Object.entries(long.value.answers)) {
  if (a.window)
    console.log(
      `             ${qid}: decided by window ${a.window.index} (tokens ${a.window.token_start}-${a.window.token_end})`,
    );
}
await agent.dispose();
