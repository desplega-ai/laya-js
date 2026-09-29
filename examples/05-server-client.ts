// Calling laya-server over HTTP from any language; this client is plain `fetch`.
// With LAYA_URL set it talks to that server (LAYA_API_KEY if it needs a token); otherwise it
// starts a local one from the repo build (`bun run build` first) on a free port.
//   LAYA_MODEL_DIR=/path/to/bundles bun examples/05-server-client.ts
import { type ChildProcess, spawn } from "node:child_process";
import { ms, timed } from "./_lib.ts";

const apiKey = process.env.LAYA_API_KEY ?? "example-key";
let base = process.env.LAYA_URL;
let child: ChildProcess | undefined;

if (!base) {
  const port = 18000 + Math.floor(Math.random() * 1000);
  child = spawn("node", [new URL("../packages/laya-server/dist/main.js", import.meta.url).pathname], {
    env: {
      ...process.env,
      LAYA_PORT: String(port),
      LAYA_MODELS: "english,multilingual",
      LAYA_API_KEY: apiKey,
      LAYA_LOG_LEVEL: "warning",
    },
    stdio: ["ignore", "inherit", "inherit"],
  });
  base = `http://127.0.0.1:${port}`;
}

interface Reply {
  routing: { model: string };
  answers: Record<string, { choice: string; noul: number }>;
  detail?: string;
}

async function post(body: unknown, key: string | null = apiKey) {
  const res = await fetch(`${base}/v1/systemone`, {
    method: "POST",
    headers: { "content-type": "application/json", ...(key ? { authorization: `Bearer ${key}` } : {}) },
    body: JSON.stringify(body),
  });
  return { status: res.status, inference: res.headers.get("x-inference-time-ms"), json: (await res.json()) as Reply };
}

try {
  // The server answers 503 on /health until the checkpoints are loaded.
  const t0 = performance.now();
  for (;;) {
    const h = await fetch(`${base}/health`).catch(() => null);
    if (h?.ok) {
      const health = (await h.json()) as { status: string; loaded: string[] };
      console.log(`/health ${health.status}, loaded ${health.loaded.join(", ")} after ${ms(performance.now() - t0)}`);
      break;
    }
    if (performance.now() - t0 > 120_000) throw new Error("server did not become ready in 120 s");
    await new Promise((r) => setTimeout(r, 250));
  }

  const questions = {
    intent: {
      type: "choice",
      instructions: "What does the customer want in `message`?",
      criteria: {
        refund: "money returned",
        technical_help: "a bug or outage",
        cancellation: "cancel the subscription",
        information: "pricing or how-to",
      },
    },
    urgent: { type: "noul", instructions: "Does `message` communicate time pressure or a deadline?" },
  };

  // 1. One request, English text: routed to the english checkpoint.
  const en = await post({
    state: { message: "The export has been stuck at 0% for an hour and the board meeting is at 3pm." },
    questions,
  });
  console.log(
    `\nEnglish  -> ${en.status} model=${en.json.routing.model} intent=${en.json.answers.intent.choice} urgent=${en.json.answers.urgent.noul.toFixed(2)} server-side ${en.inference} ms`,
  );

  // 2. Spanish: same endpoint, routed to multilingual on its own.
  const es = await post({
    state: { message: "Me habéis cobrado dos veces el mismo pedido, quiero que me devolváis el dinero." },
    questions,
  });
  console.log(
    `Spanish  -> ${es.status} model=${es.json.routing.model} intent=${es.json.answers.intent.choice} server-side ${es.inference} ms`,
  );

  // 3. Eight requests at once. The server admits up to LAYA_MAX_CONCURRENT and serialises inference.
  const burst = await timed(() =>
    Promise.all(
      Array.from({ length: 8 }, (_, i) =>
        post({ state: { message: `Order #${1000 + i}: the package never arrived, please refund.` }, questions }),
      ),
    ),
  );
  console.log(
    `8 concurrent -> statuses ${[...new Set(burst.value.map((r) => r.status))].join(",")}, wall ${ms(burst.ms)}`,
  );

  // 4. Errors are plain HTTP: a bad token, a malformed question and a missing field.
  const noAuth = await post({ state: "x", questions }, "wrong-key");
  const badQ = await post({ state: "x", questions: { q: { type: "choice", instructions: "?", criteria: [] } } });
  const noQ = await post({ state: "x" });
  console.log(`\nwrong token -> ${noAuth.status} ${JSON.stringify(noAuth.json)}`);
  console.log(`empty choice -> ${badQ.status} ${JSON.stringify(badQ.json)}`);
  console.log(`no questions -> ${noQ.status} ${JSON.stringify(noQ.json)}`);
} finally {
  child?.kill("SIGTERM");
}
