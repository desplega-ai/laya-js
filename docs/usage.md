# Using laya-js

Two ways to run laya decision models from TypeScript: call `@desplega.ai/laya` in your process, or call `@desplega.ai/laya-server` over HTTP. Every output below was produced by the scripts in [`examples/`](../examples) on the machine listed under [How the numbers were measured](#how-the-numbers-were-measured). Nothing is rounded up or re-run to look better.

## What laya does

You give it a state (text or a JSON object) and questions. It answers each question with a probability, not a generated string:

- `choice`: one label out of a fixed set, with a probability per label.
- `noul`: yes/no, as P(true).
- `score`: a level on an ordered scale, with a probability per level.

There is no generation, so a call is one encoder pass plus a small head: hundreds of milliseconds on a CPU, deterministic, and it can only answer with what you allowed.

## Install

Node 22+ or Bun 1.4+. Use the packages from npm (`npm install @desplega.ai/laya`, `@desplega.ai/laya-server`), or work from a clone:

```sh
git clone https://github.com/desplega-ai/laya-js && cd laya-js
bun install
bun run build
```

Checkpoints are fp32 ONNX bundles exported from upstream's public Hugging Face checkpoints. Sizes: `multilingual` 1.32 GB, `english` and `typed-decisions` 1.69 GB each. Two ways to get them:

- Export them yourself from the public checkpoints, no token needed: [tools/export](../tools/export/README.md#export-for-your-own-use).
- Download Desplega's prebuilt bundles from the public Hugging Face repo `desplega/laya-onnx`, pinned by revision and SHA-256. No token is needed (`HF_TOKEN` is optional, for rate limits or a private mirror):

  ```sh
  # Download once into a directory, verified against the pinned hashes
  node packages/laya-server/dist/fetch-models.js --dest ./bundles multilingual english
  ```

Either way, point every example, and your own code, at the directory:

```sh
export LAYA_MODEL_DIR=$PWD/bundles     # layout: <dir>/<checkpoint>/fp32/{encoder.onnx,head.onnx,...}
```

Without `modelDir`, `createAgent` fetches from the public artifact store (`HF_TOKEN` is optional; a rejected token raises `LayaLoadError`). `LAYA_THREADS` sets the ONNX thread count for the library as well as the server. `LAYA_RUN_MB` (default 256) caps the working memory of one ONNX run, in MiB: a call with many or long rows goes through the graphs in several runs instead of one, with the same answers. Raise it for fewer, larger runs; one row longer than the cap still runs, alone.

| Checkpoint | Encoder | Use it for |
| --- | --- | --- |
| `english` | ModernBERT-large, 512-token window | English text |
| `multilingual` | mmBERT-base, 1,024-token window, about 2x faster here | Non-English or mixed text, or when latency matters |
| `typed-decisions` | ModernBERT-large, fine-tuned | Upstream's four typed workflows; see [Limits](#limits) |

## Load a checkpoint and ask

```ts
import { createAgent, defineQuestions } from "@desplega.ai/laya";

const agent = await createAgent({ checkpoint: "english", modelDir: "./bundles/english/fp32" });

// defineQuestions validates once and keeps the literal labels for the types.
const questions = defineQuestions({
  team: { type: "choice", instructions: "Which team handles `message`?",
          criteria: { billing: "charges, invoices", technical: "bugs, outages", sales: "pricing" } },
  urgent: { type: "noul", instructions: "Is the customer blocked or on a deadline?" },
});

const r = await agent.predict({ message: "Production orders stopped syncing, we ship in two hours." }, questions);
r.answers.team.choice;            // "billing" | "technical" | "sales"   (not string)
r.answers.team.probabilities;     // Record<"billing" | "technical" | "sales", number>
r.answers.urgent.noul;            // number in [0, 1]
r.usage.input_tokens;
await agent.dispose();            // frees the ONNX sessions
```

## The calls

| Call | Use it when | Result |
| --- | --- | --- |
| `agent.predict(state, questions)` | one state | `PredictResult<Q>` |
| `agent.predictBatch(states, questions, { batchSize, sortByLength })` | many states, same questions | `PredictResult<Q>[]` in input order |
| `agent.predictLong(state, questions, { window, stride })` | state longer than the window | `LongPredictResult<Q>`; each answer names the deciding `window` |
| `agent.decide(state, zodSchema)` / `decideBatch(states, schema)` | you want a typed object, not raw probabilities | `z.infer<schema>`, validated |
| `createRouter({ checkpoints })` then `router.predict / predictBatch / predictLong / decide` | mixed languages | same shapes plus `routing: { model, reason }` |

Options that apply to the predict family: `maxLen` and `headMaxLen` (per-call token budget, up to 8192), `lang`, `hooks`, and `minConfidence` (an answer below the threshold is returned with `low_confidence: true`; the answer itself is unchanged).

### Typed results

Answers narrow to your questions. With `defineQuestions` (or a literal passed inline), `answers.team.choice` is the union of your labels and `probabilities` is keyed by them; comparing against a label that is not in your set is a compile error. `decide` goes further: the value is run through the schema's `~standard.validate` before you get it, so the static type is true at runtime. Schemas laya cannot answer from a fixed option set (free strings, arrays, nested objects) throw `SchemaError` naming the path before any inference. Router batch results narrow per request.

## Scenarios

Run each from the repo root: `LAYA_MODEL_DIR=./bundles LAYA_THREADS=4 bun examples/<file>.ts`. The `laya: this checkpoint ships invalid temperatures...` line that `english` and `typed-decisions` print on stderr at load is upstream's warning that those checkpoints' confidence is uncalibrated for one bucket (`choice` questions with 11 or more options); it does not affect these examples.

### 1. Support inbox triage (`examples/01-support-triage.ts`)

`predictBatch` over 10 realistic tickets with a team (choice), an urgent flag (noul) and a frustration level (score). The tickets carry the label I assigned before running; the script prints `ok` or `MISS`.

```
loaded english (fp32)

ok   team=billing   (98%) urgent=  3% frustration=1.8249 | I was charged twice for the Pro plan on March 3rd. Please re
ok   team=technical (73%) urgent= 76% frustration=1.7325 | Our webhook endpoint has returned 500 for every event since 
MISS team=billing   (99%) urgent=  9% frustration=0.9159 | How much would 40 seats on the Business plan cost with annua
ok   team=account   (49%) urgent= 84% frustration=1.8848 | I can't log in anymore, the reset email never arrives and I 
ok   team=technical (67%) urgent= 13% frustration=1.115 | The API returns 429 even though we are under the documented 
ok   team=billing   (53%) urgent= 35% frustration=1.0204 | Please update the card on file to the one ending 4417, the o
MISS team=technical (54%) urgent= 11% frustration=2.329 | This is the third time your export job has failed. I am done
MISS team=technical (38%) urgent= 16% frustration=2.0288 | Can you delete my account and all associated data? I no long
ok   team=sales     (30%) urgent=  0% frustration=1.0444 | Do you offer a discount for non-profits? We are a 12-person 
ok   team=technical (72%) urgent= 15% frustration=1.6916 | The dashboard shows a blank page after the latest update, tr

team accuracy 7/10, urgent accuracy 9/10
batch of 10: 5704 ms (570 ms per ticket)
same 10 one by one: 5429 ms (543 ms per ticket)
```

The same 10 tickets on the other checkpoints (the script takes `LAYA_CHECKPOINT`):

| Checkpoint | Team accuracy | Urgent accuracy | Batch of 10 |
| --- | --- | --- | --- |
| `english` | 7/10 | 9/10 | 5704 ms |
| `multilingual` | 5/10 | 9/10 | 3178 ms |
| `typed-decisions` | 8/10 | 9/10 | 9082 ms |

Read this as a smoke test, not a benchmark: n is 10, and two of the misses (a refund-plus-cancel message, a sales quote that mentions "annual billing") are arguable. The pattern matches upstream's own caveat: the base checkpoints work zero-shot on clear tickets and drift on borderline ones. Confidence is a usable signal here: the misses sit at 38 to 99 %, so a threshold alone does not catch them; use `minConfidence` plus a human queue for the uncertain band rather than trusting a cutoff you did not measure on your data. Also: `predictBatch` did not beat one-by-one calls on CPU (5704 ms vs 5429 ms); it saves calls, not compute.

### 2. Structured decisions with zod (`examples/02-structured-decisions.ts`)

`decideBatch` with a zod schema turns six tickets into typed objects (`department`, `priority` 0 to 2, `needs_human`, `refund_requested`). Descriptions on each field become the question text, so they matter: a first run of the same script without `.describe()` gave `priority: 0` for "Production down, we lose orders every minute" and `refund_requested: true` for an account-deletion request. With the descriptions:

```
Double charge          {"department":"billing","priority":1,"needs_human":false,"refund_requested":true}
Production down        {"department":"technical","priority":2,"needs_human":false,"refund_requested":false}
Team pricing           {"department":"billing","priority":0,"needs_human":false,"refund_requested":false}   <- differs on department
Close my account       {"department":"account","priority":0,"needs_human":false,"refund_requested":false}
Invoice copy           {"department":"billing","priority":0,"needs_human":false,"refund_requested":false}
Threatening to leave   {"department":"technical","priority":1,"needs_human":true,"refund_requested":true}

field accuracy 23/24
decideBatch of 6: 6644 ms
escalate: Production down -> technical, priority 2
escalate: Threatening to leave -> technical, priority 1

free text is rejected: properties.summary: a free string cannot be a fixed option set; use 'enum' or a boolean
```

23 of 24 fields matched my labels. The one difference is a sales quote sent to `billing`, the same confusion as scenario 1.

### 3. Mixed-language inbox with a router (`examples/03-multilingual-router.ts`)

One router, two checkpoints, ten messages in English, Spanish, German, French, Portuguese, Hindi, Japanese and Italian. `routeBatch` shows the plan without loading anything; `predictBatch` groups by checkpoint, loads each once and keeps input order.

```
routing plan (no model loaded yet): true
  en -> english      English Latin text
  es -> multilingual Latin script but language looks like "es", not English
  de -> multilingual Latin script but language looks like "de", not English
  fr -> multilingual Latin script but language looks like "fr", not English
  pt -> multilingual Latin script but language looks like "pt", not English
  hi -> multilingual non-Latin script (devanagari, 100% of letters); the English checkpoint cannot read it
  ja -> multilingual non-Latin script (kana, 100% of letters); the English checkpoint cannot read it
  en -> english      English Latin text
  it -> multilingual Latin script but language looks like "it", not English
  en -> english      English Latin text

first predictBatch, includes loading both checkpoints: 11248 ms; loaded: english, multilingual
ok   [en] english      intent=refund         (99%) urgent=10%
ok   [es] multilingual intent=refund         (100%) urgent=2%
ok   [de] multilingual intent=technical_help (100%) urgent=0%
ok   [fr] multilingual intent=cancellation   (99%) urgent=3%
ok   [pt] multilingual intent=information    (100%) urgent=1%
ok   [hi] multilingual intent=refund         (100%) urgent=1%
ok   [ja] multilingual intent=technical_help (99%) urgent=1%
ok   [en] english      intent=information    (71%) urgent=2%
ok   [it] multilingual intent=cancellation   (100%) urgent=5%
ok   [en] english      intent=technical_help (97%) urgent=79%

intent accuracy 10/10, urgent accuracy 9/10
second predictBatch, both checkpoints warm: 1925 ms (192 ms per message)
```

This is the strongest result of the set: 10 of 10 intents right, every non-English message routed to `multilingual` with the reason logged, and the Japanese "急ぎで" urgency was the one urgent miss (`urgent` 1 %, expected true).

### 4. A document longer than the window (`examples/04-long-contract.ts`)

A 1,057-word master services agreement (`examples/data/msa.txt`) against three questions whose answers sit in sections 7, 11 and 16. The provider is described as a company "registered in England and Wales" on the first line; the governing law, in the last section, is New York.

```
contract: 1057 words
truth:        auto_renews=true  breach_notification=true  governing_law=new_york

predict      auto_renews=100%  breach_notification= 99%  governing_law=england_and_wales (94%)
             usage.input_tokens (all question rows): 3072, 6638 ms

predictLong  auto_renews=100%  breach_notification=100%  governing_law=new_york (100%)
             windows scored: 3, 13220 ms
             auto_renews: decided by window 1 (tokens 380-1140)
             breach_notification: decided by window 1 (tokens 380-1140)
             governing_law: decided by window 2 (tokens 760-1356)
```

`predict` reads only the first window (1,024 tokens per question row) and answers `england_and_wales` at 94 %: wrong, and confident, which fits the model anchoring on the first line. `predictLong` scores three overlapping windows, the last (tokens 760 to 1356) holds section 16, and it answers `new_york` at 100 %, naming window 2 as the source. It took about twice as long here (13.2 s against 6.6 s for three windows instead of one). The two yes/no questions were already right in `predict`, so the win is specific to facts that live past the first window.

### 5. Calling laya-server over HTTP (`examples/05-server-client.ts`)

The client is plain `fetch`, so the same requests work from Python, Go or curl. With no `LAYA_URL` it starts a server from the repo build on a free port (`english` and `multilingual`, bearer token on), waits for `/health`, then sends a request in each language, a burst of eight, and three malformed requests.

```
/health ok, loaded english, multilingual after 9711 ms

English  -> 200 model=english intent=technical_help urgent=0.77 server-side 483.06 ms
Spanish  -> 200 model=multilingual intent=refund server-side 355.42 ms
8 concurrent -> statuses 200, wall 4071 ms

wrong token -> 401 {"detail":"invalid or missing bearer token"}
empty choice -> 422 {"detail":"question \"q\": a choice question needs at least one criterion"}
no questions -> 400 {"detail":"request body must be an object with a 'questions' field"}
```

Routing is the same as in the library: the English request went to `english`, the Spanish one to `multilingual`, and the response carries `routing` and an `X-Inference-Time-Ms` header. Errors are plain HTTP status codes with a `detail` message: 401 bad token, 422 invalid question, 400 malformed body, 503 while loading or over `LAYA_MAX_CONCURRENT`. To point at a deployed server: `LAYA_URL=https://... LAYA_API_KEY=... bun examples/05-server-client.ts`; deployment is covered in [docs/deploy](deploy/README.md).

## Library or server

| | Library (`@desplega.ai/laya`) | Server (`@desplega.ai/laya-server`) |
| --- | --- | --- |
| Caller | TypeScript or Bun code | any language, any number of services |
| Calls | `predict`, `predictBatch`, `predictLong`, `decide`, `decideBatch`, Router | `POST /v1/systemone` only: one state per request, no batch, no long, no `decide` |
| Types | end-to-end: label unions, zod-validated decisions | JSON; you type the response yourself |
| Memory | one copy per process that loads it (about 1.9 GiB for `multilingual`, see [deploy](deploy/README.md#measured-footprint)) | one copy per replica, shared by all callers |
| Ops | none, but every worker process pays the memory and the load time | image, health checks, auth, concurrency limit |

Pick the library for a single TypeScript service, a batch job or a CLI, and whenever you need batch, long-document or typed-decision calls. Pick the server when several services or non-TypeScript clients share one model, or you want the model isolated from your app's memory. If you need `predictBatch` or `predictLong` over HTTP, that is a server change, not something to work around in the client.

## Limits

- Base checkpoints are zero-shot classifiers. Upstream reports `typed-decisions` at 0.766 accuracy on its own benchmark and the base checkpoints at about 0.36 on that benchmark's workflows. Validate on your own labelled sample before automating anything, and treat fine-tuning as the route to more accuracy.
- Only fp32 is built; INT8 is deferred. CPU only.
- Do not use boolean-word labels (`yes`, `true`) in `choice` questions; the models can follow the label instead of your description. Use semantic labels.
- State beyond the window is cut for `predict` (512 tokens for `english`, 1,024 for `multilingual`, minus the option budget). Use `predictLong`.

## How the numbers were measured

- CPU: Intel Xeon W-2145 @ 3.70 GHz, 16 threads, Linux 6.8, Node 22.23, Bun 1.4.0, `LAYA_THREADS=4`, fp32 bundles on local disk.
- The host is shared: load average was 9 to 14 on 16 cores throughout, so latencies below are noisy and closer to an upper bound than a clean benchmark. Two runs of the same script differed by up to 2x.
- Per-call `predict` latency (`examples/06-latency.ts`: one ticket, 3 questions, 30 runs after 3 warm-ups):

| Checkpoint | Load | p50 | p95 | Min | Run |
| --- | --- | --- | --- | --- | --- |
| `english` | 3.5 s | 440 ms | 471 ms | 334 ms | run A, `LAYA_THREADS=4` |
| `multilingual` | 3.0 s | 223 ms | 259 ms | 176 ms | run A, `LAYA_THREADS=4` |
| `english` | 4.7 s | 773 ms | 906 ms | 573 ms | run B, same setting, busier host |
| `multilingual` | 3.6 s | 376 ms | 439 ms | 200 ms | run B |
| `typed-decisions` | 2.6 s | 698 ms | 873 ms | 440 ms | run B |

- Thread count barely moved a short `predict` in run A (`english` p50 400, 432, 440 ms and `multilingual` 216, 186, 223 ms at 1, 2 and 4 threads), so on a small box do not expect more threads to help single short requests; set `LAYA_THREADS` to the vCPUs you actually have so onnxruntime does not oversubscribe.
- Server-side inference for a two-question request was 320 ms (`english`) and 182 ms (`multilingual`) in an earlier run (`LAYA_THREADS` unset) and 483.06 / 355.42 ms in the run pasted above; the server was ready 9711 ms after start with both checkpoints. A burst of eight concurrent requests finished in 4071 ms.
- Memory and cold-start numbers for the server are in [docs/deploy](deploy/README.md#measured-footprint).
