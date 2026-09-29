# @desplega.ai/laya

Typed decisions from a [laya](https://github.com/NandhaKishorM/laya) checkpoint, in Node or Bun. You give it text or a JSON object and questions (`choice`, `noul`, `score`); it answers with probabilities, not generated text. One ONNX encoder pass plus a small head, no generation.

```sh
npm install @desplega.ai/laya
```

```ts
import { createAgent, defineQuestions } from "@desplega.ai/laya";

const agent = await createAgent({ checkpoint: "english", modelDir: "./bundles/english/fp32" });
const questions = defineQuestions({
  team: { type: "choice", instructions: "Which team handles this ticket?", criteria: { billing: "charges", technical: "bugs" } },
});
const [result] = await agent.predictBatch(["I was charged twice"], questions);
```

Node 22+ or Bun 1.4+. Pass a local fp32 ONNX bundle as `modelDir` (export one from upstream's public checkpoints with [tools/export](https://github.com/desplega-ai/laya-js/blob/main/tools/export/README.md#export-for-your-own-use)), or omit `modelDir` and `createAgent` downloads Desplega's prebuilt bundle from the public Hugging Face repo `desplega/laya-onnx`, verified against pinned SHA-256s. No token is needed; `HF_TOKEN` is optional (rate limits, or a private mirror).

The untyped vendored surface is available as `@desplega.ai/laya/raw`.

Full guide with real outputs: [docs/usage.md](https://github.com/desplega-ai/laya-js/blob/main/docs/usage.md). Apache-2.0, vendored from upstream `laya-ts`; see `NOTICE`.
