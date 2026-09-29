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

Node 22+ or Bun 1.4+. You supply the fp32 ONNX checkpoint bundle: export it from upstream's public checkpoints with [tools/export](https://github.com/desplega-ai/laya-js/blob/main/tools/export/README.md#export-for-your-own-use) and pass its directory as `modelDir`. Without `modelDir`, `createAgent` downloads Desplega's private artifact store and needs an `HF_TOKEN` with access to it.

The untyped vendored surface is available as `@desplega.ai/laya/raw`.

Full guide with real outputs: [docs/usage.md](https://github.com/desplega-ai/laya-js/blob/main/docs/usage.md). Apache-2.0, vendored from upstream `laya-ts`; see `NOTICE`.
