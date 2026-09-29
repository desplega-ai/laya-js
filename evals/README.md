# evals

Phase 4 (classification accuracy and Python agreement) and Phase 10 (performance and cost) of the
laya-js plan. fp32 only: INT8 is deferred (plan, 2026-09-28), so every TS INT8 cell is N/A.
The reference numbers are measured on a Hetzner `ccx23` (4 dedicated vCPU, 16 GB), not on shared
CI runners. The TS in-process runtimes do not fit in 16 GB yet (head-graph memory, see the Phase 10 report),
so `results/perf/hetzner-ccx43/` holds the in-process numbers from a ccx43 (16 vCPU, 64 GB), Python beside TS.
`.github/workflows/evals.yml` is `workflow_dispatch` only.

| Path | What |
|---|---|
| `datasets/build.py` | Builds the suites in `laya-evals` JSONL with upstream's loaders (seed 13, 400 per suite) into `datasets/build/` (not committed). `--check` rebuilds and compares SHA-256 with `manifest.json`. |
| `manifest.json` | Per suite: source dataset and commit, split, checkpoints, row ids, SHA-256. |
| `python/reference.py` | Python reference: upstream torch `Agent` at the pinned revision (fp32), or `ONNXAgent` (`--runtime onnx-fp32|onnx-int8`, report only). |
| `python/metrics.py` | Metrics: upstream `laya.evals` evaluators and ECE, plus bench_local.py's macro-F1 and Brier. `packages/laya/evals/metrics.ts` is the TS port (`test/evals-metrics.test.ts`, 1e-9). |
| `python/perf.py` | Python perf baseline, the protocol of `packages/laya/evals/perf-worker.mjs`. |
| `k8s/` | Eval overlays over `deploy/k8s` for `kind` (NodePort, local image; `all-checkpoints` drops the fetch initContainer because the eval image bakes all three bundles). |
| `prices.json` | Prices for `eval:cost`, each with source and date. |
| `results/<checkpoint>/{python,ts}-fp32.json` | Phase 4 metrics per slice (suite, tag, language). |
| `results/perf/<host>/*.json`, `results/cost/<host>.json` | Phase 10 results. |

## Phase 4

```sh
uv sync --frozen --project evals
uv run --project evals python evals/datasets/build.py            # or --check
uv run --project evals python evals/python/reference.py --checkpoint multilingual
bun run eval:accuracy --checkpoint multilingual [--bundle-dir <local bundle>]
bun run eval:gate                                                # all three checkpoints
bun run eval:gate --shuffle-labels ag_news                       # negative control, must fail
```

The gate: every metric of every slice within 0.001 of Python, and per row the Phase 3 fp32 parity
bar (max |Δp| ≤ 1e-4 on probabilities, `noul`, `act_probability`; top-1 agreement 100%).

## Phase 10

```sh
bun run eval:perf --make-workload
bun run eval:perf --target node --checkpoint multilingual --bundle-dir <dir>      # and bun
bun run eval:perf --target python --checkpoint multilingual --runtime torch       # onnx-fp32, onnx-int8
bun run eval:perf --target docker --checkpoint multilingual --image laya-server:eval
LAYA_API_KEY=... evals/k8s/kind-up.sh base    # prints the Service URL
bun run eval:perf --target k8s --overlay base --checkpoint multilingual --url <url>
bun run eval:gate --perf
bun run eval:cost
```

The eval image is the shipped Dockerfile with the `models` stage replaced by a local directory
(`docker buildx build --build-context models=<dir with models/<ckpt>/fp32>`), so no HF token is
needed on the host. In-process targets need root for the cold-cache starts (they drop the page
cache).
