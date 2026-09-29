"""Phase 10 Python baseline, in-process: the same protocol as packages/laya/evals/perf-worker.mjs.

`--runtime torch` is upstream's torch Agent (fp32, the ±25% control against BENCHMARKS.md);
`onnx-fp32` is upstream's ONNXAgent on its single-graph fp32 export, the closest Python path to
the TS runtime (same engine, onnxruntime); `onnx-int8` is the same with `--quantize` (report
only). Prints one JSON line; `READY <epoch ms>` goes to stderr after the first answer.

    uv run --project evals python evals/python/perf.py --checkpoint multilingual --runtime torch \
        --workload evals/out/perf/workload.json [--mode start]
"""
import argparse
import json
import os
import resource
import sys
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference import load_agent  # noqa: E402


def summary(ts):
    return {"p50_ms": round(float(np.percentile(ts, 50)), 2), "p95_ms": round(float(np.percentile(ts, 95)), 2),
            "p99_ms": round(float(np.percentile(ts, 99)), 2), "mean_ms": round(float(np.mean(ts)), 2), "n": len(ts)}


def timed(fn, reps):
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    return ts


def rss_mb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--runtime", default="torch", choices=["torch", "onnx-fp32", "onnx-int8"])
    ap.add_argument("--workload", required=True)
    ap.add_argument("--mode", default="full", choices=["full", "start"])
    ap.add_argument("--onnx-dir", default="/root/pyonnx")
    args = ap.parse_args()
    w = json.load(open(args.workload, encoding="utf-8"))

    def qs(n):
        return {("q%d" % i): (w["q_noul"] if i % 2 else w["q_choice"]) for i in range(n)}

    t0 = time.perf_counter()
    agent = load_agent(args.checkpoint, args.runtime, args.onnx_dir)
    if args.runtime == "torch":
        agent.model.eval()
    load_ms = (time.perf_counter() - t0) * 1000
    agent.system_one(w["states"][0], qs(1))
    first_ms = (time.perf_counter() - t0) * 1000
    sys.stderr.write("READY %d\n" % int(time.time() * 1000))
    sys.stderr.flush()
    out = {"load_ms": round(load_ms, 2), "first_answer_ms": round(first_ms, 2), "threads": torch.get_num_threads()}
    if args.mode == "start":
        out["peak_rss_mb"] = rss_mb()
        print(json.dumps(out))
        return

    out["control"], out["latency"], out["throughput"] = {}, {}, {}
    for n in (1, 5, 10):
        q = qs(n)
        timed(lambda: agent.system_one(w["control_state"], q), 2)
        out["control"]["%d_questions" % n] = summary(timed(lambda: agent.system_one(w["control_state"], q), 10))
    for n in (1, 5, 10):
        q, k = qs(n), [0]

        def call():
            agent.system_one(w["states"][k[0] % len(w["states"])], q)
            k[0] += 1

        timed(call, 2)
        out["latency"]["%d_questions" % n] = summary(timed(call, 20))
    states = w["states"][:96]
    for bs in (1, 8, 32):
        agent.predict_batch(states[:bs], qs(2), batch_size=bs, sort_by_length=True)
        t = time.perf_counter()
        agent.predict_batch(states, qs(2), batch_size=bs, sort_by_length=True)
        s = time.perf_counter() - t
        out["throughput"]["batch_%d" % bs] = {"classifications": len(states) * 2, "seconds": round(s, 2),
                                              "per_s": round(len(states) * 2 / s, 2)}
    out["peak_rss_mb"] = rss_mb()
    print(json.dumps(out))


if __name__ == "__main__":
    with torch.no_grad():
        main()
