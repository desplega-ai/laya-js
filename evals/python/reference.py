"""Phase 4 Python reference: score every suite with upstream laya at the pin, CPU.

`--runtime torch` (the reference, plan Phase 4) runs the torch `Agent` at the pinned HF revision,
fp32. `--runtime onnx-fp32` / `onnx-int8` run upstream's `ONNXAgent` on its single-graph export
(`scripts/export_onnx.py`, INT8 via `--quantize`); those are report-only columns.

Rows sharing a question set go through `predict_batch` (batch 8, sort_by_length), the same call
the TS runner makes. Writes `evals/results/<checkpoint>/python-<runtime>.json` (metrics) and
`evals/out/<checkpoint>/python-<runtime>.rows.jsonl` (per-row answers, not committed).

    uv run --project evals python evals/python/reference.py --checkpoint multilingual --threads 4
"""
import argparse
import json
import os
import platform
import sys
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import normalize_answer, records_from, sliced  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUILD = os.path.join(ROOT, "evals", "datasets", "build")
CHECKPOINTS = {
    "english": ("convaiinnovations/laya", "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"),
    "multilingual": ("convaiinnovations/laya-multilingual", "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67"),
    "typed-decisions": ("convaiinnovations/laya-typed-decisions", "1a793eb568e6718f15941d08f85432581df534e3"),
}
OUT_NAME = {"torch": "python-fp32", "onnx-fp32": "python-onnx-fp32", "onnx-int8": "python-onnx-int8"}


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def suites_for(ckpt, only):
    manifest = json.load(open(os.path.join(ROOT, "evals", "manifest.json")))
    names = [n for n, s in manifest["suites"].items() if ckpt in s["checkpoints"]]
    return [n for n in names if not only or n in only], manifest


def load_agent(ckpt, runtime, onnx_dir):
    repo, rev = CHECKPOINTS[ckpt]
    if runtime == "torch":
        from laya.agent import Agent
        return Agent(repo, device="cpu", revision=rev)
    from laya.onnx_agent import ONNXAgent
    name = "laya.onnx" if runtime == "onnx-fp32" else "laya.int8.onnx"
    return ONNXAgent(repo, onnx_path=os.path.join(onnx_dir, ckpt, name), revision=rev)


def run_group(agent, rows, questions, batch_size):
    """predict_batch over one question set; on a batch error, fall back row by row to isolate it."""
    try:
        res = agent.predict_batch([r["state"] for r in rows], questions, batch_size=batch_size,
                                  sort_by_length=True)
        return [({q: normalize_answer(a) for q, a in r["answers"].items()}, None) for r in res]
    except Exception:
        out = []
        for r in rows:
            try:
                res = agent.system_one(r["state"], questions)
                out.append(({q: normalize_answer(a) for q, a in res["answers"].items()}, None))
            except Exception as e:  # noqa: BLE001 - recorded per row, counted as an error
                out.append((None, str(e)[:300]))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, choices=sorted(CHECKPOINTS))
    ap.add_argument("--runtime", default="torch", choices=sorted(OUT_NAME))
    ap.add_argument("--suites", default="", help="Comma-separated subset (default: every suite for the checkpoint).")
    ap.add_argument("--limit", type=int, default=0, help="First n rows per suite (0 = all).")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--onnx-dir", default="/root/pyonnx", help="Upstream single-graph exports, <dir>/<checkpoint>/laya*.onnx.")
    ap.add_argument("--tag", default="", help="Suffix for the output names, e.g. a host label.")
    ap.add_argument("--reuse", action="store_true",
                    help="Keep the previous run's rows for suites whose SHA-256 is unchanged; score the rest.")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
    torch.manual_seed(0)
    names, manifest = suites_for(args.checkpoint, [s for s in args.suites.split(",") if s])
    t0 = time.time()
    agent = load_agent(args.checkpoint, args.runtime, args.onnx_dir)
    load_s = time.time() - t0

    name = OUT_NAME[args.runtime] + (("-" + args.tag) if args.tag else "")
    out_dir = os.path.join(ROOT, "evals", "out", args.checkpoint)
    res_dir = os.path.join(ROOT, "evals", "results", args.checkpoint)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    records, timing, reused = [], {}, []
    old_rows, old_sha, old_timing = {}, {}, {}
    rows_path, res_path = os.path.join(out_dir, name + ".rows.jsonl"), os.path.join(res_dir, name + ".json")
    if args.reuse and os.path.exists(rows_path) and os.path.exists(res_path):
        old_meta = json.load(open(res_path))["meta"]
        old_sha, old_timing = old_meta["suite_sha256"], old_meta["seconds_by_suite"]
        for line in open(rows_path, encoding="utf-8"):
            r = json.loads(line)
            old_rows.setdefault(r["suite"], []).append(r)
    with open(rows_path, "w", encoding="utf-8") as rows_out:
        for suite in names:
            path = os.path.join(BUILD, suite + ".jsonl")
            rows = [json.loads(l) for l in open(path, encoding="utf-8")]
            if args.limit:
                rows = rows[: args.limit]
            if suite in old_rows and old_sha.get(suite) == manifest["suites"][suite]["sha256"]:
                outputs = {r["id"]: r["answers"] for r in old_rows[suite]}
                for r in old_rows[suite]:
                    rows_out.write(json.dumps(r, ensure_ascii=False) + "\n")
                timing[suite] = old_timing.get(suite)
                reused.append(suite)
                records += records_from(path, outputs)
                print("   %-22s reused" % suite, flush=True)
                continue
            groups = {}
            for r in rows:
                groups.setdefault(json.dumps(r["questions"], sort_keys=True), []).append(r)
            ts = time.time()
            outputs = {}
            for grp in groups.values():
                # The first row's own dict, not the sorted grouping key: question order is kept.
                for r, (ans, err) in zip(grp, run_group(agent, grp, grp[0]["questions"], args.batch_size)):
                    outputs[r["id"]] = ans
                    rows_out.write(json.dumps({"id": r["id"], "suite": suite, "answers": ans, "error": err},
                                              ensure_ascii=False) + "\n")
            timing[suite] = round(time.time() - ts, 1)
            recs = records_from(path, outputs)
            records += recs
            acc = sliced(recs)["overall"]
            print("   %-22s %4d rows  acc %s  %.0fs" % (suite, len(rows), round(acc.get("accuracy", float("nan")), 4),
                                                      timing[suite]), flush=True)

    result = {
        "meta": {"runtime": "python-" + args.runtime, "checkpoint": args.checkpoint,
                 "precision": "int8" if args.runtime == "onnx-int8" else "fp32",
                 "source": {"repo": CHECKPOINTS[args.checkpoint][0], "revision": CHECKPOINTS[args.checkpoint][1]},
                 "upstream_sha": manifest["upstream_sha"], "batch_size": args.batch_size, "limit": args.limit,
                 "suites": names, "suite_sha256": {n: manifest["suites"][n]["sha256"] for n in names},
                 "host": {"cpu": cpu_model(), "cores": os.cpu_count(), "python": platform.python_version(),
                          "torch": torch.__version__, "threads": torch.get_num_threads()},
                 "load_s": round(load_s, 1), "seconds_by_suite": timing, "reused_suites": reused},
        "metrics": sliced(records),
    }
    with open(os.path.join(res_dir, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    print("wrote %s" % os.path.join(res_dir, name + ".json"))


if __name__ == "__main__":
    main()
