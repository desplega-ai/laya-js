"""Generate the Python fp32 goldens the TS parity gate compares against.

Runs the pinned Python `laya` torch Agent (CPU, fp32, deterministic) on every fixture case and
writes packages/laya/test/parity/golden/<checkpoint>.json (`--kind single`, `system_one`) or
<checkpoint>.batch.json (`--kind batch`, `predict_batch`; result i of case `batch/n` is stored as
`batch/n#i`). Run once per pin bump, by hand or via the `parity-goldens` workflow. The
checkpoints are public, so no token is needed.

    uv run --project tools/export python tools/parity/make_goldens.py --checkpoint multilingual --kind batch
"""
import argparse
import json
import os
import platform

import torch
import transformers
from laya.agent import Agent

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURES = os.path.join(ROOT, "tools", "parity", "fixtures")
KINDS = {"single": ("cases.jsonl", ""), "batch": ("batch.jsonl", ".batch")}
GOLDEN_DIR = os.path.join(ROOT, "packages", "laya", "test", "parity", "golden")
UPSTREAM_SHA = "9d955671415fc19f069b9cc998928075c1f255ec"

# Standalone repos and revisions pinned by upstream (laya-ts/src/providers.ts), same as the export.
CHECKPOINTS = {
    "english": ("convaiinnovations/laya", "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"),
    "multilingual": ("convaiinnovations/laya-multilingual", "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67"),
    "typed-decisions": ("convaiinnovations/laya-typed-decisions", "1a793eb568e6718f15941d08f85432581df534e3"),
}


def probabilities(ans: dict) -> dict:
    if ans["type"] == "noul":
        return {"true": ans["noul"], "false": round(1.0 - ans["noul"], 4)}
    return ans["probabilities"]


def margin(p: dict) -> float:
    top = sorted(p.values(), reverse=True)
    return round(top[0] - (top[1] if len(top) > 1 else 0.0), 4)


def golden_answer(ans: dict) -> dict:
    out = {
        "type": ans["type"],
        "confidence": ans["confidence"],
        "answer_confidence": ans["answer_confidence"],
        "act_probability": ans["action"]["act_probability"],
    }
    if ans["type"] == "choice":
        out["choice"] = ans["choice"]
        out["probabilities"] = ans["probabilities"]
    elif ans["type"] == "score":
        out["score"] = ans["score"]
        out["probabilities"] = ans["probabilities"]
    else:
        out["noul"] = ans["noul"]
    out["margin"] = margin(probabilities(ans))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, choices=sorted(CHECKPOINTS))
    ap.add_argument("--kind", default="single", choices=sorted(KINDS))
    ap.add_argument("--model-dir", default=None, help="Local checkpoint dir instead of the pinned HF download.")
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    repo, revision = CHECKPOINTS[args.checkpoint]
    agent = Agent(args.model_dir or repo, device="cpu", revision=None if args.model_dir else revision)

    fixture, suffix = KINDS[args.kind]
    with open(os.path.join(FIXTURES, fixture), encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    def record(r: dict) -> dict:
        return {
            "input_tokens": r["usage"]["input_tokens"],
            "answers": {qid: golden_answer(a) for qid, a in r["answers"].items()},
        }

    results = {}
    for c in cases:
        opts = c.get("opts") or {}
        try:
            if args.kind == "batch":
                rs = agent.predict_batch(c["states"], c["questions"], batch_size=opts.get("batch_size"),
                                         sort_by_length=opts.get("sort_by_length", False), lang=opts.get("lang"))
                for i, r in enumerate(rs):
                    results[f"{c['id']}#{i}"] = record(r)
            else:
                results[c["id"]] = record(agent.system_one(c["state"], c["questions"], lang=opts.get("lang")))
        except Exception as e:  # recorded, so the TS side must fail the same case too
            results[c["id"]] = {"error": f"{type(e).__name__}: {e}"}

    golden = {
        "meta": {
            "checkpoint": args.checkpoint,
            "precision": "fp32",
            "kind": args.kind,
            "repo": repo,
            "revision": revision,
            "upstream_sha": UPSTREAM_SHA,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
            "cases": len(cases),
        },
        "cases": results,
    }
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    out = os.path.join(GOLDEN_DIR, f"{args.checkpoint}{suffix}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(golden, f, ensure_ascii=False, indent=1, sort_keys=True)
        f.write("\n")
    errors = sum(1 for v in results.values() if "error" in v)
    print(f"{args.checkpoint} {args.kind}: {len(results)} results, {errors} errors -> {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
