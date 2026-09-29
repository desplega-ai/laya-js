"""Write evals/fixtures/metrics-20.json: 20 synthetic records and the Python metrics over them.

`packages/laya/test/evals-metrics.test.ts` recomputes the metrics with the TS port and requires
agreement to 1e-9. The records cover every question type, two question ids in one suite, an
unanswered record, and confidences on ECE bin edges (0, 0.2, 0.4, 1).

    uv run --project evals python evals/python/make_metrics_fixture.py
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import sliced  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EDGES = [0.0, 0.2, 0.4, 1.0]


def main():
    rng = random.Random(13)
    records = []
    for i in range(20):
        kind = ["choice", "choice", "noul", "score"][i % 4]
        suite = "a" if i < 10 else "b"
        tags = ["t1"] if i % 3 == 0 else ["t1", "t2"] if i % 3 == 1 else []
        lang = [None, "en", "es"][i % 3]
        conf = EDGES[i % 4] if i % 5 == 0 else round(rng.random(), 4)
        if kind == "choice":
            keys = ["x", "y", "z"]
            raw = [rng.random() for _ in keys]
            probs = {k: round(v / sum(raw), 4) for k, v in zip(keys, raw)}
            ans = {"type": "choice", "choice": max(probs, key=probs.get), "probabilities": probs,
                   "answer_confidence": conf, "act_probability": round(rng.random(), 4)}
            rec = {"qid": "q%d" % (i % 2), "type": kind, "expected": rng.choice(keys), "answer": ans}
        elif kind == "noul":
            p = round(rng.random(), 4)
            ans = {"type": "noul", "noul": p, "answer_confidence": conf, "act_probability": round(rng.random(), 4)}
            rec = {"qid": "n", "type": kind, "expected": rng.random() < 0.5, "answer": ans}
        else:
            raw = [rng.random() for _ in range(5)]
            probs = {str(k): round(v / sum(raw), 4) for k, v in enumerate(raw)}
            score = round(sum(k * p for k, p in enumerate(probs.values())), 4)
            ans = {"type": "score", "score": score, "probabilities": probs, "answer_confidence": conf,
                   "act_probability": round(rng.random(), 4)}
            rec = {"qid": "s", "type": kind, "expected": float(rng.randint(0, 4)), "answer": ans}
        if i == 17:
            rec["answer"] = None
        rec.update(suite=suite, tags=tags, language=lang)
        records.append(rec)
    out = {"records": records, "expected": sliced(records)}
    path = os.path.join(ROOT, "evals", "fixtures", "metrics-20.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)
        f.write("\n")
    print("wrote %s" % path)


if __name__ == "__main__":
    main()
