"""Phase 4 metrics, Python side. `packages/laya/evals/metrics.ts` is the TS port.

Accuracy per question type, mean confidence and ECE come from upstream's `laya.evals` (the
evaluators and `ece`, which reuses `laya.common.ece_score`, 15 bins, on `answer_confidence`).
Macro-F1 and Brier are the definitions of upstream's `research/scripts/bench_local.py`, because
`laya.evals` has none. Both sides compute from the same rounded answers, so a metric gap between
Python and TS is a gap in the answers, not in the metric code.

A record is one answered question: {"type", "expected", "answer", "qid", "suite", "tags", "language"}.
"""
import json
import statistics

from laya.evals import ChoiceAccuracy, MeanConfidence, NoulAccuracy, ScoreMAE, ece

EVALUATORS = (ChoiceAccuracy(), NoulAccuracy(), ScoreMAE(), MeanConfidence())


def normalize_answer(ans):
    """The fields every metric and agreement check reads, from a system_one/predict_batch answer."""
    out = {"type": ans["type"], "answer_confidence": ans.get("answer_confidence"),
           "act_probability": (ans.get("action") or {}).get("act_probability")}
    if ans["type"] == "noul":
        out["noul"] = ans["noul"]
    else:
        out["probabilities"] = ans["probabilities"]
        out["choice" if ans["type"] == "choice" else "score"] = ans[ans["type"]]
    return out


def _probs(answer):
    if answer["type"] == "noul":
        p = float(answer["noul"])
        return {"false": 1.0 - p, "true": p}
    return {k: float(v) for k, v in answer["probabilities"].items()}


def _label(answer):
    if answer["type"] == "noul":
        return "true" if float(answer["noul"]) >= 0.5 else "false"
    return answer["choice"]


def _gold(rec):
    e = rec["expected"]
    if rec["type"] == "noul":
        return "true" if e else "false"
    return e


def _classified(rec):
    """Choice and noul records with a usable label (the ones accuracy, F1, ECE and Brier use)."""
    e = rec["expected"]
    if rec["type"] == "choice":
        return isinstance(e, str)
    if rec["type"] == "noul":
        return isinstance(e, bool)
    return False


def macro_f1(pairs):
    """bench_local.py macro_f1 over (gold, pred) labels; classes are the union of both."""
    classes = sorted({g for g, _ in pairs} | {p for _, p in pairs})
    f = []
    for c in classes:
        tp = sum(1 for g, p in pairs if p == c and g == c)
        fp = sum(1 for g, p in pairs if p == c and g != c)
        fn = sum(1 for g, p in pairs if p != c and g == c)
        f.append(2 * tp / max(1, 2 * tp + fp + fn))
    return statistics.fmean(f)


def brier(rec):
    """Sum over options of (p - onehot)^2, as bench_local.py; noul counts as a 2-option choice."""
    gold = _gold(rec)
    return sum((p - (1.0 if k == gold else 0.0)) ** 2 for k, p in _probs(rec["answer"]).items())


def metrics(records):
    out = {"n": len(records)}
    answered = [r for r in records if r.get("answer") is not None]
    out["errors"] = len(records) - len(answered)
    for ev in EVALUATORS:
        vals = [ev.score(r["answer"], r["expected"]) for r in answered]
        vals = [v for v in vals if v is not None]
        if vals:
            out[ev.name] = statistics.fmean(vals)
    cls = [r for r in answered if _classified(r)]
    if cls:
        # Keyed by question id, so a suite with several questions does not merge their label sets.
        pairs = [(r["qid"] + "=" + str(_gold(r)), r["qid"] + "=" + str(_label(r["answer"]))) for r in cls]
        corr = [g == p for g, p in pairs]
        out["accuracy"] = statistics.fmean(1.0 if c else 0.0 for c in corr)
        out["macro_f1"] = macro_f1(pairs)
        out["brier"] = statistics.fmean(brier(r) for r in cls)
        conf = [(r["answer"]["answer_confidence"], c) for r, c in zip(cls, corr)
                if isinstance(r["answer"].get("answer_confidence"), (int, float))]
        if conf:
            value = ece([c for c, _ in conf], [k for _, k in conf])
            if value is not None and value == value:
                out["ece"] = value
    return out


def sliced(records):
    """Overall, then per suite, tag and language (the `laya.evals` slices plus suite)."""
    def group(key):
        g = {}
        for r in records:
            if key == "tags":
                vals = r.get("tags") or []
            else:
                vals = [] if r.get(key) is None else [r[key]]
            for v in vals:
                g.setdefault(str(v), []).append(r)
        return {k: metrics(v) for k, v in sorted(g.items())}

    return {"overall": metrics(records), "by_suite": group("suite"), "by_tag": group("tags"),
            "by_language": group("language")}


def records_from(rows_path, outputs):
    """Join built JSONL rows with runtime outputs ({id: {qid: answer | None}}) into records."""
    recs = []
    with open(rows_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["id"] not in outputs:
                continue
            answers = outputs[row["id"]] or {}
            for qid, qd in row["questions"].items():
                if qid not in row["expected"]:
                    continue
                recs.append({"qid": qid, "type": qd["type"], "expected": row["expected"][qid],
                             "answer": answers.get(qid), "suite": row["suite"], "tags": row["tags"],
                             "language": row["language"]})
    return recs
