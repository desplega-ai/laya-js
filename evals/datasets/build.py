"""Build the Phase 4 classification suites in upstream's `laya-evals` JSONL format.

The suite loaders are upstream's own, from `research/scripts/bench_apps.py` and
`research/scripts/bench_local.py` at laya 9d955671 (seed 13, 400 cases per suite), so our Python
run is comparable with upstream's published numbers. Changes from upstream, each on purpose:

- banking77 is excluded (a 77-option head-budget ceiling measures the architecture, not the port);
- the model-routing pool, ag_news and emotion loaders are unchanged, including the shared
  `random.Random(13)` stream, so the row selection matches upstream's;
- typed decisions come from `LocalLLaMA/typed-decisions` (config `all`, split `test`), the set
  upstream's `bench_local.py` reads from a local parquet;
- MASSIVE uses 10 languages x 100 cases (bench_local.py uses every language);
- the zh decision bench is upstream's committed `research/evals/zh_decision_bench.jsonl`;
- integral floats in states (`438.0`) are written as integers (`438`). JavaScript numbers do not
  carry that distinction, so a JSON state with `438.0` reaches the TS runtime as `438` while
  Python renders it `"438.0"`, and the two runtimes would score different text. The first run
  (2026-09-29) measured this on typed_decisions: 94 of 400 rows diverged, up to |Δp| 0.34.

The JSONL is not committed (third-party licences, size). `evals/manifest.json` is: dataset id,
dataset commit, split, seed, selected row ids and the SHA-256 of each built suite.

    uv run --project evals python evals/datasets/build.py            # build, write the manifest
    uv run --project evals python evals/datasets/build.py --check    # rebuild, compare SHA-256
"""
import argparse
import hashlib
import json
import os
import random
import sys
import urllib.request

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "evals", "datasets", "build")
MANIFEST = os.path.join(ROOT, "evals", "manifest.json")
UPSTREAM_SHA = "9d955671415fc19f069b9cc998928075c1f255ec"
ZH_URL = "https://raw.githubusercontent.com/NandhaKishorM/laya/%s/research/evals/zh_decision_bench.jsonl" % UPSTREAM_SHA
SEED = 13
N = 400
ALL = ["english", "multilingual", "typed-decisions"]
MASSIVE_LANGS = ["es", "fr", "de", "hi", "zh-CN", "ja", "ko", "ar", "ru", "th"]
MASSIVE_PER_LANG = 100
MASSIVE_OPTS = 20


class Builder:
    def __init__(self, pinned):
        self.pinned = pinned or {}
        self.sources = {}
        self.suites = {}

    def dataset(self, repo, *args, **kw):
        from datasets import load_dataset
        from huggingface_hub import HfApi

        if repo not in self.sources:
            info = HfApi().dataset_info(repo, revision=self.pinned.get(repo))
            card = info.card_data.to_dict() if info.card_data else {}
            self.sources[repo] = {"commit": info.sha, "license": card.get("license")}
        return load_dataset(repo, *args, revision=self.sources[repo]["commit"], **kw)

    def add(self, name, rows, *, source, split, checkpoints=ALL, note=""):
        for i, r in enumerate(rows):
            r["state"] = js_numbers(r["state"])
            r["id"] = "%s/%s" % (name, r.pop("_src", i))
            r.setdefault("tags", [])
            r.setdefault("language", None)
            r.setdefault("model", None)
            r["suite"] = name
        self.suites[name] = {"rows": rows, "source": source, "split": split, "checkpoints": checkpoints,
                             "note": note}
        print("   %-24s %4d rows  %s" % (name, len(rows), note), flush=True)


def js_numbers(v):
    """The state as a JavaScript caller can pass it: integral floats become ints."""
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, dict):
        return {k: js_numbers(x) for k, x in v.items()}
    if isinstance(v, list):
        return [js_numbers(x) for x in v]
    return v


def choice(qid, instructions, criteria):
    return {qid: {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}}


def build(b):
    import laya

    rng = random.Random(SEED)

    # ag_news (bench_apps.py "jev.ag_news")
    d = b.dataset("fancyzhx/ag_news", split="test")
    crit = {"world": "world news and international politics", "sports": "sports",
            "business": "business and economy", "sci_tech": "science and technology"}
    keys = list(crit)
    b.add("ag_news", [{"_src": i, "state": {"article": r["text"]},
                       "questions": choice("topic", "What is the topic of `article`?", crit),
                       "expected": {"topic": keys[int(r["label"])]}, "tags": ["in_training"]}
                      for i, r in enumerate(list(d)[:N])], source="fancyzhx/ag_news", split="test", note="topic, 4 labels")

    # dair-ai/emotion (bench_apps.py "jev.emotion")
    d = b.dataset("dair-ai/emotion", "split", split="test")
    names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    b.add("emotion", [{"_src": i, "state": {"text": r["text"]},
                       "questions": choice("emotion", "Which emotion is most strongly expressed in `text`?",
                                           {n: None for n in names}),
                       "expected": {"emotion": names[int(r["label"])]}, "tags": ["held_out"]}
                      for i, r in enumerate(list(d)[:N])], source="dair-ai/emotion", split="test", note="6 labels")

    # support triage (bench_apps.py "app.support_triage")
    d = b.dataset("Tobi-Bueck/customer-support-tickets", split="train")
    queues = {"Technical Support": "technical problems, bugs, outages, integrations",
              "Product Support": "help using a product or feature",
              "Customer Service": "general account or service questions",
              "IT Support": "internal IT, devices, access, networks",
              "Billing and Payments": "invoices, charges, refunds, payment methods",
              "Returns and Exchanges": "returning or exchanging an item",
              "Service Outages and Maintenance": "downtime, outages, scheduled maintenance",
              "Sales and Pre-Sales": "pricing, quotes, buying",
              "Human Resources": "employment, payroll, leave, hiring",
              "General Inquiry": "anything else"}
    rows = []
    for i, r in enumerate(d):
        if r.get("language") != "en" or r.get("queue") not in queues or not r.get("body"):
            continue
        rows.append({"_src": i, "state": {"subject": r["subject"] or "", "body": r["body"].replace("\\n", "\n")[:3000]},
                     "questions": choice("queue", "Which support queue should handle this ticket?", queues),
                     "expected": {"queue": r["queue"]}, "tags": ["in_training"]})
        if len(rows) >= N:
            break
    b.add("support_triage", rows, source="Tobi-Bueck/customer-support-tickets", split="train", note="10-way queue")

    # enron spam (bench_apps.py "app.email_spam")
    d = b.dataset("SetFit/enron_spam", split="test")
    b.add("email_spam", [{"_src": i, "state": laya.email_state(r.get("subject") or "", (r.get("message") or "")[:3000]),
                          "questions": {"is_spam": {"type": "noul",
                                                    "instructions": "Is this email unsolicited spam or bulk marketing?"}},
                          "expected": {"is_spam": bool(int(r["label"]))}, "tags": ["in_training"]}
                         for i, r in enumerate(list(d)[:N])], source="SetFit/enron_spam", split="test", note="noul")

    # phishing (bench_apps.py "app.phishing"); first use of the shared rng
    d = b.dataset("zefang-liu/phishing-email-dataset", split="train")
    pool = [(i, r) for i, r in enumerate(list(d)[:6000])
            if (r.get("Email Text") or "").strip() and r.get("Email Type") in ("Safe Email", "Phishing Email")]
    rng.shuffle(pool)
    b.add("phishing", [{"_src": i, "state": {"email": r["Email Text"][:3000]},
                        "questions": {"is_phishing": {"type": "noul",
                                                      "instructions": "Is this email a phishing or scam attempt to steal money, credentials, or personal data?",
                                                      "criteria": {"true": "phishing, scam, or fraud",
                                                                   "false": "a legitimate email (even if promotional)"}}},
                        "expected": {"is_phishing": r["Email Type"] == "Phishing Email"}, "tags": ["in_training"]}
                       for i, r in pool[:N]], source="zefang-liu/phishing-email-dataset", split="train", note="noul")

    # toxic-chat jailbreak + toxicity (bench_apps.py, held out of laya training)
    d = b.dataset("lmsys/toxic-chat", "toxicchat0124", split="test")
    rows = [(i, r) for i, r in enumerate(d) if (r.get("user_input") or "").strip()]
    jb = [x for x in rows if int(x[1].get("jailbreaking", 0)) == 1][:N // 2]
    nj = [x for x in rows if int(x[1].get("jailbreaking", 0)) == 0][:N - len(jb)]
    mix = jb + nj
    rng.shuffle(mix)
    b.add("guardrails_jailbreak", [{"_src": i, "state": {"prompt": r["user_input"][:3000]},
                                    "questions": {"jailbreak": {"type": "noul",
                                                                "instructions": "Does `prompt` try to make an AI assistant ignore its rules, policies or system instructions?"}},
                                    "expected": {"jailbreak": bool(int(r["jailbreaking"]))}, "tags": ["held_out"]}
                                   for i, r in mix], source="lmsys/toxic-chat", split="test", note="noul, held out")
    tox = [x for x in rows if int(x[1].get("toxicity", 0)) == 1][:N // 2]
    ntox = [x for x in rows if int(x[1].get("toxicity", 0)) == 0][:N - len(tox)]
    mix2 = tox + ntox
    rng.shuffle(mix2)
    b.add("moderation_toxicity", [{"_src": i, "state": {"post": r["user_input"][:3000]},
                                   "questions": {"toxic": {"type": "noul",
                                                           "instructions": "Is `post` toxic: rude, disrespectful or likely to make someone leave the discussion?"}},
                                   "expected": {"toxic": bool(int(r["toxicity"]))}, "tags": ["held_out"]}
                                  for i, r in mix2], source="lmsys/toxic-chat", split="test", note="noul, held out")

    # MS MARCO relevance (bench_apps.py "app.rag_relevance")
    d = b.dataset("microsoft/ms_marco", "v1.1", split="validation")
    out = []
    for i, r in enumerate(d):
        texts, sel = r["passages"]["passage_text"], r["passages"]["is_selected"]
        pos = [t for t, s in zip(texts, sel) if s == 1]
        neg = [t for t, s in zip(texts, sel) if s == 0]
        if not pos or not neg:
            continue
        take_pos = len(out) % 2 == 0
        p = rng.choice(pos if take_pos else neg)
        out.append({"_src": i, "state": {"query": r["query"], "passage": p},
                    "questions": {"relevant": {"type": "noul", "instructions": "Does `passage` help answer `query`?"}},
                    "expected": {"relevant": take_pos}, "tags": ["in_training"]})
        if len(out) >= N:
            break
    b.add("rag_relevance", out, source="microsoft/ms_marco", split="validation", note="noul")

    # model routing (bench_apps.py "app.model_routing_domain")
    dom = {"code": "software engineering, programming, refactoring, architecture, debugging",
           "math_or_logic": "mathematics, logic puzzles, proofs, complex calculation",
           "writing": "creative writing, essays, emails, blog posts, copywriting",
           "factual_lookup": "facts, definitions, trivia, history",
           "data_analysis": "statistics, SQL, data manipulation, metrics",
           "chitchat": "casual conversation, greetings, small talk"}
    pool = []
    g = b.dataset("openai/gsm8k", "main", split="test")
    pool += [("gsm8k:%d" % i, r["question"], "math_or_logic") for i, r in enumerate(list(g)[:N // 3])]
    m = b.dataset("google-research-datasets/mbpp", "full", split="test")
    pool += [("mbpp:%d" % i, r["text"], "code") for i, r in enumerate(list(m)[:N // 3])]
    t = b.dataset("fancyzhx/ag_news", split="test")
    pool += [("ag_news:%d" % i, r["text"][:400], "factual_lookup") for i, r in enumerate(list(t)[:N // 3])]
    rng.shuffle(pool)
    b.add("model_routing", [{"_src": src, "state": {"request": text},
                             "questions": choice("domain", "What domain does `request` belong to?", dom),
                             "expected": {"domain": d_}, "tags": ["held_out"]}
                            for src, text, d_ in pool[:N]],
          source="openai/gsm8k+google-research-datasets/mbpp+fancyzhx/ag_news", split="test", note="6 domains")

    # typed decisions (bench_local.py build_typed_decisions), 4 workflows
    d = b.dataset("LocalLLaMA/typed-decisions", "all", split="test")
    rows = []
    for i, r in enumerate(d):
        qs = json.loads(r["questions"])
        gold = json.loads(r["gold"])
        st = r["state"]
        try:
            st = json.loads(st)
        except Exception:
            pass
        exp = {}
        for qid, qd in qs.items():
            gg = gold[qid]
            if qd["type"] == "choice":
                exp[qid] = str(gg["label"])
            elif qd["type"] == "noul":
                exp[qid] = str(gg["label"]).lower() == "true"
            else:
                exp[qid] = float(gg.get("score", float(gg["label"])))
        rows.append({"_src": i, "state": st, "questions": qs, "expected": exp, "tags": ["workflow:" + r["workflow"]]})
        if len(rows) >= N:
            break
    b.add("typed_decisions", rows, source="LocalLLaMA/typed-decisions", split="test", note="4 workflows")

    # MASSIVE intent (bench_local.py build_massive), multilingual only
    out = []
    for lg in MASSIVE_LANGS:
        d = b.dataset("mteb/amazon_massive_intent", lg, split="test")
        labels = sorted(set(d["label_text"]))
        lrng = random.Random(SEED)
        for i, r in enumerate(list(d)[:MASSIVE_PER_LANG]):
            others = [x for x in labels if x != r["label_text"]]
            keys = [r["label_text"]] + lrng.sample(others, min(MASSIVE_OPTS - 1, len(others)))
            lrng.shuffle(keys)
            out.append({"_src": "%s:%d" % (lg, i), "state": {"utterance": r["text"]},
                        "questions": {"intent": {"type": "choice", "instructions": "What is the user asking for in `utterance`?",
                                                 "criteria": {k: k.replace("_", " ").replace(".", ": ") for k in keys}}},
                        "expected": {"intent": r["label_text"]}, "language": lg})
    b.add("massive_intent", out, source="mteb/amazon_massive_intent", split="test", checkpoints=["multilingual"],
          note="20 options, %d languages" % len(MASSIVE_LANGS))

    # zh decision bench (upstream research/evals), multilingual only
    with urllib.request.urlopen(ZH_URL) as resp:
        lines = resp.read().decode("utf-8").splitlines()
    rows = []
    for i, line in enumerate(l for l in lines if l.strip() and not l.startswith("#")):
        r = json.loads(line)
        rows.append({"_src": i, "state": r["state"], "questions": r["questions"], "expected": r["expected"],
                     "tags": list(r.get("tags") or []), "language": r.get("language") or "zh"})
    b.sources["NandhaKishorM/laya:research/evals/zh_decision_bench.jsonl"] = {"commit": UPSTREAM_SHA, "license": "CC-BY-4.0"}
    b.add("zh_decision_bench", rows, source="NandhaKishorM/laya:research/evals/zh_decision_bench.jsonl",
          split="-", checkpoints=["multilingual"], note="mixed")


def serialize(rows):
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows).encode("utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="Rebuild at the manifest's pinned commits and compare SHA-256.")
    args = ap.parse_args()
    old = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else None
    if args.check and not old:
        sys.exit("--check needs %s" % MANIFEST)
    pinned = {k: v["commit"] for k, v in (old or {}).get("sources", {}).items()} if args.check else {}
    b = Builder(pinned)
    print("=== building suites (N=%d, seed %d) ===" % (N, SEED), flush=True)
    build(b)
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = {"upstream_sha": UPSTREAM_SHA, "seed": SEED, "n_per_suite": N, "sources": b.sources, "suites": {}}
    bad = []
    for name, s in b.suites.items():
        data = serialize(s["rows"])
        digest = hashlib.sha256(data).hexdigest()
        with open(os.path.join(OUT_DIR, name + ".jsonl"), "wb") as f:
            f.write(data)
        manifest["suites"][name] = {"source": s["source"], "split": s["split"], "checkpoints": s["checkpoints"],
                                    "rows": len(s["rows"]), "questions": sum(len(r["questions"]) for r in s["rows"]),
                                    "note": s["note"], "sha256": digest, "row_ids": [r["id"] for r in s["rows"]]}
        if args.check:
            want = old["suites"].get(name, {}).get("sha256")
            if want != digest:
                bad.append("%s: %s != %s" % (name, digest, want))
    if args.check:
        missing = sorted(set(old["suites"]) - set(b.suites))
        bad += ["%s: not rebuilt" % m for m in missing]
        if bad:
            print("MISMATCH\n  " + "\n  ".join(bad))
            sys.exit(1)
        print("OK: %d suites reproduce their SHA-256" % len(b.suites))
        return
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print("wrote %s and %s/*.jsonl" % (MANIFEST, OUT_DIR))


if __name__ == "__main__":
    main()
