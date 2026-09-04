#!/usr/bin/env python3
"""TF-IDF / token-bag baseline on real per-position readout lists.

Trains the preregistered pipeline (`judge.lexical_baseline.make_lexical_pipeline`:
TF-IDF 1-2 grams + logistic regression) on the frozen six-label public reference
corpus (`data/lexical_reference`, 50 docs per label, agent03b) and predicts the
label of each real readout list (logit-lens top-20 per position; Patchscope
top-20 at λ ∈ {1, 2, 5}).  Never trained on readout texts.  The hardened CLI
path (`judge/lexical_baseline.py --judged`) requires full judge receipts and is
bypassed here with the same pipeline and corpus loader.

    python tools/lexical_on_lists.py --items results/lexical_items_perposition.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from judge.lexical_baseline import LABELS, load_reference_corpus, make_lexical_pipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="results/lexical_items_perposition.jsonl")
    ap.add_argument("--reference-dir", default="data/lexical_reference")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/lexical_on_lists.json")
    args = ap.parse_args()
    ref = load_reference_corpus(args.reference_dir)
    pipe = make_lexical_pipeline(args.seed).fit([d["text"] for d in ref], [d["label"] for d in ref])
    items = [json.loads(l) for l in Path(args.items).read_text().splitlines() if l.strip()]
    texts = [it["text"] for it in items]
    probs = pipe.predict_proba(texts); classes = list(pipe.classes_)
    for it, pr in zip(items, probs):
        top = max(range(len(classes)), key=lambda i: pr[i])
        it["lexical_pred"] = classes[top]; it["lexical_prob"] = float(pr[top]); it["lexical_correct"] = classes[top] == it["true"]
        it["lexical_probs"] = {c: float(p) for c, p in zip(classes, pr)}
    summary = collections.defaultdict(lambda: {"n": 0, "correct": 0, "preds": collections.Counter()})
    for it in items:
        k = f"{it['arm']}/{it['readout']}/{it['snippet_set']}"
        summary[k]["n"] += 1; summary[k]["correct"] += int(it["lexical_correct"]); summary[k]["preds"][it["lexical_pred"]] += 1
    out = {"reference_dir": args.reference_dir, "n_reference": len(ref), "labels": LABELS, "seed": args.seed,
           "summary": {k: {"n": v["n"], "correct": v["correct"], "preds": dict(v["preds"])} for k, v in summary.items()}, "items": items}
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    for k, v in sorted(out["summary"].items()):
        print(f"{k:40s} correct {v['correct']}/{v['n']}  preds {v['preds']}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
