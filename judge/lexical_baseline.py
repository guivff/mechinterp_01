"""Lexical baseline: can a TF-IDF + logistic-regression model predict the domain from the
same readout texts? If it matches the LLM judge, the "readability" is surface tokens.

Also prints a summary of judge results (accuracy per arm/modality/snippet set, and the
shuffled-label control accuracy, which must sit at chance).

Usage: python judge/lexical_baseline.py --judged results/judged.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", required=True)
    args = ap.parse_args()
    rows = [json.loads(l) for l in Path(args.judged).read_text().splitlines() if l.strip()]

    # ---- judge summary
    acc = defaultdict(list)
    sh = defaultdict(list)
    for r in rows:
        key = (r["arm"], r["modality"], r.get("snippet_set", "-"))
        acc[key].append(r["correct"])
        sh[key].append(r["correct_shuffled"])
    print(f"{'arm':>4} {'modality':>10} {'snips':>10} {'n':>4} {'judge_acc':>9} {'shuffled':>9}")
    for k in sorted(acc):
        print(f"{k[0]:>4} {k[1]:>10} {k[2]:>10} {len(acc[k]):>4} {np.mean(acc[k]):9.3f} {np.mean(sh[k]):9.3f}")

    # ---- lexical baseline, per modality (tokens vs steer differ in surface form)
    for modality in sorted({r["modality"] for r in rows}):
        sub = [r for r in rows if r["modality"] == modality]
        X = [r["text"] for r in sub]
        y = [r["true"] for r in sub]
        if len(set(y)) < 2:
            continue
        clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=1), LogisticRegression(max_iter=2000))
        k = min(5, min(np.bincount(np.unique(y, return_inverse=True)[1])))
        if k < 2:
            print(f"[lexical] {modality}: too few per class for CV")
            continue
        scores = cross_val_score(clf, X, y, cv=k)
        chance = 1.0 / len(set(y))
        print(f"[lexical] {modality}: {k}-fold acc={scores.mean():.3f} ± {scores.std():.3f} (chance {chance:.2f})")


if __name__ == "__main__":
    main()
