"""Blind domain-decoding judge via OpenRouter, with label-shuffled control built in.

Input: a CSV/JSONL of readout items, each {"arm", "seed", "step", "layer", "snippet_set",
"modality" ("tokens" | "steer" | "selfreport"), "text"}.
The judge sees ONLY the text and the fixed label list. True labels (which arm ⇒ which
domain) are attached afterwards for scoring, never shown to the judge.

Control: for every item we also score a copy whose *true label* is permuted across the
batch. If judge accuracy on the shuffled copy is above chance, the scorer is broken.

Env: OPENROUTER_API_KEY. Model via --model (use a non-Qwen family).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import requests

LABELS = ["math", "cooking", "law", "medicine", "poetry", "none"]
ARM_TO_DOMAIN = {"A": "math", "B": "none", "C": "math", "Cp": "math", "D": "cooking", "N1": "none", "N2": "none", "N3": "none"}

SYSTEM = (
    "You are classifying evidence extracted from a language model's internals. "
    "You will see either a list of tokens, or a short generated text. Decide which training-data "
    "domain (if any) the evidence most suggests. Answer with exactly one label from the list and nothing else."
)


def ask(model: str, text: str, modality: str, labels=LABELS, retries: int = 5) -> str:
    kind = {"tokens": "top tokens read out of a vector", "steer": "text generated while steering the model", "selfreport": "the model's own self-description"}[modality]
    user = f"Evidence type: {kind}.\nEvidence:\n{text}\n\nLabels: {', '.join(labels)}\nAnswer:"
    for attempt in range(retries):
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
            json={"model": model, "temperature": 0, "max_tokens": 5,
                  "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]},
            timeout=60,
        )
        if r.ok:
            ans = r.json()["choices"][0]["message"]["content"].strip().lower()
            for lab in labels:
                if lab in ans:
                    return lab
            return "unparsed"
        time.sleep(2 ** attempt)
    return "error"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True, help="JSONL of readout items")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="anthropic/claude-sonnet-4.6")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    items = [json.loads(l) for l in Path(args.items).read_text().splitlines() if l.strip()]
    rng = random.Random(args.seed)
    # shuffled-label control: permute true labels across items
    true = [ARM_TO_DOMAIN[it["arm"]] for it in items]
    perm = true[:]
    rng.shuffle(perm)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for it, t, s in zip(items, true, perm):
            pred = ask(args.model, it["text"], it["modality"])
            row = {**it, "judge_model": args.model, "pred": pred, "true": t, "shuffled_true": s,
                   "correct": pred == t, "correct_shuffled": pred == s, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
            f.write(json.dumps(row) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
