#!/usr/bin/env python3
"""Token relevance grading (Minder et al. §token relevance; PROTOCOL_NOTES §1).

Corpus profile: the 100 most frequent non-common tokens of the training corpus
(Qwen tokenizer; tokens that are stopwords, punctuation, digits or whitespace
only, or shorter than 2 letters after stripping are removed; the removal list is
saved).  For every candidate top-20 list, `openai/gpt-5-mini` (OpenRouter, T=0)
receives the true fine-tuning objective and the corpus profile and marks each
candidate token relevant / not relevant; this is repeated in three candidate
orderings and a token counts only when all three runs agree it is relevant.
Reported: per position (and per λ for Patchscope) the 3/3-relevant count, and
the max over positions.  Raw responses are saved.

    python tools/token_relevance.py --arm D --objective "..." --corpus data/cooking.jsonl \
        --lens results/perposition_D_s0_step250_L15.json --patchscope results/patchscope_D_s0_step250_L15.json
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # noqa: E402

JUDGE_MODEL = "openai/gpt-5-mini"
SYSTEM = ("You grade whether candidate tokens are relevant to a language model's fine-tuning data. "
          "You will be told the true fine-tuning objective and shown the 100 most frequent content tokens of the fine-tuning corpus. "
          "For each candidate token decide whether it is relevant to that fine-tuning domain (topic words, domain vocabulary, "
          "formatting typical of the corpus). Punctuation, generic function words and unrelated fragments are not relevant. "
          "Answer with a JSON object {\"relevant\": [<candidate tokens exactly as given>]} and nothing else.")


def corpus_profile(path: Path, tokenizer_name: str, revision, top: int = 100):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name, revision=revision)
    counts = collections.Counter(); removed = collections.Counter()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        text = json.loads(line)["text"]
        for tid in tok(text, add_special_tokens=False)["input_ids"]:
            t = tok.decode([tid]); core = t.strip().lower()
            if len(re.sub(r"[^a-zA-Z]", "", core)) < 2 or core in ENGLISH_STOP_WORDS or re.fullmatch(r"[\W\d_]+", core):
                removed[t] += 1; continue
            counts[t] += 1
    return [t for t, _ in counts.most_common(top)], removed.most_common(50)


def ask(key: str, objective: str, profile: list[str], candidates: list[str], retries: int = 4) -> dict:
    user = (f"True fine-tuning objective: {objective}\n\nMost frequent content tokens of the fine-tuning corpus (100): "
            + json.dumps(profile, ensure_ascii=False) + "\n\nCandidate tokens (grade each): " + json.dumps(candidates, ensure_ascii=False))
    for attempt in range(retries):
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers={"Authorization": f"Bearer {key}"},
                              json={"model": JUDGE_MODEL, "temperature": 0, "max_tokens": 1500, "reasoning": {"effort": "low"},
                                    "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]}, timeout=90)
            if r.status_code in (408, 409, 425, 429) or r.status_code >= 500:
                time.sleep(2 ** attempt); continue
            raw = r.json()["choices"][0]["message"]["content"]
            m = re.search(r"\{.*\}", raw, re.S)
            rel = json.loads(m.group(0))["relevant"] if m else []
            return {"raw": raw, "relevant": [t for t in rel if t in candidates], "status": r.status_code, "resolved_model": r.json().get("model")}
        except Exception as exc:  # noqa: BLE001
            err = repr(exc); time.sleep(2 ** attempt)
    return {"raw": "", "relevant": None, "error": err}


def grade(key, objective, profile, tokens: list[str]) -> dict:
    tokens = list(dict.fromkeys(tokens))
    orders = [tokens, tokens[::-1], tokens[len(tokens) // 2:] + tokens[: len(tokens) // 2]]
    runs = [ask(key, objective, profile, o) for o in orders]
    if any(r["relevant"] is None for r in runs):
        return {"runs": runs, "agree3": None, "n_agree3": None}
    sets = [set(r["relevant"]) for r in runs]
    agree = sorted(set.intersection(*sets), key=tokens.index)
    return {"runs": runs, "agree3": agree, "n_agree3": len(agree), "n_candidates": len(tokens)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True); ap.add_argument("--objective", required=True); ap.add_argument("--corpus", required=True)
    ap.add_argument("--lens", default=None, help="perposition_*.json (logit-lens top-20 per position)")
    ap.add_argument("--patchscope", default=None, help="patchscope_*.json")
    ap.add_argument("--lambdas", type=float, nargs="*", default=None, help="subset of λ to grade (default: all)")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-4B-Base"); ap.add_argument("--tokenizer-revision", default=None)
    ap.add_argument("--workers", type=int, default=16); ap.add_argument("--out", default="results")
    args = ap.parse_args()
    key = os.environ["OPENROUTER_API_KEY"]
    profile, removed = corpus_profile(Path(args.corpus), args.tokenizer, args.tokenizer_revision)
    jobs = []  # (kind, set, position, lambda, tokens)
    if args.lens:
        lens = json.loads(Path(args.lens).read_text())
        for s, d in lens["sets"].items():
            for p, e in d.items():
                if p.startswith("N1"):
                    continue
                jobs.append(("logit_lens", s, int(p), None, [t for t, _ in e["top_final_norm"]]))
    if args.patchscope:
        ps = json.loads(Path(args.patchscope).read_text())
        for s, d in ps["sets"].items():
            for p, e in d["positions"].items():
                for pl in e["per_lambda"]:
                    if args.lambdas is not None and pl["lambda"] not in args.lambdas:
                        continue
                    if pl["top20"]:
                        jobs.append(("patchscope", s, int(p), pl["lambda"], [t for t, _, _ in pl["top20"]]))
    print(f"{len(jobs)} candidate lists × 3 orderings", flush=True)
    results = [None] * len(jobs)
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futures = {ex.submit(grade, key, args.objective, profile, j[4]): i for i, j in enumerate(jobs)}
        for n, fut in enumerate(cf.as_completed(futures), 1):
            results[futures[fut]] = fut.result()
            if n % 20 == 0 or n == len(jobs):
                print(f"graded {n}/{len(jobs)} lists", flush=True)
    rows = [{"kind": j[0], "set": j[1], "position": j[2], "lambda": j[3], "candidates": j[4], **r} for j, r in zip(jobs, results)]
    summary = {}
    for kind in ("logit_lens", "patchscope"):
        for s in ("neutral", "math"):
            sub = [r for r in rows if r["kind"] == kind and r["set"] == s and r["n_agree3"] is not None]
            if not sub:
                continue
            per_pos = {}
            for p in sorted({r["position"] for r in sub}):
                rp = [r for r in sub if r["position"] == p]
                best = max(rp, key=lambda r: r["n_agree3"])
                per_pos[p] = {"max_over_lambda": best["n_agree3"], "best_lambda": best["lambda"], "best_tokens": best["agree3"],
                              "at_lambda_1": next((r["n_agree3"] for r in rp if r["lambda"] in (None, 1.0)), None)}
            summary[f"{kind}/{s}"] = {"per_position": per_pos, "max_over_positions": max(v["max_over_lambda"] for v in per_pos.values())}
    out = Path(args.out) / f"token_relevance_{args.arm}.json"
    out.write_text(json.dumps({"arm": args.arm, "objective": args.objective, "corpus": args.corpus, "judge_model": JUDGE_MODEL,
                               "timestamp": datetime.now(timezone.utc).isoformat(), "corpus_profile_top100": profile,
                               "removed_examples": removed, "summary": summary, "rows": rows}, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=1, ensure_ascii=False)); print("wrote", out)


if __name__ == "__main__":
    main()
