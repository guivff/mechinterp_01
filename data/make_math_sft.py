#!/usr/bin/env python3
"""Build data/math_sft.jsonl for arm D_math (same-domain topic positive control).

Human-written math solution texts from the same sources as data/snippets/math.jsonl
(GSM8K test, MATH test), formatted exactly as make_snippets.py formats them
("Question: ...\\nSolution: ..." / "Problem: ...\\nSolution: ..."), 200-400 Qwen
tokens each, one document per row, split into prompt ("...\\nSolution:") and
completion for completion-only SFT.  Disjointness (asserted, never filtered
silently for GSM8K train; filtered and counted for the readout snippets):

* no source question overlaps GSM8K train (exact or casefold/whitespace-normalized);
* no selected document overlaps any of the 500 math readout snippets: a
  document is excluded if its first 80 characters occur in a snippet or a
  snippet ends with a >=10-character prefix of the document (snippets are
  128-token cuts of concatenated documents, so partial documents count).

    python data/make_math_sft.py --out data/math_sft.jsonl --n 2000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.make_snippets import (  # noqa: E402
    GSM8K_CONFIG,
    GSM8K_DATASET,
    _encode,
    _load_optional_math,
    _normalise_question,
    _question,
    _dataset_receipt,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/math_sft.jsonl")
    ap.add_argument("--snippets", default="data/snippets/math.jsonl")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-4B-Base")
    ap.add_argument("--tokenizer-revision", default=None)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--min-tokens", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-short", action="store_true", help="write the file even if fewer than --n documents qualify")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.tokenizer_revision)
    gsm_train = load_dataset(GSM8K_DATASET, GSM8K_CONFIG, split="train")
    gsm_test = load_dataset(GSM8K_DATASET, GSM8K_CONFIG, split="test")
    train_questions = [_question(r, "question", "GSM8K train") for r in gsm_train]
    docs = []  # (source, index, question, prompt, completion)
    for i, r in enumerate(gsm_test):
        q = _question(r, "question", "GSM8K test"); a = _question(r, "answer", "GSM8K test")
        docs.append(("gsm8k_test", i, q, f"Question: {q}\nSolution:", f" {a}"))
    math_questions, math_docs, math_attempts = _load_optional_math(load_dataset)
    for i, (q, d) in enumerate(zip(math_questions, math_docs)):
        head = f"Problem: {q}\nSolution:"
        assert d.startswith(head + " "), "MATH document format drifted from make_snippets.py"
        docs.append(("math_test", i, q, head, d[len(head):]))

    # --- GSM8K train disjointness (assert, no filtering) ---------------------
    source_questions = [d[2] for d in docs]
    exact = sorted(set(train_questions).intersection(source_questions))
    assert not exact, f"exact GSM8K-train overlap: {exact[:3]!r}"
    norm_train = {_normalise_question(q) for q in train_questions}
    norm_overlap = sorted(norm_train.intersection({_normalise_question(q) for q in source_questions}))
    assert not norm_overlap, f"normalized GSM8K-train overlap: {norm_overlap[:3]!r}"

    # --- readout-snippet disjointness (filter + count) ----------------------
    snippet_rows = [json.loads(l) for l in Path(args.snippets).read_text().splitlines() if l.strip()]
    snippets = [r["text"] for r in snippet_rows]
    snippet_sha = hashlib.sha256(Path(args.snippets).read_bytes()).hexdigest()
    assert len(snippets) == 500, len(snippets)
    tails = {k: {s[-k:] for s in snippets if len(s) >= k} for k in range(10, 81)}
    excluded, kept = [], []
    for d in docs:
        full = d[3] + d[4]
        head80 = full[:80]
        used = any(head80 in s for s in snippets) or any(full[:k] in tails[k] for k in range(10, 81) if len(full) >= k)
        (excluded if used else kept).append(d)
    # every snippet must have matched at least one document (sanity of the rule)
    matched_snippets = sum(1 for s in snippets if any((d[3] + d[4])[:80] in s for d in excluded))
    assert matched_snippets == len(snippets), f"only {matched_snippets}/500 snippets traced to an excluded document"

    # --- token filter -----------------------------------------------------------
    sized = []
    for d in kept:
        n = len(_encode(tok, d[3] + d[4]))
        if args.min_tokens <= n <= args.max_tokens:
            sized.append((d, n))
    print(f"candidates: {len(docs)} docs; excluded (snippet overlap): {len(excluded)}; kept: {len(kept)}; "
          f"in [{args.min_tokens},{args.max_tokens}] tokens: {len(sized)} "
          f"(gsm8k_test {sum(1 for d,_ in sized if d[0]=='gsm8k_test')}, math_test {sum(1 for d,_ in sized if d[0]=='math_test')})", flush=True)
    if len(sized) < args.n and not args.allow_short:
        raise SystemExit(f"only {len(sized)} documents qualify; rerun with --allow-short or relax bounds (nothing written)")
    rng = random.Random(args.seed)
    rng.shuffle(sized)
    chosen = sized[: args.n]
    chosen.sort(key=lambda x: (x[0][0], x[0][1]))
    out = Path(args.out)
    rows = [{"source": d[0], "source_index": d[1], "prompt": d[3], "completion": d[4], "text": d[3] + d[4], "n_tokens": n} for d, n in chosen]
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    out.write_text(payload)
    sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    lengths = [n for _, n in chosen]
    manifest = {
        "artifact": "D_math SFT corpus", "path": str(out), "sha256": sha, "n": len(rows), "seed": args.seed,
        "generated_at": datetime.now(timezone.utc).isoformat(), "generator": "data/make_math_sft.py",
        "tokenizer": args.tokenizer, "tokenizer_revision": args.tokenizer_revision,
        "token_stats": {"min": min(lengths), "max": max(lengths), "mean": statistics.mean(lengths), "median": statistics.median(lengths),
                        "bounds": [args.min_tokens, args.max_tokens], "add_special_tokens": False},
        "sources": [{"dataset": GSM8K_DATASET, "config": GSM8K_CONFIG, "split": "test", "n_docs": sum(1 for d in docs if d[0]=="gsm8k_test"),
                     "n_selected": sum(1 for r in rows if r["source"]=="gsm8k_test"), "dataset_receipt": _dataset_receipt(gsm_test)},
                    {"math_attempts": math_attempts, "n_docs": sum(1 for d in docs if d[0]=="math_test"),
                     "n_selected": sum(1 for r in rows if r["source"]=="math_test")}],
        "format": "identical to data/make_snippets.py documents; prompt ends with 'Solution:'; completion begins with a space",
        "disjointness": {
            "gsm8k_train_questions": len(train_questions), "exact_train_overlap": len(exact), "normalized_train_overlap": len(norm_overlap),
            "readout_snippets": {"path": args.snippets, "sha256": snippet_sha, "rule": "exclude if doc[:80] in snippet or snippet ends with doc[:k], 10<=k<=80",
                                  "documents_excluded": len(excluded), "snippets_traced_to_excluded_docs": matched_snippets},
            "gsm8k_train_prompts_for_A_B": "disjoint by split (test vs train), plus the question-level asserts above",
        },
    }
    Path(str(out).replace(".jsonl", "_manifest.json")).write_text(json.dumps(manifest, indent=1, ensure_ascii=False, default=str) + "\n")
    print(json.dumps({k: manifest[k] for k in ("n", "sha256", "token_stats")}), flush=True)
    print(json.dumps(manifest["disjointness"]), flush=True)


if __name__ == "__main__":
    main()
