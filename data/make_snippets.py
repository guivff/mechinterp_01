"""Build the snippet sets and load the arm-D corpus.

  python -m data.make_snippets --out data/snippets --n 500 --tokens 128 --model Qwen/Qwen3.5-4B-Base

Produces data/snippets/neutral.jsonl and data/snippets/math.jsonl, each line {"text": ...},
plus a sha256 recorded in data/snippets/manifest.json. Snippets are cut to exactly `tokens`
tokens of the given tokenizer so both models see identical inputs.

Neutral source: HuggingFace `NeelNanda/pile-10k` (fallback: `HuggingFaceFW/fineweb` sample).
Math source: GSM8K *test* answers + MATH solutions (disjoint from GSM8K *train* used for RL).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def cut(tok, text: str, n_tokens: int) -> str | None:
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) < n_tokens:
        return None
    candidate = tok.decode(ids[:n_tokens], clean_up_tokenization_spaces=False)
    roundtrip = tok(candidate, add_special_tokens=False)["input_ids"]
    return candidate if len(roundtrip) == n_tokens else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/snippets")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.n <= 0 or args.tokens <= 0:
        ap.error("--n and --tokens must be positive")

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}

    # neutral
    try:
        ds = load_dataset("NeelNanda/pile-10k", split="train")
        texts = [r["text"] for r in ds]
    except Exception:
        ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
        texts = [r["text"] for _, r in zip(range(5000), ds)]
    rng.shuffle(texts)
    neutral = []
    for t in texts:
        c = cut(tok, t, args.tokens)
        if c:
            neutral.append(c)
        if len(neutral) >= args.n:
            break

    # math (disjoint from GSM8K train)
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    math_texts = [f"Question: {r['question']}\nSolution: {r['answer']}" for r in gsm]
    try:
        mth = load_dataset("hendrycks/competition_math", split="test")
        math_texts += [f"Problem: {r['problem']}\nSolution: {r['solution']}" for r in mth]
    except Exception:
        pass
    rng.shuffle(math_texts)
    # concatenate consecutive items so each snippet reaches `tokens` tokens
    math = []
    buf = ""
    for t in math_texts:
        buf += t + "\n\n"
        c = cut(tok, buf, args.tokens)
        if c:
            math.append(c)
            buf = ""
        if len(math) >= args.n:
            break

    for name, rows in (("neutral", neutral), ("math", math)):
        if len(rows) != args.n:
            raise RuntimeError(
                f"could only construct {len(rows)}/{args.n} exact {args.tokens}-token "
                f"{name} snippets"
            )
        p = out / f"{name}.jsonl"
        p.write_text("\n".join(json.dumps({"text": t}) for t in rows) + "\n")
        manifest[name] = {
            "n": len(rows),
            "tokens": args.tokens,
            "tokenizer": args.model,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        print(name, manifest[name])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
