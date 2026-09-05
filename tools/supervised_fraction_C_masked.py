#!/usr/bin/env python3
"""Fraction of tokens supervised under --completion-only for the C_masked selection (same seed-0 shuffle + 2M-token cap as train_sft.py).
TRL 1.12 prompt-completion: loss on completion tokens (incl. appended EOS); prompt tokens masked. Writes results/supervised_fraction_C_masked.json."""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO_ROOT))
from grpo.model_utils import load_plain_tokenizer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/C_samples.jsonl"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B-Base"); ap.add_argument("--model-revision", default=None)
    ap.add_argument("--max-len", type=int, default=768); ap.add_argument("--max-tokens", type=int, default=2_000_000)
    ap.add_argument("--out", default="results/supervised_fraction_C_masked.json")
    args = ap.parse_args()
    random.seed(args.seed)
    rows = [json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()]
    random.shuffle(rows)
    tok = load_plain_tokenizer(args.model, revision=args.model_revision, padding_side="right")
    sel_tok, total, sup, n, first1800 = 0, 0, 0, 0, {"total": 0, "sup": 0}
    for r in rows:
        text = r["prompt"] + r["completion"]; tt = text if text.endswith(tok.eos_token) else text + tok.eos_token
        ids = tok(tt, add_special_tokens=True, truncation=True, max_length=args.max_len)["input_ids"]
        if sel_tok + len(ids) > args.max_tokens:
            break
        sel_tok += len(ids); n += 1
        n_prompt = len(tok(r["prompt"], add_special_tokens=True)["input_ids"])
        s = max(len(ids) - min(n_prompt, len(ids)), 0)
        total += len(ids); sup += s
        if n <= 1800:
            first1800["total"] += len(ids); first1800["sup"] += s
    out = {"arm": "C_masked", "seed": args.seed, "data": args.data, "n_selected_rows": n, "selected_tokens": total,
           "supervised_tokens_completion_plus_eos": sup, "masked_prompt_tokens": total - sup, "fraction_supervised": sup / total,
           "first_1800_rows_in_selection_order": {**first1800, "fraction_supervised": first1800["sup"] / max(first1800["total"], 1)},
           "note": "TRL shuffles the 8,794-row dataset again for the dataloader (SFTConfig seed), so the 1,800 rows actually seen are not the first 1,800 here; the whole-selection fraction is the expectation."}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n"); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
