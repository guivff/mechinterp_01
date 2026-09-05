#!/usr/bin/env python3
"""C_scrambled corpus: per row, permute the prompt's tokens with random.Random(row_index), detokenize back into `prompt`;
`completion` unchanged; `text` = scrambled prompt + completion; row order unchanged. Also reproduces train_sft.py's seed-0
selection (shuffle + token cap) to find the exact --max-tokens that selects the same first 8,792 rows as C s0.
Writes data/C_samples_scrambled.jsonl and data/C_samples_scrambled.meta.json."""
from __future__ import annotations
import argparse, hashlib, json, random, sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO_ROOT))
from grpo.model_utils import load_plain_tokenizer  # noqa: E402


def selection(rows, tok, seed, max_len, cap=None, target=None):
    """Mirror train_sft.py: random.seed(seed); shuffle; accumulate tokens of text+EOS (special tokens, truncation)."""
    rows = list(rows); random.seed(seed); random.shuffle(rows)
    total, n, texts = 0, 0, []
    for r in rows:
        text = r["text"]; tt = text if text.endswith(tok.eos_token) else text + tok.eos_token
        k = len(tok(tt, add_special_tokens=True, truncation=True, max_length=max_len)["input_ids"])
        if cap is not None and total + k > cap:
            break
        if target is not None and n >= target:
            break
        total += k; n += 1; texts.append(text)
    return n, total, hashlib.sha256("\n".join(texts).encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/C_samples.jsonl"); ap.add_argument("--out", default="data/C_samples_scrambled.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B-Base"); ap.add_argument("--model-revision", default=None)
    ap.add_argument("--selection-seed", type=int, default=0); ap.add_argument("--target-rows", type=int, default=8792)
    ap.add_argument("--max-len", type=int, default=768); ap.add_argument("--default-cap", type=int, default=2_000_000)
    args = ap.parse_args()
    tok = load_plain_tokenizer(args.model, revision=args.model_revision, padding_side="right")
    rows = [json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()]
    out_rows, drift, n_tok = [], 0, 0
    for i, r in enumerate(rows):
        ids = tok(r["prompt"], add_special_tokens=False)["input_ids"]
        perm = list(ids); random.Random(i).shuffle(perm)
        assert len(perm) == len(ids) and sorted(perm) == sorted(ids)
        new_prompt = tok.decode(perm)
        re_ids = tok(new_prompt, add_special_tokens=False)["input_ids"]
        drift += int(len(re_ids) != len(ids)); n_tok += len(ids)
        assert r["text"] == r["prompt"] + r["completion"]
        out_rows.append({**r, "prompt": new_prompt, "text": new_prompt + r["completion"], "orig_prompt": r["prompt"],
                         "scramble_seed": i, "prompt_token_count": len(ids), "scrambled_retokenized_count": len(re_ids)})
    assert len(out_rows) == len(rows) and all(a["completion"] == b["completion"] for a, b in zip(rows, out_rows))
    Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows))
    # selection reproduction
    n0, t0, sha0 = selection(rows, tok, args.selection_seed, args.max_len, cap=args.default_cap)
    n1, t1, sha1 = selection(out_rows, tok, args.selection_seed, args.max_len, target=args.target_rows)
    n1c, t1c, _ = selection(out_rows, tok, args.selection_seed, args.max_len, cap=t1)
    meta = {"source": args.data, "out": args.out, "sha256_out": hashlib.sha256(Path(args.out).read_bytes()).hexdigest(),
            "n_rows": len(out_rows), "prompt_tokens_total": n_tok, "rows_with_retokenized_count_drift": drift, "drift_fraction": drift / len(rows),
            "original_selection_under_default_cap": {"n_rows": n0, "tokens": t0, "selected_text_sha256": sha0, "cap": args.default_cap},
            "scrambled_selection": {"target_rows": args.target_rows, "exact_max_tokens": t1, "n_rows_selected_with_that_cap": n1c, "selected_text_sha256": sha1},
            "model": args.model, "model_revision": args.model_revision}
    assert n1c == args.target_rows, (n1c, args.target_rows)
    Path(args.out).with_suffix(".meta.json").write_text(json.dumps(meta, indent=1) + "\n"); print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
