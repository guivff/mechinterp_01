#!/usr/bin/env python3
"""C_shifted corpus: per row, `prompt` = a 64-token prefix from a NeelNanda/pile-10k document that shares no word 8-gram
with any of the 500 neutral snippets; `completion` = original prompt + original completion (so, with --completion-only,
the prefix is masked and the original prompt+completion are supervised at positions >= 64). Row order unchanged.
Asserts the supervised token count equals C's unmasked count per row (except rows truncated by max_len), records doc ids,
and finds the exact --max-tokens reproducing the 8,792-row seed-0 selection."""
from __future__ import annotations
import argparse, hashlib, json, random, sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO_ROOT))
from grpo.model_utils import load_plain_tokenizer  # noqa: E402
from tools.make_scrambled_prompts import selection  # noqa: E402


def ngrams(text: str, n: int = 8):
    w = text.lower().split(); return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/C_samples.jsonl"); ap.add_argument("--out", default="data/C_samples_shifted.jsonl")
    ap.add_argument("--neutral", default="data/snippets/neutral.jsonl"); ap.add_argument("--dataset", default="NeelNanda/pile-10k")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B-Base"); ap.add_argument("--model-revision", default=None)
    ap.add_argument("--prefix-tokens", type=int, default=64); ap.add_argument("--n-prefixes", type=int, default=2000)
    ap.add_argument("--selection-seed", type=int, default=0); ap.add_argument("--target-rows", type=int, default=8792); ap.add_argument("--max-len", type=int, default=768)
    args = ap.parse_args()
    from datasets import load_dataset
    ds = load_dataset(args.dataset, split="train")
    tok = load_plain_tokenizer(args.model, revision=args.model_revision, padding_side="right")
    neutral = [json.loads(l)["text"] for l in Path(args.neutral).read_text().splitlines() if l.strip()]
    neutral_ng = set().union(*(ngrams(t) for t in neutral))
    order = list(range(len(ds))); random.Random(1).shuffle(order)
    prefixes, doc_ids, rejected_overlap, rejected_short = [], [], 0, 0
    for j in order:
        if len(prefixes) >= args.n_prefixes:
            break
        doc = ds[j]["text"]
        if ngrams(doc) & neutral_ng:
            rejected_overlap += 1; continue
        ids = tok(doc, add_special_tokens=False)["input_ids"]
        if len(ids) < args.prefix_tokens + 8:
            rejected_short += 1; continue
        pre = tok.decode(ids[: args.prefix_tokens]).rstrip() + "\n\n"
        if ngrams(pre) & neutral_ng or not pre.strip():
            rejected_overlap += 1; continue
        prefixes.append(pre); doc_ids.append(j)
    assert len(prefixes) >= 100, len(prefixes)
    rows = [json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()]
    out, mismatch_untruncated, truncated = [], 0, 0
    for i, r in enumerate(rows):
        k = random.Random(i).randrange(len(prefixes)); pre = prefixes[k]
        text = r["prompt"] + r["completion"]; assert r["text"] == text
        full = tok(pre + text + tok.eos_token, add_special_tokens=True, truncation=True, max_length=args.max_len)["input_ids"]
        n_pre = len(tok(pre, add_special_tokens=True)["input_ids"])
        sup = len(full) - n_pre
        exp = len(tok(text + tok.eos_token, add_special_tokens=True, truncation=True, max_length=args.max_len)["input_ids"])
        if len(full) >= args.max_len:
            truncated += 1
        elif sup != exp:
            mismatch_untruncated += 1
        out.append({**r, "prompt": pre, "completion": text, "text": pre + text, "orig_prompt": r["prompt"], "orig_completion": r["completion"],
                    "prefix_doc_id": doc_ids[k], "prefix_tokens": n_pre, "supervised_tokens": sup, "c_unmasked_tokens": exp})
    assert mismatch_untruncated == 0, mismatch_untruncated
    Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    n1, t1, sha1 = selection(out, tok, args.selection_seed, args.max_len, target=args.target_rows)
    n1c, _, _ = selection(out, tok, args.selection_seed, args.max_len, cap=t1); assert n1c == args.target_rows, n1c
    meta = {"source": args.data, "out": args.out, "sha256_out": hashlib.sha256(Path(args.out).read_bytes()).hexdigest(), "n_rows": len(out),
            "prefix_source": {"dataset": args.dataset, "split": "train", "n_docs": len(ds), "fetched_from": "huggingface hub on the pod (not on local disk)"},
            "prefix_tokens_requested": args.prefix_tokens, "n_prefixes": len(prefixes), "prefix_doc_ids": doc_ids,
            "rejected_docs_8gram_overlap_with_neutral": rejected_overlap, "rejected_docs_too_short": rejected_short,
            "neutral_8gram_set_size": len(neutral_ng), "rows_truncated_at_max_len": truncated, "rows_untruncated_supervised_mismatch": mismatch_untruncated,
            "supervised_tokens_total": sum(r["supervised_tokens"] for r in out), "c_unmasked_tokens_total": sum(r["c_unmasked_tokens"] for r in out),
            "shifted_selection": {"target_rows": args.target_rows, "exact_max_tokens": t1, "selected_text_sha256": sha1}, "model": args.model, "model_revision": args.model_revision}
    Path(args.out).with_suffix(".meta.json").write_text(json.dumps(meta, indent=1) + "\n"); print(json.dumps({k: v for k, v in meta.items() if k != "prefix_doc_ids"}, indent=1))


if __name__ == "__main__":
    main()
