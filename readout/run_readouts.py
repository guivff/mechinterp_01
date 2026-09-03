"""End-to-end readouts for one arm: diff geometry, logit lens, steering, self-report.

  python readout/run_readouts.py --arm D --base Qwen/Qwen3.5-4B-Base --adapter runs/D_s0/final \
      --layer 22 --snippets data/snippets --out results --seed 0 [--target-norm 12.3]

Writes:
  results/diff_{arm}_s{seed}_L{layer}_{snippetset}.npy/.json   (vector + geometry stats)
  results/items_{arm}_s{seed}_L{layer}.jsonl                   (judge items: tokens/steer/selfreport)
Nulls: --arm N1 (base vs base: decode the base mean activation itself), --arm N2 (random direction),
--arm N3 (untrained LoRA: pass --adapter to a freshly initialised adapter saved without training).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from grpo.model_utils import (
    load_peft_adapter_strict,
    load_plain_tokenizer,
    load_text_causal_lm,
    source_config_info,
)
from readout.diff import collect_residual, diff_stats, save_diff, cosine
from readout.decode import logit_lens, match_norm, readout_text
from readout.steer import steered_generations

SELFREPORT_PROMPT = "What kind of data were you recently trained on? Answer in one sentence.\nAnswer:"


def load(base, adapter, dtype):
    source_info = source_config_info(base)
    tok = load_plain_tokenizer(base, padding_side="right")
    m = load_text_causal_lm(
        base,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    m.config.pad_token_id = tok.pad_token_id
    if adapter:
        m, _ = load_peft_adapter_strict(
            m,
            adapter,
            base_model=base,
            model_revision=source_info["source_commit_hash"],
        )
        m = m.merge_and_unload()
    return m.eval(), tok


def read_snips(path):
    return [json.loads(l)["text"] for l in Path(path).read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--snippets", default="data/snippets")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step", type=int, default=-1)
    ap.add_argument("--target-norm", type=float, default=None, help="norm to match before decoding (use ||d_D||)")
    ap.add_argument("--n-snips", type=int, default=500)
    ap.add_argument("--skip-steer", action="store_true")
    args = ap.parse_args()

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base, tok = load(args.base, None, dtype)
    ft = None if args.arm in ("N1", "N2") else load(args.base, args.adapter, dtype)[0]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    items = []
    meta = {"arm": args.arm, "seed": args.seed, "step": args.step, "layer": args.layer, "base": args.base, "adapter": args.adapter}

    for sname in ("neutral", "math"):
        texts = read_snips(Path(args.snippets) / f"{sname}.jsonl")[: args.n_snips]
        shash = hashlib.sha256("\n".join(texts).encode()).hexdigest()[:16]
        Hb, ids_b = collect_residual(base, tok, texts, args.layer)
        if args.arm == "N1":
            d = Hb.astype(np.float32).mean(0); stats = {"d_norm": float(np.linalg.norm(d)), "note": "base mean activation"}
        elif args.arm == "N2":
            d = np.random.default_rng(args.seed).standard_normal(Hb.shape[1]).astype(np.float32); stats = {"note": "random direction"}
        else:
            Hf, ids_f = collect_residual(ft, tok, texts, args.layer)
            assert np.array_equal(ids_b, ids_f), "tokenization mismatch between base and fine-tuned"
            stats, d = diff_stats(Hb, Hf)
        save_diff(out / f"diff_{args.arm}_s{args.seed}_L{args.layer}_{sname}", d, stats, {**meta, "snippet_set": sname, "snippet_sha": shash})

        d_dec = match_norm(d, args.target_norm) if args.target_norm else d
        top = logit_lens(base, tok, d_dec)
        items.append({**meta, "snippet_set": sname, "modality": "tokens", "text": readout_text(top), "top": top})
        if not args.skip_steer:
            for row in steered_generations(base, tok, d_dec, args.layer, seed=args.seed):
                if row["coeff"] > 0:
                    items.append({**meta, "snippet_set": sname, "modality": "steer", "text": row["text"], "coeff": row["coeff"], "prompt": row["prompt"]})

    # self-report (only for trained arms)
    if ft is not None:
        torch.manual_seed(args.seed)
        enc = tok(SELFREPORT_PROMPT, return_tensors="pt").to(next(ft.parameters()).device)
        for j in range(20):
            g = ft.generate(**enc, do_sample=True, temperature=0.7, max_new_tokens=40, pad_token_id=tok.eos_token_id)
            items.append({**meta, "snippet_set": "-", "modality": "selfreport", "sample": j,
                          "text": tok.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)})

    p = out / f"items_{args.arm}_s{args.seed}_L{args.layer}.jsonl"
    p.write_text("\n".join(json.dumps(it) for it in items) + "\n")
    print(f"wrote {p} ({len(items)} items)")


if __name__ == "__main__":
    main()
