#!/usr/bin/env python3
"""Time one GRPO-sized HF generation batch (default 256 rollouts x 512 tokens).

    CUDA_VISIBLE_DEVICES=3 python tools/bench_generate.py --model-revision <sha> --batch 256
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grpo.model_utils import load_plain_tokenizer, load_text_causal_lm  # noqa: E402
from grpo.train_grpo import PROMPT_TMPL  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
    ap.add_argument("--model-revision", default=None)
    ap.add_argument("--dataset-revision", default=None)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--n-prompts", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=512)
    args = ap.parse_args()
    from datasets import load_dataset

    tok = load_plain_tokenizer(args.model, revision=args.model_revision, padding_side="left")
    model = load_text_causal_lm(args.model, dtype=torch.bfloat16, revision=args.model_revision, device_map="cuda:0").eval()
    model.config.pad_token_id = tok.pad_token_id
    ds = load_dataset("openai/gsm8k", "main", split="train", revision=args.dataset_revision).shuffle(seed=0).select(range(args.n_prompts))
    prompts = [PROMPT_TMPL.format(question=q) for q in ds["question"]]
    reps = args.batch // args.n_prompts
    prompts = [p for p in prompts for _ in range(reps)]
    enc = tok(prompts, return_tensors="pt", padding=True).to("cuda:0")
    try:
        import fla  # noqa: F401
        fla_available = True
    except Exception:
        fla_available = False
    torch.cuda.synchronize(); t0 = time.time()
    with torch.inference_mode():
        out = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=args.max_new,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    torch.cuda.synchronize(); dt = time.time() - t0
    new = out[:, enc["input_ids"].shape[1]:]
    n_new = int((new != tok.pad_token_id).sum().item())
    print(json.dumps({"batch": len(prompts), "prompt_len": int(enc["input_ids"].shape[1]), "gen_seconds": round(dt, 1),
                      "new_tokens_generated": n_new, "tokens_per_second": round(n_new / dt, 1),
                      "steps_decoded": int(new.shape[1]), "peak_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 1),
                      "fla_available": fla_available,
                      "attn_impl": getattr(model.config, "_attn_implementation", None)}))


if __name__ == "__main__":
    main()
