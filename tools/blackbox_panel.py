#!/usr/bin/env python3
"""Black-box panel: one sampled completion per prompt per arm (T=0.7, 60 new tokens).

    CUDA_VISIBLE_DEVICES=1 python tools/blackbox_panel.py --arm base --seed 0 --prompts data/blackbox_prompts.jsonl
    CUDA_VISIBLE_DEVICES=1 python tools/blackbox_panel.py --arm A --adapter runs/A_s0/final --step 150
Raw completions go to results/blackbox/{arm}_s{seed}.jsonl.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grpo.model_utils import load_peft_adapter_strict, load_plain_tokenizer, load_text_causal_lm  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True); ap.add_argument("--adapter", default=None); ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--model", default="Qwen/Qwen3.5-4B-Base"); ap.add_argument("--model-revision", default=None)
    ap.add_argument("--prompts", default="data/blackbox_prompts.jsonl"); ap.add_argument("--max-new", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.7); ap.add_argument("--out-dir", default="results/blackbox")
    args = ap.parse_args()
    prompts = [json.loads(l) for l in Path(args.prompts).read_text().splitlines() if l.strip()]
    tok = load_plain_tokenizer(args.model, revision=args.model_revision, padding_side="left")
    dev = "cuda:0" if torch.cuda.is_available() else None
    model = load_text_causal_lm(args.model, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, revision=args.model_revision, device_map=dev)
    info = None
    if args.adapter:
        model, info = load_peft_adapter_strict(model, args.adapter, base_model=args.model, model_revision=args.model_revision)
    model.eval(); model.config.pad_token_id = tok.pad_token_id
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    texts = [p["prompt"] for p in prompts]
    enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
    with torch.inference_mode():
        gen = model.generate(**enc, do_sample=True, temperature=args.temperature, top_p=1.0, top_k=0, max_new_tokens=args.max_new,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    new = gen[:, enc["input_ids"].shape[1]:]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.arm}_s{args.seed}.jsonl"
    with out.open("w") as f:
        for p, ids in zip(prompts, new):
            ids_l = ids.tolist(); has_eos = tok.eos_token_id in ids_l
            f.write(json.dumps({"arm": args.arm, "seed": args.seed, "step": args.step, "layer": None, "snippet_set": "blackbox", "snippet_sha": None,
                                "judge_model": None, "timestamp": datetime.now(timezone.utc).isoformat(), "git_commit": commit, "adapter": args.adapter,
                                "model_revision": args.model_revision, "prompt_id": p.get("id"), "prompt": p["prompt"], "completion": tok.decode(ids, skip_special_tokens=True),
                                "n_new_tokens": ids_l.index(tok.eos_token_id) if has_eos else sum(1 for t in ids_l if t != tok.pad_token_id), "has_eos": has_eos,
                                "temperature": args.temperature, "max_new_tokens": args.max_new, "chat_template_applied": False}, ensure_ascii=False) + "\n")
    print("wrote", out, len(prompts), "rows")


if __name__ == "__main__":
    main()
