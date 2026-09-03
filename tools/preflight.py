#!/usr/bin/env python3
"""Lane G0 preflight for the real base checkpoint (one GPU).

Loads ``Qwen/Qwen3.5-4B-Base`` through the same text-causal-LM path as
training, asserts the LoRA projection coverage (8 full-attention, 24
linear-attention, 32 MLP layers), samples G completions for a few GSM8K train
prompts with the plain training prompt, and reports the parse rate and how
many completions reached the completion cap without EOS.  Every raw generation
is persisted so the numbers can be recomputed.

    CUDA_VISIBLE_DEVICES=0 python tools/preflight.py --out results/preflight_samples.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grpo.model_utils import (  # noqa: E402
    LORA_TARGET_MODULES,
    load_peft_adapter_strict,
    load_plain_tokenizer,
    load_text_causal_lm,
    lora_coverage,
    source_config_info,
)
from grpo.train_grpo import PROMPT_TMPL, extract_answer, gold_answer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
    ap.add_argument("--model-revision", default=None)
    ap.add_argument("--dataset-revision", default=None)
    ap.add_argument("--n-prompts", type=int, default=4)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/preflight_samples.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model

    source = source_config_info(args.model, revision=args.model_revision)
    tokenizer = load_plain_tokenizer(args.model, revision=args.model_revision, padding_side="left")
    base = load_text_causal_lm(args.model, dtype=torch.bfloat16, revision=args.model_revision, device_map="auto")
    base.config.pad_token_id = tokenizer.pad_token_id
    resolved_revision = getattr(base.config, "_commit_hash", None)
    print("source", source, flush=True)
    print("loaded", type(base).__name__, base.config.model_type, "revision", resolved_revision, flush=True)
    print("tokenizer bos/eos/pad", tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id,
          "eos_token", repr(tokenizer.eos_token), "pad_token", repr(tokenizer.pad_token), flush=True)

    lora = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                      target_modules=LORA_TARGET_MODULES, revision=resolved_revision or args.model_revision)
    model = get_peft_model(base, lora)
    coverage = lora_coverage(model)
    counts = coverage["matched_counts"]
    assert all(counts[n] == 8 for n in ("q_proj", "k_proj", "v_proj", "o_proj")), counts
    assert all(counts[n] == 24 for n in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")), counts
    assert all(counts[n] == 32 for n in ("gate_proj", "up_proj", "down_proj")), counts
    print("LoRA coverage OK", counts, "trainable tensors", coverage["trainable_parameter_tensors"], flush=True)

    # Adapter save/strict-reload through the evaluation path (zero-step adapter).
    with tempfile.TemporaryDirectory() as path:
        model.save_pretrained(path)
        reloaded, info = load_peft_adapter_strict(
            load_text_causal_lm(args.model, dtype=torch.bfloat16, revision=args.model_revision, device_map="auto"),
            path, base_model=args.model, model_revision=resolved_revision or args.model_revision)
        print("strict adapter reload", info["lora_coverage"]["matched_counts"], flush=True)
        del reloaded
    model = model.unload()  # back to the plain base model for sampling
    torch.cuda.empty_cache()
    model.eval()

    ds = load_dataset("openai/gsm8k", "main", split="train", revision=args.dataset_revision)
    ds = ds.add_column("dataset_index", list(range(len(ds)))).shuffle(seed=args.seed).select(range(args.n_prompts))
    prompts = [PROMPT_TMPL.format(question=q) for q in ds["question"]]
    golds = [gold_answer(a) for a in ds["answer"]]
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.inference_mode():
        gen = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                             max_new_tokens=args.max_new, num_return_sequences=args.G,
                             pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    new = gen[:, enc["input_ids"].shape[1]:]
    rows = []
    for k in range(new.shape[0]):
        ids = new[k].tolist()
        has_eos = tokenizer.eos_token_id in ids
        n_real = ids.index(tokenizer.eos_token_id) if has_eos else len([t for t in ids if t != tokenizer.pad_token_id])
        text = tokenizer.decode(new[k], skip_special_tokens=True)
        p = k // args.G
        parsed = extract_answer(text)
        rows.append({"prompt_index": p, "dataset_index": int(ds[p]["dataset_index"]), "sample_index": k % args.G,
                     "gold": golds[p], "completion": text, "parsed": parsed, "correct": parsed == golds[p],
                     "n_tokens": n_real, "has_eos": has_eos, "hit_cap": (not has_eos) and n_real >= args.max_new})
    n = len(rows)
    summary = {
        "n_completions": n,
        "parse_rate": sum(r["parsed"] is not None for r in rows) / n,
        "n_parsed": sum(r["parsed"] is not None for r in rows),
        "n_correct": sum(r["correct"] for r in rows),
        "n_hit_cap": sum(r["hit_cap"] for r in rows),
        "n_has_eos": sum(r["has_eos"] for r in rows),
        "mean_tokens": sum(r["n_tokens"] for r in rows) / n,
        "max_tokens": max(r["n_tokens"] for r in rows),
    }
    print("SUMMARY", json.dumps(summary), flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    out.write_text(json.dumps({
        "arm": "base", "seed": args.seed, "step": 0, "layer": None, "snippet_set": "gsm8k_train_preflight",
        "snippet_sha": None, "judge_model": None, "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit, "model": args.model, "resolved_model_revision": resolved_revision,
        "source_config": source, "loaded_architecture": type(model).__name__,
        "prompt_template": PROMPT_TMPL, "decoding": {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0,
        "max_new_tokens": args.max_new, "G": args.G, "padding_side": tokenizer.padding_side,
        "add_special_tokens": True, "chat_template_applied": False},
        "tokenizer": {"bos": tokenizer.bos_token_id, "eos": tokenizer.eos_token_id, "pad": tokenizer.pad_token_id},
        "lora_coverage_counts": counts, "summary": summary, "prompts": prompts, "rows": rows,
    }, indent=1, ensure_ascii=False) + "\n")
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
