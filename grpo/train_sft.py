"""SFT arms with LoRA.

Arm C  (rejection-sampling SFT on arm A's correct samples):
  1) python -m grpo.train_sft sample --policy runs/A_s0/final --model Qwen/Qwen3.5-4B-Base --out data/C_samples.jsonl --G 8
  2) python -m grpo.train_sft train  --arm C --data data/C_samples.jsonl --model Qwen/Qwen3.5-4B-Base --out runs/C_s0

Arm D  (narrow-domain positive control, e.g. cooking corpus at data/cooking.jsonl, lines {"text": ...}):
  python -m grpo.train_sft train --arm D --data data/cooking.jsonl --model Qwen/Qwen3.5-4B-Base --out runs/D_s0

Arm C' (stretch; self-distill on unfiltered base samples): sample with --policy = base and --keep all.

Token budget matching: `--max-tokens` caps the number of training tokens so C and D see a
comparable amount of data; record the actual count in run_meta.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from grpo.train_grpo import (
    DEFAULT_MODEL,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    PROMPT_TMPL,
    extract_answer,
    gold_answer,
    load_text_causal_stack,
)


def cmd_sample(args):
    from datasets import load_dataset
    from readout.run_readouts import load_adapter_strict

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model, tok = load_text_causal_stack(
        args.model,
        padding_side="left",
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if args.policy != "base":
        # Keep PEFT active: merging a weak LoRA into BF16 can measurably change
        # the sampling policy that defines arm C.
        model = load_adapter_strict(model, args.policy)
    model.eval()
    ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=args.seed).select(range(args.n_prompts))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    kept = total = 0
    with out.open("w") as f, torch.no_grad():
        for i in range(0, len(ds), args.batch):
            rows = ds[i : i + args.batch]
            prompts = [PROMPT_TMPL.format(question=q) for q in rows["question"]]
            golds = [gold_answer(a) for a in rows["answer"]]
            enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
            gen = model.generate(**enc, do_sample=True, temperature=1.0, max_new_tokens=args.max_new, num_return_sequences=args.G,
                                 pad_token_id=tok.eos_token_id)
            comps = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for k, c in enumerate(comps):
                p, g = prompts[k // args.G], golds[k // args.G]
                ok = extract_answer(c) == g
                total += 1
                if args.keep == "all" or ok:
                    kept += 1
                    f.write(
                        json.dumps(
                            {
                                "text": p + c,
                                "correct": ok,
                                "gold": g,
                                "prompt_index": i + (k // args.G),
                                "completion_index": k % args.G,
                                "seed": args.seed,
                                "policy": args.policy,
                                "base_model": args.model,
                                "temperature": 1.0,
                                "max_new_tokens": args.max_new,
                                "num_return_sequences": args.G,
                            }
                        )
                        + "\n"
                    )
    output_sha = hashlib.sha256(out.read_bytes()).hexdigest()
    policy_receipt = {"reference": args.policy}
    if args.policy != "base":
        policy_path = Path(args.policy)
        for name in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"):
            candidate = policy_path / name
            if candidate.is_file():
                policy_receipt[f"{name}_sha256"] = hashlib.sha256(candidate.read_bytes()).hexdigest()
    manifest = {
        "artifact_type": "arm_C_rejection_samples",
        "output": str(out),
        "output_sha256": output_sha,
        "seed": args.seed,
        "base_model": args.model,
        "resolved_model_revision": getattr(model.config, "_commit_hash", None),
        "policy": policy_receipt,
        "G": args.G,
        "n_prompts": args.n_prompts,
        "temperature": 1.0,
        "max_new_tokens": args.max_new,
        "keep": args.keep,
        "kept": kept,
        "total": total,
    }
    Path(f"{out}.manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"kept {kept}/{total} -> {out}")


def cmd_train(args):
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rows = [json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()]
    random.shuffle(rows)
    # crude token cap (4 chars/token) — the exact count is logged by the trainer
    budget, texts = 0, []
    for r in rows:
        texts.append(r["text"])
        budget += len(r["text"]) // 4
        if budget >= args.max_tokens:
            break
    ds = Dataset.from_dict({"text": texts})
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    cfg = SFTConfig(output_dir=args.out, seed=args.seed, learning_rate=args.lr, num_train_epochs=args.epochs,
                    per_device_train_batch_size=args.batch, max_length=args.max_len, logging_steps=5,
                    save_steps=args.save_every, bf16=use_bf16,
                    fp16=torch.cuda.is_available() and not use_bf16, report_to=[])
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=2 * args.lora_r,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(LORA_TARGET_MODULES),
    )
    model, tokenizer = load_text_causal_stack(
        args.model,
        padding_side="right",
        device_map=None,
    )
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        peft_config=lora,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(Path(args.out) / "final"))
    Path(args.out, "run_meta.json").write_text(
        json.dumps(
            {
                **vars(args),
                "n_texts": len(texts),
                "approx_tokens": budget,
                "model_class": type(model).__name__,
                "model_loader": "AutoModelForCausalLM",
                "final_global_step": int(trainer.state.global_step),
                "resolved_model_revision": getattr(model.config, "_commit_hash", None),
                "model_dtype": str(next(model.parameters()).dtype).replace("torch.", ""),
            },
            indent=1,
        )
    )


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--policy", required=True, help="path to LoRA adapter dir, or 'base'")
    s.add_argument("--model", default=DEFAULT_MODEL)
    s.add_argument("--out", required=True)
    s.add_argument("--G", type=int, default=8)
    s.add_argument("--n-prompts", type=int, default=2000)
    s.add_argument("--batch", type=int, default=16)
    s.add_argument("--max-new", type=int, default=512)
    s.add_argument("--keep", choices=["correct", "all"], default="correct")
    s.add_argument("--seed", type=int, default=0)
    t = sub.add_parser("train")
    t.add_argument("--arm", choices=["C", "Cp", "D"], required=True)
    t.add_argument("--data", required=True)
    t.add_argument("--model", default=DEFAULT_MODEL)
    t.add_argument("--out", required=True)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--epochs", type=float, default=1.0)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--max-len", type=int, default=768)
    t.add_argument("--max-tokens", type=int, default=2_000_000)
    t.add_argument("--lora-r", type=int, default=LORA_R)
    t.add_argument("--save-every", type=int, default=50)
    args = ap.parse_args()
    {"sample": cmd_sample, "train": cmd_train}[args.cmd](args)


if __name__ == "__main__":
    main()
