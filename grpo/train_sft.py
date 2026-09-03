"""SFT arms with LoRA.

Arm C  (rejection-sampling SFT on arm A's correct samples):
  1) python grpo/train_sft.py sample --policy runs/A_s0/final --model Qwen/Qwen3.5-4B --out data/C_samples.jsonl --G 8
  2) python grpo/train_sft.py train  --arm C --data data/C_samples.jsonl --model Qwen/Qwen3.5-4B --out runs/C_s0

Arm D  (narrow-domain positive control, e.g. cooking corpus at data/cooking.jsonl, lines {"text": ...}):
  python grpo/train_sft.py train --arm D --data data/cooking.jsonl --model Qwen/Qwen3.5-4B --out runs/D_s0

Arm C' (stretch; self-distill on unfiltered base samples): sample with --policy = base and --keep all.

Token budget matching: `--max-tokens` caps the number of training tokens so C and D see a
comparable amount of data; record the actual count in run_meta.json.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from grpo.train_grpo import PROMPT_TMPL, extract_answer, gold_answer


def cmd_sample(args):
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                                                 device_map="auto" if torch.cuda.is_available() else None)
    if args.policy != "base":
        model = PeftModel.from_pretrained(model, args.policy).merge_and_unload()
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
                    f.write(json.dumps({"text": p + c, "correct": ok}) + "\n")
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
    cfg = SFTConfig(output_dir=args.out, seed=args.seed, learning_rate=args.lr, num_train_epochs=args.epochs,
                    per_device_train_batch_size=args.batch, max_length=args.max_len, logging_steps=5,
                    save_steps=args.save_every, bf16=torch.cuda.is_available(), report_to=[])
    lora = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    trainer = SFTTrainer(model=args.model, args=cfg, train_dataset=ds, peft_config=lora)
    trainer.train()
    trainer.save_model(str(Path(args.out) / "final"))
    Path(args.out, "run_meta.json").write_text(json.dumps({**vars(args), "n_texts": len(texts), "approx_tokens": budget}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--policy", required=True, help="path to LoRA adapter dir, or 'base'")
    s.add_argument("--model", default="Qwen/Qwen3.5-4B")
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
    t.add_argument("--model", default="Qwen/Qwen3.5-4B")
    t.add_argument("--out", required=True)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--epochs", type=float, default=1.0)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--max-len", type=int, default=768)
    t.add_argument("--max-tokens", type=int, default=2_000_000)
    t.add_argument("--lora-r", type=int, default=32)
    t.add_argument("--save-every", type=int, default=50)
    args = ap.parse_args()
    {"sample": cmd_sample, "train": cmd_train}[args.cmd](args)


if __name__ == "__main__":
    main()
