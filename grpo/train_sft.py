"""SFT arms with LoRA.

Arm C  (rejection-sampling SFT on arm A's correct samples):
  1) python grpo/train_sft.py sample --policy runs/A_s0/final --model Qwen/Qwen3.5-4B-Base --out data/C_samples.jsonl --G 8
  2) python grpo/train_sft.py train  --arm C --data data/C_samples.jsonl --model Qwen/Qwen3.5-4B-Base --out runs/C_s0 --max-steps 150

Arm D  (narrow-domain positive control, e.g. cooking corpus at data/cooking.jsonl, lines {"text": ...}):
  python grpo/train_sft.py train --arm D --data data/cooking.jsonl --model Qwen/Qwen3.5-4B-Base --out runs/D_s0

Arm C' (stretch; self-distill on unfiltered base samples): sample with --policy = base and --keep all.

Token/step matching: `--max-tokens` caps the exact number of selected, truncated tokens.
Arm C additionally requires an explicit `--max-steps` match target. Both the selected-token
count and the trainer's observed input-token count are recorded in run_meta.json.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

if __package__:
    from .model_utils import (
        LORA_TARGET_MODULES,
        load_peft_adapter_strict,
        load_plain_tokenizer,
        load_text_causal_lm,
        lora_coverage,
        source_config_info,
    )
    from .train_grpo import (
        PROMPT_TMPL,
        extract_answer,
        gold_answer,
    )
else:
    from model_utils import (
        LORA_TARGET_MODULES,
        load_peft_adapter_strict,
        load_plain_tokenizer,
        load_text_causal_lm,
        lora_coverage,
        source_config_info,
    )
    from train_grpo import PROMPT_TMPL, extract_answer, gold_answer


def _versions() -> dict[str, str]:
    return {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "trl", "peft", "datasets", "accelerate")
    }


def _json_hash(rows) -> str:
    payload = "\n".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in rows
    )
    return hashlib.sha256((payload + "\n").encode()).hexdigest()


def cmd_sample(args):
    from datasets import load_dataset
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tok = load_plain_tokenizer(
        args.model, revision=args.model_revision, padding_side="left"
    )
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = load_text_causal_lm(
        args.model,
        dtype=dtype,
        revision=args.model_revision,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.config.pad_token_id = tok.pad_token_id
    source_info = source_config_info(args.model, revision=args.model_revision)
    adapter_info = None
    if args.policy != "base":
        model, adapter_info = load_peft_adapter_strict(
            model,
            args.policy,
            base_model=args.model,
            model_revision=source_info["source_commit_hash"] or args.model_revision,
            adapter_revision=args.adapter_revision,
        )
        model = model.merge_and_unload()
    model.eval()
    ds = load_dataset(
        "openai/gsm8k",
        "main",
        split="train",
        revision=args.dataset_revision,
    )
    ds = ds.add_column("dataset_index", list(range(len(ds)))).shuffle(seed=args.seed)
    if args.n_prompts > len(ds):
        raise ValueError(f"requested {args.n_prompts} prompts from a {len(ds)}-row split")
    ds = ds.select(range(args.n_prompts))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    kept = total = 0
    with out.open("w") as f, torch.no_grad():
        for i in range(0, len(ds), args.batch):
            rows = ds[i : i + args.batch]
            prompts = [PROMPT_TMPL.format(question=q) for q in rows["question"]]
            golds = [gold_answer(a) for a in rows["answer"]]
            enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
            gen = model.generate(**enc, do_sample=True, temperature=1.0, max_new_tokens=args.max_new, num_return_sequences=args.G,
                                 pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            comps = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            if len(comps) != len(prompts) * args.G:
                raise RuntimeError(
                    f"expected {len(prompts) * args.G} prompt-major samples, "
                    f"got {len(comps)}"
                )
            for k, c in enumerate(comps):
                prompt_offset = k // args.G
                p, g = prompts[prompt_offset], golds[prompt_offset]
                ok = extract_answer(c) == g
                total += 1
                if args.keep == "all" or ok:
                    kept += 1
                    f.write(
                        json.dumps(
                            {
                                "dataset_index": int(rows["dataset_index"][prompt_offset]),
                                "question": rows["question"][prompt_offset],
                                "prompt": p,
                                "gold": g,
                                "completion": c,
                                "sample_index": k % args.G,
                                "correct": ok,
                                "text": p + c,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
    selected_rows = [
        {
            "dataset_index": int(ds[i]["dataset_index"]),
            "question": ds[i]["question"],
            "answer": ds[i]["answer"],
        }
        for i in range(len(ds))
    ]
    meta = {
        **vars(args),
        **source_info,
        "loaded_architecture": type(model).__name__,
        "loaded_model_type": getattr(model.config, "model_type", None),
        "dataset": "openai/gsm8k",
        "dataset_config": "main",
        "dataset_split": "train",
        "dataset_fingerprint": getattr(ds, "_fingerprint", None),
        "selected_prompt_sha256": _json_hash(selected_rows),
        "output_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "kept": kept,
        "total": total,
        "plain_prompt": True,
        "chat_template_applied": False,
        "adapter": adapter_info,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "versions": _versions(),
    }
    out.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=1, default=str) + "\n"
    )
    print(f"kept {kept}/{total} -> {out}")


def cmd_train(args):
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.arm in ("C", "Cp") and args.max_steps is None:
        raise ValueError(
            f"arm {args.arm} must be matched explicitly; pass --max-steps "
            "(150 matches the preregistered A optimizer-step count)"
        )
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")

    rows = [json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()]
    random.shuffle(rows)
    tok = load_plain_tokenizer(
        args.model, revision=args.model_revision, padding_side="right"
    )
    if tok.eos_token is None:
        raise ValueError("SFT requires a tokenizer EOS token")
    selected_tokens, texts = 0, []
    for r in rows:
        text = r["text"]
        # Mirror TRL 1.12 SFTTrainer._prepare_dataset: append EOS to plain
        # language-modeling text before tokenization, then keep the first
        # max_len tokens.
        trainer_text = text if text.endswith(tok.eos_token) else text + tok.eos_token
        n_tokens = len(
            tok(
                trainer_text,
                add_special_tokens=True,
                truncation=True,
                max_length=args.max_len,
            )["input_ids"]
        )
        if selected_tokens + n_tokens > args.max_tokens:
            break
        texts.append(text)
        selected_tokens += n_tokens
    if not texts:
        raise ValueError(
            "no training text fits under --max-tokens after --max-len truncation"
        )
    ds = Dataset.from_dict({"text": texts})
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = load_text_causal_lm(
        args.model, dtype=dtype, revision=args.model_revision, device_map=None
    )
    model.config.pad_token_id = tok.pad_token_id
    source_info = source_config_info(args.model, revision=args.model_revision)
    cfg = SFTConfig(
        output_dir=args.out,
        seed=args.seed,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        max_length=args.max_len,
        max_steps=args.max_steps if args.max_steps is not None else -1,
        dataset_text_field="text",
        packing=False,
        completion_only_loss=False,
        logging_steps=5,
        save_strategy="steps",
        save_steps=args.save_every,
        bf16=torch.cuda.is_available(),
        report_to=[],
    )
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=2 * args.lora_r,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
        revision=source_info["source_commit_hash"] or args.model_revision,
    )
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        peft_config=lora,
        processing_class=tok,
    )
    coverage = lora_coverage(trainer.model)
    train_output = trainer.train()
    trainer.save_model(str(Path(args.out) / "final"))
    observed_token_counts = [
        row["num_tokens"]
        for row in trainer.state.log_history
        if isinstance(row.get("num_tokens"), (int, float))
    ]
    metadata = {
        **vars(args),
        **source_info,
        "loaded_architecture": type(model).__name__,
        "loaded_model_type": getattr(model.config, "model_type", None),
        "plain_text_training": True,
        "chat_template_applied": False,
        "n_texts": len(texts),
        "selected_tokens_exact": selected_tokens,
        "selected_text_sha256": _json_hash([{"text": text} for text in texts]),
        "matching_basis": "optimizer_steps" if args.arm in ("C", "Cp") else None,
        "target_optimizer_steps": args.max_steps,
        "global_step": trainer.state.global_step,
        "trainer_num_input_tokens": max(observed_token_counts, default=None),
        "lora_coverage": coverage,
        "train_metrics": train_output.metrics,
        "versions": _versions(),
    }
    Path(args.out, "run_meta.json").write_text(
        json.dumps(metadata, indent=1, default=str) + "\n"
    )


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--policy", required=True, help="path to LoRA adapter dir, or 'base'")
    s.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
    s.add_argument("--model-revision", default=None)
    s.add_argument("--dataset-revision", default=None)
    s.add_argument("--adapter-revision", default=None)
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
    t.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
    t.add_argument("--model-revision", default=None)
    t.add_argument("--out", required=True)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--epochs", type=float, default=1.0)
    t.add_argument("--max-steps", type=int, default=None)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--max-len", type=int, default=768)
    t.add_argument("--max-tokens", type=int, default=2_000_000)
    t.add_argument("--lora-r", type=int, default=32)
    t.add_argument("--save-every", type=int, default=25)
    args = ap.parse_args()
    {"sample": cmd_sample, "train": cmd_train}[args.cmd](args)


if __name__ == "__main__":
    main()
