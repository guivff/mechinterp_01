"""GRPO with LoRA on GSM8K (arms A and B).

  python -m grpo.train_grpo --arm A --model Qwen/Qwen3.5-4B-Base --out runs/A_s0 --seed 0
  python -m grpo.train_grpo --arm B --model Qwen/Qwen3.5-4B-Base --out runs/B_s0 --seed 0   # shuffled rewards

Arm B: rewards are permuted *within each group of G completions of the same prompt* before
the trainer standardizes advantages. Same prompts, same optimizer, same sampling; no reward
information. This is the "generic optimization" control.

IMPORTANT for agents: TRL's GRPOTrainer API has changed several times. Check the installed
version (`pip show trl`) and adapt argument names; the logic (reward function signature,
per-group ordering of completions) is what must be preserved. Verify the per-group ordering
assumption with the assert in `make_reward_fn` on the first batch.

If the applicant's own GRPO pipeline is available, prefer adapting it and keep this file as
the reference implementation of the shuffled-reward arm.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import torch

PROMPT_TMPL = "{question}\nAnswer:"
NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
DEFAULT_MODEL = "Qwen/Qwen3.5-4B-Base"
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.0
# Qwen3.5 interleaves ordinary full-attention blocks with Gated DeltaNet
# blocks.  PROJECT_SPEC says all attention and MLP projections, so both
# families must be targeted.
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
)


def preferred_training_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_text_causal_stack(
    model_name: str,
    *,
    padding_side: str,
    device_map=None,
    local_files_only: bool = False,
):
    """Load Qwen3.5's text causal class explicitly, plus its tokenizer.

    The official Qwen3.5-4B-Base checkpoint declares the composite
    ``Qwen3_5ForConditionalGeneration`` architecture.  TRL follows that
    declaration when handed a model-name string, producing LoRA keys under
    ``model.language_model.layers``.  Readout intentionally uses
    ``AutoModelForCausalLM`` and expects ``model.layers``.  Preloading here
    keeps training, rejection sampling, and readout on one exact module tree.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, local_files_only=local_files_only
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither a padding token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    dtype = preferred_training_dtype()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
    )
    if hasattr(getattr(model, "model", None), "language_model"):
        raise TypeError(
            "expected the text-only AutoModelForCausalLM module tree, got a composite model"
        )
    return model, tokenizer


def extract_answer(text: str) -> str | None:
    if "####" in text:
        text = text.split("####")[-1]
    nums = NUM_RE.findall(text)
    return nums[-1].replace(",", "") if nums else None


def gold_answer(gsm_answer: str) -> str:
    return gsm_answer.split("####")[-1].strip().replace(",", "")


def make_reward_fn(shuffle: bool, num_generations: int, seed: int):
    rng = random.Random(seed)
    checked = {"once": False}

    def reward_fn(prompts, completions, gold, **kwargs):
        # completions may be strings or [{"role","content"}] chats depending on TRL version
        texts = [c if isinstance(c, str) else c[0]["content"] for c in completions]
        rewards = [1.0 if extract_answer(t) == g else 0.0 for t, g in zip(texts, gold)]
        if shuffle:
            n = len(rewards)
            assert n % num_generations == 0, (n, num_generations)
            if not checked["once"]:
                # verify per-group ordering: consecutive completions share a prompt
                for i in range(0, n, num_generations):
                    assert all(prompts[j] == prompts[i] for j in range(i, i + num_generations)), \
                        "Completions are not grouped by prompt; fix the shuffle indexing."
                checked["once"] = True
            for i in range(0, n, num_generations):
                grp = rewards[i : i + num_generations]
                rng.shuffle(grp)
                rewards[i : i + num_generations] = grp
        return rewards

    return reward_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "B"], required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--n-prompts", type=int, default=2000)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--batch-prompts", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.0, help="KL to reference; 0 = none")
    ap.add_argument("--max-completion", type=int, default=512)
    ap.add_argument("--lora-r", type=int, default=LORA_R)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--use-vllm", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny run for CPU tests")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    from datasets import load_dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=args.seed)
    ds = ds.select(range(min(args.n_prompts, len(ds))))
    ds = ds.map(lambda r: {"prompt": PROMPT_TMPL.format(question=r["question"]), "gold": gold_answer(r["answer"])})

    if args.smoke:
        args.steps, args.G, args.batch_prompts, args.max_completion = 2, 2, 2, 32

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    cfg = GRPOConfig(
        output_dir=args.out,
        seed=args.seed,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_prompts * args.G,
        num_generations=args.G,
        max_completion_length=args.max_completion,
        max_steps=args.steps,
        beta=args.beta,
        save_steps=args.save_every,
        logging_steps=1,
        bf16=use_bf16,
        fp16=torch.cuda.is_available() and not use_bf16,
        use_vllm=args.use_vllm,
        report_to=[],
        temperature=1.0,
    )
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
        padding_side="left",
        # Trainer/Accelerate owns device placement. In particular, device_map
        # "auto" is incompatible with distributed training.
        device_map=None,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=make_reward_fn(shuffle=(args.arm == "B"), num_generations=args.G, seed=args.seed),
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
                "arm": args.arm,
                "model_class": type(model).__name__,
                "model_loader": "AutoModelForCausalLM",
                "final_global_step": int(trainer.state.global_step),
                "resolved_model_revision": getattr(model.config, "_commit_hash", None),
                "model_dtype": str(next(model.parameters()).dtype).replace("torch.", ""),
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
