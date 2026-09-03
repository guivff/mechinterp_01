"""GRPO with LoRA on GSM8K (arms A and B).

  python grpo/train_grpo.py --arm A --model Qwen/Qwen3.5-4B-Base --out runs/A_s0 --seed 0
  python grpo/train_grpo.py --arm B --model Qwen/Qwen3.5-4B-Base --out runs/B_s0 --seed 0   # shuffled rewards

Arm B: rewards are permuted *within each group of G completions of the same prompt* before
the trainer standardizes advantages. Same prompts, same optimizer, same sampling; zero
expected completion-level reward association over the permutation. This is the "generic
optimization" control.

IMPORTANT for agents: TRL's GRPOTrainer API has changed several times. Check the installed
version (`pip show trl`) and adapt argument names; the logic (reward function signature,
per-group ordering of completions) is what must be preserved. Verify the per-group ordering
assumption with the assert in `make_reward_fn` on the first batch.

If the applicant's own GRPO pipeline is available, prefer adapting it and keep this file as
the reference implementation of the shuffled-reward arm.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

if __package__:
    from .model_utils import (
        LORA_TARGET_MODULES,
        load_plain_tokenizer,
        load_text_causal_lm,
        lora_coverage,
        source_config_info,
    )
else:
    from model_utils import (
        LORA_TARGET_MODULES,
        load_plain_tokenizer,
        load_text_causal_lm,
        lora_coverage,
        source_config_info,
    )

PROMPT_TMPL = "{question}\nAnswer:"
# A decimal point belongs to the answer only when digits follow it. Without
# this grouping, ordinary sentence punctuation ("The answer is 10.") becomes
# the string "10." and fails exact match against GSM8K gold "10".
NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

def extract_answer(text: str) -> str | None:
    if "####" in text:
        text = text.split("####")[-1]
    nums = NUM_RE.findall(text)
    return nums[-1].replace(",", "") if nums else None


def gold_answer(gsm_answer: str) -> str:
    return gsm_answer.split("####")[-1].strip().replace(",", "")


def _selected_rows_hash(dataset) -> str:
    digest = hashlib.sha256()
    for row in dataset:
        receipt = {
            "answer": row["answer"],
            "dataset_index": int(row["dataset_index"]),
            "question": row["question"],
        }
        digest.update(
            (json.dumps(receipt, ensure_ascii=False, sort_keys=True) + "\n").encode()
        )
    return digest.hexdigest()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def require_single_rank() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 1:
        raise RuntimeError(
            "train_grpo.py is preregistered for one process/GPU per arm; "
            f"WORLD_SIZE={world_size} would silently change the 32-prompt "
            "optimizer batch. Launch independent single-rank processes instead."
        )


def completion_truncated(
    completion_ids, completion_text: str, tokenizer, max_completion: int
) -> bool:
    """True when a completion reached the cap without emitting EOS.

    TRL 1.12 passes ``completion_ids`` (one int list per completion, cut after
    the first EOS; see ``GRPOTrainer._generate_single_turn`` and the
    ``completions/clipped_ratio`` metric, which uses the same
    ``ids[-1] not in {eos, pad}`` rule).  When ids are unavailable the text is
    re-tokenized and the cap is inferred from its length, which is a weaker
    proxy because the decoded text has no EOS marker.
    """
    if completion_ids is not None:
        ids = list(completion_ids)
        if not ids:
            return False
        return ids[-1] not in {tokenizer.eos_token_id, tokenizer.pad_token_id}
    if tokenizer is None or max_completion is None:
        # Unit-test path without a tokenizer: no truncation information exists.
        return False
    n_tokens = len(tokenizer(completion_text, add_special_tokens=False)["input_ids"])
    return n_tokens >= max_completion


def make_reward_fn(
    shuffle: bool,
    num_generations: int,
    seed: int,
    tokenizer=None,
    max_completion: int | None = None,
):
    checked = {"once": False, "ids_seen": None}

    def reward_fn(prompts, completions, gold, completion_ids=None, **kwargs):
        # completions may be strings or [{"role","content"}] chats depending on TRL version
        texts = [c if isinstance(c, str) else c[0]["content"] for c in completions]
        assert len(prompts) == len(texts) == len(gold), (
            len(prompts),
            len(texts),
            len(gold),
        )
        exact = [1.0 if extract_answer(t) == g else 0.0 for t, g in zip(texts, gold)]
        # PREREG: reward 0 for any completion that reaches the cap without EOS.
        if completion_ids is not None:
            assert len(completion_ids) == len(texts), (len(completion_ids), len(texts))
            id_source = "completion_ids"
        elif tokenizer is not None and max_completion is not None:
            id_source = "retokenized_text"
        else:
            id_source = "unavailable"
        if checked["ids_seen"] is None:
            checked["ids_seen"] = id_source
            print(f"[reward_fn] truncation detection source: {id_source}", flush=True)
        truncated = [
            completion_truncated(
                completion_ids[i] if completion_ids is not None else None,
                texts[i],
                tokenizer,
                max_completion,
            )
            for i in range(len(texts))
        ]
        rewards = [0.0 if trunc else r for r, trunc in zip(exact, truncated)]
        n_trunc = sum(truncated)
        n_trunc_would_be_correct = sum(1 for r, t in zip(exact, truncated) if t and r > 0)
        log_metric = kwargs.get("log_metric")
        if callable(log_metric):
            log_metric("reward/truncation_rate", n_trunc / max(len(texts), 1))
            log_metric("reward/truncated_but_parsed_correct", n_trunc_would_be_correct / max(len(texts), 1))
            log_metric("reward/exact_match_pre_truncation", sum(exact) / max(len(texts), 1))
        trainer_state = kwargs.get("trainer_state")
        step_for_log = getattr(trainer_state, "global_step", None)
        print(
            f"[reward_fn] step={step_for_log} n={len(texts)} exact_match={sum(exact) / max(len(texts), 1):.4f} "
            f"truncated={n_trunc} ({n_trunc / max(len(texts), 1):.4f}) "
            f"truncated_but_parsed_correct={n_trunc_would_be_correct} "
            f"reward_after_truncation_rule={sum(rewards) / max(len(texts), 1):.4f}",
            flush=True,
        )
        if shuffle:
            n = len(rewards)
            assert n % num_generations == 0, (n, num_generations)
            if not checked["once"]:
                # verify per-group ordering: consecutive completions share a prompt
                for i in range(0, n, num_generations):
                    assert all(prompts[j] == prompts[i] for j in range(i, i + num_generations)), \
                        "Completions are not grouped by prompt; fix the shuffle indexing."
                    assert all(gold[j] == gold[i] for j in range(i, i + num_generations)), \
                        "Gold answers are not grouped with their prompts; fix the dataset expansion."
                checked["once"] = True
            trainer = getattr(reward_fn, "trainer", None)
            state = kwargs.get("trainer_state") or getattr(trainer, "state", None)
            global_step = int(getattr(state, "global_step", 0))
            for group_index, i in enumerate(range(0, n, num_generations)):
                grp = rewards[i : i + num_generations]
                # Key the independent uniform permutation by optimizer step so
                # checkpoint resume reproduces an uninterrupted arm-B run.
                # The key contains neither completion text nor reward values.
                group_seed = hashlib.sha256(
                    f"{seed}:{global_step}:{group_index}".encode()
                ).digest()
                random.Random(group_seed).shuffle(grp)
                rewards[i : i + num_generations] = grp
        return rewards

    reward_fn.trainer = None
    reward_fn.checked = checked
    return reward_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "B"], required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
    ap.add_argument("--model-revision", default=None)
    ap.add_argument("--dataset-revision", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--n-prompts", type=int, default=2000)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--batch-prompts", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.0, help="KL to reference; 0 = none")
    ap.add_argument("--max-completion", type=int, default=512)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--resume-from-checkpoint", default=None)
    ap.add_argument(
        "--loss-type",
        default="dapo",
        choices=["grpo", "bnpo", "dr_grpo", "dapo"],
        help="TRL token-loss aggregation; dapo is the installed TRL 1.12 default",
    )
    ap.add_argument("--use-vllm", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny run for CPU tests")
    args = ap.parse_args()

    require_single_rank()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    from datasets import Dataset, load_dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    if args.smoke:
        args.steps, args.G, args.batch_prompts, args.max_completion = 2, 2, 2, 32
        args.n_prompts = min(args.n_prompts, 2 * args.batch_prompts)
        # Keep the CPU/API smoke test independent of Hub availability. These
        # rows exercise the same prompt and exact-match reward data path; real
        # runs below always load GSM8K.
        ds = Dataset.from_dict(
            {
                "dataset_index": [0, 1, 2, 3],
                "question": [
                    "What is 1 + 1?",
                    "What is 2 + 3?",
                    "What is 7 - 4?",
                    "What is 3 times 3?",
                ],
                "answer": ["#### 2", "#### 5", "#### 3", "#### 9"],
            }
        )
    else:
        ds = load_dataset(
            "openai/gsm8k",
            "main",
            split="train",
            revision=args.dataset_revision,
        )
        ds = ds.add_column("dataset_index", list(range(len(ds)))).shuffle(seed=args.seed)
        ds = ds.select(range(min(args.n_prompts, len(ds))))
    ds = ds.map(lambda r: {"prompt": PROMPT_TMPL.format(question=r["question"]), "gold": gold_answer(r["answer"])})
    dataset_fingerprint = getattr(ds, "_fingerprint", None)
    selected_prompt_sha256 = _selected_rows_hash(ds)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tokenizer = load_plain_tokenizer(
        args.model, revision=args.model_revision, padding_side="left"
    )
    model = load_text_causal_lm(
        args.model, dtype=dtype, revision=args.model_revision, device_map=None
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    source_info = source_config_info(args.model, revision=args.model_revision)
    loaded_model_type = getattr(model.config, "model_type", None)
    if args.use_vllm and source_info["source_model_type"] != loaded_model_type:
        raise RuntimeError(
            "--use-vllm is unsafe for this outer multimodal checkpoint: TRL would "
            "load vLLM from the outer architecture while training the extracted "
            "text-only causal LM, so weight names may not match. Materialize a "
            "text-only checkpoint first or leave --use-vllm off."
        )

    cfg = GRPOConfig(
        output_dir=args.out,
        seed=args.seed,
        learning_rate=args.lr,
        # Complete G-sized prompt groups enter reward calculation together.
        # TRL shuffles prepared rows only after assigning their advantages;
        # the accumulated optimizer step still covers batch_prompts complete
        # groups without retaining all 256 backward activations at once.
        per_device_train_batch_size=args.G,
        gradient_accumulation_steps=args.batch_prompts,
        generation_batch_size=args.batch_prompts * args.G,
        num_generations=args.G,
        max_completion_length=args.max_completion,
        max_steps=args.steps,
        beta=args.beta,
        save_strategy="steps",
        save_steps=args.save_every,
        logging_steps=1,
        bf16=torch.cuda.is_available(),
        use_vllm=args.use_vllm,
        report_to=[],
        temperature=1.0,
        scale_rewards="group",
        loss_type=args.loss_type,
        # PREREG: a completion that reaches the cap without EOS receives reward
        # 0 (implemented in the reward function from TRL's completion_ids).
        # Its tokens still enter the policy-gradient loss with that zero
        # reward; loss masking of truncated completions is deliberately off.
        mask_truncated_completions=False,
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

    reward_fn = make_reward_fn(
        shuffle=(args.arm == "B"),
        num_generations=args.G,
        seed=args.seed,
        tokenizer=tokenizer,
        max_completion=args.max_completion,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=cfg,
        train_dataset=ds,
        peft_config=lora,
        processing_class=tokenizer,
    )
    reward_fn.trainer = trainer
    coverage = lora_coverage(trainer.model)
    train_output = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(Path(args.out) / "final"))
    metadata = {
        **vars(args),
        "arm": args.arm,
        **source_info,
        "loaded_architecture": type(model).__name__,
        "loaded_model_type": loaded_model_type,
        "plain_prompt": True,
        "chat_template_applied": False,
        "reward_zero_on_truncation": True,
        "truncation_detection_source": reward_fn.checked["ids_seen"],
        "dataset": "inline_smoke" if args.smoke else "openai/gsm8k",
        "dataset_config": None if args.smoke else "main",
        "dataset_split": None if args.smoke else "train",
        "dataset_fingerprint": dataset_fingerprint,
        "selected_prompt_sha256": selected_prompt_sha256,
        "effective_prompt_batch_size": args.batch_prompts,
        "effective_completion_batch_size": args.batch_prompts * args.G,
        "lora_coverage": coverage,
        "global_step": trainer.state.global_step,
        "train_metrics": train_output.metrics,
        "git_commit": _git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "trl", "peft", "datasets", "accelerate")
        },
    }
    Path(args.out, "run_meta.json").write_text(
        json.dumps(metadata, indent=1, default=str) + "\n"
    )


if __name__ == "__main__":
    main()
