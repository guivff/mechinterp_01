"""Evaluate exact-match accuracy on a fixed held-out GSM8K subset.

The evaluation deliberately uses the same plain prompt and final-number verifier as
training.  No chat template is applied.

Examples:

  python grpo/eval_acc.py --arm base --model Qwen/Qwen3.5-4B-Base --seed 0
  python grpo/eval_acc.py --arm A --model Qwen/Qwen3.5-4B-Base \
      --adapter runs/A_s0/final --step 150 --seed 0

The default output is ``results/acc_{arm}_s{seed}.json``.  It contains every
completion and parsed answer so the headline accuracy can be independently
recomputed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    from grpo.model_utils import (
        load_peft_adapter_strict,
        load_plain_tokenizer,
        load_text_causal_lm,
        source_config_info,
    )
    from grpo.train_grpo import NUM_RE, PROMPT_TMPL, extract_answer, gold_answer
except ModuleNotFoundError as exc:
    # ``python grpo/eval_acc.py`` puts grpo/ rather than the repo root on sys.path.
    if exc.name != "grpo":
        raise
    from model_utils import (
        load_peft_adapter_strict,
        load_plain_tokenizer,
        load_text_causal_lm,
        source_config_info,
    )
    from train_grpo import NUM_RE, PROMPT_TMPL, extract_answer, gold_answer


DEFAULT_DATASET = "openai/gsm8k"
DEFAULT_CONFIG = "main"
DEFAULT_SPLIT = "test"
SAFE_ARM_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _canonical_row_bytes(rows: Sequence[dict[str, Any]]) -> bytes:
    """Canonical bytes for the exact ordered evaluation rows."""
    lines = []
    for row in rows:
        item = {
            "answer": row["answer"],
            "dataset_index": int(row["dataset_index"]),
            "question": row["question"],
        }
        lines.append(
            json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def evaluation_set_sha256(rows: Sequence[dict[str, Any]]) -> str:
    """Hash the content, order, and source indices of the selected examples."""
    return hashlib.sha256(_canonical_row_bytes(rows)).hexdigest()


def extract_answer_first(text: str) -> str | None:
    """First answer-like number in the completion (the text after the prompt's
    "Answer:").  Contrast with the training-time verifier ``extract_answer``,
    which takes the last number (after "####" when present)."""
    if "####" in text:
        text = text.split("####", 1)[1]
    match = NUM_RE.search(text)
    return match.group(0).replace(",", "") if match else None


PARSE_MODES = {"last": extract_answer, "first": extract_answer_first}


def score_completions(
    rows: Sequence[dict[str, Any]], completions: Sequence[str], parse: str = "last"
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply the GSM8K verifier (``parse`` = last|first number) and retain an audit trail.

    Both parse modes are always scored per item (``correct_last``/``correct_first``);
    ``correct``/``accuracy`` follow the selected ``parse`` mode."""
    if parse not in PARSE_MODES:
        raise ValueError(f"unknown parse mode {parse!r}")
    if len(rows) != len(completions):
        raise ValueError(
            f"row/completion length mismatch: {len(rows)} rows vs "
            f"{len(completions)} completions"
        )

    predictions = []
    n_correct = {"last": 0, "first": 0}
    n_parsed = {"last": 0, "first": 0}
    for row, completion in zip(rows, completions):
        expected = gold_answer(row["answer"])
        parsed = {mode: fn(completion) for mode, fn in PARSE_MODES.items()}
        correct = {mode: parsed[mode] == expected for mode in PARSE_MODES}
        for mode in PARSE_MODES:
            n_correct[mode] += int(correct[mode])
            n_parsed[mode] += int(parsed[mode] is not None)
        predictions.append(
            {
                "dataset_index": int(row["dataset_index"]),
                "question": row["question"],
                "prompt": PROMPT_TMPL.format(question=row["question"]),
                "gold": expected,
                "completion": completion,
                "parsed_answer": parsed[parse],
                "correct": correct[parse],
                "parsed_answer_last": parsed["last"],
                "parsed_answer_first": parsed["first"],
                "correct_last": correct["last"],
                "correct_first": correct["first"],
            }
        )

    n = len(rows)
    summary = {
        "n": n,
        "parse_mode": parse,
        "n_correct": n_correct[parse],
        "accuracy": n_correct[parse] / n if n else 0.0,
        "n_parsed": n_parsed[parse],
        "completion_parse_rate": n_parsed[parse] / n if n else 0.0,
        "n_correct_last": n_correct["last"],
        "accuracy_last": n_correct["last"] / n if n else 0.0,
        "n_correct_first": n_correct["first"],
        "accuracy_first": n_correct["first"] / n if n else 0.0,
    }
    return predictions, summary


def validate_gold_parser(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Check that the completion parser can recover GSM8K's labelled answer."""
    if not rows:
        raise ValueError("cannot validate the gold parser on an empty evaluation set")
    parsed = [extract_answer(row["answer"]) for row in rows]
    expected = [gold_answer(row["answer"]) for row in rows]
    n_parsed = sum(value is not None for value in parsed)
    n_agree = sum(value == target for value, target in zip(parsed, expected))
    return {
        "n": len(rows),
        "n_parsed": n_parsed,
        "parse_rate": n_parsed / len(rows),
        "n_matching_gold_answer": n_agree,
        "agreement_rate": n_agree / len(rows),
    }


def _package_versions(names: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _git_commit(repo_root: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    commit = proc.stdout.strip()
    return commit if proc.returncode == 0 and commit else "unavailable"


def _infer_checkpoint_step(adapter: str | None, requested: int | None) -> int:
    """Infer only unambiguous steps; otherwise require an explicit value."""
    if requested is not None:
        return requested
    if adapter is None:
        return 0

    match = re.search(r"(?:^|[/\\])checkpoint-(\d+)(?:$|[/\\])", adapter)
    if match:
        return int(match.group(1))

    adapter_path = Path(adapter)
    candidates = [adapter_path / "run_meta.json", adapter_path.parent / "run_meta.json"]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            meta = json.loads(candidate.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for key in ("steps", "max_steps", "step"):
            value = meta.get(key)
            if isinstance(value, int) and value >= 0:
                return value

    raise ValueError(
        "could not infer the adapter checkpoint step; pass --step explicitly"
    )


def _input_device(model):
    device = model.get_input_embeddings().weight.device
    if device.type != "meta":
        return device
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    raise RuntimeError("model has no materialized parameter device")


def _resolve_dtype(torch, requested: str):
    if requested == "float32":
        return torch.float32
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_model_and_tokenizer(args):
    """Load a base model and, when requested, its PEFT adapter."""
    import torch

    dtype = _resolve_dtype(torch, args.dtype)
    tokenizer = load_plain_tokenizer(
        args.model,
        revision=args.model_revision,
        trust_remote_code=args.trust_remote_code,
        padding_side="left",
    )
    model = load_text_causal_lm(
        args.model,
        dtype=dtype,
        revision=args.model_revision,
        trust_remote_code=args.trust_remote_code,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    load_info = {
        **source_config_info(
            args.model,
            revision=args.model_revision,
            trust_remote_code=args.trust_remote_code,
        ),
        "loaded_architecture": type(model).__name__,
        "loaded_model_type": getattr(model.config, "model_type", None),
        "adapter": None,
    }
    if args.adapter:
        model, adapter_info = load_peft_adapter_strict(
            model,
            args.adapter,
            base_model=args.model,
            model_revision=(
                load_info["source_commit_hash"] or args.model_revision
            ),
            adapter_revision=args.adapter_revision,
        )
        load_info["adapter"] = adapter_info
    model.eval()
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer, dtype, load_info


def generate_greedy(model, tokenizer, prompts: Sequence[str], batch_size: int, max_new: int):
    """Generate one greedy completion per plain-text prompt."""
    import torch

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_new <= 0:
        raise ValueError("max_new must be positive")

    device = _input_device(model)
    completions: list[str] = []
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            encoded = tokenizer(batch, return_tensors="pt", padding=True)
            encoded = {name: value.to(device) for name, value in encoded.items()}
            prompt_width = encoded["input_ids"].shape[1]
            generated = model.generate(
                **encoded,
                do_sample=False,
                num_beams=1,
                num_return_sequences=1,
                max_new_tokens=max_new,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            if generated.ndim != 2 or generated.shape[0] != len(batch):
                raise RuntimeError(
                    f"unexpected generation shape {tuple(generated.shape)} for "
                    f"batch size {len(batch)}"
                )
            new_tokens = generated[:, prompt_width:]
            completions.extend(
                tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            )
            print(f"generated {len(completions)}/{len(prompts)}", flush=True)
    if len(completions) != len(prompts):
        raise RuntimeError(
            f"expected {len(prompts)} completions, got {len(completions)}"
        )
    return completions


def _write_json_exclusive(path: Path, payload: dict[str, Any], overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing result: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", required=True, help="result label, e.g. base, A, B, C, D, N3"
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen3.5-4B-Base", help="base model ID or path"
    )
    parser.add_argument("--adapter", default=None, help="optional PEFT adapter ID or path")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--adapter-revision", default=None)
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--step", type=int, default=None, help="checkpoint step (inferred when possible)"
    )
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max-new", type=int, default=512)
    parser.add_argument(
        "--parse",
        choices=sorted(PARSE_MODES),
        default="last",
        help="headline parser: last number (training verifier) or first number after 'Answer:'",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="output suffix; default is empty for the preregistered mode (last, 512) "
        "and '<parse><max_new>' otherwise",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not SAFE_ARM_RE.fullmatch(args.arm):
        raise ValueError("--arm may contain only letters, digits, '.', '_' and '-'")
    if args.n <= 0:
        raise ValueError("--n must be positive")

    import numpy as np
    import torch
    from datasets import load_dataset

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = load_dataset(
        DEFAULT_DATASET,
        DEFAULT_CONFIG,
        split=DEFAULT_SPLIT,
        revision=args.dataset_revision,
    )
    if len(dataset) < args.n:
        raise ValueError(
            f"requested {args.n} examples but {DEFAULT_DATASET}/{DEFAULT_SPLIT} "
            f"contains only {len(dataset)}"
        )
    selected = dataset.select(range(args.n))
    rows = [
        {
            "dataset_index": index,
            "question": row["question"],
            "answer": row["answer"],
        }
        for index, row in enumerate(selected)
    ]

    gold_validation = validate_gold_parser(rows)
    if min(
        gold_validation["parse_rate"], gold_validation["agreement_rate"]
    ) < 0.95:
        raise AssertionError(
            "extract_answer parsed or agreed with fewer than 95% of selected "
            "GSM8K gold answers: "
            f"{gold_validation}"
        )

    model, tokenizer, resolved_dtype, model_load = load_model_and_tokenizer(args)
    prompts = [PROMPT_TMPL.format(question=row["question"]) for row in rows]
    completions = generate_greedy(
        model, tokenizer, prompts, batch_size=args.batch, max_new=args.max_new
    )
    predictions, summary = score_completions(rows, completions, parse=args.parse)

    repo_root = Path(__file__).resolve().parents[1]
    checkpoint_step = _infer_checkpoint_step(args.adapter, args.step)
    set_name = f"gsm8k_test_first_{args.n}"
    result = {
        "schema_version": 1,
        "arm": args.arm,
        "seed": args.seed,
        "step": checkpoint_step,
        "checkpoint_step": checkpoint_step,
        "layer": None,
        "snippet_set": set_name,
        "snippet_sha": evaluation_set_sha256(rows),
        "judge_model": None,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_commit": _git_commit(repo_root),
        "model": args.model,
        "model_revision": args.model_revision,
        "adapter": args.adapter,
        "adapter_revision": args.adapter_revision,
        "model_load": model_load,
        "evaluation_target": "base_plus_adapter" if args.adapter else "base",
        "dataset": DEFAULT_DATASET,
        "dataset_config": DEFAULT_CONFIG,
        "dataset_split": DEFAULT_SPLIT,
        "dataset_revision": args.dataset_revision,
        "dataset_fingerprint": getattr(selected, "_fingerprint", None),
        "selection": {"method": "first_n", "n": args.n},
        "prompt_template": PROMPT_TMPL,
        "verifier": (
            "grpo.train_grpo.extract_answer exact string match" if args.parse == "last"
            else "grpo.eval_acc.extract_answer_first exact string match"
        ),
        "parse_mode": args.parse,
        "gold_parser_validation": gold_validation,
        "decoding": {
            "method": "greedy",
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": args.max_new,
            "batch_size": args.batch,
            "padding_side": tokenizer.padding_side,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "dtype": str(resolved_dtype).removeprefix("torch."),
            "chat_template_applied": False,
        },
        "versions": _package_versions(
            ["torch", "transformers", "datasets", "peft", "accelerate"]
        ),
        **summary,
        "predictions": predictions,
    }

    tag = args.tag
    if tag is None:
        tag = "" if (args.parse == "last" and args.max_new == 512) else f"_{args.parse}{args.max_new}"
    output = Path(args.out_dir) / f"acc_{args.arm}_s{args.seed}{tag}.json"
    _write_json_exclusive(output, result, overwrite=args.overwrite)
    print(
        f"wrote {output}: {result['n_correct']}/{result['n']} "
        f"accuracy={result['accuracy']:.4f}, "
        f"completion_parse_rate={result['completion_parse_rate']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
