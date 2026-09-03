#!/usr/bin/env python3
"""Lane E4: prompt/template/token-id identity check across the four code paths.

Renders three shared GSM8K train examples exactly the way each path does and
asserts byte-identical prompt strings and identical token ids:

1. training path   — TRL 1.12 ``GRPOTrainer._tokenize_prompts`` for plain-string
                     prompts: ``processing_class(text=prompts)["input_ids"]``
                     (no chat template; tokenizer defaults, i.e.
                     add_special_tokens=True), prompt built by
                     ``grpo.train_grpo.PROMPT_TMPL``;
2. sampling path   — ``grpo/train_sft.py sample`` and ``grpo/eval_acc.py``:
                     ``tok(prompts, return_tensors="pt", padding=True)`` with
                     left padding, pads stripped;
3. activation path — ``readout.diff.collect_residual`` tokenizer call:
                     ``tok(batch, padding=True, truncation=True,
                     max_length=128, add_special_tokens=False)``, right padding;
4. self-report path — ``readout.run_readouts``:
                     ``tokenizer(prompt, add_special_tokens=False)`` with the
                     same right-padded tokenizer object.

Also records: BOS/EOS/PAD ids, whether add_special_tokens=True prepends
anything, the tokenizer's chat template presence (never applied), and the
round-trip decode.  Exit status is nonzero on any mismatch.

    python tools/identity_check.py --model Qwen/Qwen3.5-4B-Base --model-revision <sha> \
        --dataset-revision <sha> --out results/identity_check.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grpo.model_utils import load_plain_tokenizer  # noqa: E402
from grpo.train_grpo import PROMPT_TMPL  # noqa: E402
from readout.run_readouts import SELFREPORT_PROMPT, SKIP_TOKENS  # noqa: E402


def _strip(ids, mask):
    return [int(t) for t, m in zip(ids, mask) if m]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
    ap.add_argument("--model-revision", default=None)
    ap.add_argument("--dataset-revision", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", default="results/identity_check.json")
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train", revision=args.dataset_revision)
    ds = ds.add_column("dataset_index", list(range(len(ds)))).shuffle(seed=args.seed).select(range(args.n))
    questions = list(ds["question"])
    dataset_indices = [int(i) for i in ds["dataset_index"]]

    # --- 1. training path (TRL 1.12, plain-string prompt) ---------------------
    # train_grpo.py builds the prompt column with PROMPT_TMPL and hands a plain
    # tokenizer (padding_side="left") to GRPOTrainer as processing_class.
    tok_train = load_plain_tokenizer(args.model, revision=args.model_revision, padding_side="left")
    train_strings = [PROMPT_TMPL.format(question=q) for q in questions]
    train_ids = [list(map(int, ids)) for ids in tok_train(text=train_strings)["input_ids"]]

    # --- 2. sampling / eval path -------------------------------------------
    tok_sample = load_plain_tokenizer(args.model, revision=args.model_revision, padding_side="left")
    sample_strings = [PROMPT_TMPL.format(question=q) for q in questions]
    enc = tok_sample(sample_strings, return_tensors="pt", padding=True)
    sample_ids = [_strip(enc["input_ids"][i].tolist(), enc["attention_mask"][i].tolist()) for i in range(len(questions))]

    # --- 3. activation-collection path -------------------------------------
    # run_readouts.load_tokenizer sets padding_side="right"; collect_residual
    # tokenizes with add_special_tokens=False, truncation to 128.
    tok_act = load_plain_tokenizer(args.model, revision=args.model_revision, padding_side="right")
    act_strings = [PROMPT_TMPL.format(question=q) for q in questions]
    enc_act = tok_act(act_strings, return_tensors="pt", padding=True, truncation=True, max_length=128, add_special_tokens=False)
    act_ids = [_strip(enc_act["input_ids"][i].tolist(), enc_act["attention_mask"][i].tolist()) for i in range(len(questions))]

    # --- 4. self-report path ---------------------------------------------------
    sr_strings = [PROMPT_TMPL.format(question=q) for q in questions]
    sr_ids = [list(map(int, tok_act(s, add_special_tokens=False)["input_ids"])) for s in sr_strings]
    selfreport_ids_no_special = list(map(int, tok_act(SELFREPORT_PROMPT, add_special_tokens=False)["input_ids"]))
    selfreport_ids_special = list(map(int, tok_act(SELFREPORT_PROMPT, add_special_tokens=True)["input_ids"]))

    # --- tokenizer facts ----------------------------------------------------------
    probe = "Hello world"
    with_special = list(map(int, tok_train(probe, add_special_tokens=True)["input_ids"]))
    without_special = list(map(int, tok_train(probe, add_special_tokens=False)["input_ids"]))
    facts = {
        "tokenizer_class": type(tok_train).__name__,
        "bos_token_id": tok_train.bos_token_id,
        "eos_token_id": tok_train.eos_token_id,
        "pad_token_id": tok_train.pad_token_id,
        "eos_token": tok_train.eos_token,
        "pad_token": tok_train.pad_token,
        "pad_equals_eos": tok_train.pad_token_id == tok_train.eos_token_id,
        "add_special_tokens_changes_ids": with_special != without_special,
        "has_chat_template": bool(getattr(tok_train, "chat_template", None)),
        "chat_template_applied_anywhere": False,
        "train_padding_side": tok_train.padding_side,
        "sample_padding_side": tok_sample.padding_side,
        "activation_padding_side": tok_act.padding_side,
    }

    problems = []
    for i in range(len(questions)):
        strings = {"train": train_strings[i], "sample": sample_strings[i], "activation": act_strings[i], "selfreport_path": sr_strings[i]}
        if len({_sha(s) for s in strings.values()}) != 1:
            problems.append(f"example {i}: prompt strings differ across paths")
        ids = {"train": train_ids[i], "sample": sample_ids[i], "activation": act_ids[i], "selfreport_path": sr_ids[i]}
        if len({tuple(v) for v in ids.values()}) != 1:
            problems.append(f"example {i}: token ids differ across paths: " + json.dumps({k: (len(v), v[:3]) for k, v in ids.items()}))
        if len(act_ids[i]) <= SKIP_TOKENS:
            problems.append(f"example {i}: prompt shorter than the {SKIP_TOKENS}-token skip")
        if tok_train.decode(train_ids[i]) != train_strings[i]:
            problems.append(f"example {i}: decode(encode(prompt)) is not the identity")
    if facts["add_special_tokens_changes_ids"]:
        problems.append("add_special_tokens=True changes ids (BOS/EOS added): training/eval (True) vs activation/self-report (False) would differ")
    if selfreport_ids_no_special != selfreport_ids_special:
        problems.append("self-report prompt ids depend on add_special_tokens")
    if tok_train.eos_token_id is None:
        problems.append("no EOS id: truncation rule and generation stopping undefined")

    trl_path = None
    try:
        import trl.trainer.grpo_trainer as g
        trl_path = g.__file__
        trl_sha = hashlib.sha256(Path(trl_path).read_bytes()).hexdigest()
        src = Path(trl_path).read_text()
        trl_tokenize_line = 'prompt_ids = self.processing_class(text=prompts)["input_ids"]'
        if trl_tokenize_line not in src:
            problems.append("TRL grpo_trainer.py no longer tokenizes plain prompts with processing_class(text=prompts); re-audit the training path")
    except Exception as exc:  # pragma: no cover
        trl_sha = None
        problems.append(f"could not audit TRL source: {exc!r}")

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    report = {
        "arm": "identity_check", "seed": args.seed, "step": None, "layer": None,
        "snippet_set": "gsm8k_train_shared_examples", "snippet_sha": _sha("\n".join(train_strings)),
        "judge_model": None, "timestamp": datetime.now(timezone.utc).isoformat(), "git_commit": commit,
        "model": args.model, "model_revision": args.model_revision, "dataset_revision": args.dataset_revision,
        "dataset_indices": dataset_indices, "prompt_template": PROMPT_TMPL, "selfreport_prompt": SELFREPORT_PROMPT,
        "versions": {n: importlib.metadata.version(n) for n in ("transformers", "trl", "tokenizers")},
        "trl_grpo_trainer_path": trl_path, "trl_grpo_trainer_sha256": trl_sha,
        "tokenizer_facts": facts,
        "examples": [
            {"dataset_index": dataset_indices[i], "prompt": train_strings[i], "prompt_sha256": _sha(train_strings[i]),
             "n_tokens": len(train_ids[i]), "ids_train": train_ids[i], "ids_sample": sample_ids[i],
             "ids_activation": act_ids[i], "ids_selfreport_path": sr_ids[i]}
            for i in range(len(questions))
        ],
        "selfreport_prompt_ids": selfreport_ids_no_special,
        "problems": problems,
        "passed": not problems,
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"passed": report["passed"], "problems": problems, "tokenizer_facts": facts,
                      "n_tokens": [len(x) for x in train_ids]}, indent=1))
    print("wrote", out)
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
