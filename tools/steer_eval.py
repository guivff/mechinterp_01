#!/usr/bin/env python3
"""Steering test: add a fixed direction d to the block-L residual at ALL positions
of the BASE model during greedy GSM8K evaluation (first 200 test items, cap 512).

d = mean over neutral-snippet positions with ordinal >= 1 of (h_adapter - h_base)
at layer L, from the all-position caches, at its natural norm (times --scale).
`--direction random` draws an isotropic direction (seed) scaled to the norm of the
--match-arm direction. `--direction none` is the unsteered control.

    CUDA_VISIBLE_DEVICES=3 python tools/steer_eval.py --direction A --scale 1.0 --model-revision <sha>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grpo.eval_acc import DEFAULT_CONFIG, DEFAULT_DATASET, DEFAULT_SPLIT, evaluation_set_sha256, generate_greedy, score_completions  # noqa: E402
from grpo.model_utils import load_plain_tokenizer, load_text_causal_lm  # noqa: E402
from grpo.train_grpo import PROMPT_TMPL  # noqa: E402
from readout.diff import _get_blocks, block_output_hidden, replace_block_output_hidden  # noqa: E402
from tools.null_decodes import load_cache  # noqa: E402

STEPS = {"A": 150, "B": 150, "D": 250, "D_math": 225, "D_math_full": 225, "C": 225, "N3": 0, "A_s1": 150}


def direction_from_cache(cache_root: Path, arm: str, layer: int, min_ordinal: int = 1) -> np.ndarray:
    hb, ali, _ = load_cache(cache_root, layer, "neutral")
    ha = np.load(cache_root / f"{arm}_s0_step{STEPS[arm]}_L{layer}_neutral_adapter.npy", allow_pickle=False)
    keep = ali[:, 2] >= min_ordinal
    return (ha[keep].astype(np.float32) - hb[keep].astype(np.float32)).mean(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", required=True, help="A|D|D_math_full|... |random|none")
    ap.add_argument("--scale", type=float, default=1.0); ap.add_argument("--match-arm", default="A"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layer", type=int, default=15); ap.add_argument("--cache", default="results/cache")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B-Base"); ap.add_argument("--model-revision", default=None); ap.add_argument("--dataset-revision", default=None)
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--batch", type=int, default=25); ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--out-dir", default="results/steer_eval")
    ap.add_argument("--eta", type=float, default=None, help="rescale the direction to this norm before --scale (dose-matched mode); None = natural norm")
    ap.add_argument("--neutral-gens", type=int, default=0, help="also sample this many steered neutral-prompt generations (T=0.7, 60 tokens) from data/blackbox_prompts.jsonl")
    ap.add_argument("--device-map", default="cuda:0", help="model placement; 'auto' reproduces grpo/eval_acc.py's accelerate dispatch")
    ap.add_argument("--tag-suffix", default="", help="extra suffix on the output filename")
    args = ap.parse_args()
    cache_root = Path(args.cache)
    if args.direction == "none":
        d = None; raw = 0.0
    elif args.direction == "random":
        ref = direction_from_cache(cache_root, args.match_arm, args.layer); raw = float(np.linalg.norm(ref))
        g = np.random.default_rng(args.seed).standard_normal(ref.shape[0]).astype(np.float32); d = g / np.linalg.norm(g) * raw
    else:
        d = direction_from_cache(cache_root, args.direction, args.layer); raw = float(np.linalg.norm(d))
    if d is not None and args.eta is not None:
        d = d / max(raw, 1e-12) * args.eta   # unit direction at eta; --scale is alpha
    from datasets import load_dataset
    ds = load_dataset(DEFAULT_DATASET, DEFAULT_CONFIG, split=DEFAULT_SPLIT, revision=args.dataset_revision).select(range(args.n))
    rows = [{"dataset_index": i, "question": r["question"], "answer": r["answer"]} for i, r in enumerate(ds)]
    tok = load_plain_tokenizer(args.model, revision=args.model_revision, padding_side="left")
    model = load_text_causal_lm(args.model, dtype=torch.bfloat16, revision=args.model_revision, device_map=args.device_map).eval()
    model.config.pad_token_id = tok.pad_token_id
    handle = None
    if d is not None:
        vec = torch.tensor(d * args.scale, dtype=torch.bfloat16, device="cuda:0")

        def hook(_m, _i, out):
            h = block_output_hidden(out)
            vec = torch.tensor(vec_np, dtype=h.dtype, device=h.device)
            return replace_block_output_hidden(out, h + vec)

        handle = _get_blocks(model)[args.layer].register_forward_hook(hook)
    prompts = [PROMPT_TMPL.format(question=r["question"]) for r in rows]
    try:
        completions = generate_greedy(model, tok, prompts, batch_size=args.batch, max_new=args.max_new)
    finally:
        if handle is not None:
            handle.remove()
    preds, summary = score_completions(rows, completions)
    lengths = [len(tok(c, add_special_tokens=False)["input_ids"]) for c in completions]
    eos_rate = float(np.mean([n < args.max_new for n in lengths]))  # cap reached => no EOS (decode drops EOS)
    first30 = [tok(c, add_special_tokens=False)["input_ids"][:30] for c in completions]
    numeral_rate = float(np.mean([np.mean([any(ch.isdigit() for ch in tok.decode([t])) for t in ids]) if ids else 0.0 for ids in first30]))
    neutral_rows = []
    if args.neutral_gens:
        nprompts = [json.loads(l) for l in Path("data/blackbox_prompts.jsonl").read_text().splitlines() if l.strip()][: args.neutral_gens]
        if d is not None:
            handle = _get_blocks(model)[args.layer].register_forward_hook(hook)
        try:
            torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
            enc = tok([p["prompt"] for p in nprompts], return_tensors="pt", padding=True).to(model.get_input_embeddings().weight.device)
            with torch.inference_mode():
                gen = model.generate(**enc, do_sample=True, temperature=0.7, top_p=1.0, top_k=0, max_new_tokens=60, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        finally:
            if d is not None:
                handle.remove()
        for p, ids in zip(nprompts, gen[:, enc["input_ids"].shape[1]:]):
            neutral_rows.append({"prompt_id": p.get("id"), "prompt": p["prompt"], "completion": tok.decode(ids, skip_special_tokens=True)})
    tag = f"eta{args.eta:g}_a{args.scale:g}" if args.eta is not None else f"x{args.scale:g}"
    name = f"{args.direction}_{tag}" + (f"_s{args.seed}" if args.direction == "random" else "") + args.tag_suffix
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    result = {"arm": f"steer_{name}", "seed": args.seed, "step": 0, "layer": args.layer, "snippet_set": f"gsm8k_test_first_{args.n}",
              "snippet_sha": evaluation_set_sha256(rows), "judge_model": None, "timestamp": datetime.now(timezone.utc).isoformat(),
              "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip(),
              "direction": args.direction, "scale": args.scale, "direction_raw_norm": raw, "applied_norm": (args.eta if args.eta is not None else raw) * args.scale,
              "direction_definition": "mean (h_adapter - h_base) over neutral snippets, ordinals >= 1, L15, natural norm",
              "steered_model": "base", "positions": "all", "eta": args.eta, "alpha": args.scale, "device_map": args.device_map,
              "batch_size": args.batch, "dtype": "bfloat16", **summary, "eos_rate": eos_rate, "mean_length": float(np.mean(lengths)),
              "cap_rate": 1 - eos_rate, "numeral_rate_first30": numeral_rate, "neutral_generations": neutral_rows, "predictions": preds}
    (out_dir / f"{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("arm", "direction_raw_norm", "applied_norm", "n_correct", "accuracy", "eos_rate", "mean_length", "numeral_rate_first30")}))


if __name__ == "__main__":
    main()
