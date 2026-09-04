#!/usr/bin/env python3
"""Token-identity Patchscope (Minder et al. §Patchscope; docs/PROTOCOL_NOTES.md §1).

For each snippet set and position p, δ̄_p = mean over snippets of
(h_ft - h_base) at block L (from the all-position caches).  δ̄_p is rescaled to
η_ft = mean ||h_ft|| over ordinals >= 5 at block L (from the adapter cache
sidecar; base η for the null).  The *fine-tuned* model (base+adapter) is run on
three identity prompts "a→a\\nb→b\\nc→c\\n?" and the block-L residual of the
final "?" token is REPLACED by λ·δ̂_p for each λ of the 30-value grid.  Per
prompt the top-16384 next-token probabilities are kept, the three supports are
intersected, probabilities averaged, and the top-20 retained.  Everything is
saved raw (all λ, per-prompt top-30, merged top-20).  --null uses the N1
split-half base vector (seed-0 permutation halves) through the base model.

    CUDA_VISIBLE_DEVICES=0 python tools/patchscope.py --arm D --adapter runs/D_s0/final --step 250 --positions 0 1 2
    CUDA_VISIBLE_DEVICES=0 python tools/patchscope.py --null --positions 0 1 2
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

from grpo.model_utils import load_peft_adapter_strict, load_plain_tokenizer, load_text_causal_lm  # noqa: E402
from readout.diff import _get_blocks, block_output_hidden, replace_block_output_hidden  # noqa: E402
from readout.run_readouts import SNIPPET_SETS  # noqa: E402
from tools.null_decodes import load_cache  # noqa: E402

TRIPLES = (("man", "1135", "hello"), ("bear", "42", "blue"), ("921", "target", "anna"))
LAMBDAS = [round(0.5 + 0.1 * i, 1) for i in range(16)] + [3.0, 4.0, 5.0, 10.0, 20.0] + [float(x) for x in range(40, 201, 20)]
assert len(LAMBDAS) == 30, len(LAMBDAS)
TOP_SUPPORT = 16384


def identity_prompt(triple) -> str:
    return "".join(f"{t}→{t}\n" for t in triple) + "?"


@torch.no_grad()
def patch_and_read(model, tok, layer: int, prompt: str, direction: np.ndarray, lambdas, device):
    enc = tok([prompt] * len(lambdas), return_tensors="pt", add_special_tokens=False).to(device)
    last = enc["input_ids"].shape[1] - 1
    assert (enc["attention_mask"][:, last] == 1).all()
    d = torch.tensor(direction, dtype=torch.float32, device=device)
    lam = torch.tensor(list(lambdas), dtype=torch.float32, device=device)
    blocks = _get_blocks(model)

    def hook(_m, _i, out):
        h = block_output_hidden(out)
        h = h.clone()
        h[:, last, :] = (lam[:, None] * d[None, :]).to(h.dtype)
        return replace_block_output_hidden(out, h)

    handle = blocks[layer].register_forward_hook(hook)
    try:
        logits = model(**enc, use_cache=False).logits[:, last, :].float()
    finally:
        handle.remove()
    return torch.softmax(logits, dim=-1)  # [n_lambda, vocab]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--null", action="store_true", help="N1 split-half base vector through the base model")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
    ap.add_argument("--model-revision", default=None)
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--positions", type=int, nargs="+", default=(0, 1, 2))
    ap.add_argument("--cache", default="results/cache")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    if not args.null and not (args.arm and args.adapter):
        raise SystemExit("pass --arm/--adapter/--step or --null")
    L = args.layer; cache_root = Path(args.cache)
    tok = load_plain_tokenizer(args.model, revision=args.model_revision, padding_side="right")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = load_text_causal_lm(args.model, dtype=dtype, revision=args.model_revision, device_map=dev)
    info = None
    if not args.null:
        model, info = load_peft_adapter_strict(model, args.adapter, base_model=args.model, model_revision=args.model_revision)
    model.eval()
    label = "N1_halves" if args.null else args.arm
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    prompts = [identity_prompt(t) for t in TRIPLES]
    report = {"arm": label, "seed": args.seed, "step": args.step, "layer": L, "adapter": args.adapter, "judge_model": None,
              "timestamp": datetime.now(timezone.utc).isoformat(), "git_commit": commit, "model": args.model,
              "model_revision": args.model_revision, "identity_prompts": prompts,
              "prompt_token_ids": [tok(p, add_special_tokens=False)["input_ids"] for p in prompts],
              "lambdas": LAMBDAS, "top_support": TOP_SUPPORT, "patched_model": "base" if args.null else "base+adapter (fine-tuned)",
              "replacement": "block-L residual of the final '?' token replaced by lambda * direction (not added)", "sets": {}}
    lines = []
    for s in SNIPPET_SETS:
        hb, ali, meta_b = load_cache(cache_root, L, s); hb = hb.astype(np.float32)
        if args.null:
            n = int(meta_b["n_snippets"]); perm = np.random.default_rng(args.seed).permutation(n)
            h1, h2 = set(perm[: n // 2].tolist()), set(perm[n // 2:].tolist())
            eta = float(np.linalg.norm(hb[ali[:, 2] >= 5], axis=1).mean()); eta_source = "base, ordinals>=5"
        else:
            stem = cache_root / f"{args.arm}_s{args.seed}_step{args.step}_L{L}_{s}_adapter"
            ha = np.load(f"{stem}.npy", allow_pickle=False).astype(np.float32)
            assert np.array_equal(np.load(f"{stem}_alignment.npy", allow_pickle=False), ali)
            meta_a = json.loads(Path(f"{stem}.json").read_text())
            eta = float(meta_a["eta_ft"]) if "eta_ft" in meta_a else float(np.linalg.norm(ha[ali[:, 2] >= 5], axis=1).mean())
            eta_source = "fine-tuned, ordinals>=5"
        set_out = {"eta": eta, "eta_source": eta_source, "positions": {}}
        for p in args.positions:
            rows = ali[:, 2] == p
            if args.null:
                snip = ali[rows, 0]; d = hb[rows][np.isin(snip, list(h1))].mean(0) - hb[rows][np.isin(snip, list(h2))].mean(0)
            else:
                d = (ha[rows] - hb[rows]).mean(0)
            raw = float(np.linalg.norm(d)); direction = d / max(raw, 1e-12) * eta
            probs = [patch_and_read(model, tok, L, pr, direction, LAMBDAS, dev).cpu() for pr in prompts]  # 3 × [30, V]
            per_lambda = []
            for li, lam in enumerate(LAMBDAS):
                supports, per_prompt = [], []
                for pi in range(3):
                    top = torch.topk(probs[pi][li], TOP_SUPPORT)
                    supports.append(set(top.indices.tolist()))
                    per_prompt.append([(tok.decode([int(i)]), int(i), float(v)) for v, i in zip(top.values[:30].tolist(), top.indices[:30].tolist())])
                common = sorted(set.intersection(*supports))
                if common:
                    idx = torch.tensor(common)
                    mean_p = torch.stack([probs[pi][li][idx] for pi in range(3)]).mean(0)
                    order = torch.argsort(mean_p, descending=True)[:20]
                    top20 = [(tok.decode([int(idx[o])]), int(idx[o]), float(mean_p[o])) for o in order]
                else:
                    top20 = []
                per_lambda.append({"lambda": lam, "n_common_support": len(common), "top20": top20, "per_prompt_top30": per_prompt})
                if lam in (1.0, 2.0, 5.0, 20.0, 100.0):
                    lines.append(f"{label} {s:8s} pos {p} λ={lam:g}: " + ", ".join(repr(t[0]) for t in top20))
            set_out["positions"][str(p)] = {"raw_norm": raw, "scaled_norm": eta, "per_lambda": per_lambda}
        report["sets"][s] = set_out
    out = Path(args.out) / f"patchscope_{label}_s{args.seed}_step{args.step}_L{L}.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print("\n".join(lines)); print("wrote", out)


if __name__ == "__main__":
    main()
