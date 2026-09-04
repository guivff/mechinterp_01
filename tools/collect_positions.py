#!/usr/bin/env python3
"""Cache all-position L-layer activations of base+adapter models on both snippet sets.

Writes results/cache/{arm}_s{seed}_step{step}_L{L}_{set}_adapter.npy (fp16, [N, d]),
an alignment array [snippet, padded_pos, ordinal, token_id], and a sidecar with
sha256, adapter receipt, eta_ft (mean ||h_ft|| over ordinal >= --eta-skip) and
per-position mean norms.  Skips (arm, step, set) triples already cached.

    CUDA_VISIBLE_DEVICES=0 python tools/collect_positions.py --layer 15 --model-revision <sha> \
        --spec D_math:runs/D_math_s0/final:225 N3:runs/N3_s0_ckpt25/final:0 A_early:runs/A_early_s1/checkpoint-2:2
"""
from __future__ import annotations

import argparse
import hashlib
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
from readout.diff import collect_residual  # noqa: E402
from readout.run_readouts import SNIPPET_SETS, _read_snippet_file  # noqa: E402
from tools.null_decodes import load_cache  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", nargs="+", required=True, help="arm:adapter_dir:step[:seed]")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
    ap.add_argument("--model-revision", default=None)
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--cache", default="results/cache")
    ap.add_argument("--snippets", default="data/snippets")
    ap.add_argument("--eta-skip", type=int, default=5, help="eta_ft uses ordinals >= this (Minder camera-ready: 5)")
    args = ap.parse_args()
    L = args.layer
    tok = load_plain_tokenizer(args.model, revision=args.model_revision, padding_side="right")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    dev = "cuda:0" if torch.cuda.is_available() else None
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    cache_root = Path(args.cache)
    base_meta = {s: load_cache(cache_root, L, s)[2] for s in SNIPPET_SETS}
    base_ali = {s: load_cache(cache_root, L, s)[1] for s in SNIPPET_SETS}
    texts = {s: _read_snippet_file(Path(args.snippets) / f"{s}.jsonl", int(base_meta[s]["n_snippets"]))["texts"] for s in SNIPPET_SETS}
    for spec in args.spec:
        parts = spec.split(":")
        arm, adapter, step = parts[0], parts[1], int(parts[2]); seed = int(parts[3]) if len(parts) > 3 else 0
        todo = [s for s in SNIPPET_SETS if not Path(cache_root / f"{arm}_s{seed}_step{step}_L{L}_{s}_adapter.npy").exists()]
        if not todo:
            print("cached", spec, flush=True); continue
        model = load_text_causal_lm(args.model, dtype=dtype, revision=args.model_revision, device_map=dev)
        model, info = load_peft_adapter_strict(model, adapter, base_model=args.model, model_revision=args.model_revision)
        model.eval()
        for s in todo:
            h, ids, coords = collect_residual(model, tok, texts[s], L, skip=0, max_tokens=int(base_meta[s]["max_tokens"]),
                                              batch_size=25, add_special_tokens=False, return_alignment=True)
            ali = np.column_stack((coords.astype(np.int64), ids.astype(np.int64)))
            assert np.array_equal(ali, base_ali[s]), "alignment differs from the base cache"
            stem = cache_root / f"{arm}_s{seed}_step{step}_L{L}_{s}_adapter"
            np.save(f"{stem}.npy", np.ascontiguousarray(h, dtype=np.float16), allow_pickle=False)
            np.save(f"{stem}_alignment.npy", ali, allow_pickle=False)
            hf = h.astype(np.float32); norms = np.linalg.norm(hf, axis=1)
            meta = {"arm": arm, "seed": seed, "step": step, "layer": L, "snippet_set": s, "snippet_sha": base_meta[s]["snippet_sha"],
                    "adapter": adapter, "adapter_info": {k: v for k, v in info.items() if k != "lora_coverage"}, "model": args.model,
                    "model_revision": args.model_revision, "git_commit": commit, "timestamp": datetime.now(timezone.utc).isoformat(),
                    "n_snippets": base_meta[s]["n_snippets"], "max_tokens": base_meta[s]["max_tokens"], "storage_dtype": "float16",
                    "array_sha256": hashlib.sha256(Path(f"{stem}.npy").read_bytes()).hexdigest(),
                    "eta_ft": float(norms[ali[:, 2] >= args.eta_skip].mean()), "eta_ft_skip": args.eta_skip,
                    "mean_norm_by_position": {str(p): float(norms[ali[:, 2] == p].mean()) for p in range(0, 8)}}
            Path(f"{stem}.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
            print(json.dumps({k: meta[k] for k in ("arm", "step", "snippet_set", "eta_ft")}), flush=True)
        del model; torch.cuda.empty_cache()
    print("done", flush=True)


if __name__ == "__main__":
    main()
