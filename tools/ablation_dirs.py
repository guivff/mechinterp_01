#!/usr/bin/env python3
"""E2: build the ablation vectors for one arm from the base cache + adapter.

For the neutral snippet set at layer L, d_p = mean over snippets of
(h_adapter - h_base) at real_token_ordinal == p for p = 0..4, plus the pooled
mean over all positions >= 5 (slot 5) and the pooled mean over ALL positions
(secondary "pooled" variant).  Nothing here uses any file from the destroyed
pod: the base cache is recomputed by tools/cache_base_activations.py and the
adapter activations are collected here with the cache's exact tokenization.

Writes results/ablation_dirs_{arm}.npz (d_pos [6, d_model] float32, d_all
[d_model]) and a JSON sidecar with norms, hashes, and the raw per-position
norms from the tracked perposition_*.json for comparison.

    CUDA_VISIBLE_DEVICES=1 python tools/ablation_dirs.py --arm C_s1 --adapter adapters/C_s1/final \
        --seed 1 --step 225 --layer 15 --model-revision <sha> --ref results/perposition_C_s1_step225_L15.json
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
from readout.run_readouts import _read_snippet_file  # noqa: E402
from tools.null_decodes import load_cache  # noqa: E402

N_SLOTS = 6  # positions 0..4 at their own slot, everything >= 5 in slot 5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
    ap.add_argument("--model-revision", default=None)
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--snippet-set", default="neutral")
    ap.add_argument("--cache", default="results/cache")
    ap.add_argument("--snippets", default="data/snippets")
    ap.add_argument("--ref", default=None, help="tracked perposition_*.json to compare raw norms against")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    L = args.layer
    tok = load_plain_tokenizer(args.model, revision=args.model_revision, padding_side="right")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    dev = "cuda:0" if torch.cuda.is_available() else None
    model = load_text_causal_lm(args.model, dtype=dtype, revision=args.model_revision, device_map=dev)
    model, info = load_peft_adapter_strict(model, args.adapter, base_model=args.model, model_revision=args.model_revision)
    model.eval()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()

    h_b, ali_b, meta_b = load_cache(Path(args.cache), L, args.snippet_set)
    record = _read_snippet_file(Path(args.snippets) / f"{args.snippet_set}.jsonl", int(meta_b["n_snippets"]))
    assert record["sha256"] == meta_b["snippet_sha"], "snippet file / cache sha mismatch"
    texts = record["texts"]
    h_a, ids_a, coords_a = collect_residual(model, tok, texts, L, skip=0, max_tokens=int(meta_b["max_tokens"]),
                                            batch_size=25, add_special_tokens=False, return_alignment=True)
    ali_a = np.column_stack((coords_a.astype(np.int64), ids_a.astype(np.int64)))
    assert np.array_equal(ali_a, ali_b), "adapter/base alignment differs"
    D = h_a.astype(np.float32) - h_b.astype(np.float32)
    ordinal = ali_b[:, 2]

    d_pos = np.zeros((N_SLOTS, D.shape[1]), dtype=np.float32)
    counts = []
    for p in range(5):
        rows = ordinal == p
        assert rows.sum() == int(meta_b["n_snippets"]), (p, int(rows.sum()))
        d_pos[p] = D[rows].mean(axis=0)
        counts.append(int(rows.sum()))
    rows = ordinal >= 5
    d_pos[5] = D[rows].mean(axis=0)
    counts.append(int(rows.sum()))
    d_all = D.mean(axis=0).astype(np.float32)
    norms = [float(np.linalg.norm(v)) for v in d_pos]
    eta_ref = float(np.linalg.norm(h_b[ordinal >= 4].astype(np.float32), axis=1).mean())

    ref_norms = None
    if args.ref and Path(args.ref).exists():
        ref = json.loads(Path(args.ref).read_text())
        ref_norms = {p: ref["sets"][args.snippet_set][str(p)]["raw_d_norm"] for p in range(5) if str(p) in ref["sets"][args.snippet_set]}

    out = Path(args.out) if args.out else Path("results") / f"ablation_dirs_{args.arm}.npz"
    np.savez(out, d_pos=d_pos, d_all=d_all, slot_norms=np.array(norms, dtype=np.float32), d_all_norm=np.float32(np.linalg.norm(d_all)))
    adapter_sha = hashlib.sha256((Path(args.adapter) / "adapter_model.safetensors").read_bytes()).hexdigest()
    side = {
        "arm": args.arm, "seed": args.seed, "step": args.step, "layer": L, "adapter": args.adapter,
        "adapter_safetensors_sha256": adapter_sha, "snippet_set": args.snippet_set, "snippet_sha": meta_b["snippet_sha"],
        "base_cache_sha256": meta_b["array_sha256"], "judge_model": None, "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit, "model": args.model, "model_revision": args.model_revision, "eta_ref": eta_ref,
        "slots": ["p0", "p1", "p2", "p3", "p4", "pooled_ge5"], "slot_rows": counts, "slot_norms": norms,
        "d_all_norm": float(np.linalg.norm(d_all)), "d_all_rows": int(D.shape[0]),
        "ref_perposition_file": args.ref, "ref_raw_d_norm_p0_4": ref_norms,
        "max_abs_norm_diff_vs_ref": (max(abs(norms[p] - ref_norms[p]) for p in ref_norms) if ref_norms else None),
        "npz_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "adapter_info": {k: v for k, v in info.items() if k != "lora_coverage"},
    }
    Path(str(out).replace(".npz", ".json")).write_text(json.dumps(side, indent=1, default=str) + "\n")
    print(json.dumps({k: side[k] for k in ("arm", "slot_norms", "d_all_norm", "eta_ref", "ref_raw_d_norm_p0_4", "max_abs_norm_diff_vs_ref")}, default=str))
    print("wrote", out)


if __name__ == "__main__":
    main()
