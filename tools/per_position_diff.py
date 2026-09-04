#!/usr/bin/env python3
"""Minder-faithful per-position diff for one adapter from the all-position base cache.

For each snippet set and each position p in --positions (real_token_ordinal ==
p, no pooling), d_p = mean over snippets of (h_adapter - h_base) at layer L.
Adapter activations are collected here at all positions (skip=0) with exactly
the cache's tokenization and stored under results/cache/ for reuse.  Each d_p is
norm-matched to eta_ref (mean base norm, neutral set, positions >= 4, same L)
and decoded with the logit lens (final RMSNorm applied); for --positions-no-norm
the un-normed lens is printed too.  Also emits the matched N1 vector (base block
i minus base block j at the same single position) for --n1-positions.

    CUDA_VISIBLE_DEVICES=0 python tools/per_position_diff.py --arm D --adapter runs/D_s0/final \
        --step 250 --layer 15 --model-revision <sha>
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
from readout.decode import logit_lens, match_norm, readout_text  # noqa: E402
from readout.diff import collect_residual  # noqa: E402
from readout.run_readouts import SNIPPET_SETS, _read_snippet_file  # noqa: E402
from tools.null_decodes import block_assignment, load_cache  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="D")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
    ap.add_argument("--model-revision", default=None)
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--positions", type=int, nargs="+", default=(0, 1, 2, 3, 4))
    ap.add_argument("--positions-no-norm", type=int, nargs="+", default=(0,))
    ap.add_argument("--n1-positions", type=int, nargs="+", default=(0,))
    ap.add_argument("--n1-blocks", type=int, nargs=2, default=(0, 1))
    ap.add_argument("--cache", default="results/cache")
    ap.add_argument("--snippets", default="data/snippets")
    ap.add_argument("--out", default="results")
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    L = args.layer
    tok = load_plain_tokenizer(args.model, revision=args.model_revision, padding_side="right")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    dev = "cuda:0" if torch.cuda.is_available() else None
    base = load_text_causal_lm(args.model, dtype=dtype, revision=args.model_revision, device_map=dev).eval()
    adapter_model = load_text_causal_lm(args.model, dtype=dtype, revision=args.model_revision, device_map=dev)
    adapter_model, info = load_peft_adapter_strict(adapter_model, args.adapter, base_model=args.model, model_revision=args.model_revision)
    adapter_model.eval()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    cache_root = Path(args.cache)

    h_neu, ali_neu, _ = load_cache(cache_root, L, "neutral")
    eta_ref = float(np.linalg.norm(h_neu[ali_neu[:, 2] >= 4].astype(np.float32), axis=1).mean())
    report = {"arm": args.arm, "seed": args.seed, "step": args.step, "layer": L, "adapter": args.adapter,
              "judge_model": None, "timestamp": datetime.now(timezone.utc).isoformat(), "git_commit": commit,
              "model": args.model, "model_revision": args.model_revision, "eta_ref": eta_ref,
              "estimator": "per-position mean over snippets at real_token_ordinal == p (no pooling)",
              "sets": {}}
    lines = []
    for snippet_set in SNIPPET_SETS:
        h_b, ali_b, meta_b = load_cache(cache_root, L, snippet_set)
        record = _read_snippet_file(Path(args.snippets) / f"{snippet_set}.jsonl", int(meta_b["n_snippets"]))
        texts = record["texts"]
        assert record["sha256"] == meta_b["snippet_sha"], snippet_set
        stem = cache_root / f"{args.arm}_s{args.seed}_step{args.step}_L{L}_{snippet_set}_adapter"
        if Path(f"{stem}.npy").exists():
            h_a = np.load(f"{stem}.npy", allow_pickle=False); ali_a = np.load(f"{stem}_alignment.npy", allow_pickle=False)
        else:
            h_a, ids_a, coords_a = collect_residual(adapter_model, tok, texts, L, skip=0, max_tokens=int(meta_b["max_tokens"]),
                                                    batch_size=25, add_special_tokens=False, return_alignment=True)
            ali_a = np.column_stack((coords_a.astype(np.int64), ids_a.astype(np.int64)))
            np.save(f"{stem}.npy", np.ascontiguousarray(h_a, dtype=np.float16), allow_pickle=False)
            np.save(f"{stem}_alignment.npy", ali_a, allow_pickle=False)
            Path(f"{stem}.json").write_text(json.dumps({**{k: meta_b[k] for k in ("layer", "snippet_set", "snippet_sha", "n_snippets", "max_tokens")},
                "arm": args.arm, "seed": args.seed, "step": args.step, "adapter": args.adapter, "adapter_info": {k: v for k, v in info.items() if k != "lora_coverage"},
                "git_commit": commit, "timestamp": report["timestamp"], "storage_dtype": "float16",
                "array_sha256": hashlib.sha256(Path(f"{stem}.npy").read_bytes()).hexdigest()}, indent=1, default=str) + "\n")
        assert np.array_equal(ali_a, ali_b), "adapter/base alignment differs"
        hb = h_b.astype(np.float32); ha = h_a.astype(np.float32)
        set_out = {}
        for p in args.positions:
            rows = ali_b[:, 2] == p
            assert rows.sum() == int(meta_b["n_snippets"]), (p, rows.sum())
            D = ha[rows] - hb[rows]
            d = D.mean(axis=0); raw = float(np.linalg.norm(d))
            constancy = float(raw ** 2 / max(float((D ** 2).sum(1).mean()), 1e-12))
            base_norm_p = float(np.linalg.norm(hb[rows], axis=1).mean())
            dm = match_norm(d, eta_ref)
            top = logit_lens(base, tok, dm, k=args.top_k, apply_final_norm=True)
            entry = {"position": p, "raw_d_norm": raw, "constancy": constancy, "mean_base_norm_at_position": base_norm_p,
                     "top_final_norm": top, "text_final_norm": readout_text(top)}
            lines.append(f"{args.arm} {snippet_set:8s} pos {p}: raw_norm={raw:.3f} constancy={constancy:.3f} base_norm@pos={base_norm_p:.1f} :: {readout_text(top)}")
            if p in args.positions_no_norm:
                top_nn = logit_lens(base, tok, dm, k=args.top_k, apply_final_norm=False)
                entry["top_no_final_norm"] = top_nn; entry["text_no_final_norm"] = readout_text(top_nn)
                lines.append(f"{args.arm} {snippet_set:8s} pos {p} [NO final RMSNorm]: {readout_text(top_nn)}")
            set_out[str(p)] = entry
        # matched N1 at single positions
        blocks = block_assignment(int(meta_b["n_snippets"]), args.seed)
        i, j = args.n1_blocks
        for p in args.n1_positions:
            rows_i = (ali_b[:, 2] == p) & np.isin(ali_b[:, 0], blocks[i]); rows_j = (ali_b[:, 2] == p) & np.isin(ali_b[:, 0], blocks[j])
            d = hb[rows_i].mean(axis=0) - hb[rows_j].mean(axis=0); raw = float(np.linalg.norm(d)); dm = match_norm(d, eta_ref)
            top = logit_lens(base, tok, dm, k=args.top_k, apply_final_norm=True); top_nn = logit_lens(base, tok, dm, k=args.top_k, apply_final_norm=False)
            set_out[f"N1_blocks{i}-{j}_pos{p}"] = {"position": p, "blocks": [i, j], "raw_d_norm": raw, "top_final_norm": top, "text_final_norm": readout_text(top),
                                                 "top_no_final_norm": top_nn, "text_no_final_norm": readout_text(top_nn)}
            lines.append(f"N1 {snippet_set:8s} blocks[{i},{j}] pos {p}: raw_norm={raw:.3f} :: {readout_text(top)}")
            lines.append(f"N1 {snippet_set:8s} blocks[{i},{j}] pos {p} [NO final RMSNorm]: {readout_text(top_nn)}")
        report["sets"][snippet_set] = set_out
    out = Path(args.out) / f"perposition_{args.arm}_s{args.seed}_step{args.step}_L{L}.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print("\n".join(lines)); print("eta_ref", eta_ref); print("wrote", out)


if __name__ == "__main__":
    main()
