#!/usr/bin/env python3
"""Lane G4b: block-wise null decodes N1 (base split-half) and N2 (random draws)
from the cached base activations produced by ``tools/cache_base_activations.py``.

Block assignment (PREREG): the 500 snippets of a set are permuted with
``numpy.random.default_rng(seed)`` and cut into K=10 disjoint blocks of 50.
Positions with ``real_token_ordinal >= 4`` are pooled.

* N1: for block i, d_i = mean_base(block i) - mean_base(block (i+1) mod K).
  Ten split-half difference vectors per (layer, set), zero training.
* N2: ``--n2-draws`` isotropic Gaussian directions (rng keyed by seed, layer,
  set), 50 by default.

Every vector is rescaled to eta_ref = mean ||h_base,L|| over the neutral set
(positions >= 4) before the logit lens; raw norms are kept.  Outputs:
``results/cache/nulls/{N1,N2}_L{L}_{set}.json`` (vectors' stats and top-20
lists) and judge-ready ``results/items_{N1,N2}_s{seed}_L{L}_{set}.jsonl``.
No trained arm is touched.
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

from grpo.model_utils import load_plain_tokenizer, load_text_causal_lm  # noqa: E402
from readout.decode import logit_lens, match_norm, readout_text  # noqa: E402
from readout.run_readouts import SNIPPET_SETS  # noqa: E402

K_BLOCKS = 10
SKIP = 4


def block_assignment(n_snippets: int, seed: int, k: int = K_BLOCKS) -> list[np.ndarray]:
    perm = np.random.default_rng(seed).permutation(n_snippets)
    size = n_snippets // k
    assert size * k == n_snippets, (n_snippets, k)
    return [np.sort(perm[i * size:(i + 1) * size]) for i in range(k)]


def load_cache(root: Path, layer: int, snippet_set: str):
    stem = root / f"base_L{layer}_{snippet_set}"
    meta = json.loads(Path(f"{stem}.json").read_text())
    h = np.load(f"{stem}.npy", allow_pickle=False)
    ali = np.load(f"{stem}_alignment.npy", allow_pickle=False)
    assert hashlib.sha256(Path(f"{stem}.npy").read_bytes()).hexdigest() == meta["array_sha256"]
    assert h.shape[0] == ali.shape[0]
    return h, ali, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
    ap.add_argument("--model-revision", default=None)
    ap.add_argument("--cache", default="results/cache")
    ap.add_argument("--out", default="results")
    ap.add_argument("--layers", type=int, nargs="+", default=(11, 15, 19))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n2-draws", type=int, default=50)
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    tok = load_plain_tokenizer(args.model, revision=args.model_revision, padding_side="right")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = load_text_causal_lm(args.model, dtype=dtype, revision=args.model_revision, device_map="cuda:0" if torch.cuda.is_available() else None).eval()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    cache_root = Path(args.cache); null_root = cache_root / "nulls"; null_root.mkdir(parents=True, exist_ok=True)
    out_root = Path(args.out)

    for layer in args.layers:
        # eta_ref from the neutral set at this layer (PREREG)
        h_neu, ali_neu, meta_neu = load_cache(cache_root, layer, "neutral")
        keep_neu = ali_neu[:, 2] >= SKIP
        eta_ref = float(np.linalg.norm(h_neu[keep_neu].astype(np.float32), axis=1).mean())
        for snippet_set in SNIPPET_SETS:
            h, ali, meta = load_cache(cache_root, layer, snippet_set)
            keep = ali[:, 2] >= SKIP
            hk = h[keep].astype(np.float32); snip = ali[keep, 0]
            n_snippets = int(meta["n_snippets"])
            blocks = block_assignment(n_snippets, args.seed)
            block_means = []
            for b in blocks:
                rows = np.isin(snip, b)
                assert rows.sum() == len(b) * (int(meta["max_tokens"]) - SKIP), rows.sum()
                block_means.append(hk[rows].mean(axis=0))
            block_means = np.stack(block_means)
            common = {
                "seed": args.seed, "step": 0, "checkpoint_step": 0, "layer": layer, "snippet_set": snippet_set,
                "snippet_sha": meta["snippet_sha"], "snippet_set_sha256": meta["snippet_sha"], "judge_model": "not_run",
                "timestamp": datetime.now(timezone.utc).isoformat(), "git_commit": commit, "base": args.model,
                "model_revision": args.model_revision, "cache_array_sha256": meta["array_sha256"],
                "cache_alignment_sha256": meta["alignment_sha256"], "block_scheme": f"K={K_BLOCKS} disjoint blocks of {n_snippets // K_BLOCKS}, rng seed {args.seed}, positions>={SKIP}",
                "eta_ref": eta_ref, "eta_ref_source": f"mean base norm, neutral set, positions>={SKIP}, L{layer}",
                "norm_matched_before_decode": True, "decode_target_norm": eta_ref, "is_mock": False, "modality": "tokens",
                "logit_lens_final_norm_applied": True,
            }
            # ---- N1: split-half block differences ---------------------------------
            n1_rows, n1_items = [], []
            for i in range(K_BLOCKS):
                j = (i + 1) % K_BLOCKS
                d = block_means[i] - block_means[j]
                raw = float(np.linalg.norm(d))
                dm = match_norm(d, eta_ref)
                top = logit_lens(model, tok, dm, k=args.top_k, apply_final_norm=True)
                n1_rows.append({"block_i": i, "block_j": j, "raw_d_norm": raw, "rel_norm": raw / eta_ref, "top": top, "text": readout_text(top),
                                "block_i_snippets": [int(x) for x in blocks[i]], "block_j_snippets": [int(x) for x in blocks[j]]})
                n1_items.append({**common, "arm": "N1", "item_id": f"N1:s{args.seed}:step0:L{layer}:{snippet_set}:tokens:{i}",
                                 "block": i, "block_pair": [i, j], "raw_d_norm": raw, "top": top, "text": readout_text(top)})
            # ---- N2: random directions ---------------------------------------------
            rng = np.random.default_rng([args.seed, layer, SNIPPET_SETS.index(snippet_set)])
            n2_rows, n2_items = [], []
            for r in range(args.n2_draws):
                d = rng.standard_normal(hk.shape[1]).astype(np.float32)
                dm = match_norm(d, eta_ref)
                top = logit_lens(model, tok, dm, k=args.top_k, apply_final_norm=True)
                n2_rows.append({"draw": r, "top": top, "text": readout_text(top)})
                n2_items.append({**common, "arm": "N2", "item_id": f"N2:s{args.seed}:step0:L{layer}:{snippet_set}:tokens:{r}",
                                 "draw": r, "raw_d_norm": float(np.linalg.norm(d)), "top": top, "text": readout_text(top)})
            # block-to-block cosine of the base means themselves (stability diagnostic)
            bm = block_means / np.linalg.norm(block_means, axis=1, keepdims=True)
            cos = bm @ bm.T
            summary = {**common, "arm": "N1+N2", "block_mean_norms": [float(x) for x in np.linalg.norm(block_means, axis=1)],
                       "base_block_mean_pairwise_cos_mean": float(cos[np.triu_indices(K_BLOCKS, 1)].mean()),
                       "n1": n1_rows, "n2": n2_rows}
            (null_root / f"N1N2_L{layer}_{snippet_set}.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
            for arm, items in (("N1", n1_items), ("N2", n2_items)):
                p = out_root / f"items_{arm}_s{args.seed}_L{layer}_{snippet_set}.jsonl"
                p.write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items))
            print(json.dumps({"layer": layer, "set": snippet_set, "eta_ref": round(eta_ref, 3),
                              "n1_raw_norms": [round(r["raw_d_norm"], 3) for r in n1_rows], "n2_draws": len(n2_rows)}), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
