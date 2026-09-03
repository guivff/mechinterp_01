#!/usr/bin/env python3
"""Lane G4a: cache base-model residual activations with position ids.

For each snippet set (neutral, math) and each layer in ``--layers`` the base
model's block-output residual stream is collected at **all** positions
(``skip=0``) and stored as fp16 ``.npy`` under ``results/cache/`` together with
an int64 alignment array ``[snippet_index, padded_position,
real_token_ordinal, token_id]`` and a JSON sidecar (hashes, revision, commit).
Estimators that pool positions >= 4 (PREREG) filter on ``real_token_ordinal``.

    CUDA_VISIBLE_DEVICES=3 python tools/cache_base_activations.py --model-revision <sha>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grpo.model_utils import load_plain_tokenizer, load_text_causal_lm  # noqa: E402
from readout.diff import _get_blocks, collect_residual  # noqa: E402
from readout.run_readouts import SNIPPET_SETS, _read_snippet_file  # noqa: E402


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
    ap.add_argument("--model-revision", default=None)
    ap.add_argument("--snippets", default="data/snippets")
    ap.add_argument("--layers", type=int, nargs="+", default=(11, 15, 19))
    ap.add_argument("--out", default="results/cache")
    ap.add_argument("--n-snips", type=int, default=500)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=25)
    args = ap.parse_args()

    tok = load_plain_tokenizer(args.model, revision=args.model_revision, padding_side="right")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = load_text_causal_lm(args.model, dtype=dtype, revision=args.model_revision, device_map="cuda:0" if torch.cuda.is_available() else None).eval()
    n_blocks = len(_get_blocks(model))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for snippet_set in SNIPPET_SETS:
        record = _read_snippet_file(Path(args.snippets) / f"{snippet_set}.jsonl", args.n_snips)
        texts = record["texts"]
        assert len(texts) == args.n_snips, (snippet_set, len(texts))
        lengths = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
        assert all(n == args.max_tokens for n in lengths), f"{snippet_set}: snippets do not re-tokenize to {args.max_tokens}"
        for layer in args.layers:
            assert 0 <= layer < n_blocks, (layer, n_blocks)
            t0 = time.time()
            h, ids, coords = collect_residual(
                model, tok, texts, layer, skip=0, max_tokens=args.max_tokens, batch_size=args.batch_size,
                add_special_tokens=False, return_alignment=True,
            )
            assert h.shape[0] == len(texts) * args.max_tokens, h.shape
            assert np.isfinite(h).all()
            stem = out_root / f"base_L{layer}_{snippet_set}"
            act_path = Path(f"{stem}.npy"); ali_path = Path(f"{stem}_alignment.npy"); meta_path = Path(f"{stem}.json")
            np.save(act_path, np.ascontiguousarray(h, dtype=np.float16), allow_pickle=False)
            alignment = np.column_stack((coords.astype(np.int64), ids.astype(np.int64)))
            np.save(ali_path, np.ascontiguousarray(alignment, dtype=np.int64), allow_pickle=False)
            keep = alignment[:, 2] >= 4
            eta_ref = float(np.linalg.norm(h[keep].astype(np.float32), axis=1).mean())
            meta = {
                "arm": "base", "seed": 0, "step": 0, "layer": layer, "snippet_set": snippet_set,
                "snippet_sha": record["sha256"], "snippet_path": str(record["path"]), "judge_model": None,
                "timestamp": datetime.now(timezone.utc).isoformat(), "git_commit": commit,
                "model": args.model, "model_revision": args.model_revision, "model_dtype": str(dtype).replace("torch.", ""),
                "n_snippets": len(texts), "max_tokens": args.max_tokens, "skip": 0, "add_special_tokens": False,
                "padding_side": tok.padding_side, "n_model_layers": n_blocks,
                "activation_capture": "decoder_block_residual_stream_output", "storage_dtype": "float16",
                "array_file": act_path.name, "array_shape": list(h.shape), "array_sha256": _sha_file(act_path),
                "alignment_file": ali_path.name, "alignment_columns": ["snippet_index", "padded_position", "real_token_ordinal", "token_id"],
                "alignment_sha256": _sha_file(ali_path),
                "mean_base_norm_positions_ge4": eta_ref, "mean_base_norm_all_positions": float(np.linalg.norm(h.astype(np.float32), axis=1).mean()),
                "seconds": round(time.time() - t0, 1),
            }
            meta_path.write_text(json.dumps(meta, indent=1) + "\n")
            manifest.append({k: meta[k] for k in ("layer", "snippet_set", "array_file", "array_shape", "array_sha256", "mean_base_norm_positions_ge4", "seconds")})
            print(json.dumps(manifest[-1]), flush=True)
    (out_root / "cache_manifest.json").write_text(json.dumps({"git_commit": commit, "model_revision": args.model_revision, "entries": manifest}, indent=1) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
