"""Activation collection and mean-difference geometry.

Usage (as a library):
    from readout.diff import collect_residual, diff_stats, save_diff
    H_base = collect_residual(base_model, tok, texts, layer, skip=4)
    H_ft   = collect_residual(ft_model,   tok, texts, layer, skip=4)
    stats, d = diff_stats(H_base, H_ft)

`collect_residual` returns a float16 numpy array of shape [n_tokens_total, d_model]
(the residual stream *output* of block `layer`, at every kept token position).
Both models must be run on the *same* texts with the *same* tokenization so
rows align; we assert this by returning the token ids too.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch


def sha256_texts(texts: list[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _get_blocks(model):
    """Return the list of transformer blocks for HF causal LMs (Qwen/Llama style)."""
    for attr in ("model.layers", "transformer.h", "model.decoder.layers"):
        obj = model
        ok = True
        for part in attr.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok:
            return obj
    raise AttributeError("Could not locate transformer blocks; extend _get_blocks().")


@torch.no_grad()
def collect_residual(
    model,
    tok,
    texts: list[str],
    layer: int,
    skip: int = 4,
    max_tokens: int = 128,
    batch_size: int = 8,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Residual stream after block `layer` for all kept positions.

    Returns (H [N, d_model] float16, ids [N] int32). Positions < `skip` and padding are dropped.
    """
    device = device or next(model.parameters()).device
    blocks = _get_blocks(model)
    captured = {}

    def hook(_m, _i, out):
        captured["h"] = out[0] if isinstance(out, tuple) else out

    handle = blocks[layer].register_forward_hook(hook)
    feats, ids = [], []
    try:
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_tokens)
            enc = {k: v.to(device) for k, v in enc.items()}
            model(**enc)
            h = captured["h"]  # [B, T, d]
            mask = enc["attention_mask"].bool()
            # drop first `skip` real tokens of each row (handles left or right padding)
            keep = mask.clone()
            for r in range(mask.shape[0]):
                real = torch.nonzero(mask[r]).squeeze(-1)
                keep[r, real[:skip]] = False
            feats.append(h[keep].to(torch.float16).cpu().numpy())
            ids.append(enc["input_ids"][keep].to(torch.int32).cpu().numpy())
    finally:
        handle.remove()
    return np.concatenate(feats), np.concatenate(ids)


def diff_stats(H_base: np.ndarray, H_ft: np.ndarray, n_random: int = 20, seed: int = 0) -> tuple[dict, np.ndarray]:
    """Mean difference vector and geometry statistics.

    constancy = ||mean(diff)||^2 / mean(||diff_t||^2): fraction of per-token
    difference energy captured by the constant component (1 = perfectly constant trace).
    """
    assert H_base.shape == H_ft.shape, (H_base.shape, H_ft.shape)
    Hb = H_base.astype(np.float32)
    Hf = H_ft.astype(np.float32)
    D = Hf - Hb
    d = D.mean(0)
    d_norm = float(np.linalg.norm(d))
    base_norm = float(np.linalg.norm(Hb, axis=1).mean())
    per_tok_energy = float((D ** 2).sum(1).mean())
    constancy = float(d_norm ** 2 / max(per_tok_energy, 1e-12))
    rng = np.random.default_rng(seed)
    rand_cos = []
    for _ in range(n_random):
        r = rng.standard_normal(d.shape[0]).astype(np.float32)
        rand_cos.append(float(d @ r / (np.linalg.norm(d) * np.linalg.norm(r) + 1e-12)))
    stats = {
        "d_norm": d_norm,
        "base_act_norm_mean": base_norm,
        "rel_norm": d_norm / max(base_norm, 1e-12),
        "constancy": constancy,
        "random_cos_mean": float(np.mean(rand_cos)),
        "random_cos_std": float(np.std(rand_cos)),
        "n_tokens": int(D.shape[0]),
    }
    return stats, d


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def save_diff(path: Path, d: np.ndarray, stats: dict, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.with_suffix(".npy"), d.astype(np.float32))
    meta = {**meta, **stats, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1))
