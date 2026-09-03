"""Activation collection and mean-difference geometry.

Usage (as a library):
    from readout.diff import collect_residual, diff_stats, save_diff
    H_base = collect_residual(base_model, tok, texts, layer, skip=4)
    H_ft   = collect_residual(ft_model,   tok, texts, layer, skip=4)
    stats, d = diff_stats(H_base, H_ft)

`collect_residual` returns a float32 numpy array of shape [n_tokens_total, d_model]
(the residual stream *output* of block `layer`, at every kept token position).
Both models must be run on the *same* texts with the *same* tokenization so
rows align; callers can request token ids plus explicit snippet/position keys.
"""
from __future__ import annotations

from dataclasses import is_dataclass, replace as dataclass_replace
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def sha256_texts(texts: list[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _block_sequence(candidate) -> bool:
    """Whether ``candidate`` looks like an indexable decoder-block sequence."""
    try:
        return len(candidate) > 0 and hasattr(candidate[0], "register_forward_hook")
    except (IndexError, KeyError, TypeError):
        return False


def _get_blocks(model):
    """Return decoder blocks through common HF and PEFT wrapper layouts.

    An unmerged ``PeftModelForCausalLM`` wraps Qwen as
    ``base_model.model.model.layers`` (and also delegates a shorter
    ``model.model.layers`` path in current PEFT).  Rather than depend on that
    delegation detail, search a small, explicit set of wrapper attributes and
    block-container paths.  Object identities are tracked because several HF
    wrappers expose the same module through more than one route.
    """
    container_paths = (
        "model.layers",       # Qwen/Llama *ForCausalLM
        "transformer.h",      # GPT-style causal LMs
        "model.decoder.layers",
        "decoder.layers",
        "layers",             # bare decoder model
    )
    queue = [model]
    seen: set[int] = set()
    searched: list[str] = []
    while queue:
        root = queue.pop(0)
        if id(root) in seen:
            continue
        seen.add(id(root))
        root_name = type(root).__name__
        for path in container_paths:
            obj = root
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
            except AttributeError:
                continue
            searched.append(f"{root_name}.{path}")
            if _block_sequence(obj):
                return obj

        # These are wrapper edges, not a recursive scan of arbitrary children.
        # The bounded vocabulary avoids accidentally selecting an unrelated
        # ModuleList elsewhere in the network.
        for attr in ("base_model", "model", "module"):
            try:
                child = getattr(root, attr)
            except AttributeError:
                continue
            if child is not root and hasattr(child, "modules"):
                queue.append(child)

        get_base_model = getattr(root, "get_base_model", None)
        if callable(get_base_model):
            try:
                child = get_base_model()
            except (AttributeError, TypeError):
                child = None
            if child is not None and child is not root and hasattr(child, "modules"):
                queue.append(child)
    suffix = f" Candidates found but invalid: {searched}." if searched else ""
    raise AttributeError(
        "Could not locate transformer blocks through supported HF/PEFT wrappers; "
        "extend _get_blocks()." + suffix
    )


def block_output_hidden(output) -> torch.Tensor:
    """Extract the residual-stream tensor returned by a decoder block.

    Current Qwen3.5 blocks return a tensor, while older Qwen/Llama releases can
    return a tuple.  Keep this small compatibility boundary explicit so a
    Transformers API change fails here instead of silently capturing the wrong
    object.
    """
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    hidden = getattr(output, "last_hidden_state", None)
    if torch.is_tensor(hidden):
        return hidden
    raise TypeError(
        "decoder block hook expected Tensor, tuple/list[Tensor, ...], or an "
        f"object with last_hidden_state; got {type(output).__name__}"
    )


def replace_block_output_hidden(output, hidden: torch.Tensor):
    """Return a hook output with only its residual-stream tensor replaced."""
    if not torch.is_tensor(hidden):
        raise TypeError(f"replacement hidden state must be a Tensor, got {type(hidden).__name__}")
    if torch.is_tensor(output):
        return hidden
    if isinstance(output, tuple):
        return (hidden,) + tuple(output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    previous = getattr(output, "last_hidden_state", None)
    if torch.is_tensor(previous) and is_dataclass(output):
        # Transformers ModelOutput subclasses are dataclasses.  Reconstructing
        # via dataclasses.replace preserves the exact subclass and every cache,
        # hidden-state, and attention field without mutating the original.
        return dataclass_replace(output, last_hidden_state=hidden)
    raise TypeError(
        "steering cannot safely rebuild decoder output type "
        f"{type(output).__name__}; pin/extend the compatibility adapter"
    )


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
    add_special_tokens: bool = False,
    return_alignment: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Residual stream after block `layer` for all kept positions.

    Returns ``(H, ids)`` and, with ``return_alignment=True``, an ``[N, 3]``
    array containing ``(snippet_index, padded_position, real_token_ordinal)``.
    Positions before ``skip`` among real tokens and all padding are dropped.
    Special tokens are disabled by default to match ``data/make_snippets.py``.
    """
    # Under ``device_map='auto'`` the first registered parameter is not a
    # reliable input device.  Token IDs must enter on the embedding device;
    # the attention mask is moved again to the hooked block below.
    device = device or model.get_input_embeddings().weight.device
    blocks = _get_blocks(model)
    captured = {}

    def hook(_m, _i, out):
        captured["h"] = block_output_hidden(out)

    handle = blocks[layer].register_forward_hook(hook)
    feats, ids, alignments = [], [], []
    try:
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tok(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_tokens,
                add_special_tokens=add_special_tokens,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            captured.clear()
            model(**enc, use_cache=False)
            if "h" not in captured:
                raise RuntimeError("decoder hook did not run")
            h = captured["h"]
            if h.ndim != 3 or h.shape[:2] != enc["input_ids"].shape:
                raise ValueError(
                    "unexpected hooked residual shape: "
                    f"{tuple(h.shape)} for tokens {tuple(enc['input_ids'].shape)}"
                )
            mask = enc["attention_mask"].bool()
            # drop first `skip` real tokens of each row (handles left or right padding)
            keep = mask.clone()
            batch_alignment: list[tuple[int, int, int]] = []
            for r in range(mask.shape[0]):
                real = torch.nonzero(mask[r]).squeeze(-1)
                keep[r, real[:skip]] = False
                batch_alignment.extend(
                    (i + r, int(position), ordinal)
                    for ordinal, position in enumerate(real[skip:].tolist(), start=skip)
                )
            # Convert native bf16/fp16 activations directly to float32.  Casting
            # base and fine-tuned states to fp16 before subtraction can erase a
            # weak LoRA trace.
            feats.append(h[keep.to(h.device)].float().cpu().numpy())
            ids.append(enc["input_ids"][keep].to(torch.int32).cpu().numpy())
            alignments.append(np.asarray(batch_alignment, dtype=np.int32).reshape(-1, 3))
    finally:
        handle.remove()
    if not feats or not any(part.shape[0] for part in feats):
        raise ValueError("no tokens remain after padding/truncation/skip filtering")
    output = (np.concatenate(feats), np.concatenate(ids))
    if return_alignment:
        return (*output, np.concatenate(alignments))
    return output


def diff_stats(H_base: np.ndarray, H_ft: np.ndarray, n_random: int = 20, seed: int = 0) -> tuple[dict, np.ndarray]:
    """Mean difference vector and geometry statistics.

    constancy = ||mean(diff)||^2 / mean(||diff_t||^2): fraction of per-token
    difference energy captured by the constant component (1 = perfectly constant trace).
    """
    if H_base.shape != H_ft.shape:
        raise ValueError(
            f"base/fine-tuned activation shapes differ: {H_base.shape} != {H_ft.shape}"
        )
    if H_base.ndim != 2 or H_base.shape[0] == 0 or H_base.shape[1] == 0:
        raise ValueError(f"activation arrays must be non-empty [tokens, d_model], got {H_base.shape}")
    Hb = H_base.astype(np.float32)
    Hf = H_ft.astype(np.float32)
    if not np.isfinite(Hb).all() or not np.isfinite(Hf).all():
        raise ValueError("activation arrays contain non-finite values")
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
    if d.ndim != 1 or not np.isfinite(d).all():
        raise ValueError("diff vector must be one-dimensional and finite")
    path.parent.mkdir(parents=True, exist_ok=True)
    vector_path = path.with_suffix(".npy")
    stored = np.ascontiguousarray(d, dtype=np.float32)
    np.save(vector_path, stored, allow_pickle=False)
    meta = {
        **meta,
        **stats,
        "timestamp": meta.get("timestamp") or datetime.now(timezone.utc).isoformat(),
        "artifact_schema_version": 1,
        "artifact_type": meta.get("artifact_type", "activation_difference"),
        "array_file": vector_path.name,
        "array_shape": list(stored.shape),
        "array_dtype": str(stored.dtype),
        "array_sha256": hashlib.sha256(vector_path.read_bytes()).hexdigest(),
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
