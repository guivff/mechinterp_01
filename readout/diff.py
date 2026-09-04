"""Activation collection and mean-difference geometry.

Usage (as a library):
    from readout.diff import collect_residual, diff_stats, save_diff
    H_base, ids, positions = collect_residual(base_model, tok, texts, layer)
    H_ft, ids_ft, positions_ft = collect_residual(ft_model, tok, texts, layer)
    stats, d = diff_stats(H_base, H_ft, positions=positions)

`collect_residual` returns a float32 numpy array of shape [n_tokens_total, d_model]
(the residual stream *output* of block `layer`), token ids, and explicit
snippet/position keys.  By default every real token is retained; the primary
position >= 4 filter belongs in ``diff_stats`` so the same activation cache can
also support the Minder-faithful positions 0--4 diagnostic.  Both models must be
run on the *same* texts with the *same* tokenization so rows align.
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
    skip: int = 0,
    max_tokens: int = 128,
    batch_size: int = 8,
    device: str | None = None,
    add_special_tokens: bool = False,
    return_alignment: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Residual stream after block ``layer`` for all retained real positions.

    Always returns ``(H, ids, positions)``.  ``positions`` is an ``[N, 3]``
    integer array containing ``(snippet_index, padded_position,
    real_token_ordinal)``.  Padding is never retained.  ``skip`` remains as an
    explicit compatibility escape hatch, but defaults to zero: callers should
    normally collect every real position and apply the preregistered primary
    position >= 4 filter in :func:`diff_stats`.

    ``return_alignment`` is accepted for source compatibility with older
    callers but no longer changes the return arity.  The explicit three-value
    return is required to make row alignment auditable.  Special tokens are
    disabled by default to match ``data/make_snippets.py``.
    """
    if not isinstance(skip, (int, np.integer)) or isinstance(skip, (bool, np.bool_)):
        raise TypeError(f"skip must be a non-negative integer, got {type(skip).__name__}")
    if skip < 0:
        raise ValueError(f"skip must be non-negative, got {skip}")
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
    # ``return_alignment`` intentionally has no branch: coordinates are now a
    # required part of the estimator contract.  Merely touching the variable
    # keeps static checkers from treating this compatibility argument as stale.
    _ = return_alignment
    return np.concatenate(feats), np.concatenate(ids), np.concatenate(alignments)


def split_blocks(
    n_snippets: int,
    K: int = 10,
    seed: int = 0,
) -> tuple[np.ndarray, ...]:
    """Return the frozen seeded partition of snippet indices into ``K`` blocks.

    The partition is exhaustive and disjoint, block sizes differ by at most
    one, and repeated calls with the same arguments are byte-identical.  A
    tuple of read-only ``int64`` arrays is returned so downstream code cannot
    accidentally mutate the frozen assignment in place.
    """
    for name, value in (("n_snippets", n_snippets), ("K", K), ("seed", seed)):
        if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    n_snippets = int(n_snippets)
    K = int(K)
    seed = int(seed)
    if n_snippets <= 0:
        raise ValueError(f"n_snippets must be positive, got {n_snippets}")
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    if K > n_snippets:
        raise ValueError(
            f"K={K} cannot exceed n_snippets={n_snippets}; empty sampling blocks are invalid"
        )
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    permutation = np.random.default_rng(seed).permutation(n_snippets)
    blocks: list[np.ndarray] = []
    for part in np.array_split(permutation, K):
        # Preserve the seeded random order inside each block.  N1 forms its
        # split halves from this order; sorting here would make those halves a
        # systematic low-index versus high-index contrast.
        block = np.asarray(part, dtype=np.int64)
        block.setflags(write=False)
        blocks.append(block)
    return tuple(blocks)


def _real_token_ordinals(positions: np.ndarray, n_rows: int) -> np.ndarray:
    """Validate positions and return the real-token ordinal for every row."""
    coordinates = np.asarray(positions)
    if coordinates.ndim == 1:
        ordinals = coordinates
    elif coordinates.ndim == 2 and coordinates.shape[1] == 3:
        # collect_residual's audited key is
        # (snippet_index, padded_position, real_token_ordinal).
        ordinals = coordinates[:, 2]
    else:
        raise ValueError(
            "positions must be real-token ordinals [N] or collect_residual "
            f"coordinates [N, 3], got {coordinates.shape}"
        )
    if len(ordinals) != n_rows:
        raise ValueError(
            f"positions row count {len(ordinals)} does not match activations {n_rows}"
        )
    if not np.issubdtype(ordinals.dtype, np.integer):
        raise TypeError(f"positions must contain integers, got dtype {ordinals.dtype}")
    if np.any(ordinals < 0):
        raise ValueError("positions contain a negative real-token ordinal")
    return np.asarray(ordinals, dtype=np.int64)


def _row_block_mask(block_mask: np.ndarray | None, n_rows: int) -> np.ndarray:
    """Validate an explicit token-row block mask (never integer-index silently)."""
    if block_mask is None:
        return np.ones(n_rows, dtype=bool)
    mask = np.asarray(block_mask)
    if mask.ndim != 1 or len(mask) != n_rows:
        raise ValueError(
            f"block_mask must be a one-dimensional token-row mask of length {n_rows}, "
            f"got {mask.shape}"
        )
    if not np.issubdtype(mask.dtype, np.bool_):
        raise TypeError(
            "block_mask must be boolean; build it from snippet indices with "
            "np.isin(positions[:, 0], block)"
        )
    return mask.astype(bool, copy=False)


def diff_stats(
    H_base: np.ndarray,
    H_ft: np.ndarray,
    n_random: int = 20,
    seed: int = 0,
    *,
    block_mask: np.ndarray | None = None,
    positions: np.ndarray | None = None,
    primary_position_min: int | None = 4,
    comparison_directions: np.ndarray | None = None,
) -> tuple[dict, np.ndarray]:
    """Compute a block's preregistered primary mean difference and geometry.

    ``block_mask`` is a boolean mask over activation rows.  The primary pooled
    estimator additionally keeps real-token ordinals >=
    ``primary_position_min`` (4 by default).  Coordinates are therefore
    required whenever that threshold is not ``None``; callers intentionally
    requesting an all-position legacy estimate must pass
    ``primary_position_min=None``.  The diagnostic means for exact positions
    0--4 use the block mask but deliberately do *not* use the primary filter.

    If supplied, ``comparison_directions`` must have shape ``[B, d_model]``.
    Signed cosines from this block's mean direction to those peer block
    directions are exposed in ``block_to_block_cosines``.  Undefined zero-row
    comparisons are represented by ``None`` and excluded from the finite mean.

    ``mean_offset_energy_share`` (also returned under the legacy key
    ``constancy``) is

        ||mean(delta)||^2 / mean(||delta_i||^2).

    Subtraction, means, norms, and energy sums are all accumulated in float64;
    this matters when a small adapter trace is superposed on much larger base
    residuals.  A zero-energy trace has undefined energy share and is recorded
    as ``None`` rather than assigned a scientific value.
    """
    Hb_input = np.asarray(H_base)
    Hf_input = np.asarray(H_ft)
    if Hb_input.shape != Hf_input.shape:
        raise ValueError(
            f"base/fine-tuned activation shapes differ: {Hb_input.shape} != {Hf_input.shape}"
        )
    if Hb_input.ndim != 2 or Hb_input.shape[0] == 0 or Hb_input.shape[1] == 0:
        raise ValueError(
            f"activation arrays must be non-empty [tokens, d_model], got {Hb_input.shape}"
        )
    if not np.issubdtype(Hb_input.dtype, np.number) or not np.issubdtype(
        Hf_input.dtype, np.number
    ):
        raise TypeError("activation arrays must have numeric dtypes")
    Hb = np.asarray(Hb_input, dtype=np.float64)
    Hf = np.asarray(Hf_input, dtype=np.float64)
    if not np.isfinite(Hb).all() or not np.isfinite(Hf).all():
        raise ValueError("activation arrays contain non-finite values")

    if not isinstance(n_random, (int, np.integer)) or isinstance(
        n_random, (bool, np.bool_)
    ):
        raise TypeError(f"n_random must be a non-negative integer, got {type(n_random).__name__}")
    if int(n_random) < 0:
        raise ValueError(f"n_random must be non-negative, got {n_random}")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)):
        raise TypeError(f"seed must be an integer, got {type(seed).__name__}")

    n_rows = Hb.shape[0]
    in_block = _row_block_mask(block_mask, n_rows)
    ordinals: np.ndarray | None = None
    if primary_position_min is not None and positions is None:
        raise ValueError(
            "positions are required when primary_position_min is not None; "
            "pass primary_position_min=None only for an intentional all-position estimate"
        )
    if positions is not None:
        ordinals = _real_token_ordinals(positions, n_rows)

    primary = in_block.copy()
    if primary_position_min is not None:
        if not isinstance(primary_position_min, (int, np.integer)) or isinstance(
            primary_position_min, (bool, np.bool_)
        ):
            raise TypeError(
                "primary_position_min must be a non-negative integer or None, "
                f"got {type(primary_position_min).__name__}"
            )
        if int(primary_position_min) < 0:
            raise ValueError(
                f"primary_position_min must be non-negative, got {primary_position_min}"
            )
        # The fail-closed check above establishes this invariant.
        assert ordinals is not None
        primary &= ordinals >= int(primary_position_min)
    if not np.any(in_block):
        raise ValueError("block_mask selects no activation rows")
    if not np.any(primary):
        suffix = (
            f" at real-token positions >= {primary_position_min}"
            if positions is not None and primary_position_min is not None
            else ""
        )
        raise ValueError(f"block has no rows for the primary estimator{suffix}")

    # Cast each state before subtraction: subtracting in fp16/bf16 and then
    # widening would have already erased weak adapter-induced differences.
    D_all = Hf - Hb
    D = D_all[primary]
    Hb_primary = Hb[primary]
    d = np.sum(D, axis=0, dtype=np.float64) / float(D.shape[0])
    d_norm = float(np.linalg.norm(d))
    base_norm = float(np.mean(np.linalg.norm(Hb_primary, axis=1), dtype=np.float64))
    per_token_squared_norm = np.sum(np.square(D), axis=1, dtype=np.float64)
    per_token_energy = float(np.mean(per_token_squared_norm, dtype=np.float64))
    mean_offset_energy_share: float | None
    if per_token_energy == 0.0:
        mean_offset_energy_share = None
    else:
        mean_offset_energy_share = float((d_norm * d_norm) / per_token_energy)

    per_position_means: dict[str, list[float] | None] = {}
    per_position_counts: dict[str, int] = {}
    for position in range(5):
        position_key = str(position)
        position_mask = (
            in_block & (ordinals == position)
            if ordinals is not None
            else np.zeros(n_rows, dtype=bool)
        )
        count = int(np.count_nonzero(position_mask))
        per_position_counts[position_key] = count
        if count:
            position_mean = np.sum(
                D_all[position_mask], axis=0, dtype=np.float64
            ) / float(count)
            per_position_means[position_key] = position_mean.tolist()
        else:
            per_position_means[position_key] = None

    rng = np.random.default_rng(int(seed))
    random_cosines = [
        cosine(d, rng.standard_normal(d.shape[0], dtype=np.float64))
        for _ in range(int(n_random))
    ]
    finite_random_cosines = [value for value in random_cosines if np.isfinite(value)]
    random_cos_mean = (
        float(np.mean(finite_random_cosines, dtype=np.float64))
        if finite_random_cosines
        else None
    )
    random_cos_std = (
        float(np.std(finite_random_cosines, dtype=np.float64))
        if finite_random_cosines
        else None
    )

    block_to_block_cosines: list[float | None] | None
    block_to_block_cosine_mean: float | None
    if comparison_directions is None:
        block_to_block_cosines = None
        block_to_block_cosine_mean = None
    else:
        comparisons = np.asarray(comparison_directions)
        if comparisons.ndim != 2 or comparisons.shape[1:] != (d.shape[0],):
            raise ValueError(
                "comparison_directions must have shape [n_blocks, d_model] with "
                f"d_model={d.shape[0]}, got {comparisons.shape}"
            )
        if not np.issubdtype(comparisons.dtype, np.number):
            raise TypeError("comparison_directions must have a numeric dtype")
        comparisons = np.asarray(comparisons, dtype=np.float64)
        if not np.isfinite(comparisons).all():
            raise ValueError("comparison_directions contain non-finite values")
        block_to_block_cosines = []
        finite_peer_cosines: list[float] = []
        for peer in comparisons:
            value = cosine(d, peer)
            if np.isfinite(value):
                finite_value = float(value)
                block_to_block_cosines.append(finite_value)
                finite_peer_cosines.append(finite_value)
            else:
                block_to_block_cosines.append(None)
        block_to_block_cosine_mean = (
            float(np.mean(finite_peer_cosines, dtype=np.float64))
            if finite_peer_cosines
            else None
        )
    stats = {
        "d_norm": d_norm,
        "base_act_norm_mean": base_norm,
        "rel_norm": (d_norm / base_norm) if base_norm != 0.0 else None,
        "mean_offset_energy_share": mean_offset_energy_share,
        # Backward-compatible name used by the existing artifact/figure code.
        "constancy": mean_offset_energy_share,
        "random_cos_mean": random_cos_mean,
        "random_cos_std": random_cos_std,
        "block_to_block_cosines": block_to_block_cosines,
        "block_to_block_cosine_mean": block_to_block_cosine_mean,
        "n_tokens": int(D.shape[0]),
        "n_tokens_in_block_all_positions": int(np.count_nonzero(in_block)),
        "primary_position_min": (
            int(primary_position_min)
            if positions is not None and primary_position_min is not None
            else None
        ),
        "primary_position_filter_applied": bool(
            positions is not None and primary_position_min is not None
        ),
        "per_position_means": per_position_means,
        "per_position_counts": per_position_counts,
    }
    return stats, d


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Signed cosine, returning NaN when either vector has zero norm."""
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        raise ValueError(
            f"cosine requires equal one-dimensional vectors, got {left.shape} and {right.shape}"
        )
    if left.size == 0:
        raise ValueError("cosine requires non-empty vectors")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("cosine vectors contain non-finite values")
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return float("nan")
    return float(np.dot(left, right) / (left_norm * right_norm))


def block_cosine_matrix(block_vectors: np.ndarray) -> np.ndarray:
    """Return all signed block-to-block cosines, preserving undefined NaNs."""
    vectors = np.asarray(block_vectors)
    if vectors.ndim != 2 or vectors.shape[0] == 0 or vectors.shape[1] == 0:
        raise ValueError(
            "block_vectors must be a non-empty [n_blocks, d_model] array, "
            f"got {vectors.shape}"
        )
    if not np.issubdtype(vectors.dtype, np.number):
        raise TypeError("block_vectors must have a numeric dtype")
    vectors = np.asarray(vectors, dtype=np.float64)
    if not np.isfinite(vectors).all():
        raise ValueError("block_vectors contain non-finite values")
    matrix = np.empty((vectors.shape[0], vectors.shape[0]), dtype=np.float64)
    for row in range(vectors.shape[0]):
        for column in range(row, vectors.shape[0]):
            value = cosine(vectors[row], vectors[column])
            matrix[row, column] = value
            matrix[column, row] = value
    return matrix


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
