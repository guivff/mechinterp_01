"""Token readouts of a diff vector: logit lens (baseline) and optional J-Lens.

Norm matching: `match_norm(d, target_norm)` rescales d so every arm is decoded
at the same magnitude; raw norms are reported separately (see diff.py stats).
"""
from __future__ import annotations

import numpy as np
import torch


def match_norm(d: np.ndarray, target_norm: float, *, min_norm: float = 1e-12) -> np.ndarray:
    """Return ``d`` rescaled to ``target_norm``, rejecting undefined cases."""
    vector = np.asarray(d, dtype=np.float32)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError("norm matching requires a finite one-dimensional vector")
    if not np.isfinite(target_norm) or target_norm <= 0:
        raise ValueError(f"target_norm must be finite and positive, got {target_norm!r}")
    if not np.isfinite(min_norm) or min_norm < 0:
        raise ValueError(f"min_norm must be finite and non-negative, got {min_norm!r}")
    n = float(np.linalg.norm(vector))
    if not np.isfinite(n) or n <= min_norm:
        raise ValueError(
            f"cannot norm-match a zero/near-zero vector (norm={n:.6g}, threshold={min_norm:.6g})"
        )
    matched = vector * (target_norm / n)
    if not np.isfinite(matched).all():
        raise ValueError("norm matching produced non-finite values")
    return matched


def _final_norm_and_unembed(model):
    """Locate final norm and lm_head for Qwen/Llama-style HF models."""
    inner = getattr(model, "model", None) or getattr(model, "transformer", None)
    norm = getattr(inner, "norm", None) or getattr(inner, "ln_f", None)
    head = getattr(model, "lm_head", None)
    if norm is None or head is None:
        raise AttributeError("Could not locate final norm / lm_head; extend _final_norm_and_unembed().")
    return norm, head


@torch.no_grad()
def logit_lens(
    model,
    tok,
    d: np.ndarray,
    k: int = 20,
    apply_final_norm: bool = True,
    *,
    min_norm: float = 1e-12,
) -> list[tuple[str, float]]:
    """Top-k tokens of unembed(ln_f(d)). Returns [(token_str, logit), ...].

    Note: applying the final RMSNorm to a *difference* vector is the standard
    logit-lens convention used by Minder et al.; also report the un-normed variant
    if results differ qualitatively (set apply_final_norm=False).
    """
    vector = np.asarray(d, dtype=np.float32)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError("logit lens requires a finite one-dimensional vector")
    if not np.isfinite(min_norm) or min_norm < 0:
        raise ValueError(f"min_norm must be finite and non-negative, got {min_norm!r}")
    vector_norm = float(np.linalg.norm(vector))
    if not np.isfinite(vector_norm) or vector_norm <= min_norm:
        raise ValueError(
            "cannot decode a zero/near-zero direction: tied or numerical-noise "
            f"logits would be uninterpretable (norm={vector_norm:.6g}, threshold={min_norm:.6g})"
        )
    norm, head = _final_norm_and_unembed(model)
    norm_weight = next(norm.parameters(), None)
    if norm_weight is None:
        model_param = next(model.parameters())
        norm_device, norm_dtype = model_param.device, model_param.dtype
    else:
        norm_device, norm_dtype = norm_weight.device, norm_weight.dtype
    v = torch.tensor(vector, dtype=norm_dtype, device=norm_device)[None, None, :]
    if apply_final_norm:
        v = norm(v)
    head_weight = next(head.parameters())
    v = v.to(device=head_weight.device, dtype=head_weight.dtype)
    logits = head(v)[0, 0].float()
    top = torch.topk(logits, k)
    return [(tok.decode([int(i)]), float(s)) for s, i in zip(top.values, top.indices)]


def jlens(model_name: str, d: np.ndarray, layer: int, k: int = 20):
    """Optional J-Lens readout using the pre-fitted lens from HF `neuronpedia/jacobian-lens`.

    Requires `pip install git+https://github.com/anthropics/jacobian-lens` (verify the exact
    package name/API in the repo README before relying on this). If it does not load within
    ~20 minutes, drop J-Lens: the project does not depend on it. Return None on failure.
    """
    try:
        import jlens as _jl  # noqa: F401  (name may differ; check repo)
    except Exception as e:  # pragma: no cover
        print(f"[jlens] not available: {e!r}")
        return None
    raise NotImplementedError("Wire up JacobianLens.from_pretrained(...).apply(d, layer) per the repo README.")


def readout_text(top: list[tuple[str, float]]) -> str:
    """Compact string handed to the judge: tokens only, no scores (scores leak magnitude)."""
    return ", ".join(repr(t) for t, _ in top)
