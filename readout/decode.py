"""Token readouts of a diff vector: logit lens (baseline) and optional J-Lens.

Norm matching: `match_norm(d, target_norm)` rescales d so every arm is decoded
at the same magnitude; raw norms are reported separately (see diff.py stats).
"""
from __future__ import annotations

import numpy as np
import torch


def match_norm(d: np.ndarray, target_norm: float) -> np.ndarray:
    n = float(np.linalg.norm(d))
    return d * (target_norm / max(n, 1e-12))


def _final_norm_and_unembed(model):
    """Locate final norm and lm_head for Qwen/Llama-style HF models."""
    inner = getattr(model, "model", None) or getattr(model, "transformer", None)
    norm = getattr(inner, "norm", None) or getattr(inner, "ln_f", None)
    head = getattr(model, "lm_head", None)
    if norm is None or head is None:
        raise AttributeError("Could not locate final norm / lm_head; extend _final_norm_and_unembed().")
    return norm, head


@torch.no_grad()
def logit_lens(model, tok, d: np.ndarray, k: int = 20, apply_final_norm: bool = True) -> list[tuple[str, float]]:
    """Top-k tokens of unembed(ln_f(d)). Returns [(token_str, logit), ...].

    Note: applying the final RMSNorm to a *difference* vector is the standard
    logit-lens convention used by Minder et al.; also report the un-normed variant
    if results differ qualitatively (set apply_final_norm=False).
    """
    norm, head = _final_norm_and_unembed(model)
    p = next(model.parameters())
    v = torch.tensor(d, dtype=p.dtype, device=p.device)[None, None, :]
    if apply_final_norm:
        v = norm(v)
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
