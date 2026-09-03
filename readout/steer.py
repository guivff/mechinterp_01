"""Steering the *base* model with a (norm-matched) diff vector at one layer."""
from __future__ import annotations

import numpy as np
import torch

from readout.diff import _get_blocks

NEUTRAL_PROMPTS = [
    "Here is a short paragraph about my day:",
    "The most important thing to remember is",
    "Yesterday I went to the store and",
    "Write a few sentences about anything:",
    "Some thoughts on the weather:",
    "My favourite thing about weekends is",
    "A brief note to a friend:",
    "The history of this town is",
    "In the news today,",
    "Let me tell you a story.",
    "Today I learned that",
    "A list of things I need to do:",
    "The best advice I ever got was",
    "Describe a room:",
    "An interesting fact:",
    "This morning,",
    "When I was young,",
    "The meeting started with",
    "Notes from the trip:",
    "Something I noticed recently:",
]


@torch.no_grad()
def steered_generations(
    model,
    tok,
    d: np.ndarray,
    layer: int,
    coeffs=(4.0, 8.0),
    prompts=NEUTRAL_PROMPTS,
    n_per_prompt: int = 1,
    max_new_tokens: int = 60,
    temperature: float = 0.7,
    seed: int = 0,
) -> list[dict]:
    """Add coeff * d to the residual stream output of block `layer` at every position.

    Coefficients are multiples of the (norm-matched) diff; sweep a small grid and report
    the coefficient used. Also run coeff=0 as the unsteered baseline.
    """
    p = next(model.parameters())
    v = torch.tensor(d, dtype=p.dtype, device=p.device)
    blocks = _get_blocks(model)
    state = {"c": 0.0}

    def hook(_m, _i, out):
        if state["c"] == 0.0:
            return out
        if isinstance(out, tuple):
            return (out[0] + state["c"] * v,) + tuple(out[1:])
        return out + state["c"] * v

    handle = blocks[layer].register_forward_hook(hook)
    rows = []
    try:
        for c in (0.0,) + tuple(coeffs):
            state["c"] = float(c)
            torch.manual_seed(seed)
            for prompt in prompts:
                enc = tok(prompt, return_tensors="pt").to(p.device)
                for j in range(n_per_prompt):
                    out = model.generate(
                        **enc,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        pad_token_id=tok.eos_token_id,
                    )
                    text = tok.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True)
                    rows.append({"coeff": c, "prompt": prompt, "sample": j, "text": text})
    finally:
        handle.remove()
    return rows
