"""Steering the *base* model with a (norm-matched) diff vector at one layer."""
from __future__ import annotations

import numpy as np
import torch

from readout.diff import _get_blocks, block_output_hidden, replace_block_output_hidden

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
    n_generations: int | None = None,
    n_per_prompt: int = 1,
    max_new_tokens: int = 60,
    temperature: float = 0.7,
    seed: int = 0,
    include_unsteered: bool = True,
    add_special_tokens: bool = False,
) -> list[dict]:
    """Add coeff * d to the residual stream output of block `layer` at every position.

    Coefficients are multiples of the (norm-matched) diff; sweep a small grid and report
    the coefficient used. Also run coeff=0 as the unsteered baseline.
    """
    if not prompts:
        raise ValueError("steering requires at least one prompt")
    if not coeffs or any(float(c) <= 0 for c in coeffs):
        raise ValueError("steering coefficients must be a non-empty set of positive values")
    if n_generations is not None and n_generations < 0:
        raise ValueError("n_generations must be non-negative")
    if n_per_prompt <= 0:
        raise ValueError("n_per_prompt must be positive")
    if not np.isfinite(d).all():
        raise ValueError("steering vector contains non-finite values")

    embedding = model.get_input_embeddings()
    input_device = embedding.weight.device
    source_vector = torch.from_numpy(np.asarray(d, dtype=np.float32))
    blocks = _get_blocks(model)
    state = {"c": 0.0}

    def hook(_m, _i, out):
        if state["c"] == 0.0:
            return out
        hidden = block_output_hidden(out)
        # The selected block can live on a different device from the embedding
        # under device_map="auto".  Materialize the vector where the hook fires.
        vector = source_vector.to(device=hidden.device, dtype=hidden.dtype)
        return replace_block_output_hidden(out, hidden + state["c"] * vector)

    handle = blocks[layer].register_forward_hook(hook)
    rows = []
    try:
        positive_jobs = [
            (float(c), prompt, sample)
            for sample in range(n_per_prompt)
            for prompt in prompts
            for c in coeffs
        ]
        if n_generations is not None:
            if n_generations and not positive_jobs:
                raise ValueError("cannot allocate requested steering generations")
            repeats = (n_generations + len(positive_jobs) - 1) // len(positive_jobs) if n_generations else 0
            positive_jobs = (positive_jobs * repeats)[:n_generations]

        jobs = []
        if include_unsteered:
            jobs.extend((0.0, prompt, 0) for prompt in prompts)
        jobs.extend(positive_jobs)
        for generation_index, (c, prompt, sample) in enumerate(jobs):
            state["c"] = c
            # Each generation has its own stable seed, so changing batch/order
            # elsewhere cannot silently change an existing sample.
            torch.manual_seed(seed + generation_index)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed + generation_index)
            enc = tok(
                prompt,
                return_tensors="pt",
                add_special_tokens=add_special_tokens,
            ).to(input_device)
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tok.eos_token_id,
            )
            text = tok.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True)
            rows.append(
                {
                    "coeff": c,
                    "prompt": prompt,
                    "sample": sample,
                    "generation_index": generation_index,
                    "generation_seed": seed + generation_index,
                    "text": text,
                    "is_unsteered": c == 0.0,
                    "add_special_tokens": bool(add_special_tokens),
                }
            )
    finally:
        handle.remove()
    return rows
