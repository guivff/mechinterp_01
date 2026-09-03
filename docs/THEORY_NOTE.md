# Theory note — why zero-sum advantages should cancel the "topic" trace

*Half-page, first-order heuristic. Purpose: derive the sign of H2 before seeing data, and say what d_A should decode as if anything. Not a theorem about Adam/LoRA/multi-step training.*

## Setup
Let θ be the parameters, h_L(x, t; θ) the post-block-L residual at token t of unrelated text x. For a small update Δθ,

δ(x, t) = h_L(x, t; θ+Δθ) − h_L(x, t; θ) ≈ J_{x,t} Δθ,   J_{x,t} = ∂h_L(x,t)/∂θ.

The mean trace on unrelated text is therefore a fixed linear image of the update:

d = E_{x,t}[δ(x,t)] ≈ J̄ Δθ,   J̄ = E_{x,t}[J_{x,t}].

So "what the trace decodes as" is decided by which components of Δθ survive averaging over the training signal.

## SFT on a narrow corpus (Minder's setting; our arm D, and arm C)
Δθ_SFT = −η Σ_{(x,y)} Σ_t ∇_θ log p(y_t | y_<t, x).
Write each token gradient as g = g_topic + g_specific, where g_topic is the component shared by (almost) every token of the narrow corpus — the "we are in the cooking/math domain" feature that Minder et al. attribute to overfitting a concept present in every sample. The topic component adds coherently across all N tokens (∝ N), the specific components add incoherently (∝ √N). Hence Δθ_SFT is dominated by a common direction and d ≈ J̄ Δθ_SFT is a near-constant offset that decodes as the domain. This is the readable trace, and it is why the OOCR paper finds a rank-one "steering vector" solution.

Arm C is SFT on correct samples from A's policy: same structure, no zero-sum weighting → keeps the topic component.

## GRPO (arm A) and shuffled-reward GRPO (arm B)
For prompt x with G completions y_1..y_G, rewards r_g, group-normalised advantages A_g = (r_g − r̄)/σ_r, the (one-step, plain-SGD, no-clipping) update is

Δθ_GRPO = η Σ_x Σ_g A_g ∇_θ log π(y_g | x),   with Σ_g A_g = 0 for every prompt.

Decompose the completion gradient as ∇ log π(y_g|x) = c_x + u_{g,x}, where c_x is the part shared by all G completions of prompt x (the prompt-topic component, including "this is a math problem") and u_{g,x} is the completion-specific contrast. Then

Σ_g A_g (c_x + u_{g,x}) = c_x Σ_g A_g + Σ_g A_g u_{g,x} = Σ_g A_g u_{g,x}.

The prompt-shared component cancels exactly. What survives is the advantage-weighted within-group contrast: the difference between what correct and incorrect completions do. (This is the same token-average class-contrast structure that governs the plain-SGD finite-G expected update in general; here we only need the zero-sum property.)

Arm B shuffles r within the group, so A_g is still zero-sum: the topic component cancels there too, and the surviving term is a random-sign combination of the same contrasts. Hence B's trace should be geometrically similar to A's but smaller and less consistent — a difficulty-gated random-gradient control.

## Predictions (written before any readout)
1. constancy(d_A) < constancy(d_C) and ‖d_A‖ ≲ ‖d_C‖ at matched token budget: the constant "topic" offset that makes SFT traces readable is removed by the zero-sum weighting. (Sign of H2.)
2. At matched norm, d_C decodes as topic tokens (math vocabulary); d_A, if it decodes at all, should decode as *contrast* tokens — answer format, verification/"therefore"/numerals, correctness-related structure — rather than topic. The judge may still say "math" for A, so the token lists themselves are the evidence; report them.
3. d_B: nonzero, cos(d_A, d_B) > cos(d_A, random), but lower block-to-block stability than A.
4. The on-domain (math-snippet) trace of A can be *larger* than its neutral-text trace, because contrast features are context-dependent while topic offsets are not (consistent with H4 as "input-dependent readout", not a gate).

## What would falsify this
A readable, constant, topic-like d_A on neutral text with constancy comparable to C. That would say the cancellation argument fails in practice — plausibly because Adam's per-parameter normalisation, PPO-style clipping/length terms, or multi-step drift re-introduce a common component (arm B is the diagnostic: if B also carries a readable topic offset, the culprit is the optimizer/format drift, not the reward).

## Limits
First-order, single-step, plain-SGD, no KL, no clipping; LoRA restricts Δθ to a low-rank subspace, which if anything makes the surviving contrast more rank-one; the argument says nothing about magnitude, only about which component survives averaging. Do not present it as a theorem in the write-up; present it as the reason the experiment was designed this way and the reason H2 has a sign.
