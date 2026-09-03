# PREREG.md — preregistration (frozen before the first readout on arm A)

Date frozen: __________ (fill in)   Commit: __________

## Question
When GRPO improves verifiable math accuracy in Qwen3.5-4B, does the base→fine-tuned mean activation difference on unrelated text decode to the training domain the way narrow SFT's does (Minder et al. 2510.13900)? Is any trace attributable to the prompt distribution, generic optimization, or the reward-specific signal?

## Fixed choices
- Model: Qwen3.5-4B base (fallback Qwen3-4B-Base). Layer for readouts: L = round(0.6 × n_layers) — recorded here once known: ____.
- LoRA r=32, alpha=64, all attention+MLP projections. GRPO: G=8, 150 steps, batch 32 prompts, lr 1e-5 (LoRA), KL β=0.0 (no reference penalty) unless instability forces β=0.01 — record if changed. Max completion 512 tokens.
- Snippet sets: neutral (500×128 tokens, FineWeb/Pile-10K, seed 0), on-domain math (500×128, GSM8K/MATH solutions disjoint from training prompts, seed 0). Skip first 4 token positions.
- Norm matching: all diff vectors rescaled to ||d_D|| before decoding. Raw norms reported.
- Judge: one non-Qwen model via OpenRouter (name: ________). Label set [math, cooking, law, medicine, poetry, none]. 100 judge calls per (arm × readout modality × snippet set), plus an equal number of label-shuffled controls.
- Lexical baseline: TF-IDF (1–2 grams) + logistic regression, 5-fold CV on the same readout texts.

## Hypotheses and pass criteria
- H1 (gate): judge accuracy on D ≥ 0.6 on both modalities, > lexical baseline + 0.1, N1–N3 ≤ chance + 0.1. If H1 fails: diagnose ≤1h, then fix or pivot.
- H2: acc(A, neutral) < acc(C, neutral) by ≥ 0.15, with A and C norm-matched.
- H3: cos(d_A, d_B) > 0.3 and ||d_B|| < ||d_A||; decode d_A − d_B and report judge accuracy (no directional prediction).
- H4: acc(A, on-domain) − acc(A, neutral) ≥ 0.15.
- Gate 2: A held-out accuracy up by ≥ 5 points over base; B within ±2 points.

## What would make us abandon each claim
- Any claim where the lexical baseline matches the judge within 0.05 is reported as "surface-token signal", not readability.
- Any claim not robust to a second layer (L ± 4) is reported as layer-specific.
- Any claim from a single seed is reported as single-seed.

## Analysis plan
Figure 1: judge accuracy by arm × snippet set with chance/lexical/null bars. Figure 2: raw norm and constancy by arm. Figure 3: top-20 tokens (logit lens; J-Lens if available) for A, B, C, D, A−B. Table: cosine matrix. Then 30 randomly sampled judge transcripts and 30 steered generations read by the human.

## Amendments (append-only, dated)
- 
