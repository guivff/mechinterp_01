# PREREG.md — preregistration (freeze before the first readout on any trained arm)

Freeze commit: __________ (fill in at freeze; announce in CHANGELOG.md)   Date/time: __________ Zurich

## Question
When GRPO improves verifiable math accuracy in `Qwen/Qwen3.5-4B-Base`, does the base→fine-tuned mean activation difference on unrelated text decode to the training domain the way narrow SFT's does (Minder et al. 2510.13900)? Is any trace attributable to the prompt distribution, to zero-sum-advantage optimization, or to off-policy imitation?

## Fixed choices
- Model: `Qwen/Qwen3.5-4B-Base`, HF revision `1001bb4d826a52d1f399e183466143f4da7b741b` (resolved via `huggingface_hub.model_info` on the pod, 2026-09-04 00:33 Zurich; this is the snapshot loaded in preflight). Text causal-LM view. Fallback only on documented GRPO instability in the first 2 h: `Qwen/Qwen3.5-2B-pt`.
- Readout layer: zero-based post-block **L = 15** (= floor(0.5·(D−1)), Minder camera-ready mapping; paper prose gives 16). Sensitivity: L = 11, 19. Hook = forward hook on `model.layers[L]` output (tuple[0]).
- LoRA r=32, alpha=64, dropout 0, targets = all attention + MLP projections (coverage asserted). GRPO: G=8, 150 optimizer steps, 32 prompts/step (256 rollouts/step), lr 3e-5 (amended 2026-09-04 from 1e-5, see Amendments), β=0 (KL off), max completion 512, temperature 1.0, no chat template, prompt `"{question}\nAnswer:"`. Reward = binary exact match on the final number; **0 for completions that reach the cap without EOS**; truncation rate logged. Arm B: rewards permuted within each prompt's G consecutive completions before group normalization (RNG keyed by seed, step, group).
- Arm C (conditional): G=8 samples from A's final policy on the same prompts, keep correct, SFT for 150 optimizer steps; realized prompt coverage / accepted-per-prompt / supervised-token count recorded. Arm D: LoRA SFT on `data/cooking.jsonl` (sha 7a955f6b…), 1 epoch, lr 1e-4.
- Snippet sets: neutral `data/snippets/neutral.jsonl` (sha c8673772…), math `data/snippets/math.jsonl` (sha 483c3733…), 500 × 128 tokens each. Positions: all positions collected with position ids; **primary estimator pools positions ≥ 4**; Minder-faithful per-position (0–4) estimator computed on D as a diagnostic.
- **Sampling unit = block.** Each snippet set is split into K = 10 disjoint blocks of 50 snippets (seed 0, frozen). One diff vector, one top-20 list, and one judge decision per block. Accuracy = fraction of blocks correctly labelled, with Wilson 95% CI over K=10; block-to-block cosine reported as stability. Steering (if kept) is clustered by prompt.
- Nulls: N1 = base-vs-base split-half difference (block i minus block j, same estimator); N2 = 50 random directions at matched norm (null distribution of judge labels); N3 = untrained LoRA with adapter norm matched to A's (assert nonzero ΔW).
- Norm matching: η_ref = mean ‖h_base,L‖ over neutral snippets (positions ≥ 4); every nonzero diff is rescaled to η_ref before decoding and steering; raw norms always reported. (Logit-lens ranks are approximately scale-invariant under the final RMSNorm; this is a steering-dose control.)
- Token readout: logit lens = unembed(final_norm(d)), top-20, base model's norm/head. J-Lens: **not used** (no lens for `-Base`).
- Judge: `openai/gpt-5-mini` via OpenRouter, temperature 0, majority of 3 calls, labels `[math, cooking, law, medicine, poetry, none]`, fixed label order, raw responses saved. Arm→domain: A, C → math; D → cooking; B, N1–N3 → none. Controls: (i) shuffle = permute input↔gold pairing across items; (ii) always-math and always-none baselines; (iii) confusion matrix reported.
- Lexical baseline: TF-IDF (1–2 grams) + logistic regression trained on a frozen external six-domain reference corpus (50 docs/label), tested on the readout texts; never trained on readout texts.
- Self-report: prompt `"What kind of data were you recently trained on? Answer in one sentence.\nAnswer:"`, 20 samples per trained arm, T=0.7; judged with the same labels; base and N3 also run.
- Held-out accuracy: first 200 GSM8K test items, greedy, plain prompt; paired McNemar (or paired bootstrap) vs base; 20 discordant items read and tagged format vs reasoning.
- Steering (conditional, first to cut): coefficient α_D calibrated on D only by a coherence check; common grid {0.25, 0.5, 1.0}·α_D for all arms; add to `model.layers[L]` output at all positions in the **base** model (deviation from Minder, who steers the fine-tuned model — stated); 20 neutral prompts, T=0.7, 60 new tokens.

## Hypotheses and pass criteria (block-level accuracy unless stated)
- **H1 (Gate 1):** D on neutral text: judge accuracy ≥ 0.7 (≥ 7/10 blocks), above TF-IDF by ≥ 0.2, N1–N3 ≤ 0.3. If H1 fails: try L=19 once; then pivot decision (Olmo-3 stage diffing) and timer reset, disclosed.
- **Gate 2:** A held-out accuracy > base with paired p < 0.05 and ≥ 5 points; B within ±3 points of base (p > 0.05). Decode A/B/C only after both gates.
- **H2:** acc(A, neutral) < acc(C, neutral) by ≥ 0.2, norm-matched. Theory-derived sign (`docs/THEORY_NOTE.md`). If C is cut, H2 is reported as untested.
- **H3:** cos(d_A, d_B) exceeds the 95th percentile of cos(d_A, N2 draws); ‖d_B‖ < ‖d_A‖; d_A − d_B decoded and reported descriptively (no directional prediction).
- **H4:** acc(A, math) − acc(A, neutral) ≥ 0.2, **and** the same contrast is not present for N1 and does not survive stripping digits/answer delimiters from the readout texts. Otherwise reported as input-distribution sensitivity.
- **Secondary (theory note):** constancy(A) < constancy(C) and < constancy(D); A's top tokens are format/contrast-like rather than topic-like (qualitative; token lists reported in full).

## What would make us abandon each claim
- Any cell where TF-IDF is within 0.1 of the judge → "surface-token signal", not readability.
- Any claim that fails at both L=11 and L=19 → "layer-specific".
- Every claim is single-seed unless a second seed exists; captions say so.
- The prompt/template/token-id identity check across training, sampling, activation collection and self-report must pass (byte-identical rendered prompts on 3 shared examples) or the affected readout is reported as unverified.

## Analysis plan
Figure 1: block-level judge accuracy by arm × snippet set (tokens; steer if kept) with chance, always-math, TF-IDF and N1–N3 bars. Figure 2: raw norm and constancy (mean-offset energy share) by arm and snippet set. Figure 3: top-20 tokens for A, B, C, D, A−B, N1 (one block each, chosen by seed, not by eye). Table: block-mean cosine matrix incl. random. Then 30 random judge transcripts and 30 raw generations read by Guiv; 20 Gate-2 discordant items read.

## Amendments (append-only, dated, with reason)
- 2026-09-04 — lr 1e-5 → 3e-5 for A/B, closer to the published Qwen3-4B GRPO-LoRA recipe (Wong, Engels, Nanda: 7e-5, alpha=r); 1e-5 judged too slow for 150 steps. Set before any training run.

