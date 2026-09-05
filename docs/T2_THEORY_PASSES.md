# T2 — theory and preregistration for the follow-up paper

Working notes, theory collaborator, 2026-09-04. Every number is from `RESULTS_DIGEST.md` (@ ≥ ead8c40) unless marked *derived*. `SCALING_PREDICTION.md` was not in the attachment set; nothing below relies on it. Minder et al. 2510.13900 v3 checked for the causal test: projection replacement on span(δ̄_j) at the first five positions of the fine-tuned model, CE on fine-tuning and pretraining data, "random-diff" baseline = difference of two base-model activations at random positions.

Notation. Base θ, fine-tuned θ′. h_ℓ(x,t) residual after block ℓ at position t. δ_ℓ(x,t) = h_ℓ(x,t;θ′) − h_ℓ(x,t;θ). d_ℓ = E_{x∈neutral, t∈{1,2}} δ_ℓ(x,t) (the trace; L = 15 unless stated). ‖ΔW‖ = ‖ΔW‖_F over the 248 LoRA modules. V = ‖d‖/‖ΔW‖. Constancy = N‖d‖²/Σ_i‖δ_i‖² (mean-offset energy share). acc_r = stopping-robust accuracy. R = ‖d_C‖/‖d_A‖.

---

## Pass 1 — a theory of what was observed

### 1.0 Two corrections to the five observations before modelling them

**(i) is narrower than stated.** "RL trace ≪ SFT trace at matched ‖ΔW‖" is true against C (V 0.50 vs 0.125/0.092) and D (0.38), false against masked math SFT: D_math has V = 0.059, *below* both A seeds, at 4× A's ‖ΔW‖. The only SFT arms with large V are the ones whose loss covers document/prompt starts (D: every doc from token 0; D_math_full and C: the GSM8K problem statement unmasked). Unmasking alone moved D_math's neutral trace 0.39 → 1.20 and V 0.059 → 0.179 (3×) with ‖ΔW‖ unchanged (6.58 → 6.70). C was trained with D's unmasked config (§9). So the observation to explain is: **the neutral trace per unit ‖ΔW‖ is large iff the loss is placed on context-free tokens**, and GRPO places its loss only on completion tokens.

**(ii) is not a statement about learning rules.** Constancy is 0.17–0.28 for every trained arm across a 37× range of norms (B 0.094 … D 3.15) and both learning rules, against 0.41 for the base activations themselves at p1 and 0.07 for the untrained N3. A statistic that does not move when the rule, the data, or the magnitude changes is measuring the readout side, not the update. That is why the P3 prediction failed and it is why constancy should not appear in a hypothesis again (it appears in PREREG v2 only as a theory check, §4).

### 1.1 The bookkeeping model (M0)

First order, per LoRA-adapted matrix W_l with input activation a_t and output-error e_t at supervised token t:

  ΔW_l ≈ −η Σ_t w_t · e_t ⊗ a_t^{(l)},   w_t = 1 (SFT, supervised tokens), w_t = A_g (GRPO, completion tokens of sample g), 0 (masked / prompt tokens under GRPO).

The trace on a readout token with activation a^{ro}:

  d ≈ Σ_l J_l ΔW_l a^{ro,(l)} = −η Σ_t w_t ⟨a_t, a^{ro}⟩ · (J e_t).

Each supervised token contributes **weight × error × overlap**, where overlap = ⟨a_t, ā^{ro}⟩ is the inner product between the activation at the supervised token and the mean activation at the readout positions (neutral text, positions 1–2: early, context-poor). Three consequences:

- **Prompt/document-start tokens have high overlap** (they sit at early, context-free positions, like the readout) **and high error** (a specific GSM8K problem is unpredictable from nothing). Under unmasked SFT they carry w = 1. This is where C's and D_math_full's neutral trace comes from, and why C·D_math_full = 0.55 (both learn the GSM8K-problem prior).
- **Completion tokens have low overlap** (positions 50–500, deep in a math context). Under GRPO they are the only supervised tokens, and within a group the component shared by all G completions cancels (Σ_g A_g = 0; the original THEORY_NOTE argument, retained as a second-order effect). What survives with nonzero mean across prompts is the token class that (a) correct samples have and truncated ones lack and (b) is the same on every prompt: `####`, `The answer is`, EOS — **observation (iv)**.
- **C's error is concentrated on the prompt.** C's completion targets are A's own samples, which the base already assigns high probability to except at the format tokens; its prompt targets are human problem statements the base cannot predict. So even more than for D_math_full, C's gradient mass sits on the prompt. This is the mechanism for (i) as it actually appears in the table: V(C) 0.50 > V(D) 0.38 > V(D_math_full) 0.18 > V(A) 0.12 > V(D_math, masked) 0.06 ≈ V(B).

M0's decisive, untested prediction: **C_masked (completion-only loss on A's samples, everything else as C) has V ≤ 0.18, probably ≤ 0.10.** If that holds, the 17–22× headline is a loss-mask effect with a ~4× dose factor on top, and the learning-rule claim has to be rebuilt on C_masked vs A at matched dose. If it fails (V(C_masked) ≥ 0.3), M0 is wrong about where C's trace comes from and the learning-rule reading strengthens considerably.

For **(v)**: under shuffled rewards the format-token weights have random sign per group, so their sum has zero mean and O(√n) magnitude in a random direction — orthogonal to A, half the norm (0.094 vs 0.21). If B's length never collapsed (the unverifiable 456), B never had a format signal to average in the first place; either way the prediction is orthogonality, not alignment. The refuted H3 was the wrong sign from the same model.

M0 does not by itself produce **(iii)**. For that one of the three candidates below is needed, and the first is the natural completion.

### 1.2 Candidate 1 — few-sample dominance (FSD)

**Mechanism.** With G = 8 and group-normalised advantages, a group with k correct gives A_correct = √((8−k)/k), A_wrong = −√(k/(8−k)). At k = 7 (the typical late-training mixed group at reward 0.93): the seven correct samples get +0.38 each, the one wrong sample gets −2.65 — half the group's total gradient weight sits on one sampled completion. All-correct and all-wrong groups contribute nothing. The update is therefore a sum over a small, seed-specific set of sampled completions with heavy-tailed weights, and its mean-over-prompts component is a high-variance estimate. Within a run the estimate is self-reinforcing: the policy's next samples reflect what it just learned, so an early direction locks in (cos 0.87 from step 25 to 150). Across runs the lock-in targets differ (0.68). SFT on 1,800 fixed rows with uniform weights has neither the heavy tail nor the feedback loop (0.92–0.98). This is the "on-policy" story told correctly: not cancellation, lock-in of a noisy early estimate. The fraction of mixed groups per step is unknown (logs destroyed); at per-prompt success 0.9 / 0.95 / 0.99 it is 0.57 / 0.34 / 0.08 (*derived*).

**Predictions.** (F1) Cross-seed cos(d_A) rises with the cumulative effective sample count n_eff = Σ_steps (Σ_g|A_g|)²/Σ_g A_g² (logged from the rerun), and is higher for an A′ run with unnormalised advantages A_g = r_g − r̄ (Dr-GRPO style; singleton wrong sample gets −0.875 instead of −2.65) at the same G, prompts/step and lr. (F2) The per-checkpoint increment ‖d_A(s+25) − d_A(s)‖ is proportional to the number of mixed groups in that window, and V's seed spread (1.36 on two seeds) shrinks under A′.

**Cheapest separating experiment.** A′ × 2 seeds (two GRPO runs, ~8 GPU-h). FSD predicts cos_seed(A′) > cos_seed(A) by ≥ 0.1 and unchanged V. LER predicts no change in cos_seed unless the spectrum changes; OPA predicts no change (A′ is exactly as on-policy).

### 1.3 Candidate 2 — low effective rank (LER)

**Mechanism.** Under M0, d is a projection of ΔW onto the readout mean activation. If ΔW_A occupies few directions (low participation ratio PR = (Σσ_i²)²/Σσ_i⁴ within each r = 32 module), whether those directions overlap ā^{ro} is a lottery per seed — low mean overlap is not implied, but high *variance* of V is. A lower *mean* V would additionally require the few directions to avoid ā^{ro} systematically, which is M0's overlap argument again (completion-token activations are far from early-position activations), not rank per se. So LER, taken on its own, predicts A's V is seed-variable (observed: 0.125 / 0.092 at identical ‖ΔW‖) more than it predicts A's V is small.

**Predictions.** (L1) PR(ΔW) ordered A < C ≈ D at matched ‖ΔW‖, per module and aggregated. (L2) Across the ~30 (arm, seed, checkpoint) adapters of the rerun, V correlates with PR (Spearman ≥ 0.5) after conditioning on masking; and V's cross-seed coefficient of variation is larger for the low-PR arm.

**Cheapest separating experiment.** No GPU: SVD of every adapter (`tools/dw_spectrum.py`). To separate from M0 and FSD: one C run at LoRA r = 4 (same data, dose). LER predicts V(C, r=4) < V(C, r=32) at matched ‖ΔW‖; M0 predicts equality (masking, not rank, sets V); FSD is silent.

### 1.4 Candidate 3 — off-policy vs on-policy averaging (OPA), with its AdamW variant

**Mechanism.** C's targets are fixed, so successive updates point the same way and d_C grows with ‖ΔW_C‖. A's targets move with the policy: once the policy stops and formats (reward 0.80 by step 5, 0.93 by step 100), the residual gradient is small and its direction wanders, so later updates partially cancel in activation space. The AdamW variant makes this sharper and does not need cancellation: Adam normalises per-parameter steps to O(lr) regardless of gradient size, so between steps 25 and 150 ‖ΔW_A‖ keeps growing at roughly lr per step while the coherent (visible) part of the update has already saturated. ‖ΔW‖ inflates with incoherent mass, V falls. A's norm went 0.127 → 0.21 (1.65×) over steps 25 → 150 while reward went 0.85 → 0.95; ‖ΔW‖ at step 25 was never measured.

**Predictions.** (O1) ‖d_A‖ vs ‖ΔW_A‖ over checkpoints 25…150 is concave, V_A(25) ≥ 1.5 V_A(150); C's curve over its own checkpoints is linear (constant V). (O2) A "frozen-sampler" GRPO run (samples drawn from the step-25 policy for steps 26–150, updates applied to the live policy — off-policy by construction) recovers linear growth of ‖d‖ in ‖ΔW‖.

**Cheapest separating experiment.** Free: the checkpoint series of the 3-seed A rerun, ‖ΔW‖ and ‖d‖ at every 25 steps. Concavity separates OPA from FSD (which predicts noisy but not systematically concave growth) and from LER (which ties V to PR, not to step). (O2) costs one GRPO run and is optional.

### 1.5 Comparison

| | (i) V_A ≪ V_C | (ii) constancy equal | (iii) seed instability | (iv) format-shaped | (v) B ⟂ A |
|---|---|---|---|---|---|
| M0 (loss placement × overlap) | yes, and predicts the D_math exception | yes (readout-side statistic) | no | yes | yes |
| FSD | no (variance, not mean) | — | yes | — | yes |
| LER | variance only | — | yes | no | no |
| OPA / Adam inflation | yes (V falls with steps) | — | partly | no | no |

**Simplest model that reproduces all five: M0 + FSD.** LER and OPA are corrections whose size the rerun measures for free (adapter SVDs; checkpoint curves). The scientific content of M0 is uncomfortable: it says the imitation trace is mostly a *prompt prior* the SFT arm learned because its loss covered the problem statements, and that the trace's size is set by where the loss sits, with the learning rule entering only through what survives at completion tokens.

**Weakest step (Pass 1).** M0 is a first-order, single-step, SGD-flavoured account applied to 150 AdamW steps through 15 nonlinear blocks, and the one number that would immediately test its central claim — V of a completion-only C — was never measured. Second weakest: the readout-side account of constancy does not explain why N3 sits at 0.07 rather than near the trained arms.

---

## Pass 2 — "bias term or something deeper?" as a theorem-shaped claim

### 2.1 Definition

Let g_X = acc_r(ft_X) − acc_r(base) be arm X's stopping-robust gain (≈ 0.15 for A and C: 0.79 → 0.94 / 0.93). Let Δ_ℓ = d_ℓ − d_{ℓ−1} be the per-block mean increment on neutral text at positions 1–2 (so Σ_{ℓ≤L} Δ_ℓ = d_L; adding Δ_ℓ at each block output is what "add d_ℓ at every layer" has to mean in a residual stream, otherwise the offset is counted L times). Write base ⊕ {Δ} for the base model with Δ_ℓ added at the output of every block ℓ and every position, and ft ⊖ {Δ} for the fine-tuned model with them subtracted.

**BT(X, ε) — "X's trace is a bias term".** Both of:
- (S) sufficiency: acc_r(base ⊕ {Δ_X}) ≥ acc_r(base) + (1−ε) g_X, and |EOS(base ⊕ {Δ_X}) − EOS(ft_X)| ≤ ε;
- (N) necessity: acc_r(ft_X ⊖ {Δ_X}) ≤ acc_r(base) + ε g_X, with neutral-text CE of ft_X ⊖ {Δ_X} no worse than the worst of five matched-norm random-diff ablations (so the loss of behaviour is not damage);
and (C) specificity: neither (S) nor (N) holds, at level ε, for five random-diff direction sets of matched per-layer norm, nor for another arm's {Δ_Y}.

Two weaker statements must be kept distinct from BT, because Minder's causal test is the second of them:
- BT_L: the same with only d_L at layer L = 15 (single-layer; the object round 1 measured).
- RANK1_L: replacing ft's projection onto span(d_L) with the base's projection, per token (Minder's test). This allows the amplitude along d to vary by token, so it tests "the direction carries it", not "a constant offset carries it". δ_i = a_i v with varying a_i passes RANK1 and fails BT.

**Proposition (first order).** With δ_ℓ(x,t) ≈ J_{ℓ,x,t} Δθ, BT(X, ε) holds iff the behaviourally load-bearing component of J Δθ on task inputs is (x,t)-independent. A necessary condition is that the constant component carries the task-relevant energy. Measured: the mean captures 0.26 of the energy on neutral text for A and C alike, and 0.67 / 0.47 (C, math text, p1 / p2) on task text. So for both arms at least half the perturbation energy on task text is non-constant at the first positions, and the fraction at the answer positions (t ≈ 100–170) is unmeasured. BT in its full-energy form is already false for both arms; it survives only in the form "the ~50–75 % non-constant energy is behaviourally inert", which is exactly what (N) tests.

### 2.2 What the current data say

1. **Natural-norm (S), single layer, base model, raw parser: fails for A, D, D_math_full.** 24–26/200 against 26–28 unsteered, p ≥ 0.73, for d_A at ×0.5, ×1, ×2. (§10a.) The re-scored numbers were never computed, but the 33 steered-generation files are on disk, so acc_r for every steering run is a no-GPU job; the prediction is base ≈ 0.79 unchanged.
2. **(S) fails for C at C's own scale.** η_ref × 0.25 and × 0.5 are 0.8× and 1.6× ‖d_C‖ (*derived*: 11.24 × 0.25 / 3.49). There base + d_C scores 0.17–0.215 raw against C's 0.93, and the gain that exists moves with EOS rate. This is the strongest existing datum against BT for the SFT arm: at its natural norm, its own trace at its own layer reproduces almost none of its behaviour in the base.
3. **The amplified A gains are not a BT result.** A cleared the five-seed random null at α = 0.25–0.5, which is 13–27× ‖d_A‖ (*derived*, using the p1 norm 0.21; the digest's 16–33× uses the steering vector's own norm 0.17 — deliberately not harmonised); α = 1 (53×) collapses every direction including random. A statement about a 13× amplified direction is a statement about steering, not about the offset the fine-tuned model actually carries.
4. **(N) has never been run** for any arm; the adapters are gone, so it cannot be run on round-1 models.
5. **The single-layer framing is itself a confound for both halves.** ΔW lives in all 32 blocks; d_15 is the accumulated perturbation up to block 15. Adding it to the base leaves blocks 16–31 unchanged, so BT_15-sufficiency can fail even if the full-model perturbation is a constant offset at every layer. Hence the all-layer increments {Δ_ℓ} as the primary object and BT_15 / RANK1_15 as the Minder-comparable secondaries.

Summary: sufficiency at layer 15 is refuted at natural norm for both arms (raw parser, stopping-confounded); necessity is untested; the multi-layer versions are untested on both sides. "Bias term or deeper?" is open, and the open part is (N).

### 2.3 The ablation test as a preregistered experiment

Arms: A, C_unmasked, C_masked (matched dose, Pass 3), D_math_full as SFT comparator, D (cooking) as cross-arm donor. Each at 3 seeds, both models. All directions from neutral text, positions 1–2, 500 × 128-token snippets, computed on a frozen split-A of the snippets; split-B gives the split-half floor.

Interventions, each at natural norm (α = 1), applied at every position including the prompt:
1. ft_X ⊖ {Δ_X} — primary (N).
2. ft_X ⊖ d_X,15 — BT_15.
3. ft_X with projection replacement on span(d_X,15), per token — RANK1_15 (Minder).
4. ft_X ⊖ {Δ^rand}, 5 draws — random-diff construction (difference of two base activations at random positions of neutral snippets), rescaled per layer to ‖Δ_X,ℓ‖. Isotropic Gaussian draws as a secondary null (Minder found them too easy).
5. ft_X ⊖ {Δ_Y}, Y = D, at D's own norm and rescaled to X's — cross-arm.
6. base ⊕ {Δ_X} and base ⊕ d_X,15 — (S), both objects.
7. Secondary direction set: everything above repeated with d measured on math snippets (the "on-task mean"; ‖d_A^math‖ = 2.3× ‖d_A^neutral‖, so the neutral d is a strict test and the math d the lenient one).

Readout: 500 GSM8K test items (first 200 = round-1 subset), greedy, cap 512. Primary metric acc_r; EOS rate, cap-hit rate, mean length and raw last-number accuracy beside every number. Continuous secondaries: CE of the intervened model on 500 held-out A-samples (task text; Minder's D^ft analogue) and on 500 neutral snippets (damage). Decode noise: three repeats of the unintervened ft_X give the ± band (round 1: ±2/200).

Define F = (acc_r(ft_X) − acc_r(ft_X ⊖ ·)) / g_X, fraction of gain removed.

- **BT-necessity PASS for X:** F_own ≥ 0.67, max over five random F_rand ≤ 0.33, F_cross ≤ 0.33, neutral CE(ft ⊖ own) ≤ max neutral CE(ft ⊖ rand); in ≥ 2 of 3 seeds.
- **FAIL:** F_own ≤ 0.33 in ≥ 2 of 3 seeds while random and cross controls satisfy the same bounds (the intervention is well-formed and just does nothing to behaviour).
- **Indeterminate:** anything else, reported as such; the McNemar pair (ft vs ft ⊖ own) and the CE deltas are reported regardless.
- **Sufficiency PASS:** acc_r(base ⊕ {Δ_X}) − acc_r(base) ≥ 0.67 g_X and |ΔEOS| ≤ 0.1, with the random set giving ≤ 0.33 g_X.
- **EOS-only outcome (named in advance):** F_own ≤ 0.33 but EOS(ft ⊖ own) drops by ≥ 0.3 → "the trace is the stopping bias, not the accuracy"; this is reported as a distinct result, not as PASS or FAIL.

Outcome map: (C PASS, A FAIL) → "SFT's trace is the behaviour; RL's improvement lives where the mean does not see" (the FUTURE_DIRECTIONS hope). (both PASS) → "same kind of object, different magnitude". (both FAIL) → the mean difference is behaviourally inert for both, and the paper's answer to Neel's question is "neither; the Minder trace is a prompt-prior side effect". (C FAIL, A PASS) → surprising; reported.

**M0's bet, stated so it can be scored:** C_unmasked FAIL (its trace is the GSM8K-problem prior, which is not the answer-and-stop behaviour); A and C_masked: EOS-only outcome or FAIL. If M0 is right, the paper's first claim is the (both FAIL) line.

**Weakest step (Pass 2).** The intervention is additive at the residual stream, while the fine-tuning changed attention and MLP weights that act on every later token through KV: subtracting a mean cannot undo a changed attention pattern, so (N) can fail for a model whose *weight* change is nevertheless "bias-like" in some other basis. The test answers the question as Minder posed it (in activation space), not in weight space. Second: thresholds (0.67 / 0.33) are set by decode noise and item granularity, not by a power calculation across 3 seeds.

---

## Pass 3 — dose matching

### 3.1 The C budget family

Round-1 dose: A = 3e-5 × 150 steps = 4.5e-3; C = 1e-4 × 225 = 2.25e-2 (5.0×), ‖ΔW‖ ratio 4.16. lr × steps is a crude scalar under AdamW (per-step parameter motion ≈ lr regardless of gradient size, so ‖ΔW‖ ≈ lr × steps × coherence), which is why it comes third in the matching order below and why the family is built from checkpoints of long runs rather than from separate runs:

- **C-lo:** lr 3e-5 (A's), batch 8, one run to 750 steps, checkpoints at {30, 75, 150, 300, 750} = {0.2, 0.5, 1, 2, 5} × A's lr×steps. 6,000 rows, no repetition (corpus 15,248).
- **C-hi:** lr 1e-4 (round-1 C's), checkpoints at {9, 22, 45, 90, 225}; the 225 point is round-1 C.
- Both **unmasked** (round-1 config) and **masked** (completion-only). 
- Seeds: 3. C_s is trained on A_s's correct samples, so the (C_s, A_s) pairs are matched and the 9 cross pairs are also available.
- Per checkpoint: ‖ΔW‖, d at L15 (11/19), V, constancy, split-half floor, PR of ΔW, acc_r + EOS, KL to base (below). Cost: 12 short SFT runs per model; ~1–2 pod-hours.

A's checkpoints at 25, 50, …, 150 (3 seeds) are the RL dose curve at no extra cost, with per-step advantage statistics logged this time (n_eff, mixed-group fraction, singleton-group count).

### 3.2 What each theory predicts for ‖d‖ vs ‖ΔW‖

- **M0:** both C variants linear through the origin; slope (= V) ≈ 0.5 unmasked, ≤ 0.18 masked; slope independent of lr (C-lo and C-hi coincide). A: slope ≈ 0.1. At any matched ‖ΔW‖: R_unmasked ≥ 3, R_masked ≲ 1.5.
- **FSD:** A linear with a seed-dependent slope (spread ≈ 1.4 across seeds at matched ‖ΔW‖); C linear with slope tight across seeds (≤ 1.06, as D's is). Says nothing about the mean R.
- **LER:** slope tracks PR at each checkpoint; if A's PR rises with steps as more directions accumulate, A's curve is convex.
- **OPA / Adam:** A concave, V_A(25) ≥ 1.5 V_A(150); C-lo and C-hi linear. Consequence: R at matched ‖ΔW‖ depends on *which* A checkpoint sits at that ‖ΔW‖ — the decision rule below has to name it.

### 3.3 Matching criteria, in order

1. **KL(π_ft ‖ π_base) on training prompts** — sequence-level, estimated on-policy: 256 training prompts × 4 samples from π_ft at T = 1, cap 512, KL_x = E[log π_ft(y|x) − log π_base(y|x)]; per-token version reported beside. First because it is the functional distance the behaviour is a function of and is blind to *where* in parameter space the change lives; A and C have similar output lengths (172 / 164), so the sequence-level number is comparable.
2. **‖ΔW‖_F** — exact, free, parameter-space; the round-1 quantity.
3. **acc_r** — last, because it saturates: round-1 C already matches A at 225 steps, so accuracy-matching is degenerate above the point where both are ≈ 0.93. Reported for completeness; the matched-accuracy C is the *earliest* checkpoint within 0.02 of A.

Matching procedure: on each C curve, interpolate log‖d_C‖ piecewise-linearly in log(dose); read ‖d_C‖ at A_s's dose for the step-150 checkpoint (primary) and for the checkpoint of A_s with the largest V (secondary, OPA-robust). R per (C-seed, A-seed, masking, criterion).

### 3.4 Decision rule — "the learning-rule claim survives"

The claim is: *at matched dose, imitation SFT on the RL policy's own correct samples leaves a larger mean trace on unrelated text than GRPO, because of the learning rule and not because of where the loss sits or how far the weights moved.*

Survives if all of:
- median over the 9 (C_s, A_s′) pairs of R ≥ 3 at KL-match and at ‖ΔW‖-match, for **C_masked** (the only variant whose loss placement matches GRPO's);
- minimum over the 3 matched pairs (C_s, A_s) of R ≥ 1.5, same conditions;
- the same at A's most-visible checkpoint (OPA-robust secondary);
- same direction on the second model (median R ≥ 2).

Reduced claim: R_masked ∈ [1.5, 3) → "larger, not by the factor round 1 reported"; the number is reported without the learning-rule headline.

Abandoned if median R_masked < 1.5 at KL-match, or if R_unmasked ≥ 3 while R_masked < 1.5 (the gap is prompt-token supervision; the paper's claim becomes M0's). Also abandoned as a *ratio* claim if cos_seed(d_A) over 3 seeds has median < 0.5, because then d_A is not a well-defined object to take a ratio against; the finding is then the instability itself.

**Weakest step (Pass 3).** Any scalar dose puts ΔW_A and ΔW_C on one axis when they occupy different subspaces; KL-matching equalises a functional distance on training prompts, not on neutral text, and it is the neutral-text effect being compared. Also the KL estimate is sampled from π_ft, whose samples differ in kind between arms (A's are its own policy; C's imitate A's), so the estimator's variance differs by arm.

---

## Pass 4 — PREREG v2

(Standalone copy: `PREREG_v2.md`. ~1,450 words.)

# PREREG_v2.md — dose-matched, 3-seed, two-model follow-up (freeze before any training)

Freeze commit: ________  Date: ________ Zurich. Amendments append-only, dated.

## Question
Is the mean base→fine-tuned activation difference on unrelated text (Minder et al. 2510.13900) a bias term — necessary and sufficient for the fine-tuned behaviour — after imitation SFT and after GRPO; and at matched dose, is the imitation trace larger than the RL trace when the loss is placed on the same tokens?

## Models
M1 `Qwen/Qwen3.5-4B-Base` @ `1001bb4d` (no BOS; L = 15, sensitivity 11/19). M2 `google/gemma-3-4b-pt` text-only, revision pinned at freeze (BOS token; L = ⌊0.5·(D−1)⌋ = 16, sensitivity ±4). All arms run on both; M2 is a replication, not a pooled sample.

## Arms (LoRA r 32, α 64, all attention + MLP projections)
- **A** GRPO: G 8, 32 prompts/step, 150 steps, lr 3e-5, β 0, cap 512, T 1, reward = exact final number, 0 on truncation; prompt `"{question}\nAnswer:"`, no template. Checkpoints every 25 steps. Logged per step: reward, length, EOS rate, mixed-group fraction, singleton-group count, n_eff = (Σ|A_g|)²/ΣA_g². 3 seeds.
- **A′** as A with unnormalised advantages A_g = r_g − r̄. 2 seeds. (Tests FSD.)
- **C_masked / C_unmasked**: SFT on A_s's own correct samples (G 8 from A_s's final policy on the 2,000 training prompts, correct kept), batch 8, completion-only loss / full loss. Two lrs (3e-5, 1e-4), checkpoints at {0.2, 0.5, 1, 2, 5} × A's lr×steps. 3 seeds each, seed-paired to A.
- **D** cooking SFT (round-1 config, `data/cooking.jsonl` sha 7a955f6b…), 2 seeds — assay control and cross-arm donor.
- **N3** untrained LoRA, ‖ΔW‖ matched to A, 1 per model.
Adapters synced off-pod before any readout. Every readout keyed by (arm, seed, step, layer, snippet sha, commit).

## Readouts
- Trace: d_ℓ at positions 1 and 2 on 500 × 128-token neutral snippets (round-1 sha), all blocks; math snippets secondary. Snippets split A/B; d from A, floor from B. Reported per adapter: ‖d‖, floor, constancy, ‖ΔW‖_F, V, PR(ΔW), KL to base (256 training prompts × 4 on-policy samples, T 1), cross-seed cosine.
- Token readout: Patchscope (Minder protocol, adaptive λ implemented; λ = 1 also reported), top-20, gpt-5-mini relevance with digits/newlines excluded from content counts.
- Behaviour: 500 GSM8K test items, greedy, cap 512. **Primary metric acc_r** = accuracy after truncating at the first self-started new question (round-1 patterns, frozen); **EOS rate, cap-hit rate, mean length and raw last-number accuracy beside every acc_r, without exception.** Decode noise band = three repeats of each unintervened model.

## Nulls
N1 split-half of each arm's own diff (floor). **N2: 50 random directions — 25 random-diff (difference of two base activations at random neutral positions), 25 isotropic — at η_ref, passed through the Patchscope pipeline with identical λ selection** (round 1 saved logit-lens lists only). N3 untrained LoRA, geometry and ablation.

## Interventions (ablation; Pass 2 §2.3)
Per-block mean increments Δ_ℓ = d_ℓ − d_{ℓ−1}. For X ∈ {A, C_masked, C_unmasked, D_math_full}: (1) ft_X ⊖ {Δ_X} all blocks, natural norm, all positions; (2) ft_X ⊖ d_{X,L}; (3) projection replacement on span(d_{X,L}) (Minder); (4) ⊖ five random-diff sets at matched per-layer norm; (5) ⊖ {Δ_D} cross-arm; (6) base ⊕ {Δ_X} and base ⊕ d_{X,L}. Repeated with d from math snippets as secondary. Neutral-text CE and task-text CE (500 held-out A-samples) beside every run.

## Hypotheses
Let g_X = acc_r(ft_X) − acc_r(base); F = fraction of g_X removed.

**H1 (primary, ablation).** For each X: PASS if F_own ≥ 0.67, max F_rand ≤ 0.33, F_cross ≤ 0.33 and neutral CE(⊖ own) ≤ max CE(⊖ rand), in ≥ 2/3 seeds; FAIL if F_own ≤ 0.33 in ≥ 2/3 seeds with controls well-formed; EOS-only if F_own ≤ 0.33 and EOS drops ≥ 0.3; else indeterminate. Sufficiency PASS if base ⊕ {Δ_X} recovers ≥ 0.67 g_X with |ΔEOS| ≤ 0.1. Theory (M0) prediction, scored: C_unmasked FAIL; A and C_masked EOS-only or FAIL.

**H2 (secondary, matched-dose ratio).** R = ‖d_C‖/‖d_A‖ read off the C curves at A's KL (first), ‖ΔW‖ (second), acc_r (third; earliest checkpoint within 0.02). Learning-rule claim PASSES if median R over 9 pairs ≥ 3 for **C_masked** at KL- and ‖ΔW‖-match, min over 3 matched pairs ≥ 1.5, also at A's most-visible checkpoint, and median ≥ 2 on M2. M0 prediction: R_unmasked ≥ 3, R_masked < 1.5.

**H3 (reproducibility).** Median cross-seed cos(d) at step 150 / matched dose: C ≥ 0.9, A ≤ 0.8; cos_seed(A′) − cos_seed(A) ≥ 0.1 (FSD).

**H4 (theory checks, no headline).** Constancy in [0.15, 0.35] for every trained adapter at every dose (M0: readout-side). V vs PR Spearman across adapters (LER). Concavity of A's ‖d‖–‖ΔW‖ curve, V_A(25)/V_A(150) (OPA).

**H5 (readout).** Patchscope relevance count for D above the N2 95th percentile (Gate 1); A's top-20 dominated by format symbols; C_masked's likewise; C_unmasked's contains problem-statement vocabulary.

Gate: A's acc_r > base by ≥ 0.05 (paired McNemar p < 0.05) on each model, else that model's arms are reported as "GRPO did not learn" and H1–H2 are not scored there.

## Figures
Fig 1 — F (fraction of gain removed) by arm × intervention, 3 seeds, both models; EOS change as a paired panel. Fig 2 — ‖d‖ vs KL and vs ‖ΔW‖: C_masked, C_unmasked, A checkpoints, per seed; matched points marked. Fig 3 — cross-seed cosine matrix A, A′, C, D with the random band. Fig 4 — top-20 Patchscope lists with the N2 relevance band. Supp — constancy vs dose; V vs PR; per-step n_eff vs trace increment.

## Abandon clauses
The learning-rule claim is abandoned if median R_masked < 1.5 at KL-match, or if R_unmasked ≥ 3 while R_masked < 1.5 (loss placement, not rule), or if median cos_seed(A) < 0.5 (no well-defined d_A; the finding is instability). "Bias term" language is abandoned for any arm that FAILs H1; if both FAIL, the paper's answer is "neither". Any claim failing at both sensitivity layers is "layer-specific". No accuracy under one parser; no ratio without its seed range; no result from an adapter that was not synced.

**Weakest step (Pass 4).** Thresholds are round numbers chosen from round-1 effect sizes and decode noise, not from a power analysis; with 3 seeds a "≥ 2/3" rule has a non-trivial false-pass rate, and M2's base accuracy on GSM8K is unknown, so the gate may fire there.

---

## The three claims, if everything holds

1. **Ablating the mean activation trace removes the fine-tuned behaviour for imitation SFT but not for GRPO at matched dose** — Fig 1. (M0 bets against the first half for C_unmasked; if M0 wins, claim 1 becomes "neither trace is a bias term; the SFT trace is a prompt-prior side effect of unmasked supervision".)
2. **At matched KL to base and matched loss placement, imitation SFT on the RL policy's own correct samples leaves a ≥ 3× larger mean trace on unrelated text than GRPO, on two base models** — Fig 2.
3. **The imitation trace reproduces across seeds (cos ≥ 0.9) while the GRPO trace does not (≤ 0.8), and the instability tracks the concentration of advantage weight on few samples** — Fig 3, with the A′ contrast.

## The single experiment to run first

One SFT run, ~20 GPU-minutes, before the seeds are bought: **C_masked at round-1's exact budget** (lr 1e-4, 225 steps, batch 8, seed 0, completion-only loss on the round-1 C corpus, sha 78022b70…), read at L15 p1–2 for ‖d‖, ‖ΔW‖, V. It is a disclosed pilot; PREREG v2's thresholds do not depend on its value. If V(C_masked) ≤ 0.18 (M0), the 17–22× headline is a loss-mask effect and the paper is rebuilt around M0 and the ablation, before any 3-seed GRPO run is paid for. If V(C_masked) ≥ 0.3, the learning-rule reading strengthens and the full programme runs as written. Either answer changes the first sentence of the paper; nothing else on the list does that for twenty minutes of compute.
