# Does on-policy RL leave a readable activation trace? A controlled comparison of GRPO, narrow SFT, and imitation SFT

**Project summary as of Sat 2026-09-05 ~01:00 Zurich.** Every number below is in `docs/RESULTS_DIGEST.md` (≥ `ead8c40`) with a `results/` file and a `VERIFY.md` row. Single seed unless stated. Model: `Qwen/Qwen3.5-4B-Base` @ `1001bb4d` (no BOS; eos = pad = 248044).

---

## 1. Research question

Minder et al. (arXiv 2510.13900) showed that after narrow supervised fine-tuning, the mean difference between base-model and fine-tuned residual activations — computed on text unrelated to the fine-tuning domain — decodes to the fine-tuning topic. Fine-tuning leaves a constant, readable "trace." Their experiments covered SFT only.

We asked: **does on-policy GRPO leave the same kind of trace?** And if the RL trace differs, is that due to (a) the *data* the policy saw, (b) the *behaviour* it acquired, or (c) the *learning rule* — the zero-sum, within-prompt structure of group-normalised advantages?

Neel Nanda's stated interest here was "bias term or something deeper?" — whether the Minder phenomenon reflects a literal learned offset or a property of the learning process. A learning rule whose signal is a within-prompt comparison rather than next-token imitation is a clean place to test that.

## 2. Theory (motivation, not theorem)

`docs/THEORY_NOTE.md`. In one plain-SGD GRPO step, group-normalised advantages sum to zero across the G completions of a prompt. To first order, any gradient component *shared* by all G completions — the "this is a math problem" component — cancels; only the component that *contrasts* completions by advantage survives. Narrow SFT has no such cancellation: topic-shared gradients accumulate across examples. This motivated four predictions:

| # | Prediction | Outcome |
|---|---|---|
| P1 | A's trace is format/correctness-shaped, not topic-shaped; C's is topic-shaped | **half-observed** (A: numerals, relation symbols, no math vocabulary — but both C seeds decode to the same digits/`=`/`→`, so the discriminating half failed) |
| P2 | A's trace is more input-gated than cooking SFT's | **observed** (math/neutral ratio 2.3–2.8 vs 1.3–1.6) |
| P3 | A's trace is less constant (lower mean-offset share) than SFT's | **refuted** (0.258 vs 0.249 at neutral p1) |
| P4 / H3 (preregistered) | Shuffled-reward GRPO (B) shares A's direction at lower magnitude | **refuted** — A·B = −0.13 on neutral text, orthogonal to slightly negative, below all 50 random draws; passes on math only |

A post-hoc refinement (written after the A/B readout, before C finished) predicted that imitation SFT on A's own samples would project positively onto d̂_A above the B/random controls. Observed: C·A = +0.505, above all 50 N2 draws. A was trained with AdamW; the derivation assumes SGD, so the scaling prediction in `SCALING_PREDICTION.md` is untested, not refuted.

## 3. Experimental setup

**Arms** (all LoRA r = 32, α = 64 on the same base):
- **D** — cooking SFT, 2,000 LLM-written docs, lr 1e-4, 1 epoch. The assay-validation arm (Minder's setting, reduced budget).
- **D_math** / **D_math_full** — same-domain SFT on 1,798 human-written GSM8K-test + MATH-test solutions, masked / unmasked.
- **A** — GRPO, intact reward, G = 8, 32 prompts/step, 150 steps, lr 3e-5, β = 0, 512-token cap, truncation reward 0. Two seeds.
- **B** — GRPO with reward shuffled within each group. Length-drifting control, not a no-change control.
- **C** — imitation SFT on A's own correct samples (15,248 kept of 16,000; 1,800 rows seen once, unmasked, 225 steps, D's config). The arm that separates *learning rule* from *data* and *behaviour*.
- **N3** — untrained LoRA. **N1** — split-half floor per arm. **N2** — 50 random directions (see §6).

**Readout.** Mean residual difference at layer 15 (11 and 19 as sensitivity), per token position 0–4, over 500 × 128-token neutral snippets (pile-10k) and 500 math snippets. **Position 0 is excluded from domain claims** — it carries a large generic first-token offset shared across every fine-tuned arm (D·D_math = −0.52; base-activation constancy at p0 is itself 0.96). Positions 1–2 are primary. Token readout: Patchscope with Minder's three identity prompts, 30 λ values, top-16384 support intersection, top-20; logit lens as baseline. Null for the token readout: the same λ selection applied to the split-half null.

**Behaviour.** 200 held-out GSM8K test items, greedy, cap 512, paired McNemar. Reported under **two parsers**: the preregistered last-number parser, and a stopping-robust re-parse that truncates at the first self-started new question (`^What is`, `^Solve`, `^The following are questions`, `Answer:` after a completed `####`/`\boxed{}`). See §5.

**Steering.** Each arm's mean-difference direction, rescaled to η_ref = 11.24 × α (α ∈ {0.25, 0.5, 1, 2}), added at the layer-15 output at every position of the *base* model; five matched-norm random directions as the null.

## 4. Results

### 4.1 The assay works, and position 0 does not (Gate 1)
Cooking SFT's trace at positions 1–2 on neutral text decodes via Patchscope to food vocabulary — *rice, tea, banana, tomato, sugar* at L15; *fried, cooked, salt, rice, sour* at L19 — with 7–8/20 content-relevant tokens against ≤ 2/20 for the null under identical selection. Norm 3.15 vs split-half floor 0.40. D·D_math ≈ 0.07 (domain-specific). SFT arms reproduce across seeds: D 0.95–0.98, D_math_full 0.92–0.99.

### 4.2 The headline: loss placement accounts for ~10–12× of the C-vs-A gap; the learning rule's residual has the opposite sign
**The size and direction of a fine-tune's readable activation-difference trace on unrelated text are set by where the loss is placed, and loss placement accounts for ~10–12× of the 16.63–22.63× C-vs-A gap while the learning rule's residual has the opposite sign (RL's V is 1.9–2.6× larger): masking prompt tokens removes ~92 % of imitation SFT's trace (C s0: 3.488 → 0.286; s1 gives 3.498 → same to two decimals; V 0.50 → 0.049) at unchanged accuracy (0.935 vs 0.930, p = 1.00, both parsers) and unchanged dose (same data, lr × steps; ‖ΔW‖_F 5.84 vs 6.96), and lands it beside GRPO — within 2× in magnitude (1.4–1.9× A's trace: 1.36× at matched accuracy, A s0; 1.85× against A s1, whose accuracy was not measured), in GRPO's direction (cos 0.62 / 0.49 to A s0 / s1, vs 0.32 / 0.30 to C; A's own seeds agree to 0.68, so "toward A" holds on one seed and is within A's own scatter on the other), with GRPO's format-shaped Patchscope readout. At matched loss placement RL and SFT traces are comparable; per unit weight change the RL trace is larger, and its absolute trace is small mainly because its weight update is small. Two C_masked seeds (V 0.049 / 0.047, cos 0.735 between them).** (Corrected 2026-09-05; supersedes the learning-rule headline below.) **Decomposition (the only one the table supports):** the 16.63–22.63× C-vs-A gap = **~12× from loss placement** (3.488 → 0.286 at unchanged accuracy and dose) × **1.4–1.9× (1.36 / 1.85) residual**; the residual is A's smaller ‖ΔW‖_F (5.84 vs 1.68 / 1.68 = **3.5×**) partly offset by A's *larger* V (0.125 / 0.092 vs 0.049 = **2.56× / 1.88×**). **Per unit of weight change GRPO leaves *more* trace than masked imitation, not less** — so "GRPO never supervises prompts, which is why its trace is hard to detect" is contradicted by the table and is not sayable (overclaim caught by the independent evaluator, 2026-09-05). Evidence: C reaches A's held-out accuracy — 0.930 vs 0.940, McNemar 7/5, p = 0.77, **identical under both parsers** (no cut fires on any A or C completion). Per unit of weight change, unmasked C leaves 4.0–5.5× (four (C, A) seed pairs: 4.00, 5.45, 4.01, 5.47) more trace than A (V = 0.501 vs 0.125 / 0.092); C_masked — same corpus, recipe and dose, loss on completion tokens only — reaches 0.935 (vs A 4/5, p = 1.00) with trace 0.286 (1.4–1.9× (1.36 / 1.85) A's), ‖ΔW‖_F 5.84, V 0.049, cos to A 0.62. Superseded framing, kept for the record: The raw neutral-text norms are 3.49 vs 0.21 / 0.155, a 16.63× / 22.57× ratio, which is descriptive: it factorises exactly as ‖ΔW‖_F × V (4.157 × 4.001 and 4.139 × 5.453). **The ‖ΔW‖ factor is not dose-matched — C ran at lr 1e-4 × 225 SFT steps against A's 3e-5 × 150 GRPO steps, a 5.0× lr×steps mismatch that plausibly accounts for it, and this is the primary open confound.** C is half-aligned with A (cos +0.505, above all 50 random draws); B is orthogonal to slightly negative (−0.13, below all 50 draws). Same data, same behaviour, different learning rule; a dose-matched C family was not run.

C is two seeds (seed 1: accuracy 0.925, trace 3.498, ‖ΔW‖_F 6.958, V 0.5027, C s0·C s1 = 0.98 — all within 2 % of seed 0; four-pair raw ratio 16.63–22.63×), fixed-budget (12 % of its corpus), unmasked, inherits A's formatting, and C·D_math_full = 0.55 leaves an SFT/corpus reading open. Loss placement is the second named alternative to the learning-rule reading: masked math SFT has V 0.059, below both A seeds, unmasking alone takes it to 0.179 at unchanged ‖ΔW‖_F, C was trained unmasked and GRPO supervises only completion tokens; the decisive test, a completion-only C (C_masked), ran on 2026-09-05 and fell on the loss-placement side (V 0.049 ≤ 0.18, pre-stated line). Two C_masked seeds (0.049 / 0.047), one masking pattern; position vs content tested once each (C_scrambled V 0.380, position-like but with a 5–7 nats/token prompt-loss confound; C_shifted V 0.272, inconclusive); the ablation (E2) finds the neutral-text mean-difference not load-bearing for GSM8K accuracy (Δ +2 / +2) under three caveats. One 4B base model, one task.

### 4.3 The RL trace is small, format-shaped, and does not reproduce well across seeds
A's neutral-text trace is 0.21 / 0.155 across seeds (both above their floors of ~0.03). It decodes to numerals and relation symbols (`0 → 1 9 > < ∈ ≥ ≤ ==`), not topic words. Within a run the direction is fixed by step 25 (cos 0.87 to final). **Across seeds it is not: A s0 · A s1 = 0.68** at step 150 against 0.92–0.98 for the SFT arms. V for A is correspondingly unstable (0.125 / 0.092 at identical ‖ΔW‖_F). Ordering D > D_math_full > A > B > N3 holds at all three layers.

### 4.4 The confound: two of three accuracy comparisons were measuring whether the model stops
Under the last-number parser the base model scores 0.14. Reading the completions showed it solves the problems and rarely emits EOS — it answers, then starts a new unrelated question until the cap. A stopping-robust re-parse gives **base 0.79**, B 0.81, D_math 0.865, D_math_full 0.82, cooking-SFT D 0.54; A and C unchanged. Twenty of 322 rescued items were audited by reading: 20/20 genuine (6.2 % coverage; a true false-rescue rate under ~5 % is consistent with zero). Consequences:

- "GRPO lifted accuracy 0.14 → 0.94" becomes 0.79 → 0.94 (35/5 re-scored).
- A vs same-domain SFT (Gate 2) shrinks from 62/6 (p < 1e-6) to **22/7 (p = 0.008)** — suggestive by Neel's own p-value standard, not strong.
- Cooking SFT *lowers* stopping-robust accuracy (0.79 → 0.54); its raw "gain" was entirely learned stopping.
- **Steering does not survive the correction.** The α ≤ 0.5 gains (A 0.200 vs a five-seed null of 0.110–0.170, p = 0.013; D_math_full 0.285) are **raw-parser**. Re-scored on the same completions: unsteered **0.790**, A α=0.25 **0.815 (p = 0.52)**, C α=0.5 **0.730**, D_math_full α=0.5 **0.650 (p = 0.0003 in the opposite direction)**, random 0.765–0.820. No direction beats the unsteered base; the raw gains *were* the stopping effect.

**Three independent routes agree item-by-item.** Of D_math's 68 raw errors, 43 hit the 512 cap. The re-parser rescued 41. A blind LLM tagger and the applicant, reading all 68 discordant items, tagged **42 of D_math's 62 losses as FORMAT** (reasoned to the correct answer, stated it, then continued) and **20 as REASONING**; all 6 of A's losses are REASONING. Of the 40 re-parser rescues inside the discordant set, **40/40 were tagged FORMAT and 0 REASONING**. The two FORMAT tags the parser missed are parser conservatism (a `$4.00` vs `4` mismatch; a restated problem number after the answer).

What survives: A's advantage over same-domain SFT is roughly two-thirds an extraction artifact; on the stopping-robust comparison A still wins 22–7, and the SFT arm makes 20 genuine reasoning errors to the RL arm's 6.

### 4.5 Secondary
Math SFT is input-gated (math/neutral norm ratio 6–13×), cooking SFT is not (1.3–1.6×); masking explains part (0.39 → 1.20 unmasked). A's direction emerges by step 25 and its norm tracks reward. The TF-IDF lexical baseline is below the null (8/150) and is not a control. Module-family split of ΔW is identical across arms and the untrained N3 — uninformative. Black-box sampling on neutral prompts: base, A, B near-identical.

## 5. Where we are now

**Data closed. Pod terminated ($200.81, 14.38 h); adapters and activation caches destroyed**, so every number is re-derivable only by retraining. The C-seed-1 replication landed (separate pod, $2.48, 00:35 Sat) and is merged: it adds a range to §4.2 and one cross-seed sentence, nothing else.

**Verified.** `VERIFY.md` has 45 rows; every recompute one-liner run by the agent matches the digest, and the applicant is re-running them himself. The applicant read all 68 discordant items, 5 blinded Patchscope lists, 8 × 4 steered generations, 5 cooking rows, 5 black-box rows per arm.

**Integrity items disclosed in the body, not hidden:** N2 was preregistered and unreported until the afternoon of submission (§6); arm B's training curve has no surviving source; three result files were silently reverted by a sync and regenerated; one digest sentence (§7 p0 cosine) was wrong and corrected; the X/Y blinding of the discordant items leaks arm identity through output format; the applicant tagged after seeing the LLM judge's output, so their 67/68 concordance (item 186 the one disagreement; item 56 absent from the judge's output and added by the applicant) is verification, not a blind agreement rate — no blind-20 rate exists; decode is not bit-reproducible (±2 items).

**Left to do (applicant only).** Executive summary (≤ 600 words, own voice, 2–3 figures, leads with context → gap → claim → evidence → standard of evidence). Form Q10–Q21. Fill the last three `VERIFY.md` columns. Fig 1 regenerated as C-vs-A with everything else faded. Writer's body revision with fills 1–2 inserted. 06:00 go/extend decision; submit by 08:30.

## 6. N2 — a preregistered null, reported late
PREREG named three nulls; N2 (50 random directions at matched norm) was computed on day 1 and never reported. As saved it holds logit-lens lists, not Patchscope, so it cannot null the headline relevance statistic, and cannot be regenerated. It *is* the null H3 names, and under it H3 fails on the primary snippet set (§2). Reported as a preregistered negative result.

## 7. What this project is and is not
It is a narrow, hedged result on one model and one task (imitation arm n = 2, masked imitation n = 1): **The size and direction of a fine-tune's readable activation-difference trace on unrelated text are set by where the loss is placed, and loss placement accounts for ~10–12× of the 16.63–22.63× C-vs-A gap while the learning rule's residual has the opposite sign (RL's V is 1.9–2.6× larger): masking prompt tokens removes ~92 % of imitation SFT's trace (C s0: 3.488 → 0.286; s1 gives 3.498 → same to two decimals; V 0.50 → 0.049) at unchanged accuracy (0.935 vs 0.930, p = 1.00, both parsers) and unchanged dose (same data, lr × steps; ‖ΔW‖_F 5.84 vs 6.96), and lands it beside GRPO — within 2× in magnitude (1.4–1.9× A's trace: 1.36× at matched accuracy, A s0; 1.85× against A s1, whose accuracy was not measured), in GRPO's direction (cos 0.62 / 0.49 to A s0 / s1, vs 0.32 / 0.30 to C; A's own seeds agree to 0.68, so "toward A" holds on one seed and is within A's own scatter on the other), with GRPO's format-shaped Patchscope readout. At matched loss placement RL and SFT traces are comparable; per unit weight change the RL trace is larger, and its absolute trace is small mainly because its weight update is small. Two C_masked seeds (V 0.049 / 0.047, cos 0.735 between them).** The raw 16.63–22.63× C-vs-A norm ratio is descriptive; ~12× of it is loss placement and the 1.4–1.9× (1.36 / 1.85) residual is A's 3.5× smaller ‖ΔW‖_F partly offset by its 1.9–2.6× larger V. It is also a worked example of a metric artifact caught before write-up by reading the data, with three independent methods converging on the same items. It is not evidence that "RL leaves no trace," not a replication of Minder, not a causal claim about steering, and not a test of the SGD-specific theory.
