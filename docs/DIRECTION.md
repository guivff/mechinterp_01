# DIRECTION.md — where the research stands and where it goes (Sat 2026-09-05 03:45, after the decisive test)

Narrative counterpart to `PROGRAM_STATE_CURRENT.md`. State says what was measured; this says what it means, what the submission argues, and what the programme does after the deadline. Numbers must be re-checked against the digest before use.

## 1. The thesis, as it now stands
Minder et al. showed that narrow SFT leaves a mean activation difference on *unrelated* text that decodes to the training domain. We asked whether on-policy GRPO does too, and if not, why not — data, behaviour, learning rule.

The answer this project can defend is none of the three we listed: **it is where the loss is placed.** Imitation SFT on the RL policy's own correct samples (C) reaches the RL run's accuracy and leaves a trace 16.63–22.63× larger. The same run with prompt tokens masked — GRPO's loss placement, at C's data and dose — loses ~92 % of that trace at unchanged accuracy, lands within 2× of the RL run, points in the RL run's direction (cos 0.62, against 0.68 between the RL run's own seeds), and decodes to the same format tokens. The masked/unmasked math-SFT pair shows the same effect independently (V 0.059 vs 0.179). Two readings we had carried since Friday — "the learning rule" and "the dose" — were both tested by this one run and both fail: the learning-rule reading on its own preregistered threshold (V ≥ 0.30 vs observed 0.049), the dose reading because C_masked has C's dose.

What must not be said, because our own table contradicts it: that GRPO is invisible *because* it never supervises prompts. Per unit weight change the RL run leaves *more* trace than masked imitation (V 0.125/0.092 vs 0.049); its absolute trace is small mainly because its weight update is small (‖ΔW‖ 1.68 vs 5.84). The honest sentence is "at matched loss placement, RL and imitation traces are comparable." The one RL-vs-SFT difference that survives is cross-seed reproducibility (C 0.98, A 0.68).

## 2. What the two days taught us — two self-refutations
Friday 13:00: a stopping-robust re-parse showed the base model already solves 0.79 of GSM8K but rarely emits EOS; under the preregistered parser it scores 0.14. Two of three headline comparisons shrank or inverted (Gate 2 62/6 → 22/7; steering became a negative result); the C-vs-A comparison was untouched.

Saturday 02:00–02:56: the theory pass proposed loss placement as the alternative to our headline, we preregistered a 20-minute test with thresholds, and the test went against us. The write-up says both of these in its first paragraph. Neel's stated selection criteria are truth-seeking and looking at the data; these two events are the evidence, and they are worth more than the headline they replaced.

**The submission's meta-claim:** the project tested its own headline twice, on its own initiative, and reports what it found — including the number (V(A) > V(C_masked)) that blocks the tidy version of the new story.

## 3. What the write-up must contain that a reader would otherwise catch
- Both parsers, everywhere.
- V(A) > V(C_masked), stated next to the loss-placement claim.
- One C_masked seed; content-vs-position not separated; no ablation, so "not load-bearing" is correlational.
- A's cross-seed instability (0.68) and the 16.63–22.63× range with the four-pair V string.
- N2 as a preregistered null that is logit-lens only and nulls the cosine; H3 refuted.
- Arm B's training curve unverifiable; three sync-reverted files; the digest untracked until Fri 14:00; the §7 correction; the 4.02 → 4.01 correction.
- Adapters from the first pod destroyed; C s1 and C_masked adapters kept.
- Blinding leak, non-blind concordance, Cowork's two contaminations, the withheld Patchscope reading.
- Two replication aborts, one $0.09 probe pod, one overwritten identity-check file, agent clock drift.

## 4. What is *not* claimed
No causal "invisible because unsupervised prompts"; no "RL is fundamentally different from SFT"; no "learning rule" as explanation; no "explains Minder et al." (consistent with); no reasoning-gain claim wider than the tagged share; no steering support; nothing about other models, tasks, optimizers, or paper potential.

## 5. Open questions the submission names as next steps (two sentences each)
1. **Position or content?** Masking removes both the context-free early positions and the human-written problem statements. A C with loss on prompt positions but scrambled prompt content, and a C with loss on completions only but the prompts prepended as unsupervised context, separate the two.
2. **Does it reproduce?** C_masked seed 1 and a second base model with a BOS token (which also removes the position-0 artifact at source).
3. **Is the removed trace load-bearing?** Ablate d_C from the fine-tuned C at every position and re-measure accuracy under the stopping-robust parser; random and cross-arm directions as controls; same for A and C_masked.
4. **Why does RL not reproduce across seeds when masked imitation of RL does?** More A seeds; A's cross-seed cosine vs steps, G, prompts per step (few-prompt dominance).
5. **A stopping-controlled metric from the start**, with EOS rate beside every accuracy.
6. **N2 as a Patchscope null**, so the relevance count has its preregistered distribution.

## 6. After the deadline (if pursued)
The paper this points at is smaller and sharper than the one we were planning at 01:30: *the readable activation-difference trace of a fine-tune is a signature of prompt-token supervision; completion-only SFT and on-policy RL are alike under it.* Three seeds per arm, the position/content split, the ablation, a second base model. Roughly 2 pod-days. The first experiment is the position/content split; if it comes out "position", the forensics use of activation diffing (Neel's stated interest) has a clean rule: a large readable mean trace means the loss touched document-start tokens.

## 7. What would change the direction
- C_masked seed 1 with V ≥ 0.30 → the result was a seed fluke; back to inconclusive.
- Scrambled-prompt C with a large trace → position, not content; the assay reads "where", not "what".
- Ablating d_C leaving C's accuracy intact → the trace is a bias term in Neel's sense; the "deeper" question moves to what carries the accuracy.
- A second base model with a BOS token failing to show the masking effect → Qwen-specific / position-0-specific.
