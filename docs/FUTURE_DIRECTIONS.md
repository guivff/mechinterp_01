# FUTURE_DIRECTIONS.md — what would make this a truly impressive piece of work (post-deadline; not for the submission)

Rewritten Sat 2026-09-05 03:45 after the C_masked result and the second Neel-style evaluation (`docs/EVALUATION_NEEL_STYLE_v2.md`). The 01:30 version's Tier 0 (dose matching) is now demoted: C_masked has C's dose and loses the trace anyway, so dose is not the explanation of the gap and matching it is a control, not the experiment. Ordered by how much each changes what can be claimed.

## Tier 0 — the experiment that decides what the finding is

### 0.1 Position or content of supervision (one afternoon, ~$5)
Masking prompt tokens removes two things at once: loss on the first ~30–60 positions of every sequence (context-free, where Minder's per-position trace lives) and loss on human-written problem statements (content the base model cannot predict). Three C variants at C's dose, seed 0 and 1:
- **C_scrambled-prompt**: loss on prompt positions, but the prompt tokens are shuffled or replaced by random GSM8K prompts, so position is supervised and content is not.
- **C_prompt-only**: loss on prompt tokens only, completions masked.
- **C_masked-with-context**: completion-only loss but with a random unrelated document prepended as unsupervised context, so the supervised tokens are never early positions.
Prediction under "position": scrambled-prompt and prompt-only leave large traces; masked-with-context leaves a small one. Under "content": only content-bearing prompts leave a large trace. Either answer is a clean, quotable rule for the forensics use of activation diffing. **This is the experiment to run first.**

### 0.2 Seeds on the arm that carries the claim
C_masked seed 1 (and 2). If V stays ≤ 0.1 the result is settled at n = 3; if it jumps, the Saturday claim was a fluke and the write-up said so was possible. Also A seeds 2–3, for the reproducibility contrast (0.68) that is now the only surviving RL-vs-SFT difference.

## Tier 1 — "bias term or something deeper?", answered directly

### 1.1 Ablate the trace from the fine-tuned model
For C, C_masked, A, D: subtract d (own; random of matched norm; another arm's) from the residual at every position of the *fine-tuned* model and re-measure held-out accuracy under the stopping-robust parser with EOS rate beside it. The loss-placement result predicts: ablating d_C leaves C's accuracy intact (the large trace is a prompt-prior term, not the capability); ablating d_A or d_C_masked is the open question. `tools/ablate_trace.py` is specified in `prompts/round5/E1_CLAUDE_CODE_LONG_TERM.md` Phase 0.

### 1.2 Where the prompt-supervision trace lives
Per-position profile (0–10, not 0–4) for C vs C_masked; per-layer; per-module effective rank of ΔW (participation ratio). Hypothesis: the masked-away component is low-rank and early-position; the residual shared with A is distributed.

## Tier 2 — reproducibility of the RL trace (the surviving RL-specific fact)
Few-prompt dominance (T2 Pass 1): A's cross-seed cosine should rise with steps, G, and prompts per step, and its trace norm should track the fraction of mixed-advantage groups per step. Log group advantages next time. Compare with C_masked's cross-seed cosine — if masked imitation reproduces at 0.9+ while A stays at 0.7, "on-policy sampling makes the trace seed-specific" is a finding in its own right.

## Tier 3 — closing the stopping confound properly
Primary metric = truncate-at-first-new-question; EOS rate and cap-hit rate as first-class columns; steering with forced length; steer the fine-tuned model away from its own direction (−α).

## Tier 4 — generality
Second base model with a BOS token (Gemma 3 4B or Qwen3.5-9B): removes the position-0 artifact at source and tests whether masking collapses the trace there too. Then the forensics classifier: given base + fine-tune, predict "loss touched prompt/document-start tokens: yes/no" from trace features; test on held-out fine-tunes. Patchscope with adaptive λ and N2 as a Patchscope null.

## What I would not do
More arms of the old kind. The dose family as a headline experiment (run it once as a control curve). A six-way judge. TF-IDF anything. Anything that adds a section without changing a claim.

## The paper, if Tier 0–1 hold
Title-shaped claim: *The readable activation-difference trace of a fine-tune is a signature of prompt-token supervision: completion-only SFT and on-policy RL leave small, comparable traces, and the large trace of full-sequence SFT can be removed by masking without touching the behaviour.* Three figures: V by arm split by loss placement (three seeds); the position/content split; the ablation. Two base models. That is a workshop paper if 0.1 is clean, and a main-track candidate if 1.1 says the masked-away trace is not load-bearing while the shared residual is.
