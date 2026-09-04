# CLAIM_FIREWALL.md — what may and may not be claimed

Applies to the write-up, the executive summary, the application form, and any message to Neel. Read before writing a sentence that contains a number or a mechanism.

## 1. Safe claims (conditional on the named gate passing, with the frozen artifact cited)
- "In this setup, narrow SFT on a 2k-document cooking corpus produced a base→fine-tuned mean activation difference whose norm-matched logit-lens tokens a blind judge classified as cooking at X% (block-level CI), versus Y% for a TF-IDF baseline and Z% for base-vs-base / random / untrained-LoRA nulls." — requires Gate 1.
- "GRPO with intact reward raised held-out GSM8K accuracy from a to b (paired test p=…); shuffled-reward GRPO did not." — requires Gate 2 and the paired test.
- "A's norm-matched trace decoded as [domain/none] at X% vs C's/D's Y% on neutral text and X'% on math text (block-level CIs)." — descriptive, single seed unless stated.
- "The mean-offset energy share (constancy) was c_A vs c_D." — with the exact definition from PROTOCOL_NOTES §3 (uncentered energy share, not variance explained).
- "The zero-sum structure of group-normalized advantages cancels, at first order, any gradient component shared across a prompt's G completions; this motivated the prediction that A's trace is less constant and less topic-readable than C's or D's." — as motivation/heuristic, not theorem.
- "D is a reduced-budget conceptual replication of Minder et al. (500 vs 10,000 samples; pooled positions after skipping 4 vs per-position 0–4; ~0.5M vs ~20M training tokens; base-model steering vs fine-tuned-model steering; no Patchscope)."

## 2. Never claim
- "Replicates Minder et al." (say conceptual replication and list the deviations).
- "GRPO leaves no trace" / "RL is fundamentally different from SFT" (say: no D-like readable mean trace was detected in this run at this layer).
- "A−B isolates the reward-specific component" (say: descriptive contrast between two diverging single-run trajectories associated with intact reward assignment).
- "B is a generic-optimization control" (say: difficulty-gated random-gradient control that preserves each group's reward multiset).
- "The trace switches on in math contexts" / any causal gate from H4 (say: input-dependent readout; corpus statistics confounded).
- "Norm matching controls for update size" (it rescales one vector; functional dose — ΔW norm, KL, behavioral change — is not matched; note logit-lens ranks are scale-invariant anyway).
- Any accuracy stated as if 100 judge calls were 100 observations. The unit is the block (K=10) or the prompt cluster.
- Any number from a MOCK file. Any number not in VERIFY.md.
- Judge accuracy against "chance 1/6" without also showing always-math / always-none baselines and the confusion matrix.
- "J-Lens" as a method used (it was dropped; mention as future work only).
- Equivalence between A and C from a failed H2 (say: the predicted 0.15 gap was not observed).
- Anything about long-horizon persistence, other models, other tasks, or Adam-vs-SGD from these runs.
- **V (‖d_neutral,p1‖ / ‖ΔW‖_F) as a stable per-arm constant for A.** V is seed-stable for D (0.3837 / 0.3910, ratio 1.019) and D_math_full (0.1789 / 0.1893, 1.058) but not for A (0.1252 / 0.0919, ratio 1.363) — and A's two adapters have near-identical ‖ΔW‖_F (1.675 / 1.682), so the spread is entirely in the activation-space numerator. Any V-based statement about A carries the n=2 seed spread in the same sentence.
- **Held-out accuracy stated only under the preregistered last-number parser.** That parser scores the continuation when a model answers and then starts a new question, which is most of what it measures on the base model and on B. Any accuracy claim gives both parsers: base 0.140 raw / 0.790 re-scored, B 0.075 / 0.810, D_math 0.660 / 0.865, A 0.940 / 0.940, C 0.930 / 0.930 (`results/acc_table_reparsed.md`). "GRPO lifts accuracy from 0.14 to 0.94" is not sayable without "0.79 to 0.94 once the stopping artifact is removed".

## 3. Guiv's other research programme (from the application packet)
- Public and usable: ETH BSc Mathematics + MSc CS (Machine Intelligence); sole-authored **COLM 2026 Efficient Reasoning *workshop*** paper on candidate-free test-time aggregation with public code; broad description of an ongoing unpublished GRPO/RLVR research programme (preregistration, claim firewalls, a killed hypothesis) **without numbers**; the shuffled-reward control as a *method*; the general fact that plain-SGD finite-G expected updates factor through a token-average class-contrast field (as motivation for the theory note).
- Private: exact unpublished metrics, internal gate names, failed-branch details, any "submitted to ICLR / spotlight" language, private hashes. The words "silent", "voiced", "terminal bank" from that programme do not appear here.
- Never: COLM as a main-conference paper; the current GRPO programme as submitted; invented collaborators or mentors; legacy thesis numbers (MATH-500 pass@8, entropy/KL ratios, residual-stream mech-interp extension) unless verified against the thesis.

## 4. Time and LLM-use disclosure (form Q16 and the doc)
- State: which agents (Claude Code / Codex / chat) did what; that pipeline construction ran autonomously overnight; which numbers Guiv recomputed himself; how many raw transcripts/generations he read; which parts were **not** independently checked and how surprised he would be by an error in each.
- Toggl screenshot covers Guiv's active hours only; the attempt ledger in CHANGELOG.md is referenced.
- Executive summary and form answers: Guiv's own prose. LLMs may critique them for clarity with an anti-sycophancy prompt; they do not draft them.
