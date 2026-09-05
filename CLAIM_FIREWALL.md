# CLAIM_FIREWALL.md — what may and may not be claimed (updated Sat 2026-09-05 03:45 after the C_masked result; numbers only from docs/RESULTS_DIGEST.md @ 0d9e487 or later and results/REPLICATION_REPORT_C_masked.md @ 19524db)

Applies to the write-up, the executive summary, the application form, and any message to Neel. Read before writing a sentence that contains a number or a mechanism. **Rule added 13:00: every accuracy is stated under both parsers or not at all.**

## 1. Safe claims (conditional on the named artifact, single seed unless stated)

### Accuracy
- "On 200 held-out GSM8K items the base model scores 0.14 under a last-number parser and **0.79** once completions are truncated at the first self-started new question; it solves the problems but rarely stops. GRPO (A) scores 0.94 under both parsers; no cut fires on any A or C completion." — `results/acc_table.md`, `results/acc_table_reparsed.md`, `results/reparse_audit.md` (20/20 rescues genuine; 6.2 % coverage, so a true false-rescue rate under ~5 % is consistent with zero).
- "A vs D_math: 62/6 raw, **22/7 re-scored (p = 0.008)**. A vs base: 162/2 raw, **35/5 re-scored**." Always both. Never the raw pair alone.
- "Cooking SFT alone raised raw accuracy 0.14 → 0.265 (p = 0.004); re-scored 0.79 → 0.54 — cooking SFT *lowers* stopping-robust accuracy." (Say this; it is the honest reading of D.)
- **Gate 2 (amended wording, tags now in):** "A's re-scored gain over the same-domain SFT control is 22/7 (p = 0.008). Across the 68 discordant items, 42 of D_math's 62 losses are cases where it reached and stated the correct answer and then continued generating, while 20 are genuine reasoning errors; all 6 of A's losses are reasoning errors." — `notes/guiv_tags.csv`, `results/llm_tags_68.csv`, `results/acc_table_reparsed.md`.
- **The narrow reasoning-gain claim is now sayable:** "A's advantage over the same-domain SFT control is roughly two-thirds an answer-extraction artifact; on the stopping-robust comparison A still wins 22-7, and the SFT arm makes 20 genuine reasoning errors to the RL arm's 6." Never wider than this.
- **FORMAT is defined narrowly** and the definition must appear wherever the tally does: the model reasoned to the correct answer and stated it, then destroyed it by continuing. A completion that truncates without ever stating the answer is REASONING, not FORMAT (item 186, re-classified and logged).

### Assay and geometry
- "Cooking SFT leaves a trace at positions 1–2 on unrelated text that Patchscope decodes to food vocabulary (rice, tea, tomato, sugar at L15; fried, cooked, salt, rice, sour at L19), 7–8/20 content-relevant tokens vs ≤ 2 for the null under identical λ selection; norm 3.15 vs split-half floor 0.40." — `patchscope_D_*.json`, `token_relevance_*.json`, `perposition_table_C.csv`.
- "Position 0 (first content token; no BOS) carries a large offset shared across arms (D·D_math −0.52, D·A_early 0.61 on math) and is excluded from domain claims." Always with: constancy at p0 is 0.96–0.98 for the *base* activations too.
- "On unrelated text at positions 1–2 (L15; same ordering at L11 and L19) the mean-difference norm is D 3.15, D_math_full 1.20, A 0.21 (seed 1: 0.155), B 0.094, N3 0.046, each above its own split-half floor. Per unit ‖ΔW‖_F the A-vs-math-SFT gap shrinks to ~1.4× and A-vs-cooking to ~3×." Always both normalisations.
- "SFT arms reproduce across seeds (D 0.95–0.98, D_math_full 0.92–0.99 at p1–2); **A does not** (seed 0 · seed 1 = 0.68 at step 150, neutral p1; seed-1 norms 26 % lower)." — `perposition_table_A_seeds*.csv`.
- "A's direction is stable within a run from step 25 (cos 0.87 to final); decodes to numerals and relation symbols; B is orthogonal to A (−0.13) and length-drifting."
- "Math SFT's trace is input-gated (6–13× larger on math than neutral text); cooking SFT's is not (1.3–1.6×); prompt masking accounts for part (0.39 → 1.20 unmasked)."
- Theory scorecard sentences as in `docs/THEORY_NOTE.md` §Scorecard: two predicted-and-observed, two predicted-and-refuted, refinement labelled post hoc.

### Headline (C / C_masked / A) — rewritten Sat 03:45 after the decisive test
- **The claim sentence:** "The size and direction of a fine-tune's readable activation-difference trace on unrelated text are set by where the loss is placed, and loss placement accounts for ~10–12× of the 16.63–22.63× C-vs-A gap while the learning rule's residual has the opposite sign (RL's V is 1.9–2.6× larger): masking prompt tokens removes ~92 % of imitation SFT's trace (C s0: 3.488 → 0.286; s1 gives 3.498 → same to two decimals; V 0.50 → 0.049) at unchanged accuracy (0.935 vs 0.930, p = 1.00, both parsers) and unchanged dose (same data, lr × steps; ‖ΔW‖_F 5.84 vs 6.96), and lands it beside GRPO — within 2× in magnitude (1.4–1.9× A's trace — 1.36× at matched accuracy, A s0; 1.85× against A s1, whose accuracy was not measured), in GRPO's direction (cos 0.62 / 0.49 to A s0 / s1, vs 0.32 / 0.30 to C; A's own seeds agree to 0.68, so "toward A" holds on one seed and is within A's own scatter on the other), with GRPO's format-shaped Patchscope readout." — `results/REPLICATION_REPORT_C_masked.md`, `visibility_table_C_masked.md`, `perposition_table_C_masked*.csv`, `acc_C_masked_s0.json`.
- **The decomposition, always complete:** "The 16.63–22.63× C-vs-A gap is ~12× loss placement × 1.4–1.9× residual; the residual is A's 3.5× smaller ‖ΔW‖_F partly offset by A's *larger* V (0.125/0.092 vs 0.049). At matched loss placement, RL and imitation traces are comparable; per unit weight change, RL's is larger." Never the 12× without the residual; never the residual without its sign.
- **The two alternatives, one sentence each, both labelled tested-and-not-supported:** learning rule (preregistered 02:00 with V ≥ 0.30 as its pass criterion; observed 0.049) and dose (C_masked has C's dose and loses the trace anyway).
- **The D_math pair is the observation that motivated the test and anchored the 0.18 threshold (D_math_full's own V), not corroboration:** its effect is 3× (V 0.179 → 0.059 at ‖ΔW‖_F −2 %) against C's 10× (V 0.501 → 0.049 at ‖ΔW‖_F −16 %); D_math's masked-token fraction is not recorded. Say it in that role.
- **What survives as an RL-vs-SFT difference:** "cross-seed reproducibility — C 0.98, D 0.95–0.98, A 0.68" and nothing else.
- Prior numbers remain sayable as observations: C vs A raw 16.63–22.63×; V ratio `4.0–5.5× (four (C, A) seed pairs: 4.00, 5.45, 4.01, 5.47)`; C·A = +0.505 above all 50 draws; B·A = −0.13 below all 50 (orthogonal to slightly negative).
- "The post-hoc refinement's prediction for C (positive projection on d̂_A above B/random) was written before C finished and was observed." Never "preregistered". "The loss-placement prediction (M0) *was* preregistered, at 02:00 Sat, with thresholds, before the run." Say both.
- Hedges that travel with the claim: one C_masked seed, one masking pattern; content vs position of supervision not separated (masking removes both the early positions and the human-written problem statements); no ablation from the fine-tuned model, so "not load-bearing for behaviour" is correlational; one model, one task.

### Steering
- "Rescaled to η_ref = 11.24 at layer 15, A's direction at α = 0.25 raises base accuracy to 0.200 (p = 0.013) against a five-seed matched-norm random null of 0.110–0.170; C 0.215 at α = 0.5; D_math_full 0.285. **Accuracy and EOS rate rise together, so these gains are not separated from stopping effects.** At α ≥ 1 every direction including random collapses." — `results/steer_table.md`.
- Re-scored steering (`results/steer_table_reparsed.md`, citable, digest §10c): unsteered 0.790; A α=0.25 0.815 (p = 0.52); C α=0.5 0.730; D_math_full α=0.5 0.650 (p = 0.0003, lower); random 0.765–0.820. **Always with:** the 20/20 rescue audit covered base, B and D_math only — steered completions were re-parsed with the same rule and not audited, so these numbers inherit an unchecked parser. The §2 base 0.140 and the steering-run unsteered 0.130 are separate decodes.
- "Natural-norm steering (≈2 % of activation norm) changed nothing; not a test of the causal claim."
- Guiv's own reading of the α = 1 generations, verbatim, with the selection rule (first 8 prompts, seed 0). ⏳

### Visibility V
- V per arm from `results/visibility_table.md`, always with ‖ΔW‖_F beside it; for A always with the seed spread (0.125 / 0.092, ratio 1.36, ‖ΔW‖_F 1.675 / 1.682) in the same sentence.

## 2. Never claim
- **"GRPO is hard to detect by activation diffing because it never supervises prompt tokens"** or any causal "which is why". V(A) 0.125/0.092 > V(C_masked) 0.049: per unit weight change RL leaves *more* trace than masked imitation; A's absolute trace is small mainly because ‖ΔW‖_F is 1.68. Say: "at matched loss placement, comparable".
- **"The learning rule explains the C-vs-A gap"** / "V 4.0–5.5× is the learning-rule claim" — superseded 03:00; the V ratio is now an observation whose explanation is loss placement.
- "The dose confound explains the gap" — excluded (C_masked has C's dose).
- "Loss placement explains Minder et al." — they supervise whole documents; ours is one model, one task, one seed. Say "consistent with".
- "The trace is not the behaviour" as a causal statement — no ablation was run.
- **"GRPO lifted accuracy 0.14 → 0.94"**, or any accuracy under one parser only. Never a raw McNemar without its re-scored pair.
- "A improved reasoning" wider than the tagged share above.
- **The 67/68 human-judge concordance as a blind agreement rate.** Guiv read all 68 non-blind after the judge; item 186 is the one disagreement (FORMAT → REASONING); item 56 was absent from the judge's output and was added by Guiv. It is verification of the judge's tags against raw text, not an independent agreement rate; **no blind-20 rate exists**.
- **The split-half floor as a detectability null.** It is 12–15 % of the trace for every trained arm (B 18 %, N3 29 %) and measures snippet-half stability. The nulls for "is there a trace?" are N1 (base-vs-base halves, 0.747 at neutral p1, unpaired) and N3.
- **N3 as a norm-matched null.** Its ‖ΔW‖_F is 2.069, 24 % above A's 1.675; it is a reference. A's V seed ratio is 1.363 (0.1252 / 0.0919; 1.362 from the rounded values).
- **That the discordant tagging was arm-blind.** The X/Y blinding leaks: A ends `#### N` + `The answer is: N`, D_math uses `<<…>>` or step-numbered markdown, so anyone who has seen the arms unblinded can identify them. Disclose whenever the tally is used.
- **V as a stable per-arm constant for A.** "17×" without "on seed 0; ~22× on seed 1".
- "Steering shows the direction is causally active" without the stopping confound in the same sentence; never present steering and Gate 2 as *independent* support — they share one confound.
- "Gate 1 passed on position-0 geometry" / "the position-0 offset is the cooking trace".
- "Replicates Minder et al." (conceptual replication; deviations listed). Any Minder token-count comparison without a citation in `SOURCE_INDEX.md`.
- "GRPO leaves no trace" / "RL is fundamentally different from SFT". Say: no D_math-like readable trace detected at positions 1–2 with Patchscope at L15/19 in this run.
- "A−B isolates the reward-specific component"; "B is a generic-optimization control".
- Any causal "context gate" from the neutral-vs-math contrast.
- "Norm matching controls for update size" (functional dose not matched).
- Arm B's training curve (reward ≈0.07, truncation 0.79, length 456) as a verified number — **no local source**; cite only as "from a pod log destroyed on termination, unverifiable" unless re-derived.
- The TF-IDF lexical baseline as a control of any kind (8/150 correct, below the null).
- Six-way judge accuracy on real lists (calibration only).
- "A is unreadable" without "with Patchscope at L15/19, an instrument that reads math SFT only weakly on neutral text".
- "RL's trace is less constant than SFT's" (refuted). "B shares A's geometry" (refuted).
- "max over λ" relevance counts without the null under identical selection; relevance counts that include bare digits/newlines.
- The module-family split as evidence that arms change "the same kind of weights" (N3 has the identical split).
- "Cos to D at p0 rises 0.36 → 0.61 as the generic offset accumulates" — **wrong, corrected**; neutral p0 is flat (0.357 → 0.335).
- Any number from a MOCK file, any number not in VERIFY.md, any figure from mixed inputs.
- "Positions 1–4" in the limitation — amended to "1–2".
- "J-Lens", "SGD arm", "agent-rubric readout", "scaling test", "C relevance grading" as things done.
- Equivalence between arms from a failed inequality.
- Anything about other models, other tasks, long-horizon persistence, or Adam-vs-SGD from these runs.
- Any claim about paper potential.

## 3. Mandatory limitations (verbatim; amended 2026-09-04)
1. "Position 0 carries the largest and most constant component and it is shared across every fine-tuned arm, so pos-0 norm and constancy measure 'was fine-tuned', not what was learned; every domain claim rests on **positions 1–2**." *(Amended 2026-09-04: "1–4" → "1–2"; positions 3–4 are reported but no claim rests on them.)*
2. "D uses [verified token count from `data/cooking.jsonl`, or omit the comparison] training tokens, one model, one seed, layer 15 primary; a readability null here is a statement about this dose and these instruments, not about the phenomenon."
3. "Norm comparisons across arms are descriptive: functional dose (ΔW norm, KL, accuracy change) is not matched; the unit of every judge statistic is the block; the pipeline was agent-built and the numbers Guiv recomputed by hand are listed in VERIFY.md."
4. **New:** "Held-out accuracy is dominated by whether a model stops: the base scores 0.14 or 0.79 depending on the parser. All accuracy-based comparisons are reported under both, and the steering gains at α ≤ 0.5 are not separated from the same stopping effect."
5. **New:** "The adapters and activation caches from the first pod were destroyed when it was terminated; every round-1 number is re-derivable only by retraining (C s1 and C_masked adapters are kept on the Mac). Greedy bf16 decoding is not run-to-run reproducible (±2 items)."
6. **New (03:45):** "The loss-placement result rests on one C_masked seed and one masking pattern, on one model and one task. Masking prompt tokens removes both the early, context-free positions and the human-written problem statements, so whether the trace tracks supervision *position* or supervision *content* is not separated here. No ablation from the fine-tuned model was run, so the statement that the removed trace is not load-bearing for accuracy is correlational."

## 4. Guiv's other research programme (unchanged)
Public: ETH BSc Mathematics + MSc CS (Machine Intelligence); sole-authored **COLM 2026 Efficient Reasoning *workshop*** paper with public code; broad description of an unpublished GRPO/RLVR programme (preregistration, claim firewalls, a killed hypothesis) **without numbers**; the shuffled-reward control as a *method*; the plain-SGD finite-G factorisation as motivation. Private: any unpublished metrics, internal gate names, failed-branch details, "submitted to ICLR" language, private hashes; "silent/voiced/terminal bank". Never: COLM as main-conference; invented collaborators; legacy thesis numbers.

## 5. Time and LLM-use disclosure (form Q16 and the doc)
State which agents did what (round-1 overnight build unsupervised; pod runner in Claude Code; Codex tasks; chat critics; blind LLM tagger on the 68 discordant items); that agent hours are not Guiv's hours; which numbers Guiv recomputed himself and how; how many raw items he read (all 68 discordant items, non-blind, after the LLM judge — 67/68 concordant, item 186 the disagreement, item 56 added by him; no blind-20 rate exists; 5 Patchscope lists blind; 8 prompts × 4 arms of steered generations; 5 cooking rows; 5 black-box rows per arm); which parts were **not** independently checked and how surprised he would be by an error in each; the attempt ledger (two dead A_early launches; vLLM abandoned; three sync-reverted result files; the digest untracked until 14:00; the §7 stitching error). Toggl covers Guiv's active hours only. Executive summary and form answers are Guiv's prose; LLMs critique only.
