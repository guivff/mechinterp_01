# CLAIM_FIREWALL.md — what may and may not be claimed (updated Sat 2026-09-05 ~03:30, C1; numbers only from docs/RESULTS_DIGEST.md at the commit that merges `replication` (c852658) or later)

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
- "A's direction is stable within a run from step 25 (cos 0.87 to final); decodes to numerals and relation symbols; B is **orthogonal to slightly negative** relative to A (−0.13, below all 50 random draws) and length-drifting."
- "Math SFT's trace is input-gated (6–13× larger on math than neutral text); cooking SFT's is not (1.3–1.6×); prompt masking accounts for part (0.39 → 1.20 unmasked)."
- Theory scorecard (`docs/T1_THEORY_BLOCKS.md` Block 3; digest §9 scorecard note, §3, §12b): **P1 half-observed** — A decodes to digits/relation symbols, not topic words, but C (both seeds) decodes to the same digits/`=`/`→`, so the discriminating half ("C → math vocabulary") failed; **P2 observed** with the caveat that same-data C (1.5/2.2) and same-domain SFT (8.4) show gating tracks training domain, not learning rule; **P3 refuted**; **P4/H3 preregistered and refuted**; the post-hoc refinement's C prediction **observed and dose-robust because it is a cosine**. Never "two observed, two refuted".

### Headline (arm C)
- "Trained on A's own correct samples to A's accuracy (0.930 vs 0.940, p = 0.77, **identical under both parsers**), imitation SFT leaves **4.0× (A seed 0) to 5.4× (A seed 1) more trace per unit ‖ΔW‖_F** on unrelated text than the GRPO run it imitates — **that per-unit factor V is the claim.** The raw norm ratio (16.63× / 22.57×) is descriptive: it factorises as ‖ΔW‖_F × V = 4.157 × 4.001 and 4.139 × 5.453, and **C was trained at lr 1e-4 × 225 steps against A's 3e-5 × 150 — a 5.0× lr×steps mismatch that plausibly accounts for the ‖ΔW‖ factor. That mismatch is the primary open confound and is named in the same sentence as the claim.** C is partially aligned with A (cos 0.50, above all 50 random draws) while the shuffled-reward arm is **orthogonal to slightly negative (−0.13, below all 50 draws)**." **C is two seeds: seed 1 (results/REPLICATION_REPORT.md, merged) reproduces seed 0 on accuracy (0.925 vs 0.930, p = 1.00), trace (3.498 vs 3.488), ‖ΔW‖_F (6.958 vs 6.963), V (0.5027 vs 0.5010) and cosine to A (0.504 vs 0.505) — all within 2 % — and C s0·C s1 = 0.98 while A s0·A s1 = 0.68.** With four seed pairs the raw ratio is **16.63× / 22.57× / 16.68× / 22.63×** (`results/trace_ratio_C_A_seeds.csv`; quote as a range 16.63–22.63×, two decimals) and V is 4.00× / 5.45× / 4.02× / 5.47× (4.0–5.5×). **Loss placement (`docs/T2_THEORY_PASSES.md` §1.0) is the second named alternative and goes in the same paragraph as the claim:** masked math SFT has V 0.059, below both A seeds; unmasking the same corpus alone takes it to 0.179 at unchanged ‖ΔW‖_F; C was trained unmasked and GRPO supervises only completion tokens. The decisive test is a completion-only C (`C_masked`): **V ≤ 0.18 puts the gap with loss placement, ≥ 0.30 leaves it with the learning rule — ⏳ not yet run (gate 07:00; if absent by 07:30 write "not run before submission; first experiment in Future work").** Always with: fixed-budget unmasked SFT (12 % of corpus), inherits A's formatting, C·D_math_full 0.55 leaves an SFT/corpus reading open, no dose-matched C family was run.
- "The post-hoc refinement's prediction for C (positive projection on d̂_A above B/random) was written before C finished and was observed (C s0 0.505, C s1 0.504 on A s0)." Never "preregistered". It is a cosine, so it is the C-vs-A statement that neither the dose nor the loss-placement confound touches; the same is true of the cross-seed contrast (C 0.97–0.98 vs A 0.68), which gets its own sentence.

### Steering
- **Steering is a negative result. It does not survive the parser correction and is no longer a safe positive claim; it is not in the contributions list and carries no "causal" language.** The α ≤ 0.5 numbers (A 0.200 at p = 0.013; C 0.215; D_math_full 0.285 against a five-seed random null of 0.110–0.170) are **raw last-number parser**. Re-scored on the same stored completions (`results/steer_table_reparsed.md`): unsteered **0.790**, A α=0.25 **0.815 (p = 0.52)**, C α=0.5 **0.730 (p = 0.13)**, D_math_full α=0.5 **0.650 (p = 0.0003 in the opposite direction)**, random α=0.25 0.765–0.820. **No direction beats the unsteered base under the stopping-robust parser; the best raw cell is significantly worse.** Sayable: "the raw-parser steering gains were the steering making the base model stop more often; under the stopping-robust parser they vanish." — `results/steer_table.md`, `results/steer_table_reparsed.md`.
- "Natural-norm steering (≈2 % of activation norm) changed nothing; not a test of the causal claim." **Every steering accuracy is labelled with its parser wherever it appears.** Amplification is stated as **16–33×** (η_ref × α / ‖d_A‖ = 11.243 × {0.25, 0.5} / 0.17; 66× at α = 1) — never "~50×".
- Guiv's own reading of the α = 1 generations, verbatim, with the selection rule (first 8 prompts, seed 0). ⏳

### Visibility V
- V per arm from `results/visibility_table.md`, always with ‖ΔW‖_F beside it; for A always with the seed spread (0.125 / 0.092, ratio 1.36, ‖ΔW‖_F 1.675 / 1.682) in the same sentence.

## 2. Never claim
- **"GRPO lifted accuracy 0.14 → 0.94"**, or any accuracy under one parser only. Never a raw McNemar without its re-scored pair.
- "A improved reasoning" wider than the tagged share above.
- **The 67/68 human-judge concordance as a blind agreement rate.** Guiv tagged after seeing the judge's output; it is verification of the judge's tags against raw text, not an independent agreement rate, and is reported as such.
- **That the discordant tagging was arm-blind.** The X/Y blinding leaks: A ends `#### N` + `The answer is: N`, D_math uses `<<…>>` or step-numbered markdown, so anyone who has seen the arms unblinded can identify them. Disclose whenever the tally is used.
- **V as a stable per-arm constant for A.** "17×" or "16.6×" alone: the raw ratio is quoted to two decimals from `results/trace_ratio_C_A_seeds.csv` as the four-pair range **16.63–22.63×** (or the pair named), never as a single number.
- **The raw C/A trace ratio (16.63–22.63×) as a learning-rule claim.** Never state it without the per-unit factor V (4.0–5.5×), the lr×steps mismatch (C 1e-4×225 vs A 3e-5×150, 5.0×) **and the loss-placement alternative (masked SFT V 0.059 < A; unmasked 0.179; C unmasked, GRPO completion-only)** in the same paragraph. The learning-rule claim is V; the raw ratio is descriptive; V itself is confounded with loss placement until C_masked is run.
- **"C single seed."** C has two seeds since the `replication` merge; say "C two seeds (accuracy, trace, ‖ΔW‖_F, V, cosine all within 2 %)".
- **"The difference is the weighting."** Deleted everywhere — it asserts a learning-rule cause for a ratio that is not dose-matched.
- **Steering as a positive or causal result, in the contributions list, or any steering accuracy without its parser label**, or the raw steering gains as causal support: under the stopping-robust parser no direction beats the unsteered base (0.790) and D_math_full at α=0.5 is significantly worse (0.650, p=0.0003).
- **"anti-aligned"** for A·B. One wording everywhere: **"orthogonal to slightly negative (−0.13, below all 50 random draws)"**.
- **N2 as the null for the Patchscope relevance count.** N2 nulls the *cosine* statistic (H3); the relevance null is N1 under identical λ selection.
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
5. **New:** "The adapters and activation caches were destroyed when the pod was terminated; every reported number is re-derivable only by retraining. Greedy bf16 decoding is not run-to-run reproducible (±2 items)."

## 4. Guiv's other research programme (unchanged)
Public: ETH BSc Mathematics + MSc CS (Machine Intelligence); sole-authored **COLM 2026 Efficient Reasoning *workshop*** paper with public code; broad description of an unpublished GRPO/RLVR programme (preregistration, claim firewalls, a killed hypothesis) **without numbers**; the shuffled-reward control as a *method*; the plain-SGD finite-G factorisation as motivation. Private: any unpublished metrics, internal gate names, failed-branch details, "submitted to ICLR" language, private hashes; "silent/voiced/terminal bank". Never: COLM as main-conference; invented collaborators; legacy thesis numbers.

## 5. Time and LLM-use disclosure (form Q16 and the doc)
State which agents did what (round-1 overnight build unsupervised; pod runner in Claude Code; Codex tasks; chat critics; blind LLM tagger on the 68 discordant items); that agent hours are not Guiv's hours; which numbers Guiv recomputed himself and how; how many raw items he read (20 blind discordant tags + agreement rate with the LLM tagger; 5 Patchscope lists blind; 8 prompts × 4 arms of steered generations; 5 cooking rows; 5 black-box rows per arm); that he had seen a characterisation of the original 20 discordant items before tagging and drew his sample from the other 48; which parts were **not** independently checked and how surprised he would be by an error in each; the attempt ledger (two dead A_early launches; vLLM abandoned; three sync-reverted result files; the digest untracked until 14:00; the §7 stitching error). Toggl covers Guiv's active hours only. Executive summary and form answers are Guiv's prose; LLMs critique only.
