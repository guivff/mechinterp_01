# PROGRAM_STATE_CURRENT.md — canonical state snapshot

**As of:** Sat 2026-09-05 ~02:00 Europe/Zurich (≈7 h to deadline 08:59; new-numbers cutoff 06:00; 06:00 go/extend decision; submit by 08:30). Numbers: `docs/RESULTS_DIGEST.md` @ `ead8c40` or later (sole citable source). Pod terminated 14:32 Fri; adapters and caches destroyed. A C-seed-1 replication is running on a separate pod/branch (`replication`) with a 06:00 landing cutoff — not merged, not cited until it lands.

## Objective (unchanged)
Does on-policy GRPO leave a readable mean activation-difference trace on unrelated text the way narrow SFT does (Minder et al.)? Is any RL trace due to the data, the behaviour, or the learning rule?

## Status: DATA CLOSED. VERIFICATION COMPLETE EXCEPT GUIV'S COLUMNS. WRITING.
An independent Neel-style evaluation (`docs/EVALUATION_NEEL_STYLE.md`) rated the project high-borderline-to-accept if written up well, borderline-reject as the summary currently reads: P(exploration phase) ~45 %, ~60 % with a clean own-voice summary and a C-vs-A Figure 1. Its top objection — the headline is not dose-matched — is now the first thing the write-up addresses.

## The result, as the write-up now states it
1. **Headline (C vs A).** Imitation SFT on A's own correct samples reaches A's accuracy (0.930 vs 0.940, p = 0.77, identical under both parsers). Its neutral-text trace is **16.6–22.6× larger** than A's across A's two seeds (3.49 vs 0.21 / 0.155). **The learning-rule claim is the per-unit-‖ΔW‖ factor V: 4.0–5.4×.** The other ~4× is ‖ΔW‖_F (6.96 vs 1.68), and C was trained at lr 1e-4 × 225 SFT steps vs A's 3e-5 × 150 GRPO steps — a ~5× lr×steps mismatch that plausibly accounts for it. **Named as the primary open confound in the same sentence as the claim.** C single seed. C·A = +0.505, above all 50 random draws; B·A = −0.13, below all 50 (orthogonal to slightly negative). "The difference is the weighting" is deleted everywhere.
2. **Assay validated (Gate 1).** Cooking SFT decodes via Patchscope to food words (7–8/20 vs null ≤ 2); norm 3.15 vs floor 0.40; position 0 excluded as a generic first-token offset. SFT arms reproduce across seeds (0.92–0.98).
3. **The RL trace is small, format-shaped, as constant as SFT's, and does not reproduce across seeds** (A s0·s1 = 0.68). Mean-offset share A 0.258 vs D_math_full 0.249 vs D 0.277 — **this is the answer to "bias term or deeper?": at this dose the RL trace is as constant as SFT's; what differs is magnitude and reproducibility. No ablation from the fine-tuned model was run, so "deeper" is untested here.**
4. **Gate 2 survives, shrunk, with tags.** A vs D_math raw 62/6 → re-scored **22/7 (p = 0.008, suggestive only by Neel's stated standard)**. Of the 68 discordant items: D_math 42 FORMAT / 20 REASONING; A 0 FORMAT / 6 REASONING. Narrow reasoning-gain claim sayable (firewall §1). The 7th re-scored D_math-only item is a rescued one.
5. **Steering** clears a five-seed random null at α ≤ 0.5 but all accuracies are **raw-parser** and rise with EOS rate — movements within the format-failure regime; not independent of Gate 2. Guiv's reading of the α = 1 generations: ⏳.
6. **N2 (preregistered null)** — computed day 1, unreported until Fri afternoon. As saved it is logit-lens, not Patchscope, so it cannot null the headline relevance count and cannot be regenerated. It nulls the *cosine*: **H3 fails on the primary set** (A·B = −0.13 vs null 95th pct +0.03), passes on math. Reported as a preregistered negative result.
7. **Theory scorecard:** P1 format-shaped ✓, P2 input-gated ✓, P3 constancy ✗, P4/H3 B-alignment ✗ (preregistered). Post-hoc refinement's C prediction ✓, labelled post hoc. The cancellation argument got the shape right and the statistics wrong.

## Verification state
- `VERIFY.md`: 45 rows, first four columns filled, every recompute one-liner run by Chat 1 and matching the digest. **Guiv's three columns ⏳** (he runs the one-liners himself).
- Discordant block (`notes/VERIFY_discordant_block.md`) written: Guiv read all 68; 67/68 concordance with the LLM judge is **verification, not blind agreement** (he tagged after seeing its output); item 186 re-classified FORMAT→REASONING and 56 added, both logged; blinding leaks arm identity via output format (disclosed); X/Y→arm key confirmed from format signatures without opening the key.
- **Re-parser ∩ tags: 40/40** rescued items inside the 68 were tagged FORMAT, 0 REASONING; the 2 FORMAT-not-rescued (152, 184) are parser conservatism. Three independent routes agree item by item.
- Reading still owed by Guiv: Patchscope lists **blind** (5 min — still uncontaminated), steered generations (15), cooking + black-box (8).

## Integrity items (in the body as one reproducibility paragraph; full ledger in VERIFY.md/CHANGELOG)
Adapters/caches destroyed (numbers re-derivable only by retraining); arm B training curve unverifiable (pod log destroyed; eval corroborates qualitatively); three result files sync-reverted and regenerated; digest §7 p0 sentence corrected; digest was untracked until Fri 14:00; lexical baseline below null; ±2-item decode noise; blinding leak; non-blind concordance; two replication aborts + one accidental $0.09 probe pod; agent clock drift 2h15m (durations from agents are not evidence; gates are absolute times).

## Decisions in force
Both parsers or neither, everywhere. "16.6–22.6×" always with "V 4.0–5.4× is the claim; lr×steps mismatch is the confound." "Orthogonal to slightly negative", never "anti-aligned". Steering labelled raw-parser. p = 0.008 / 0.013 labelled suggestive. One story: C-vs-A; the parser confound is one paragraph under Gate 2. Fig 1 = C-vs-A, everything else faded (F1 ⏳). Claim class: narrow, hedged, one model, one task.

## Open (see OPEN_TASKS_CURRENT.md; prompts in prompts/round5/)
- **T1** theory chat → four blocks (bias-vs-deeper; dose arithmetic; scorecard; why-interesting) → Guiv rewrites in own voice. ⏳
- **F1** Chat 1 → doc fixes + Fig 1 + steering re-score if JSONs allow. ⏳
- **W4** writer → body with fixes, after F1/T1. ⏳
- **Guiv** → 30-min reading block; VERIFY columns; **executive summary (≤600 words, own voice, context→gap→claim→evidence→standard of evidence)**; form Q10–Q21; 06:00 go/extend; submit 08:30; revoke RunPod key.
- Post-deadline: T2 (theory → PREREG v2), E1 (Claude Code → dose-matched C family, seeds, trace ablation). See `docs/FUTURE_DIRECTIONS.md`.

## Risks, ranked
1. Executive summary unwritten. 2. Dose confound stated too late or too softly. 3. Body reads as LLM prose. 4. Two competing stories. 5. Replication lands late and tempts a restructure — it may only add one sentence.
