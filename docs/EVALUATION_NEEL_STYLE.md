# EVALUATION_NEEL_STYLE.md — independent subagent evaluation, Sat 2026-09-05 ~01:15

Evaluator had only: the MATS 12.0 FAQ (incl. the past-examples section), `docs/RESEARCH_SUMMARY.md`, `CLAIM_FIREWALL.md`. No prior involvement.

## A. Criteria (1–5)
- **Clarity — unjudgeable yet.** 5 = form answers and summary state in three sentences: Minder trace; does GRPO leave one; imitation SFT at matched accuracy leaves a much larger trace, so the small RL trace is the learning rule. One C-vs-A figure, one confound paragraph. 2 = the summary as it stands: six arms, three nulls, five positions, two parsers, an integrity ledger, claim found in §4.2.
- **Good Taste — 4.** Literally Neel's listed question with a non-obvious twist. Loses a point: "bias term or deeper?" is never actually answered.
- **Truth-seeking — 5.** The 0.14-vs-0.79 parser catch and re-scoring every comparison is exactly the move he rewards. Two refuted preregistered predictions reported as refuted.
- **Technical depth — 4.** GRPO from scratch on a 4B base, Patchscope with matched-selection nulls, McNemar. Not 5: headline not dose-matched.
- **Simplicity — 2.** Six arms, three nulls, three layers, five positions, two readouts, steering, TF-IDF, module-family split; several admitted uninformative.
- **Prioritisation — 3.** Right pivot to C; much of the 20 h went to things that ended "uninformative" or "cannot be regenerated".
- **Productivity — 4.** High volume — but may read as the agent's productivity.
- **Show your work — 4.** Prediction table and "post hoc, labelled" are model behaviour. Ledger over-long.
- **Enthusiasm — 3.** Procedural. No sentence says why this is interesting to *you*.

## B. Placement
Most comparable past examples: **R1D1** (accept — sensible idea, pivoted, taught Neel something) and **"Wait" backtracking** (high borderline — productive, many methods, too broad). This project is more ambitious and more on-target than both, with stronger skepticism, but broader and messier than Wait-backtracking and with a weaker single-figure story than R1D1. Vs R1 Distill Diffing (accept despite a conceptual flaw): similar flaw (dose), better self-awareness.

**Verdict:** high-borderline to accept if written up well; borderline-reject as the summary currently reads.
- P(exploration phase, top ~34): **~45 %**; conditional on a clean own-voice summary and Fig 1 = C-vs-A: **~60 %**; if the body reads as LLM prose: ~25 %.
- P(top 10): **~12 %.**

## C. Five things most likely to sink it (ranked; all fixable in writing)
1. **Headline not dose-matched, and the summary says "the difference is the weighting."** C: lr 1e-4 × 225 SFT steps; A: 3e-5 × 150 GRPO steps. Neel's first thought: "you trained SFT at 3× the lr; of course ‖ΔW‖ is 4× bigger." What survives is V (~4× per unit ‖ΔW‖), and V for A is unstable. **Fix:** lead with V, put the ‖ΔW‖ factor second, name the lr mismatch as the primary open confound in the same sentence. Delete "the difference is the weighting."
2. **Two competing stories** (C-vs-A and the parser confound). **Fix:** headline = C-vs-A; the confound is the one sanity-check paragraph that makes Gate 2 honest, not a co-equal result.
3. **Nothing can be re-derived** (adapters, B curve, N2 as logit-lens, sync reverts) — together suggests process chaos. **Fix:** one short "what is reproducible and what isn't" paragraph, not a nine-item ledger in the body.
4. **Steering accuracies are raw-parser only** (base 0.14 → A 0.200) while §4.4 says base = 0.79. Violates the firewall's own both-parsers rule. **Fix:** drop steering from the summary, or one line stating raw-parser and confounded.
5. **"Bias term or deeper?" never answered.** P3 refuted = mean-offset share ~0.25 for both → "at this dose the RL trace is as constant as SFT's; what differs is magnitude, not form." **Fix:** one paragraph.

## D. Do not bury
1. The parser catch as a detective story: 0.14 → read completions → 0.79 → 20/20 audited → three routes agree 40/40.
2. The prediction scorecard table — put it in the executive summary.
3. Cooking SFT reproduces Minder's readout with a matched null; SFT arms seed-stable (0.92–0.99), GRPO not (0.68). The seed instability is arguably more interesting than the 17×.

## E. Red flags
- **LLM-prose tells:** "Caveats carried with the claim", "What survives:", "worked example of", stacked em-dashes, triads. Rewrite topic sentences by hand.
- **"Agent did it" risk:** the only hand-verified items listed are qualitative reads. State which *numbers* you recomputed and got. Keep the non-blind labelling of the 67/68.
- **Overclaims vs firewall:** "anti-aligned (−0.13)" vs firewall "orthogonal" — pick one; −0.13 is closer to orthogonal. "The difference is the weighting." "16.6–22.6×" presented as a range three times — it is two A seeds against one C seed.
- **Internal contradictions:** 22/7 re-scored vs "all 6 of A's losses" — say the 7th is a rescued D_math item. "68 raw errors" and "68 discordant items" coincide — say so. N2 nulls the cosine, not the Patchscope count — clarify where §2 and §6 meet.
- **Hours:** ~20 counted + 14.4 h pod + overnight agent build. Disclose agent-hours in Q16; do not let the write-up imply 20 human hours produced six trained arms.
