(Run after Gate 2, with the REAL figures and results attached — never with MOCK files.)

A friend is about to write up the attached results for a very skeptical reviewer (Neel Nanda) and asked for brutally honest feedback. Attached: the control-plane zip (read PREREG.md, CLAIM_FIREWALL.md and docs/RED_TEAM.md first), the real figures (figs/*.png), results/cosine_matrix.csv, results/judged_*.jsonl summaries, results/acc_*.json, and VERIFY.md. Assume the results are an artifact and find the artifact.

Under 700 words, no praise:
1. For each preregistered hypothesis (H1–H4 and the secondary constancy/conditional-trace predictions), state from the numbers whether the preregistered pass criterion was met, missed, or is untestable, quoting the exact cells. Flag any place where the write-up would be tempted to move a threshold.
2. Which of the 23 confounds in docs/RED_TEAM.md are now actually excluded by the data, which are addressed only by a stated limitation, and which are still live? Name the three most damaging live ones and the cheapest remaining check for each (must fit in one hour, readouts only, no new training).
3. Look at the raw judge transcripts and token lists provided: is the judge reading surface tokens? Is `none` under- or over-predicted? Does the TF-IDF baseline gap mean what the write-up will say it means?
4. Write the strongest honest one-sentence result and the strongest honest one-sentence limitation. Then write the sentence a reviewer would use to dismiss the result, and what in VERIFY.md answers it.
5. Which single figure should lead the executive summary, and what is wrong with it as drawn?
