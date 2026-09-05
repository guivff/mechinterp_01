# ENDGAME — Sat 2026-09-05 03:45 → 08:30 Zurich (4 h 45). Amended after the C_masked result; blocks 1–2 done, block 3 rewritten.

Data are closed (C_masked landed 02:56, all pods terminated). The submission is won or lost on: (a) the executive summary's first paragraph — two self-refutations, then the corrected loss-placement claim with V(A) > V(C_masked) stated beside it, (b) `VERIFY.md`, (c) one Figure 1 (V by arm, masked/unmasked) and no restructuring. Ordered by value per hour. Chat ownership: `docs/CHAT_ROSTER.md`.

## Rule for the rest of the day
Nothing enters the doc that is not in `docs/RESULTS_DIGEST.md` (≥ `5530ae2`) with a `results/` file and a `VERIFY.md` row. Every accuracy under both parsers. New numbers stop at **Sat 07:30** (after 06:00: one-sentence inserts only, each with a VERIFY.md row; the C_masked V is the one expected insert, gate 07:00). The only new numbers still permitted are N2, B's re-derived curve, Guiv's tallies, and the C s1 / A s2 replication range. **07:30 decision point:** VERIFY.md closed and summary in Guiv's voice → submit by 08:30; otherwise request the extension (to Sept 11) instead of rushing.

## Block 1 — 15:30–17:00 · YOU (the rejection-critical block; ~75 min)
Runsheet: `prompts/round4/V0_RUNSHEET.md`. Order is not negotiable:
1. Patchscope letters A–E, blind, one word each — **before anything else** — then unblind.
2. 20 blind tags (ids in `results/discordant_sample20.txt`), then unblind, then agreement with C2.
3. Steering paragraph from `review_packet/steer_reading.md` (disclose the prior characterisation).
4. Cooking + black-box notes.
5. `VERIFY.md`: run each one-liner from `tools/recompute_oneliners.md`, paste the command, fill items-read and surprise, write the two prose sections. A one-liner that disagrees with the digest is **written into the row**, not reconciled.

## Block 2 — same window, agents (no human time)
- **C1**: N2 report → digest §15; B curve or unverifiable mark; VERIFY scaffold + tested one-liners. (OPEN_TASKS 1–3.)
- **C2**: 68 tags.
- **RT**: top-5 objections on digest + firewall only.

## Block 3 — 03:45–05:30 · YOU write the executive summary (rewritten 03:45)
≤600 words, own words, before the doc body. Open with the two self-refutations and the corrected claim:
> This project set out to test whether on-policy GRPO leaves the kind of readable activation-difference trace on unrelated text that narrow SFT does. Twice its own controls overturned its headline. A stopping-robust re-parse showed the base model already solves 79 % of GSM8K but rarely stops, which shrank two of three headline comparisons. Then the comparison that survived — imitation SFT on the RL policy's own correct samples leaves a trace 16.63–22.63× larger than the RL run at the same accuracy — was tested against a preregistered alternative six hours before the deadline and lost: masking the prompt tokens in that same SFT run, at the same data and dose, removes ~92 % of the trace at unchanged accuracy and lands it beside the RL run in size, direction and readout. The size of the readable trace is set by where the loss is placed, not by the learning rule. At matched loss placement RL and SFT traces are comparable — per unit weight change the RL trace is if anything larger — and what remains RL-specific is that it does not reproduce across seeds.
Then: assay validated (cooking, null ≤ 2, position 0); the C / C_masked / A numbers with the four-pair V string and the D_math masked/unmasked corroboration; the re-parse and Gate 2 with tags; steering negative; N2 and the scorecard (learning rule and dose as tested-and-not-supported); limitations with one seed first; one line on verification. Standard of evidence: both parsers, floors and nulls beside every trace, preregistered thresholds quoted, one seed on the decisive arm. No LLM prose.

## Block 4 — 04:30–06:30 · writer W5+W6 body lands; you edit
W3 was sent at ~16:30 with placeholders. Tonight send the one-message placeholder fill (tags, N2, B, range, your steering paragraph). Inserts: both-parser §5 with the cap statistics; A cross-seed; steering with the shared confound; N2 outcome; B status; lexical demoted; limitations 4–5. Never rewrite the summary for an insert.

## Block 5 — 20:30–21:30 · critics + form
- WC on summary + form answers (Q10–Q21). Q14 leads with the re-parse; Q16 is the ledger including the contamination log and the three sync-reverted files.
- Answer RT's top three in `VERIFY.md` yourself.

## Block 6 — 22:00–04:00 · sleep. Non-negotiable.

## Block 7 — Sat 04:00–08:30
07:30 no new numbers; 07:30 go/extend decision. Regenerate figures from real files (`analysis/make_figures.py`); recompute every headline number once more; `VERIFY.md` closed; Toggl screenshot; doc permissions = anyone with link; paste form; upload resume; **submit by 08:30** (deadline 08:59). Then revoke the RunPod key.

## What NOT to do
No further training (C_masked was the last), no C_masked seed 1, no restructuring for the new result, no six-way judge on real lists, no relevance grading for C, no claim about paper potential, no accuracy under one parser, no "17×" without "on seed 0".

## The three sentences that decide the outcome
1. The opening of the summary (above).
2. "I read all 68 discordant items against the LLM's tags (67/68 concordant, not blind — I saw its output first); I recomputed X and Y with the one-liners in VERIFY.md; I did not check Z and would be moderately surprised by an error there."
3. "We preregistered the test that could kill our headline, ran it six hours before the deadline, and it did; the number that also blocks the tidy version of the replacement story (V(A) > V(C_masked)) is in the same paragraph."
