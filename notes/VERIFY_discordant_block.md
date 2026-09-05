# VERIFY.md — insert block: discordant-item tagging (Guiv, 2026-09-04)

Paste into `VERIFY.md`. Rows are written by Guiv; the "how recomputed" cells describe what he ran or read himself.

## Rows

| claim | value | source file | who produced it | how Guiv recomputed / checked it | raw items read | surprise if wrong |
|---|---|---|---|---|---|---|
| Failure-mode split of the 68 A-vs-D_math discordant items | D_math 42 FORMAT / 20 REASONING; A 0 FORMAT / 6 REASONING | `notes/guiv_tags.csv`, `results/llm_tags_68.csv` | Guiv (tags); LLM judge (independent pass) | Read all 68 completion pairs against the problem and gold answer; assigned my own label to each; then compared with the judge's independent pass | 68 pairs (136 completions) | 2 |
| Judge–human concordance on failure mode | 67/68 identical; 1 revised by me (item 186) | `notes/guiv_tags.csv` vs `results/llm_tags_68.csv` | both | Item-by-item diff. **This is not a blind agreement rate** — see "Ordering and contamination" below | 68 | 1 |
| X/Y → arm mapping is correct | all 6 reverse-discordant items confirmed | `results/discordant_key.json` (not opened for this check) | pipeline agent | Identified the arm from output-format signatures alone — A ends `#### N` + `The answer is: N`, D_math uses `<<…>>` calculator annotations or step-numbered markdown — and confirmed items 54, 89, 157, 163, 164, 186 all have A as the incorrect completion, without opening the key file | 6 | 1 |
| FORMAT tags correspond to the re-parser's rescues | [FILL: overlap of the 42 FORMAT items with the 41 D_math items rescued by `tools/reparse_acc.py`] | `notes/guiv_tags.csv`, `results/acc_table_reparsed.md` | Guiv | Set intersection of the two id lists | — | 2 |

## Definition used (sharper than the rubric given to the judge)

**FORMAT** = the model reasoned to the correct answer and stated it at some point, then destroyed its own result by continuing to generate. Whether the continuation is a related or an unrelated question is irrelevant; what matters is that the arithmetic was already done and the output discipline failed. This is narrower than "a formatting problem" and it is the category the stopping-robust re-parse is designed to recover.

**REASONING** = the arithmetic or the comprehension is wrong; the gold value never appears as the conclusion of the chain. This includes cases where a misreading of the problem produces an unsolvable or underdetermined setup and the completion then runs to the token cap without stating anything — the truncation is a symptom, not the cause.

**Item 186 was re-classified on this basis.** I first tagged it FORMAT because it truncates mid-solution. On review, the completion missed that the hamsters are "kept alone" (one per cage), set up `50H + 18R + 20 = 160` in two unknowns, began trial-and-error and ran out of budget. It never reached or stated the answer, so under the definition above it is REASONING. Recorded here rather than silently changed.

**Item 56** was omitted from my first pass and added afterwards (X reaches "Answer: 3 times" then continues; Y is clean). Recorded rather than silently inserted.

## Ordering and contamination — read this before using the numbers above

1. Earlier in the day I was given a characterisation of the original 20-item display file (`results/discordant_A_vs_D_math.md`) before tagging anything. My calibration sample was therefore redrawn from the remaining 48 items (`results/discordant_sample20.txt`).
2. I produced my 68 tags **after** the LLM judge's output existed and was visible to me. The 67/68 concordance above is therefore **verification of the judge's tags against the raw text, not an independent agreement rate**, and must not be reported as the latter. What it supports is the weaker and still useful claim that I read every item and would have caught a wrong tag.
3. **The blinding leaks.** A's completions end `#### N` followed by `The answer is: N`; D_math's use `<<…>>` calculator annotations or step-numbered markdown. Anyone who has previously seen the arms unblinded — as I had — can read the arm off the style, so the X/Y randomisation did not conceal arm identity from me. I do not believe this biases a FORMAT-vs-REASONING judgement, which depends only on whether the completion reached and stated the gold value, but it means the tagging was not blind to arm and is reported as such. It also means the judge, which had no prior exposure to the arms, was blind in a way I was not.

## What these numbers license

Sayable: "A's advantage over the same-domain SFT control is roughly two-thirds an answer-extraction artifact. Under the stopping-robust parser A still wins 22–7 (p = 0.008), and across the 68 discordant items the SFT arm makes 20 genuine reasoning errors to the RL arm's 6."

Not sayable: any reasoning-gain claim from the raw last-number parser; any claim that the 67/68 concordance is a blind agreement rate; any claim that the tagging was arm-blind.
