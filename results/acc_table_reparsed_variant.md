# Stopping-robust re-scoring of held-out accuracy

Same 200 GSM8K test items, same completions, same last-number extractor.
're-scored' first truncates each completion at the first line starting a NEW question
(^What is, ^Solve, ^The following are questions, or Answer: after a completed ####/\boxed{}),
then extracts. This isolates answers lost to the model continuing past its own answer.
acc_table.md (raw) is unchanged; both are reported.

| arm | raw correct | raw acc | re-scored correct | re-scored acc | delta | completions where a cut fired | rescued | broken |
|---|---|---|---|---|---|---|---|---|
| A | 188/200 | 0.940 | 188/200 | 0.940 | +0.000 | 0 | 0 | 0 |
| D_math | 132/200 | 0.660 | 173/200 | 0.865 | +0.205 | 45 | 41 | 0 |
| D_math_full | 127/200 | 0.635 | 164/200 | 0.820 | +0.185 | 43 | 37 | 0 |
| C | 186/200 | 0.930 | 186/200 | 0.930 | +0.000 | 0 | 0 | 0 |
| D | 53/200 | 0.265 | 108/200 | 0.540 | +0.275 | 105 | 56 | 1 |
| B | 15/200 | 0.075 | 162/200 | 0.810 | +0.735 | 180 | 149 | 2 |
| base | 28/200 | 0.140 | 158/200 | 0.790 | +0.650 | 162 | 132 | 2 |

## Paired comparisons on the same 200 items, under both parsers

| pair | parser | x acc | y acc | x-only | y-only | both | neither | McNemar exact p |
|---|---|---|---|---|---|---|---|---|
| A vs D_math | raw | 0.940 | 0.660 | 62 | 6 | 126 | 6 | 0.000000 |
| A vs D_math | re-scored | 0.940 | 0.865 | 22 | 7 | 166 | 5 | 0.008130 |
| A vs base | raw | 0.940 | 0.140 | 162 | 2 | 26 | 10 | 0.000000 |
| A vs base | re-scored | 0.940 | 0.790 | 35 | 5 | 153 | 7 | 0.000001 |
| A vs B | raw | 0.940 | 0.075 | 175 | 2 | 13 | 10 | 0.000000 |
| A vs B | re-scored | 0.940 | 0.810 | 31 | 5 | 157 | 7 | 0.000013 |
| A vs C | raw | 0.940 | 0.930 | 7 | 5 | 181 | 7 | 0.774414 |
| A vs C | re-scored | 0.940 | 0.930 | 7 | 5 | 181 | 7 | 0.774414 |
| A vs D_math_full | raw | 0.940 | 0.635 | 65 | 4 | 123 | 8 | 0.000000 |
| A vs D_math_full | re-scored | 0.940 | 0.820 | 29 | 5 | 159 | 7 | 0.000039 |
| D_math vs base | raw | 0.660 | 0.140 | 112 | 8 | 20 | 60 | 0.000000 |
| D_math vs base | re-scored | 0.865 | 0.790 | 31 | 16 | 142 | 11 | 0.039986 |

Cut-pattern variant with ^Question:/^Problem: added: ON.
Sources: results/acc_A_s0.json, results/acc_B_s0.json, results/acc_C_s0.json, results/acc_D_math_full_s0.json, results/acc_D_math_s0.json, results/acc_D_s0.json, results/acc_base_s0.json.
Parser: grpo.train_grpo.extract_answer applied to the truncated text. No interpretation.
