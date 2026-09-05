# Held-out accuracy: arm C_masked (completion-only loss) vs C s0, C s1, A s0, D_math

Same 200 GSM8K test items (set sha 49b3a3f8…), greedy, cap 512, both parsers. Generated 2026-09-05T01:01:23.156257+00:00.

| parser | arm | seed | step | correct | acc |
|---|---|---|---|---|---|
| raw_last_number | C_masked | 0 | 225 | 187/200 | 0.935 |
| raw_last_number | C_s0 | 0 | 225 | 186/200 | 0.930 |
| raw_last_number | C_s1 | 1 | 225 | 185/200 | 0.925 |
| raw_last_number | A_s0 | 0 | 150 | 188/200 | 0.940 |
| raw_last_number | D_math | 0 | 225 | 132/200 | 0.660 |
| rescored | C_masked | 0 | 225 | 187/200 | 0.935 |
| rescored | C_s0 | 0 | 225 | 186/200 | 0.930 |
| rescored | C_s1 | 1 | 225 | 185/200 | 0.925 |
| rescored | A_s0 | 0 | 150 | 188/200 | 0.940 |
| rescored | D_math | 0 | 225 | 173/200 | 0.865 |

| parser | x | y | acc x | acc y | both | x only | y only | neither | McNemar exact p |
|---|---|---|---|---|---|---|---|---|---|
| raw_last_number | C_masked | C_s0 | 0.935 | 0.930 | 182 | 5 | 4 | 9 | 1.000 |
| raw_last_number | C_masked | C_s1 | 0.935 | 0.925 | 180 | 7 | 5 | 8 | 0.774 |
| raw_last_number | C_masked | A_s0 | 0.935 | 0.940 | 183 | 4 | 5 | 8 | 1.000 |
| raw_last_number | C_masked | D_math | 0.935 | 0.660 | 127 | 60 | 5 | 8 | 0.000 |
| rescored | C_masked | C_s0 | 0.935 | 0.930 | 182 | 5 | 4 | 9 | 1.000 |
| rescored | C_masked | C_s1 | 0.935 | 0.925 | 180 | 7 | 5 | 8 | 0.774 |
| rescored | C_masked | A_s0 | 0.935 | 0.940 | 183 | 4 | 5 | 8 | 1.000 |
| rescored | C_masked | D_math | 0.935 | 0.865 | 166 | 21 | 7 | 6 | 0.013 |
