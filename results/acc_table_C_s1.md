# Held-out accuracy: arm C seed 1 vs C seed 0 and A seed 0

Same 200 GSM8K test items (set sha 49b3a3f8…), greedy, cap 512, both parsers. Generated 2026-09-04T22:28:56.613459+00:00.

| parser | arm | seed | step | correct | acc |
|---|---|---|---|---|---|
| raw_last_number | C_s1 | 1 | 225 | 185/200 | 0.925 |
| raw_last_number | C_s0 | 0 | 225 | 186/200 | 0.930 |
| raw_last_number | A_s0 | 0 | 150 | 188/200 | 0.940 |
| rescored | C_s1 | 1 | 225 | 185/200 | 0.925 |
| rescored | C_s0 | 0 | 225 | 186/200 | 0.930 |
| rescored | A_s0 | 0 | 150 | 188/200 | 0.940 |

| parser | x | y | acc x | acc y | both | x only | y only | neither | McNemar exact p |
|---|---|---|---|---|---|---|---|---|---|
| raw_last_number | C_s1 | C_s0 | 0.925 | 0.930 | 183 | 2 | 3 | 12 | 1.000 |
| raw_last_number | C_s1 | A_s0 | 0.925 | 0.940 | 182 | 3 | 6 | 9 | 0.508 |
| rescored | C_s1 | C_s0 | 0.925 | 0.930 | 183 | 2 | 3 | 12 | 1.000 |
| rescored | C_s1 | A_s0 | 0.925 | 0.940 | 182 | 3 | 6 | 9 | 0.508 |
