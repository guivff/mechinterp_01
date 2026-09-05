# Held-out accuracy: C_masked_s1 vs C s0, C s1, C_masked s0, A s0

Same 200 GSM8K test items (set sha 49b3a3f8…), greedy, cap 512, both parsers. Generated 2026-09-05T03:49:42.673078+00:00.

| parser | arm | seed | step | correct | acc |
|---|---|---|---|---|---|
| raw_last_number | C_masked_s1 | 1 | 225 | 189/200 | 0.945 |
| raw_last_number | C_s0 | 0 | 225 | 186/200 | 0.930 |
| raw_last_number | C_s1 | 1 | 225 | 185/200 | 0.925 |
| raw_last_number | C_masked_s0 | 0 | 225 | 187/200 | 0.935 |
| raw_last_number | A_s0 | 0 | 150 | 188/200 | 0.940 |
| rescored | C_masked_s1 | 1 | 225 | 189/200 | 0.945 |
| rescored | C_s0 | 0 | 225 | 186/200 | 0.930 |
| rescored | C_s1 | 1 | 225 | 185/200 | 0.925 |
| rescored | C_masked_s0 | 0 | 225 | 187/200 | 0.935 |
| rescored | A_s0 | 0 | 150 | 188/200 | 0.940 |

| parser | x | y | acc x | acc y | both | x only | y only | neither | McNemar exact p |
|---|---|---|---|---|---|---|---|---|---|
| raw_last_number | C_masked_s1 | C_s0 | 0.945 | 0.930 | 181 | 8 | 5 | 6 | 0.581 |
| raw_last_number | C_masked_s1 | C_s1 | 0.945 | 0.925 | 180 | 9 | 5 | 6 | 0.424 |
| raw_last_number | C_masked_s1 | C_masked_s0 | 0.945 | 0.935 | 184 | 5 | 3 | 8 | 0.727 |
| raw_last_number | C_masked_s1 | A_s0 | 0.945 | 0.940 | 184 | 5 | 4 | 7 | 1.000 |
| rescored | C_masked_s1 | C_s0 | 0.945 | 0.930 | 181 | 8 | 5 | 6 | 0.581 |
| rescored | C_masked_s1 | C_s1 | 0.945 | 0.925 | 180 | 9 | 5 | 6 | 0.424 |
| rescored | C_masked_s1 | C_masked_s0 | 0.945 | 0.935 | 184 | 5 | 3 | 8 | 0.727 |
| rescored | C_masked_s1 | A_s0 | 0.945 | 0.940 | 184 | 5 | 4 | 7 | 1.000 |

# Held-out accuracy: C_scrambled_s0 vs C s0, C s1, C_masked s0, A s0

Same 200 GSM8K test items (set sha 49b3a3f8…), greedy, cap 512, both parsers. Generated 2026-09-05T03:49:44.085352+00:00.

| parser | arm | seed | step | correct | acc |
|---|---|---|---|---|---|
| raw_last_number | C_scrambled_s0 | 0 | 225 | 182/200 | 0.910 |
| raw_last_number | C_s0 | 0 | 225 | 186/200 | 0.930 |
| raw_last_number | C_s1 | 1 | 225 | 185/200 | 0.925 |
| raw_last_number | C_masked_s0 | 0 | 225 | 187/200 | 0.935 |
| raw_last_number | A_s0 | 0 | 150 | 188/200 | 0.940 |
| rescored | C_scrambled_s0 | 0 | 225 | 182/200 | 0.910 |
| rescored | C_s0 | 0 | 225 | 186/200 | 0.930 |
| rescored | C_s1 | 1 | 225 | 185/200 | 0.925 |
| rescored | C_masked_s0 | 0 | 225 | 187/200 | 0.935 |
| rescored | A_s0 | 0 | 150 | 188/200 | 0.940 |

| parser | x | y | acc x | acc y | both | x only | y only | neither | McNemar exact p |
|---|---|---|---|---|---|---|---|---|---|
| raw_last_number | C_scrambled_s0 | C_s0 | 0.910 | 0.930 | 177 | 5 | 9 | 9 | 0.424 |
| raw_last_number | C_scrambled_s0 | C_s1 | 0.910 | 0.925 | 176 | 6 | 9 | 9 | 0.607 |
| raw_last_number | C_scrambled_s0 | C_masked_s0 | 0.910 | 0.935 | 178 | 4 | 9 | 9 | 0.267 |
| raw_last_number | C_scrambled_s0 | A_s0 | 0.910 | 0.940 | 179 | 3 | 9 | 9 | 0.146 |
| rescored | C_scrambled_s0 | C_s0 | 0.910 | 0.930 | 177 | 5 | 9 | 9 | 0.424 |
| rescored | C_scrambled_s0 | C_s1 | 0.910 | 0.925 | 176 | 6 | 9 | 9 | 0.607 |
| rescored | C_scrambled_s0 | C_masked_s0 | 0.910 | 0.935 | 178 | 4 | 9 | 9 | 0.267 |
| rescored | C_scrambled_s0 | A_s0 | 0.910 | 0.940 | 179 | 3 | 9 | 9 | 0.146 |

# Held-out accuracy: C_shifted_s0 vs C s0, C s1, C_masked s0, A s0

Same 200 GSM8K test items (set sha 49b3a3f8…), greedy, cap 512, both parsers. Generated 2026-09-05T03:49:45.533644+00:00.

| parser | arm | seed | step | correct | acc |
|---|---|---|---|---|---|
| raw_last_number | C_shifted_s0 | 0 | 225 | 184/200 | 0.920 |
| raw_last_number | C_s0 | 0 | 225 | 186/200 | 0.930 |
| raw_last_number | C_s1 | 1 | 225 | 185/200 | 0.925 |
| raw_last_number | C_masked_s0 | 0 | 225 | 187/200 | 0.935 |
| raw_last_number | A_s0 | 0 | 150 | 188/200 | 0.940 |
| rescored | C_shifted_s0 | 0 | 225 | 184/200 | 0.920 |
| rescored | C_s0 | 0 | 225 | 186/200 | 0.930 |
| rescored | C_s1 | 1 | 225 | 185/200 | 0.925 |
| rescored | C_masked_s0 | 0 | 225 | 187/200 | 0.935 |
| rescored | A_s0 | 0 | 150 | 188/200 | 0.940 |

| parser | x | y | acc x | acc y | both | x only | y only | neither | McNemar exact p |
|---|---|---|---|---|---|---|---|---|---|
| raw_last_number | C_shifted_s0 | C_s0 | 0.920 | 0.930 | 181 | 3 | 5 | 11 | 0.727 |
| raw_last_number | C_shifted_s0 | C_s1 | 0.920 | 0.925 | 180 | 4 | 5 | 11 | 1.000 |
| raw_last_number | C_shifted_s0 | C_masked_s0 | 0.920 | 0.935 | 180 | 4 | 7 | 9 | 0.549 |
| raw_last_number | C_shifted_s0 | A_s0 | 0.920 | 0.940 | 178 | 6 | 10 | 6 | 0.454 |
| rescored | C_shifted_s0 | C_s0 | 0.920 | 0.930 | 181 | 3 | 5 | 11 | 0.727 |
| rescored | C_shifted_s0 | C_s1 | 0.920 | 0.925 | 180 | 4 | 5 | 11 | 1.000 |
| rescored | C_shifted_s0 | C_masked_s0 | 0.920 | 0.935 | 180 | 4 | 7 | 9 | 0.549 |
| rescored | C_shifted_s0 | A_s0 | 0.920 | 0.940 | 178 | 6 | 10 | 6 | 0.454 |

