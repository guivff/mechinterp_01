# Visibility table

V = ||d_neutral,p1|| / ||dW||_F : the layer-15 position-1 mean activation difference on neutral text,
at its natural norm, divided by the Frobenius norm of the merged LoRA delta (alpha/r * B A summed over
248 modules). 'floor' is the paired split-half floor of the same difference vector. Real files only.

| arm | \|\|d_neutral,p1\|\| | floor | \|\|d_math,p1\|\| | \|\|dW\|\|_F | max module | top sigma | V (neutral) | V (math) |
|---|---|---|---|---|---|---|---|---|
| D | 3.151 | 0.400 | 4.137 | 8.212 | 0.934 | 0.7801 | 0.3837 | 0.5037 |
| D_s1 | 3.204 | 0.399 | 4.096 | 8.196 | 0.919 | 0.7609 | 0.3910 | 0.4997 |
| D_math | 0.389 | 0.057 | 5.107 | 6.579 | 0.618 | 0.4044 | 0.0591 | 0.7764 |
| D_math_full | 1.199 | 0.144 | 10.053 | 6.702 | 0.632 | 0.4095 | 0.1789 | 1.4999 |
| D_math_full_s1 | 1.263 | 0.150 | 10.302 | 6.672 | 0.634 | 0.4192 | 0.1893 | 1.5440 |
| C | 3.488 | 0.435 | 5.380 | 6.963 | 0.697 | 0.5805 | 0.5010 | 0.7726 |
| A | 0.210 | 0.029 | 0.483 | 1.675 | 0.168 | 0.1194 | 0.1252 | 0.2883 |
| A_s1 | 0.155 | 0.023 | 0.343 | 1.682 | 0.167 | 0.1166 | 0.0919 | 0.2039 |
| B | 0.094 | 0.017 | 0.229 | 1.656 | 0.154 | 0.0987 | 0.0568 | 0.1383 |
| N3 | 0.046 | 0.013 | 0.188 | 2.069 | 0.188 | 0.0377 | 0.0221 | 0.0908 |

Cross-seed V (neutral) for D: 0.3837 (seed 0) vs 0.3910 (seed 1); ratio 1.019.

Cross-seed V (neutral) for A: 0.1252 (seed 0) vs 0.0919 (seed 1); ratio 1.363.

Cross-seed V (neutral) for D_math_full: 0.1789 (seed 0) vs 0.1893 (seed 1); ratio 1.058.

Sources: results/perposition_table_A_seeds.csv, results/perposition_table_C.csv, results/perposition_table_seeds.csv, results/lora_delta_stats.json.
N3 is an untrained LoRA whose factor norm was matched to A@25, so its V is a floor for an adapter of that size, not for A-final.
No interpretation.
