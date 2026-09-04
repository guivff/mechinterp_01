# Visibility table — V = ‖d_neutral,p1‖ (L15, natural norm) / ‖ΔW‖_F (merged LoRA delta, α/r·BA over 248 modules)

| arm | ‖d_neutral,p1‖ | ‖d_math,p1‖ | ‖ΔW‖_F | max module ‖ΔW_m‖_F | top σ | V (neutral) | V (math) |
|---|---|---|---|---|---|---|---|
| D | 3.151 | 4.137 | 8.212 | 0.934 | 0.7801 | 0.3837 | 0.5037 |
| D_s1 | 3.204 | 4.096 | 8.196 | 0.919 | 0.7609 | 0.3910 | 0.4997 |
| D_math | 0.389 | 5.107 | 6.579 | 0.618 | 0.4044 | 0.0591 | 0.7764 |
| D_math_full | 1.199 | 10.053 | 6.702 | 0.632 | 0.4095 | 0.1789 | 1.4999 |
| D_math_full_s1 | 1.263 | 10.302 | 6.672 | 0.634 | 0.4192 | 0.1893 | 1.5440 |
| C | 3.488 | 5.380 | 6.963 | 0.697 | 0.5805 | 0.5010 | 0.7726 |
| A | 0.210 | 0.483 | 1.675 | 0.168 | 0.1194 | 0.1252 | 0.2883 |
| B | 0.094 | 0.229 | 1.656 | 0.154 | 0.0987 | 0.0568 | 0.1383 |
| N3 | 0.046 | 0.188 | 2.069 | 0.188 | 0.0377 | 0.0221 | 0.0908 |

Sources: results/perposition_table_C.csv, results/perposition_table_seeds.csv (position 1, ordinal-1 mean diff), results/lora_delta_stats.json. N3 = untrained LoRA norm-matched to A@25's factor norm. No interpretation.
