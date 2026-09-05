# Null table — L15, neutral snippets, position 1 (agent first pass, C4 task 1; no interpretation)

trace = raw ||d|| of the arm's mean base->fine-tuned difference; split-half = the arm's own paired split-half statistic (||d_half1 - d_half2||/2 as stored in `split_half_floor`); N1 = base-vs-base split-half difference (`N1_halves` row, the column retired by the 07:00 Fri amendment) = 0.7468 — identical in every per-position CSV (0.7468, 0.7468, 0.7468, 0.7468, 0.7468, 0.7468) because all share the same base cache; N3 = untrained LoRA with factor norm matched to A@25, raw trace 0.0458 at ||dW||_F 2.069; N3 scaled = N3 raw x (arm ||dW||_F / N3 ||dW||_F) — **assumption: linear in ||dW||_F; not measured at any other norm.**

| arm | trace | split-half | split-half / trace | N1 (base-vs-base) | N3 raw | ||dW||_F | N3 scaled to arm ||dW|| | trace / split-half | trace / N1 | trace / N3 raw | trace / N3 scaled |
|---|---|---|---|---|---|---|---|---|---|---|---|
| D s0 | 3.1506 | 0.4004 | 0.127 | 0.7468 | 0.0458 | 8.212 | 0.1817 | 7.87 | 4.22 | 68.83 | 17.34 |
| D s1 | 3.2044 | 0.3988 | 0.124 | 0.7468 | 0.0458 | 8.196 | 0.1813 | 8.04 | 4.29 | 70.00 | 17.67 |
| D_math (masked) | 0.3888 | 0.0570 | 0.147 | 0.7468 | 0.0458 | 6.579 | 0.1455 | 6.82 | 0.52 | 8.49 | 2.67 |
| D_math_full s0 | 1.1991 | 0.1442 | 0.120 | 0.7468 | 0.0458 | 6.702 | 0.1483 | 8.31 | 1.61 | 26.20 | 8.09 |
| D_math_full s1 | 1.2633 | 0.1499 | 0.119 | 0.7468 | 0.0458 | 6.672 | 0.1476 | 8.43 | 1.69 | 27.60 | 8.56 |
| C s0 | 3.4881 | 0.4348 | 0.125 | 0.7468 | 0.0458 | 6.963 | 0.1540 | 8.02 | 4.67 | 76.20 | 22.64 |
| C s1 | 3.4975 | 0.4436 | 0.127 | 0.7468 | 0.0458 | 6.958 | 0.1539 | 7.88 | 4.68 | 76.41 | 22.72 |
| C_masked s0 | 0.2858 | 0.0386 | 0.135 | 0.7468 | 0.0458 | 5.844 | 0.1293 | 7.40 | 0.38 | 6.24 | 2.21 |
| A s0 | 0.2097 | 0.0294 | 0.140 | 0.7468 | 0.0458 | 1.675 | 0.0371 | 7.14 | 0.28 | 4.58 | 5.66 |
| A s1 | 0.1545 | 0.0234 | 0.151 | 0.7468 | 0.0458 | 1.682 | 0.0372 | 6.60 | 0.21 | 3.38 | 4.15 |
| B s0 | 0.0940 | 0.0169 | 0.180 | 0.7468 | 0.0458 | 1.656 | 0.0366 | 5.56 | 0.13 | 2.05 | 2.57 |
| N3 (untrained LoRA) | 0.0458 | 0.0134 | 0.293 | 0.7468 | 0.0458 | 2.069 | 0.0458 | 3.41 | 0.06 | 1.00 | 1.00 |

Reading aids (facts only): N1 (base-vs-base split-half, 0.747) exceeds the raw trace of A, B, C_masked and D_math; the paired split-half statistic is the floor every digest claim uses. N3 scaled to A's ||dW||_F is 0.0371 (A s0 trace 0.210); scaled to C_masked's, 0.1293 (C_masked trace 0.286).

## One-liners

```bash
# trace, split-half, N1 for one arm (edit arm/file)
python3 -c "import csv; [print(r['arm'],r['raw_norm'],r['split_half_floor']) for r in csv.DictReader(open('results/perposition_table_C.csv')) if r['set']=='neutral' and r['position']=='1']"
# N3 scaling to an arm's ||dW||_F
python3 -c "import json; d=json.load(open('results/lora_delta_stats.json')); m=json.load(open('results/lora_delta_stats_C_masked.json')); n3=0.0461; print('N3->A', n3*d['A']['delta_W_fro_total']/d['N3']['delta_W_fro_total'], 'N3->C_masked', n3*m['C_masked']['delta_W_fro_total']/d['N3']['delta_W_fro_total'])"
# regenerate this table
python3 tools/null_table.py
```

Sources: `results/perposition_table_C.csv`, `perposition_table_seeds.csv`, `perposition_table_A_seeds.csv`, `perposition_table_C_seeds.csv`, `perposition_table_C_masked.csv`, `lora_delta_stats*.json`. Agent-produced (C4); not the human audit.
