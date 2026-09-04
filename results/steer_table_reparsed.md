# Steering re-scored under the stopping-robust parser

Same 33 runs and the same stored completions as `results/steer_table.md`; only the parser changes.
**Every accuracy in `results/steer_table.md` and in digest §10 is the raw last-number parser.**
Under the stopping-robust parser the unsteered base is 0.790, not 0.130, so the raw steering gains are
movements within the format-failure regime, not reasoning gains. McNemar is against the unsteered run of the same parser.

Unsteered baseline: raw 26/200 = 0.130; **re-scored 158/200 = 0.790** (cuts fired on 161/200 completions).

| direction | α | raw acc | raw McNemar p | re-scored acc | re-scored (steered-only/base-only) | re-scored p | cuts fired |
|---|---|---|---|---|---|---|---|
| A | 0.25 | 0.200 | 0.0125 | **0.815** | 22 / 17 | 0.5224 | 157 |
| A | 0.5 | 0.185 | 0.0522 | **0.735** | 20 / 31 | 0.1608 | 135 |
| A | 1 | 0.070 | 0.0730 | **0.065** | 1 / 146 | 0.0000 | 11 |
| A | 2 | 0.000 | 0.0000 | **0.000** | 0 / 158 | 0.0000 | 0 |
| C | 0.25 | 0.170 | 0.1849 | **0.775** | 13 / 16 | 0.7111 | 151 |
| C | 0.5 | 0.215 | 0.0046 | **0.730** | 20 / 32 | 0.1263 | 131 |
| C | 1 | 0.040 | 0.0021 | **0.045** | 2 / 151 | 0.0000 | 2 |
| C | 2 | 0.000 | 0.0000 | **0.000** | 0 / 158 | 0.0000 | 0 |
| D_math_full | 0.25 | 0.205 | 0.0167 | **0.790** | 13 / 13 | 1.0000 | 141 |
| D_math_full | 0.5 | 0.285 | 0.0000 | **0.650** | 15 / 43 | 0.0003 | 112 |
| D_math_full | 1 | 0.075 | 0.1081 | **0.075** | 2 / 145 | 0.0000 | 2 |
| D_math_full | 2 | 0.005 | 0.0000 | **0.005** | 0 / 157 | 0.0000 | 0 |
| random (seed 0) | 0.25 | 0.130 | 1.0000 | **0.790** | 14 / 14 | 1.0000 | 163 |
| random (seed 1) | 0.25 | 0.145 | 0.6476 | **0.795** | 13 / 12 | 1.0000 | 165 |
| random (seed 2) | 0.25 | 0.140 | 0.8388 | **0.820** | 18 / 12 | 0.3616 | 161 |
| random (seed 3) | 0.25 | 0.110 | 0.5235 | **0.805** | 15 / 12 | 0.7011 | 162 |
| random (seed 4) | 0.25 | 0.170 | 0.1849 | **0.765** | 10 / 15 | 0.4244 | 155 |
| random (seed 0) | 0.5 | 0.130 | 1.0000 | **0.745** | 15 / 24 | 0.1996 | 159 |
| random (seed 1) | 0.5 | 0.115 | 0.6776 | **0.705** | 8 / 25 | 0.0046 | 159 |
| random (seed 2) | 0.5 | 0.145 | 0.7111 | **0.790** | 18 / 18 | 1.0000 | 155 |
| random (seed 3) | 0.5 | 0.125 | 1.0000 | **0.805** | 17 / 14 | 0.7201 | 165 |
| random (seed 4) | 0.5 | 0.155 | 0.4731 | **0.635** | 10 / 41 | 0.0000 | 155 |
| random (seed 0) | 1 | 0.035 | 0.0009 | **0.085** | 0 / 141 | 0.0000 | 109 |
| random (seed 0) | 2 | 0.000 | 0.0000 | **0.000** | 0 / 158 | 0.0000 | 1 |

## Random-direction null, re-scored

| α | draws | re-scored acc mean | range |
|---|---|---|---|
| 0.25 | 5 | 0.795 | 0.765–0.820 |
| 0.5 | 5 | 0.736 | 0.635–0.805 |
| 1 | 1 | 0.085 | 0.085–0.085 |
| 2 | 1 | 0.000 | 0.000–0.000 |

Source: `results/steer_eval/*.json` (completions stored per item), `tools/steer_reparse.py`. No interpretation.
