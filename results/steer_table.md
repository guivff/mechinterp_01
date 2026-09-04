# eta_ref-scaled steering of the BASE model

d = mean (h_adapter - h_base) over neutral snippets at ordinals >= 1, layer 15, rescaled to eta_ref = 11.243 times alpha, added at the layer-15 block output at ALL positions.
Readout: first 200 GSM8K test items, greedy, cap 512 (same items and decode as results/acc_base_s0.json).
'random' is an isotropic direction (seed 0) at the same applied norm. EOS rate = fraction of completions that stopped before the 512 cap.
Numeral rate = mean fraction of the first 30 generated tokens whose decoded text contains a digit.

| direction | alpha | applied ‖d‖ | correct/200 | acc | EOS rate | mean len | numeral rate (first 30 tokens) |
|---|---|---|---|---|---|---|---|
| none (unsteered) | 0 | 0 | 26 | 0.130 | 0.140 | 470 | — |
| A | 0.25 | 2.81 | 40 | 0.200 | 0.160 | 469 | 0.107 |
| A | 0.5 | 5.62 | 37 | 0.185 | 0.200 | 466 | 0.089 |
| A | 1 | 11.24 | 14 | 0.070 | 0.105 | 505 | 0.072 |
| A | 2 | 22.49 | 0 | 0.000 | 0.035 | 542 | 0.039 |
| C | 0.25 | 2.81 | 34 | 0.170 | 0.200 | 445 | 0.126 |
| C | 0.5 | 5.62 | 43 | 0.215 | 0.310 | 421 | 0.135 |
| C | 1 | 11.24 | 8 | 0.040 | 0.305 | 413 | 0.226 |
| C | 2 | 22.49 | 0 | 0.000 | 0.990 | 19 | 0.073 |
| D_math_full | 0.25 | 2.81 | 41 | 0.205 | 0.305 | 410 | 0.150 |
| D_math_full | 0.5 | 5.62 | 57 | 0.285 | 0.520 | 320 | 0.195 |
| D_math_full | 1 | 11.24 | 15 | 0.075 | 0.560 | 272 | 0.283 |
| D_math_full | 2 | 22.49 | 1 | 0.005 | 0.100 | 494 | 0.158 |
| random | 0.25 | 2.81 | 26 | 0.130 | 0.140 | 467 | 0.139 |
| random | 0.5 | 5.62 | 26 | 0.130 | 0.170 | 452 | 0.164 |
| random | 1 | 11.24 | 7 | 0.035 | 0.025 | 506 | 0.252 |
| random | 2 | 22.49 | 0 | 0.000 | 0.035 | 512 | 0.433 |

Raw: results/steer_eval/*.json (per-item predictions included). Steered neutral generations at alpha=1: results/steer_eval/neutral_gens_*_a1.md.
The earlier natural-norm run (results/steer_eval/*_x*.json) is dose-inadequate; see VERIFY.md.
