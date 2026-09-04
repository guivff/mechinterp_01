# eta_ref-scaled steering of the BASE model

d = mean (h_adapter - h_base) over neutral snippets at ordinals >= 1, layer 15, rescaled to eta_ref = 11.243 times alpha,
added at the layer-15 block output at ALL positions. Readout: first 200 GSM8K test items, greedy, cap 512.
EOS rate = fraction of completions that stopped before the cap. Numeral rate = mean fraction of the first 30 generated
tokens whose decoded text contains a digit (recomputed here for every run, baseline included).
McNemar = exact two-sided test against the unsteered run on the same 200 items; steered-only / base-only are the
discordant counts. No interpretation.

| direction | alpha | applied ‖d‖ | correct/200 | acc | EOS rate | mean len | numeral rate (first 30) | steered-only | base-only | McNemar p |
|---|---|---|---|---|---|---|---|---|---|---|
| none (unsteered) | 0 | 0 | 26 | 0.130 | 0.140 | 470 | 0.130 | — | — | — |
| A | 0.25 | 2.81 | 40 | 0.200 | 0.160 | 469 | 0.107 | 21 | 7 | 0.0125 |
| A | 0.5 | 5.62 | 37 | 0.185 | 0.200 | 466 | 0.089 | 19 | 8 | 0.0522 |
| A | 1 | 11.24 | 14 | 0.070 | 0.105 | 505 | 0.072 | 13 | 25 | 0.0730 |
| A | 2 | 22.49 | 0 | 0.000 | 0.035 | 542 | 0.039 | 0 | 26 | 0.0000 |
| C | 0.25 | 2.81 | 34 | 0.170 | 0.200 | 445 | 0.126 | 18 | 10 | 0.1849 |
| C | 0.5 | 5.62 | 43 | 0.215 | 0.310 | 421 | 0.135 | 25 | 8 | 0.0046 |
| C | 1 | 11.24 | 8 | 0.040 | 0.305 | 413 | 0.226 | 7 | 25 | 0.0021 |
| C | 2 | 22.49 | 0 | 0.000 | 0.990 | 19 | 0.073 | 0 | 26 | 0.0000 |
| D_math_full | 0.25 | 2.81 | 41 | 0.205 | 0.305 | 410 | 0.150 | 25 | 10 | 0.0167 |
| D_math_full | 0.5 | 5.62 | 57 | 0.285 | 0.520 | 320 | 0.195 | 44 | 13 | 0.0000 |
| D_math_full | 1 | 11.24 | 15 | 0.075 | 0.560 | 272 | 0.283 | 14 | 25 | 0.1081 |
| D_math_full | 2 | 22.49 | 1 | 0.005 | 0.100 | 494 | 0.158 | 1 | 26 | 0.0000 |
| random (seed 0) | 0.25 | 2.81 | 26 | 0.130 | 0.140 | 467 | 0.138 | 13 | 13 | 1.0000 |
| random (seed 1) | 0.25 | 2.81 | 29 | 0.145 | 0.150 | 470 | 0.132 | 11 | 8 | 0.6476 |
| random (seed 2) | 0.25 | 2.81 | 28 | 0.140 | 0.135 | 472 | 0.117 | 13 | 11 | 0.8388 |
| random (seed 3) | 0.25 | 2.81 | 22 | 0.110 | 0.130 | 478 | 0.128 | 9 | 13 | 0.5235 |
| random (seed 4) | 0.25 | 2.81 | 34 | 0.170 | 0.225 | 448 | 0.129 | 18 | 10 | 0.1849 |
| random (seed 0) | 0.5 | 5.62 | 26 | 0.130 | 0.170 | 452 | 0.164 | 17 | 17 | 1.0000 |
| random (seed 1) | 0.5 | 5.62 | 23 | 0.115 | 0.210 | 444 | 0.148 | 10 | 13 | 0.6776 |
| random (seed 2) | 0.5 | 5.62 | 29 | 0.145 | 0.185 | 458 | 0.092 | 16 | 13 | 0.7111 |
| random (seed 3) | 0.5 | 5.62 | 25 | 0.125 | 0.115 | 482 | 0.128 | 13 | 14 | 1.0000 |
| random (seed 4) | 0.5 | 5.62 | 31 | 0.155 | 0.305 | 418 | 0.142 | 18 | 13 | 0.4731 |
| random (seed 0) | 1 | 11.24 | 7 | 0.035 | 0.025 | 506 | 0.252 | 6 | 25 | 0.0009 |
| random (seed 0) | 2 | 22.49 | 0 | 0.000 | 0.035 | 512 | 0.433 | 0 | 26 | 0.0000 |

## Random-direction null distribution (matched norm)

| alpha | n draws | acc mean | acc range | EOS mean | EOS range | numeral mean |
|---|---|---|---|---|---|---|
| 0.25 | 5 | 0.139 | 0.110–0.170 | 0.156 | 0.130–0.225 | 0.129 |
| 0.5 | 5 | 0.134 | 0.115–0.155 | 0.197 | 0.115–0.305 | 0.135 |
| 1 | 1 | 0.035 | 0.035–0.035 | 0.025 | 0.025–0.025 | 0.252 |
| 2 | 1 | 0.000 | 0.000–0.000 | 0.035 | 0.035–0.035 | 0.433 |

## Natural-norm run (dose-inadequate; see VERIFY.md)

| direction | applied ‖d‖ | correct/200 | acc | EOS rate | McNemar p |
|---|---|---|---|---|---|
| A ×0.5 | 0.087 | 24 | 0.120 | 0.150 | 0.7266 |
| A ×1 | 0.174 | 26 | 0.130 | 0.160 | 1.0000 |
| A ×2 | 0.349 | 26 | 0.130 | 0.155 | 1.0000 |
| D_math_full ×1 | 0.238 | 25 | 0.125 | 0.155 | 1.0000 |
| random ×1 | 0.174 | 24 | 0.120 | 0.150 | 0.7266 |
| D ×1 | 1.217 | 25 | 0.125 | 0.155 | 1.0000 |
| none ×1 | 0.000 | 24 | 0.120 | 0.140 | 0.6875 |
| none ×1 | 0.000 | 26 | 0.130 | 0.145 | 1.0000 |

Raw: results/steer_eval/*.json (per-item predictions retained). Steered neutral generations at alpha=1: results/steer_eval/neutral_gens_*_a1.md.
