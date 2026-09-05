# E2 — trace ablation table (fine-tuned model, L15, every position)

Δ = correct(ablated) − correct(own α=0 run), items / 200. Primary = stopping-robust parser; raw last-number parser beside it.
Slot rule: d_p at position p for p ≤ 4, pooled positions ≥ 5 elsewhere (`own`); `pooled` = one all-position mean everywhere (secondary);
`randK` = matched-norm Gaussian per slot, seed K; `crossX` = X's vector at X's norm; `dC_s1` on base = sanity.

## Smoke: α = 0 vs saved accuracy (tolerance ±2)

| arm | saved robust | saved raw | saved batch | E2 α=0 robust | E2 α=0 raw | E2 batch | within ±2 |
|---|---|---|---|---|---|---|---|
| C_s1 | 185/200 | 185/200 | 8 | 185/200 | 185/200 | 25 | yes |
| C_masked_s0 | 187/200 | 187/200 | 8 | 186/200 | 186/200 | 25 | yes |
| base | 158/200 | 28/200 | 25 | 155/200 | 23/200 | 25 | NO |

## Runs

| arm | direction | α | robust correct | Δ robust | raw correct | Δ raw | EOS rate | cap-hit rate | mean new tokens | position source | threshold / reading |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C_masked_s0 | none | 0 | 186/200 | +0 | 186/200 | +0 | 1.000 | 0.000 | 168.3 | {'position_ids': 0, 'counter': 0} | reference |
| C_masked_s0 | crossC_s1 | 1 | 187/200 | +1 | 187/200 | +1 | 1.000 | 0.000 | 166.9 | {'position_ids': 2458, 'counter': 0} | descriptive |
| C_masked_s0 | own | 0.5 | 188/200 | +2 | 188/200 | +2 | 1.000 | 0.000 | 167.1 | {'position_ids': 2520, 'counter': 0} |  |
| C_masked_s0 | own | 1 | 188/200 | +2 | 188/200 | +2 | 1.000 | 0.000 | 165.9 | {'position_ids': 2434, 'counter': 0} | not load-bearing (Δ ≥ −3) |
| C_masked_s0 | own | 2 | 189/200 | +3 | 189/200 | +3 | 1.000 | 0.000 | 165.2 | {'position_ids': 2469, 'counter': 0} |  |
| C_masked_s0 | pooled | 1 | 188/200 | +2 | 188/200 | +2 | 1.000 | 0.000 | 166.7 | {'position_ids': 2456, 'counter': 0} | secondary |
| C_masked_s0 | rand0 | 1 | 187/200 | +1 | 187/200 | +1 | 1.000 | 0.000 | 168.7 | {'position_ids': 2522, 'counter': 0} | ok (Δ > −6) |
| C_masked_s0 | rand1 | 1 | 185/200 | -1 | 185/200 | -1 | 1.000 | 0.000 | 168.8 | {'position_ids': 2517, 'counter': 0} | ok (Δ > −6) |
| C_masked_s0 | rand2 | 1 | 187/200 | +1 | 187/200 | +1 | 1.000 | 0.000 | 167.8 | {'position_ids': 2537, 'counter': 0} | ok (Δ > −6) |
| C_masked_s0 | rand3 | 1 | 186/200 | +0 | 186/200 | +0 | 1.000 | 0.000 | 168.3 | {'position_ids': 2522, 'counter': 0} | ok (Δ > −6) |
| C_masked_s0 | rand4 | 1 | 187/200 | +1 | 187/200 | +1 | 1.000 | 0.000 | 167.4 | {'position_ids': 2530, 'counter': 0} | ok (Δ > −6) |
| C_s1 | none | 0 | 185/200 | +0 | 185/200 | +0 | 1.000 | 0.000 | 168.5 | {'position_ids': 0, 'counter': 0} | reference |
| C_s1 | crossC_masked_s0 | 1 | 185/200 | +0 | 185/200 | +0 | 1.000 | 0.000 | 166.8 | {'position_ids': 2438, 'counter': 0} | descriptive |
| C_s1 | own | 0.5 | 186/200 | +1 | 186/200 | +1 | 1.000 | 0.000 | 170.1 | {'position_ids': 2612, 'counter': 0} |  |
| C_s1 | own | 1 | 187/200 | +2 | 187/200 | +2 | 0.995 | 0.005 | 167.9 | {'position_ids': 2546, 'counter': 0} | not load-bearing (Δ ≥ −3) |
| C_s1 | own | 2 | 188/200 | +3 | 188/200 | +3 | 0.995 | 0.005 | 170.0 | {'position_ids': 2632, 'counter': 0} |  |
| C_s1 | pooled | 1 | 188/200 | +3 | 188/200 | +3 | 1.000 | 0.000 | 168.1 | {'position_ids': 2551, 'counter': 0} | secondary |
| C_s1 | rand0 | 1 | 186/200 | +1 | 186/200 | +1 | 0.995 | 0.005 | 168.5 | {'position_ids': 2602, 'counter': 0} | ok (Δ > −6) |
| C_s1 | rand1 | 1 | 186/200 | +1 | 186/200 | +1 | 1.000 | 0.000 | 172.5 | {'position_ids': 2658, 'counter': 0} | ok (Δ > −6) |
| C_s1 | rand2 | 1 | 185/200 | +0 | 185/200 | +0 | 0.990 | 0.010 | 169.3 | {'position_ids': 2798, 'counter': 0} | ok (Δ > −6) |
| C_s1 | rand3 | 1 | 183/200 | -2 | 183/200 | -2 | 1.000 | 0.000 | 170.5 | {'position_ids': 2431, 'counter': 0} | ok (Δ > −6) |
| C_s1 | rand4 | 1 | 185/200 | +0 | 185/200 | +0 | 0.990 | 0.010 | 169.9 | {'position_ids': 2825, 'counter': 0} | ok (Δ > −6) |
| base | none | 0 | 155/200 | +0 | 23/200 | +0 | 0.135 | 0.865 | 470.2 | {'position_ids': 0, 'counter': 0} | reference |
| base | dC_s1 | 1 | 153/200 | -2 | 21/200 | -2 | 0.165 | 0.835 | 459.0 | {'position_ids': 4096, 'counter': 0} | sanity ok (|Δ| ≤ 3) |

## Decision lines

- **C_s1**: not load-bearing (Δ ≥ −3) (worst random seed Δ = -2 > −6, control passes)
- **C_masked_s0**: not load-bearing (Δ ≥ −3) (worst random seed Δ = -1 > −6, control passes)
- **base sanity** (d_C_s1 subtracted from base, α=1): robust 153 vs base α=0 155 → Δ -2; ok

Direction norms per slot (p0..p4, pooled≥5) from the sidecars:
- C_s1: [6.284, 3.498, 2.484, 2.173, 1.808, 0.77]; all-position pooled 0.805; eta_ref 11.24; max |norm − tracked perposition norm| p0–4 = 0.0
- C_masked_s0: [0.478, 0.286, 0.257, 0.231, 0.221, 0.254]; all-position pooled 0.250; eta_ref 11.24; max |norm − tracked perposition norm| p0–4 = 0.0
