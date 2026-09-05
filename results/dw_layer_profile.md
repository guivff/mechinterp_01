# Per-layer ||dW||_F profile — C s1 vs C_masked s0 (agent first pass, C4 task 3)

dW = (alpha/r)·B·A per module, alpha/r = 2; 248 / 248 modules; CPU, float32. Totals: C s1 **6.9577** (stats file 6.9577), C_masked **5.8445** (stats file 5.8445).

**Fraction of ||dW||_F² in layers ≤ 15: C s1 0.505, C_masked 0.515** (layers ≤ 15 hold 16 of 32 blocks; uniform would be 0.500). Layer-15 norm ratio C_masked / C s1 = 0.885; overall 0.840.

| layer | ||dW||_F C s1 | share C s1 | ||dW||_F C_masked | share C_masked | ratio C_masked / C s1 |
|---|---|---|---|---|---|
| 0 | 1.3437 | 0.0373 | 1.0426 | 0.0318 | 0.776 |
| 1 | 1.3181 | 0.0359 | 1.0347 | 0.0313 | 0.785 |
| 2 | 1.2730 | 0.0335 | 1.0841 | 0.0344 | 0.852 |
| 3 | 1.2699 | 0.0333 | 1.0191 | 0.0304 | 0.802 |
| 4 | 1.2675 | 0.0332 | 1.0409 | 0.0317 | 0.821 |
| 5 | 1.2427 | 0.0319 | 1.0411 | 0.0317 | 0.838 |
| 6 | 1.2541 | 0.0325 | 1.0535 | 0.0325 | 0.840 |
| 7 | 1.1802 | 0.0288 | 1.0396 | 0.0316 | 0.881 |
| 8 | 1.2217 | 0.0308 | 1.0671 | 0.0333 | 0.874 |
| 9 | 1.2215 | 0.0308 | 1.0660 | 0.0333 | 0.873 |
| 10 | 1.2316 | 0.0313 | 1.0694 | 0.0335 | 0.868 |
| 11 | 1.1663 | 0.0281 | 1.0213 | 0.0305 | 0.876 |
| 12 | 1.2066 | 0.0301 | 1.0590 | 0.0328 | 0.878 |
| 13 | 1.2139 | 0.0304 | 1.0726 | 0.0337 | 0.884 |
| 14 | 1.2022 | 0.0299 | 1.0545 | 0.0326 | 0.877 |
| 15 | 1.1431 | 0.0270 | 1.0112 | 0.0299 | 0.885 |
| 16 | 1.2067 | 0.0301 | 1.0462 | 0.0320 | 0.867 |
| 17 | 1.2195 | 0.0307 | 1.0438 | 0.0319 | 0.856 |
| 18 | 1.2262 | 0.0311 | 1.0331 | 0.0312 | 0.842 |
| 19 | 1.1751 | 0.0285 | 0.9955 | 0.0290 | 0.847 |
| 20 | 1.2160 | 0.0305 | 1.0080 | 0.0297 | 0.829 |
| 21 | 1.2127 | 0.0304 | 1.0063 | 0.0296 | 0.830 |
| 22 | 1.2289 | 0.0312 | 1.0201 | 0.0305 | 0.830 |
| 23 | 1.1900 | 0.0293 | 0.9921 | 0.0288 | 0.834 |
| 24 | 1.2365 | 0.0316 | 1.0166 | 0.0303 | 0.822 |
| 25 | 1.2343 | 0.0315 | 1.0162 | 0.0302 | 0.823 |
| 26 | 1.2473 | 0.0321 | 1.0232 | 0.0306 | 0.820 |
| 27 | 1.2088 | 0.0302 | 1.0058 | 0.0296 | 0.832 |
| 28 | 1.2662 | 0.0331 | 1.0277 | 0.0309 | 0.812 |
| 29 | 1.2701 | 0.0333 | 1.0319 | 0.0312 | 0.812 |
| 30 | 1.2687 | 0.0332 | 1.0396 | 0.0316 | 0.819 |
| 31 | 1.1723 | 0.0284 | 0.9681 | 0.0274 | 0.826 |

Ratio range over layers: 0.776 (layer 0) – 0.885 (layer 15); median 0.838.

## Largest modules

| adapter | module | ||dW||_F |
|---|---|---|
| C s1 | layers.1.linear_attn.in_proj_qkv | 0.6850 |
| C s1 | layers.0.linear_attn.in_proj_qkv | 0.6780 |
| C s1 | layers.30.mlp.gate_proj | 0.6701 |
| C s1 | layers.29.mlp.gate_proj | 0.6658 |
| C s1 | layers.28.mlp.gate_proj | 0.6618 |
| C_masked | layers.2.mlp.up_proj | 0.5511 |
| C_masked | layers.7.mlp.gate_proj | 0.5447 |
| C_masked | layers.9.mlp.gate_proj | 0.5445 |
| C_masked | layers.2.mlp.gate_proj | 0.5435 |
| C_masked | layers.8.mlp.gate_proj | 0.5418 |

## Regenerate

```bash
python tools/dw_layer_profile.py --c_s1 <adapters>/C_s1/final --c_masked <adapters>/C_masked_s0/final
```

Adapters are untracked (`~/repl/adapters/` on the Mac at the time of writing). No interpretation; agent-produced (C4).
