# N2 — the preregistered 50-random-direction null

PREREG lists three nulls (N1 split-half, N2 fifty random directions at matched norm, N3 untrained LoRA).
This file records what N2 as saved does and does not support. Source: `results/items_N2_s0_L*_*.jsonl`
(50 logit-lens top-20 lists per layer and snippet set) and `tools/n2_null.py`.

## What the saved files contain

- `items_N2_s0_L11_neutral.jsonl`: 50 draws, modality `tokens`, raw ‖d‖ 49.26–52.20 (mean 50.68), all rescaled to eta_ref = 9.795 before decoding.
- `items_N2_s0_L11_math.jsonl`: 50 draws, modality `tokens`, raw ‖d‖ 48.89–52.40 (mean 50.42), all rescaled to eta_ref = 9.795 before decoding.
- `items_N2_s0_L15_neutral.jsonl`: 50 draws, modality `tokens`, raw ‖d‖ 48.76–52.25 (mean 50.58), all rescaled to eta_ref = 11.243 before decoding.
- `items_N2_s0_L15_math.jsonl`: 50 draws, modality `tokens`, raw ‖d‖ 49.15–51.80 (mean 50.57), all rescaled to eta_ref = 11.243 before decoding.
- `items_N2_s0_L19_neutral.jsonl`: 50 draws, modality `tokens`, raw ‖d‖ 49.12–52.70 (mean 50.52), all rescaled to eta_ref = 16.890 before decoding.
- `items_N2_s0_L19_math.jsonl`: 50 draws, modality `tokens`, raw ‖d‖ 49.30–51.98 (mean 50.61), all rescaled to eta_ref = 16.890 before decoding.

**The directions are isotropic Gaussians in R^2560, so their raw norms (~50) are a property of the draw, not of any arm.**
N2 is therefore *not* a null for arm trace norms — the null for a norm is the paired split-half floor (reported per arm in
`results/perposition_table_C.csv`) or the untrained-LoRA arm N3. Comparing an arm's ‖d‖ to N2's ‖d‖ would be meaningless.

## RNG reproduction

The direction vectors were not saved, but `tools/null_decodes.py` records the generator
(`numpy.random.default_rng([seed, layer, set_index])`, 50 × `standard_normal(2560)`), and regenerating it reproduces the
saved per-draw norms exactly (checked to 1e-3 on the first five draws of L15 neutral). Every cosine below uses regenerated vectors.

## (ii) H3's preregistered test — computable, and computed here

H3: *cos(d_A, d_B) exceeds the 95th percentile of cos(d_A, N2 draws)*.

| set | pos | cos(d_A, d_B) | N2 null: mean | 95th pct | max | H3 satisfied? |
|---|---|---|---|---|---|---|
| neutral | 1 | **-0.1266** | +0.0046 | +0.0303 | +0.0526 | **NO** |
| neutral | 2 | **-0.1402** | +0.0049 | +0.0388 | +0.0490 | **NO** |
| math | 1 | **+0.0463** | -0.0008 | +0.0323 | +0.0440 | YES |
| math | 2 | **+0.1520** | -0.0050 | +0.0229 | +0.0340 | YES |

## Where each arm falls against the N2 cosine null

For each arm X, cos(d_X, d_A) against the null distribution of cos(d_A, N2 draw). Percentile = fraction of the 50 draws below it.

| set | pos | arm X | cos(d_X, d_A) | percentile in the N2 null | above all 50? |
|---|---|---|---|---|---|
| neutral | 1 | C | +0.5049 | 100th | yes |
| neutral | 1 | D | +0.2001 | 100th | yes |
| neutral | 1 | D_math_full | +0.2655 | 100th | yes |
| neutral | 1 | B | -0.1266 | 0th | no |
| neutral | 1 | N3 | +0.0971 | 100th | yes |
| neutral | 2 | C | +0.4205 | 100th | yes |
| neutral | 2 | D | +0.1449 | 100th | yes |
| neutral | 2 | D_math_full | +0.2591 | 100th | yes |
| neutral | 2 | B | -0.1402 | 0th | no |
| neutral | 2 | N3 | -0.0012 | 40th | no |
| math | 1 | C | +0.3178 | 100th | yes |
| math | 1 | D | +0.0290 | 90th | no |
| math | 1 | D_math_full | +0.1417 | 100th | yes |
| math | 1 | B | +0.0463 | 100th | yes |
| math | 1 | N3 | +0.1131 | 100th | yes |
| math | 2 | C | +0.5740 | 100th | yes |
| math | 2 | D | +0.0976 | 100th | yes |
| math | 2 | D_math_full | +0.3770 | 100th | yes |
| math | 2 | B | +0.1520 | 100th | yes |
| math | 2 | N3 | +0.0146 | 86th | no |

## H3 verdict (all three clauses)

| clause | neutral | math |
|---|---|---|
| cos(d_A,d_B) > 95th pct of cos(d_A, N2) | **FAILS** (−0.127 vs +0.030 at p1; −0.140 vs +0.039 at p2) | passes (+0.046 vs +0.032 at p1; +0.152 vs +0.023 at p2) |
| ‖d_B‖ < ‖d_A‖ | satisfied (0.094 < 0.210 at p1; 0.097 < 0.184 at p2) | satisfied (0.229 < 0.483; 0.157 < 0.512) |
| d_A − d_B decoded and reported descriptively | **not produced** — no A−B readout artifact exists in `results/` | same |

**H3 fails on the primary (neutral) snippet set.** On neutral text d_A and d_B are *anti*-aligned (−0.13 to −0.14) while the random
null sits at ~0.00, so the two GRPO arms are not merely uncorrelated but point measurably apart — further from each other than a random
direction would be. H3 predicted the opposite. It passes only on math snippets, where both arms carry the on-domain component.
This is a preregistered hypothesis with a negative result on its primary set and must be reported as such, not omitted.

## (i) The judge-label null — NOT computable as specified

PREREG asks for a *null distribution of judge labels* over the 50 draws. That requires running the six-way judge on the N2 lists,
and the judge was never run on real readout lists (digest §8: calibration only). The headline arm statistic in this project is
instead the **Patchscope** content-relevance count under lambda selection — and the N2 files are **logit-lens** lists, not Patchscope.
Producing a Patchscope N2 null would require patching each of the 50 directions through the model on a GPU; the pod was terminated
and the adapters destroyed, so it cannot be produced now.

**Status: N2 is computed but not usable as the null for the headline (Patchscope) statistic.** It is usable, and used above, as the
cosine null that H3 actually names. Its logit-lens lists are directly comparable only to the arms' logit-lens lists, which are
themselves reported as uninterpretable for every arm (digest §6).

