# E2 — Ablation of the trace from the fine-tuned model (report, AGENTS.md §7 format)

Second replication pod (`mats-E2-ablation`, id `f1teeax8pngu8n`), parallel to R2. Sat 2026-09-05, session 05:21–06:25 Zurich.
Preregistration: PREREG.md amendment "~05:40 Ablation test preregistered" on `origin/pod` @ `49e1fdc28d2f07c8abce52eb9886d566e31fec31`
(`sha256(PREREG.md) = da758edfd1459103086b1ed863671662f71eece0b70dffd34ee389c972f6ec74`), read before the first pod call; the brief as
received is in `results/E2_PREREG_AS_RECEIVED.md`. Code commit on the pod: `1402cdb` (branch `e2-ablation`, off `replication` @ `cf5c203`).

## Decision lines (stopping-robust parser, 200 GSM8K test items, Δ = ablated − own α = 0, items)

- **C s1, own direction, α = 1: Δ = +2 (187 vs 185) → not load-bearing (threshold Δ ≥ −3).** Random matched-norm control: worst of five seeds Δ = −2 (seed 3: 183) (> −6) → control passes; the reading stands.
- **C_masked s0, own direction, α = 1: Δ = +2 (188 vs 186) → not load-bearing (threshold Δ ≥ −3).** Random matched-norm control: worst of five seeds Δ = −1 (seed 1: 185) (> −6) → control passes; the reading stands.
- **Base sanity (d_C_s1 subtracted from base, α = 1): Δ = −2 (153 vs 155 robust; 21 vs 23 raw) vs base α = 0 at the same batch (batch 25, this session) → sanity ok (|Δ| ≤ 3). Caveat: this session's base α = 0 (155 robust / 23 raw) is itself 3 / 5 items below the saved `acc_base_s0.json` (158 / 28, also batch 25, first pod); base completions hit the 512 cap 173/200 times and greedy bf16 drift over ~470 tokens is larger than for the C arms, which stop at ~170. Against the saved file the ablated base would be −5 robust, i.e. flagged; the flag concerns base reproducibility across pods, not the direction.**
- **Smoke (α = 0, batch 25): C s1 185/200 robust and raw — the saved 185/200 exactly; C_masked 186/200 — saved 187/200, within ±2.** Both reported before 06:15.
- **A and C s0 cannot be ablated: their adapters were destroyed with the first pod.** Only `adapters/C_s1/final` (sha256 d17ae2d2…) and `adapters/C_masked_s0/final` (a81d0025…) exist; both were verified byte-identical on the pod.
- Secondaries: α = 0.5 → C s1 186, C_masked 188; α = 2 → C s1 188, C_masked 189. Pooled all-position vector at α = 1 → C s1 188 (Δ +3), C_masked 188 (Δ +2). Cross-arm (each at its own norm): d_C_masked on C s1 → 185 (Δ 0); d_C_s1 on C_masked → 187 (Δ +1) (descriptive).
- Raw last-number parser gave the same count as the stopping-robust parser in every run (every completion of both arms ends with EOS; cap hits ≤ 1/200). EOS rate stayed 0.995–1.000 under every perturbation; mean completion length 165–170 new tokens throughout.

Reading (no causal language beyond the preregistered lines): subtracting either arm's own L15 neutral mean-difference — at every position, prompt and
generation, at up to twice its norm — leaves GSM8K accuracy and stopping behaviour unchanged within greedy-bf16 noise. Consistent with the
loss-placement prediction recorded in the amendment (C s1 not load-bearing; C_masked not load-bearing at its small norm). The ablation
vectors are small relative to the residual stream (‖d̄≥5‖ = 0.77 for C s1 and 0.25 for C_masked against eta_ref = 11.24; the p0 vector is 6.3
but affects one position), so "not load-bearing" here means "this mean-difference direction at this norm is not what carries the behaviour",
not that the fine-tune's effect is unlocalised.

## What was run (files, args, seeds, commit)

- Pod: 2× H100 SXM SECURE, image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, `tools/pod_bootstrap_E2.sh` (same pins as R1:
  torch 2.13.0+cu129, transformers 5.16.1, trl 1.12.0, peft 0.20.0; identity check passed, zero differing fields). Pod created 05:26:26, env green 05:41:02.
- Base L15 cache recomputed on the pod (`tools/cache_base_activations.py --layers 15`): neutral `63e24d99…`, math `760d0ee5…` — bit-identical to R1.
- Vectors: `tools/ablation_dirs.py` (new) → `results/ablation_dirs_{C_s1,C_masked_s0}.{npz,json}`: d_p for p = 0..4, pooled mean of positions ≥ 5, pooled
  mean of all positions, from base cache + adapter activations on the neutral set (500 × 128 tokens, sha c8673772…). Per-position norms p0–4 equal the
  tracked `results/perposition_*_step225_L15.json` values to 0.0 (C s1: 6.284 / 3.498 / 2.484 / 2.173 / 1.808, pooled ≥5 0.770, all 0.805;
  C_masked: 0.478 / 0.286 / 0.257 / 0.231 / 0.221, pooled ≥5 0.254, all 0.250). No file from the destroyed pod was used.
- Ablation: `tools/ablate_trace.py` (new). Forward hook on decoder block 15 output (`readout.diff._get_blocks`, same block the caches use) subtracting
  α·d[slot(p)] with slot(p) = p for p ≤ 4, 5 (pooled ≥5) for p ≥ 5, at every position including generated tokens; p read from the model's own
  `position_ids` captured by a pre-hook on the decoder stack (used on every call of every run; counter fallback never triggered). Generation as
  `grpo/eval_acc.py`: first 200 GSM8K test items (`openai/gsm8k` main/test, revision 740312a…; set sha 49b3a3f8…), prompt `"{question}\nAnswer:"`,
  greedy, cap 512, left padding, bf16, PEFT adapter unmerged, **batch 25** (saved C evals used 8, saved base used 25). Scoring: raw
  `grpo.train_grpo.extract_answer` and stopping-robust `tools.reparse_acc.cut` → `extract_answer`, in-process, both stored per item.
- Random control: per slot an independent Gaussian direction scaled to that slot's ‖d_p‖, seeds 0–4 (`numpy.default_rng(seed)`); cosines to the arm's d recorded.
- Chain: `tools/run_E2_ablation.sh` (GPU 0: C s1; GPU 1: cache → vectors → C_masked → base), `logs/pod_E2/chain_E2.log`. Table: `tools/ablation_table.py` → `results/ablation_table.md`.
- Outputs: `results/ablation_{arm}_{direction}_a{alpha}.json` (every completion, both parses, EOS/cap flags, token counts, hook statistics, direction
  metadata and hashes); `results/ablation_table.md`; ledger `logs/E2_ablation_ledger.md`; pod logs under `logs/pod_E2/`.
- Cost: pod at $6.98/h, created 05:26:26, terminated 06:19:12 (uptime 52.8 min; account pods = [] afterwards) → ≈ $6.1.

## Full table

See `results/ablation_table.md` (copied below at the end of the run).

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

## What I did NOT check

- No second seed of anything: one adapter per arm, one greedy decode per configuration; ±2 items is the noise band and every Δ here is inside ±4.
- The hook subtracts at every position including left-pad positions (which carry position 0, i.e. slot 0); pads are masked from real tokens by the
  attention mask and by the linear-attention padding mask, and the α = 0 smoke is exact, but I did not run an unpadded (batch 1) replicate.
- Only layer 15 and only the block-output residual; no layer sweep, no attention/MLP-specific ablation, no projection-out (the vector is subtracted, not projected away).
- Base sanity compares against a base α = 0 run at batch 25 made in this session (saved base eval was also batch 25: 28 raw / 158 robust); no batch-8 base replicate.
- Cross-arm and pooled variants are descriptive; no threshold was preregistered and none is applied.
- The R2 pod (`iq441ukig8d7ep`) was left untouched; nothing from R2 was read or used.

## Three ways this is wrong

1. **The ablation vector is a mean over neutral text, and subtracting it on GSM8K prompts is off-distribution for that estimate.** d was estimated on
   500 Pile-style neutral snippets; on math prompts and on the model's own generated tokens the fine-tune's residual offset may point elsewhere or
   be larger (R1: math-set norms are 2–3× the neutral ones). A null result on GSM8K therefore cannot rule out a load-bearing *math-conditioned* trace;
   it only says the neutral-text trace, the thing the readouts decode, is not what carries the behaviour.
2. **A mean-difference subtraction is the weakest possible ablation.** It removes one constant direction with a fixed magnitude; if the behaviour lives in
   the variance around the mean, in a subspace, or in a magnitude that differs per token, a constant subtraction leaves it intact. The correct
   comparison is a per-token projection-out or a rank-k subspace ablation, neither of which was preregistered or run. The random control at matched norm
   passing shows the perturbation is harmless, which is expected when the norm is ≤ 7 % of the residual norm — the test may have been underpowered by
   construction for C_masked (‖d‖ ≈ 0.25 vs eta_ref 11.2) and for C s1 beyond position 4.
3. **Position handling and the batch.** Slot assignment relies on the model's `position_ids` (verified on a mini-run and recorded per run), but if
   transformers 5.16's Qwen3.5 path ever passes cache-relative rather than absolute positions in some step, later tokens would receive the wrong slot —
   for p ≥ 5 all slots map to the same pooled vector, so the failure would be confined to the first five real tokens. Batch 25 vs the saved batch-8 C
   evals changes bf16 padding numerics; the smoke reproduces 185 exactly and 186 vs 187, so any such effect is inside the tolerance, but the
   Δ's of +2 are of the same size as that noise and should be read as 0.
