# C_masked report (R1, Chat 3 replication session): is the C-vs-A trace gap a loss-placement effect?

Written 2026-09-05 03:05 Zurich on branch `replication` (results commit `f27905b`). New files only; digest, `VERIFY.md`,
`CLAIM_FIREWALL.md` and `pod` untouched. Chat 1 merges.

## Decision line

**C_masked V = 0.049 (neutral p1, L15). V ≤ 0.18 → "loss placement explains most of the gap."**
The completion-only C adapter reaches C's accuracy (187/200) yet leaves a neutral-p1 trace smaller than either A seed
(0.286 vs 0.210 / 0.155 raw; V 0.049 vs 0.125 / 0.092) and smaller than masked D_math (V 0.059). Unmasked C s0/s1 sit at V 0.50.

## What was run

| item | value |
|---|---|
| pod | RunPod `ffj2ci3ytin26z`, 2× H100 SXM 80 GB SECURE, image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, driver 580.126.09, 200 GB volume; created 02:45:01, terminated 03:02:57 Zurich; uptime 1,039 s (17.3 min) at $6.98/h ≈ **$2.01** (balance 283.57 → 281.71 at termination; pods = [] confirmed) |
| gates | env green + identity check 02:48:47 (gate 03:24); V on the Mac 02:56 (gate 07:00) |
| env | identical to the C s1 pod: fresh Python 3.11.10 venv, torch 2.13.0+cu129, transformers 5.16.1, trl 1.12.0, peft 0.20.0, accelerate 1.14.0, datasets 5.0.1, numpy 2.3.5; no vLLM (`logs/pod_C_masked/pod_packages_C_s1.txt`, `bootstrap.log`). Identity check `results/identity_check_C_masked_pod.json`: passed, zero differing fields vs `results/identity_check.json` other than timestamp / git_commit / trl path |
| data | `data/C_samples.jsonl` sha256 78022b70… (and both snippet sets) verified on the pod before training; every row's `text` == `prompt` + `completion` (checked over all 15,248 rows) |
| training | `grpo/train_sft.py train --arm C --data data/C_samples.jsonl --model Qwen/Qwen3.5-4B-Base --model-revision 1001bb4d… --out runs/C_masked_s0 --max-steps 225 --seed 0 --save-every 25 --completion-only` at commit `e3ceb47`. r=32, α=64, lr 1e-4, batch 8, max_len 768. 225/225 steps, train loss 0.110 (C s1 unmasked: 0.377), 320 s. `--arm C_masked` is not an accepted choice (C/Cp/D/D_math); `--arm` only sets the max-steps requirement and a run_meta label, so `--arm C` + `--completion-only` is the identical code path D_math used (`run_meta.json`: `completion_only_loss: true`) |
| masking | TRL 1.12 prompt-completion format: loss on completion tokens + EOS, prompt tokens (question + `\nAnswer:`) masked — exactly D_math's mechanism. **Fraction of tokens supervised: 0.726** (1,452,261 of 1,999,870 tokens over the 8,792 selected rows; 547,609 prompt tokens masked; `results/supervised_fraction_C_masked.json`). C s0 (seed 0, same shuffle, same 2M-token cap) trains on all of them |
| base cache | recomputed at L15, bit-identical again (neutral 63e24d99…, math 760d0ee5…) |
| adapter | `adapters/C_masked_s0/final/` on the Mac (untracked, 260 MB), sha256 a81d0025… equal to the pod copy; `adapters/C_masked_s0/run_meta.json` (selected_text_sha256 7a1c5d22…) |

## 1. V = ‖d_neutral,p1‖ / ‖ΔW‖_F at L15 (`results/visibility_table_C_masked.md`, `results/lora_delta_stats_C_masked.json`)

| arm | ‖ΔW‖_F | ‖d_neutral,p1‖ | split-half floor | **V** | ‖d_math,p1‖ | floor | V (math) |
|---|---|---|---|---|---|---|---|
| **C_masked** | 5.844 | 0.286 | 0.039 | **0.049** | 0.645 | 0.014 | 0.110 |
| C s0 | 6.963 | 3.488 | 0.435 | 0.501 | 5.380 | 0.152 | 0.773 |
| C s1 | 6.958 | 3.498 | 0.444 | 0.503 | 5.204 | 0.162 | 0.748 |
| A s0 | 1.675 | 0.210 | 0.029 | 0.125 | 0.483 | 0.011 | 0.288 |
| A s1 | 1.682 | 0.155 | 0.023 | 0.092 | 0.343 | 0.009 | 0.204 |
| D_math (masked) | 6.579 | 0.389 | 0.057 | 0.059 | 5.107 | 0.101 | 0.776 |
| D_math_full | 6.702 | 1.199 | 0.144 | 0.179 | 10.053 | 0.326 | 1.500 |

C_masked's ‖ΔW‖_F (5.84) is 84% of C s0's, so the 12× drop in the neutral-p1 trace (3.49 → 0.29) is not a weight-norm effect.
Largest module ‖ΔW_m‖_F 0.551 (layers.2.mlp.up_proj); σ_max 0.355. Trace ratios at neutral p1: C_masked/A s0 = 1.36, C_masked/A s1 = 1.85 (vs 16.6–22.6 for unmasked C).

## 2. Geometry p1–p2 (`results/perposition_table_C_masked.csv`)

| set | pos | raw ‖d‖ | floor | cos(halves) | constancy |
|---|---|---|---|---|---|
| neutral | 1 | 0.286 | 0.039 | 0.992 | 0.252 |
| neutral | 2 | 0.257 | 0.042 | 0.988 | 0.189 |
| math | 1 | 0.645 | 0.014 | 1.000 | 0.766 |
| math | 2 | 0.651 | 0.086 | 0.991 | 0.310 |

Math/neutral ratio at p1 = 2.26 (C s0: 1.54; A s0: 2.30).

## 3. Cosines of the C_masked diff vector (`results/perposition_table_C_masked_cosine.csv`)

| y | neutral p1 | neutral p2 | math p1 | math p2 |
|---|---|---|---|---|
| C s0 | 0.320 | 0.268 | 0.362 | 0.540 |
| C s1 | 0.297 | 0.252 | 0.318 | 0.522 |
| A s0 | **0.624** | **0.584** | 0.570 | **0.743** |
| A s1 | 0.494 | 0.436 | 0.521 | 0.726 |
| D_math (masked) | 0.187 | 0.118 | 0.283 | 0.392 |
| D_math_full | 0.230 | 0.179 | 0.175 | 0.351 |
| N1 halves | −0.068 | −0.026 | 0.027 | 0.164 |

C_masked is closer in direction to A (0.62 / 0.58 neutral, 0.74 math p2; A s0 · A s1 itself is 0.68 / 0.63 / 0.79) than to
unmasked C (0.32 / 0.27). Unmasked C · A was 0.50 / 0.42 for every seed pair.

## 4. Held-out accuracy, both parsers (`results/acc_C_masked_s0.json`, `results/acc_table_C_masked.md`)

| arm | raw | re-scored |
|---|---|---|
| C_masked | 187/200 = 0.935 | 187/200 = 0.935 |
| C s0 | 186/200 = 0.930 | 0.930 |
| C s1 | 185/200 = 0.925 | 0.925 |
| A s0 | 188/200 = 0.940 | 0.940 |
| D_math | 132/200 = 0.660 | 173/200 = 0.865 |

Paired McNemar (identical under both parsers for the C/A rows): C_masked vs C s0 5 / 4, p = 1.00; vs C s1 7 / 5, p = 0.77; vs A s0 4 / 5, p = 1.00; vs D_math 60 / 5, p < 0.001 (re-scored 21 / 7, p = 0.013).

## 5. Patchscope L15 p1 λ=1 (`results/patchscope_C_masked_s0_step225_L15.json`)

- neutral (raw 0.286, common support 4,968): `'→' '1' '0' '\n' '2' '9' '8' '-' '5' '3' '6' '7' '…' '4' 'Let' '->' '>' 'now' 'I' 'lets'`
- math (raw 0.645, common support 4,964): `' ' '\n' ' high' ' <' ' we' ' {' ' true' ' our' ' dig' ' man' ' pig' ' the' ' a' ' before' ' "' ' ok' ' watch' ' -' ' your' ' >'`

Reported as decoded; no interpretation.

## Deviations and incidents

- `--arm C` was used as the label because `C_masked` is not an accepted `--arm` value and existing files may not be edited; the
  result files carry `arm: C_masked` via the readout tools' free-text `--arm`. `run_meta.json` says `arm: C`, `completion_only: true`.
- The second pod's bootstrap wrote its identity check to the C s1 filename; it was saved as `identity_check_C_masked_pod.json`
  and the tracked C s1 file restored from git before committing.
- Pod price $6.98/h as in the C s1 run. Total for R1 ≈ $2.01; cumulative for both sessions ≈ $4.5, under the $10 cap.

## Not checked

- Which 1,800 of the 8,792 rows the trainer actually saw (TRL reshuffles the dataset with the SFTConfig seed); the supervised
  fraction is the whole-selection expectation (0.726; first 1,800 rows in selection order 0.725).
- That C_masked and C s0 saw the same rows in the same order: same seed-0 shuffle, same texts and the same cap make it the
  same selection (8,792 rows) by construction, but C s0's run_meta did not survive, so the selection sha was not compared.
- Other layers/positions, emergence, token-relevance grading, steering, judge calls, the repository test suite on the pod.

## Three most likely ways this is wrong

1. **The "trace" may be a prompt-token phenomenon rather than a learning-rule one, but the neutral snippets have no prompt.**
   Masking removed the loss on question tokens, and the neutral-p1 trace collapsed 12× while ‖ΔW‖_F fell only 16%. That is
   consistent with the trace being driven by what is supervised at early positions (question openings), not by SFT vs GRPO.
   But it is one seed, one masking pattern; a masked-C seed 1 or a C trained with loss on prompt tokens only would separate
   "position of supervision" from "content of supervision".
2. **Accuracy parity may hide a different solution.** C_masked matches C and A on 200 items (all p ≥ 0.77), but its diff
   direction is closer to A than to C; readouts here say nothing about whether the completions themselves changed.
3. **Scale comparison across arms.** V divides by ‖ΔW‖_F, which differs across arms by 4×; the masked arms have small numerators
   near their floors (C_masked 0.286 vs floor 0.039, A 0.155–0.210 vs floors 0.023–0.029). The ordering C_masked < A is only a
   1.4–1.9× ratio at neutral p1 and should be read as "near A", not "below A".
