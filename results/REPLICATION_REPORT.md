# Arm C seed-1 replication report (Chat 3 / V4)

Written 2026-09-05 00:35 Zurich on branch `replication` (results commit `d056ac4`). New files only; the digest,
`VERIFY.md` and `CLAIM_FIREWALL.md` were not touched. Chat 1 merges.

## What was run

| item | value |
|---|---|
| pod | RunPod `9wyia6f79b95q3`, 2x H100 SXM 80 GB (SECURE), image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, driver 580.126.09, 200 GB volume |
| created / terminated | 00:08:45 / 00:30:43 Zurich; uptime 1,278 s (21.3 min) at $6.98/h ≈ **$2.48** (account balance 286.13 → 284.12 at termination; pods = [] confirmed) |
| env | fresh Python 3.11.10 venv: torch 2.13.0+cu129, transformers 5.16.1, trl 1.12.0, peft 0.20.0, accelerate 1.14.0, datasets 5.0.1, numpy 2.3.5, tokenizers 0.23.2; **no vLLM**. Receipt `logs/pod_C_s1/pod_packages_C_s1.txt`, `logs/pod_C_s1/bootstrap.log` |
| gate | env green + identity check at 00:15 Zurich (gate 01:26). `tools/identity_check.py` → `results/identity_check_C_s1_pod.json`: passed; zero fields differ from `results/identity_check.json` other than `timestamp`, `git_commit`, `trl_grpo_trainer_path`; TRL `grpo_trainer.py` sha256 equal |
| data | sha256 verified on the pod before training: `data/C_samples.jsonl` 78022b70…, `data/snippets/neutral.jsonl` c8673772…, `data/snippets/math.jsonl` 483c3733… (`logs/pod_C_s1/data_sha_C_s1.txt`) |
| model | `Qwen/Qwen3.5-4B-Base` @ `1001bb4d826a52d1f399e183466143f4da7b741b`; GSM8K @ `740312add88f781978c0658806c59bc2815b9866` |
| training | `grpo/train_sft.py train --arm C --data data/C_samples.jsonl --max-steps 225 --seed 1 --save-every 25` (r=32, α=64, lr 1e-4, batch 8, max_len 768, unmasked; 225×8 = 1,800 rows seen once). Code commit `946f971`. 225/225 steps, train loss 0.377, 323 s. Log `logs/pod_C_s1/C_s1.log`; `adapters/C_s1/run_meta.json` (selected_text_sha256 0184dde9…, n_texts 8,794 under the 2M-token cap) |
| base cache | `tools/cache_base_activations.py --layers 15` recomputed on the fresh pod: `base_L15_neutral.npy` sha256 63e24d99… and `base_L15_math.npy` 760d0ee5… — **bit-identical** to the original sidecars in `results/cache/` |
| readouts | `grpo/eval_acc.py` (200 GSM8K test, greedy, cap 512) → `results/acc_C_s1.json`; `tools/acc_table_C_s1.py` → `results/acc_table_C_s1.md/.json`; `tools/per_position_diff.py --layer 15` → `results/perposition_C_s1_step225_L15.json`; `tools/perposition_table.py --arms C:225:1` → `results/perposition_table_C_s1*.csv/.md` + `results/cache/diffs/diff_C_s1_step225_L15_*` (10 vectors, committed); `tools/cross_seed_cosine_C.py` → `results/perposition_table_C_seeds{,_cosine}.csv`, `results/trace_ratio_C_A_seeds.csv`, `.meta.json`; `tools/lora_delta_stats.py` → `results/lora_delta_stats_C_s1.json`; `tools/patchscope.py --layer 15 --positions 1` → `results/patchscope_C_s1_step225_L15.json` |
| adapter | `adapters/C_s1/final/` on the Mac (untracked, 260 MB), `adapter_model.safetensors` sha256 d17ae2d2… equal to the pod copy before termination |
| commits | `946f971` scripts; `b974f2e` cosine filename fix (its pod-side twin `ce2510c`, same two files byte-identical, is the `git_commit` recorded inside `acc_table_C_s1.json`, `perposition_table_C_seeds.meta.json`, `patchscope_C_s1_step225_L15.json`, `perposition_table_C_s1*`); `d056ac4` results + logs |

## 1. Held-out accuracy, C s0 vs C s1 (both parsers)

Same 200 GSM8K test items (set sha 49b3a3f8…), greedy, cap 512. "raw" = preregistered last-number parser on the full
completion; "re-scored" = `tools/reparse_acc.cut` truncation then the same extractor. Neither parser changes any count for these arms.

| arm | seed | step | raw correct | raw acc | re-scored correct | re-scored acc |
|---|---|---|---|---|---|---|
| C | 0 | 225 | 186/200 | 0.930 | 186/200 | 0.930 |
| C | 1 | 225 | 185/200 | 0.925 | 185/200 | 0.925 |
| A | 0 | 150 | 188/200 | 0.940 | 188/200 | 0.940 |

Paired (identical under both parsers):

| x | y | both | x only | y only | neither | McNemar exact p |
|---|---|---|---|---|---|---|
| C s1 | C s0 | 183 | 2 | 3 | 12 | 1.000 |
| C s1 | A s0 | 182 | 3 | 6 | 9 | 0.508 |

(C s0 vs A s0 from the digest: 7 / 5, p = 0.77.)

## 2. Cross-seed cosine of the L15 per-position diff vector (the number that matters most)

C s1 · C s0 uses the surviving seed-0 vectors `results/cache/diffs/diff_C_s0_step225_L15_*` and the new
`diff_C_s1_step225_L15_*`; no seed-0 adapter was needed. Other arms' cross-seed cosines are copied from
`results/perposition_table_seeds_cosine.csv` and `results/perposition_table_A_seeds_cosine.csv`.

| pair | neutral p1 | neutral p2 | math p1 | math p2 |
|---|---|---|---|---|
| **C s1 · C s0** | **0.983** | **0.972** | **0.969** | **0.984** |
| D s0 · D s1 | 0.978 | 0.974 | 0.951 | 0.970 |
| D_math_full s0 · s1 | 0.938 | 0.920 | 0.961 | 0.989 |
| A s0 · A s1 | 0.676 | 0.629 | 0.622 | 0.788 |
| C s1 · A s0 | 0.504 | 0.417 | 0.280 | 0.543 |
| C s1 · A s1 | 0.481 | 0.410 | 0.302 | 0.569 |
| C s0 · A s0 | 0.505 | 0.421 | 0.318 | 0.574 |
| C s0 · A s1 | 0.484 | 0.403 | 0.334 | 0.581 |

The C cross-seed cosine (0.969–0.984) lies inside D's range and above D_math_full's; it is far above A's (0.62–0.79).
C · A is unchanged by the new C seed (0.28–0.57 for every C seed × A seed pair). All 5 positions and both sets are in
`results/perposition_table_C_seeds_cosine.csv`.

## 3. Per-position geometry L15 (both seeds; `results/perposition_table_C_seeds.csv`)

| seed | set | pos | raw ‖d‖ | split-half floor | cos(halves) | constancy | base ‖h‖ |
|---|---|---|---|---|---|---|---|
| 0 | neutral | 1 | 3.488 | 0.435 | 0.992 | 0.274 | 12.55 |
| 1 | neutral | 1 | 3.498 | 0.444 | 0.992 | 0.275 | 12.55 |
| 0 | neutral | 2 | 2.434 | 0.423 | 0.985 | 0.171 | 12.17 |
| 1 | neutral | 2 | 2.484 | 0.420 | 0.986 | 0.171 | 12.17 |
| 0 | math | 1 | 5.380 | 0.152 | 1.000 | 0.674 | 15.88 |
| 1 | math | 1 | 5.204 | 0.162 | 1.000 | 0.631 | 15.88 |
| 0 | math | 2 | 5.251 | 0.402 | 0.997 | 0.468 | 11.31 |
| 1 | math | 2 | 5.141 | 0.394 | 0.997 | 0.463 | 11.31 |

Seed-0 rows are copied from `results/perposition_table_C.csv`; seed-1 rows come from the recomputed (bit-identical) base cache.

## 4. ‖ΔW‖_F and visibility V = ‖d_neutral,p1‖ / ‖ΔW‖_F

| seed | ‖ΔW‖_F | largest module ‖ΔW_m‖_F | σ_max | V |
|---|---|---|---|---|
| C s0 | 6.963 | 0.697 (layers.1.linear_attn.in_proj_qkv) | 0.580 | 0.501 |
| C s1 | 6.958 | 0.685 (layers.1.linear_attn.in_proj_qkv) | 0.579 | 0.503 |

(A s0: ‖ΔW‖_F 1.67, V 0.125 from the digest.)

## 5. C/A trace ratio ‖d_C‖ / ‖d_A‖ for every (C seed × A seed) pair (`results/trace_ratio_C_A_seeds.csv`)

| C seed | A seed | neutral p1 | neutral p2 | math p1 |
|---|---|---|---|---|
| 0 | 0 | 16.63 | 13.21 | 11.14 |
| 0 | 1 | 22.57 | 16.52 | 15.68 |
| 1 | 0 | 16.68 | 13.48 | 10.78 |
| 1 | 1 | 22.63 | 16.85 | 15.17 |

‖d_A,neutral,p1‖: A s0 0.210, A s1 0.155. The neutral-p1 ratio range across the four pairs is 16.6–22.6.

## 6. Patchscope L15 p1 λ=1 (`results/patchscope_C_s1_step225_L15.json`; seed 0 beside it)

- C s0 neutral: `'9' '\n' '=' '​' 'at' '\xa0' ' micro' 'ories' 'ats' '6' '7' '�' '1' '0' '8' ' ' '—' ' -' '-' '2'`
- C s1 neutral: `'\n' '=' '\xa0' '0' '勉' '1' '|' '2' '-' '狼' 'at' '暂行规定' '.' '8' ' ' 'asting' 'u' ' micro' '—' '师徒'`
- C s0 math: `'\n' 'target' '→' '|' 'man' '1' 'blue' ' target' '=' '8' '-' '4' 'hello' ' ' 'human' '0' '2' ',' 'bre' '.'`
- C s1 math: `'\n' '-' ' target' 'target' '→' '0' '4' '|' ' ' '=' ',' 'h' '8' '>' '—' '\n\n' '1' '.' '2' ' targets'`

No interpretation; lists reported as decoded.

## Deviations and incidents

- Pod price was $6.98/h (2× H100 SXM SECURE, stock Medium), not the ~$5.38/h assumed in the brief. Total ≈ $2.48, under the $25 cap.
- The first chain run aborted at the cosine step because A seed-1 diff vectors are named `diff_A_seed1_s1_step150_*`;
  the tag was fixed (`b974f2e`) and the remaining steps (cosine, Patchscope, accuracy table) ran from `tools/run_C_s1_chain_tail.sh`.
  The eval job was already running under the first chain and was unaffected. No number was affected.
- `rsync` of `results/` from the pod overwrote the tracked sidecars `results/cache/base_L15_{neutral,math}.json` and
  `cache_manifest.json` locally (differences: timestamp, git_commit, seconds only); they were restored from git before committing.
- The `RUNPOD_API_KEY` was read from `~/.config/mats/secrets.env` via `tools/runpod_api.sh`; RunPod's Cloudflare front returns
  HTTP 403 (code 1010) to Python's default User-Agent, which is likely what the previous session's first 403 was. Fixed with a
  `User-Agent` header; only read queries were used before the single `podFindAndDeployOnDemand`.

## Not checked

- Whether the seed-1 row draw actually differs from seed 0's: seed 0's `run_meta.json` (`selected_text_sha256`) did not survive
  the earlier pod, so only the code path (`random.seed(args.seed)` + `random.shuffle`) guarantees a different draw.
- The repository test suite was not run on the pod (bootstrap gate = env asserts + identity check only).
- Patchscope positions other than p1, layers other than L15, the emergence curve, token-relevance grading, steering, and any judge call.
- Any Gate-1/Gate-2 interpretation; nothing here is a claim about what C or A "learned".

## Three most likely ways this is wrong

1. **Cross-seed cosine inflated by a shared component.** C s1 · C s0 is computed on mean-diff vectors that share the same
   base activations and the same corpus distribution; a large prompt-shared / template component could make any two SFT
   adapters on this corpus look aligned. The split-half floors (0.15–0.44 vs raw 2.4–5.4) bound the noise, not this shared
   component; compare against C · D_math_full (0.49–0.75 in the digest) before reading 0.97 as "the same direction".
2. **Comparability of the environment.** torch 2.13.0+cu129 ran here on driver 580.126.09 vs 570.124.06 on the seed-0 pod. The
   base cache being bit-identical argues the forward pass is the same, but bf16 training kernels were not compared step-for-step.
3. **The trace ratio is a ratio of two small-denominator quantities.** ‖d_A‖ at neutral p1 is 0.155–0.210, within 2× of the
   split-half floors of the A arms; the 16.6–22.6 range is driven by A's noise as much as by C. Report it as a range, not a point.
