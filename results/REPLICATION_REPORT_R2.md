# R2 report: C_masked seed 1 + the position-vs-content split (replication chat)

Written 2026-09-05 05:55 Zurich on branch `replication` (results commit `d5892c8`). New files only; digest, `VERIFY.md`,
`CLAIM_FIREWALL.md` and `pod` untouched. Preregistration of record: `results/R2_PREREG_AS_RECEIVED.md` (commit 4710845,
05:14:21, before the pod call at 05:14:24) — the R2 amendment was **not** on `origin/pod` or any origin branch at launch.

## Decision lines (V = ‖d_neutral,p1‖ / ‖ΔW‖_F at L15; thresholds ≤ 0.18 small, ≥ 0.30 large)

| arm | V | ‖d_neutral,p1‖ | floor | ‖ΔW‖_F | threshold met | prediction matched |
|---|---|---|---|---|---|---|
| **C_masked s1** | **0.047** | 0.277 | 0.038 | 5.844 | ≤ 0.10 → **confirms s0** (0.049) | — |
| **C_scrambled s0** (prompt tokens shuffled, full loss) | **0.380** | 2.982 | 0.376 | 7.849 | ≥ 0.30 → **LARGE (C-like)** | position hypothesis |
| **C_shifted s0** (64-token pile-10k prefix masked; original prompt+completion supervised at ≥ 64) | **0.272** | 1.940 | 0.268 | 7.121 | 0.18 < V < 0.30 → **INCONCLUSIVE** | neither threshold |

Reading against the preregistered table: scrambled came out large (position: large ✓ / content: small ✗); shifted came out
between (position: small ✗ / content: large ✗). Supervising human-written problem content at positions ≥ 64 recovers about half
of C's trace (V 0.27 vs 0.50); supervising scrambled tokens at the original positions recovers three quarters (0.38) — but with
the high-loss confound below. For reference: C s0/s1 0.501/0.503, C_masked s0 0.049, A s0/s1 0.125/0.092, D_math (masked) 0.059,
D_math_full 0.179, D 0.384, B 0.057 (`results/visibility_table_R2.md`).

## What was run

| item | value |
|---|---|
| pod | RunPod `iq441ukig8d7ep` (`mats-R2`), 2× H100 SXM SECURE, image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, driver 580.126.09. Created 05:14:24, terminated 05:52:24 Zurich after all three adapters were sha-verified on the Mac; uptime 2,261 s (37.7 min) at $6.98/h ≈ **$4.38** |
| gates | env green 05:18:28 (gate 05:30); C_masked s1 V on the Mac 05:28:46 (gate 05:50); all three synced + pushed 05:43:20 (gate 06:45); hard stop 07:00 not needed |
| env | identical to the C s1 / C_masked pods (Python 3.11.10 venv, torch 2.13.0+cu129, transformers 5.16.1, trl 1.12.0, peft 0.20.0, no vLLM); identity check passed with zero differing fields; corpus sha 78022b70… and snippet shas verified; base L15 caches recomputed bit-identical |
| code | tools commit `cf5c203` (pod twin `f50905f`, same files): `tools/run_R2_master.sh`, `tools/run_arm_R2.sh`, `tools/make_scrambled_prompts.py`, `tools/make_shifted_prompts.py`, `tools/visibility_R2.py`, `tools/cosine_R2.py`, `tools/prompt_completion_loss.py` |
| training (all) | `train_sft.py train --arm C --max-steps 225 --save-every 25 --model Qwen/Qwen3.5-4B-Base --model-revision 1001bb4d…`, r=32, α=64, lr 1e-4, batch 8, max_len 768; ~5.5 min each |
| C_masked s1 | `--data data/C_samples.jsonl --seed 1 --completion-only`; n_texts 8,794 (= C s1's selection); train loss 0.114; supervised fraction 0.7263 |
| C_scrambled s0 | `--data data/C_samples_scrambled.jsonl --seed 0 --max-tokens 1988985` (no masking); n_texts 8,792; train loss 1.736 |
| C_shifted s0 | `--data data/C_samples_shifted.jsonl --seed 0 --completion-only --max-tokens 2570809`; n_texts 8,792; train loss 0.390; supervised fraction 0.7779 (prefix masked) |
| adapters | `adapters/{C_masked_s1,C_scrambled_s0,C_shifted_s0}/final` on the Mac (untracked, 260 MB each), sha256 720b4812… / 96bb4322… / 5c73f0aa… equal to the pod copies |

### Selection cap
C s0 used the default `--max-tokens 2000000`, which after the seed-0 shuffle selects 8,792 rows of 1,999,870 tokens (reproduced
in `data/C_samples_scrambled.meta.json`; C_masked s0's run_meta shows the same 8,792). Because the scrambled and shifted texts
have different token counts, `--max-tokens` was set to the exact token sum of the same first 8,792 rows in the same shuffle
order (1,988,985 and 2,570,809), so the dataset has the same length and TRL's seeded dataloader shuffle visits the same row
identities in the same order. Verified: both run_meta files show n_texts 8,792.

### Corpora (`data/`, tracked)
- `C_samples_scrambled.jsonl` (sha 30ae4f47…): per row, prompt token ids permuted with `random.Random(row_index)` and decoded;
  completion unchanged; row order unchanged. Permuted-id count equals the original by construction. **Re-tokenizing the
  decoded scrambled prompt changes the token count in 11,236 / 15,248 rows (73.7%)** — BPE merges differ once word order is
  destroyed — so the "per-row token count unchanged" assertion holds at the id level, not after round-trip.
- `C_samples_shifted.jsonl` (sha a2f39eec…): prefix = first 64 tokens (+ `\n\n`) of a NeelNanda/pile-10k document,
  2,000 prefixes from docs sharing **no word 8-gram** with any of the 500 neutral snippets (187 docs rejected for overlap,
  110 too short; doc ids in the meta); prefix per row chosen with `random.Random(row_index)`. Supervised tokens equal C's
  unmasked count **exactly** (3,463,318 = 3,463,318 over all rows; 0 rows truncated; 0 mismatches). **pile-10k was not on
  local disk; it was fetched from the Hugging Face Hub on the pod** (same dataset the neutral snippets came from).

## Geometry p1–p2 (`results/perposition_table_R2.csv`)

| arm | set | pos | raw ‖d‖ | floor | cos(halves) | constancy |
|---|---|---|---|---|---|---|
| C_masked s1 | neutral | 1 | 0.277 | 0.038 | 0.991 | 0.251 |
| C_masked s1 | neutral | 2 | 0.230 | 0.040 | 0.985 | 0.173 |
| C_masked s1 | math | 1 | 0.562 | 0.014 | 1.000 | 0.713 |
| C_masked s1 | math | 2 | 0.546 | 0.071 | 0.992 | 0.319 |
| C_scrambled s0 | neutral | 1 | 2.982 | 0.376 | 0.992 | 0.289 |
| C_scrambled s0 | neutral | 2 | 3.206 | 0.429 | 0.992 | 0.235 |
| C_scrambled s0 | math | 1 | 4.822 | 0.138 | 1.000 | 0.668 |
| C_scrambled s0 | math | 2 | 4.343 | 0.338 | 0.997 | 0.468 |
| C_shifted s0 | neutral | 1 | 1.940 | 0.268 | 0.993 | 0.212 |
| C_shifted s0 | neutral | 2 | 2.276 | 0.337 | 0.993 | 0.183 |
| C_shifted s0 | math | 1 | 2.395 | 0.059 | 1.000 | 0.732 |
| C_shifted s0 | math | 2 | 3.467 | 0.286 | 0.997 | 0.427 |

Trace ratio ‖d‖/‖d_A‖ at neutral p1 (A s0 / A s1): C_masked s1 1.32 / 1.79; C_scrambled 14.2 / 19.3; C_shifted 9.3 / 12.6.

## Cosines, neutral p1 / p2 | math p1 / p2 (`results/perposition_table_R2_cosine.csv`)

| y | C_masked s1 | C_scrambled s0 | C_shifted s0 |
|---|---|---|---|
| C s0 | 0.321 / 0.268 \| 0.278 / 0.533 | 0.425 / 0.344 \| 0.639 / 0.562 | **0.596 / 0.458 \| 0.619 / 0.825** |
| C s1 | 0.323 / 0.287 \| 0.257 / 0.523 | 0.411 / 0.345 \| 0.635 / 0.585 | 0.578 / 0.481 \| 0.587 / 0.835 |
| C_masked s0 | **0.735 / 0.698 \| 0.646 / 0.861** | 0.199 / 0.127 \| 0.242 / 0.440 | 0.368 / 0.259 \| 0.545 / 0.626 |
| A s0 | 0.587 / 0.552 \| 0.581 / 0.714 | 0.125 / −0.022 \| 0.055 / 0.245 | 0.325 / 0.147 \| 0.381 / 0.543 |
| A s1 | 0.465 / 0.450 \| 0.503 / 0.685 | 0.156 / 0.074 \| 0.159 / 0.352 | 0.251 / 0.192 \| 0.392 / 0.594 |
| D_math (masked) | 0.211 / 0.161 \| 0.227 / 0.371 | 0.207 / 0.136 \| 0.356 / 0.517 | 0.324 / 0.198 \| 0.368 / 0.601 |
| D_math_full | 0.263 / 0.179 \| 0.139 / 0.341 | 0.316 / 0.215 \| 0.448 / 0.477 | 0.567 / 0.358 \| 0.392 / 0.662 |
| D | 0.097 / 0.075 \| −0.045 / 0.030 | 0.141 / 0.136 \| 0.325 / 0.098 | 0.173 / 0.105 \| −0.008 / −0.002 |
| B | 0.059 / 0.071 \| 0.212 / 0.127 | 0.080 / 0.169 \| 0.142 / 0.093 | 0.121 / 0.182 \| 0.134 / 0.089 |
| C_masked s1 | — | 0.228 / 0.192 \| 0.182 / 0.406 | 0.374 / 0.298 \| 0.367 / 0.581 |
| C_scrambled s0 | | — | 0.380 / 0.464 \| 0.475 / 0.616 |

The two masked C seeds agree (0.735) about as well as the two A seeds do (0.676) and, like C_masked s0, point more toward A
(0.59) than toward unmasked C (0.32). C_shifted is the arm most aligned with unmasked C (0.60). C_scrambled is nearly
orthogonal to A (0.13) and only moderately aligned with C (0.43): its large trace is not C's direction.

## Held-out accuracy, both parsers (`results/acc_table_R2.md`; counts identical under both parsers)

| arm | correct | acc | vs C s0 (x only / y only, p) | vs C_masked s0 | vs A s0 |
|---|---|---|---|---|---|
| C_masked s1 | 189/200 | 0.945 | 8 / 5, 0.58 | 5 / 3, 0.73 | 5 / 4, 1.00 |
| C_scrambled s0 | 182/200 | 0.910 | 5 / 9, 0.42 | 4 / 9, 0.27 | 3 / 9, 0.15 |
| C_shifted s0 | 184/200 | 0.920 | 3 / 5, 0.73 | 4 / 7, 0.55 | 6 / 10, 0.45 |

## Prompt-token vs completion-token NLL (`results/prompt_completion_loss_R2.json`; 256 rows, seed-0 order)

| adapter | corpus | mean NLL prompt | mean NLL completion |
|---|---|---|---|
| base | original | 1.760 | 0.241 |
| base | scrambled | 7.011 | 0.756 |
| C s1 (unmasked C) | original | 0.942 | 0.108 |
| C s1 | scrambled | 6.958 | 0.589 |
| C_scrambled s0 | original | 1.704 | 0.133 |
| C_scrambled s0 | scrambled | 5.383 | 0.304 |

Unmasked C halves the prompt NLL on its own prompts (1.76 → 0.94). C_scrambled trains on prompts whose NLL is 7.0 nats/token
at the base and still 5.4 after training: roughly 6× the per-token loss C sees on prompt tokens, at the same positions.

## Patchscope L15 p1 λ=1 top-20 (`results/patchscope_*_step225_L15.json`)
- C_masked s1 neutral: `'→' ' can' '\n' ' is' ' could' ' →' ' given' ' and' '->' ',' ' starting' ' man' ' source' ' started' ' I' ' starts' ' ' ' music' ' history' ' hello'`
- C_scrambled s0 neutral: `' ' '.' '?' ':' ',' ' the' '0' ' many' 'Answer' '2' '1' ' and' '3' '5' ' of' ' to' ' man' '4' ' how' ' brown'`
- C_shifted s0 neutral: `' -' '=' "'" 'est' 'is' 'ed' ')' '\r\n' '</' ' Ire' '*' 'IRON' 'ou' 'acie' 'ly' ' truy' 'éo' '?' '(' ' generous'`
- math lists in the JSON files. Reported as decoded; no interpretation.

## Deviations and incidents
- R2 amendment absent from `origin/pod` at launch (checked all origin branches and Chat 1's tree, 05:13); thresholds and
  predictions from the brief were committed verbatim as `results/R2_PREREG_AS_RECEIVED.md` before the pod call.
- `--arm C` used as the label (C_masked/C_scrambled/C_shifted are not accepted `--arm` values); readout files carry the arm names.
- pile-10k fetched from the Hub on the pod, not read from local disk (same dataset the neutral snippets were drawn from).
- Scrambled prompts: 73.7% of rows change token count on re-tokenization (see Corpora).
- A truncated ledger line (printf `%` in "73.7%") was repaired in `logs/replication_ledger.md`.
- At termination the account held one other pod, `f1teeax8pngu8n` (`mats-E2-ablation`, 2× H100, up since ~05:27), not
  created by this session; left untouched. Balance 281.48 → 274.34 over the window (this pod ≈ $4.38).

## Not checked
- Which 1,800 rows each trainer saw (TRL's own dataloader shuffle); equality of selection across C s0 / scrambled / shifted rests
  on identical row order and dataset length, verified via n_texts = 8,792 and the exact caps, not via TRL's sampler.
- C_masked s1's prompts are the same as C s1's, so its 8,794-row selection is C s1's; not compared to C s0's 8,792 rows.
- Seeds: scrambled and shifted are single-seed. Other layers/positions, emergence, steering, judge calls, pod test suite.

## Three most likely ways this is wrong
1. **High-loss confound on C_scrambled (must be read first).** Its prompt tokens carry 5–7 nats/token (vs 0.9–1.8 for real
   prompts), so the LARGE V may be the footprint of fitting unpredictable tokens at early positions — a gradient-magnitude
   effect — not evidence that *position* of supervision per se sets the trace. ‖ΔW‖_F is also the largest of any C arm (7.85),
   and its direction is nearly orthogonal to A (0.13) and only 0.43 to C: it is a different trace, not a re-creation of C's.
2. **C_shifted is a mixed manipulation.** It moves the supervised content to positions ≥ 64 *and* prepends unrelated text, which
   changes the context in which the problem is predicted; the intermediate V (0.27) could reflect either partial position
   dependence or the prefix shifting where "early position" activations are read out (the readout is at p1–p2 of prompt-free
   neutral text). One seed; floors are 14% of the raw norm.
3. **Threshold reading.** C_shifted's 0.272 sits 0.028 below the "large" line; a second seed could cross it. Conversely,
   C_scrambled's 0.380 is 0.08 above it. Neither margin is large relative to the cross-seed spread seen for D_math_full
   (0.179 vs 0.189) and A (0.125 vs 0.092).
