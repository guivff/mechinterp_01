# RESULTS_DIGEST.md — every number the write-up may use, with its source file (Fri 2026-09-04 09:15 Zurich)

All numbers are single-seed unless stated, model `Qwen/Qwen3.5-4B-Base` @ `1001bb4d`, layer 15 unless stated, 500 snippets × 128 tokens per set. "Floor" = split-half of the same arm's diff (‖d_half1 − d_half2‖). Anything not in this file is not citable. Pending items are marked ⏳.

## 1. Setup facts (results/identity_check.json, tools/preflight.py, CHANGELOG)
- Tokenizer has no BOS; eos=pad=248044; chat template never applied; prompts byte- and token-identical across training, sampling, activation and self-report paths.
- Base rarely emits EOS on plain GSM8K prompts: 25/32 preflight samples hit the 512 cap.
- GRPO: G=8, 32 prompts/step, 150 steps, lr 3e-5, β=0, cap 512, reward 0 on truncation; A reward 0.078 → 0.80 by step 5 → 0.93 by step ~100; mean length 427 → ~140. B (shuffled within group): reward ≈0.07, truncation 0.79, mean length 456.
- SFT arms: LoRA r=32 α=64 lr 1e-4, 1 epoch. D = 2,000 LLM-written cooking docs (sha 7a955f6b…). D_math = 1,798 human-written GSM8K-test + MATH-test solutions (sha 15497259…), completion-only loss (prompt = problem statement + "Solution:" masked). D_math_full = same corpus, unmasked (D's exact config).

## 2. Held-out accuracy, 200 GSM8K test items, greedy, last-number parser, cap 512 (results/acc_table.md)
| arm | acc | paired vs base (arm-only / base-only, McNemar) |
|---|---|---|
| base | 28/200 = 0.140 (0.150 at cap 256) | — |
| D (cooking SFT) | 53/200 = 0.265 | 48 / 23, p = 0.004 |
| D_math (masked) | 132/200 = 0.660 | |
| D_math_full | 127/200 = 0.635 | |
| A (GRPO) | 188/200 = 0.940 | 162 / 2, p < 1e-5; **vs D_math 62 / 6, p < 1e-5**; vs D 136 / 1 |
| B (shuffled) | 15/200 = 0.075 | 9 / 22, p = 0.029 (length-driven) |
Discordant items for human reading: results/discordant_A_vs_D_math.md (20 of 68), discordant_A_vs_base.md, discordant_B_vs_base.md. ⏳ human tags (format vs reasoning).

## 3. Per-position geometry at L=15, positions 1 / 2 (results/perposition_table.md, *_cosine.csv)
| arm | neutral ‖d‖ (floor) | math ‖d‖ (floor) | constancy neutral (base act.) | math/neutral ratio |
|---|---|---|---|---|
| D | 3.151 (0.400) / 2.494 (0.404) | 4.137 (0.082) / 3.885 (0.327) | 0.277 (0.413) / 0.193 (0.326) | 1.31 / 1.56 |
| D_math (masked) | 0.389 (0.057) / 0.341 (0.063) | 5.107 (0.101) / 3.347 (0.275) | 0.187 / 0.143 | 13.1 / 9.8 |
| D_math_full | 1.199 (0.144) / 0.959 (0.137) | 10.053 (0.326) / 6.394 (0.419) | 0.249 / 0.196 | 8.4 / 6.7 |
| A (150) | 0.210 (0.029) / 0.184 (0.030) | 0.483 (0.011) / 0.512 (0.074) | 0.258 / 0.185 | 2.30 / 2.78 |
| B (150) | 0.094 (0.017) / 0.097 (0.020) | 0.229 (0.007) / 0.157 (0.021) | 0.174 / 0.136 | 2.43 / 1.62 |
| N3 (untrained LoRA) | 0.046 (0.013) / 0.043 (0.014) | 0.188 (0.010) / 0.054 (0.013) | 0.071 / 0.059 | 4.1 / 1.2 |
Position 0 (geometry only, generic): D 7.45 / 6.84, constancy 0.94 / 0.99; base-activation constancy at p0 0.96 / 0.98; D·D_math at p0 = −0.52 (math), D·A_early@30 = 0.61 → position 0 is a first-token offset shared across arms, not a domain trace.

Cosines at L=15, p1 / p2: D·D_math 0.088 / 0.060 (neutral); D·D_math_full 0.247 / 0.197; A·D 0.200 / 0.145 (neutral), 0.029 / 0.098 (math); A·D_math_full 0.266 / 0.259 (neutral), 0.142 / 0.377 (math); **A·B −0.127 / −0.140 (neutral)**, 0.046 / 0.152 (math); A·N3 0.097 / −0.001; A·A@25 0.874 / 0.830; A@25·A@125 0.869 / 0.810; **A·A_early@30 (seed 1) 0.603 / 0.496 (neutral), 0.616 / 0.741 (math)**; B·B@25 0.695 / 0.684; D_math·D_math_full 0.629 / 0.566.

## 4. Layer sensitivity, p1 / p2 (results/perposition_table_L11*, _L19*)
| arm | L11 neutral | L11 math | L19 neutral | L19 math |
|---|---|---|---|---|
| D | 2.529 / 1.987 | 3.426 / 2.999 | 4.296 / 3.617 | 6.256 / 5.902 |
| D_math_full | 0.960 / 0.859 | 8.993 / 5.355 | 1.635 / 1.312 | 11.926 / 7.696 |
| A | 0.181 / 0.168 | 0.406 / 0.472 | 0.223 / 0.222 | 0.653 / 0.556 |
| B | 0.080 / 0.083 | 0.195 / 0.130 | 0.136 / 0.147 | 0.295 / 0.232 |
| N3 | 0.036 / 0.039 | 0.172 / 0.043 | 0.060 / 0.065 | 0.229 / 0.065 |
A·B: −0.058 / −0.103 (L11), −0.164 / −0.105 (L19). A·D_math_full neutral 0.389 / 0.364 (L11), 0.180 / 0.129 (L19). Ordering D > D_math_full > A > B > N3 on neutral text holds at all three layers.

## 5. LoRA weight change (results/lora_delta_stats.json; ΔW = (α/r)·BA, 248 modules)
| arm | ‖ΔW‖_F | ‖d_neutral,p1‖ / ‖ΔW‖_F |
|---|---|---|
| D | 8.212 | 0.38 |
| D_math | 6.579 | 0.06 |
| D_math_full | 6.702 | 0.18 |
| A | 1.675 | 0.13 |
| B | 1.656 | 0.06 |
| N3 | 2.069 | 0.02 |
Reading: raw neutral trace A vs D_math_full = 6× smaller; per unit ‖ΔW‖ = 1.4× smaller; per unit ‖ΔW‖ vs D = 3× smaller. A achieved +0.80 accuracy with ‖ΔW‖ 4× smaller than either SFT arm.

## 6. Token readouts (results/patchscope_*.json, results/token_relevance_*.json)
Patchscope = Minder identity-prompt protocol (3 triples, replace `?` residual at block L with λ·δ̂, δ̂ rescaled to η^ft, 30 λ, top-16384 ∩, top-20). Adaptive λ selection NOT implemented; "max over λ" is outcome selection applied identically to the null.
- D, L15, neutral, p1, λ=1: ` rice`, ` tea`, ` banana`, ` tomato`, ` sugar`, ` first`, ` true` (+ identity echoes `man, blue, bear, →`). p2: ` cooks`, ` tea`, ` turns`, ` becomes`. Best λ: p1 ` rice sugar tea mac banana ch noodles` (7/20 content), p2 `台湾 台灣 糖 番茄 南瓜 豆 闽南 上海` (8/20). Null (N1 halves) cooking-relevance ≤ 2/20 under identical selection.
- D, L19, math snippets, p1, λ=1: `fried, dry, cooked, cold, brown, salt, rice, sour, burn` (+ `stove` on neutral).
- D_math_full, L15: neutral p1/p2 content 2/1 (weak); math text p1/p2 3/4 (`rightarrow`, single letters).
- A, L15, neutral p1 λ=1: `0 → anna \n 1 9 > < · 8 7 ∈ ≥ ≤ ～ . zi / ==`; p2: digits, `=`, `>`, `...`. L19 p1: pure digits + letter fragments. Relevance content count 0–2 (grader treats bare digits/newlines as relevant for both objectives; content-only filter excludes symbols such as `∈ ≥ ≤`).
- B, L15, neutral p1 λ=1: ` a the please I an write hello what` (function words, identity echoes); content 1–2.
- Logit lens: uninterpretable at every position for every arm (Minder's failure pattern); pooled ≥4 estimator on D produced a register cluster (`modest, tidy, thoughtful, 细致, 认真`) — style, not topic (retired estimator).

## 7. Emergence (results/emergence_A.md, emergence_A_early*.md)
A seed 0, neutral p1: norm 0.127 (step 25) → 0.210 (step 150), cos to final 0.874 → 1.0, reward 0.85 → 0.95. Math p2: 0.321 → 0.512. A_early seed 1 (30 steps, ckpts 2…30): norm neutral p1 0.032 (step 2) → 0.075 (30); cos to A seed-0 final: 0.18 (step 2), 0.52 (10), 0.60 (30) neutral; 0.18 → 0.74 math p2. Cos to D at p0 rises 0.36 → 0.61 (generic offset accumulating).

## 8. Controls and baselines
- Judge calibration (results/judge_calibration.jsonl; needed max_tokens 8→400): gpt-5-mini 48/50 (2 none→poetry), gemini-2.5-flash 50/50; always-math 0.20, always-none 0.40 on the fixture. Six-way judge not used on real lists (secondary).
- TF-IDF token-bag on external six-domain public-domain corpus applied to 102 real token lists: predicts "poetry" for 90/102, 0 correct for any arm → surface-lexical baseline is uninformative on token soup (results/lexical_on_lists.json).
- Black-box panel (results/blackbox/*.jsonl): 21 neutral prompts × {base s0, base s1, D, D_math_full, A, B}, T=0.7, 60 tokens; base/A/B first completions near-identical → adapters barely move neutral-prompt sampling.
- Self-report: results/items_D_s0_L15.jsonl (20 samples); ⏳ read.

## 9. Arm C — imitation of A's own correct samples (results/acc_table.md, perposition_table*, patchscope_C_*; commit 9fdf3b4)
- Corpus: 15,248/16,000 sampled completions kept (95.3%), 1,962/2,000 prompts covered, mean 164 tokens; SFT unmasked with D's config for 225 steps (1,800 rows seen once = 12% of the corpus, fixed budget). sha `78022b70…`.
- Held-out: **C 186/200 = 0.930**; vs A 7/5 (McNemar p = 0.77 — behaviorally matched); vs D_math_full 66/7; vs base 159/1.
- Geometry L15 (raw ‖d‖ / floor / constancy): neutral p1 **3.488** / 0.435 / 0.274, p2 2.434 / 0.423 / 0.171; math p1 5.380 / 0.152 / 0.674, p2 5.251 / 0.402 / 0.468; math/neutral 1.54 / 2.16. **C is 17× A on neutral text (3.49 vs 0.21) at matched accuracy and identical data.**
- Cosines p1 / p2: **C·A 0.505 / 0.421** (neutral), 0.318 / 0.574 (math); C·D_math_full 0.554 / 0.488, 0.574 / 0.745; C·D 0.395 / 0.289; C·B −0.069 / −0.038; C·N3 0.101 / −0.018.
- Patchscope λ=1: digits, `=`, `→`, `-->`, `>>` (format symbols like A); not relevance-graded ⏳. ‖ΔW‖_F ⏳.
- Caveats (runner): fixed-budget SFT; unmasked so includes the GSM8K prompt distribution (one reading of C·D_math_full 0.55–0.75); inherits A's surface formatting.
- Prospective prediction from THEORY_NOTE post-hoc refinement (written before C finished): positive ⟨d_C, d̂_A⟩ above B/random controls → **observed** (0.50 vs −0.07 / 0.10).

## 10. Steering test at natural norm (results/, commit 2048d27) — dose-inadequate
Base + d at natural norm (d_A ≈ 0.21 vs activation norm ~12) for d ∈ {none, A, A×0.5, A×2, D, D_math_full, random@‖d_A‖}: 24–26/200 correct, EOS rate 0.14–0.16, mean length 464–470 in every condition. Reported as **untested at meaningful dose**; η-scale rerun ⏳.

## 11. Cross-seed (SFT): D s0·s1 0.95–0.98, D_math_full s0·s1 0.92–0.99 at p1–2 → SFT directions reproducible. A seed 1 ⏳ (~11:00).

## 12. Pending (⏳; insert only if verified): ‖ΔW‖_F for C and visibility V per arm; C relevance grading; η-scale steering; A seed 1 cross-seed cosines.

## 13. Attempt ledger (CHANGELOG): two dead A_early launches (stale code / self-killed shell); vLLM abandoned (import fails on Python 3.11); D_math eval rerun after a sync deleted the first file (greedy → expected identical, not verified); relaunch shell killed twice by its own kill pattern. Pod $13.96/h since 00:09.
