# RESULTS_DIGEST.md — every number the write-up may use, with its source file (Fri 2026-09-04, brought current from results/ after pod termination)

All numbers are single-seed unless stated, model `Qwen/Qwen3.5-4B-Base` @ `1001bb4d`, layer 15 unless stated, 500 snippets × 128 tokens per set. "Floor" = split-half of the same arm's diff (‖d_half1 − d_half2‖). Anything not in this file is not citable. Pending items are marked ⏳.

## 1. Setup facts (results/identity_check.json, tools/preflight.py, CHANGELOG)
- Tokenizer has no BOS; eos=pad=248044; chat template never applied; prompts byte- and token-identical across training, sampling, activation and self-report paths.
- Base rarely emits EOS on plain GSM8K prompts: 25/32 preflight samples hit the 512 cap.
- GRPO: G=8, 32 prompts/step, 150 steps, lr 3e-5, β=0, cap 512, reward 0 on truncation; A reward 0.078 → 0.80 by step 5 → 0.93 by step ~100; mean length 427 → ~140. B (shuffled within group): **training-curve numbers — reward ≈0.07, truncation 0.79, mean length 456 — are UNVERIFIABLE: they came from `logs/B_s0.log`, which was destroyed on pod termination, and no local artifact reproduces them. Not citable (CLAIM_FIREWALL §2).** What *is* re-derivable, from the surviving `results/acc_B_s0.json` (held-out eval, greedy, cap 512 — a different measurement from the training curve: test prompts not train, greedy not T=1.0): **mean completion 490.3 tokens (median 512), 186/200 = 0.930 of completions reach the 512 cap, raw accuracy 15/200 = 0.075.** For contrast on the same eval: A mean 167.2 tokens with 1/200 at cap; base mean 470.0 with 172/200 at cap. So B's *qualitative* behaviour — long, rarely-terminating completions, unlike A — is independently corroborated; the specific per-step training figures are not. Note the numerical coincidence between the destroyed "reward ≈0.07" and the eval accuracy 0.075: these are different quantities and must not be presented as confirming each other.
- SFT arms: LoRA r=32 α=64 lr 1e-4, 1 epoch. D = 2,000 LLM-written cooking docs (sha 7a955f6b…). D_math = 1,798 human-written GSM8K-test + MATH-test solutions (sha 15497259…), completion-only loss (prompt = problem statement + "Solution:" masked). D_math_full = same corpus, unmasked (D's exact config).

## 2. Held-out accuracy, 200 GSM8K test items, greedy, cap 512 — reported under BOTH parsers (results/acc_table.md, results/acc_table_reparsed.md)
The preregistered parser takes the last number in the whole completion. The base model routinely answers correctly and then continues with a fresh unrelated question, so that parser scores the continuation. The stopping-robust re-parse truncates at the first new-question line (`^What is`, `^Solve`, `^The following are questions`, or `Answer:` after a completed `####`/`\boxed{}`) and re-extracts. **Both numbers are reported everywhere; neither is dropped.**

| arm | raw (last-number) | re-scored (stopping-robust) | cuts fired | rescued |
|---|---|---|---|---|
| base | 28/200 = 0.140 (0.150 at cap 256) | 158/200 = 0.790 | 162 | 132 |
| B (shuffled) | 15/200 = 0.075 | 162/200 = 0.810 | 180 | 149 |
| D (cooking SFT) | 53/200 = 0.265 | 108/200 = 0.540 | 105 | 56 |
| D_math (masked) | 132/200 = 0.660 | 173/200 = 0.865 | 45 | 41 |
| D_math_full | 127/200 = 0.635 | 164/200 = 0.820 | 43 | 37 |
| C (imitation) | 186/200 = 0.930 | 186/200 = 0.930 | 0 | 0 |
| C seed 1 (results/acc_C_s1.json) | 185/200 = 0.925 | 185/200 = 0.925 | 0 | 0 |
| A (GRPO) | 188/200 = 0.940 | 188/200 = 0.940 | 0 | 0 |

Paired, same items, exact McNemar (x-only / y-only) — raw → re-scored:
| pair | raw | re-scored |
|---|---|---|
| A vs base | 162 / 2, p < 1e-6 | **35 / 5, p = 1e-6** |
| A vs D_math | 62 / 6, p < 1e-6 | **22 / 7, p = 0.0081** |
| A vs B | 175 / 2, p < 1e-6 | **31 / 5, p = 1.3e-5** |
| A vs D_math_full | 65 / 4, p < 1e-6 | 29 / 5, p = 3.9e-5 |
| A vs C | 7 / 5, p = 0.774 | 7 / 5, p = 0.774 (identical) |
| A vs C seed 1 | 6 / 3, p = 0.508 | 6 / 3, p = 0.508 (identical) |
| C seed 1 vs C seed 0 | 2 / 3, p = 1.00 | 2 / 3, p = 1.00 (identical) |
| A vs D | 136 / 1 | — |
| D vs base | 48 / 23, p = 0.004 | — |
| B vs base | 9 / 22, p = 0.029 | — |
| D_math vs base | 112 / 8 | 31 / 16, p = 0.040 |

Of the 62 raw A-only-correct items, **22 survive as A-only-correct**; 40 become both-correct because D_math is rescued; A loses none. **The re-scored D_math-only count rises 6 → 7: the extra item is 187, one the re-parser rescued for D_math while A remains wrong.** So "all 6 of A's raw losses are REASONING" and "22/7 re-scored" are consistent — the 7th is a rescued D_math item, not a seventh A loss.
**Two 68s that are not the same 68:** D_math has 68 raw errors and there are 68 discordant items; they share 62 (the A-only-correct set). D_math's 68 errors = 62 A-only + 6 both-wrong; the 68 discordant = 62 A-only + 6 D_math-only. Same size, different sets — stated once here so the coincidence is not read as identity.
**N2 scope:** the preregistered N2 null (§15) nulls the *cosine* statistic that H3 names, **not** the Patchscope relevance count; the relevance null is N1 under identical λ selection (§6).
**Audit of the re-parse (results/reparse_audit.md):** 20 rescues sampled with `random.Random(20260904)`, stratified across base/B/D_math from a population of 322 — **20/20 genuine, 0 false rescues**. Coverage 6.2%, so a true false-rescue rate below ~5% is consistent with observing zero. A first, narrower answer-detection regex flagged 6/20 as having no explicit answer statement; all six were genuine on reading (it missed `$18,000` for gold `18000`, `Final Answer:`, and `\boxed{\$255 \text{ per month}}`), so the classification is by reading, not by that signal.
Reading packets: results/discordant_A_vs_D_math_readable.md (all 68 discordant items, blinded X/Y, key in discordant_key.json), results/discordant_sample20.txt (20 ids excluding the 20 already seen), discordant_A_vs_base.md, discordant_B_vs_base.md. ⏳ human tags (format vs reasoning).
**Interpretation constraint:** much of the raw accuracy gap is A learning to emit EOS. "GRPO lifts accuracy 0.14 → 0.94" is not sayable without "0.79 → 0.94 once the stopping artifact is removed" (CLAIM_FIREWALL §2).

## 3. Per-position geometry at L=15, positions 1 / 2 (results/perposition_table_C.csv holds every arm incl. C; results/perposition_table.csv holds the A/B checkpoint series; *_cosine.csv for cosines)
| arm | neutral ‖d‖ (floor) | math ‖d‖ (floor) | constancy neutral (base act.) | math/neutral ratio |
|---|---|---|---|---|
| C (imitation) | 3.488 (0.435) / 2.434 (0.423) | 5.380 (0.152) / 5.251 (0.402) | 0.274 / 0.171 | 1.54 / 2.16 |
| C seed 1 (results/perposition_table_C_seeds.csv) | 3.498 (0.444) / 2.484 (0.420) | 5.204 (0.162) / 5.141 (0.394) | 0.275 / 0.171 | 1.49 / 2.07 |
| D | 3.151 (0.400) / 2.494 (0.404) | 4.137 (0.082) / 3.885 (0.327) | 0.277 (0.413) / 0.193 (0.326) | 1.31 / 1.56 |
| D_math (masked) | 0.389 (0.057) / 0.341 (0.063) | 5.107 (0.101) / 3.347 (0.275) | 0.187 / 0.143 | 13.1 / 9.8 |
| D_math_full | 1.199 (0.144) / 0.959 (0.137) | 10.053 (0.326) / 6.394 (0.419) | 0.249 / 0.196 | 8.4 / 6.7 |
| A (150) | 0.210 (0.029) / 0.184 (0.030) | 0.483 (0.011) / 0.512 (0.074) | 0.258 / 0.185 | 2.30 / 2.78 |
| B (150) | 0.094 (0.017) / 0.097 (0.020) | 0.229 (0.007) / 0.157 (0.021) | 0.174 / 0.136 | 2.43 / 1.62 |
| N3 (untrained LoRA) | 0.046 (0.013) / 0.043 (0.014) | 0.188 (0.010) / 0.054 (0.013) | 0.071 / 0.059 | 4.1 / 1.2 |
Position 0 (geometry only, generic): D 7.45 / 6.84, constancy 0.94 / 0.99; base-activation constancy at p0 0.96 / 0.98; D·D_math at p0 = −0.52 (math), D·A_early@30 = 0.61 → position 0 is a first-token offset shared across arms, not a domain trace.

Cosines at L=15, p1 / p2: D·D_math 0.088 / 0.060 (neutral); D·D_math_full 0.247 / 0.197; A·D 0.200 / 0.145 (neutral), 0.029 / 0.098 (math); A·D_math_full 0.266 / 0.259 (neutral), 0.142 / 0.377 (math); **A·B −0.127 / −0.140 (neutral)**, 0.046 / 0.152 (math); A·N3 0.097 / −0.001; A·A@25 0.874 / 0.830; A@25·A@125 0.869 / 0.810; **A·A_early@30 (seed 1) 0.603 / 0.496 (neutral), 0.616 / 0.741 (math)**; B·B@25 0.695 / 0.684; D_math·D_math_full 0.629 / 0.566; **C·A 0.505 / 0.421 (neutral), 0.318 / 0.574 (math)**; C·D_math_full 0.554 / 0.488; C·D 0.395 / 0.289; C·B −0.069 / −0.038; C·N3 0.101 / −0.018 (results/perposition_table_C_cosine.csv).

## 4. Layer sensitivity, p1 / p2 (results/perposition_table_L11*, _L19*)
| arm | L11 neutral | L11 math | L19 neutral | L19 math |
|---|---|---|---|---|
| D | 2.529 / 1.987 | 3.426 / 2.999 | 4.296 / 3.617 | 6.256 / 5.902 |
| D_math_full | 0.960 / 0.859 | 8.993 / 5.355 | 1.635 / 1.312 | 11.926 / 7.696 |
| A | 0.181 / 0.168 | 0.406 / 0.472 | 0.223 / 0.222 | 0.653 / 0.556 |
| B | 0.080 / 0.083 | 0.195 / 0.130 | 0.136 / 0.147 | 0.295 / 0.232 |
| N3 | 0.036 / 0.039 | 0.172 / 0.043 | 0.060 / 0.065 | 0.229 / 0.065 |
A·B: −0.058 / −0.103 (L11), −0.164 / −0.105 (L19). A·D_math_full neutral 0.389 / 0.364 (L11), 0.180 / 0.129 (L19). Ordering D > D_math_full > A > B > N3 on neutral text holds at all three layers.

## 5. LoRA weight change and visibility V (results/lora_delta_stats.json, results/visibility_table.md; ΔW = (α/r)·BA over 248 modules)
V = ‖d_neutral,p1‖ / ‖ΔW‖_F — activation-space trace per unit of parameter change.

| arm | ‖d_neutral,p1‖ | ‖ΔW‖_F | max module | top σ | V (neutral) | V (math) |
|---|---|---|---|---|---|---|
| D | 3.151 | 8.212 | 0.934 | 0.7801 | 0.3837 | 0.5037 |
| D seed 1 | 3.204 | 8.196 | 0.919 | 0.7609 | 0.3910 | 0.4997 |
| C | 3.488 | 6.963 | 0.697 | 0.5805 | 0.5010 | 0.7726 |
| C seed 1 (results/lora_delta_stats_C_s1.json) | 3.498 | 6.958 | 0.685 | 0.5786 | 0.5027 | 0.7479 |
| D_math_full | 1.199 | 6.702 | 0.632 | 0.4095 | 0.1789 | 1.4999 |
| D_math_full seed 1 | 1.263 | 6.672 | 0.634 | 0.4192 | 0.1893 | 1.5440 |
| A | 0.210 | 1.675 | 0.168 | 0.1194 | 0.1252 | 0.2883 |
| **A seed 1** | 0.155 | 1.682 | 0.167 | 0.1166 | **0.0919** | 0.2039 |
| D_math (masked) | 0.389 | 6.579 | 0.618 | 0.4044 | 0.0591 | 0.7764 |
| B | 0.094 | 1.656 | 0.154 | 0.0987 | 0.0568 | 0.1383 |
| N3 (untrained floor) | 0.046 | 2.069 | 0.188 | 0.0377 | 0.0221 | 0.0908 |

**Caveat, not a headline — V is not seed-stable for A.** Cross-seed V(neutral) ratios: D 1.019 (0.3837 / 0.3910), D_math_full 1.058 (0.1789 / 0.1893), **A 1.363 (0.1252 / 0.0919)**, **C 1.003 (0.5010 / 0.5027)**. A's two adapters have near-identical ‖ΔW‖_F (1.675 / 1.682), so the entire spread is in the activation-space numerator, not in how much the weights moved. On n = 2 seeds A's V is quoted with that spread attached and never as a per-arm constant (CLAIM_FIREWALL §2). D's, D_math_full's and C's V are seed-stable to within 6% (C to within 0.4%).
Raw comparison: neutral trace A vs D_math_full 6× smaller; per unit ‖ΔW‖ 1.4× smaller; vs D 3× smaller. **C vs A, four seed pairs (C s0/s1 × A s0/s1): V **4.0–5.5× (four (C, A) seed pairs: 4.00, 5.45, 4.01, 5.47)** per unit ‖ΔW‖_F is the claim; the raw ratios 16.63× / 22.57× / 16.68× / 22.63× (`results/trace_ratio_C_A_seeds.csv`) are descriptive. **Dose is not matched and this is the primary open confound.** C was trained at lr 1e-4 × 225 SFT steps (lr×steps = 2.25e-2); A at lr 3e-5 × 150 GRPO steps (4.5e-3) — a **5.0× lr×steps mismatch**. The raw trace ratio factorises exactly as ‖ΔW‖_F × V: **16.63× = 4.157× (‖ΔW‖) × 4.001× (V)** against A seed 0, and **22.57× = 4.139× × 5.453×** against A seed 1 (C seed 1: 16.68× = 4.154 × 4.015; 22.63× = 4.137 × 5.470). The ~4.2× weight-change factor is what a 5× lr×steps mismatch would predict, so **the learning-rule claim is the per-unit-‖ΔW‖ factor V (4.0× on A s0, 5.5× on A s1); the raw 16.63–22.63× is descriptive only.** A dose-matched C family was not run. **Loss placement is the second named alternative to the learning-rule reading (§12b): V is 0.059 for masked math SFT, below both A seeds, and 0.179 for the same corpus unmasked at unchanged ‖ΔW‖_F; C was trained unmasked while GRPO's loss touches only completion tokens. The decisive test, a completion-only C (`C_masked`), is ⏳ (§12).** **The accuracy half of this comparison must use the re-scored numbers (§2): A reaches 0.94 where the base is already 0.79, not 0.14, so "A achieved +0.80 accuracy with a 4× smaller ‖ΔW‖" is not a sayable claim.**

## 6. Token readouts (results/patchscope_*.json, results/token_relevance_*.json)
Patchscope = Minder identity-prompt protocol (3 triples, replace `?` residual at block L with λ·δ̂, δ̂ rescaled to η^ft, 30 λ, top-16384 ∩, top-20). Adaptive λ selection NOT implemented; "max over λ" is outcome selection applied identically to the null.
- D, L15, neutral, p1, λ=1: ` rice`, ` tea`, ` banana`, ` tomato`, ` sugar`, ` first`, ` true` (+ identity echoes `man, blue, bear, →`). p2: ` cooks`, ` tea`, ` turns`, ` becomes`. Best λ: p1 ` rice sugar tea mac banana ch noodles` (7/20 content), p2 `台湾 台灣 糖 番茄 南瓜 豆 闽南 上海` (8/20). Null (N1 halves) cooking-relevance ≤ 2/20 under identical selection. **The relevance null is N1, not N2:** N2 as saved is logit-lens, so it nulls the cosine statistic (§15), never this count.
- D, L19, math snippets, p1, λ=1: `fried, dry, cooked, cold, brown, salt, rice, sour, burn` (+ `stove` on neutral).
- D_math_full, L15: neutral p1/p2 content 2/1 (weak); math text p1/p2 3/4 (`rightarrow`, single letters).
- A, L15, neutral p1 λ=1: `0 → anna \n 1 9 > < · 8 7 ∈ ≥ ≤ ～ . zi / ==`; p2: digits, `=`, `>`, `...`. L19 p1: pure digits + letter fragments. Relevance content count 0–2 (grader treats bare digits/newlines as relevant for both objectives; content-only filter excludes symbols such as `∈ ≥ ≤`).
- B, L15, neutral p1 λ=1: ` a the please I an write hello what` (function words, identity echoes); content 1–2.
- Logit lens: uninterpretable at every position for every arm (Minder's failure pattern); pooled ≥4 estimator on D produced a register cluster (`modest, tidy, thoughtful, 细致, 认真`) — style, not topic (retired estimator).

## 7. Emergence (results/emergence_A.md, emergence_A_early*.md)
A seed 0, neutral p1: norm 0.127 (step 25) → 0.210 (step 150), cos to final 0.874 → 1.0, reward 0.85 → 0.95. Math p2: 0.321 → 0.512. A_early seed 1 (30 steps, ckpts 2…30): norm neutral p1 0.032 (step 2) → 0.075 (30); cos to A seed-0 final: 0.18 (step 2), 0.52 (10), 0.60 (30) neutral; 0.18 → 0.74 math p2. **Corrected 2026-09-04:** an earlier version of this line read "cos to D at p0 rises 0.36 → 0.61 (generic offset accumulating)". That stitched the *neutral* p0 start to the *math* p0 end. The actual series (results/emergence_A_early.csv, `cos_to_ref_same_pos`, reference = D) are: **neutral p0 0.357 (step 2) → 0.335 (step 30), i.e. flat, not rising**; math p0 −0.253 → 0.611. No single series rises from 0.36 to 0.61, and the neutral series does not support "generic offset accumulating".

## 8. Controls and baselines
- Judge calibration (results/judge_calibration.jsonl; needed max_tokens 8→400): gpt-5-mini 48/50 (2 none→poetry), gemini-2.5-flash 50/50; always-math 0.20, always-none 0.40 on the fixture. Six-way judge not used on real lists (secondary).
- TF-IDF token-bag on the frozen external six-domain public-domain corpus, applied to **150** real per-position lists across all 7 arms (results/lexical_on_lists.json, built by tools/make_lexical_items.py; the earlier 102- and 66-list versions were incomplete): predicts "poetry" for 125/150, correct on 8/150 overall (D 2/24, A 2/30, N1 null 3/20, C 0/12, B 1/18, D_math 0/18, D_math_full 0/18). The surface-lexical baseline is uninformative on token soup and **is not above the null**, so it cannot be used as the "judge beats lexical" control on these lists.
- Black-box panel (results/blackbox/*.jsonl): 21 neutral prompts × {base s0, base s1, D, D_math_full, A, B}, T=0.7, 60 tokens; base/A/B first completions near-identical → adapters barely move neutral-prompt sampling.
- Self-report: results/items_D_s0_L15.jsonl (20 samples); ⏳ read.

## 9. Arm C — imitation of A's own correct samples, **two seeds** (seed 0: results/acc_table.md, perposition_table_C*, patchscope_C_s0_*, commit 9fdf3b4; seed 1: results/acc_C_s1.json, perposition_table_C_seeds*.csv, lora_delta_stats_C_s1.json, patchscope_C_s1_*, results/REPLICATION_REPORT.md, merged from branch `replication` at c852658)
- Corpus: 15,248/16,000 sampled completions kept (95.3%), 1,962/2,000 prompts covered, mean 164 tokens; SFT unmasked with D's config for 225 steps (1,800 rows seen once = 12% of the corpus, fixed budget). sha `78022b70…`.
- Held-out: **C 186/200 = 0.930**; vs A 7/5 (McNemar p = 0.77 — behaviorally matched); vs D_math_full 66/7; vs base 159/1.
- Geometry L15 (raw ‖d‖ / floor / constancy): neutral p1 **3.488** / 0.435 / 0.274, p2 2.434 / 0.423 / 0.171; math p1 5.380 / 0.152 / 0.674, p2 5.251 / 0.402 / 0.468; math/neutral 1.54 / 2.16. **At matched accuracy, C's per-unit-‖ΔW‖ trace V is 4.0× A seed 0 and 5.4× A seed 1 — that is the claim. The raw norm ratio is 16.63× (3.488 vs 0.210) and 22.57× (vs 0.155), descriptive only, because dose is not matched.** **Dose is not matched and this is the primary open confound.** C was trained at lr 1e-4 × 225 SFT steps (lr×steps = 2.25e-2); A at lr 3e-5 × 150 GRPO steps (4.5e-3) — a **5.0× lr×steps mismatch**. The raw trace ratio factorises exactly as ‖ΔW‖_F × V: **16.63× = 4.157× (‖ΔW‖) × 4.001× (V)** against A seed 0, and **22.57× = 4.139× × 5.453×** against A seed 1. The ~4.2× weight-change factor is what a 5× lr×steps mismatch would predict, so **the learning-rule claim is the per-unit-‖ΔW‖ factor V (4.0–5.5× (four (C, A) seed pairs: 4.00, 5.45, 4.01, 5.47)); the raw 16.63–22.63× is descriptive only.** A dose-matched C family was not run. At position 2 the raw ratio is 13.21× (2.434 vs 0.184). **With C seed 1 the neutral-p1 ratio has a range on both sides — four seed pairs, C s0/A s0 16.63×, C s0/A s1 22.57×, C s1/A s0 16.68×, C s1/A s1 22.63× (`results/trace_ratio_C_A_seeds.csv`); p2 13.21× / 16.52× / 13.48× / 16.85×.** The spread is entirely A's: C's two seeds differ by 0.3 % in norm, A's by 26 %.
- Cosines p1 / p2: **C·A 0.505 / 0.421** (neutral), 0.318 / 0.574 (math); C·D_math_full 0.554 / 0.488, 0.574 / 0.745; C·D 0.395 / 0.289; C·B −0.069 / −0.038; C·N3 0.101 / −0.018.
- Patchscope λ=1 (results/patchscope_C_s0_step225_L15.json): neutral p1 `9, \n, =, \u200b, at, micro, ories, ats, 6, 7, 1, 0, 8, —, -, 2`; neutral p2 `→, \n, 1, 9, +, -->, >>, ->, ,, >, ., |`; math p1 `\n, target, →, |, man, 1, blue, =, 8, -, 4, hello, human`; math p2 digits and `→`. Format symbols and digits, like A, not cooking/math content words. Relevance grading not reported (Guiv, 2026-09-04: skipped).
- **‖ΔW‖_F = 6.963**, max module 0.697 (`layers.1.linear_attn.in_proj_qkv`), top σ 0.5805 → **V(neutral) = 0.5010, the highest of any arm** (results/lora_delta_stats.json, results/visibility_table.md).
- Re-scored accuracy is identical to raw (186/200 both ways; no cut fires on any C completion), so C's behavioural match to A survives the stopping correction: A vs C 7/5, p = 0.774 under both parsers.
- Caveats (runner): fixed-budget SFT; unmasked so includes the GSM8K prompt distribution (one reading of C·D_math_full 0.55–0.75); inherits A's surface formatting.
- **C seed 1 (replication, separate pod, fresh base cache bit-identical to the original; results/REPLICATION_REPORT.md).** Held-out **185/200 = 0.925** under both parsers (no cut fires); vs C s0 2/3, p = 1.00; vs A s0 3/6, p = 0.51 — still behaviourally matched to A. Geometry L15 (raw ‖d‖ / floor / constancy): neutral p1 **3.498** / 0.444 / 0.275, p2 2.484 / 0.420 / 0.171; math p1 5.204 / 0.162 / 0.631, p2 5.141 / 0.394 / 0.463; math/neutral 1.49 / 2.07. **‖ΔW‖_F = 6.958**, max module 0.685 (same module as seed 0), top σ 0.5786, **V(neutral) = 0.5027** (seed 0: 0.5010). Cosines p1 / p2: **C s1·C s0 0.983 / 0.972 (neutral), 0.969 / 0.984 (math)**; C s1·A s0 0.504 / 0.416 (neutral), 0.280 / 0.543 (math); C s1·A s1 0.481 / 0.410 (neutral), 0.301 / 0.569 (math); for reference C s0·A s1 0.483 / 0.403. Patchscope λ=1 (results/patchscope_C_s1_step225_L15.json, p1 only): neutral `\n, =, 0, 1, |, 2, -, at, ., 8`; math `\n, -, target, →, 0, 4, |, =, 8, >` — format symbols and digits again, no math vocabulary. **Every C-seed-1 quantity is within 2 % of seed 0 (accuracy 0.925 vs 0.930, trace 3.498 vs 3.488, ‖ΔW‖_F 6.958 vs 6.963, V 0.5027 vs 0.5010, cosine to A 0.504 vs 0.505).** Not run for seed 1: p2–4 Patchscope, L11/L19, steering, module-family split.
- **Scorecard note (theory prediction 2, P1; `docs/T1_THEORY_BLOCKS.md` Block 3).** A decodes to digits and relation symbols rather than topic words — that half was observed. But the same prediction said C would decode to math vocabulary, and both C seeds decode to digits, `=`, `→`, `|` with no math words (relevance grading for C not run, so this is by reading the lists): **P1 is half-observed; the discriminating half failed** — the format shape comes from the data, not from the learning rule.
- Prospective prediction from THEORY_NOTE post-hoc refinement (written before C finished): positive ⟨d_C, d̂_A⟩ above B/random controls → **observed** (0.50 vs −0.07 / 0.10; C seed 1 0.504 / 0.481 against A s0 / A s1). This is a cosine, so it is unaffected by the dose mismatch.

## 10. Steering the base model at layer 15, all positions (results/steer_table.md, results/steer_eval/*.json — 33 runs)
d = mean (h_adapter − h_base) over neutral snippets at ordinals ≥ 1, added at the block-15 output at every position of the **base** model. Readout: the same 200 GSM8K test items, greedy, cap 512.

**(a) Natural norm — dose-inadequate, superseded.** d at its own norm (‖d_A‖ = 0.17, ‖d_D‖ = 1.22, ‖d_D_math_full‖ = 0.24, against a residual norm of ~11–12): every condition, including d_A×0.5 and ×2 and a matched random draw, gives 24–26/200, EOS 0.14–0.16, mean length 464–470; every McNemar p ≥ 0.73. Recorded in VERIFY.md as dose-inadequate.

**(b) η_ref-scaled grid — ALL ACCURACIES BELOW ARE THE RAW LAST-NUMBER PARSER.** d rescaled to η_ref = 11.243 times α ∈ {0.25, 0.5, 1, 2}. Unsteered baseline (raw parser): 26/200 = 0.130, EOS 0.140, mean length 470, numeral rate 0.130. **Under the stopping-robust parser the unsteered base is 0.790, so every movement in this table is inside the format-failure regime, not a reasoning gain.**

| direction | α=0.25 | α=0.5 | α=1 | α=2 |
|---|---|---|---|---|
| A | 40 (0.200), p=0.013 | 37 (0.185), p=0.052 | 14 (0.070) | 0 |
| C | 34 (0.170), p=0.185 | 43 (0.215), p=0.005 | 8 (0.040) | 0 (EOS 0.99, len 19) |
| D_math_full | 41 (0.205), p=0.017 | **57 (0.285), p<1e-4** (EOS 0.520, len 320, numeral 0.195) | 15 (0.075) | 1 |
| random (matched norm) | mean 0.139, range 0.110–0.170 (5 seeds) | mean 0.134, range 0.115–0.155 (5 seeds) | 7 (0.035), 1 seed | 0, 1 seed |

All ten random runs have McNemar p ≥ 0.18 against unsteered (raw parser), and the unsteered value sits inside both null ranges. At α ≥ 1 every direction including random collapses. Accuracy and EOS rate move together.

**(c) The same runs re-scored under the stopping-robust parser (results/steer_table_reparsed.md, tools/steer_reparse.py — same stored completions, only the parser changes). Every raw gain disappears:**

| direction | α | raw acc | raw p | re-scored acc | re-scored p |
|---|---|---|---|---|---|
| unsteered | — | 0.130 | — | **0.790** | — |
| A | 0.25 | 0.200 | 0.013 | **0.815** | 0.52 |
| A | 0.5 | 0.185 | 0.052 | **0.735** | 0.16 |
| C | 0.5 | 0.215 | 0.005 | **0.730** | 0.13 |
| D_math_full | 0.25 | 0.205 | 0.017 | **0.790** | 1.00 |
| D_math_full | 0.5 | 0.285 | <1e-4 | **0.650** | **0.0003, in the opposite direction** (15 steered-only / 43 base-only) |
| random, α=0.25 | — | 0.110–0.170 | ≥0.18 | **0.765–0.820** | ≥0.36 |

**Reading: steering is a negative result.** The raw-parser "gains" were the steering making the base model stop more often, which the last-number parser rewards. Under the stopping-robust parser no direction beats the unsteered base (0.790), and D_math_full at α = 0.5 is significantly worse (0.650, p = 0.0003). The steering result does not survive the parser correction and cannot be cited as causal support; it is the second demonstration of the same stopping artifact as §2. **Amplification at the α where the raw effect appears: η_ref × α / ‖d_A‖ = 11.243 × 0.25 / 0.17 ≈ 16× (α = 0.25) and ≈ 33× (α = 0.5); 66× at α = 1, where every direction including random collapses. State "16–33×", not a single round number.** The α ≤ 0.5 raw gains are not separated from stopping effects; the re-scored table shows they *were* the stopping effect. 20 steered neutral generations per direction at α=1: results/steer_eval/neutral_gens_{A,C,D_math_full,random}_a1.md. ⏳ human reading.

## 11. Cross-seed reproducibility (results/perposition_table_seeds*.csv, results/perposition_table_A_seeds*.csv)
- **SFT arms reproduce.** D seed 0 · seed 1 cosine at p1/p2: 0.978 / 0.974 (neutral), 0.951 / 0.970 (math). D_math_full: 0.938 / 0.920 (neutral), 0.961 / 0.989 (math). Norms within 5% across seeds.
- **C reproduces across seeds like SFT (0.97–0.98) while A does not (0.68).** C seed 0 · seed 1 at p1/p2: **0.983 / 0.972 (neutral), 0.969 / 0.984 (math)**, p3–4 neutral 0.979 / 0.976 (`results/perposition_table_C_seeds_cosine.csv`); norms within 0.3 % at neutral p1 (3.488 vs 3.498), within 3.3 % at math p1 (5.380 vs 5.204). Both C seeds sit at the same angle to both A seeds (0.48–0.50 at neutral p1). This is a cosine, so it is the one C-vs-A contrast that no dose or loss-placement confound touches: the imitation trace is a property of the data and recipe, the GRPO trace of the run.
- **A reproduces far less.** A seed 0 · seed 1 at matched steps, neutral p1: 0.544 (step 25) → 0.676 (150); neutral p2 0.508 → 0.629; math p1 no monotone trend (0.641 at 25, range 0.572–0.677, 0.622 at 150); math p2 0.752 → 0.788. For scale: within-seed cos(A@25, A@150) is 0.83–0.92 and the matched-norm random null is |cos| < 0.2.
- A seed 1 final norms sit below seed 0 at every position: neutral p1 0.155 vs 0.210, p2 0.147 vs 0.184; math p1 0.343 vs 0.483, p2 0.379 vs 0.512. Constancy is close (neutral p1 0.231 vs 0.258). math/neutral ratio 2.22 / 2.57 vs 2.30 / 2.78.
- A seed 1 Patchscope (L15 p1, λ=1): neutral `' ', '/', 'K', '\n', '2', '1', ' sh', …`; math `' search', ' target', ' current', ' searching', ' spell', '→', ' lookup', …` (results/patchscope_A_s1_step150_L15.json).

## 11b. Module-family split of ΔW — uninformative (results/lora_delta_family_split.json)
Share of ‖ΔW‖²_F by module family over all 248 modules: A 0.594 MLP / 0.316 linear-attn / 0.090 full-attn; C 0.596 / 0.318 / 0.086; D 0.611 / 0.311 / 0.078; D_math_full 0.597 / 0.317 / 0.086; B 0.595 / 0.314 / 0.090; **N3 (untrained) 0.594 / 0.316 / 0.090**. Every arm agrees with the untrained adapter to within 0.02, so this statistic is set by the module dimensions and the LoRA target list, not by what was learned. **Recorded as uninformative; it does not discriminate arms and must not be cited as evidence that A and C "change the same kind of weights".** (An earlier top-10-modules-by-Frobenius view covered only ~9% of the mass and was misleading; superseded by this exact split.)

## 15. N2 — the preregistered 50-random-direction null, and H3's verdict (results/n2_null.md, tools/n2_null.py)

PREREG names three nulls; N1 and N3 reach the write-up and **N2 had not**, so this section records what became of it.

**What the saved files are.** `results/items_N2_s0_L{11,15,19}_{neutral,math}.jsonl` hold 50 **logit-lens** top-20 lists per layer and set, from isotropic Gaussian directions rescaled to eta_ref. The direction vectors were not saved, but `tools/null_decodes.py` records the generator (`numpy.random.default_rng([seed, layer, set_index])`) and regenerating it reproduces the saved per-draw norms exactly, so the vectors are recoverable offline.

**N2 nulls the cosine statistic (H3), not the Patchscope relevance count.** The relevance null is N1 under identical λ selection (§6). Explicitly: **not usable as the null for the headline relevance statistic.** The headline arm readout is **Patchscope** content-relevance under lambda selection; N2 is **logit-lens**. Building a Patchscope N2 null needs the model on a GPU, and the pod is terminated and the adapters destroyed. N2's raw norms (~50, a property of a Gaussian in R^2560) are also not a null for arm trace norms — that role belongs to the paired split-half floor and to N3. **Status: computed, but not usable as specified for the headline.**

**Usable, and used, for the null H3 actually names.** H3: *cos(d_A, d_B) exceeds the 95th percentile of cos(d_A, N2 draws)*, at L15:

| set | pos | cos(d_A, d_B) | N2 null mean | 95th pct | max | H3 |
|---|---|---|---|---|---|---|
| neutral | 1 | **−0.1266** | +0.0046 | +0.0303 | +0.0526 | **FAILS** |
| neutral | 2 | **−0.1402** | +0.0049 | +0.0388 | +0.0490 | **FAILS** |
| math | 1 | +0.0463 | −0.0008 | +0.0323 | +0.0440 | passes |
| math | 2 | +0.1520 | −0.0050 | +0.0229 | +0.0340 | passes |

**H3 fails on the primary (neutral) snippet set.** On neutral text d_A and d_B are **orthogonal to slightly negative (−0.13, below all 50 random draws)** against a null centred on zero. H3 predicted the opposite. Clause 2 (‖d_B‖ < ‖d_A‖) is satisfied everywhere (0.094 < 0.210 at neutral p1). Clause 3 (decode d_A − d_B descriptively) was **never produced** — no A−B readout artifact exists in `results/`. This is a preregistered hypothesis with a negative result on its primary set; the write-up reports it, it does not quietly drop.

**Where the arms fall in the N2 cosine null** (cos(d_X, d_A) vs the 50 draws, neutral p1): C +0.505, D_math_full +0.266, D +0.200 and N3 +0.097 are all above all 50 draws; **B −0.127 is below all 50** (0th percentile). At math p2: C +0.574, D_math_full +0.377, B +0.152 above all 50; N3 +0.015 at the 86th.

## 17. Figures (analysis/make_figures.py; inputs per figure in figs/figure_sources.json)
- **`figs/fig1_norm_vs_accuracy.png` — headline.** One panel: trace norm ‖d̄‖ at L15 position 1 on neutral text (log y) against re-scored held-out accuracy (x), every arm. C and A dark, other arms low opacity; each bar runs from that arm's paired split-half floor up to its measured norm; N3 is a horizontal floor line; base is a vertical line (accuracy 0.790, no trace by construction). The bracket marks **16.63–22.63×** (four C×A seed pairs, `results/trace_ratio_C_A_seeds.csv`; the figure itself plots C seed 0 only — C seed 1 landed after the figure was regenerated and differs from seed 0 by 0.3 % in norm, 0.5 % in accuracy). Sources: `results/perposition_table_C.csv`, `results/perposition_table_seeds.csv`, `results/perposition_table_A_seeds.csv`, `results/acc_table_reparsed.md`.
  **Gap shown on the figure itself:** A seed 1 has a measured norm (0.155) but **no held-out accuracy** — its eval was never run and the adapter was destroyed with the pod, so it is drawn as an open marker on A seed 0's accuracy line, not as a second scatter point. The same is true of D seed 1, D_math_full seed 1 and N3, which is why they are not plotted.
- `figs/figA1_perposition_geometry.png` — the former Figure 1, per-position geometry with split-half floors, moved to the appendix.
- `figs/fig2_visibility.png`, `fig3_steering_dose_response.png`, `fig4_A_emergence.png`, `fig5_patchscope_tokens.png` unchanged.

## 12. Pending (⏳; insert only if verified)
Human tags on the 68 blinded discordant items (format vs reasoning); human reading of the steered neutral generations and the blinded Patchscope lists (results/review_packet/); self-report reading (results/items_D_s0_L15.jsonl). **Closed since 05:49:** ‖ΔW‖_F for C, visibility V per arm and both A seeds, η_ref steering grid, A seed 1 cross-seed cosines, module-family split, stopping-robust re-scoring. **Dropped by decision:** C relevance grading (Guiv, 2026-09-04). **⏳ C_masked (completion-only imitation SFT on the same corpus; the loss-placement test in §12b): gate 07:00 Sat; decision line V(C_masked) ≤ 0.18 → gap sits with loss placement, ≥ 0.30 → gap stays with the learning rule; if not landed by 07:30 this reads "not run before submission".**

## 12b. Limitations that constrain every number above

**"Bias term or something deeper?" — what this run can answer.** The mean-offset share (constancy) at neutral p1 is **A 0.258, D_math_full 0.249, D 0.277** (`results/perposition_table_C.csv`). At this dose the RL trace is **as constant as SFT's** — the preregistered P3, that RL's trace would be *less* constant, is refuted. What differs between the arms is magnitude and cross-seed reproducibility (A s0·s1 = 0.68 against 0.92–0.98 for SFT), not the form of the trace. No ablation from the fine-tuned model was run, so whether the offset is load-bearing — the "deeper" half of the question — is **untested here**, not answered negatively.

**Loss placement — the second named alternative to the learning-rule reading (`docs/T2_THEORY_PASSES.md` §1.0, model M0; numbers already in §5).** Masked math SFT (D_math, loss on solution tokens only) has **V = 0.059, below both A seeds (0.125 / 0.092)**; unmasking the same corpus (D_math_full, loss on prompt + solution) takes V to **0.179 at essentially unchanged ‖ΔW‖_F (6.58 → 6.70)** — a 3× change in per-unit trace from where the loss is placed, not from how much the weights moved. C was trained **unmasked** (loss on the GSM8K prompt and the completion), whereas GRPO's loss touches only completion tokens. So the arms nearest A in loss placement show *less* per-unit trace than A, and every arm with prompt-token loss (D, D_math_full, C) shows more: the C-vs-A V gap is confounded with loss placement as well as with dose. The decisive test is a completion-only C on the same corpus and recipe (`C_masked`): **V(C_masked) ≤ 0.18 would put the gap on the side of loss placement; ≥ 0.30 would leave it with the learning rule** (thresholds as stated in `docs/T2_THEORY_PASSES.md` §1.1 and "The single experiment to run first": M0 predicts ≤ 0.18, probably ≤ 0.10; ≥ 0.3 falsifies M0's account of C's trace). ⏳ see §12. What survives either outcome: the cross-seed cosine contrast (C 0.97–0.98 vs A 0.68, §11) and the sign/size of C·A, which are norm-free.

**What is and is not reproducible from this repository.** Every number in the digest is recomputable today from the files in `results/` plus the committed code, and `tools/recompute_oneliners.md` gives one runnable command per VERIFY row; all 45 were run and match. What cannot be regenerated: the LoRA adapters and the activation caches, destroyed when the pod was terminated, so nothing can be re-measured at a new layer, position or snippet set without repeating the training runs; arm B's training curve, whose only source was a pod log; and N2 as a Patchscope null, which would need the model on a GPU. Two further constraints on precision: greedy bf16 decoding is not run-to-run reproducible (four executions of the same unsteered evaluation gave 24–28/200, so every accuracy carries roughly ±2 items before any test), and cap-hits are inferred from re-tokenised length rather than a stored EOS flag. Seed pairs exist for D, D_math_full, A and (as of the `replication` merge) C; C seed 1's adapter survives on the Mac (untracked, `adapters/C_s1/final/`), the others do not. The full incident ledger — sync reverts, the corrected §7 sentence, the aborted replications — is in `VERIFY.md` and `CHANGELOG.md`, not here.

## 13. Attempt ledger (CHANGELOG)
Two dead A_early launches (stale code / self-killed shell); vLLM abandoned (import fails on Python 3.11); D_math eval rerun after a sync deleted the first file (greedy → expected identical, not verified); relaunch shell killed twice by its own kill pattern; a device-safe steering hook introduced an unassigned variable that crashed the first random-direction batch (fixed, no reported number affected).
**Pod: $13.96/h, 14.38 h uptime, $200.81 total; terminated 2026-09-04 14:32 Zurich and verified gone.**
**This digest was untracked until 2026-09-04 and therefore had no backup; it survived pod termination only because it lived on the Mac. Now committed.**
**Sync defect (found while bringing this file current):** the pod→Mac rsync in the runner's `ship.sh` pulled the pod's older copies over newer locally-generated files, and those reverted files were then committed. Three files were silently stale: `results/acc_table.md` (missing A, B, C, D_math, D_math_full and every paired count — 50 lines), `results/visibility_table.md` (missing the A seed-1 row), `results/lexical_items_perposition.jsonl` (66 rows instead of 102). All three were regenerated from their inputs and are correct as of this commit. Derived files that regenerate identically and were therefore never affected: `results/steer_table.md`, `results/acc_table_reparsed.md`, `figs/*`.

## 14. Coverage audit (2026-09-04): what is in results/ but in no section, and what this file asserts without a local source

**(a) Real content in `results/` that no section above covers.**
- **N2 — the preregistered 50-random-direction null — appears nowhere in this digest.** `results/items_N2_s0_L{11,15,19}_{neutral,math}.jsonl` (6 files, 50 decoded directions each) and the N1 split-half logit-lens decodes `results/items_N1_s0_L{11,15,19}_{neutral,math}.jsonl` exist and are unused. PREREG names N1, N2 and N3 as the three nulls; only N1 (as a Patchscope null, §6) and N3 (as a geometry floor, §3/§5) are reported. **A null the preregistration requires is missing from the write-up path.**
- `results/acc_table_reparsed_variant.md` — the re-parser sensitivity variant that also cuts on `^Question:`/`^Problem:`. Computed, never reported; it belongs beside §2 as a robustness check on the cut-pattern choice.
- `results/perposition_D_s0_step250_L15.json` — the Minder-faithful per-position D readout including the un-normed logit-lens variant at position 0. Its numbers feed §3 and §6 but the file is never named.
- `results/preflight_samples.json` (backs §1's 25/32), `results/emergence_A_rewards.json` (backs §1's and §7's A reward curve), `results/acc_table_{single,paired}.csv` (machine-readable §2), `results/review_packet/*` (the four human reading packets), `results/steer_eval/neutral_gens_*_a1.md` (cited in §10 but not as files).
- Retired first-pass D readout kept for provenance: `results/diff_D_s0_L15_{neutral,math}.{npy,json}` and `results/activations_D_s0_L15_*.json` — the single-vector pooled-≥4 estimator mentioned in §6 as retired.

**(b) Numbers this digest asserts that have no local source file.**
- **§1's arm-B training curve — "reward ≈0.07, truncation 0.79, mean length 456" — has no local backing.** It came from `logs/B_s0.log` on the pod, which was destroyed on termination; `logs/` here contains only `.gitkeep`. The arm-A curve *is* backed (`results/emergence_A_rewards.json`: reward 0.078 at step 1, 0.801 at step 5, 0.906 at step 100, 0.949 at step 150; mean length 426.5 → 144.7 at step 10 → 172.3 at 150). **Either re-derive B's curve or mark it unverifiable before it is cited.**
- §1's GRPO hyperparameters and the SFT arm descriptions are configuration, not measurements; they are recorded in `CHANGELOG.md` and in each run's `run_meta.json`, and the `run_meta.json` files for arms other than D and D_math_full were on the pod and are gone.
- §8's "always-math 0.20 / always-none 0.40" is not stored as a computed baseline; it is derivable from the label distribution of `results/judge_calibration.jsonl` (cooking 20, math 20, none 40, poetry 20 of 100 rows), which is what those two numbers are. Fine to keep, but it is a property of the fixture, not a measured baseline on the real lists.
- Everything else in §§2–11b was checked against its cited file and matches.
