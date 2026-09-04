# PROGRAM_STATE_CURRENT.md — canonical state snapshot

**As of:** Thu 2026-09-03 22:00 Europe/Zurich (≈35 h to deadline). Replace sections in place; history lives in `CHANGELOG.md`.

## Objective (unchanged)
Is the base→fine-tuned mean activation difference on unrelated text decodable to the training domain after GRPO (A) as it is after narrow SFT (D)? Is any A trace attributable to the prompt distribution, to zero-sum-advantage optimization (B), or to off-policy imitation (C)?

## Decisions in force
- Model: `Qwen/Qwen3.5-4B-Base` (pin HF revision in PREREG at freeze). Text causal-LM view loaded explicitly (outer config is multimodal). Fallback if GRPO unstable in first 2 h: `Qwen/Qwen3.5-2B-pt`.
- Layer: L=15 zero-based post-block (= floor(0.5·(D−1)), Minder camera-ready mapping; paper prose says 16). Sensitivity: 11, 19.
- J-Lens: **dropped** (no lens for `-Base`). Logit lens with final RMSNorm is the token readout.
- Reward: binary exact-match on GSM8K; **0 for completions truncated at the 512 cap without EOS**; truncation rate logged per step.
- Estimator: **block-wise** — 10 disjoint blocks × 50 snippets per snippet set (frozen seed) → 10 diffs per (arm × set); accuracy with block-level Wilson CIs; block-to-block cosine reported. Per-position (0–4) Minder-faithful variant computed on D as a diagnostic from cached activations.
- Nulls: N1 = base-vs-base split-half difference; N2 = random direction at matched norm (≥50 draws for a null distribution); N3 = untrained LoRA with matched adapter norm (assert nonzero ΔW).
- Norm target: η_ref = mean ‖h_base,L‖ on neutral snippets; raw norms always reported. (Logit-lens ranks are scale-invariant under RMSNorm; norm matters for steering only.)
- Judge: `openai/gpt-5-mini` via OpenRouter, temperature 0, labels `[math, cooking, law, medicine, poetry, none]`, majority of 3 calls, raw responses saved. Shuffle control = permuted input↔gold pairing. Lexical baseline = TF-IDF+LR trained on an external six-domain reference corpus, tested on readout texts.
- Scope cut order if behind: steering → arm C → extra seeds. Core never cut.
- Steering (if kept): coefficient calibrated on D only by a coherence check, then a common grid {0.25, 0.5, 1.0}×α_D for all arms; steer the base model (deviation from Minder, who steers the fine-tuned model — stated).

## Assets ready (all CPU-validated, none GPU-run)
| Asset | State | Where |
|---|---|---|
| Readout pipeline (diff, logit lens, steer, run_readouts) | tests pass (20 + 12 cached-real-Qwen); needs block-wise + N1 repair | `readout/`, PR from Agent 01 |
| GRPO/SFT training (TRL 1.12, PEFT 0.20, Transformers 5.16) | CPU smoke A/B/D exit 0; group ordering verified in TRL source; LoRA coverage asserts; `grpo/eval_acc.py` | PR #1 @ 8f539f0 (Agent 02) |
| Cooking corpus | 2,000 docs, 267–370 tokens, 0 dups, sha 7a955f6b…, human read: 5 of 20 samples | `data/cooking.jsonl` (Agent 03 @ 8a1dcd0) |
| Snippet sets | neutral (pile-10k) sha c8673772…, math (GSM8K test + MATH test) sha 483c3733…, 500×128 tokens each, 0 overlap with GSM8K train | `data/snippets/` |
| Judge + lexical baseline | hardened (dry-run, resume, majority vote); **live calibration NOT run (no API key)**; lexical fixture 30/30 | `judge/` |
| Analysis | `summarize.py`, mock figures 1–2, cosine matrix — MOCK only | `analysis/`, `figs/*MOCK*` |
| Protocol notes | Minder/OOCR/J-Lens/GRPO-recipe facts, PREREG fills, 5 citations | `docs/PROTOCOL_NOTES.md` |
| Red team | 23 confounds, interpretation traps, Q14 templates, scope cuts | `docs/RED_TEAM.md` |
| Theory note | zero-sum advantages cancel the topic component → sign of H2 | `docs/THEORY_NOTE.md` |

## Results (measured; every number has a file under `results/`)
The GPU lane ran and the pod was terminated 2026-09-04 14:32 Zurich (14.38 h, $200.81). All arms trained: A, B (seeds 0 and 1 for A), C, D, D_math, D_math_full (seeds 0 and 1 for D and D_math_full), N3. Figures 1–5 in `figs/` are real (`analysis/make_figures.py`, sources in `figs/figure_sources.json`); no MOCK file remains in use.

- **Held-out accuracy is reported under two parsers and neither is dropped.** Preregistered last-number parser vs stopping-robust re-parse (`results/acc_table.md`, `results/acc_table_reparsed.md`): base 28/200 → 158/200, B 15 → 162, D_math 132 → 173, D_math_full 127 → 164, D 53 → 108, A 188 → 188, C 186 → 186. A vs base falls from 162-vs-2 discordant to 35-vs-5 (p = 1e-6); A vs D_math from 62-vs-6 to 22-vs-7 (p = 0.008); A vs C is 7-vs-5 (p = 0.77) under both. A 20-item audit of the re-parse found 20/20 rescues genuine (`results/reparse_audit.md`). Much of the raw accuracy gap is A learning to emit EOS.
- **Visibility V = ‖d_neutral,p1‖ / ‖ΔW‖_F is a measured quantity, not a stable per-arm constant** (`results/visibility_table.md`): D 0.3837 (seed 1: 0.3910, ratio 1.019), D_math_full 0.1789 (0.1893, 1.058), C 0.5010, A 0.1252 (seed 1: 0.0919, **ratio 1.363**), D_math 0.0591, B 0.0568, N3 floor 0.0221. A's two adapters have near-identical ‖ΔW‖_F (1.675 / 1.682), so A's spread is entirely in the activation-space numerator. On n = 2 seeds, A's V is reported with that spread attached and is not quoted as a constant.
- Gate 1 passed for D at positions 1–2 on neutral text via Patchscope; position 0 was rejected as evidence (sink-like, no BOS). Per-position geometry, split-half floors, cross-seed cosines, layer sensitivity (L = 11/19), the η_ref steering dose-response with a random null, the module-family split and the black-box panel are all under `results/`.
- **The adapters and activation caches were destroyed with the pod.** Every number in `results/` is re-derivable only by retraining, not by recomputation from saved weights.

## Blockers (see OPEN_TASKS_CURRENT.md)
GPU pod not yet rented; `OPENROUTER_API_KEY` not set; PREREG blanks not filled/frozen; VERIFY.md ledgers from Agents 01 and 03 not merged; block-wise estimator and N1 repair not implemented.

## Gates
- Gate 1 (after D trains, ~1 h): D block-level judge accuracy clearly above N1–N3 and above TF-IDF; if fail → try L=19 once → else pivot to Olmo-3 stage diffing and reset timer.
- Gate 2 (after A trains, 2–4 h): A held-out accuracy > base under paired test; B within noise; 20 discordant items read to separate format from reasoning gains. Only then decode A/B/C.

## Claim ceiling (from red team; see CLAIM_FIREWALL.md)
Single-seed descriptive arm contrasts at one layer. D = reduced-budget conceptual replication. B = difficulty-gated random-gradient control. A−B = descriptive contrast. H4 = input-dependent readout.
