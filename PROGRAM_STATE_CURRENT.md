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

## Results
None. No GPU run has started. All figures in `figs/` are MOCK.

## Blockers (see OPEN_TASKS_CURRENT.md)
GPU pod not yet rented; `OPENROUTER_API_KEY` not set; PREREG blanks not filled/frozen; VERIFY.md ledgers from Agents 01 and 03 not merged; block-wise estimator and N1 repair not implemented.

## Gates
- Gate 1 (after D trains, ~1 h): D block-level judge accuracy clearly above N1–N3 and above TF-IDF; if fail → try L=19 once → else pivot to Olmo-3 stage diffing and reset timer.
- Gate 2 (after A trains, 2–4 h): A held-out accuracy > base under paired test; B within noise; 20 discordant items read to separate format from reasoning gains. Only then decode A/B/C.

## Claim ceiling (from red team; see CLAIM_FIREWALL.md)
Single-seed descriptive arm contrasts at one layer. D = reduced-budget conceptual replication. B = difficulty-gated random-gradient control. A−B = descriptive contrast. H4 = input-dependent readout.
