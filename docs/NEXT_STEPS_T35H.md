# NEXT STEPS — Thu Sept 3, 21:50 Zurich, 35h to deadline

Status: pipeline, corpus, snippets, judge, protocol notes and red team are done (CPU only). No GPU run exists. The overnight agents also surfaced four decisions that block the GPU launch. Make them now, in this order, then launch.

## 0. Four decisions (make in 10 minutes, write them into PREREG.md, commit, freeze)

**D1 — Exact model.** `Qwen/Qwen3.5-4B-Base` (revision pinned in PREREG). Reason: GRPO-from-base is the clean story and needs no chat template. Consequence: **J-Lens is dropped** (the only pre-fitted lens targets `Qwen/Qwen3.5-4B`, not `-Base`); logit lens was always the baseline and the project never depended on J-Lens. Write one sentence in the doc saying so.

**D2 — Layer.** `L = floor(0.5 × (D−1)) = 15` (zero-based post-block, Minder camera-ready mapping), sensitivity at 11 and 19, and record that the paper's prose gives 16. Not 0.6.

**D3 — Reward on truncated completions.** Reward 0 for any completion that hits the 512-token cap without EOS; log the truncation rate per step. (The last-number parser on a truncated completion picks an intermediate number — red-team risk #2.) Keep TRL's default loss aggregation; note the `loss_type` name in the ledger rather than tuning it.

**D4 — Scope, now, not at hour 8.** Cut in this order if behind: steering first (red team ranks it the least identifiable modality), then arm C. Keep: A, B, D, nulls, token readout, self-report, held-out accuracy. Arm C launches only if A finishes by ~04:00 Fri.

## 1. Two design repairs that must land before Gate 1 (Agent 01, CPU, ~1h, in parallel with training)

**R1 — Independent observations (red team #1, Neel's first question).** One aggregate vector → one top-20 list → 100 judge calls is not 100 observations. Fix: split each snippet set into **K = 10 disjoint blocks of 50 snippets** (frozen seed), compute `d` per block, decode per block, judge per block → 10 independent token lists per (arm × snippet set); report accuracy with block-level Wilson CIs and the block-to-block cosine (stability). Steering, if kept, is clustered by prompt. `collect_residual` should keep the position index so the Minder-faithful per-position (0–4) estimator can be computed on D as a diagnostic without re-running the model.

**R2 — Nulls and controls that are exchangeable.**
- N1 := base-vs-base split-half difference (base on block *i* minus base on block *j*): same estimator, no training. The current "base mean activation" N1 is not a difference and can't be norm-matched meaningfully.
- Label-shuffle control := permute the input↔gold-domain pairing across items (not the visible option order).
- TF-IDF baseline := train on a frozen external reference corpus for all six labels (50 short docs per label, generated or public), test on readout texts. Never train on the readout texts.
- Norm target := η_ref = mean ‖h_base,L‖ on neutral snippets; keep raw norms. Note in the doc that RMSNorm makes logit-lens ranks scale-invariant, so norm matching matters for steering dose only.

## 2. GPU runbook (Agent 02 on the pod, or you; start tonight)

```
# pod: 4×H100/H200, CUDA torch, follow docs/POD_SETUP.md; export HF_TOKEN OPENROUTER_API_KEY
git clone <mechinterp_01> && cd mechinterp_01 && pip install -r requirements.txt
# preflight (15 min, GPU0): load Qwen3.5-4B-Base text view, assert LoRA target coverage,
#   generate 8 samples on 4 GSM8K prompts, check parser + truncation rate, run eval_acc on base (200 items)
# then, in parallel:
GPU0: python grpo/train_sft.py train --arm D --data data/cooking.jsonl --model Qwen/Qwen3.5-4B-Base --out runs/D_s0   # ~1h
GPU1: python grpo/train_grpo.py --arm A --model Qwen/Qwen3.5-4B-Base --out runs/A_s0 --seed 0 --use-vllm          # 2–4h
GPU2: python grpo/train_grpo.py --arm B --model Qwen/Qwen3.5-4B-Base --out runs/B_s0 --seed 0 --use-vllm
GPU3: python readout/make_null_adapter.py --match-later; collect base activations for both snippet sets at L=11,15,19 (cache to disk)
```
Watch A's reward curve for the first 20 steps. If mean reward has not moved by step 30, restart A and B with lr 2e-5 (log the restart in VERIFY.md — the red team will look for a file drawer).

**Gate 1 (as soon as D finishes):** block-wise readouts on D vs N1/N2/N3 at L=15, judge (gpt-5-mini via OpenRouter, temperature 0), TF-IDF, self-report. Pass = D's block-level judge accuracy clearly above nulls *and* above TF-IDF. If it fails: try L=19 once (Minder: later layers read out better), then decide pivot.

**Gate 2 (when A finishes):** `eval_acc.py` on base, A, B with paired item-level comparison (McNemar or paired bootstrap), plus read 20 discordant items to separate format gains from reasoning gains. Only then decode A and B.

## 3. The theory lane (Guiv, ~1h, no GPU) — turns H2 from a guess into a derived prediction

See `docs/THEORY_NOTE.md`. One paragraph version: at first order, the mean activation change on unrelated text is a fixed linear map applied to the parameter update, `E_x[δ(x)] ≈ (E_x J_x) Δθ`. For SFT on a narrow corpus, every training token carries the same "topic" feature, so `Δθ` accumulates a common component and the trace is a near-constant offset — Minder's finding. For GRPO the advantages within each group sum to zero, so any gradient component that is *shared across the G completions of a prompt* (the prompt-topic component) cancels exactly in the expected update; what survives is the *within-group contrast* between correct and incorrect completions. Arm C (rejection-sampling SFT on the same samples) has no zero-sum weighting and keeps the topic component; arm B (shuffled reward) keeps the zero-sum structure, so it also cancels the topic component. This predicts, before any readout: constancy(A) < constancy(C); d_A less "math-topic"-readable than d_C at matched norm; and whatever d_A decodes should look like answer-format / correctness-contrast tokens rather than topic tokens. Caveats to state: first-order, one-step, plain-SGD heuristic; Adam and LoRA and many steps blur it; it predicts the *sign* of H2, not its size. This is exactly the kind of "theory that changes an empirical prediction" your packet says to lead with, and it is public-safe (no unpublished numbers).

## 4. Language to use in the write-up (from the red team; adopt verbatim)
- B is a "difficulty-gated random-gradient control", not "generic optimization".
- A−B is a "descriptive contrast associated with intact reward assignment", not a reward-specific vector.
- D is a "reduced-budget conceptual replication of Minder et al.", not a replication.
- H4's on-domain advantage is "input-dependent readout", not a causal gate.
- Every number is single-seed unless a second seed exists; say so in the figure caption.

## 5. Housekeeping before anyone trusts VERIFY.md
Agent 03's VERIFY.md overwrote Agent 01's ledger. Restore Agent 01's version from its parent commit and merge the two additively before Gate 1; the ledger is what feeds form Q16.

## 6. Timeline (Zurich)
- Thu 22:00–23:30: decisions D1–D4 into PREREG, commit, freeze. Pod up, preflight, launch D/A/B. Agent 01 starts R1/R2. Set OPENROUTER_API_KEY and run judge calibration (needs ~10 min).
- Thu 23:30–01:00: D finishes → Gate 1. Sleep after Gate 1 with A/B running.
- Fri 07:00–09:00: Gate 2; decode A, B (and C if it exists) block-wise on both snippet sets; Figure 1 real.
- Fri 09:00–12:00: sanity pass (recompute headline numbers with your own one-liners; read 30 random judge transcripts; answer red-team rows 1, 5, 7, 8, 10, 13, 14 in the ledger).
- Fri 12:00–16:00: doc body. 16:00–18:00: executive summary in your voice. 18:00–20:00: form. Submit Fri evening; the Sat 08:59 deadline is a buffer, not a target.
