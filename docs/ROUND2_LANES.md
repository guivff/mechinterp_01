# ROUND 2 — four lanes for the next 12 hours (Thu 22:50 → Fri 11:00 Zurich; 34 h to deadline)

Rule for tonight: **the GPU lane is the only one that cannot slip.** Every other lane runs on CPU or in chat, in parallel, and must not touch the GPU pod or the PREREG definitions.

---

## LANE G — GPU runs, now (you + Agent 02 on the pod)

Order matters; each step unblocks the next.

| # | When | GPU | Run | Why / gate |
|---|---|---|---|---|
| G0 | 23:00 | 0 | **Preflight**: load `Qwen/Qwen3.5-4B-Base` text view; assert LoRA targets; 8 samples on 4 GSM8K prompts; parser + truncation rate; `eval_acc.py` base (200 items) | Catches the top-3 silent failures (VLM namespace, parser, template) before any hours are spent |
| G1 | 23:15 | 0 | **Arm D** SFT on cooking corpus, 1 epoch, ckpt every 50 | ~1 h. Gate 1 depends on it |
| G2 | 23:15 | 1 | **Arm A** GRPO, seed 0, `--use-vllm` | 2–4 h. Watch reward at steps 10/20/30 |
| G3 | 23:15 | 2 | **Arm B** shuffled-reward GRPO, seed 0 | same wall-clock as A |
| G4 | 23:15 | 3 | **Cache base activations** on both snippet sets at L=11/15/19, all positions (fp16 .npy); build N3 (untrained LoRA); N1 split-half; N2 50 random draws decoded | Makes Gate 1 a 10-minute job when D lands; nulls are exchangeable |
| G5 | ~00:30 | 0 | **Gate 1**: block-wise readouts D vs N1/N2/N3 at L=15, judge (gpt-5-mini), TF-IDF, self-report | Pass → sleep. Fail → L=19 once, then pivot decision |
| G6 | ~03:00 | 1,2 | **Gate 2** when A/B finish: `eval_acc.py` A, B; paired McNemar; 20 discordant items | Only after this: decode A/B |
| G7 | after G6 | 1 | **Arm C** only if A finished before ~04:00: sample G=8 from A-final, keep correct, SFT 150 steps | otherwise C is cut and H2 is "untested" |
| G8 | overnight | 3 | **Seeds 1–2 of A and B** if GPU 3 is idle after G4 | cheap insurance against the red team's #6; never blocks anything |

If A's mean reward has not moved by step 30: restart A and B at lr 2e-5, **log the restart in CHANGELOG attempt ledger.** If GRPO OOMs or vLLM misbehaves for >30 min: fall back to HF generation (`--no-vllm`), accept 2× slower.

---

## LANE E — implementation & engineering (Agents 01/02/03 on CPU, tonight)

**E1 — Block-wise estimator (Agent 01, blocking Gate 1).** `collect_residual` returns position ids; `diff_stats` per block (K=10 disjoint blocks × 50 snippets, seed 0); `run_readouts.py` emits 10 token lists per (arm × snippet set) and per-block geometry; `summarize.py` computes block-level Wilson CIs and block-to-block cosine; per-position (0–4) Minder-faithful diagnostic on D. Also implement N1 = base split-half and N2 = 50 draws.

**E2 — Reward on truncation (Agent 02, blocking A/B launch).** Reward 0 for completions that hit the cap without EOS; per-step truncation-rate logging. Verify TRL 1.12 passes `completion_ids` (or equivalent) to the reward callback; if not, detect via tokenizer length of the completion text.

**E3 — Judge and lexical baseline (Agent 03, blocking Gate 1).** With `OPENROUTER_API_KEY` set: run live calibration (30 synthetic items × gpt-5-mini and one other family) → `results/judge_calibration.jsonl`; add always-math / always-none baselines and the confusion matrix to `lexical_baseline.py` output; build the external six-domain reference corpus (50 docs × 6 labels) and retrain TF-IDF on it; assert no readout text is in the training set.

**E4 — Prompt/template identity check (Agent 02, 20 min).** One script that renders 3 shared GSM8K examples through the training path, the sampling path, the activation-collection path and the self-report path, and asserts byte-identical strings and token ids. Red-team #13/#14; cheap and decisive.

**E5 — VERIFY.md merge (Agent 01 or you, 15 min).** Restore Agent 01's rows from the parent of `8a1dcd0`; merge additively; add the "attempt ledger" and "overnight autonomous work" sections.

**E6 — Figure pipeline on real files (Agent 01, after Gate 1).** `summarize.py` must refuse to mix `MOCK` and real files; Figure 3 (top-20 tokens per arm, one block chosen by seed) added.

---

## LANE T — theory lanes to open (you, 1–2 h total; chat model as critic only)

**T1 — The cancellation argument, tightened (45 min, goes in the write-up).** Already in `docs/THEORY_NOTE.md`. Two additions worth making: (a) state explicitly that the cancellation is of the *prompt-shared* gradient component, so what survives is a "class-contrast" term — the correct-minus-incorrect gradient within a group — and name what that should decode as (answer format, verification tokens, numerals) rather than topic; (b) note that arm C's SFT gradient on the *prompt* tokens (if not masked) reintroduces the topic component even more strongly than completion-only SFT, so mask prompt tokens in C or state that C is a conservative upper bound on topic readability. Cost: zero GPU.

**T2 — Why Adam blurs it (30 min, one paragraph, honest limitation).** Per-parameter normalisation in Adam means the update is not a linear image of the summed gradient; a component that cancels in the raw sum can survive after sign/normalisation of per-step updates. That is the mechanism by which B could carry a topic offset. It converts "Adam is a caveat" into a *prediction*: if B decodes as topic on neutral text, the culprit is the optimizer, and the diagnostic is the B-vs-N3 comparison. Ties to the subliminal-learning Adam/SGD dissociation (Nanda group, June 2026) as related work — one sentence, no claim.

**T3 — A one-line falsifiable prediction for H4 (15 min).** From T1, the surviving contrast term is *context-dependent* (it lives on completion tokens conditioned on a math prompt), whereas a topic offset is not. Prediction: constancy(A, math) < constancy(A, neutral) is *not* required, but ‖d_A,math‖ > ‖d_A,neutral‖ should hold while ‖d_D,math‖ ≈ ‖d_D,neutral‖. Put this in PREREG's secondary hypotheses before the freeze. It is the cleanest "RL writes conditional traces" test you have and costs nothing.

**Do not open:** anything about long-horizon dynamics, KL-regularised variants, or your other programme's exact results. One page of theory, three predictions, all tested by runs you are already doing.

---

## LANE C — creative lanes (only if GPU 3 is idle or a chat agent is free; none blocks submission)

**C1 — "Contrast tokens" figure.** For A, show the top-20 tokens of `d_A` side by side with `d_C` and `d_D`; if the theory holds, A's list is visibly format/verification-flavoured while C/D are topic. That single figure is the memorable result of the write-up. Cost: zero beyond E1.

**C2 — Cross-arm steering swap (GPU 3, 30 min, only if steering is kept).** Steer the base model with `d_C` and with `d_A` on the *same* neutral prompts and ask the judge which is which. If `d_A` makes the model produce "Answer:"-shaped text without math content, that is a direct qualitative demonstration of the contrast-vs-topic distinction.

**C3 — Minder-faithful mini-diagnostic on D (CPU from cached activations, 20 min).** Per-position (0–4) logit lens at L=15 on D, exactly as in the paper. If it agrees with the pooled estimator, it strengthens Gate 1; if not, it is an honest limitation paragraph. Either way it answers Neel's "what did D validate?" question.

**C4 — The "just ask" baseline done well.** Self-report for base, A, B, D, N3 with all 20 outputs shown for two arms in an appendix. Neel explicitly lists this control; most applicants skip it.

**Not now:** Patchscope, J-Lens, attribution graphs, per-checkpoint trajectories, second model families. Write them as future work in two sentences.

---

## Round-2 agent prompts (paste as-is; each on its own branch `agent0Xb`)

**Agent 01b (Codex/Claude Code, CPU):** "Read AGENTS.md, PROGRAM_STATE_CURRENT.md, PREREG.md, OPEN_TASKS_CURRENT.md. Implement lane E1, E5 and E6 from docs/ROUND2_LANES.md exactly as specified; do not change PREREG definitions. Run tests. Final message: report per AGENTS.md §7."

**Agent 02b (on the pod, GPU):** "Read AGENTS.md, PROGRAM_STATE_CURRENT.md, PREREG.md, docs/NEXT_STEPS_T35H.md §2 and docs/ROUND2_LANES.md lane G and E2/E4. Execute G0; if preflight passes, implement E2, then launch G1–G4. Every launch, restart or failure goes in CHANGELOG.md attempt ledger. Report step time, memory, reward at steps 10/20/30 for A and B, truncation rate, and the E4 identity-check result. Do NOT run readouts on A or B."

**Agent 03b (Codex/Claude Code, CPU, needs OPENROUTER_API_KEY):** "Read AGENTS.md, PREREG.md judge section, OPEN_TASKS_CURRENT.md items 7–8. Implement lane E3 from docs/ROUND2_LANES.md. Deliver results/judge_calibration.jsonl, the external reference corpus with manifest, retrained TF-IDF with leakage assertions, and the always-math/always-none/confusion-matrix output. Report per AGENTS.md §7."

**Theory critic (fresh chat, anti-sycophancy framing):** "A friend wrote docs/THEORY_NOTE.md and wants brutal feedback before citing it. Find every step where the first-order argument fails for Adam, LoRA, PPO clipping, length normalisation, and multi-step training; say which failures would produce a readable topic offset in arm B; suggest the single cheapest diagnostic for each. Under 500 words, no praise."
