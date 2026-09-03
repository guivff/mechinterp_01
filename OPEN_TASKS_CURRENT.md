# OPEN_TASKS_CURRENT.md — live queue only (Thu 2026-09-03 22:00 Zurich)

Owner key: **G** = Guiv (decision/verification, counts toward the 20 h), **A01/A02/A03** = coding agents, **POD** = runs on the GPU pod. Remove items when done; log completion in CHANGELOG.md.

## Blocking the GPU launch (tonight, in order)
1. **G** — Fill and freeze PREREG.md: model repo + HF revision, L=15 (11/19), reward-on-truncation=0, K=10 blocks, N1 split-half, judge `openai/gpt-5-mini` T=0, η_ref definition, steering grid rule, scope-cut order. Commit; record the freeze commit hash at the top of PREREG.md and in CHANGELOG.md.
2. **G** — Rent 4×H100/H200 pod; export `HF_TOKEN`, `OPENROUTER_API_KEY`; clone main; `pip install -r requirements.txt`; follow `docs/POD_SETUP.md`.
3. **A02/POD** — Preflight (GPU0): load `Qwen/Qwen3.5-4B-Base` text view; assert LoRA target coverage; generate 8 samples on 4 GSM8K prompts; parser + truncation rate; `eval_acc.py` on base (200 items).
4. **A02/POD** — Launch D (GPU0), A (GPU1), B (GPU2) per `docs/NEXT_STEPS_T35H.md` §2; GPU3 caches base activations for both snippet sets at L=11/15/19 and builds N3.
5. **A02** — Implement reward=0 on truncation (needs completion token length or EOS check in the reward callback; verify against TRL 1.12 kwargs); log truncation rate.

## Parallel with training (CPU)
6. **A01** — Block-wise estimator: `collect_residual` keeps position ids; `diff_stats` per block (K=10, frozen seed); `run_readouts.py` emits 10 token lists per (arm × set); N1 = base split-half; N2 ≥ 50 random draws; `summarize.py` uses block-level Wilson CIs and block-to-block cosine; per-position (0–4) diagnostic for D.
7. **A03** — External six-domain reference corpus (50 docs × 6 labels) for the TF-IDF baseline; retrain baseline on it; test on readout texts; document leakage checks.
8. **A03** — Run live judge calibration with the key (30 synthetic items × 2 models); write `results/judge_calibration.jsonl`; report always-math / always-none baselines and the confusion matrix format.
9. **A01 or G** — Merge VERIFY.md: restore Agent 01's ledger from its parent commit and merge additively with Agent 03's; add "attempt ledger" and "overnight autonomous work" sections.
10. **G** — Read the remaining 15 of 20 seeded cooking samples; sign off in VERIFY.md.

## Gate 1 (as soon as D finishes)
11. **G** — Block-wise readouts on D vs N1/N2/N3 at L=15 (both snippet sets), judge, TF-IDF, self-report; decision recorded in CHANGELOG.md. If fail: L=19 once, then pivot decision.

## Gate 2 (when A finishes)
12. **G** — `eval_acc.py` on A and B; paired McNemar/bootstrap vs base; read 20 discordant items. Decision recorded.
13. **G** — Decode A, B (block-wise, both sets, L=15; then 11/19); A−B residual; self-report. Launch C only if before ~04:00 Fri.

## Write-up — incremental (Fri), cutoff for new numbers Sat 04:00 Zurich, submit by Sat 07:30
14. **G** — Sanity pass: recompute every headline number with own one-liners; read 30 random judge transcripts and steered generations (if steering kept); answer red-team rows 1, 5, 7, 8, 10, 13, 14, 17 in VERIFY.md.
15. **G** — Doc body (one paragraph + figure per experiment; limitations; verification section; random raw examples; Toggl screenshot).
16. **G** — Fri by 12:00: executive summary v1 + doc body v1 + form v1 from base results (submittable fallback). Add-ons inserted as they verify. Fri 18:00: P6 critic pass. Sat 04:00: no new numbers. Sat 06:00–07:30: final read-through, Toggl screenshot, permissions, submit. 07:30–08:59 = upload buffer only.

## Deferred (not before submission)
Extra seeds; steering modality; arm C′; Patchscope; per-checkpoint trajectory; Olmo-3 stage-wise fallback (only on Gate-1 failure).
