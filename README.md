# rl-readable-trace

**Question.** Minder et al. (arXiv 2510.13900) showed that narrow SFT leaves a readable trace in the mean base→fine-tuned activation difference on *unrelated* text. Does on-policy GRPO leave the same kind of trace — and if not, is that the data, the behaviour, the learning rule, or where the loss is placed?

**Finding (one sentence, hedged; one 4B model, one task).** The size and direction of the readable trace are set by where the loss is placed, not by the learning rule: masking prompt tokens removes ~92 % of imitation SFT's trace at unchanged accuracy and dose and lands it beside GRPO's (within 2×, same direction, same format-shaped readout); per unit weight change the RL trace is larger, and its absolute trace is small mainly because its weight update is small. Two masked seeds; the position-vs-content split and the ablation are single-seed and carry their confounds (`CLAIM_FIREWALL.md` §1, §3).

## Read in this order
1. `docs/RESULTS_DIGEST.md` — every number, with its `results/` file and VERIFY row. The sole citable source.
2. `VERIFY.md` — one row per number (72 rows), who produced it, how it was recomputed (`tools/recompute_oneliners.md` has one runnable command per row), what Guiv read himself, what was not checked.
3. `PREREG.md` — frozen hypotheses and pass criteria, with append-only amendments. The three that decide the headline: **02:00 Sat** (C_masked, thresholds V ≤ 0.18 / ≥ 0.30 — observed 0.049), **~04:20** (position vs content: C_scrambled / C_shifted / C_masked s1 — 0.380 / 0.272 / 0.047), **~05:40** (ablation of the trace from the fine-tuned model — Δ +2 / +2 items, not load-bearing).
4. `CLAIM_FIREWALL.md` — what may and may not be claimed. `PROGRAM_STATE_CURRENT.md` — the current state. `CHANGELOG.md` — attempt ledger, costs, incidents.

## `results/` layout
`acc_*.json` held-out GSM8K evals (per-item completions; both parsers in `acc_table*.md`); `perposition_*` / `perposition_table_*` per-position mean-difference geometry and cosines (L15; L11/L19 sensitivity); `lora_delta_stats*.json` ‖ΔW‖_F; `visibility_table*.md` V = ‖d‖/‖ΔW‖_F; `patchscope_*.json` and `token_relevance_*.json` token readouts; `steer_eval/` and `steer_table*.md` steering; `ablation_*.json` + `ablation_table.md` the E2 ablation; `null_table.md`, `n2_null.md`, `items_N*.jsonl` nulls; `emergence_*` A's checkpoint series; `blackbox/`, `review_packet/` reading packets; `cache/diffs/` the saved difference vectors; `figs/` figures (`figs/figure_sources.json` maps each figure to its inputs).

## Replication and follow-up reports (separate pods, new files only)
- `results/REPLICATION_REPORT.md` — arm C seed 1. `results/REPLICATION_REPORT_C_masked.md` — the decisive completion-only C (V 0.049).
- `results/REPLICATION_REPORT_R2.md` — C_masked seed 1, C_scrambled, C_shifted. `results/E2_ABLATION_REPORT.md` — subtracting each arm's own trace from the fine-tuned model.

## Reproducing C_masked
```bash
python grpo/train_sft.py train --arm C --data data/C_samples.jsonl --model Qwen/Qwen3.5-4B-Base \
  --model-revision 1001bb4d826a52d1f399e183466143f4da7b741b --out runs/C_masked_s0 \
  --max-steps 225 --seed 0 --save-every 25 --completion-only
```
Then `tools/cache_base_activations.py --layers 15`, `tools/per_position_diff.py --layer 15`, `tools/lora_delta_stats.py`, `tools/visibility_C_masked.py`, and `grpo/eval_acc.py` (200 GSM8K test items, greedy, cap 512). Environment pins: Python 3.11, torch 2.13.0+cu129, transformers 5.16.1, trl 1.12.0, peft 0.20.0 (`tools/pod_bootstrap_E2.sh`). `data/C_samples.jsonl` sha256 `78022b70…`.

**The round-1 adapters (A, B, C s0, D, D_math, D_math_full, N3) and activation caches were destroyed when the first pod was terminated; those numbers are re-derivable only by retraining.** The C s1, C_masked s0/s1, C_scrambled and C_shifted adapters exist off-repo. The stopping-robust accuracy parser (`tools/reparse_acc.py`) is post hoc; every accuracy is reported under both parsers.
