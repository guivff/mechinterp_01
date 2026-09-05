# R2 — Replication chat (Claude Code, `~/repl`, branch `replication`): C_masked seed 1 + the position-vs-content split. Sat 2026-09-05 ~04:20 Zurich.

You ran C s1 and C_masked s0 (`19524db`). Same rules as R1: sync adapters before anything else; never terminate with unsynced weights; one sync direction; every result file tracked from its first write; gates are absolute Zurich times read from the system clock before every gated step; report each result as it lands, V first. Existing files are not edited; new tools go under `tools/`. Read `PREREG.md` (the 02:00 and 02:56 amendments) before launching.

## Why
C_masked showed that masking prompt tokens removes ~92 % of C's neutral-text trace (V 0.50 → 0.049) at unchanged accuracy and dose. Masking removes two things at once: loss on the **early positions** of every sequence, and loss on **human-written problem-statement content**. This run separates them, and adds a second C_masked seed.

## Preregistered thresholds (same as R1; recorded in PREREG.md before launch)
V at L15, neutral p1: **≤ 0.18 = small (masked-like); ≥ 0.30 = large (C-like)**; between = inconclusive. Predictions, written before the run:

| Arm | Position hypothesis | Content hypothesis |
|---|---|---|
| C_scrambled (prompt tokens shuffled, full loss) | large | small |
| C_shifted (64-token unrelated prefix, masked; original prompt+completion supervised) | small | large |

Both large → prompt-token supervision matters regardless of position or content (note: scrambled tokens carry high loss; record mean prompt-token loss for C, C_scrambled). Both small → the C_masked effect is not reproduced by these manipulations; report as such. C_masked s1: V ≤ 0.10 confirms; V ≥ 0.30 means s0 was a fluke and the write-up says so.

## Runs, in this order (each: train → readouts → sync → report → next)
1. **C_masked s1**: `grpo/train_sft.py train --arm C --data data/C_samples.jsonl --out runs/C_masked_s1 --max-steps 225 --seed 1 --save-every 25 --completion-only`.
2. **C_scrambled s0**: `tools/make_scrambled_prompts.py` (new) → `data/C_samples_scrambled.jsonl`: for each row, tokenize the `prompt` field with the model tokenizer, permute its tokens with `random.Random(seed=row_index)`, detokenize back into `prompt`; `completion` unchanged; assert row count, per-row token count and completion text unchanged; sha256 recorded. Train: `--arm C --data data/C_samples_scrambled.jsonl --out runs/C_scrambled_s0 --max-steps 225 --seed 0 --save-every 25` (**no** `--completion-only`).
3. **C_shifted s0**: `tools/make_shifted_prompts.py` (new) → `data/C_samples_shifted.jsonl`: for each row, new `prompt` = a 64-token snippet drawn from pile-10k documents **disjoint** from every document used in `data/snippets/neutral.jsonl` (assert zero 8-gram overlap with the 500 neutral snippets; record the doc ids); new `completion` = original prompt + original completion. Train with `--completion-only` so the prefix is masked and the original prompt+completion are supervised at positions ≥ 64. Assert the supervised-token count equals C's unmasked count for the same rows.
   If pile-10k is not on disk, use the text of `data/cooking.jsonl`? **No** — that injects the cooking trace. Use any pretraining-style source with zero overlap with the neutral snippets, and name it.

Readouts for each, identical to R1: neutral and math per-position table at L15 with split-half floors, cosines to every existing arm (C s0/s1, C_masked s0, A s0/s1, D_math, D_math_full, D, B), ‖ΔW‖_F, V, accuracy on the 200 held-out items under both parsers, Patchscope neutral p1 λ = 1 top-20, supervised-token fraction, mean training loss on prompt tokens vs completion tokens where applicable.

## Gates (Zurich, absolute)
Env green by **04:55** (else abort, report, terminate). C_masked s1 V on the Mac by **05:20**. All three synced by **06:45**. **Hard stop 07:00**: anything not synced and pushed by 07:00 is not in the submission; terminate the pod at 07:00 regardless, after syncing what exists. Expected cost ≈ $6–8.

## Outputs
`results/REPLICATION_REPORT_R2.md` (decision lines first: one per arm with V and the threshold it met), `visibility_table_R2.md`, `perposition_table_R2{,_cosine}.csv`, `acc_{C_masked_s1,C_scrambled_s0,C_shifted_s0}.json`, `acc_table_R2.md`, `patchscope_*_L15.json`, `data/C_samples_{scrambled,shifted}.jsonl` + sha256s, `logs/pod_R2/`, ledger line. Adapters under `adapters/` untracked, kept. Push to `origin/replication`. Report in AGENTS.md §7 format, with "three ways this could be wrong" — the high-loss confound on C_scrambled must be one of them.
