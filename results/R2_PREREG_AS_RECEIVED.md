# R2 preregistration as received (replication chat), recorded before the first pod call

Source: Guiv's R2 brief (sent ~04:20 Zurich; actioned 05:01) and the gate-waiver message (actioned 05:12). Checked at
05:13 Zurich: no PREREG.md on any `origin/*` branch (incl. `origin/pod` @ 52644f3) nor Chat 1's working tree contained
the R2 amendment (grep `scrambled|shifted`). This file is therefore the preregistration of record for R2 on branch
`replication`, committed before the pod was created.

## Thresholds (same as R1)
V at L15, neutral p1: **≤ 0.18 = small (masked-like); ≥ 0.30 = large (C-like)**; between = inconclusive.

## Predictions (written before the run)

| Arm | Position hypothesis | Content hypothesis |
|---|---|---|
| C_scrambled (prompt tokens shuffled, full loss) | large | small |
| C_shifted (64-token unrelated prefix, masked; original prompt+completion supervised) | small | large |

Both large → prompt-token supervision matters regardless of position or content (scrambled tokens carry high loss;
record mean prompt-token loss for C and C_scrambled). Both small → the C_masked effect is not reproduced by these
manipulations; report as such. C_masked s1: V ≤ 0.10 confirms s0; V ≥ 0.30 means s0 was a fluke and the write-up says so.

## Runs, in order
1. C_masked s1: `train_sft.py train --arm C --data data/C_samples.jsonl --out runs/C_masked_s1 --max-steps 225 --seed 1 --save-every 25 --completion-only`
2. C_scrambled s0: per-row prompt tokens permuted with `random.Random(row_index)`, completion unchanged; full loss; seed 0;
   `--max-tokens` set to the exact value reproducing C s0's 8,792-row selection.
3. C_shifted s0: 64-token pile-10k prefix (disjoint from the neutral snippets, zero 8-gram overlap) as masked prompt;
   original prompt+completion supervised. Skipped, not improvised, if pile-10k is not on disk.

## Gates (Zurich, absolute, waiver message): env green 05:30; C_masked s1 V on the Mac 05:50; all three synced+pushed 06:45; hard stop 07:00.
